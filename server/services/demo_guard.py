"""
Spend caps for the public demo.

Token Factory bills per model call, and a demo URL in a submission is open to
anyone who finds it. The account is capped at the provider (prepaid credits
only, a key made just for the demo so it can be revoked); these are the layers
inside the app, so the worst case is a few dollars rather than the whole
balance:

    access code     DEMO_ACCESS_CODE        judges get a link carrying ?code=…
    daily budget    DEMO_DAILY_LIMIT        model calls per UTC day, everyone
    per-IP rate     DEMO_IP_HOURLY_LIMIT    model calls per address per rolling hour
    concurrency     MAX_CONCURRENT_JOBS     scoring runs in flight at once
    run time        JOB_TIMEOUT_SECONDS     wall clock per run (600 unless set; 0 = off)
    clip length     MAX_CLIP_SECONDS        the scored window, after trimming
    frames          MAX_FRAMES              images per lift (see pipeline.py)
    proxies         TRUSTED_PROXIES         who may tell us the visitor's address

The budgets count model calls, not uploads. One lift costs three (triage,
observation, judge), a repair retry or the fallback judge one more each, and a
session clip one triage per window on top — so counting uploads left the bill
open-ended. A run is admitted when there is room for one whole lift and is then
charged call by call; if the budget runs out part-way, the run stops there and
keeps the lifts already scored.

Every limit except the run timeout is off until its variable is set, so local
runs, the CLIs and the tests are untouched; the Docker image sets them all.
Counters live in this process, like the job store: a restart resets the day's
count, which errs on the side of letting a judge in. A limit turns a request
away with a plain sentence and a machine-readable reason; the page uses the
reason to offer the cached sample clip instead.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import math
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

DAY_SECONDS = 86_400
HOUR_SECONDS = 3_600
BUSY_RETRY_SECONDS = 60
# Distinct IPs remembered before idle ones are forgotten.
IP_TABLE_SOFT_LIMIT = 2_000
# The least a lift costs: triage, observation, judge. A run is only admitted
# with this much room, so nobody pays for the vision call and then loses the
# verdict to the budget.
CALLS_PER_LIFT = 3
DEFAULT_JOB_TIMEOUT_SECONDS = 600.0
# In TRUSTED_PROXIES: the loopback, link-local and RFC 1918 ranges a container
# network or a proxy on the same host comes from. Spelled out rather than
# Python's `is_private`, which also counts the documentation and benchmark
# ranges.
PRIVATE_RANGES = "private"
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "169.254.0.0/16", "::1/128", "fc00::/7", "fe80::/10")
)
# In TRUSTED_PROXIES: trust whatever reaches the port. Only right when nothing
# but your proxy can reach it (Cloud Run, Fly, a port that is not published).
ANY_PEER = "*"


class DemoRefused(Exception):
    """A request the demo turns away, with what the HTTP layer needs to say so."""

    def __init__(self, status: int, reason: str, detail: str, *, retry_after: Optional[int] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail
        self.retry_after = retry_after


class BudgetExhausted(DemoRefused):
    """The budget ran out during a run. The call it was raised for never happens."""


# ---------- configuration (read live, so tests and operators can change it) ----------


def _number_env(name: str) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; treating it as unset", name, raw)
        return 0.0
    return max(0.0, value)


def access_codes() -> List[str]:
    """DEMO_ACCESS_CODE, comma-separated so one code can be revoked without the rest."""
    raw = os.getenv("DEMO_ACCESS_CODE") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def access_required() -> bool:
    return bool(access_codes())


def daily_limit() -> int:
    """Model calls per UTC day, all visitors together. 0 = no limit."""
    return int(_number_env("DEMO_DAILY_LIMIT"))


def ip_hourly_limit() -> int:
    """Model calls per visitor address per rolling hour. 0 = no limit."""
    return int(_number_env("DEMO_IP_HOURLY_LIMIT"))


def max_concurrent_jobs() -> int:
    return int(_number_env("MAX_CONCURRENT_JOBS"))


def max_clip_seconds() -> float:
    return _number_env("MAX_CLIP_SECONDS")


def job_timeout_seconds() -> float:
    """
    Wall clock a scoring run may take, model calls and the replay render
    included. Unlike the other caps this one is on unless set to 0: an upload
    must not hold a worker, a budget slot and a phone forever.
    """
    raw = os.getenv("JOB_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return DEFAULT_JOB_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        logger.warning("JOB_TIMEOUT_SECONDS=%r is not a number; using %s", raw, DEFAULT_JOB_TIMEOUT_SECONDS)
        return DEFAULT_JOB_TIMEOUT_SECONDS
    return max(0.0, value)


def trusted_proxies() -> List[str]:
    """TRUSTED_PROXIES: addresses, CIDR ranges, `private`, or `*`, comma-separated."""
    raw = os.getenv("TRUSTED_PROXIES") or ""
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def demo_caps_set() -> bool:
    """Whether any public-demo cap is configured — the sign this is a public deploy."""
    return bool(access_required() or daily_limit() or ip_hourly_limit() or max_concurrent_jobs() or max_clip_seconds())


# ---------- the checks ----------


def check_access(code: Optional[str]) -> None:
    """Raise unless `code` is one of the configured codes (or none are configured)."""
    codes = access_codes()
    if not codes:
        return
    given = (code or "").strip()
    if not given:
        raise DemoRefused(
            401,
            "access_code",
            "This demo needs an access code. It is on the submission page, or open the link that carries it.",
        )
    if not any(hmac.compare_digest(given.encode("utf-8"), known.encode("utf-8")) for known in codes):
        raise DemoRefused(403, "access_code", "That access code is not right. Check it against the submission page.")


def enforce_clip_length(start_sec: float, end_sec: float) -> None:
    """
    Refuse a scored window longer than MAX_CLIP_SECONDS. The cap is on the
    window, not the upload: a long gallery clip trimmed to one lift is fine.
    An unknown length (no ffprobe) cannot be measured and is let through.
    """
    cap = max_clip_seconds()
    if not cap or end_sec <= start_sec:
        return
    length = end_sec - start_sec
    if length > cap + 0.05:
        raise DemoRefused(
            400,
            "too_long",
            f"That window is {length:.0f} s long, and this demo scores at most {cap:.0f} s at a time. "
            "Trim it to the one lift.",
        )


# ---------- the caller's address ----------


def _parse_address(host: str):
    try:
        return ipaddress.ip_address(host.strip().strip("[]").split("%", 1)[0])
    except ValueError:
        return None


def _is_trusted_proxy(host: str, trusted: List[str]) -> bool:
    """Whether `host` is one of the configured proxies. `*` is the caller's business."""
    address = _parse_address(host)
    if address is None:
        return False
    for entry in trusted:
        if entry == ANY_PEER:
            continue
        if entry == PRIVATE_RANGES:
            if any(address.version == net.version and address in net for net in _PRIVATE_NETWORKS):
                return True
            continue
        try:
            network = ipaddress.ip_network(entry, strict=False)
        except ValueError:
            continue
        if address.version == network.version and address in network:
            return True
    return False


def client_ip(request) -> str:
    """
    The visitor's address for the per-IP limit.

    X-Forwarded-For is believed only when the connection comes from a proxy
    named in TRUSTED_PROXIES. Otherwise the socket's peer address is used and
    the header is ignored, so a direct caller cannot choose their own address
    by forging it — which used to buy a fresh hourly allowance per header.

    Behind a trusted proxy the visitor is the rightmost entry that is not
    itself a trusted proxy: that is the one the proxy appended, and anything to
    its left came from the client. With `*` only the peer is trusted, so the
    last entry — the one that proxy wrote — is the answer.
    """
    client = getattr(request, "client", None)
    peer = getattr(client, "host", None) or "unknown"
    trusted = trusted_proxies()
    if not trusted:
        return peer
    if ANY_PEER not in trusted and not _is_trusted_proxy(peer, trusted):
        return peer
    headers = getattr(request, "headers", None) or {}
    forwarded = headers.get("x-forwarded-for") or ""
    chain = [part.strip() for part in forwarded.split(",") if part.strip()]
    if not chain:
        return peer
    for address in reversed(chain):
        if not _is_trusted_proxy(address, trusted):
            return address
    return chain[0]


# ---------- the counters ----------


class DemoGuard:
    """Per-process counters behind the daily, per-IP and concurrency limits."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._day = -1
        self._day_calls = 0
        self._by_ip: Dict[str, Deque[float]] = {}
        self._running = 0

    @property
    def running(self) -> int:
        return self._running

    def used_today(self) -> int:
        """Model calls charged so far today."""
        self._roll_day(self._clock())
        return self._day_calls

    def admit(self, ip: str) -> None:
        """
        Take a slot for one scoring run from `ip`, or raise DemoRefused.
        Nothing is counted yet — calls are charged as they are made — but a run
        is turned away up front unless there is room for a whole lift, so no
        one pays for the vision call and then loses the verdict. A refusal
        records nothing. Pair every admit with a release().
        """
        now = self._clock()
        concurrent = max_concurrent_jobs()
        if concurrent and self._running >= concurrent:
            raise DemoRefused(
                429,
                "busy",
                "The demo is scoring other clips right now. Give it a minute and try again.",
                retry_after=BUSY_RETRY_SECONDS,
            )
        self._check_ip(ip, now, CALLS_PER_LIFT)
        self._check_day(now, CALLS_PER_LIFT)
        self._running += 1

    def charge(self, ip: str) -> None:
        """
        Count one model call for `ip`. Raises BudgetExhausted — and counts
        nothing — when the hour's or the day's budget is spent, so the call is
        never made. Checked per call because what a session clip will cost is
        not known when it is admitted.
        """
        now = self._clock()
        try:
            self._check_ip(ip, now, 1)
            self._check_day(now, 1)
        except DemoRefused as exc:
            if exc.reason == "ip_limit":
                minutes = max(1, math.ceil((exc.retry_after or 60) / 60))
                detail = f"This hour's share of the demo ran out part-way through this clip. Try again in {minutes} min."
            else:
                detail = "Today's demo budget ran out part-way through this clip. Come back tomorrow."
            raise BudgetExhausted(exc.status, exc.reason, detail, retry_after=exc.retry_after) from None
        self._recent(ip, now).append(now)
        self._day_calls += 1
        self._prune(now)

    def meter(self, ip: str) -> Callable[[], None]:
        """A no-argument charge for `ip`, to hand to the model client."""
        return lambda: self.charge(ip)

    def release(self) -> None:
        """A run has ended, pass or fail."""
        self._running = max(0, self._running - 1)

    def _check_ip(self, ip: str, now: float, needed: int) -> None:
        per_ip = ip_hourly_limit()
        if not per_ip:
            return
        needed = min(needed, per_ip)  # a limit under one lift's cost still admits
        recent = self._recent(ip, now)
        if len(recent) + needed <= per_ip:
            return
        # Room appears when enough of the oldest calls fall out of the hour.
        short_by = len(recent) + needed - per_ip
        wait = recent[short_by - 1] + HOUR_SECONDS - now
        minutes = max(1, math.ceil(wait / 60))
        clips = max(1, per_ip // CALLS_PER_LIFT)
        raise DemoRefused(
            429,
            "ip_limit",
            f"You've used this hour's share of the demo ({per_ip} model calls, about {clips} clips). "
            f"Try again in {minutes} min.",
            retry_after=int(wait) + 1,
        )

    def _check_day(self, now: float, needed: int) -> None:
        self._roll_day(now)
        daily = daily_limit()
        if not daily:
            return
        needed = min(needed, daily)
        if self._day_calls + needed <= daily:
            return
        wait = int((self._day + 1) * DAY_SECONDS - now) + 1
        clips = max(1, daily // CALLS_PER_LIFT)
        raise DemoRefused(
            429,
            "daily_limit",
            f"Today's demo budget ({daily} model calls, about {clips} clips) is used up. Come back tomorrow.",
            retry_after=wait,
        )

    def _roll_day(self, now: float) -> None:
        today = int(now // DAY_SECONDS)
        if today != self._day:
            self._day = today
            self._day_calls = 0

    def _recent(self, ip: str, now: float) -> Deque[float]:
        window = self._by_ip.setdefault(ip, deque())
        cutoff = now - HOUR_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()
        return window

    def _prune(self, now: float) -> None:
        if len(self._by_ip) <= IP_TABLE_SOFT_LIMIT:
            return
        cutoff = now - HOUR_SECONDS
        for ip in [ip for ip, window in self._by_ip.items() if not window or window[-1] <= cutoff]:
            del self._by_ip[ip]


_guard: Optional[DemoGuard] = None


def get_guard() -> DemoGuard:
    global _guard
    if _guard is None:
        _guard = DemoGuard()
    return _guard


def reset_guard(guard: Optional[DemoGuard] = None) -> DemoGuard:
    """Replace the process-wide guard. Used by tests."""
    global _guard
    _guard = guard
    return get_guard()
