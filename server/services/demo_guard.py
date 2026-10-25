"""
Spend caps for the public demo.

Token Factory bills per model call, and a demo URL in a submission is open to
anyone who finds it. The account is capped at the provider (prepaid credits
only, a key made just for the demo so it can be revoked); these are the layers
inside the app, so the worst case is a few dollars rather than the whole
balance:

    access mode     ACCESS_MODE             open | link | code (default code)
    access code     DEMO_ACCESS_CODE        the secret, for link and code modes
    kill switch     DEMO_DISABLED           true = the sample only, no scoring
    daily budget    DEMO_DAILY_LIMIT        model calls per day, everyone
    day boundary    DEMO_DAY_TIMEZONE       when the day resets (UTC unless set)
    newcomer share  DEMO_DAILY_RESERVE_PERCENT  the last slice of the day, kept
                                            for addresses that have not scored
                                            yet today (20 unless set)
    per-IP rate     DEMO_IP_HOURLY_LIMIT    model calls per address per rolling hour
    concurrency     MAX_CONCURRENT_JOBS     scoring runs in flight at once
    run time        JOB_TIMEOUT_SECONDS     wall clock per run (600 unless set; 0 = off)
    clip length     MAX_CLIP_SECONDS        the scored window, after trimming
    lifts per clip  MAX_LIFTS_PER_CLIP      windows a single upload may score
    frames          MAX_FRAMES              images per lift (see pipeline.py)
    proxies         TRUSTED_PROXIES         who may tell us the visitor's address

The budgets count model calls, not uploads. One lift costs three (triage,
observation, judge), a repair retry or the fallback judge one more each, and a
session clip one triage per window on top — so counting uploads left the bill
open-ended. A run is admitted when there is room for one whole lift; once the
clip's lift windows are known it must fit the budget whole (windows × three)
or it stops before the first call; and it is then charged call by call.

Every limit except the run timeout is off until its variable is set, so local
runs, the CLIs and the tests are untouched; the deploys set them all, and in
`open` mode the app refuses to start without them (see missing_open_mode_caps).
Counters live in this process, like the job store: a restart resets the day's
count, which errs on the side of letting a judge in. A limit turns a request
away with a plain sentence and a machine-readable reason; the page uses the
reason to offer the cached sample clip instead. The sentences name nothing a
visitor cannot act on — no variable, no model, no call count.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from datetime import time as time_of_day
from functools import lru_cache
from typing import Callable, Deque, Dict, List, Optional
from zoneinfo import ZoneInfo

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
DEFAULT_DAILY_RESERVE_PERCENT = 20.0
ACCESS_MODES = ("open", "link", "code")
DEFAULT_ACCESS_MODE = "code"
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
# but your proxy can reach it (Cloud Run, Fly, Heroku's router).
ANY_PEER = "*"

# What the visitor reads. One sentence each, and always a way on: the sample
# result, a wait, or tomorrow. Nothing here names a variable or a model.
DAILY_LIMIT_MESSAGE = "Today's free demo limit is used up. See the sample result, or come back tomorrow."
DAILY_RESERVE_MESSAGE = (
    "Today's free demo limit is nearly used up, and what is left is kept for first-time visitors. "
    "See the sample result, or come back tomorrow."
)
DAILY_LIMIT_MID_RUN_MESSAGE = (
    "Today's free demo limit ran out part-way through this clip. See the sample result, or come back tomorrow."
)
BUSY_MESSAGE = "Another lift is being scored. You're next; this usually takes under a minute."
DISABLED_MESSAGE = "The live demo is paused right now. The sample result shows everything it does."
CODE_MISSING_MESSAGE = "This demo needs an access code. It is on the submission page, or open the link that carries it."
CODE_WRONG_MESSAGE = "That access code is not right. Check it against the submission page."
LINK_MISSING_MESSAGE = "This demo opens from the link on the submission page. Use that link, or see the sample result."
LINK_WRONG_MESSAGE = "That link is out of date. Open the one on the submission page, or see the sample result."


def _minutes_phrase(seconds: float) -> str:
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def ip_limit_message(wait_seconds: float) -> str:
    return f"You've hit the hourly limit. Try again in {_minutes_phrase(wait_seconds)} or see the sample."


def ip_limit_mid_run_message(wait_seconds: float) -> str:
    return (
        f"You hit the hourly limit part-way through this clip. "
        f"Try again in {_minutes_phrase(wait_seconds)} or see the sample."
    )


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


def _flag_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in ("1", "true", "yes", "on")


def access_mode() -> str:
    """
    ACCESS_MODE: `open` (no code anywhere), `link` (the code rides in the
    URL the page is opened from, never typed) or `code` (typed on the first
    screen). Unset or unrecognised means `code`, the gated behaviour.
    """
    raw = (os.getenv("ACCESS_MODE") or "").strip().lower()
    if not raw:
        return DEFAULT_ACCESS_MODE
    if raw not in ACCESS_MODES:
        logger.warning("ACCESS_MODE=%r is not open, link or code; using %s", raw, DEFAULT_ACCESS_MODE)
        return DEFAULT_ACCESS_MODE
    return raw


def access_codes() -> List[str]:
    """DEMO_ACCESS_CODE, comma-separated so one code can be revoked without the rest."""
    raw = os.getenv("DEMO_ACCESS_CODE") or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


def access_required() -> bool:
    """Whether scoring needs a code: never in open mode, otherwise when one is configured."""
    return access_mode() != "open" and bool(access_codes())


def demo_disabled() -> bool:
    """DEMO_DISABLED=true: the kill switch. The sample still serves; nothing else runs."""
    return _flag_env("DEMO_DISABLED")


def daily_limit() -> int:
    """Model calls per day, all visitors together. 0 = no limit."""
    return int(_number_env("DEMO_DAILY_LIMIT"))


def daily_reserve_percent() -> float:
    """
    DEMO_DAILY_RESERVE_PERCENT: the share of the daily budget an address that
    has already scored a clip today may not touch, so one heavy user cannot
    lock out a judge arriving in the evening. 20 unless set; 0 turns it off.
    """
    raw = os.getenv("DEMO_DAILY_RESERVE_PERCENT")
    if raw is None or not raw.strip():
        return DEFAULT_DAILY_RESERVE_PERCENT
    try:
        value = float(raw)
    except ValueError:
        logger.warning("DEMO_DAILY_RESERVE_PERCENT=%r is not a number; using %s", raw, DEFAULT_DAILY_RESERVE_PERCENT)
        return DEFAULT_DAILY_RESERVE_PERCENT
    return min(90.0, max(0.0, value))


@lru_cache(maxsize=8)
def _zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except Exception:  # ZoneInfoNotFoundError, or a bad key
        logger.warning("DEMO_DAY_TIMEZONE=%r is not a known time zone; the day resets at UTC midnight", name)
        return timezone.utc


def day_timezone() -> tzinfo:
    """DEMO_DAY_TIMEZONE: where midnight is. UTC unless set (the deploys say Asia/Jerusalem)."""
    raw = (os.getenv("DEMO_DAY_TIMEZONE") or "").strip()
    return _zone(raw) if raw else timezone.utc


def ip_hourly_limit() -> int:
    """Model calls per visitor address per rolling hour. 0 = no limit."""
    return int(_number_env("DEMO_IP_HOURLY_LIMIT"))


def max_concurrent_jobs() -> int:
    return int(_number_env("MAX_CONCURRENT_JOBS"))


def max_clip_seconds() -> float:
    return _number_env("MAX_CLIP_SECONDS")


def max_lifts_per_clip() -> int:
    """Lift windows one upload may score (the rest are listed as skipped). 0 = no cap."""
    return int(_number_env("MAX_LIFTS_PER_CLIP"))


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


def missing_open_mode_caps() -> List[str]:
    """
    What `open` mode will not run without. With no code in front of it, every
    cap on spend must be set on purpose — unset or zero is a deploy that
    forgot one, and the app refuses to start rather than run uncapped. Each
    entry names the variable and what it must be.
    """
    from server.services.nebius_client import vision_max_images
    from server.services.pipeline import max_frames
    from server.utils.uploads import max_upload_bytes

    def is_set(name: str) -> bool:
        return bool((os.getenv(name) or "").strip())

    missing: List[str] = []
    if not daily_limit():
        missing.append("DEMO_DAILY_LIMIT (model calls per day)")
    if not ip_hourly_limit():
        missing.append("DEMO_IP_HOURLY_LIMIT (model calls per address per hour)")
    if not max_concurrent_jobs():
        missing.append("MAX_CONCURRENT_JOBS (runs at once)")
    if not max_clip_seconds():
        missing.append("MAX_CLIP_SECONDS (longest scored window)")
    if not is_set("MAX_UPLOAD_MB") or not max_upload_bytes():
        missing.append("MAX_UPLOAD_MB (upload ceiling)")
    if not is_set("MAX_FRAMES") or max_frames() > vision_max_images():
        missing.append(f"MAX_FRAMES (at most {vision_max_images()}: one vision call per lift)")
    if not max_lifts_per_clip():
        missing.append("MAX_LIFTS_PER_CLIP (lift windows one upload may score)")
    if not is_set("JOB_TIMEOUT_SECONDS") or not job_timeout_seconds():
        missing.append("JOB_TIMEOUT_SECONDS (wall clock per run)")
    return missing


# ---------- the checks ----------


def check_enabled() -> None:
    """Raise when the kill switch is on. Only the sample is served then."""
    if demo_disabled():
        raise DemoRefused(503, "disabled", DISABLED_MESSAGE)


def check_access(code: Optional[str]) -> None:
    """
    Raise unless `code` is one of the configured codes. Nothing is required in
    open mode, or when no code is configured. The sentence depends on how the
    code is meant to arrive: typed (code mode) or inside the link (link mode).
    """
    if not access_required():
        return
    by_link = access_mode() == "link"
    given = (code or "").strip()
    if not given:
        raise DemoRefused(401, "access_code", LINK_MISSING_MESSAGE if by_link else CODE_MISSING_MESSAGE)
    if not any(hmac.compare_digest(given.encode("utf-8"), known.encode("utf-8")) for known in access_codes()):
        raise DemoRefused(403, "access_code", LINK_WRONG_MESSAGE if by_link else CODE_WRONG_MESSAGE)


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
            f"Clips need to be under {cap:.0f} seconds. This window is {length:.0f} s long; trim it to the one lift.",
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
    last entry — the one that proxy wrote — is the answer. The first entry is
    never believed on its own: it is whatever the client chose to send.
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


# A random key per process: the log can tell one visitor's jobs from another's
# for as long as the process lives, and never holds an address.
_IP_HASH_KEY = os.urandom(16)


def ip_hash(ip: str) -> str:
    """A short, keyed digest of an address for the log line. Never the address."""
    return hashlib.blake2b((ip or "").encode("utf-8"), key=_IP_HASH_KEY, digest_size=6).hexdigest()


# ---------- the counters ----------


@dataclass
class Admission:
    """
    One admitted scoring run: who it is for, what it has cost so far, and the
    hooks the run uses — `meter()` for the model client, `reserve()` once the
    clip's windows are known, `release()` when it ends. Pair every admission
    with exactly one release.
    """

    guard: "DemoGuard" = field(repr=False)
    ip: str
    first_today: bool
    started_at: float
    calls: int = 0

    def meter(self) -> Callable[[], None]:
        """A no-argument charge for this run, to hand to the model client."""

        def charge() -> None:
            self.guard.charge(self.ip)
            self.calls += 1

        return charge

    def reserve(self, calls: int) -> None:
        """
        The run now knows what it will cost. Raise — before any model call is
        made — unless that much is left, so a clip is turned away whole rather
        than scored part-way and stopped.
        """
        self.guard.ensure_room(self.ip, calls)

    def release(self) -> None:
        self.guard.release(self)


class DemoGuard:
    """Per-process counters behind the daily, per-IP and concurrency limits."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._day = -1
        self._day_calls = 0
        # Scoring runs admitted today per address: the first is a newcomer's
        # and may use the whole budget; later ones stop short of the reserve.
        self._jobs_today: Dict[str, int] = {}
        self._by_ip: Dict[str, Deque[float]] = {}
        self._running = 0

    @property
    def running(self) -> int:
        return self._running

    def used_today(self) -> int:
        """Model calls charged so far today."""
        self._roll_day(self._clock())
        return self._day_calls

    def remaining_today(self) -> Optional[int]:
        """Model calls left in the day's budget, or None when there is no budget."""
        daily = daily_limit()
        if not daily:
            return None
        return max(0, daily - self.used_today())

    def precheck(self, ip: str, calls: int = CALLS_PER_LIFT) -> None:
        """
        Would a run from `ip` be admitted right now? Raises DemoRefused if
        not, and records nothing either way. The HTTP layer asks this from
        the request headers, before the upload body is read, so a refused
        clip is never received.
        """
        now = self._clock()
        concurrent = max_concurrent_jobs()
        if concurrent and self._running >= concurrent:
            raise DemoRefused(429, "busy", BUSY_MESSAGE, retry_after=BUSY_RETRY_SECONDS)
        self._check_ip(ip, now, calls)
        self._check_day(now, calls, ip)

    def admit(self, ip: str, calls: int = CALLS_PER_LIFT) -> Admission:
        """
        Take a slot for one scoring run from `ip`, or raise DemoRefused.
        Nothing is counted yet — calls are charged as they are made — but a run
        is turned away up front unless there is room for a whole lift, so no
        one pays for the vision call and then loses the verdict. A refusal
        records nothing. Pair every admit with a release().
        """
        now = self._clock()
        self.precheck(ip, calls)
        first_today = self._jobs_today.get(ip, 0) == 0
        self._jobs_today[ip] = self._jobs_today.get(ip, 0) + 1
        self._running += 1
        return Admission(guard=self, ip=ip, first_today=first_today, started_at=now)

    def ensure_room(self, ip: str, calls: int) -> None:
        """
        Raise unless `calls` more model calls fit this hour's and today's
        budgets. Counts nothing. A clip that needs more than one lift's worth
        is told so in as many words, since trimming it is the way through.
        """
        now = self._clock()
        try:
            self._check_ip(ip, now, calls)
            self._check_day(now, calls, ip)
        except DemoRefused as exc:
            if calls <= CALLS_PER_LIFT:
                raise
            lifts = max(1, math.ceil(calls / CALLS_PER_LIFT))
            if exc.reason == "ip_limit":
                detail = (
                    f"This clip holds {lifts} lifts, more than this hour's limit has room for. "
                    f"Trim it to one lift, or try again in {_minutes_phrase(exc.retry_after or 60)}."
                )
            else:
                detail = (
                    f"This clip holds {lifts} lifts, more than today's free demo has room for. "
                    "Trim it to one lift, see the sample result, or come back tomorrow."
                )
            raise DemoRefused(exc.status, exc.reason, detail, retry_after=exc.retry_after) from None

    def charge(self, ip: str) -> None:
        """
        Count one model call for `ip`. Raises BudgetExhausted — and counts
        nothing — when the hour's or the day's budget is spent, so the call is
        never made. Checked per call because what a clip will cost is not
        known exactly when it is admitted.
        """
        now = self._clock()
        try:
            self._check_ip(ip, now, 1)
            self._check_day(now, 1, ip)
        except DemoRefused as exc:
            if exc.reason == "ip_limit":
                detail = ip_limit_mid_run_message(exc.retry_after or 60)
            else:
                detail = DAILY_LIMIT_MID_RUN_MESSAGE
            raise BudgetExhausted(exc.status, exc.reason, detail, retry_after=exc.retry_after) from None
        self._recent(ip, now).append(now)
        self._day_calls += 1
        self._prune(now)

    def meter(self, ip: str) -> Callable[[], None]:
        """A no-argument charge for `ip`, to hand to the model client."""
        return lambda: self.charge(ip)

    def release(self, admission: Optional[Admission] = None) -> None:
        """
        A run has ended, pass or fail. One that cost nothing — stopped by its
        own estimate before the first call, or by a fault — is forgotten
        entirely, so the address is still a newcomer for the reserve.
        """
        self._running = max(0, self._running - 1)
        if admission is not None and admission.calls == 0:
            left = self._jobs_today.get(admission.ip, 0) - 1
            if left > 0:
                self._jobs_today[admission.ip] = left
            else:
                self._jobs_today.pop(admission.ip, None)

    @staticmethod
    def _room_needed(needed: int, limit: int) -> int:
        # A limit under one lift's cost still admits one lift — the per-call
        # charge enforces it — so a misconfiguration never locks everyone out.
        # A bigger estimate is held to the limit as it stands.
        return min(needed, limit) if needed <= CALLS_PER_LIFT else needed

    def _check_ip(self, ip: str, now: float, needed: int) -> None:
        per_ip = ip_hourly_limit()
        if not per_ip:
            return
        needed = self._room_needed(needed, per_ip)
        recent = self._recent(ip, now)
        if len(recent) + needed <= per_ip:
            return
        # Room appears when enough of the oldest calls fall out of the hour.
        short_by = len(recent) + needed - per_ip
        wait = recent[min(short_by, len(recent)) - 1] + HOUR_SECONDS - now if recent else HOUR_SECONDS
        raise DemoRefused(429, "ip_limit", ip_limit_message(wait), retry_after=int(wait) + 1)

    def _check_day(self, now: float, needed: int, ip: Optional[str] = None) -> None:
        self._roll_day(now)
        daily = daily_limit()
        if not daily:
            return
        needed = self._room_needed(needed, daily)
        # A returning address stops short of the slice kept for newcomers;
        # an address on its first run today may use the whole budget.
        returning = ip is not None and self._jobs_today.get(ip, 0) > 1
        ceiling = daily
        if returning:
            ceiling = daily - math.ceil(daily * daily_reserve_percent() / 100)
        if self._day_calls + needed <= ceiling:
            return
        wait = int(self._next_midnight(now) - now) + 1
        if returning and self._day_calls + needed <= daily:
            raise DemoRefused(429, "daily_limit", DAILY_RESERVE_MESSAGE, retry_after=wait)
        raise DemoRefused(429, "daily_limit", DAILY_LIMIT_MESSAGE, retry_after=wait)

    def _roll_day(self, now: float) -> None:
        today = datetime.fromtimestamp(now, day_timezone()).toordinal()
        if today != self._day:
            self._day = today
            self._day_calls = 0
            self._jobs_today.clear()

    @staticmethod
    def _next_midnight(now: float) -> float:
        """The next local midnight (DEMO_DAY_TIMEZONE) as a timestamp, clock changes included."""
        zone = day_timezone()
        local = datetime.fromtimestamp(now, zone)
        tomorrow = datetime.combine(local.date() + timedelta(days=1), time_of_day.min, tzinfo=zone)
        return tomorrow.timestamp()

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
