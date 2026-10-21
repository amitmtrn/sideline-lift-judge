"""
Spend caps for the public demo.

Token Factory bills per call, and a demo URL in a submission is open to anyone
who finds it. The account is capped at the provider (prepaid credits only, a
key made just for the demo so it can be revoked); these are the layers inside
the app, so the worst case is a few dollars rather than the whole balance:

    access code    DEMO_ACCESS_CODE        judges get a link carrying ?code=…
    daily budget   DEMO_DAILY_LIMIT        clips scored per UTC day, everyone
    per-IP rate    DEMO_IP_HOURLY_LIMIT    clips per IP per rolling hour
    concurrency    MAX_CONCURRENT_JOBS     scoring runs in flight at once
    clip length    MAX_CLIP_SECONDS        the scored window, after trimming
    frames         MAX_FRAMES              images per lift (see pipeline.py)

Every limit is off until its variable is set, so local runs, the CLIs and the
tests are untouched. Counters live in this process, like the job store: a
restart resets the day's count, which errs on the side of letting a judge in.
A limit turns a request away with a plain sentence and a machine-readable
reason; the page uses the reason to offer the cached sample clip instead.
"""
from __future__ import annotations

import hmac
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


class DemoRefused(Exception):
    """A request the demo turns away, with what the HTTP layer needs to say so."""

    def __init__(self, status: int, reason: str, detail: str, *, retry_after: Optional[int] = None) -> None:
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail
        self.retry_after = retry_after


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
    return int(_number_env("DEMO_DAILY_LIMIT"))


def ip_hourly_limit() -> int:
    return int(_number_env("DEMO_IP_HOURLY_LIMIT"))


def max_concurrent_jobs() -> int:
    return int(_number_env("MAX_CONCURRENT_JOBS"))


def max_clip_seconds() -> float:
    return _number_env("MAX_CLIP_SECONDS")


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


def client_ip(request) -> str:
    """
    The caller's address for the per-IP limit. A hosted deploy sits behind a
    proxy, so the first X-Forwarded-For entry wins when present. A direct
    caller could forge that header, but all it buys them is one IP's share:
    the daily budget still holds.
    """
    forwarded = request.headers.get("x-forwarded-for") if request.headers else None
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    client = getattr(request, "client", None)
    host = getattr(client, "host", None)
    return host or "unknown"


class DemoGuard:
    """Per-process counters behind the daily, per-IP and concurrency limits."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._day = -1
        self._day_count = 0
        self._by_ip: Dict[str, Deque[float]] = {}
        self._running = 0

    @property
    def running(self) -> int:
        return self._running

    def used_today(self) -> int:
        self._roll_day(self._clock())
        return self._day_count

    def admit(self, ip: str) -> None:
        """
        Count one scoring run for `ip`, or raise DemoRefused. Nothing is
        recorded when a request is refused, so a refusal never costs the caller
        part of their allowance. Pair every admit with a release().
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

        recent = self._recent(ip, now)
        per_ip = ip_hourly_limit()
        if per_ip and len(recent) >= per_ip:
            wait = recent[0] + HOUR_SECONDS - now
            minutes = max(1, math.ceil(wait / 60))
            raise DemoRefused(
                429,
                "ip_limit",
                f"You've scored {per_ip} clips in the last hour, which is this demo's limit. "
                f"Try again in {minutes} min.",
                retry_after=int(wait) + 1,
            )

        self._roll_day(now)
        daily = daily_limit()
        if daily and self._day_count >= daily:
            wait = int((self._day + 1) * DAY_SECONDS - now) + 1
            raise DemoRefused(
                429,
                "daily_limit",
                f"Today's demo budget of {daily} clips is used up. Come back tomorrow.",
                retry_after=wait,
            )

        recent.append(now)
        self._day_count += 1
        self._running += 1
        self._prune(now)

    def release(self) -> None:
        """A run has ended, pass or fail."""
        self._running = max(0, self._running - 1)

    def _roll_day(self, now: float) -> None:
        today = int(now // DAY_SECONDS)
        if today != self._day:
            self._day = today
            self._day_count = 0

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
