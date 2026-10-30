"""
Small per-process budgets for the Pro features' model calls.

The judge's own budget (demo_guard) stays as it is. The coach draft and the
ops agent get budgets of their own — DRAFT_DAILY_LIMIT and AGENT_DAILY_LIMIT
model calls per day, AGENT_HOURLY_LIMIT agent requests per visitor per hour —
so a judge playing with the agent cannot spend the clip-scoring budget, and
the payment loop on the cached sample always has a draft (the template
fallback costs nothing). The day turns where DEMO_DAY_TIMEZONE says.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Deque, Dict, Optional

from server.services.demo_guard import HOUR_SECONDS, day_timezone


class BudgetExceeded(Exception):
    def __init__(self, detail: str, *, retry_after: Optional[int] = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.retry_after = retry_after


class DailyBudget:
    """Calls per local day, plus an optional per-key rolling-hour count."""

    def __init__(self, name: str, limit: Callable[[], int], *, hourly: Optional[Callable[[], int]] = None, clock: Callable[[], float] = time.time) -> None:
        self.name = name
        self._limit = limit
        self._hourly = hourly
        self._clock = clock
        self._lock = threading.Lock()
        self._day = -1
        self._used = 0
        self._by_key: Dict[str, Deque[float]] = {}

    def _roll(self, now: float) -> None:
        today = datetime.fromtimestamp(now, day_timezone()).toordinal()
        if today != self._day:
            self._day = today
            self._used = 0

    def used_today(self) -> int:
        with self._lock:
            self._roll(self._clock())
            return self._used

    def remaining_today(self) -> Optional[int]:
        limit = self._limit()
        if not limit:
            return None
        return max(0, limit - self.used_today())

    def check(self, key: Optional[str] = None, calls: int = 1) -> None:
        """Raise unless `calls` more fit today (and this hour for `key`). Counts nothing."""
        now = self._clock()
        with self._lock:
            self._roll(now)
            limit = self._limit()
            if limit and self._used + calls > limit:
                raise BudgetExceeded(f"Today's {self.name} budget is used up. Try again tomorrow.")
            if key is not None and self._hourly is not None:
                per_hour = self._hourly()
                if per_hour:
                    recent = self._recent(key, now)
                    if len(recent) + 1 > per_hour:
                        wait = int(recent[0] + HOUR_SECONDS - now) + 1
                        raise BudgetExceeded(
                            f"You've hit the hourly limit for the {self.name}. Try again in {max(1, wait // 60 + 1)} minutes.",
                            retry_after=wait,
                        )

    def charge(self, key: Optional[str] = None, calls: int = 1) -> None:
        """Count `calls` (checking first)."""
        self.check(key, calls)
        now = self._clock()
        with self._lock:
            self._used += calls
            if key is not None and self._hourly is not None:
                self._recent(key, now).append(now)

    def meter(self, key: Optional[str] = None) -> Callable[[], None]:
        return lambda: self.charge(key)

    def _recent(self, key: str, now: float) -> Deque[float]:
        window = self._by_key.setdefault(key, deque())
        cutoff = now - HOUR_SECONDS
        while window and window[0] <= cutoff:
            window.popleft()
        return window


_draft: Optional[DailyBudget] = None
_agent: Optional[DailyBudget] = None


def draft_budget() -> DailyBudget:
    global _draft
    if _draft is None:
        from server.services.pro_config import draft_daily_limit

        _draft = DailyBudget("coach draft", draft_daily_limit)
    return _draft


def agent_budget() -> DailyBudget:
    global _agent
    if _agent is None:
        from server.services.pro_config import agent_daily_limit, agent_hourly_limit

        _agent = DailyBudget("ops agent", agent_daily_limit, hourly=agent_hourly_limit)
    return _agent


def reset_budgets_for_tests() -> None:
    global _draft, _agent
    _draft = None
    _agent = None
