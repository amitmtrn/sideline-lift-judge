"""The spend caps for the public demo, counted against a fake clock."""
import pytest

from server.services.demo_guard import (
    DAY_SECONDS,
    HOUR_SECONDS,
    DemoGuard,
    DemoRefused,
    access_codes,
    access_required,
    check_access,
    client_ip,
    enforce_clip_length,
    max_clip_seconds,
)


class Clock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def guard(clock):
    return DemoGuard(clock=clock)


@pytest.fixture(autouse=True)
def no_limits(monkeypatch):
    for name in ("DEMO_ACCESS_CODE", "DEMO_DAILY_LIMIT", "DEMO_IP_HOURLY_LIMIT", "MAX_CONCURRENT_JOBS", "MAX_CLIP_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def refused(guard, ip="1.2.3.4"):
    with pytest.raises(DemoRefused) as caught:
        guard.admit(ip)
    return caught.value


# ---------- nothing configured ----------


def test_with_nothing_set_everything_is_admitted(guard):
    for _ in range(100):
        guard.admit("1.2.3.4")
    assert guard.running == 100
    assert guard.used_today() == 100


def test_access_is_open_when_no_code_is_configured():
    assert access_required() is False
    check_access(None)
    check_access("anything")


# ---------- access code ----------


def test_access_codes_are_comma_separated_and_trimmed(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", " judge-1, judge-2 ,, ")
    assert access_codes() == ["judge-1", "judge-2"]
    assert access_required() is True


def test_missing_code_is_401_and_wrong_code_is_403(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1,judge-2")
    with pytest.raises(DemoRefused) as missing:
        check_access(None)
    assert missing.value.status == 401 and missing.value.reason == "access_code"
    with pytest.raises(DemoRefused) as blank:
        check_access("   ")
    assert blank.value.status == 401
    with pytest.raises(DemoRefused) as wrong:
        check_access("judge-3")
    assert wrong.value.status == 403 and wrong.value.reason == "access_code"
    check_access("judge-2")
    check_access("  judge-1 ")  # whitespace around a pasted code is forgiven


# ---------- per-IP ----------


def test_per_ip_limit_is_a_rolling_hour(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "2")
    guard.admit("1.2.3.4")
    clock.now += 600
    guard.admit("1.2.3.4")
    error = refused(guard)
    assert error.status == 429 and error.reason == "ip_limit"
    assert "2 clips" in error.detail
    # The first admission is 50 minutes old, so that is how long to wait.
    assert 0 < error.retry_after <= HOUR_SECONDS - 600 + 1
    assert "50 min" in error.detail
    # Another address is not held back by this one.
    guard.admit("5.6.7.8")
    # Once the oldest admission falls out of the hour, the address is back.
    clock.now += HOUR_SECONDS - 600 + 1
    guard.admit("1.2.3.4")


def test_a_refusal_does_not_count_against_the_caller(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "1")
    guard.admit("1.2.3.4")
    for _ in range(5):
        refused(guard)
    clock.now += HOUR_SECONDS + 1
    guard.admit("1.2.3.4")  # the five refusals did not extend the wait


# ---------- daily budget ----------


def test_daily_budget_counts_everyone_and_resets_at_utc_midnight(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    clock.now = 1_700_000_000.0  # mid-day UTC
    for ip in ("a", "b", "c"):
        guard.admit(ip)
    error = refused(guard, "d")
    assert error.status == 429 and error.reason == "daily_limit"
    assert "3 clips" in error.detail
    next_midnight = (clock.now // DAY_SECONDS + 1) * DAY_SECONDS
    assert error.retry_after == int(next_midnight - clock.now) + 1
    clock.now = next_midnight + 1
    guard.admit("d")
    assert guard.used_today() == 1


# ---------- concurrency ----------


def test_only_so_many_runs_at_once(guard, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "2")
    guard.admit("a")
    guard.admit("b")
    error = refused(guard, "c")
    assert error.status == 429 and error.reason == "busy"
    assert error.retry_after == 60
    guard.release()
    guard.admit("c")
    assert guard.running == 2


def test_release_never_goes_negative(guard):
    guard.release()
    assert guard.running == 0


def test_busy_is_checked_before_the_allowance_is_spent(guard, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "1")
    guard.admit("a")
    refused(guard, "b")  # busy
    guard.release()
    guard.admit("b")  # the busy refusal did not use b's one clip


# ---------- clip length ----------


def test_clip_cap_is_off_until_set():
    assert max_clip_seconds() == 0
    enforce_clip_length(0.0, 600.0)


def test_clip_cap_applies_to_the_window_not_the_upload(monkeypatch):
    monkeypatch.setenv("MAX_CLIP_SECONDS", "15")
    enforce_clip_length(0.0, 15.0)
    enforce_clip_length(40.0, 55.04)  # slider rounding is forgiven
    with pytest.raises(DemoRefused) as caught:
        enforce_clip_length(0.0, 23.0)
    assert caught.value.status == 400 and caught.value.reason == "too_long"
    assert "23 s" in caught.value.detail and "15 s" in caught.value.detail
    # An unknown length (no ffprobe) cannot be measured, so it is let through.
    enforce_clip_length(0.0, 0.0)


def test_bad_numbers_mean_off(monkeypatch):
    monkeypatch.setenv("MAX_CLIP_SECONDS", "fifteen")
    assert max_clip_seconds() == 0
    monkeypatch.setenv("MAX_CLIP_SECONDS", "-4")
    assert max_clip_seconds() == 0


# ---------- the caller's address ----------


class FakeRequest:
    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": host})()


def test_client_ip_prefers_the_first_forwarded_address():
    assert client_ip(FakeRequest({"x-forwarded-for": "203.0.113.9, 10.0.0.2"})) == "203.0.113.9"
    assert client_ip(FakeRequest({"x-forwarded-for": "  "})) == "10.0.0.1"
    assert client_ip(FakeRequest()) == "10.0.0.1"
    assert client_ip(FakeRequest(host=None)) == "unknown"
