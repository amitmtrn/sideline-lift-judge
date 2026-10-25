"""The spend caps for the public demo, counted against a fake clock."""
import pytest

from server.services.demo_guard import (
    CALLS_PER_LIFT,
    DAY_SECONDS,
    DEFAULT_JOB_TIMEOUT_SECONDS,
    HOUR_SECONDS,
    BudgetExhausted,
    DemoGuard,
    DemoRefused,
    access_codes,
    access_required,
    check_access,
    client_ip,
    demo_caps_set,
    enforce_clip_length,
    job_timeout_seconds,
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
    from tests.conftest import DEMO_ENV_VARS

    for name in DEMO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def refused(guard, ip="1.2.3.4"):
    with pytest.raises(DemoRefused) as caught:
        guard.admit(ip)
    return caught.value


def charge(guard, ip, calls):
    for _ in range(calls):
        guard.charge(ip)


# ---------- nothing configured ----------


def test_with_nothing_set_everything_is_admitted_and_counted(guard):
    for _ in range(100):
        guard.admit("1.2.3.4")
    assert guard.running == 100
    charge(guard, "1.2.3.4", 100)
    assert guard.used_today() == 100
    assert demo_caps_set() is False


def test_access_is_open_when_no_code_is_configured():
    assert access_required() is False
    check_access(None)
    check_access("anything")


# ---------- access code ----------


def test_access_codes_are_comma_separated_and_trimmed(monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", " judge-1, judge-2 ,, ")
    assert access_codes() == ["judge-1", "judge-2"]
    assert access_required() is True
    assert demo_caps_set() is True


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


# ---------- the budgets count model calls ----------


def test_a_run_is_admitted_only_with_room_for_a_whole_lift(guard, monkeypatch):
    # Two clips' worth of calls an hour. Admission does not count anything...
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", str(2 * CALLS_PER_LIFT))
    guard.admit("1.2.3.4")
    assert guard.used_today() == 0
    # ...the calls do. After one clip there is room for another whole one.
    charge(guard, "1.2.3.4", CALLS_PER_LIFT)
    guard.admit("1.2.3.4")
    charge(guard, "1.2.3.4", CALLS_PER_LIFT - 1)
    # One call of room is not a lift: a third run is turned away up front,
    # rather than after it has paid for its vision call.
    error = refused(guard)
    assert error.status == 429 and error.reason == "ip_limit"
    # Plain English: a wait and the sample, never a call count or a variable.
    assert error.detail.startswith("You've hit the hourly limit. Try again in ")
    assert "see the sample" in error.detail
    assert "model" not in error.detail and "DEMO_" not in error.detail


def test_per_ip_limit_is_a_rolling_hour(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "6")
    guard.admit("1.2.3.4")
    charge(guard, "1.2.3.4", 3)
    clock.now += 600
    guard.admit("1.2.3.4")
    charge(guard, "1.2.3.4", 3)
    error = refused(guard)
    assert error.status == 429 and error.reason == "ip_limit"
    # Room for a lift appears when the first three calls, made 10 minutes ago,
    # fall out of the hour: 50 minutes from now.
    assert "50 min" in error.detail
    assert 0 < error.retry_after <= HOUR_SECONDS - 600 + 1
    # Another address is not held back by this one.
    guard.admit("5.6.7.8")
    # Once those calls leave the window, the address is back.
    clock.now += HOUR_SECONDS - 600 + 1
    guard.admit("1.2.3.4")


def test_a_refusal_does_not_count_against_the_caller(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "3")
    guard.admit("1.2.3.4")
    charge(guard, "1.2.3.4", 3)
    for _ in range(5):
        refused(guard)
    with pytest.raises(BudgetExhausted):
        guard.charge("1.2.3.4")
    assert guard.used_today() == 3  # neither the refusals nor the refused call counted
    clock.now += HOUR_SECONDS + 1
    guard.admit("1.2.3.4")


def test_running_out_mid_run_stops_the_next_call_with_a_plain_sentence(guard, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "4")
    guard.admit("a")  # room for three: admitted
    charge(guard, "a", 4)  # a clean & jerk needs five
    with pytest.raises(BudgetExhausted) as caught:
        guard.charge("a")
    error = caught.value
    assert isinstance(error, DemoRefused)
    assert error.status == 429 and error.reason == "daily_limit"
    assert "part-way through this clip" in error.detail
    assert guard.used_today() == 4

    monkeypatch.setenv("DEMO_DAILY_LIMIT", "0")
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "4")
    with pytest.raises(BudgetExhausted) as caught:
        guard.charge("a")
    assert caught.value.reason == "ip_limit"
    assert "part-way" in caught.value.detail and "min" in caught.value.detail


def test_daily_budget_counts_everyone_and_resets_at_utc_midnight(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "9")
    clock.now = 1_700_000_000.0  # mid-day UTC
    for ip in ("a", "b", "c"):
        guard.admit(ip)
        charge(guard, ip, 3)
    error = refused(guard, "d")
    assert error.status == 429 and error.reason == "daily_limit"
    assert error.detail == "Today's free demo limit is used up. See the sample result, or come back tomorrow."
    next_midnight = (clock.now // DAY_SECONDS + 1) * DAY_SECONDS
    assert error.retry_after == int(next_midnight - clock.now) + 1
    clock.now = next_midnight + 1
    guard.admit("d")
    guard.charge("d")
    assert guard.used_today() == 1


def test_a_limit_smaller_than_one_lift_still_admits(guard, monkeypatch):
    # Misconfigured, but it must not lock everyone out forever: the per-call
    # charge is what enforces it.
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "2")
    guard.admit("a")
    charge(guard, "a", 2)
    with pytest.raises(BudgetExhausted):
        guard.charge("a")


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


def test_busy_is_checked_before_the_allowance(guard, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "3")
    guard.admit("a")
    refused(guard, "b")  # busy
    guard.release()
    guard.admit("b")  # the busy refusal did not use b's hour


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


# ---------- the run timeout ----------


def test_the_run_timeout_is_on_unless_set_to_zero(monkeypatch):
    assert job_timeout_seconds() == DEFAULT_JOB_TIMEOUT_SECONDS
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "240")
    assert job_timeout_seconds() == 240
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "0")
    assert job_timeout_seconds() == 0
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "forever")
    assert job_timeout_seconds() == DEFAULT_JOB_TIMEOUT_SECONDS


# ---------- the caller's address ----------


class FakeRequest:
    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = type("Client", (), {"host": host})()


FORGED = {"x-forwarded-for": "203.0.113.9, 10.0.0.2"}


def test_without_trusted_proxies_the_forwarded_header_is_ignored():
    # A direct caller used to pick their own address — and a fresh hourly
    # allowance — by sending the header. Now it is the socket's peer, always.
    assert client_ip(FakeRequest(FORGED, host="198.51.100.7")) == "198.51.100.7"
    assert client_ip(FakeRequest()) == "10.0.0.1"
    assert client_ip(FakeRequest(host=None)) == "unknown"


def test_behind_a_trusted_proxy_the_visitor_is_the_rightmost_untrusted_entry(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXIES", "private")
    # Caddy on the container network appends (or sets) the real client last;
    # anything to its left was sent by the client and is not believed.
    assert client_ip(FakeRequest({"x-forwarded-for": "1.1.1.1, 9.9.9.9"}, host="172.18.0.3")) == "9.9.9.9"
    assert client_ip(FakeRequest({"x-forwarded-for": "9.9.9.9"}, host="127.0.0.1")) == "9.9.9.9"
    assert client_ip(FakeRequest({"x-forwarded-for": "2001:db8::9"}, host="fd00::3")) == "2001:db8::9"
    # Two trusted hops (a CDN in front of the proxy) are both skipped.
    monkeypatch.setenv("TRUSTED_PROXIES", "private, 198.51.100.0/24")
    assert client_ip(FakeRequest({"x-forwarded-for": "9.9.9.9, 198.51.100.4"}, host="172.18.0.3")) == "9.9.9.9"
    # Only the named ranges are private: a routable peer is not a proxy, so
    # from it the header still means nothing.
    assert client_ip(FakeRequest(FORGED, host="8.8.8.8")) == "8.8.8.8"
    # No header from a trusted peer: the peer itself.
    assert client_ip(FakeRequest({}, host="172.18.0.3")) == "172.18.0.3"
    assert client_ip(FakeRequest({"x-forwarded-for": "  "}, host="172.18.0.3")) == "172.18.0.3"


def test_star_trusts_the_peer_only_and_takes_the_last_entry(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXIES", "*")
    assert client_ip(FakeRequest({"x-forwarded-for": "1.1.1.1, 203.0.113.9"}, host="testclient")) == "203.0.113.9"
    assert client_ip(FakeRequest({}, host="testclient")) == "testclient"


def test_a_peer_that_is_not_an_address_is_never_a_trusted_proxy(monkeypatch):
    monkeypatch.setenv("TRUSTED_PROXIES", "private, not-an-address")
    assert client_ip(FakeRequest(FORGED, host="testclient")) == "testclient"
