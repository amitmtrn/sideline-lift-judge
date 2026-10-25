"""
The open public demo: no access code, bounded by server-side caps, and never
a dead end — the sample result is always one tap away.

Covers the PRD's protection layers (daily budget and its local-midnight
reset, the per-IP limit against forged headers, the run deadline, the
pre-admission cost estimate, the newcomer reserve, the lifts-per-clip cap),
the three access modes, the kill switch, the docs toggle and the startup
refusal. Model calls are mocked throughout.
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from server.services.demo_guard import (
    CALLS_PER_LIFT,
    HOUR_SECONDS,
    DemoGuard,
    DemoRefused,
    access_mode,
    access_required,
    check_access,
    demo_disabled,
    missing_open_mode_caps,
)
from tests.conftest import (  # noqa: F401  (client and clip are fixtures)
    DEMO_ENV_VARS,
    SNATCH_PATH,
    client,
    clip,
    judge_reply,
    observation_reply,
    triage_reply,
    upload,
    wait_for,
    write_lift_video,
)
from tests.test_demo_api import ONE_SNATCH, post_clip, sample_dir  # noqa: F401  (sample_dir is a fixture)

# Every cap open mode insists on, with the PRD's values.
OPEN_MODE_CAPS = {
    "DEMO_DAILY_LIMIT": "150",
    "DEMO_IP_HOURLY_LIMIT": "12",
    "MAX_CONCURRENT_JOBS": "1",
    "MAX_CLIP_SECONDS": "15",
    "MAX_UPLOAD_MB": "60",
    "MAX_FRAMES": "10",
    "MAX_LIFTS_PER_CLIP": "2",
    "JOB_TIMEOUT_SECONDS": "120",
}


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
    for name in DEMO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def set_open_mode_caps(monkeypatch, **overrides):
    for name, value in {**OPEN_MODE_CAPS, **overrides}.items():
        monkeypatch.setenv(name, value)


def two_lift_clip(tmp_path):
    """A 9 s stand-in for a session clip: two rises far enough apart to be two windows."""
    return write_lift_video(str(tmp_path / "two.mp4"), busy_windows=((1.0, 3.0), (5.0, 7.0)), duration=9.0)


def sync_post(client, clip, **data):
    with open(clip, "rb") as handle:
        return client.post("/api/score", data={"lift": "snatch", **data}, files={"video": ("clip.mp4", handle, "video/mp4")})


# ---------- the modes ----------


def test_the_mode_defaults_to_code_and_an_unknown_value_falls_back_to_it(monkeypatch):
    assert access_mode() == "code"
    monkeypatch.setenv("ACCESS_MODE", " Open ")
    assert access_mode() == "open"
    monkeypatch.setenv("ACCESS_MODE", "link")
    assert access_mode() == "link"
    monkeypatch.setenv("ACCESS_MODE", "anyone")
    assert access_mode() == "code"


def test_open_mode_needs_no_code_even_when_one_is_set(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    monkeypatch.setenv("ACCESS_MODE", "open")
    assert access_required() is False
    check_access(None)
    check_access("wrong")

    page = client.get("/").text
    assert 'accessMode: "open"' in page and "accessRequired: false" in page
    # R2: the code field is not on the page at all in open mode.
    assert 'id="access-field" hidden' in page
    assert "No account, no code." in page

    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"
    assert client.get("/api/access/check").json() == {"ok": True, "mode": "open"}


def test_link_mode_hides_the_field_but_still_gates_the_upload(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    monkeypatch.setenv("ACCESS_MODE", "link")
    page = client.get("/").text
    assert 'accessMode: "link"' in page and "accessRequired: true" in page
    assert 'id="access-field" hidden' in page
    assert "Scoring opens from the link on the submission page." in page

    missing = post_clip(client, clip)
    assert missing.status_code == 401 and missing.json()["reason"] == "access_code"
    assert "link on the submission page" in missing.json()["detail"]
    assert "see the sample result" in missing.json()["detail"]
    wrong = post_clip(client, clip, headers={"X-Access-Code": "judge-9"})
    assert wrong.status_code == 403 and "link is out of date" in wrong.json()["detail"]

    _, payload = upload(client, clip, ONE_SNATCH(), headers={"X-Access-Code": "judge-1"})
    assert payload["status"] == "done"


def test_code_mode_shows_the_field_and_is_the_old_behaviour(client, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    page = client.get("/").text
    assert 'accessMode: "code"' in page and "accessRequired: true" in page
    assert 'id="access-field" hidden' not in page
    assert 'id="access-field"' in page


def test_the_access_check_validates_the_code_with_no_model_call(client, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    with patch("server.services.nebius_client.requests.post") as post:
        missing = client.get("/api/access/check")
        assert missing.status_code == 401 and missing.json()["reason"] == "access_code"
        wrong = client.get("/api/access/check", headers={"X-Access-Code": "judge-9"})
        assert wrong.status_code == 403
        right = client.get("/api/access/check", headers={"X-Access-Code": "judge-1"})
        assert right.status_code == 200 and right.json() == {"ok": True, "mode": "code"}
        # The page may also pass it as a query parameter.
        assert client.get("/api/access/check?code=judge-1").status_code == 200
    assert post.call_count == 0


def test_the_config_endpoint_reports_the_mode_and_the_caps(client, monkeypatch):
    body = client.get("/api/config").json()
    assert body["access_mode"] == "code" and body["access_required"] is False
    assert body["demo_disabled"] is False and body["sample_available"] is False
    assert body["daily_limit"] is None and body["daily_calls_remaining"] is None

    monkeypatch.setenv("ACCESS_MODE", "open")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "150")
    monkeypatch.setenv("MAX_LIFTS_PER_CLIP", "2")
    body = client.get("/api/config").json()
    assert body["access_mode"] == "open"
    assert body["daily_limit"] == 150 and body["daily_calls_remaining"] == 150
    assert body["max_lifts_per_clip"] == 2
    assert body["judge"] == "NVIDIA Nemotron 3 Ultra" and body["vision"] == "MiniCPM-V 4.5"
    # Never the secret.
    assert "DEMO_ACCESS_CODE" not in json.dumps(body) and "key" not in json.dumps(body).lower()


def test_open_mode_turns_the_api_docs_off(client, monkeypatch):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    monkeypatch.setenv("ACCESS_MODE", "open")
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    # The app itself is untouched.
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200


# ---------- the startup refusal (A3) ----------


def test_open_mode_refuses_to_start_without_every_cap(monkeypatch):
    from fastapi.testclient import TestClient

    from server.main import app

    monkeypatch.setenv("ACCESS_MODE", "open")
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    assert len(missing_open_mode_caps()) == len(OPEN_MODE_CAPS)
    with pytest.raises(RuntimeError, match="ACCESS_MODE=open needs every spend cap set") as caught:
        with TestClient(app):
            pass
    for name in OPEN_MODE_CAPS:
        assert name in str(caught.value)

    # One missing, or zero, is still a refusal — and it is named.
    set_open_mode_caps(monkeypatch, DEMO_DAILY_LIMIT="0")
    assert missing_open_mode_caps() == ["DEMO_DAILY_LIMIT (model calls per day)"]
    with pytest.raises(RuntimeError, match="DEMO_DAILY_LIMIT"):
        with TestClient(app):
            pass
    # Frames over the vision model's per-call limit would cost two calls a lift.
    set_open_mode_caps(monkeypatch, MAX_FRAMES="12")
    assert missing_open_mode_caps() == ["MAX_FRAMES (at most 10: one vision call per lift)"]

    # With them all set it starts.
    set_open_mode_caps(monkeypatch)
    assert missing_open_mode_caps() == []
    with TestClient(app) as test_client:
        assert test_client.get("/healthz").status_code == 200

    # Without a model key nothing can be spent, so a local run starts anyway.
    monkeypatch.delenv("NEBIUS_API_KEY")
    for name in OPEN_MODE_CAPS:
        monkeypatch.delenv(name)
    with TestClient(app) as test_client:
        assert test_client.get("/healthz").status_code == 200

    # And code mode never refuses: the code is its gate.
    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("ACCESS_MODE", "code")
    with TestClient(app):
        pass


# ---------- the kill switch (P9) ----------


def test_the_kill_switch_leaves_only_the_sample(client, clip, sample_dir, monkeypatch):
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(sample_dir))
    monkeypatch.setenv("DEMO_DISABLED", "true")
    assert demo_disabled() is True

    page = client.get("/").text
    assert "demoDisabled: true" in page
    assert "Live scoring is paused" in page
    assert "See a sample result" in page
    # No camera button on the start screen while scoring is off.
    assert "Score my lift" not in page

    with patch("server.services.nebius_client.requests.post") as post:
        refused = post_clip(client, clip)
        assert refused.status_code == 503 and refused.json()["reason"] == "disabled"
        assert "paused" in refused.json()["detail"] and "sample" in refused.json()["detail"]
        assert sync_post(client, clip).status_code == 503
        check = client.get("/api/access/check")
        assert check.status_code == 503 and check.json()["reason"] == "disabled"
    assert post.call_count == 0
    assert client.get("/api/sample").status_code == 200
    assert client.get("/api/sample/replay/1.mp4").status_code == 200
    assert client.get("/api/config").json()["demo_disabled"] is True

    monkeypatch.setenv("DEMO_DISABLED", "false")
    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"


# ---------- refused from the headers, before the body ----------


def test_limits_are_answered_before_the_body_is_read(monkeypatch):
    """
    Like the access code, a paused demo, a busy server and a spent budget
    are answered from the headers alone: `receive` is never called, so a
    60 MB clip that cannot be scored is never received.
    """
    from server.main import DemoGateMiddleware
    from server.services.demo_guard import get_guard, reset_guard

    reset_guard()
    ran = []

    async def inner(scope, receive, send):
        ran.append(scope["path"])
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        raise AssertionError("the request body must not be read before the limits are checked")

    def run(path="/api/jobs", method="POST"):
        scope = {
            "type": "http",
            "method": method,
            "path": path,
            "headers": [(b"content-type", b"multipart/form-data; boundary=x")],
            "client": ("203.0.113.9", 50000),
        }
        sent = []

        async def send(message):
            sent.append(message)

        asyncio.run(DemoGateMiddleware(inner)(scope, receive, send))
        return sent[0]["status"], json.loads(sent[1]["body"] or b"{}")

    monkeypatch.setenv("DEMO_DISABLED", "true")
    assert run() == (503, {"detail": "The live demo is paused right now. The sample result shows everything it does.", "reason": "disabled"})
    monkeypatch.delenv("DEMO_DISABLED")

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    admission = get_guard().admit("1.2.3.4")
    status, body = run()
    assert status == 429 and body["reason"] == "busy"
    assert body["detail"] == "Another lift is being scored. You're next; this usually takes under a minute."
    admission.release()

    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    for _ in range(3):
        get_guard().charge("1.2.3.4")
    status, body = run()
    assert status == 429 and body["reason"] == "daily_limit"
    assert ran == []

    monkeypatch.setenv("DEMO_DAILY_LIMIT", "30")
    assert run()[0] == 202
    assert run("/api/sample", "GET")[0] == 202
    assert ran == ["/api/jobs", "/api/sample"]
    reset_guard()


# ---------- P2 / A4: forged X-Forwarded-For ----------


def test_forged_forwarded_entries_do_not_buy_a_fresh_allowance(client, clip, monkeypatch):
    # Heroku's router appends the visitor as the LAST entry; anything before it
    # is whatever the client typed. Ten requests with ten different fake first
    # entries are limited exactly as ten requests with none.
    monkeypatch.setenv("TRUSTED_PROXIES", "*")
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "3")
    visitor = "203.0.113.9"

    _, payload = upload(client, clip, ONE_SNATCH(), headers={"X-Forwarded-For": f"10.9.9.9, {visitor}"})
    assert payload["status"] == "done"

    with patch("server.services.nebius_client.requests.post") as post:
        for fake in range(10):
            forged = post_clip(client, clip, headers={"X-Forwarded-For": f"198.51.100.{fake}, {visitor}"})
            assert forged.status_code == 429 and forged.json()["reason"] == "ip_limit", fake
        for _ in range(10):
            plain = post_clip(client, clip, headers={"X-Forwarded-For": visitor})
            assert plain.status_code == 429 and plain.json()["reason"] == "ip_limit"
    assert post.call_count == 0
    # A different visitor, as the router would report one, is unaffected.
    _, payload = upload(client, clip, ONE_SNATCH(), headers={"X-Forwarded-For": "198.51.100.4, 203.0.113.10"})
    assert payload["status"] == "done"


# ---------- P1 / A5: the daily budget ----------


def test_a_spent_budget_shows_the_message_and_makes_no_more_calls(client, clip, sample_dir, monkeypatch):
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(sample_dir))
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"

    with patch("server.services.nebius_client.requests.post") as post:
        for _ in range(5):
            refused = post_clip(client, clip)
            assert refused.status_code == 429 and refused.json()["reason"] == "daily_limit"
            assert refused.json()["detail"] == "Today's free demo limit is used up. See the sample result, or come back tomorrow."
        # The sample is the way on, and it costs nothing.
        assert client.get("/api/sample").status_code == 200
    assert post.call_count == 0
    assert client.get("/api/config").json()["daily_calls_remaining"] == 0


def test_the_day_resets_at_local_midnight_not_utc(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    monkeypatch.setenv("DEMO_DAY_TIMEZONE", "Asia/Jerusalem")
    jerusalem = ZoneInfo("Asia/Jerusalem")
    # 23:30 in Israel is 21:30 UTC in winter: UTC midnight is 2.5 h away,
    # Israel's is 30 minutes away.
    clock.now = datetime(2026, 11, 3, 23, 30, tzinfo=jerusalem).timestamp()
    guard.admit("a")
    for _ in range(3):
        guard.charge("a")
    with pytest.raises(DemoRefused) as caught:
        guard.admit("b")
    assert caught.value.reason == "daily_limit"
    assert caught.value.retry_after == 30 * 60 + 1

    clock.now += 30 * 60 + 1  # 00:00:01 in Israel, still 3 November in UTC
    guard.admit("b")
    assert guard.used_today() == 0


def test_the_reset_follows_the_clock_change(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "1")
    monkeypatch.setenv("DEMO_DAY_TIMEZONE", "Asia/Jerusalem")
    jerusalem = ZoneInfo("Asia/Jerusalem")
    # The night Israel leaves summer time (last Sunday of October 2026, 02:00
    # → 01:00): the day is 25 hours long, and midnight is still midnight.
    clock.now = datetime(2026, 10, 24, 23, 0, tzinfo=jerusalem).timestamp()
    guard.admit("a")
    guard.charge("a")
    with pytest.raises(DemoRefused) as caught:
        guard.admit("a")
    assert caught.value.retry_after == 60 * 60 + 1
    clock.now = datetime(2026, 10, 25, 0, 0, 1, tzinfo=jerusalem).timestamp()
    guard.admit("a")
    guard.charge("a")
    with pytest.raises(DemoRefused) as caught:
        guard.admit("a")
    # 24 h plus the hour the clocks give back, less the one second past midnight.
    assert caught.value.retry_after == 25 * 60 * 60


def test_an_unknown_time_zone_means_utc(guard, clock, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "1")
    monkeypatch.setenv("DEMO_DAY_TIMEZONE", "Mars/Olympus_Mons")
    clock.now = 1_700_000_000.0
    guard.admit("a")
    guard.charge("a")
    with pytest.raises(DemoRefused) as caught:
        guard.admit("a")
    next_utc_midnight = (clock.now // 86_400 + 1) * 86_400
    assert caught.value.retry_after == int(next_utc_midnight - clock.now) + 1


# ---------- P8: the newcomer reserve ----------


def test_the_last_fifth_of_the_day_is_kept_for_first_time_visitors(guard, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "10")
    # A returning address may spend up to 8 of the 10; the last 2 are for
    # addresses that have not scored today.
    heavy = guard.admit("heavy")
    assert heavy.first_today is True
    for _ in range(3):
        heavy.meter()()
    assert heavy.calls == 3
    heavy.release()

    again = guard.admit("heavy")
    assert again.first_today is False
    for _ in range(3):
        again.meter()()
    again.release()
    assert guard.used_today() == 6

    with pytest.raises(DemoRefused) as caught:
        guard.admit("heavy")  # 6 + 3 > 8
    assert caught.value.reason == "daily_limit"
    assert "kept for first-time visitors" in caught.value.detail
    # A newcomer still gets in: 6 + 3 <= 10.
    newcomer = guard.admit("new")
    assert newcomer.first_today is True
    for _ in range(3):
        newcomer.meter()()
    newcomer.release()
    assert guard.used_today() == 9
    # Now the day is really spent for everyone, and the message says so.
    with pytest.raises(DemoRefused) as caught:
        guard.admit("another")
    assert caught.value.detail.startswith("Today's free demo limit is used up.")


def test_the_reserve_is_configurable_and_off_at_zero(guard, monkeypatch):
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "10")
    monkeypatch.setenv("DEMO_DAILY_RESERVE_PERCENT", "0")
    for _ in range(3):
        admission = guard.admit("heavy")
        for _ in range(3):
            admission.meter()()
        admission.release()
    assert guard.used_today() == 9

    monkeypatch.setenv("DEMO_DAILY_RESERVE_PERCENT", "50")
    other = DemoGuard(clock=lambda: 1_700_000_000.0)
    first = other.admit("heavy")
    for _ in range(3):
        first.meter()()
    first.release()
    second = other.admit("heavy")  # 3 + 3 <= 5
    for _ in range(2):
        second.meter()()
    second.release()
    with pytest.raises(DemoRefused):
        other.admit("heavy")  # 5 + 3 > 5
    other.admit("new")  # 5 + 3 <= 10


def test_a_refusal_does_not_make_an_address_a_returning_one(guard, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "10")
    busy = guard.admit("a")
    with pytest.raises(DemoRefused):
        guard.admit("b")  # busy: records nothing
    busy.release()
    assert guard.admit("b").first_today is True


# ---------- P7: the cost estimate before the first call ----------


def test_the_run_is_held_to_its_estimate_before_any_model_call(client, tmp_path, monkeypatch):
    """
    A session clip with two lift windows is six calls. With five left for
    the day it is turned away whole — before triage, not after the first
    lift — and the message says what to do.
    """
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "5")
    two = two_lift_clip(tmp_path)
    with patch("server.services.nebius_client.requests.post") as post:
        response = post_clip(client, two, lift="", mode="auto")
        assert response.status_code == 202
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "error"
    assert payload["reason"] == "daily_limit"
    assert payload["error"] == (
        "This clip holds 2 lifts, more than today's free demo has room for. "
        "Trim it to one lift, see the sample result, or come back tomorrow."
    )
    assert post.call_count == 0
    from server.services.demo_guard import get_guard

    assert get_guard().used_today() == 0 and get_guard().running == 0

    # With room for both it runs: one triage per window, then two calls a lift.
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "6")
    _, payload = upload(
        client,
        two,
        [
            triage_reply(SNATCH_PATH),
            triage_reply(SNATCH_PATH),
            observation_reply(),
            judge_reply("snatch", 7, "solid", 0.8),
            observation_reply(),
            judge_reply("snatch", 6, "mixed", 0.7),
        ],
        lift="",
        mode="auto",
    )
    assert payload["status"] == "done"
    assert [item["score"] for item in payload["lifts"]] == [7, 6]


def test_the_estimate_is_checked_against_the_hourly_limit_too(client, tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "4")
    two = two_lift_clip(tmp_path)
    with patch("server.services.nebius_client.requests.post") as post:
        response = post_clip(client, two, lift="", mode="auto")
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "error" and payload["reason"] == "ip_limit"
    assert payload["error"].startswith("This clip holds 2 lifts, more than this hour's limit has room for.")
    assert post.call_count == 0


def test_the_pipeline_asks_for_its_estimate_once_per_run(client, clip, tmp_path):
    """The reserve hook is told windows × one lift's calls, before triage."""
    from server.services.nebius_client import NebiusChatClient
    from server.services.pipeline import ScoreRequest, run_scoring

    asked = []
    two = two_lift_clip(tmp_path)
    with patch("server.services.nebius_client.requests.post", side_effect=ONE_SNATCH()) as post:
        asyncio.run(
            run_scoring(
                ScoreRequest(video_path=clip, lift=None, mode="single", render_replay=False),
                client=NebiusChatClient(api_key="k"),
                prompts_dir=str(Path(__file__).resolve().parents[1] / "prompts"),
                reserve=asked.append,
            )
        )
    assert asked == [CALLS_PER_LIFT]
    assert post.call_count == 3

    asked.clear()
    with patch("server.services.nebius_client.requests.post", side_effect=[triage_reply(SNATCH_PATH)] * 2 + ONE_SNATCH() * 2):
        asyncio.run(
            run_scoring(
                ScoreRequest(video_path=two, lift=None, mode="auto", render_replay=False),
                client=NebiusChatClient(api_key="k"),
                prompts_dir=str(Path(__file__).resolve().parents[1] / "prompts"),
                reserve=asked.append,
            )
        )
    assert asked == [2 * CALLS_PER_LIFT]


# ---------- P4: lifts per clip ----------


def test_the_lifts_per_clip_cap_scores_the_first_and_lists_the_rest(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_LIFTS_PER_CLIP", "1")
    two = two_lift_clip(tmp_path)
    with patch("server.services.nebius_client.requests.post", side_effect=ONE_SNATCH()) as post:
        response = post_clip(client, two, lift="", mode="auto")
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "done"
    assert len(payload["lifts"]) == 1 and payload["lifts"][0]["score"] == 7
    assert payload["candidate_windows"] == 1
    assert any("up to 1 lifts per clip" in line for line in payload["skipped"])
    # Three calls, not six: the capped window never reached triage.
    assert post.call_count == 3
    assert "Scores up to 1 attempts it finds" in client.get("/").text


# ---------- P5 / A6: the deadline ----------


def test_the_deadline_frees_the_slot_and_reads_as_a_plain_sentence(client, clip, monkeypatch):
    from server.services import errors
    from server.services.demo_guard import get_guard

    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")

    async def forever(*_, **__):
        await asyncio.sleep(30)

    with patch("server.main.run_scoring", side_effect=forever):
        response = post_clip(client, clip)
        assert response.status_code == 202
        # The slot is held while it runs...
        assert post_clip(client, clip).json()["reason"] == "busy"
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "error" and payload["error"] == errors.TOO_LONG
    assert "reason" not in payload
    # ...and free once the deadline stops it.
    assert get_guard().running == 0
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "30")
    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"


# ---------- the log line ----------


def test_an_admitted_run_logs_one_line_without_the_address(client, clip, caplog):
    with caplog.at_level(logging.INFO, logger="server.main"):
        _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"
    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("job status=")]
    assert len(lines) == 1
    line = lines[0]
    assert "status=done" in line and "calls=3" in line and "first_today=yes" in line
    assert "remaining_today=unlimited" in line and "running=0" in line
    # A keyed hash, never the peer address, and never the clip's name.
    assert "testclient" not in line and "clip.mp4" not in line
    assert "ip=" in line and len(line.split("ip=")[1].split()[0]) == 12


# ---------- the first screen (A1) ----------


def test_the_start_screen_leads_with_the_sample(client, sample_dir, monkeypatch):
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(sample_dir))
    monkeypatch.setenv("ACCESS_MODE", "open")
    monkeypatch.setenv("MAX_CLIP_SECONDS", "15")
    monkeypatch.setenv("MAX_UPLOAD_MB", "60")
    page = client.get("/").text
    start = page.index('data-screen="start" data-active="true"')
    lift = page.index('data-screen="lift"')
    assert start < lift
    hero = page[start:lift]
    assert "Film a snatch or clean from the side. Get a bar-path score and an annotated replay in" in hero
    assert "See a sample result" in hero and "Score my lift" in hero
    assert hero.index("See a sample result") < hero.index("Score my lift")
    assert "Clips need to be under 15 seconds and 60 MB." in hero
    assert "Open demo, no code needed" in page
    # The sample is the way on next to every refusal too.
    assert page.count('data-action="sample"') >= 3
