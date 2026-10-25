"""
The public-demo layers over HTTP: access code, daily and per-IP budgets,
one-or-two-at-a-time, the run timeout, the clip-length and frame caps, and
the cached sample clip that costs nothing. Model calls are mocked throughout.
"""
import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from server.services import errors
from server.services.demo_guard import get_guard
from server.services.jobs import get_store
from tests.conftest import (  # noqa: F401  (client and clip are fixtures)
    CLEAN_AND_JERK_PATH,
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

ONE_SNATCH = lambda: [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)]  # noqa: E731


def post_clip(client, clip, *, headers=None, **fields):
    data = {"lift": "snatch"}
    data.update(fields)
    with open(clip, "rb") as handle:
        return client.post(
            "/api/jobs",
            data=data,
            files={"video": ("clip.mp4", handle, "video/mp4")},
            headers=headers or {},
        )


# ---------- access code ----------


def test_without_a_code_configured_the_page_does_not_ask_for_one(client):
    assert "accessRequired: false" in client.get("/").text


def test_access_code_gates_scoring(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1, judge-2")
    assert "accessRequired: true" in client.get("/").text

    missing = post_clip(client, clip)
    assert missing.status_code == 401
    assert missing.json()["reason"] == "access_code"
    assert "access code" in missing.json()["detail"]

    wrong = post_clip(client, clip, headers={"X-Access-Code": "judge-9"})
    assert wrong.status_code == 403
    assert wrong.json()["reason"] == "access_code"

    _, payload = upload(client, clip, ONE_SNATCH(), headers={"X-Access-Code": "judge-2"})
    assert payload["status"] == "done"
    # The code travels as a header only. A form field is inside the body, which
    # is exactly what must not be read before the code is checked.
    assert post_clip(client, clip, access_code="judge-1").status_code == 401


def test_the_sync_endpoint_is_gated_too(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    with open(clip, "rb") as handle:
        response = client.post("/api/score", data={"lift": "snatch"}, files={"video": ("clip.mp4", handle, "video/mp4")})
    assert response.status_code == 401


def test_the_access_code_is_checked_before_the_body_is_read(monkeypatch):
    """
    FastAPI reads the whole multipart body before any route code runs, so the
    check lives in ASGI middleware. A refusal must be answered from the
    headers alone: `receive` is never called, so a 200 MB upload with no code
    costs the server nothing.
    """
    from server.main import DemoGateMiddleware

    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    ran = []

    async def inner(scope, receive, send):
        ran.append(scope["path"])
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        raise AssertionError("the request body must not be read before the code is checked")

    def request(path, method="POST", code=None):
        headers = [(b"content-type", b"multipart/form-data; boundary=x")]
        if code is not None:
            headers.append((b"x-access-code", code.encode()))
        return {"type": "http", "method": method, "path": path, "headers": headers}

    def run(scope):
        sent = []

        async def send(message):
            sent.append(message)

        asyncio.run(DemoGateMiddleware(inner)(scope, receive, send))
        return sent[0]["status"], json.loads(sent[1]["body"] or b"{}")

    assert run(request("/api/jobs")) == (401, {"detail": errors.user_message(_missing_code()), "reason": "access_code"})
    assert run(request("/api/score", code="judge-9"))[0] == 403
    assert ran == []
    # The right code, and anything that costs nothing, goes through untouched.
    assert run(request("/api/jobs", code="judge-1"))[0] == 202
    assert run(request("/api/sample", method="GET"))[0] == 202
    assert run(request("/api/jobs/abc", method="GET"))[0] == 202
    assert ran == ["/api/jobs", "/api/sample", "/api/jobs/abc"]


def _missing_code():
    from server.services.demo_guard import DemoRefused, check_access

    try:
        check_access(None)
    except DemoRefused as exc:
        return exc
    raise AssertionError("expected a refusal")


def test_a_refused_upload_leaves_no_job_behind(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    monkeypatch.setenv("MAX_CLIP_SECONDS", "3")
    post_clip(client, clip)  # 401: refused before a job exists
    post_clip(client, clip, headers={"X-Access-Code": "judge-1"})  # 400: the window is too long
    root = get_store().root
    assert not root.exists() or not any(root.iterdir())


# ---------- budgets, in model calls ----------


def test_daily_budget_counts_model_calls_and_turns_the_demo_off_for_the_day(client, clip, monkeypatch):
    # One single-lift clip is three calls: triage, observation, judge.
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"
    assert get_guard().used_today() == 3

    response = post_clip(client, clip)
    assert response.status_code == 429
    body = response.json()
    assert body["reason"] == "daily_limit"
    assert body["detail"] == "Today's free demo limit is used up. See the sample result, or come back tomorrow."
    assert int(response.headers["retry-after"]) > 0

    with open(clip, "rb") as handle:
        sync = client.post("/api/score", data={"lift": "snatch"}, files={"video": ("clip.mp4", handle, "video/mp4")})
    assert sync.status_code == 429 and sync.json()["reason"] == "daily_limit"


def test_a_budget_that_runs_out_mid_run_keeps_what_was_scored(client, clip, monkeypatch):
    # Room for one lift admits the clip; a clean & jerk needs five calls, so
    # the jerk's observation call is refused and the clean is kept.
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "4")
    _, payload = upload(
        client,
        clip,
        [
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 6, "mixed", 0.7),
            observation_reply(),  # never reached
            judge_reply("jerk", 5, "mixed", 0.7),
        ],
        lift="clean",
    )
    assert payload["status"] == "done"
    assert [item["lift"] for item in payload["lifts"]] == ["clean", "jerk"]
    assert payload["lifts"][0]["score"] == 6
    jerk = payload["lifts"][1]
    assert "score" not in jerk
    assert "ran out part-way" in jerk["error"]
    assert get_guard().used_today() == 4
    assert get_guard().running == 0


def test_a_retry_is_a_billed_call_and_is_counted(client, clip, monkeypatch):
    from server.services import nebius_client

    monkeypatch.setattr(nebius_client, "RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "3")
    from unittest.mock import Mock

    busy = Mock()
    busy.status_code = 503
    busy.text = "upstream unavailable"
    # Triage, observation, then a 503 from the judge: the retry would be a
    # fourth call, and the budget stops it.
    _, payload = upload(client, clip, [triage_reply(SNATCH_PATH), observation_reply(), busy, judge_reply()])
    assert payload["status"] == "error"
    assert "ran out part-way" in payload["error"]
    assert get_guard().used_today() == 3


def test_per_ip_limit_uses_the_forwarded_address_only_behind_a_trusted_proxy(client, clip, monkeypatch):
    monkeypatch.setenv("DEMO_IP_HOURLY_LIMIT", "3")
    judge = {"X-Forwarded-For": "203.0.113.9"}

    # Nothing trusted: the header is ignored and every caller on this socket
    # shares one allowance, so a forged header buys nothing.
    _, payload = upload(client, clip, ONE_SNATCH(), headers=judge)
    assert payload["status"] == "done"
    forged = post_clip(client, clip, headers={"X-Forwarded-For": "198.51.100.4"})
    assert forged.status_code == 429 and forged.json()["reason"] == "ip_limit"
    assert forged.json()["detail"].startswith("You've hit the hourly limit. Try again in 60 minutes")
    assert int(forged.headers["retry-after"]) > 0

    # Behind a trusted proxy the forwarded address is the visitor, and a
    # different judge is unaffected by the first one's hour.
    monkeypatch.setenv("TRUSTED_PROXIES", "*")
    _, payload = upload(client, clip, ONE_SNATCH(), headers={"X-Forwarded-For": "198.51.100.4"})
    assert payload["status"] == "done"
    again = post_clip(client, clip, headers={"X-Forwarded-For": "198.51.100.4"})
    assert again.status_code == 429 and again.json()["reason"] == "ip_limit"


# ---------- the run timeout ----------


def test_a_run_past_the_timeout_is_stopped_and_frees_its_slot(client, clip, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "0.2")
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")

    async def forever(*_, **__):
        await asyncio.sleep(30)

    with patch("server.main.run_scoring", side_effect=forever):
        response = post_clip(client, clip)
        assert response.status_code == 202
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "error"
    assert payload["error"] == errors.TOO_LONG
    assert get_guard().running == 0

    with patch("server.main.run_scoring", side_effect=forever):
        with open(clip, "rb") as handle:
            sync = client.post("/api/score", data={"lift": "snatch"}, files={"video": ("clip.mp4", handle, "video/mp4")})
    assert sync.status_code == 504 and sync.json()["detail"] == errors.TOO_LONG
    assert get_guard().running == 0


def test_only_so_many_jobs_run_at_once(client, clip, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    gate = threading.Event()
    replies = iter(ONE_SNATCH() + ONE_SNATCH())

    def slow_post(*_, **__):
        # The first job's model calls wait on the gate, so it stays running.
        gate.wait(timeout=30)
        return next(replies)

    with patch("server.services.nebius_client.requests.post", side_effect=slow_post):
        first = post_clip(client, clip)
        assert first.status_code == 202

        second = post_clip(client, clip)
        assert second.status_code == 429
        assert second.json()["reason"] == "busy"
        assert second.headers["retry-after"] == "60"

        gate.set()
        assert wait_for(client, first.json()["id"])["status"] == "done"
        assert get_guard().running == 0

        third = post_clip(client, clip)
        assert third.status_code == 202
        assert wait_for(client, third.json()["id"])["status"] == "done"


def test_a_failed_run_still_frees_its_slot(client, clip, monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_JOBS", "1")
    boom = RuntimeError("Nebius API error 503: upstream unavailable")
    _, payload = upload(client, clip, [triage_reply(SNATCH_PATH), boom, boom])
    assert payload["status"] == "error"
    assert get_guard().running == 0
    _, payload = upload(client, clip, ONE_SNATCH())
    assert payload["status"] == "done"


# ---------- clip length and frames ----------


def test_clip_cap_refuses_a_long_window_but_not_a_trimmed_one(client, clip, monkeypatch):
    monkeypatch.setenv("MAX_CLIP_SECONDS", "3")
    assert "maxClipSeconds: 3" in client.get("/").text

    whole = post_clip(client, clip)  # the synthetic clip is 5 s
    assert whole.status_code == 400
    assert whole.json()["reason"] == "too_long"
    assert "5 s" in whole.json()["detail"] and "3 s" in whole.json()["detail"]

    _, payload = upload(client, clip, ONE_SNATCH(), trim_start="0.8", trim_end="3.5")
    assert payload["status"] == "done"


def test_clip_cap_applies_to_the_sync_endpoint(client, clip, monkeypatch):
    monkeypatch.setenv("MAX_CLIP_SECONDS", "3")
    with open(clip, "rb") as handle:
        response = client.post("/api/score", data={"lift": "snatch"}, files={"video": ("clip.mp4", handle, "video/mp4")})
    assert response.status_code == 400 and response.json()["reason"] == "too_long"


def test_frame_cap_keeps_every_lift_to_one_vision_call(client, clip, monkeypatch):
    monkeypatch.setenv("MAX_FRAMES", "10")
    assert "maxFrames: 10" in client.get("/").text
    # 12 frames would be chunked into two observation calls; capped to 10 it is one.
    with open(clip, "rb") as handle:
        with patch("server.services.nebius_client.requests.post", side_effect=ONE_SNATCH()) as post:
            response = client.post(
                "/api/score",
                data={"lift": "snatch", "num_frames": "12"},
                files={"video": ("clip.mp4", handle, "video/mp4")},
            )
    assert response.status_code == 200
    assert post.call_count == 3


# ---------- which model did what ----------


def test_the_footer_names_the_judge_that_is_configured(client, monkeypatch):
    # The footer reads as the pipeline: tracker measured → vision observed → judge judged.
    assert "CV tracker measured → MiniCPM-V 4.5 observed → NVIDIA Nemotron 3 Ultra judged" in client.get("/").text
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    assert "NVIDIA Nemotron 3 Super judged" in client.get("/").text
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "some-lab/other-judge")
    assert "some-lab/other-judge judged" in client.get("/").text


def test_the_verdict_says_which_model_did_what(client, clip, monkeypatch):
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")
    _, payload = upload(client, clip, ONE_SNATCH())
    lift = payload["lifts"][0]
    assert lift["judged_by"] == "nvidia/nemotron-3-super-120b-a12b"
    assert lift["models"] == {
        "vision": "openbmb/MiniCPM-V-4_5",
        "vision_name": "MiniCPM-V 4.5",
        "judge": "nvidia/nemotron-3-super-120b-a12b",
        "judge_name": "NVIDIA Nemotron 3 Super",
        "judge_fallback": False,
    }


def test_the_verdict_names_the_fallback_judge_when_it_answered(client, clip, monkeypatch):
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")
    prose = observation_reply("Let me think about this")  # never valid JSON
    _, payload = upload(client, clip, [triage_reply(SNATCH_PATH), observation_reply(), prose, prose, judge_reply()])
    lift = payload["lifts"][0]
    assert lift["judged_by"] == "nvidia/Nemotron-3-Ultra-550b-a55b"
    assert lift["models"]["judge_name"] == "NVIDIA Nemotron 3 Ultra"
    assert lift["models"]["judge_fallback"] is True


def test_with_the_fallback_disabled_a_failed_judge_is_an_error_not_a_second_model(client, clip, monkeypatch):
    # The public demo runs Super with no fallback to Ultra: a judge that fails
    # fails, rather than quietly escalating to the expensive model.
    monkeypatch.setenv("NEBIUS_JUDGE_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    monkeypatch.setenv("NEBIUS_JUDGE_FALLBACK_MODEL", "")
    prose = observation_reply("Let me think about this")
    with patch("server.services.nebius_client.requests.post", side_effect=[triage_reply(SNATCH_PATH), observation_reply(), prose, prose, judge_reply()]) as post:
        response = post_clip(client, clip)
        payload = wait_for(client, response.json()["id"])
    assert payload["status"] == "error"
    assert payload["error"] == errors.NO_VERDICT
    assert post.call_count == 4
    assert "nvidia/Nemotron-3-Ultra-550b-a55b" not in [call.kwargs["json"]["model"] for call in post.call_args_list]


def test_the_drift_line_is_written_from_the_measurement_when_the_bar_was_tracked(client, clip):
    _, payload = upload(client, clip, ONE_SNATCH())
    lift = payload["lifts"][0]
    assert lift["drift"]["text"]
    if lift["bar_path"]["tracked"]:
        assert lift["drift"]["source"] == "measurement"
        assert f'{lift["bar_path"]["max_drift_plate_diameters"]:.2f} plate widths' in lift["drift"]["text"]
    else:
        assert lift["drift"]["source"] == "judge"
        assert lift["drift"]["text"] == lift["max_horizontal_drift"]


# ---------- the cached sample clip ----------


@pytest.fixture
def sample_dir(tmp_path):
    """A sample built the way the tool builds it, with the model mocked."""
    from server.schemas.judgment import LiftType
    from server.services.nebius_client import NebiusChatClient
    from server.services.sample import make_sample

    source = write_lift_video(str(tmp_path / "source.mp4"))
    out = tmp_path / "demo_sample"
    with patch("server.services.nebius_client.requests.post", side_effect=ONE_SNATCH()):
        asyncio.run(
            make_sample(
                video_path=source,
                out_dir=out,
                client=NebiusChatClient(api_key="test-key"),
                prompts_dir=str(Path(__file__).resolve().parents[1] / "prompts"),
                lift=LiftType.snatch,
            )
        )
    return out


def test_without_a_sample_the_page_offers_none(client):
    assert client.get("/api/sample").status_code == 404
    body = client.get("/").text
    assert "sampleAvailable: false" in body
    assert 'data-action="sample"' not in body


def test_the_sample_is_served_without_a_code_or_a_budget(client, sample_dir, monkeypatch):
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(sample_dir))
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    monkeypatch.setenv("DEMO_DAILY_LIMIT", "0")

    body = client.get("/").text
    assert "sampleAvailable: true" in body
    assert 'data-action="sample"' in body

    response = client.get("/api/sample")
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample"] is True and payload["status"] == "done"
    assert payload["lifts"][0]["score"] == 7 and payload["lifts"][0]["band"] == "solid"
    assert payload["lifts"][0]["annotated_video_url"] == "/api/sample/replay/1.mp4"

    replay = client.get("/api/sample/replay/1.mp4")
    assert replay.status_code == 200
    assert replay.headers["content-type"] == "video/mp4"
    assert replay.headers["content-disposition"].startswith("inline")
    assert len(replay.content) > 1024
    assert client.get("/api/sample/replay/2.mp4").status_code == 404
    assert get_guard().used_today() == 0


def test_a_sample_whose_replay_went_missing_still_shows_the_verdict(client, sample_dir, monkeypatch):
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(sample_dir))
    for video in sample_dir.glob("*.mp4"):
        video.unlink()
    payload = client.get("/api/sample").json()
    assert payload["lifts"][0]["score"] == 7
    assert payload["lifts"][0]["annotated_video_url"] is None
    assert client.get("/api/sample/replay/1.mp4").status_code == 404


def test_the_sample_tool_parses_its_arguments(tmp_path):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "tools" / "make_demo_sample.py"
    spec = importlib.util.spec_from_file_location("make_demo_sample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["clip.mp4", "--lift", "snatch", "--out", str(tmp_path / "s"), "--trim-end", "4"])
    assert args.lift == "snatch" and args.trim_end == 4.0 and args.out.endswith("s")
    # No lift and no --auto is refused before any model call.
    assert asyncio.run(module.run(module.parse_args([str(tmp_path / "missing.mp4"), "--lift", "snatch"]))) == 2
