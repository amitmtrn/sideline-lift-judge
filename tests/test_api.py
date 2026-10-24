"""
The HTTP surface a phone talks to: PWA assets, upload, polling, replay.

Model calls are mocked throughout — nothing here reaches Token Factory.
"""
from unittest.mock import patch

from tests.conftest import (  # noqa: F401  (client and clip are fixtures)
    CLEAN_AND_JERK_PATH,
    SNATCH_PATH,
    client,
    clip,
    judge_reply,
    observation_reply,
    triage_reply,
    upload,
)


# ---------- the shell a judge opens on a phone ----------


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_home_page_is_mobile_first_and_installable(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert 'name="viewport"' in body and "viewport-fit=cover" in body
    assert '<link rel="manifest" href="/manifest.webmanifest"' in body
    assert 'rel="apple-touch-icon"' in body
    # FR-1: film straight from the page, not upload only.
    assert 'capture="environment"' in body


def test_manifest_is_served_from_the_root_scope(client):
    response = client.get("/manifest.webmanifest")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "/" and body["display"] == "standalone"
    assert any(icon["sizes"] == "512x512" for icon in body["icons"])
    assert any(icon.get("purpose") == "maskable" for icon in body["icons"])


def test_service_worker_is_served_from_the_root(client):
    # A worker under /static could not control the whole app.
    response = client.get("/service-worker.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert "/api/" in response.text  # it explicitly refuses to cache the API


def test_icons_exist(client):
    for name in ("icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png"):
        assert client.get(f"/static/icons/{name}").status_code == 200


# ---------- uploads ----------


def test_rejects_a_file_that_is_not_a_video(client, tmp_path):
    response = client.post(
        "/api/jobs",
        data={"lift": "snatch"},
        files={"video": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "not a video" in response.json()["detail"]


def test_rejects_an_oversized_clip(client, tmp_path):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * (6 * 1024 * 1024))  # MAX_UPLOAD_MB is 5 in this fixture
    with open(big, "rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"lift": "snatch"},
            files={"video": ("big.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 413
    assert "5 MB" in response.json()["detail"]


def test_rejects_an_impossible_trim_window(client, clip):
    with open(clip, "rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"lift": "snatch", "trim_start": "4", "trim_end": "2"},
            files={"video": ("clip.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 400
    assert "after trim start" in response.json()["detail"]


def test_rejects_an_unknown_lift(client, clip):
    with open(clip, "rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"lift": "deadlift"},
            files={"video": ("clip.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 400
    assert "Unknown lift" in response.json()["detail"]


def test_missing_api_key_is_a_plain_message_not_a_crash(client, clip, monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with open(clip, "rb") as handle:
        response = client.post(
            "/api/jobs",
            data={"lift": "snatch"},
            files={"video": ("clip.mp4", handle, "video/mp4")},
        )
    assert response.status_code == 503
    assert "NEBIUS_API_KEY" in response.json()["detail"]
    assert "Traceback" not in response.text


# ---------- scoring end to end ----------


def test_single_lift_round_trip(client, clip):
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    )
    assert payload["status"] == "done"
    assert payload["progress"] == 1.0
    assert len(payload["lifts"]) == 1
    lift = payload["lifts"][0]
    assert lift["lift"] == "snatch"
    assert lift["score"] == 7
    assert lift["band"] == "solid"
    assert lift["cues"] and len(lift["cues"]) <= 2
    assert lift["max_horizontal_drift"]
    assert 0.0 <= lift["confidence"] <= 1.0
    # The card says which model did what, and its drift line has a source.
    assert lift["models"]["vision_name"] == "MiniCPM-V 4.5"
    assert lift["models"]["judge_name"] == "NVIDIA Nemotron 3 Ultra"
    assert lift["drift"]["source"] in ("measurement", "judge") and lift["drift"]["text"]


def test_replay_is_served_and_playable_inline(client, clip):
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    )
    url = payload["lifts"][0]["annotated_video_url"]
    assert url, "no replay was rendered"
    response = client.get(url)
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"].startswith("inline")
    assert len(response.content) > 1024


def test_clean_and_jerk_comes_back_as_two_scored_lifts(client, clip):
    _, payload = upload(
        client,
        clip,
        [
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 6, "mixed", 0.7),
            observation_reply(),
            judge_reply("jerk", 5, "mixed", 0.7),
        ],
        lift="clean",
    )
    assert payload["split_detected"] is True
    assert [item["lift"] for item in payload["lifts"]] == ["clean", "jerk"]
    # Each half gets its own replay.
    assert len({item["annotated_video_url"] for item in payload["lifts"]}) == 2


def test_mixed_clip_that_cannot_be_split_lowers_confidence(client, clip, monkeypatch):
    # AC-4: the 0.85-on-a-mixed-clip case must not happen again.
    monkeypatch.setattr("server.services.pipeline.split_segment_at_valley", lambda *a, **k: None)
    _, payload = upload(
        client,
        clip,
        [
            triage_reply(CLEAN_AND_JERK_PATH),
            observation_reply(),
            judge_reply("clean", 3, "poor", 0.85),
            observation_reply(),
            judge_reply("jerk", 3, "poor", 0.85),
        ],
        lift="clean",
    )
    assert payload["split_detected"] is False
    for item in payload["lifts"]:
        assert item["confidence"] < 0.5, "a mixed clip must not come back confident"
        assert any("more than one lift" in warning for warning in item["warnings"])


def test_upstream_failure_reports_an_error_status(client, clip):
    boom = RuntimeError("Nebius API error 503: upstream unavailable")
    _, payload = upload(client, clip, [triage_reply(SNATCH_PATH), boom, boom])
    assert payload["status"] == "error"
    # One plain sentence for the phone; the provider's text stays in the log.
    assert "busy" in payload["error"]
    assert "503" not in payload["error"] and "upstream unavailable" not in payload["error"]


def test_the_result_carries_what_the_tracker_measured(client, clip):
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    )
    lift = payload["lifts"][0]
    assert "bar_path" in lift
    assert lift["bar_path"]["tracked"] == lift["bar_path_tracked"]
    if lift["bar_path"]["tracked"]:
        assert lift["bar_path"]["max_drift_plate_diameters"] >= 0


def test_trim_window_is_passed_through(client, clip):
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
        trim_start="1.0",
        trim_end="3.0",
    )
    lift = payload["lifts"][0]
    assert lift["start_sec"] >= 1.0
    assert lift["end_sec"] <= 3.0


def test_the_uploaded_clip_is_deleted_once_scoring_ends(client, clip):
    from server.services.jobs import get_store

    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    )
    job = get_store().get(payload["id"])
    assert job is not None and job.source_path is None
    # Only rendered replays remain in the job directory.
    assert not any(path.name.startswith("tmp") for path in job.workdir.iterdir())


def test_expired_job_explains_itself(client):
    response = client.get("/api/jobs/does-not-exist")
    assert response.status_code == 404
    assert "expired" in response.json()["detail"]


def test_replay_for_an_unknown_lift_is_404(client, clip):
    _, payload = upload(
        client,
        clip,
        [triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 7, "solid", 0.8)],
    )
    assert client.get(f"/api/jobs/{payload['id']}/replay/99.mp4").status_code == 404


# ---------- the synchronous endpoint ----------


def test_sync_endpoint_returns_one_verdict(client, clip):
    with open(clip, "rb") as handle:
        with patch(
            "server.services.nebius_client.requests.post",
            side_effect=[triage_reply(SNATCH_PATH), observation_reply(), judge_reply("snatch", 9, "elite", 0.9)],
        ):
            response = client.post(
                "/api/score",
                data={"lift": "snatch"},
                files={"video": ("clip.mp4", handle, "video/mp4")},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["lift"] == "snatch" and body["score"] == 9 and body["band"] == "elite"


def test_sync_endpoint_still_accepts_the_original_file_field(client, clip):
    # 12 frames is over the 10-image limit, so Stage 1 is chunked into two calls
    # (triage, two observation chunks, judge).
    with open(clip, "rb") as handle:
        with patch(
            "server.services.nebius_client.requests.post",
            side_effect=[triage_reply(SNATCH_PATH), observation_reply(), observation_reply(), judge_reply()],
        ) as post:
            response = client.post(
                "/api/score",
                data={"lift": "snatch", "num_frames": "12"},
                files={"file": ("clip.mp4", handle, "video/mp4")},
            )
    assert response.status_code == 200
    assert post.call_count == 4


def test_image_assets_are_served(client):
    for path in ("/static/img/mark.svg", "/static/img/filming-guide.svg"):
        response = client.get(path)
        assert response.status_code == 200
        assert "svg" in response.headers["content-type"]
        assert "<title>" in response.text  # accessible name


def test_the_page_references_the_image_assets(client):
    body = client.get("/").text
    assert "/static/img/mark.svg" in body
    assert "/static/img/filming-guide.svg" in body
