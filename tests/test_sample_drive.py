"""
With a committed sample bundle (JSON + evidence stills, no video) and a
SAMPLE_CLIP_URL pointing at Google Drive, the dyno downloads the clip once
and re-renders the annotated replay from the serialised bar-path. The whole
sample path is zero-cost: no model calls, no counter bumps, no access code.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from server.main import app
from server.services import sample as sample_module
from server.services.demo_guard import get_guard, reset_guard
from server.utils import drive_download
from tests.conftest import write_lift_video


def _write_sample_bundle(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "version": 2,
        "source": "clean.mp4",
        "replays": {},
        "payload": {
            "lifts": [
                {
                    "index": 1,
                    "lift": "clean",
                    "start_sec": 0.0,
                    "end_sec": 3.5,
                    "score": 7,
                    "band": "solid",
                    "cues": ["drive hard off the floor; keep the bar close"],
                    "max_horizontal_drift": "forward loop in the second pull",
                    "confidence": 0.75,
                    "annotated_video_url": None,
                }
            ],
            "sample_bar_path": {
                "points": [[i / 10.0, 0.42 + 0.02 * (i % 3 - 1), 0.90 - 0.07 * i] for i in range(12)],
                "plate_radius": 0.09,
                "coverage": 1.0,
                "sampled": 12,
                "frame_width": 96,
                "frame_height": 160,
                "reason": "",
            },
            "status": "done",
            "stage": "done",
            "message": "sample",
            "progress": 1.0,
        },
    }
    (directory / "sample.json").write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def client_env(tmp_path, monkeypatch):
    """
    One FastAPI client with:
      - a committed sample bundle in tmp
      - SAMPLE_CLIP_URL pointing at a fake Drive URL
      - the Drive downloader fast-tracked (no network, no cv2.VideoCapture
        check on bytes that are not a real video — the downloader test file
        covers verification separately)
    """
    bundle = tmp_path / "demo_sample"
    _write_sample_bundle(bundle)
    cache = tmp_path / "cache"

    monkeypatch.setenv("NEBIUS_API_KEY", "sk-test")
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(bundle))
    monkeypatch.setenv("SAMPLE_CLIP_URL", "https://drive.google.com/uc?id=abc")
    monkeypatch.setenv("SAMPLE_CACHE_DIR", str(cache))
    monkeypatch.delenv("SAMPLE_RESULT_URL", raising=False)

    reset_guard(None)
    sample_module.reset_remote_cache_for_tests()
    drive_download.reset_locks_for_tests()

    # The Drive downloader writes whatever the fake server hands back. Point
    # it at a real synthetic clip so OpenCV can decode and the replay renderer
    # has frames to encode.
    fake_clip = write_lift_video(str(tmp_path / "fake_drive.mp4"), width=96, height=160, duration=2.0)
    clip_bytes = Path(fake_clip).read_bytes()

    download_calls = {"count": 0}

    def _fake_download(url, destination, **kwargs):  # noqa: ARG001
        download_calls["count"] += 1
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(clip_bytes)
        return Path(destination)

    with patch("server.utils.drive_download.download_drive_file", side_effect=_fake_download):
        with TestClient(app) as test_client:
            yield test_client, download_calls

    sample_module.reset_remote_cache_for_tests()
    drive_download.reset_locks_for_tests()


def test_sample_endpoint_renders_the_replay_from_drive_clip(client_env):
    client, download_calls = client_env
    response = client.get("/api/sample")
    assert response.status_code == 200
    body = response.json()
    assert body["sample"] is True
    first = body["lifts"][0]
    assert first["lift"] == "clean"
    assert first["score"] == 7
    assert first["annotated_video_url"] == "/api/sample/replay/1.mp4"

    replay = client.get("/api/sample/replay/1.mp4")
    assert replay.status_code == 200
    assert replay.headers["content-type"] == "video/mp4"
    assert len(replay.content) > 1024

    assert download_calls["count"] == 1


def test_sample_endpoint_needs_no_access_code(client_env, monkeypatch):
    """A code is required for /api/jobs but never for the sample."""
    monkeypatch.setenv("DEMO_ACCESS_CODE", "judge-1")
    client, _ = client_env
    assert client.get("/api/sample").status_code == 200


def test_sample_endpoint_counts_zero_against_the_budget(client_env):
    """
    The demo guard is not touched by /api/sample. Fetch it twice; the day
    counter must still be zero and the running counter must be zero after.
    """
    client, _ = client_env
    assert client.get("/api/sample").status_code == 200
    assert client.get("/api/sample").status_code == 200
    guard = get_guard()
    assert guard.used_today() == 0
    assert guard.running == 0


def test_sample_endpoint_never_builds_a_nebius_client(client_env):
    """A model call during /api/sample would be a regression; prove there is none."""
    client, _ = client_env

    def _should_not_be_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Nebius was called while serving /api/sample")

    with patch("server.services.nebius_client.requests.post", side_effect=_should_not_be_called):
        assert client.get("/api/sample").status_code == 200
        assert client.get("/api/sample/replay/1.mp4").status_code == 200


def test_sample_is_hidden_when_the_drive_download_fails(tmp_path, monkeypatch):
    bundle = tmp_path / "demo_sample"
    _write_sample_bundle(bundle)
    monkeypatch.setenv("NEBIUS_API_KEY", "sk-test")
    monkeypatch.setenv("DEMO_SAMPLE_DIR", str(bundle))
    monkeypatch.setenv("SAMPLE_CLIP_URL", "https://drive.google.com/uc?id=broken")
    monkeypatch.setenv("SAMPLE_CACHE_DIR", str(tmp_path / "cache"))
    reset_guard(None)
    sample_module.reset_remote_cache_for_tests()
    drive_download.reset_locks_for_tests()

    def _fail(url, destination, **kwargs):  # noqa: ARG001
        raise RuntimeError("drive 403")

    with patch("server.utils.drive_download.download_drive_file", side_effect=_fail):
        with TestClient(app) as client:
            response = client.get("/api/sample")
            assert response.status_code == 200
            # The verdict still shows even when the replay could not be rendered.
            assert response.json()["lifts"][0]["annotated_video_url"] is None
            # And the main app keeps working.
            assert client.get("/healthz").status_code == 200
