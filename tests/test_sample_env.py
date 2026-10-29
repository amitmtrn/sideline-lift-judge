"""
The sample can also be served from env-configured URLs: a hosted sample.json
(SAMPLE_RESULT_URL) and a hosted MP4 (SAMPLE_CLIP_URL) that stand in for the
local directory when the deploy ships no video. The sample path still makes
no model calls and still does not charge the demo's budgets, either source.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from server.main import app
from server.services.demo_guard import get_guard, reset_guard
from server.services import sample as sample_module


SAMPLE_BODY = {
    "payload": {
        "lifts": [
            {
                "index": 1,
                "lift": "snatch",
                "score": 7,
                "band": "solid",
                "cues": ["bar close through the pull"],
                "annotated_video_url": "/api/sample/replay/1.mp4",
                "start_sec": 0.0,
                "end_sec": 4.0,
                "confidence": 0.72,
            }
        ],
        "status": "done",
        "stage": "done",
        "progress": 1.0,
    },
    "replays": {"1": "sample.mp4"},
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "sk-test")
    reset_guard(None)
    sample_module.reset_remote_cache_for_tests()
    with TestClient(app) as test_client:
        yield test_client
    sample_module.reset_remote_cache_for_tests()


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _fake_urlopen(body: bytes):
    def _opener(url, timeout=10):  # noqa: ARG001
        return _FakeResponse(body)
    return _opener


def test_sample_served_from_remote_json_without_a_file(client, monkeypatch):
    monkeypatch.setenv("SAMPLE_RESULT_URL", "https://files.example/sample.json")
    monkeypatch.delenv("SAMPLE_CLIP_URL", raising=False)
    monkeypatch.delenv("DEMO_SAMPLE_DIR", raising=False)

    body = json.dumps(SAMPLE_BODY).encode("utf-8")
    with patch("server.services.sample.urllib.request.urlopen", _fake_urlopen(body)):
        response = client.get("/api/sample")
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample"] is True
    assert payload["lifts"][0]["score"] == 7
    # No local replay was served, and no SAMPLE_CLIP_URL was set, so the URL
    # falls back to null — the verdict still shows, without a video.
    assert payload["lifts"][0]["annotated_video_url"] is None
    assert get_guard().used_today() == 0


def test_sample_clip_url_overrides_the_replay(client, monkeypatch):
    monkeypatch.setenv("SAMPLE_RESULT_URL", "https://files.example/sample.json")
    monkeypatch.setenv("SAMPLE_CLIP_URL", "https://cdn.example/replay.mp4")
    monkeypatch.delenv("DEMO_SAMPLE_DIR", raising=False)

    body = json.dumps(SAMPLE_BODY).encode("utf-8")
    with patch("server.services.sample.urllib.request.urlopen", _fake_urlopen(body)):
        response = client.get("/api/sample")
    assert response.status_code == 200
    assert response.json()["lifts"][0]["annotated_video_url"] == "https://cdn.example/replay.mp4"


def test_sample_endpoint_makes_no_model_calls(client, monkeypatch):
    """
    /api/sample must not touch NebiusChatClient — the sample is cached JSON,
    not a run. Replace the only post call site; a hit would raise.
    """
    monkeypatch.setenv("SAMPLE_RESULT_URL", "https://files.example/sample.json")
    body = json.dumps(SAMPLE_BODY).encode("utf-8")

    def _should_not_be_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("Nebius was called during /api/sample")

    with patch("server.services.sample.urllib.request.urlopen", _fake_urlopen(body)), \
            patch("server.services.nebius_client.requests.post", side_effect=_should_not_be_called):
        response = client.get("/api/sample")
        assert response.status_code == 200
    # And the demo budget was not charged.
    assert get_guard().used_today() == 0


def test_remote_sample_cache_is_hit_on_second_call(client, monkeypatch):
    monkeypatch.setenv("SAMPLE_RESULT_URL", "https://files.example/sample.json")
    body = json.dumps(SAMPLE_BODY).encode("utf-8")

    calls = {"count": 0}

    def _counting(url, timeout=10):  # noqa: ARG001
        calls["count"] += 1
        return _FakeResponse(body)

    with patch("server.services.sample.urllib.request.urlopen", _counting):
        for _ in range(3):
            assert client.get("/api/sample").status_code == 200
    assert calls["count"] == 1
