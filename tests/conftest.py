import time
from unittest.mock import patch

import numpy as np  # type: ignore
import pytest

# Every demo limit the app reads from the environment. The API fixtures clear
# them so a developer's .env cannot leak a cap into the tests.
DEMO_ENV_VARS = (
    "ACCESS_MODE",
    "DEMO_ACCESS_CODE",
    "DEMO_DISABLED",
    "DEMO_DAILY_LIMIT",
    "DEMO_DAILY_RESERVE_PERCENT",
    "DEMO_DAY_TIMEZONE",
    "DEMO_IP_HOURLY_LIMIT",
    "DEMO_SAMPLE_DIR",
    "MAX_CONCURRENT_JOBS",
    "MAX_CLIP_SECONDS",
    "MAX_LIFTS_PER_CLIP",
    "MAX_FRAMES",
    "JOB_TIMEOUT_SECONDS",
    "TRUSTED_PROXIES",
    "NEBIUS_JUDGE_MODEL",
    "NEBIUS_JUDGE_FALLBACK_MODEL",
)

SNATCH_PATH = ["floor", "knees", "overhead", "overhead", "overhead", "overhead"]
CLEAN_AND_JERK_PATH = ["floor", "knees", "shoulders", "shoulders", "overhead", "overhead"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """The app under TestClient, with a fresh job store and demo guard, no limits set."""
    from fastapi.testclient import TestClient

    from server.services.demo_guard import reset_guard
    from server.services.jobs import JobStore, reset_store

    monkeypatch.setenv("NEBIUS_API_KEY", "test-key")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    reset_store(JobStore(root=tmp_path / "jobs"))
    reset_guard()
    from server.main import app  # load_dotenv runs here, so clear the env after it

    for name in DEMO_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with TestClient(app) as test_client:
        yield test_client
    reset_store(JobStore(root=tmp_path / "jobs-after"))
    reset_guard()


@pytest.fixture
def clip(tmp_path):
    return write_lift_video(str(tmp_path / "clip.mp4"))


def upload(client, clip, responses, *, headers=None, **fields):
    """POST a clip to /api/jobs with the model mocked, and wait for the verdict."""
    data = {"lift": "snatch", "mode": "single"}
    data.update(fields)
    with open(clip, "rb") as handle:
        with patch("server.services.nebius_client.requests.post", side_effect=responses):
            response = client.post(
                "/api/jobs",
                data=data,
                files={"video": ("clip.mp4", handle, "video/mp4")},
                headers=headers or {},
            )
            if response.status_code != 202:
                return response, None
            job_id = response.json()["id"]
            payload = wait_for(client, job_id)
    return response, payload


def wait_for(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def write_lift_video(
    path: str,
    *,
    busy_windows=((1.0, 3.0),),
    duration: float = 5.0,
    fps: int = 20,
    width: int = 96,
    height: int = 72,
) -> str:
    """
    A stand-in for a phone clip: a still frame, with a bright block travelling up
    the frame during each busy window so the motion scan has something to find.
    """
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer"
    try:
        for index in range(int(duration * fps)):
            now = index / fps
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            for start, end in busy_windows:
                if start <= now <= end:
                    progress = (now - start) / max(1e-6, end - start)
                    y = int((height - 20) * (1 - progress)) + 2
                    cv2.rectangle(frame, (20, y), (70, min(height - 1, y + 16)), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def lift_video(tmp_path):
    return write_lift_video(str(tmp_path / "lift.mp4"))


def chat_response(content: str):
    """An OpenAI-compatible 200 reply carrying `content`."""
    from unittest.mock import Mock

    response = Mock()
    response.status_code = 200
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


JUDGE_JSON = (
    '{"lift":"%s","score":%d,"band":"%s","max_horizontal_drift":"forward loop in the second pull",'
    '"cues":["keep the bar close to the body"],"confidence":%s}'
)


def judge_reply(lift: str = "snatch", score: int = 6, band: str = "mixed", confidence: float = 0.85):
    return chat_response(JUDGE_JSON % (lift, score, band, confidence))


def triage_reply(bar_by_frame, *, activity: str = "barbell lift", confidence: float = 0.8):
    import json

    return chat_response(
        json.dumps(
            {
                "bar_by_frame": list(bar_by_frame),
                "athlete_lifting": True,
                "activity": activity,
                "confidence": confidence,
            }
        )
    )


def observation_reply(text: str = "Bar stays close through the first pull; catch over mid-foot."):
    return chat_response(text)
