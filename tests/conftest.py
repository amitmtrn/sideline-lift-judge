import numpy as np  # type: ignore
import pytest


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
