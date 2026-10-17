import os
import tempfile
from typing import List

import numpy as np  # type: ignore

from server.utils.frames import sample_video_frames_bgr


def _write_dummy_video(path: str, width: int = 64, height: int = 48, frames: int = 20, fps: int = 10) -> None:
    import cv2  # type: ignore

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened()
    try:
        for i in range(frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(img, f"{i}", (5, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(img)
    finally:
        writer.release()


def test_sample_video_frames_even_spacing():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.mp4")
        _write_dummy_video(path, frames=24, fps=12)
        frames = sample_video_frames_bgr(path, target_count=12)
        assert len(frames) == 12
        # Each frame is an image array
        assert all(hasattr(f, "shape") and len(f.shape) == 3 for f in frames)


def test_sample_video_frames_clamped_bounds():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "clip.mp4")
        _write_dummy_video(path, frames=6, fps=6)
        frames = sample_video_frames_bgr(path, target_count=8)
        # With only 6 frames available we should get up to 6
        assert 1 <= len(frames) <= 6

