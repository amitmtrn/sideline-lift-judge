from __future__ import annotations

from typing import List


def sample_frames_in_range_bgr(
    video_path: str,
    *,
    start_sec: float,
    end_sec: float,
    target_count: int = 12,
) -> List["np.ndarray"]:
    """
    Evenly sample up to `target_count` frames from the [start_sec, end_sec] window.
    Returns frames as BGR numpy arrays. Skips unreadable frames.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (fps > 0) or fps > 480:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        first = max(0, int(round(max(0.0, start_sec) * fps)))
        last = int(round(max(start_sec, end_sec) * fps))
        if total_frames > 0:
            last = min(last, total_frames - 1)
        if last <= first:
            last = first
        samples = max(1, min(target_count, last - first + 1))
        indices = np.unique(np.linspace(first, last, num=samples, dtype=int))

        frames: List[np.ndarray] = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def sample_video_frames_bgr(video_path: str, target_count: int = 12) -> List["np.ndarray"]:
    """
    Evenly sample up to `target_count` frames across the clip using OpenCV.
    Returns frames as BGR numpy arrays. Skips unreadable frames.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            # Fallback: read sequentially until exhaustion and downsample
            all_frames: List[np.ndarray] = []
            ok, frame = cap.read()
            while ok:
                all_frames.append(frame)
                ok, frame = cap.read()
            if not all_frames:
                return []
            step = max(1, len(all_frames) // max(1, target_count))
            return [all_frames[i] for i in range(0, len(all_frames), step)][:target_count]

        # Compute sample indices (avoid absolute first/last which may be black)
        samples = min(target_count, max(1, total_frames))
        indices = np.linspace(1, total_frames - 2, num=samples, dtype=int) if total_frames >= 3 else list(range(samples))
        frames: List[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        cap.release()

