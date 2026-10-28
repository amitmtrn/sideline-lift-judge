from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple

# How far before the wanted start a seek aims. OpenCV's seek lands near, not
# on, the target — up to ~0.4 s either side on phone HEVC — so it aims early
# and decodes forward; if it still lands late it tries from further back.
SEEK_MARGIN_SEC = 1.0


def frame_time(capture) -> float:
    """
    The presentation time, in seconds, of the frame `capture` last grabbed.
    Rounded to the microsecond: container timestamps are whole ticks, and the
    float noise of dividing them (0.30000000000000004) must not decide which
    side of a threshold a frame falls.
    """
    import cv2  # type: ignore

    return round(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 6)


def _grabbed_from(video_path: str, start_sec: float):
    """
    A capture with a frame grabbed at or before start_sec, and that frame's
    presentation time; (None, 0.0) when the video cannot be read. The caller
    releases the capture.
    """
    import cv2  # type: ignore

    start_sec = max(0.0, start_sec)
    for margin in (SEEK_MARGIN_SEC, 4 * SEEK_MARGIN_SEC, None):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return None, 0.0
        aim = 0.0 if margin is None else max(0.0, start_sec - margin)
        if aim > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, aim * 1000.0)
        if not capture.grab():
            capture.release()
            return None, 0.0
        t = frame_time(capture)
        if t > start_sec + 1e-3 and aim > 0:
            capture.release()  # landed past the start: aim further back
            continue
        return capture, t
    return None, 0.0


def read_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    *,
    max_fps: Optional[float] = None,
) -> Iterator[Tuple[float, "np.ndarray"]]:
    """
    (t, BGR frame) for the frames presented between start_sec and end_sec,
    thinned to at most `max_fps`, where t is the frame's own presentation
    time in seconds — the clock ffmpeg cuts by and a browser plays by.

    Every reader of the video goes through here or read_frames_at, so the
    whole app shares one clock. Frame number / fps is not that clock: phone
    footage is variable frame rate, and on a clip recorded at 30-60 fps it
    drifts from real time by up to two seconds. Seeking by frame number is
    worse still — OpenCV turns the number into a time at the average rate,
    and lands as much as 1.4 s from the frame asked for.
    """
    import cv2  # type: ignore

    capture, t = _grabbed_from(video_path, start_sec)
    if capture is None:
        return
    try:
        due = start_sec
        while t <= end_sec + 1e-6:
            if t >= start_sec - 1e-6 and t >= due - 1e-6:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    yield t, frame
                    if max_fps:
                        # A hair under the period, so jitter in the timestamps
                        # does not skip a frame that is due.
                        due = t + 0.98 / max_fps
            if not capture.grab():
                return
            t = frame_time(capture)
    finally:
        capture.release()


def read_frames_at(video_path: str, times: Sequence[float]) -> List[Optional["np.ndarray"]]:
    """
    The frame presented closest to each of `times` (within half a frame), in
    one pass, converting only the frames that are kept.
    """
    import cv2  # type: ignore

    out: List[Optional["np.ndarray"]] = [None] * len(times)
    if not times:
        return out
    order = sorted(range(len(times)), key=lambda k: times[k])
    capture, t = _grabbed_from(video_path, times[order[0]])
    if capture is None:
        return out
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        half_frame = 0.5 / fps if 0 < fps <= 480 else 1.0 / 60
        position = 0
        while position < len(order):
            if t >= times[order[position]] - half_frame:
                ok, frame = capture.retrieve()
                while position < len(order) and t >= times[order[position]] - half_frame:
                    out[order[position]] = frame if ok else None
                    position += 1
            if position >= len(order) or not capture.grab():
                break
            t = frame_time(capture)
    finally:
        capture.release()
    return out


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
    import numpy as np  # type: ignore

    end_sec = max(start_sec, end_sec)
    times = [float(t) for t in np.linspace(max(0.0, start_sec), end_sec, num=max(1, target_count))]
    frames = read_frames_at(video_path, times)
    out: List[np.ndarray] = []
    for frame in frames:
        # A short window has fewer frames than samples; do not repeat one.
        if frame is not None and not any(frame is kept for kept in out):
            out.append(frame)
    return out


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

