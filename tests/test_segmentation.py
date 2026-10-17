import os
import tempfile
from typing import List, Tuple

import numpy as np  # type: ignore

from server.utils.clips import format_timestamp
from server.utils.segmentation import (
    MotionProfile,
    compute_motion_profile,
    detect_segments,
    split_segment_at_valley,
)


def _write_session_video(
    path: str,
    *,
    busy_windows: List[Tuple[float, float]],
    duration: float = 30.0,
    fps: int = 20,
    width: int = 96,
    height: int = 72,
) -> None:
    """
    Fake a session clip: a still frame most of the time, with a bright square
    flying across the frame during each busy window (an "exercise").
    """
    import cv2  # type: ignore

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    try:
        total = int(duration * fps)
        for index in range(total):
            time_value = index / fps
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            for start, end in busy_windows:
                if start <= time_value <= end:
                    progress = (time_value - start) / max(1e-6, end - start)
                    y = int((height - 20) * (1 - progress)) + 2
                    cv2.rectangle(frame, (20, y), (70, min(height - 1, y + 16)), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_detect_segments_finds_each_busy_window():
    windows = [(3.0, 6.0), (13.0, 16.5), (23.0, 26.0)]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=windows)
        profile = compute_motion_profile(path, sample_fps=10.0)
        assert profile and profile.duration_sec > 25

        segments = detect_segments(profile)
        assert len(segments) == len(windows)
        for segment, (start, end) in zip(segments, windows):
            # Padding widens the window, so assert containment rather than equality.
            assert segment.start_sec <= start + 0.5
            assert segment.end_sec >= end - 0.5
            assert segment.peak_energy > 0
        assert [s.index for s in segments] == [1, 2, 3]


def test_detect_segments_merges_windows_inside_the_gap():
    # Two bursts 0.6s apart: one exercise with a pause, not two exercises.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(2.0, 5.0), (5.6, 8.0)], duration=14.0)
        profile = compute_motion_profile(path, sample_fps=10.0)
        assert len(detect_segments(profile, merge_gap=1.0)) == 1


def test_detect_segments_drops_windows_under_min_duration():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(3.0, 6.0), (14.0, 14.4)], duration=20.0)
        profile = compute_motion_profile(path, sample_fps=10.0)
        segments = detect_segments(profile, min_duration=1.5)
        assert len(segments) == 1
        assert segments[0].start_sec < 3.5


def test_detect_segments_respects_max_segments():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(2.0, 5.0), (10.0, 13.0), (18.0, 21.0)], duration=26.0)
        profile = compute_motion_profile(path, sample_fps=10.0)
        segments = detect_segments(profile, max_segments=2)
        assert len(segments) == 2
        # Kept in clip order, not energy order.
        assert segments[0].start_sec < segments[1].start_sec


def test_detect_segments_trims_overlong_window():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(1.0, 19.0)], duration=22.0)
        profile = compute_motion_profile(path, sample_fps=10.0)
        segments = detect_segments(profile, max_duration=8.0)
        assert len(segments) == 1
        assert segments[0].duration_sec <= 8.0 + 1e-6


def test_static_clip_falls_back_to_whole_clip():
    # A clip with no quiet/busy contrast must still produce one segment, so a
    # plain single-lift clip keeps working.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "session.mp4")
        _write_session_video(path, busy_windows=[(0.0, 12.0)], duration=12.0)
        profile = compute_motion_profile(path, sample_fps=10.0)
        segments = detect_segments(profile)
        assert len(segments) >= 1
        assert segments[0].start_sec < 1.0


def test_empty_profile_yields_no_segments():
    assert detect_segments(MotionProfile()) == []


def test_split_segment_at_valley_cuts_at_the_quiet_moment():
    from server.utils.segmentation import CandidateSegment

    times = [i * 0.1 for i in range(101)]
    # Two humps of motion with a trough at ~5.0s: a clean, then a jerk.
    energy = [1.0 + np.cos((t - 2.5) / 2.5 * np.pi) if t < 5 else 1.0 + np.cos((t - 7.5) / 2.5 * np.pi) for t in times]
    profile = MotionProfile(times=times, energy=energy, duration_sec=10.0, fps=30.0, sample_fps=10.0)
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=10.0)

    split = split_segment_at_valley(profile, segment)
    assert split is not None
    first, second = split
    assert 4.0 < first.end_sec < 6.0
    assert second.start_sec == first.end_sec
    assert second.end_sec == 10.0


def test_split_segment_at_valley_refuses_short_window():
    from server.utils.segmentation import CandidateSegment

    profile = MotionProfile(times=[0.0, 0.5, 1.0], energy=[1.0, 0.2, 1.0], duration_sec=1.0, sample_fps=2.0)
    assert split_segment_at_valley(profile, CandidateSegment(index=1, start_sec=0.0, end_sec=1.5)) is None


def test_format_timestamp():
    assert format_timestamp(0) == "0:00.0"
    assert format_timestamp(9.25) == "0:09.2"
    assert format_timestamp(75.5) == "1:15.5"
