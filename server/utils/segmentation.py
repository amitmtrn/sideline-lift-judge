from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class MotionProfile:
    """Per-sample motion energy across a clip (Stage 0a: no model involved)."""

    times: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    duration_sec: float = 0.0
    fps: float = 0.0
    sample_fps: float = 0.0

    def __bool__(self) -> bool:
        return len(self.energy) > 0


@dataclass
class CandidateSegment:
    index: int
    start_sec: float
    end_sec: float
    peak_sec: float = 0.0
    peak_energy: float = 0.0
    mean_energy: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


# Defaults tuned for phone clips of a lifting session filmed from the side:
# an attempt runs a couple of seconds, with the athlete walking up and resetting
# between attempts.
DEFAULT_SAMPLE_FPS = 8.0
DEFAULT_SENSITIVITY = 0.25
# This measures the above-threshold core, not the window: the fast part of a
# jerk is about a second, so anything stricter drops jerks while keeping the
# bar being dumped at the end of a lift. Padding turns the core into a window
# wide enough to judge.
DEFAULT_MIN_DURATION = 0.6
DEFAULT_MAX_DURATION = 20.0
DEFAULT_MERGE_GAP = 1.0
DEFAULT_PAD = 0.8


def compute_motion_profile(
    video_path: str,
    *,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    analysis_width: int = 160,
) -> MotionProfile:
    """
    Walk the clip once and record how much the picture changes over time.

    Frames are decoded at `sample_fps`, shrunk to `analysis_width` and blurred,
    so sensor noise and compression artifacts do not register as motion. Frames
    between samples are grabbed without being converted, which keeps a long clip
    cheap to scan.
    """
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    from server.utils.frames import frame_time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return MotionProfile()
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (fps > 0) or fps > 480:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, int(round(fps / max(0.5, sample_fps))))

        times: List[float] = []
        energy: List[float] = []
        previous = None
        frame_index = 0
        last_time = 0.0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if frame_index % stride == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    height, width = gray.shape[:2]
                    if width > analysis_width:
                        scale = analysis_width / float(width)
                        gray = cv2.resize(
                            gray,
                            (analysis_width, max(1, int(round(height * scale)))),
                            interpolation=cv2.INTER_AREA,
                        )
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    # The frame's own presentation time: frame_index / fps drifts
                    # from it by seconds on variable-frame-rate phone footage.
                    timestamp = frame_time(cap)
                    if previous is not None:
                        times.append(timestamp)
                        energy.append(float(np.mean(cv2.absdiff(previous, gray))))
                    previous = gray
                    last_time = timestamp
            frame_index += 1

        duration = (total_frames / fps) if total_frames > 0 else last_time
        return MotionProfile(
            times=times,
            energy=energy,
            duration_sec=max(duration, last_time),
            fps=fps,
            sample_fps=fps / stride,
        )
    finally:
        cap.release()


def detect_segments(
    profile: MotionProfile,
    *,
    sensitivity: float = DEFAULT_SENSITIVITY,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    merge_gap: float = DEFAULT_MERGE_GAP,
    pad: float = DEFAULT_PAD,
    max_segments: Optional[int] = None,
) -> List[CandidateSegment]:
    """
    Turn a motion profile into candidate exercise windows.

    Samples above an adaptive threshold form runs; runs separated by less than
    `merge_gap` join (the pause inside a clean & jerk, or a brief stall at the
    knee); runs shorter than `min_duration` are dropped as fidgeting. A clip with
    no usable contrast between quiet and busy falls back to one whole-clip
    segment, so a single-lift clip still scores.
    """
    import numpy as np  # type: ignore

    if not profile or len(profile.energy) < 4:
        return _whole_clip(profile, max_duration)

    times = np.asarray(profile.times, dtype=float)
    energy = _smooth(np.asarray(profile.energy, dtype=float), profile.sample_fps)

    baseline = float(np.percentile(energy, 20))
    peak = float(np.percentile(energy, 98))
    span = peak - baseline
    # A locked-off camera on an empty platform, or continuous motion throughout:
    # either way there is nothing to cut on.
    if span <= 1e-6 or span < 0.02 * max(1.0, peak):
        return _whole_clip(profile, max_duration)

    threshold = baseline + max(0.0, min(1.0, sensitivity)) * span
    runs = _runs_above(times, energy, threshold)
    runs = _merge_close(runs, merge_gap)

    segments: List[CandidateSegment] = []
    for start, end in runs:
        if (end - start) < min_duration:
            continue
        segments.append(CandidateSegment(index=0, start_sec=start, end_sec=end))

    if not segments:
        return _whole_clip(profile, max_duration)

    padded = _merge_close(
        [(max(0.0, s.start_sec - pad), min(profile.duration_sec or s.end_sec + pad, s.end_sec + pad)) for s in segments],
        0.0,
    )
    segments = [CandidateSegment(index=0, start_sec=s, end_sec=e) for s, e in padded]
    segments = [_trim_to_max(s, times, energy, max_duration) for s in segments]
    for segment in segments:
        _annotate(segment, times, energy)

    if max_segments is not None and len(segments) > max_segments:
        # Keep the busiest windows, then restore clip order.
        segments = sorted(segments, key=lambda s: -s.peak_energy)[:max_segments]
        segments.sort(key=lambda s: s.start_sec)

    for position, segment in enumerate(segments, start=1):
        segment.index = position
    return segments


def split_segment_at_valley(
    profile: MotionProfile,
    segment: CandidateSegment,
    *,
    min_part: float = 1.0,
) -> Optional[Tuple[CandidateSegment, CandidateSegment]]:
    """
    Split one window in two at its quietest interior moment.

    Used when triage reports a clean & jerk in a single window: the athlete is
    briefly still between racking the clean and dipping for the jerk, which shows
    up as a dip in motion energy. Returns None when the window is too short to
    split or no interior dip exists.
    """
    import numpy as np  # type: ignore

    if segment.duration_sec < 2 * min_part or not profile:
        return None

    times = np.asarray(profile.times, dtype=float)
    energy = _smooth(np.asarray(profile.energy, dtype=float), profile.sample_fps)
    inside = (times >= segment.start_sec + min_part) & (times <= segment.end_sec - min_part)
    if not bool(inside.any()):
        return None

    candidate_times = times[inside]
    split_at = float(candidate_times[int(np.argmin(energy[inside]))])
    first = CandidateSegment(index=segment.index, start_sec=segment.start_sec, end_sec=split_at)
    second = CandidateSegment(index=segment.index, start_sec=split_at, end_sec=segment.end_sec)
    _annotate(first, times, energy)
    _annotate(second, times, energy)
    return first, second


def _whole_clip(profile: MotionProfile, max_duration: float = DEFAULT_MAX_DURATION) -> List[CandidateSegment]:
    """
    Fallback when the clip offers no quiet/busy contrast to cut on: one segment.
    Still capped at `max_duration` around the busiest moment — a long clip of
    uninterrupted motion is no more judgeable as one window than as none.
    """
    end = profile.duration_sec or (profile.times[-1] if profile.times else 0.0)
    if end <= 0:
        return []
    segment = CandidateSegment(index=1, start_sec=0.0, end_sec=end)
    if profile:
        import numpy as np  # type: ignore

        times = np.asarray(profile.times, dtype=float)
        energy = _smooth(np.asarray(profile.energy, dtype=float), profile.sample_fps)
        segment = _trim_to_max(segment, times, energy, max_duration)
        segment.index = 1
        _annotate(segment, times, energy)
    return [segment]


def _smooth(energy, sample_fps: float):
    import numpy as np  # type: ignore

    window = max(1, int(round(0.4 * max(1.0, sample_fps))))
    if window <= 1 or energy.size < window:
        return energy
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(energy, kernel, mode="same")


def _runs_above(times, energy, threshold: float) -> List[Tuple[float, float]]:
    runs: List[Tuple[float, float]] = []
    start: Optional[float] = None
    previous_time = float(times[0]) if len(times) else 0.0
    for time_value, energy_value in zip(times, energy):
        above = energy_value >= threshold
        if above and start is None:
            start = float(time_value)
        elif not above and start is not None:
            runs.append((start, previous_time))
            start = None
        previous_time = float(time_value)
    if start is not None:
        runs.append((start, previous_time))
    return runs


def _merge_close(runs: Sequence[Tuple[float, float]], gap: float) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(runs):
        if merged and start - merged[-1][1] <= gap:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _trim_to_max(segment: CandidateSegment, times, energy, max_duration: float) -> CandidateSegment:
    """Center an over-long window on its busiest moment rather than cutting the tail."""
    import numpy as np  # type: ignore

    if max_duration <= 0 or segment.duration_sec <= max_duration:
        return segment
    inside = (times >= segment.start_sec) & (times <= segment.end_sec)
    if bool(inside.any()):
        center = float(times[inside][int(np.argmax(energy[inside]))])
    else:
        center = (segment.start_sec + segment.end_sec) / 2
    half = max_duration / 2
    start = max(segment.start_sec, center - half)
    end = min(segment.end_sec, start + max_duration)
    start = max(segment.start_sec, end - max_duration)
    return CandidateSegment(index=segment.index, start_sec=start, end_sec=end)


def _annotate(segment: CandidateSegment, times, energy) -> None:
    import numpy as np  # type: ignore

    if times is None or len(times) == 0:
        return
    inside = (times >= segment.start_sec) & (times <= segment.end_sec)
    if not bool(inside.any()):
        return
    window = energy[inside]
    segment.peak_energy = float(np.max(window))
    segment.mean_energy = float(np.mean(window))
    segment.peak_sec = float(times[inside][int(np.argmax(window))])
