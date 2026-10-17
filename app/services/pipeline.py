"""
One scoring run, shared by the web app and the CLIs.

    Stage 0a  motion scan (local OpenCV)    -> candidate windows
    Stage 0b  triage (vision model)         -> which lift(s) a window holds
    Stage 1   observations (vision model)   -> factual bar-path description
    Stage 2   judge (Nemotron)              -> validated 1-10 JSON
              confidence (local)            -> caps + warnings (FR-8)
              replay (ffmpeg)               -> annotated MP4 (FR-10)

A clean & jerk is two lifts in one window: triage reports both, the window
splits at the quiet moment between them, and each half is judged on its own, as
the spec requires (§3, FR-14). That split happens in `auto` mode and also in
`single` mode, where it is what catches the mixed clip of FR-9.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

from app.schemas.judgment import LiftJudgment, LiftType
from app.schemas.segment import SegmentTriage
from app.services.confidence import assess_clip_quality, apply_quality
from app.services.nebius_client import (
    NebiusChatClient,
    classify_segment,
    collect_observations,
    judge_model,
    request_judgment,
    vision_max_images,
    vision_model,
)
from app.services.replay import render_annotated_replay
from app.utils.frames import sample_frames_in_range_bgr
from app.utils.image_ops import frames_to_data_urls, prepare_frames
from app.utils.segmentation import (
    CandidateSegment,
    MotionProfile,
    compute_motion_profile,
    detect_segments,
    split_segment_at_valley,
)
from app.utils.video_meta import VideoInfo, probe_video

logger = logging.getLogger(__name__)

MIN_FRAMES = 8
MAX_FRAMES = 16
DEFAULT_FRAMES = 10
# Triage reads the bar's height frame by frame, so it needs Stage-1 resolution:
# at 512px the model mistook a front rack for a lockout overhead.
TRIAGE_FRAMES = 8
TRIAGE_LONG_SIDE = 768
# A session clip can hold a whole training hour; cap the work a single upload
# can trigger so one request cannot exhaust the token budget.
MAX_AUTO_SEGMENTS = 8


class ProgressReporter(Protocol):
    def __call__(self, stage: str, message: str, fraction: float) -> None: ...


def _noop_progress(stage: str, message: str, fraction: float) -> None:
    return None


async def _offload(function, *args, **kwargs):
    """
    Run blocking work — OpenCV decoding, ffprobe, ffmpeg — on a worker thread.
    These take seconds; on the event loop they would stall every other request
    the server is handling, including the progress polls for this very job.
    """
    if kwargs:
        from functools import partial

        function = partial(function, **kwargs)
    return await asyncio.to_thread(function, *args)


def clamp_frames(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_FRAMES
    try:
        value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_FRAMES
    return max(MIN_FRAMES, min(MAX_FRAMES, value))


class TrimError(ValueError):
    """The requested trim window is not inside the clip."""


def resolve_trim(
    *,
    duration_sec: float,
    trim_start: Optional[float],
    trim_end: Optional[float],
) -> Tuple[float, float, bool]:
    """
    Validate the requested window against the clip, per spec §10:
    0 <= start < end <= duration. Returns (start, end, trimmed).
    A missing duration (no ffprobe) means we cannot check the upper bound, so the
    request is taken at face value.
    """
    if trim_start is None and trim_end is None:
        return 0.0, duration_sec, False

    start = 0.0 if trim_start is None else float(trim_start)
    end = float(trim_end) if trim_end is not None else duration_sec
    if start < 0:
        raise TrimError("Trim start cannot be negative.")
    if duration_sec > 0:
        if start >= duration_sec:
            raise TrimError("Trim start is past the end of the clip.")
        end = min(end, duration_sec) if end > 0 else duration_sec
    if end <= start:
        raise TrimError("Trim end must come after trim start.")
    return start, end, True


@dataclass
class ScoredLift:
    """One judged lift, with where it came from and what we rendered for it."""

    index: int
    lift: LiftType
    start_sec: float
    end_sec: float
    judgment: Optional[LiftJudgment] = None
    was_split: bool = False
    activity: str = ""
    observations: Optional[str] = None
    replay_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ScoreOutcome:
    lifts: List[ScoredLift] = field(default_factory=list)
    duration_sec: float = 0.0
    start_sec: float = 0.0
    end_sec: float = 0.0
    candidate_windows: int = 0
    skipped: List[str] = field(default_factory=list)
    split_detected: bool = False
    info: Optional[VideoInfo] = None

    @property
    def scored(self) -> List[ScoredLift]:
        return [item for item in self.lifts if item.judgment is not None]


@dataclass
class ScoreRequest:
    """
    `lift=None` means let triage decide. `mode="auto"` scans a longer clip for
    every attempt in it; `mode="single"` judges one window but still splits a
    clean & jerk it finds there.
    """

    video_path: str
    lift: Optional[LiftType] = None
    trim_start: Optional[float] = None
    trim_end: Optional[float] = None
    frames: int = DEFAULT_FRAMES
    mode: str = "single"
    render_replay: bool = True
    replay_dir: Optional[str] = None
    keep_observations: bool = False
    detect_split: bool = True


async def run_scoring(
    request: ScoreRequest,
    *,
    client: NebiusChatClient,
    prompts_dir: str,
    progress: ProgressReporter = _noop_progress,
) -> ScoreOutcome:
    """
    Score one upload end to end. Never raises for a single lift that fails: that
    lift carries its `error` and the rest of the run continues.
    """
    frames_wanted = clamp_frames(request.frames)
    info = await _offload(probe_video, request.video_path)
    rotation = info.rotation

    progress("preparing", "Reading the clip…", 0.05)
    start, end, trimmed = resolve_trim(
        duration_sec=info.duration_sec,
        trim_start=request.trim_start,
        trim_end=request.trim_end,
    )
    outcome = ScoreOutcome(
        duration_sec=info.duration_sec,
        start_sec=start,
        end_sec=end if end > start else info.duration_sec,
        info=info,
    )

    profile: MotionProfile = MotionProfile()
    plan: List[Tuple[CandidateSegment, LiftType, SegmentTriage, bool]] = []

    if request.mode == "auto":
        progress("splitting", "Finding the lifts in this clip…", 0.12)
        profile = await _offload(compute_motion_profile, request.video_path)
        windows = [
            window
            for window in detect_segments(profile, max_segments=MAX_AUTO_SEGMENTS)
            if window.end_sec > start and window.start_sec < outcome.end_sec
        ]
        outcome.candidate_windows = len(windows)
        for window in windows:
            # Clip each window to the trimmed range the user asked for.
            window.start_sec = max(window.start_sec, start)
            window.end_sec = min(window.end_sec, outcome.end_sec)
        plan = await _plan_windows(
            client=client,
            request=request,
            prompts_dir=prompts_dir,
            windows=windows,
            profile=profile,
            rotation=rotation,
            outcome=outcome,
            progress=progress,
        )
    else:
        window = CandidateSegment(index=1, start_sec=start, end_sec=outcome.end_sec)
        outcome.candidate_windows = 1
        if request.detect_split:
            progress("splitting", "Checking whether this is one lift…", 0.12)
            profile = await _offload(compute_motion_profile, request.video_path)
            plan = await _plan_windows(
                client=client,
                request=request,
                prompts_dir=prompts_dir,
                windows=[window],
                profile=profile,
                rotation=rotation,
                outcome=outcome,
                progress=progress,
                fallback_lift=request.lift,
            )
        if not plan:
            lift = request.lift or LiftType.snatch
            plan = [(window, lift, SegmentTriage(lifts=[lift], activity="as selected"), False)]

    if not plan:
        return outcome

    outcome.split_detected = any(was_split for *_, was_split in plan)
    total = len(plan)
    for position, (window, lift, triage, was_split) in enumerate(plan, start=1):
        share_from = 0.25 + 0.7 * (position - 1) / total
        share_to = 0.25 + 0.7 * position / total
        scored = await _score_one_lift(
            client=client,
            request=request,
            prompts_dir=prompts_dir,
            window=window,
            lift=lift,
            triage=triage,
            was_split=was_split,
            index=position,
            total=total,
            rotation=rotation,
            info=info,
            frames_wanted=frames_wanted,
            progress=progress,
            progress_span=(share_from, share_to),
        )
        outcome.lifts.append(scored)
    return outcome


async def _plan_windows(
    *,
    client: NebiusChatClient,
    request: ScoreRequest,
    prompts_dir: str,
    windows: List[CandidateSegment],
    profile: MotionProfile,
    rotation: int,
    outcome: ScoreOutcome,
    progress: ProgressReporter,
    fallback_lift: Optional[LiftType] = None,
) -> List[Tuple[CandidateSegment, LiftType, SegmentTriage, bool]]:
    """
    Triage each window and expand it into one entry per lift. A window holding
    two lifts splits at its quietest interior moment so the clean and the jerk
    are judged separately.
    """
    plan: List[Tuple[CandidateSegment, LiftType, SegmentTriage, bool]] = []
    for window in windows:
        if window.end_sec - window.start_sec <= 0.2:
            continue
        frames = await _offload(
            sample_frames_in_range_bgr,
            request.video_path,
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            target_count=min(TRIAGE_FRAMES, vision_max_images()),
        )
        if not frames:
            outcome.skipped.append(f"{_stamp(window)} — no readable frames")
            continue
        try:
            images = await _offload(
                _encode_frames, frames, rotation, TRIAGE_LONG_SIDE
            )
            triage = await classify_segment(
                client=client,
                images=images,
                prompts_dir=prompts_dir,
                model=vision_model(),
            )
        except Exception as exc:
            logger.warning("Triage failed for %s: %s", _stamp(window), exc)
            triage = SegmentTriage(lifts=[], activity="triage failed")

        lifts = list(triage.lifts)
        if not lifts:
            if fallback_lift is not None:
                # Single-clip mode: the user told us what this is, and triage is
                # only here to catch a clip holding more than one lift.
                plan.append((window, fallback_lift, triage, False))
                continue
            outcome.skipped.append(f"{_stamp(window)} — {triage.activity or 'no scored lift'}")
            continue

        if len(lifts) == 1:
            plan.append((window, lifts[0], triage, False))
            continue

        parts: List[CandidateSegment] = []
        if len(lifts) == 2:
            split = split_segment_at_valley(profile, window)
            if split is not None:
                parts = list(split)
        if len(parts) != len(lifts):
            # No usable quiet moment: judge the same window once per lift and
            # let the confidence rule flag that the phases may be blended.
            parts = [window] * len(lifts)
            for part, lift in zip(parts, lifts):
                plan.append((part, lift, triage, False))
            continue
        for part, lift in zip(parts, lifts):
            plan.append((part, lift, triage, True))
    return plan


async def _score_one_lift(
    *,
    client: NebiusChatClient,
    request: ScoreRequest,
    prompts_dir: str,
    window: CandidateSegment,
    lift: LiftType,
    triage: SegmentTriage,
    was_split: bool,
    index: int,
    total: int,
    rotation: int,
    info: VideoInfo,
    frames_wanted: int,
    progress: ProgressReporter,
    progress_span: Tuple[float, float],
) -> ScoredLift:
    low, high = progress_span
    label = f"{lift.value}" if total == 1 else f"{lift.value} ({index} of {total})"
    scored = ScoredLift(
        index=index,
        lift=lift,
        start_sec=round(window.start_sec, 2),
        end_sec=round(window.end_sec, 2),
        was_split=was_split,
        activity=triage.activity,
    )

    frames = await _offload(
        sample_frames_in_range_bgr,
        request.video_path,
        start_sec=window.start_sec,
        end_sec=window.end_sec,
        target_count=frames_wanted,
    )
    if not frames:
        scored.error = "Could not read frames from this part of the clip."
        return scored
    processed = await _offload(prepare_frames, frames, rotation_deg=rotation)

    try:
        progress("observing", f"Reading bar path — {label}…", low + 0.1 * (high - low))
        observations = await collect_observations(
            client=client,
            lift=lift.value,
            images=await _offload(frames_to_data_urls, processed),
            prompts_dir=prompts_dir,
            model=vision_model(),
        )
        if request.keep_observations:
            scored.observations = observations

        progress("judging", f"Judging — {label}…", low + 0.45 * (high - low))
        judgment = await request_judgment(
            client=client,
            lift=lift.value,
            observations=observations,
            prompts_dir=prompts_dir,
            model=judge_model(),
        )
    except Exception as exc:
        logger.exception("Scoring failed for %s", label)
        scored.error = f"{type(exc).__name__}: {exc}"
        return scored

    quality = assess_clip_quality(
        observations=observations,
        info=info,
        frame_count=len(processed),
        duration_sec=window.end_sec - window.start_sec,
        lifts_in_window=len(triage.lifts),
        was_split=was_split,
    )
    scored.judgment = apply_quality(judgment, quality)

    if request.render_replay and request.replay_dir:
        progress("rendering", f"Rendering replay — {label}…", low + 0.75 * (high - low))
        replay = await _offload(
            render_annotated_replay,
            request.video_path,
            scored.judgment,
            str(Path(request.replay_dir) / f"replay_{index:02d}_{lift.value}.mp4"),
            start_sec=window.start_sec,
            end_sec=window.end_sec,
        )
        scored.replay_path = replay
        if replay is None:
            logger.info("Replay rendering unavailable for %s", label)
    return scored


def _encode_frames(frames, rotation_deg: int, long_side: int) -> List[str]:
    """Rotate, downscale and JPEG-encode in one hop onto the worker thread."""
    return frames_to_data_urls(prepare_frames(frames, rotation_deg=rotation_deg, long_side=long_side))


def _stamp(window: CandidateSegment) -> str:
    from app.utils.clips import format_timestamp

    return f"{format_timestamp(window.start_sec)}–{format_timestamp(window.end_sec)}"
