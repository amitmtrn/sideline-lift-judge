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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from server.schemas.judgment import LiftJudgment, LiftType
from server.schemas.segment import SegmentTriage
from server.services.confidence import assess_clip_quality, apply_quality, cross_check
from server.services.demo_guard import CALLS_PER_LIFT, BudgetExhausted
from server.services.errors import detail_of, user_message
from server.services.measurement import BarPathMeasurement, describe, measure
from server.services.nebius_client import (
    NebiusChatClient,
    classify_segment,
    collect_observations,
    judge_model,
    model_display_name,
    request_judgment,
    vision_max_images,
    vision_model,
)
from server.services.replay import render_annotated_replay
from server.utils.bar_track import BarPath, LiftWindow, detect_lift_windows, track_bar_path
from server.utils.frames import sample_frames_in_range_bgr
from server.utils.image_ops import frames_to_data_urls, prepare_frames
from server.utils.segmentation import (
    CandidateSegment,
    MotionProfile,
    compute_motion_profile,
    detect_segments,
    split_segment_at_valley,
)
from server.utils.video_meta import VideoInfo, probe_video, rotation_still_needed

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


class ReserveCalls(Protocol):
    """
    Told how many model calls the run is about to need, once its lift windows
    are known and before the first call. Raises (DemoRefused) to stop the run
    there, so a clip is turned away whole rather than scored part-way.
    """

    def __call__(self, calls: int) -> None: ...


def _noop_progress(stage: str, message: str, fraction: float) -> None:
    return None


def _noop_reserve(calls: int) -> None:
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


def max_frames() -> int:
    """
    Frames per lift ceiling. MAX_FRAMES in the environment lowers it (never
    above the hard ceiling, never below the minimum): at 10 or fewer every lift
    is one vision call, which is what the public demo wants.
    """
    raw = os.getenv("MAX_FRAMES")
    if raw is None or not raw.strip():
        return MAX_FRAMES
    try:
        value = int(float(raw))
    except ValueError:
        return MAX_FRAMES
    return max(MIN_FRAMES, min(MAX_FRAMES, value))


def clamp_frames(value: Optional[int]) -> int:
    if value is None:
        return min(DEFAULT_FRAMES, max_frames())
    try:
        value = int(value)
    except (TypeError, ValueError):
        return min(DEFAULT_FRAMES, max_frames())
    return max(MIN_FRAMES, min(max_frames(), value))


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
    bar_path_tracked: bool = False
    # What the tracker measured; goes to the judge, the API and the replay.
    measurement: Optional[BarPathMeasurement] = None
    # Which model did what for this verdict (see models_used), so the card can
    # say so — and say when the fallback judge answered instead of the primary.
    models: Dict[str, Any] = field(default_factory=dict)
    # `error` is one plain sentence for the user; `error_detail` is the raw
    # exception for logs and --debug.
    error: Optional[str] = None
    error_detail: Optional[str] = None
    # The demo budget ran out on this lift: the run stops here, keeping the
    # lifts before it.
    out_of_budget: bool = False


def models_used(judgment: Optional[LiftJudgment] = None) -> Dict[str, Any]:
    """
    The models behind a verdict, by id and by the name the page shows. The
    judge is the one that actually answered when there is a verdict, which may
    be the fallback; before one exists it is the configured primary.
    """
    vision = vision_model()
    primary = judge_model()
    answered = judgment.judged_by if judgment is not None and judgment.judged_by else None
    judge = answered or primary
    return {
        "vision": vision,
        "vision_name": model_display_name(vision),
        "judge": judge,
        "judge_name": model_display_name(judge),
        "judge_fallback": bool(answered) and answered != primary,
    }


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
    # Lift windows this one upload may score; the rest are listed as skipped.
    # 0 is no cap. The public demo sets it so one clip cannot cost a dozen calls.
    max_lifts: int = 0


async def run_scoring(
    request: ScoreRequest,
    *,
    client: NebiusChatClient,
    prompts_dir: str,
    progress: ProgressReporter = _noop_progress,
    reserve: ReserveCalls = _noop_reserve,
) -> ScoreOutcome:
    """
    Score one upload end to end. Never raises for a single lift that fails: that
    lift carries its `error` and the rest of the run continues. `reserve` is
    asked for the run's estimated cost — one lift's calls per window — once
    the windows are known and before any model call; what it raises ends the
    run there.
    """
    frames_wanted = clamp_frames(request.frames)
    info = await _offload(probe_video, request.video_path)
    # Rotation is decided per batch of decoded frames in _rotation_for, not from
    # info.rotation: OpenCV's decoder already applies the display matrix on most
    # builds, and rotating again would put a portrait clip on its side.

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
    lift_windows: Dict[int, LiftWindow] = {}

    if request.mode == "auto":
        progress("splitting", "Finding the lifts in this clip…", 0.12)
        # The motion profile is still what splits a clean & jerk at its quiet
        # moment; the windows themselves come from where the plate rises.
        profile = await _offload(compute_motion_profile, request.video_path)
        windows = await _lift_windows(request.video_path, start, outcome.end_sec, lift_windows)
        if not windows:
            # Nothing tracked — fall back to motion energy, which at least finds
            # where something happened.
            windows = [
                window
                for window in detect_segments(profile, max_segments=MAX_AUTO_SEGMENTS)
                if window.end_sec > start and window.start_sec < outcome.end_sec
            ]
            for window in windows:
                window.start_sec = max(window.start_sec, start)
                window.end_sec = min(window.end_sec, outcome.end_sec)
        windows = _cap_windows(windows, request.max_lifts, outcome)
        outcome.candidate_windows = len(windows)
        reserve(len(windows) * CALLS_PER_LIFT)
        plan = await _plan_windows(
            client=client,
            request=request,
            prompts_dir=prompts_dir,
            windows=windows,
            profile=profile,
            info=info,
            outcome=outcome,
            progress=progress,
            lift_windows=lift_windows,
        )
    else:
        window = CandidateSegment(index=1, start_sec=start, end_sec=outcome.end_sec)
        outcome.candidate_windows = 1
        if request.detect_split:
            progress("splitting", "Checking whether this is one lift…", 0.12)
            profile = await _offload(compute_motion_profile, request.video_path)
            # Inside what the user trimmed, narrow to where the bar actually
            # rises so the sampled frames are the lift, not the setup — and a
            # clean & jerk comes out as its two rises.
            windows = await _lift_windows(request.video_path, start, outcome.end_sec, lift_windows) or [window]
            windows = _cap_windows(windows, request.max_lifts, outcome)
            outcome.candidate_windows = len(windows)
            reserve(len(windows) * CALLS_PER_LIFT)
            plan = await _plan_windows(
                client=client,
                request=request,
                prompts_dir=prompts_dir,
                windows=windows,
                profile=profile,
                info=info,
                outcome=outcome,
                progress=progress,
                fallback_lift=request.lift,
                lift_windows=lift_windows,
            )
        if not plan:
            lift = request.lift or LiftType.snatch
            plan = [(window, lift, SegmentTriage(lifts=[lift], activity="as selected"), False)]
            if not request.detect_split:
                reserve(CALLS_PER_LIFT)

    if not plan:
        return outcome

    if request.max_lifts and len(plan) > request.max_lifts:
        # A window that split into two lifts can push the plan past the cap
        # the windows were held to; the cap is on what is scored.
        for window, lift, _triage, _split in plan[request.max_lifts :]:
            outcome.skipped.append(
                f"{_stamp(window)} — {lift.value} not scored: this demo scores up to "
                f"{request.max_lifts} lifts per clip"
            )
        plan = plan[: request.max_lifts]

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
            info=info,
            frames_wanted=frames_wanted,
            progress=progress,
            progress_span=(share_from, share_to),
            lift_window=lift_windows.get(window.index),
        )
        outcome.lifts.append(scored)
        if scored.out_of_budget:
            # Every later lift would fail the same way before its first call;
            # do not spend the tracking and frame work on them.
            break
    return outcome


def _cap_windows(windows: List[CandidateSegment], cap: int, outcome: ScoreOutcome) -> List[CandidateSegment]:
    """
    Hold a clip to `cap` lift windows (0 = no cap). The earliest are kept —
    they are the ones the visitor filmed first — and the rest are listed as
    skipped, with the reason, rather than silently dropped.
    """
    if not cap or len(windows) <= cap:
        return windows
    for window in windows[cap:]:
        outcome.skipped.append(f"{_stamp(window)} — not scored: this demo scores up to {cap} lifts per clip")
    return windows[:cap]


async def _plan_windows(
    *,
    client: NebiusChatClient,
    request: ScoreRequest,
    prompts_dir: str,
    windows: List[CandidateSegment],
    profile: MotionProfile,
    info: VideoInfo,
    outcome: ScoreOutcome,
    progress: ProgressReporter,
    fallback_lift: Optional[LiftType] = None,
    lift_windows: Optional[Dict[int, LiftWindow]] = None,
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
                _encode_frames, frames, _rotation_for(info, frames), TRIAGE_LONG_SIDE
            )
            triage = await classify_segment(
                client=client,
                images=images,
                prompts_dir=prompts_dir,
                model=vision_model(),
            )
        except BudgetExhausted:
            # Not a bad window: the demo budget is spent. Say so instead of
            # reporting that nothing was found.
            raise
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
        tracked_window = (lift_windows or {}).get(window.index)
        if len(lifts) == 2 and tracked_window is not None:
            # The window's own trail knows how many times the bar rose. Two
            # rises: split between them.
            parts, lifts = _split_by_rises(window, tracked_window, lifts)
            if lifts != list(triage.lifts):
                # Resolved by the bar's own travel: the confidence rule must
                # not cap this for "holding more than one lift".
                triage = triage.model_copy(update={"lifts": lifts})
        if len(lifts) == 2 and not parts:
            # One rise only (a clean & jerk so quick the two run together), or
            # no trail: split at the quiet moment between them, as long as each
            # half could hold a lift. A one-second "jerk" is a slice of the
            # clean, not a lift, and then only the first lift is scored.
            split = split_segment_at_valley(profile, window)
            if split is not None and all(part.duration_sec >= MIN_SPLIT_PART for part in split):
                parts = list(split)
            else:
                # Score the first lift only — but the frames still hold both,
                # so triage keeps its two lifts and the confidence rule caps
                # the verdict and says why (AC-4).
                lifts = lifts[:1]
        if len(lifts) == 1:
            plan.append((parts[0] if parts else window, lifts[0], triage, False))
            continue
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


# A jerk from the rack — dip, drive, split, recovery — takes at least this long;
# anything shorter that a split produces is a slice of the other lift.
MIN_SPLIT_PART = 1.2


def _split_by_rises(
    window: CandidateSegment, tracked: LiftWindow, lifts: List[LiftType]
) -> Tuple[List[CandidateSegment], List[LiftType]]:
    from server.utils.bar_track import rise_episodes

    episodes = [e for e in rise_episodes(tracked.points) if e.credible and e.rise >= 0.10]
    episodes.sort(key=lambda e: e.start_sec)
    if len(episodes) >= 2:
        first, second = episodes[0], episodes[-1]
        cut = (first.end_sec + second.start_sec) / 2
        parts = [
            CandidateSegment(index=window.index, start_sec=window.start_sec, end_sec=cut),
            CandidateSegment(index=window.index, start_sec=cut, end_sec=window.end_sec),
        ]
        if all(part.duration_sec >= MIN_SPLIT_PART for part in parts):
            return parts, lifts
    # One rise, or none we trust: the bar's travel cannot tell a snatch from a
    # clean & jerk done in one motion (both are ~1.5 s floor to overhead), so
    # this is left to the caller's valley split rather than second-guessing
    # the vision model's call.
    return [], lifts




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
    info: VideoInfo,
    frames_wanted: int,
    progress: ProgressReporter,
    progress_span: Tuple[float, float],
    lift_window: Optional[LiftWindow] = None,
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
        models=models_used(),
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
    processed = await _offload(prepare_frames, frames, rotation_deg=_rotation_for(info, frames))

    # Local CV first, before any model call: the tracker's numbers are evidence
    # for the judge as well as the trace on the replay, so the drift the judge
    # describes and the drift the replay draws are the same measurement.
    progress("observing", f"Following the bar — {label}…", low + 0.05 * (high - low))
    bar_path = await _offload(_trace_for, request.video_path, window, lift_window, label)
    measurement = measure(bar_path)
    scored.measurement = measurement
    scored.bar_path_tracked = measurement.tracked

    try:
        progress("observing", f"Reading bar path — {label}…", low + 0.15 * (high - low))
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
            measurement=describe(measurement, lift.value),
        )
        scored.models = models_used(judgment)
    except BudgetExhausted as exc:
        logger.info("Demo budget ran out on %s: %s", label, exc.reason)
        scored.error = exc.detail
        scored.error_detail = detail_of(exc)
        scored.out_of_budget = True
        return scored
    except Exception as exc:
        logger.exception("Scoring failed for %s", label)
        scored.error = user_message(exc)
        scored.error_detail = detail_of(exc)
        return scored

    quality = assess_clip_quality(
        observations=observations,
        info=info,
        frame_count=len(processed),
        duration_sec=window.end_sec - window.start_sec,
        lifts_in_window=len(triage.lifts),
        was_split=was_split,
        measurement=measurement,
    )
    # The re-check: a verdict that contradicts the measured path is flagged.
    quality = cross_check(judgment, measurement, quality)
    scored.judgment = apply_quality(judgment, quality)

    if request.render_replay and request.replay_dir:
        progress("rendering", f"Rendering replay — {label}…", low + 0.8 * (high - low))
        replay = await _offload(
            render_annotated_replay,
            request.video_path,
            scored.judgment,
            str(Path(request.replay_dir) / f"replay_{index:02d}_{lift.value}.mp4"),
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            bar_path=bar_path,
        )
        scored.replay_path = replay
        if replay is None:
            logger.info("Replay rendering unavailable for %s", label)
    return scored


def _trace_for(
    video_path: str, window: CandidateSegment, lift_window: Optional[LiftWindow], label: str
) -> Optional[BarPath]:
    """
    The bar's path through this window, from local CV and no model call. The
    trail that found the lift is used first; a second tracking pass samples
    different frames and does not reliably agree, so it runs only when there is
    no trail or the trail does not cover this part of the clip.
    """
    bar_path: Optional[BarPath] = None
    if lift_window is not None:
        bar_path = lift_window.bar_path(window.start_sec, window.end_sec)
    if bar_path is None or not bar_path.tracked:
        try:
            bar_path = track_bar_path(video_path, start_sec=window.start_sec, end_sec=window.end_sec)
        except Exception:
            logger.exception("Bar tracking failed for %s", label)
    if bar_path is not None and not bar_path.tracked:
        logger.info("No bar trace for %s: %s", label, bar_path.reason)
    return bar_path


async def _lift_windows(
    video_path: str, start_sec: float, end_sec: float, keep: Dict[int, LiftWindow]
) -> List[CandidateSegment]:
    """
    Windows around each rise of the plate, as candidate segments. The LiftWindow
    itself is kept in `keep` by segment index so the trace can be drawn from
    the trail that found the lift.
    """
    found = await _offload(detect_lift_windows, video_path, start_sec=start_sec, end_sec=end_sec)
    segments = []
    for position, w in enumerate(found, start=1):
        keep[position] = w
        segments.append(CandidateSegment(index=position, start_sec=w.start_sec, end_sec=w.end_sec, peak_sec=w.peak_sec))
    return segments


def _rotation_for(info: VideoInfo, frames) -> int:
    """Degrees these decoded frames still need, judged from their actual size."""
    if not frames:
        return 0
    height, width = frames[0].shape[:2]
    return rotation_still_needed(info, width, height)


def _encode_frames(frames, rotation_deg: int, long_side: int) -> List[str]:
    """Rotate, downscale and JPEG-encode in one hop onto the worker thread."""
    return frames_to_data_urls(prepare_frames(frames, rotation_deg=rotation_deg, long_side=long_side))


def _stamp(window: CandidateSegment) -> str:
    from server.utils.clips import format_timestamp

    return f"{format_timestamp(window.start_sec)}–{format_timestamp(window.end_sec)}"
