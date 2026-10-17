#!/usr/bin/env python3
"""
Score a whole training clip: split it into exercises, then rank each one.

Stage 0a  motion scan (local, OpenCV)   -> candidate windows
Stage 0b  triage (vision model)         -> is this a lift, and which one
Stage 1   observations (vision model)   -> factual bar-path description per lift
Stage 2   judge (Nemotron)              -> validated 1-10 JSON per lift
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from app.schemas.judgment import LiftType
from app.schemas.segment import ExerciseReport, SessionReport, SegmentTriage, SkippedSegment
from app.services.nebius_client import (
    NebiusChatClient,
    classify_segment,
    collect_observations,
    judge_model,
    request_judgment,
    vision_max_images,
    vision_model,
)
from app.utils.clips import cut_segment, format_timestamp, has_ffmpeg, write_contact_sheet
from app.utils.frames import sample_frames_in_range_bgr
from app.utils.image_ops import frames_to_data_urls, prepare_frames
from app.utils.video_meta import get_rotation_degrees
from app.utils.segmentation import (
    DEFAULT_MAX_DURATION,
    DEFAULT_MERGE_GAP,
    DEFAULT_MIN_DURATION,
    DEFAULT_PAD,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_SENSITIVITY,
    CandidateSegment,
    MotionProfile,
    compute_motion_profile,
    detect_segments,
    split_segment_at_valley,
)

# Triage reads the bar's height frame by frame, so it needs the same resolution
# as Stage 1: at 512px the model mistook a front rack for a lockout overhead.
TRIAGE_FRAMES = 8
TRIAGE_LONG_SIDE = 768


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a training clip into exercises and score each one's bar path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to the session video")
    parser.add_argument(
        "--lift",
        type=str,
        choices=[e.value for e in LiftType],
        default=None,
        help="Force every segment to this lift and skip vision triage",
    )
    parser.add_argument("--frames", type=int, default=10, help="Frames per exercise sent to the judge pipeline (8-16)")
    parser.add_argument("--out-dir", type=str, default=None, help="Where to write report.json, cut clips and frames")
    parser.add_argument("--prompts-dir", type=str, default=str(Path(__file__).parent / "prompts"))
    parser.add_argument("--concurrency", type=int, default=2, help="Exercises scored in parallel")
    parser.add_argument("--max-segments", type=int, default=None, help="Keep only the N busiest candidate windows")

    splitting = parser.add_argument_group("splitting")
    splitting.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY, help="0-1; lower splits more eagerly")
    splitting.add_argument(
        "--min-duration",
        type=float,
        default=DEFAULT_MIN_DURATION,
        help="Drop a window whose high-motion core is shorter than this (s)",
    )
    splitting.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION, help="Trim windows longer than this (s)")
    splitting.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP, help="Join windows closer than this (s)")
    splitting.add_argument("--pad", type=float, default=DEFAULT_PAD, help="Padding added to each side of a window (s)")
    splitting.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS, help="Motion-scan sampling rate")

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--no-clips", action="store_true", help="Do not cut per-exercise mp4 files")
    outputs.add_argument("--no-frames", action="store_true", help="Do not save the sampled frames per exercise")
    outputs.add_argument("--json", type=str, default=None, help="Also write the report to this path")
    outputs.add_argument(
        "--split-only",
        action="store_true",
        help="Run the motion scan and print the windows without calling any model",
    )
    outputs.add_argument("--debug", action="store_true", help="Print triage replies and Stage-1 observations")
    return parser.parse_args(argv)


def expand_for_lifts(
    segment: CandidateSegment,
    triage: SegmentTriage,
    profile: MotionProfile,
) -> List[Tuple[CandidateSegment, LiftType]]:
    """
    One window to one (window, lift) pair per scored lift.

    A clean & jerk arrives as a single window holding two lifts; the athlete is
    briefly still between them, so the window splits at its quietest interior
    moment and each half is judged on its own.
    """
    lifts = triage.lifts
    if len(lifts) <= 1:
        return [(segment, lift) for lift in lifts]

    parts: List[CandidateSegment] = []
    if len(lifts) == 2:
        split = split_segment_at_valley(profile, segment)
        if split is not None:
            parts = list(split)
    if len(parts) != len(lifts):
        # No usable valley: judge the same window once per lift.
        parts = [segment] * len(lifts)
    return list(zip(parts, lifts))


async def score_exercise(
    *,
    client: NebiusChatClient,
    video_path: str,
    rotation_deg: int,
    segment: CandidateSegment,
    lift: LiftType,
    index: int,
    args: argparse.Namespace,
    out_dir: Optional[Path],
    triage: SegmentTriage,
) -> ExerciseReport:
    report = ExerciseReport(
        index=index,
        segment_index=segment.index,
        start_sec=round(segment.start_sec, 2),
        end_sec=round(segment.end_sec, 2),
        lift=lift,
        activity=triage.activity,
        triage_confidence=triage.confidence,
    )
    frames = sample_frames_in_range_bgr(
        video_path,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        target_count=max(8, min(16, args.frames)),
    )
    if not frames:
        report.error = "no readable frames in this window"
        return report
    processed = prepare_frames(frames, rotation_deg=rotation_deg)

    if out_dir is not None and not args.no_frames:
        report.frames_dir = write_contact_sheet(processed, str(out_dir / f"exercise_{index:02d}_frames"))
    if out_dir is not None and not args.no_clips:
        report.clip_path = cut_segment(
            video_path,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            out_path=str(out_dir / f"exercise_{index:02d}_{lift.value}.mp4"),
        )

    try:
        observations = await collect_observations(
            client=client,
            lift=lift.value,
            images=frames_to_data_urls(processed),
            prompts_dir=args.prompts_dir,
            model=vision_model(),
        )
        if args.debug:
            report.observations = observations
            print(f"--- exercise {index} ({lift.value}) observations ---\n{observations}\n", file=sys.stderr)
        # request_judgment already repairs invalid JSON once and falls back to
        # the secondary judge model before giving up.
        report.judgment = await request_judgment(
            client=client,
            lift=lift.value,
            observations=observations,
            prompts_dir=args.prompts_dir,
            model=judge_model(),
        )
    except Exception as exc:  # one failed exercise must not sink the session
        report.error = f"{type(exc).__name__}: {exc}"
    return report


async def triage_segments(
    *,
    client: NebiusChatClient,
    video_path: str,
    rotation_deg: int,
    segments: List[CandidateSegment],
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> List[SegmentTriage]:
    forced = LiftType(args.lift) if args.lift else None

    async def run(segment: CandidateSegment) -> SegmentTriage:
        if forced is not None:
            return SegmentTriage(lifts=[forced], activity=f"forced {forced.value}", confidence=None)
        frames = sample_frames_in_range_bgr(
            video_path,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            target_count=min(TRIAGE_FRAMES, vision_max_images()),
        )
        if not frames:
            return SegmentTriage(lifts=[], activity="unreadable window")
        async with semaphore:
            triage = await classify_segment(
                client=client,
                images=frames_to_data_urls(
                    prepare_frames(frames, rotation_deg=rotation_deg, long_side=TRIAGE_LONG_SIDE)
                ),
                prompts_dir=args.prompts_dir,
                model=vision_model(),
            )
        if args.debug:
            print(
                f"--- window {segment.index} {format_timestamp(segment.start_sec)}-{format_timestamp(segment.end_sec)}: "
                f"{triage.lifts or 'no lift'} ({triage.activity})",
                file=sys.stderr,
            )
        return triage

    return list(await asyncio.gather(*(run(segment) for segment in segments)))


def print_segment_table(segments: List[CandidateSegment]) -> None:
    print(f"{'#':>3}  {'start':>7}  {'end':>7}  {'dur':>5}  {'peak':>6}")
    for segment in segments:
        print(
            f"{segment.index:>3}  {format_timestamp(segment.start_sec):>7}  {format_timestamp(segment.end_sec):>7}  "
            f"{segment.duration_sec:>5.1f}  {segment.peak_energy:>6.2f}"
        )


def print_ranking(report: SessionReport, stream=sys.stdout) -> None:
    ranked = report.ranked()
    write = partial(print, file=stream)
    write()
    write(f"Session: {report.source}  ({format_timestamp(report.duration_sec)})")
    write(f"Candidate windows: {report.candidate_segments}  scored: {len(ranked)}  skipped: {len(report.skipped)}")
    write()
    if ranked:
        # "ex#" is the exercise_NN_ prefix of the files written for it.
        write(f"{'rank':>4}  {'ex#':>3}  {'window':>15}  {'lift':<7}  {'score':>5}  {'band':<7}  {'conf':>4}  cues")
        for rank, item in enumerate(ranked, start=1):
            window = f"{format_timestamp(item.start_sec)}-{format_timestamp(item.end_sec)}"
            judgment = item.judgment
            score = "-" if not judgment or judgment.score is None else str(judgment.score)
            band = judgment.band if judgment else "error"
            confidence = f"{judgment.confidence:.2f}" if judgment and judgment.confidence is not None else "-"
            # Cues contain semicolons of their own, so separate them with a pipe.
            cues = " | ".join(judgment.cues) if judgment and judgment.cues else (item.error or "")
            write(
                f"{rank:>4}  {item.index:>3}  {window:>15}  {item.lift.value:<7}  {score:>5}  {band:<7}  "
                f"{confidence:>4}  {cues}"
            )
    if report.skipped:
        write()
        write("Skipped windows (not a scored lift):")
        for item in report.skipped:
            write(
                f"  window {item.index}  {format_timestamp(item.start_sec)}-{format_timestamp(item.end_sec)}  "
                f"{item.activity or item.reason}"
            )


async def run(args: argparse.Namespace) -> int:
    video_path = args.video
    if not Path(video_path).is_file():
        print(f"ERROR: no such file: {video_path}", file=sys.stderr)
        return 2

    # Stage 0a: motion scan (local, no API cost)
    profile = compute_motion_profile(video_path, sample_fps=args.sample_fps)
    if not profile and profile.duration_sec <= 0:
        print("ERROR: could not decode the video.", file=sys.stderr)
        return 3
    segments = detect_segments(
        profile,
        sensitivity=args.sensitivity,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        merge_gap=args.merge_gap,
        pad=args.pad,
        max_segments=args.max_segments,
    )
    if not segments:
        print("No exercise windows found. Try a lower --sensitivity or --min-duration.", file=sys.stderr)
        return 4
    print(f"Motion scan: {len(segments)} candidate window(s) in {format_timestamp(profile.duration_sec)}", file=sys.stderr)

    if args.split_only:
        print_segment_table(segments)
        return 0

    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set. See README for setup.", file=sys.stderr)
        return 2
    client = NebiusChatClient(api_key=api_key, base_url=os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL)

    out_dir: Optional[Path] = None
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_clips and not has_ffmpeg():
            print("NOTE: ffmpeg not found; skipping per-exercise clip cutting.", file=sys.stderr)

    rotation_deg = get_rotation_degrees(video_path)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    # Stage 0b: triage each window, then expand clean & jerk into two exercises
    triages = await triage_segments(
        client=client,
        video_path=video_path,
        rotation_deg=rotation_deg,
        segments=segments,
        args=args,
        semaphore=semaphore,
    )

    report = SessionReport(
        source=str(video_path),
        duration_sec=round(profile.duration_sec, 2),
        candidate_segments=len(segments),
    )
    scored: List[Tuple[CandidateSegment, LiftType, SegmentTriage]] = []
    for segment, triage in zip(segments, triages):
        if not triage.is_lift:
            report.skipped.append(
                SkippedSegment(
                    index=segment.index,
                    start_sec=round(segment.start_sec, 2),
                    end_sec=round(segment.end_sec, 2),
                    activity=triage.activity,
                    reason=triage.notes or "no snatch, clean or jerk detected",
                )
            )
            continue
        for part, lift in expand_for_lifts(segment, triage, profile):
            scored.append((part, lift, triage))

    if not scored:
        print("No snatch, clean or jerk found in this clip.", file=sys.stderr)
        print_ranking(report)
        return 5
    print(f"Triage: scoring {len(scored)} exercise(s), skipping {len(report.skipped)} window(s)", file=sys.stderr)

    # Stages 1 + 2: observe and judge each exercise
    async def guarded(index: int, part: CandidateSegment, lift: LiftType, triage: SegmentTriage) -> ExerciseReport:
        async with semaphore:
            return await score_exercise(
                client=client,
                video_path=video_path,
                rotation_deg=rotation_deg,
                segment=part,
                lift=lift,
                index=index,
                args=args,
                out_dir=out_dir,
                triage=triage,
            )

    report.exercises = list(
        await asyncio.gather(
            *(guarded(index, part, lift, triage) for index, (part, lift, triage) in enumerate(scored, start=1))
        )
    )

    payload = json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
    json_path = Path(args.json) if args.json else (out_dir / "report.json" if out_dir else None)
    if json_path is not None:
        # The report has a home, so stdout is the human-readable table.
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {json_path}", file=sys.stderr)
        print_ranking(report, sys.stdout)
    else:
        # Nowhere to put the report: keep stdout pipeable (as score_clip.py does)
        # and send the table to stderr.
        print(payload)
        print_ranking(report, sys.stderr)
    return 0 if any(item.judgment for item in report.exercises) else 6


def main(argv: Optional[List[str]] = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
