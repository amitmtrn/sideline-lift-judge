#!/usr/bin/env python3
"""
Score one clip from the command line.

Runs exactly the pipeline the web app runs (FR-13), so a result here and a
result on the phone cannot drift apart: triage, Stage-1 observations, the
Nemotron judge with its fallback, and the clip-quality confidence rule. A clip
holding a clean & jerk is split and scored as two lifts.

    python score_clip.py clip.mp4 --lift snatch
    python score_clip.py clip.mp4 --lift clean --trim-start 2.5 --trim-end 7 --debug
    python score_clip.py session.mp4 --auto --replay-dir out/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from server.schemas.judgment import LiftType
from server.services.nebius_client import NebiusChatClient
from server.services.pipeline import (
    DEFAULT_FRAMES,
    MAX_FRAMES,
    MIN_FRAMES,
    ScoreOutcome,
    ScoreRequest,
    TrimError,
    clamp_frames,
    run_scoring,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a lift clip with Nebius Token Factory (NVIDIA Nemotron judge).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument(
        "--lift",
        type=str,
        choices=[lift.value for lift in LiftType],
        default=None,
        help="Which lift this clip holds. Omit with --auto to let the vision model decide",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Scan a longer clip and score every attempt found in it",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help=f"Frames to sample ({MIN_FRAMES}-{MAX_FRAMES})")
    parser.add_argument("--trim-start", type=float, default=None, help="Start of the window to judge, in seconds")
    parser.add_argument("--trim-end", type=float, default=None, help="End of the window to judge, in seconds")
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip the triage call that separates a clean & jerk into two lifts",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default=None,
        help="Render the annotated replay MP4 into this directory",
    )
    parser.add_argument(
        "--prompts-dir",
        "--prompts_dir",
        dest="prompts_dir",
        type=str,
        default=str(Path(__file__).parent / "prompts"),
        help="Prompts directory",
    )
    parser.add_argument("--debug", action="store_true", help="Print Stage-1 vision observations to stderr")
    parser.add_argument("--progress", action="store_true", help="Print pipeline stages to stderr as they run")
    return parser.parse_args(argv)


def build_payload(outcome: ScoreOutcome) -> object:
    """
    One lift prints the verdict object, as this CLI always has. Several lifts —
    a clean & jerk, or an --auto session — print the whole run.
    """
    def one(item) -> dict:
        body = item.judgment.model_dump() if item.judgment else {"error": item.error}
        body.update(
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "was_split": item.was_split,
            }
        )
        if item.observations:
            body["observations"] = item.observations
        if item.replay_path:
            body["annotated_video_url"] = item.replay_path
        return body

    if len(outcome.lifts) == 1:
        return one(outcome.lifts[0])
    return {
        "lifts": [one(item) for item in outcome.lifts],
        "split_detected": outcome.split_detected,
        "skipped": outcome.skipped,
        "duration_sec": round(outcome.duration_sec, 2),
    }


async def run(args: argparse.Namespace) -> int:
    if not Path(args.video).is_file():
        print(f"ERROR: no such file: {args.video}", file=sys.stderr)
        return 2
    if not args.lift and not args.auto:
        print("ERROR: pass --lift snatch|clean|jerk, or --auto to detect the lifts.", file=sys.stderr)
        return 2
    if args.frames < MIN_FRAMES or args.frames > MAX_FRAMES:
        print(f"NOTE: clamping frames to [{MIN_FRAMES}, {MAX_FRAMES}]", file=sys.stderr)

    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set. See README for setup.", file=sys.stderr)
        return 2
    client = NebiusChatClient(
        api_key=api_key,
        base_url=os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL,
    )

    if args.replay_dir:
        Path(args.replay_dir).mkdir(parents=True, exist_ok=True)

    def progress(stage: str, message: str, fraction: float) -> None:
        if args.progress:
            print(f"[{fraction:5.0%}] {message}", file=sys.stderr)

    request = ScoreRequest(
        video_path=args.video,
        lift=LiftType(args.lift) if args.lift else None,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        frames=clamp_frames(args.frames),
        mode="auto" if args.auto else "single",
        render_replay=bool(args.replay_dir),
        replay_dir=args.replay_dir,
        keep_observations=bool(args.debug),
        detect_split=not args.no_split,
    )

    try:
        outcome = await run_scoring(request, client=client, prompts_dir=args.prompts_dir)
    except TrimError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.debug:
        for item in outcome.lifts:
            if item.observations:
                print(f"=== Stage 1 observations — {item.lift.value} ===", file=sys.stderr)
                print(item.observations, file=sys.stderr)
        print("=== End observations ===", file=sys.stderr)

    if not outcome.lifts:
        print("No snatch, clean or jerk found in this clip.", file=sys.stderr)
        return 5

    print(json.dumps(build_payload(outcome), indent=2, ensure_ascii=False))
    for item in outcome.lifts:
        if item.error:
            print(f"WARNING: {item.lift.value}: {item.error}", file=sys.stderr)
    return 0 if outcome.scored else 6


def main(argv: Optional[List[str]] = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
