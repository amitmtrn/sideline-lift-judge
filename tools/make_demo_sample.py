#!/usr/bin/env python3
"""
Score one clip once and cache the result as the public demo's sample.

    python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
    python tools/make_demo_sample.py samples/cj.mp4 --lift clean --trim-start 2 --trim-end 9 --out demo_sample/

Then set DEMO_SAMPLE_DIR=demo_sample on the server. The app serves the cached
verdict and replay from GET /api/sample at no model cost, so a judge who
arrives after the daily budget is spent still sees a full result.

Pick a clip you are allowed to show in public: the replay is served to anyone
who opens the demo. The directory holds video, so it is never committed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from server.schemas.judgment import LiftType  # noqa: E402
from server.services.nebius_client import NebiusChatClient, judge_model  # noqa: E402
from server.services.pipeline import DEFAULT_FRAMES, MAX_FRAMES, MIN_FRAMES  # noqa: E402
from server.services.sample import SAMPLE_FILE, make_sample  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a clip once and cache it as the demo's sample result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to the clip")
    parser.add_argument("--lift", choices=[lift.value for lift in LiftType], default=None, help="Which lift it holds")
    parser.add_argument("--auto", action="store_true", help="Let the pipeline find the lifts instead")
    parser.add_argument("--trim-start", type=float, default=None, help="Window start, seconds")
    parser.add_argument("--trim-end", type=float, default=None, help="Window end, seconds")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help=f"Frames to sample ({MIN_FRAMES}-{MAX_FRAMES})")
    parser.add_argument("--out", type=str, default="demo_sample", help="Directory to write the sample into")
    parser.add_argument("--prompts-dir", type=str, default=str(ROOT / "prompts"), help="Prompts directory")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    if not Path(args.video).is_file():
        print(f"ERROR: no such file: {args.video}", file=sys.stderr)
        return 2
    if not args.lift and not args.auto:
        print("ERROR: pass --lift snatch|clean|jerk, or --auto.", file=sys.stderr)
        return 2

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set.", file=sys.stderr)
        return 2
    client = NebiusChatClient(
        api_key=api_key,
        base_url=os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL,
    )

    def progress(stage: str, message: str, fraction: float) -> None:
        print(f"[{fraction:5.0%}] {message}", file=sys.stderr)

    print(f"Judge: {judge_model()}", file=sys.stderr)
    try:
        outcome = await make_sample(
            video_path=args.video,
            out_dir=Path(args.out),
            client=client,
            prompts_dir=args.prompts_dir,
            lift=LiftType(args.lift) if args.lift else None,
            mode="auto" if args.auto else "single",
            trim_start=args.trim_start,
            trim_end=args.trim_end,
            frames=args.frames,
            progress=progress,
        )
    except Exception as exc:  # the error is the whole story here
        print(f"ERROR: {exc}", file=sys.stderr)
        return 6

    for item in outcome.lifts:
        if item.judgment is None:
            print(f"  {item.lift.value}: not scored ({item.error})")
            continue
        verdict = item.judgment
        replay = "replay rendered" if item.replay_path else "no replay"
        print(f"  {item.lift.value}: {verdict.score}/10 {verdict.band}, confidence {verdict.confidence:.2f} — {replay}")
    print(f"\nWrote {Path(args.out) / SAMPLE_FILE}. Set DEMO_SAMPLE_DIR={args.out} on the server.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
