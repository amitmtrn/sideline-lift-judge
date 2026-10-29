#!/usr/bin/env python3
"""
Score one clip once and cache the result as the public demo's sample.

Takes a local file or a URL (Google Drive "anyone-with-link" is handled):

    python tools/make_demo_sample.py samples/snatch1.mp4 --lift snatch --out demo_sample/
    python tools/make_demo_sample.py "https://drive.google.com/uc?export=download&id=..." --lift clean --out demo_sample/

The resulting `sample.json` carries the verdict **and** the per-frame
bar-path points, so the dyno re-renders the annotated replay from the
downloaded clip at /api/sample time (no model call, no video commit). Only
`sample.json` and the evidence stills are expected to be committed; the
replay MP4 the pipeline rendered locally is not part of the shippable
bundle (videos are not committed — spec §12).

Deploy flow:
    1. Build the bundle locally with this tool.
    2. Commit `demo_sample/sample.json` and `demo_sample/*.jpg` (NOT the mp4).
    3. `heroku config:set SAMPLE_CLIP_URL=<drive URL> DEMO_SAMPLE_DIR=demo_sample -a <app>`.
    4. Deploy.
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
from server.utils.drive_download import download_drive_file  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a clip once and cache it as the demo's sample result.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=str, help="Path to the clip, or a Google Drive URL")
    parser.add_argument("--lift", choices=[lift.value for lift in LiftType], default=None, help="Which lift it holds")
    parser.add_argument("--auto", action="store_true", help="Let the pipeline find the lifts instead")
    parser.add_argument("--trim-start", type=float, default=None, help="Window start, seconds")
    parser.add_argument("--trim-end", type=float, default=None, help="Window end, seconds")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES, help=f"Frames to sample ({MIN_FRAMES}-{MAX_FRAMES})")
    parser.add_argument("--out", type=str, default="demo_sample", help="Directory to write the sample into")
    parser.add_argument("--prompts-dir", type=str, default=str(ROOT / "prompts"), help="Prompts directory")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    video_source = args.video
    if video_source.startswith("http://") or video_source.startswith("https://"):
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        local_clip = out_dir / "_source_download.mp4"
        print(f"Downloading {video_source} -> {local_clip}", file=sys.stderr)
        try:
            download_drive_file(video_source, local_clip)
        except Exception as exc:
            print(f"ERROR: could not download clip: {exc}", file=sys.stderr)
            return 2
        video_source = str(local_clip)
    if not Path(video_source).is_file():
        print(f"ERROR: no such file: {video_source}", file=sys.stderr)
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
            video_path=video_source,
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
