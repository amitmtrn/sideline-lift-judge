#!/usr/bin/env python3
import argparse
import asyncio
import base64
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import ValidationError

from app.schemas.judgment import parse_and_validate_json, LiftType
from app.services.nebius_client import NebiusChatClient, assemble_messages_for_lift
from app.utils.frames import sample_video_frames_bgr


def bgr_frames_to_data_urls(frames_bgr: List["np.ndarray"]) -> List[str]:
    import cv2  # type: ignore
    data_urls: List[str] = []
    for frame in frames_bgr:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        data_urls.append(f"data:image/jpeg;base64,{b64}")
    return data_urls


async def main() -> int:
    parser = argparse.ArgumentParser(description="Score a lift clip with Nebius Token Factory (NVIDIA vision model)")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--lift", type=str, choices=[e.value for e in LiftType], required=True, help="Lift type")
    parser.add_argument("--frames", type=int, default=12, help="Frames to sample (8–16)")
    parser.add_argument("--prompts_dir", type=str, default=str(Path(__file__).parent / "prompts"), help="Prompts directory")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set. See README for setup.", flush=True)
        return 2
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    model = os.getenv("NEBIUS_MODEL") or NebiusChatClient.DEFAULT_MODEL

    if args.frames < 8 or args.frames > 16:
        print("NOTE: clamping frames to [8, 16]")
    frames = sample_video_frames_bgr(args.video, target_count=max(8, min(16, args.frames)))
    if not frames:
        print("ERROR: Failed to read frames from the video.")
        return 3
    images = bgr_frames_to_data_urls(frames)

    client = NebiusChatClient(api_key=api_key, base_url=base_url, default_model=model)
    messages = assemble_messages_for_lift(lift=args.lift, images=images, prompts_dir=args.prompts_dir)
    # First attempt
    text = await client.create_chat_completion(messages=messages, model=None)
    try:
        judgment = parse_and_validate_json(text)
    except (ValueError, ValidationError):
        # Retry once with repair instruction
        messages.append(
            {"role": "system", "content": "Your previous response was not valid JSON. Reply again with JSON only matching the schema, no markdown."}
        )
        text2 = await client.create_chat_completion(messages=messages, model=None)
        judgment = parse_and_validate_json(text2)

    print(judgment.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

