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
from app.services.nebius_client import NebiusChatClient, assemble_vision_messages, assemble_judge_messages
from app.utils.frames import sample_video_frames_bgr
from app.utils.video_meta import get_rotation_degrees
from app.utils.image_ops import rotate_by_degrees, resize_long_side


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
    parser.add_argument("--frames", type=int, default=10, help="Frames to sample (8–16)")
    parser.add_argument("--prompts_dir", type=str, default=str(Path(__file__).parent / "prompts"), help="Prompts directory")
    parser.add_argument("--debug", action="store_true", help="Print Stage 1 vision observations")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY is not set. See README for setup.", flush=True)
        return 2
    base_url = os.getenv("NEBIUS_BASE_URL") or NebiusChatClient.DEFAULT_BASE_URL
    vision_model = os.getenv("NEBIUS_VISION_MODEL") or "openbmb/MiniCPM-V-4_5"
    judge_model = os.getenv("NEBIUS_JUDGE_MODEL") or "nvidia/Nemotron-3-Ultra-550b-a55b"
    max_images = int(os.getenv("NEBIUS_VISION_MAX_IMAGES") or "10")

    if args.frames < 8 or args.frames > 16:
        print("NOTE: clamping frames to [8, 16]")
    frames = sample_video_frames_bgr(args.video, target_count=max(8, min(16, args.frames)))
    if not frames:
        print("ERROR: Failed to read frames from the video.")
        return 3
    # Handle rotation metadata and downscale
    rot = get_rotation_degrees(args.video)
    processed = [resize_long_side(rotate_by_degrees(f, rot), 768) for f in frames]
    images = bgr_frames_to_data_urls(processed)

    client = NebiusChatClient(api_key=api_key, base_url=base_url)
    # Stage 1: vision observations
    # Clamp or chunk to max_images
    if len(images) <= max_images:
        v_messages = assemble_vision_messages(lift=args.lift, images=images, prompts_dir=args.prompts_dir)
        observations = await client.create_chat_completion(messages=v_messages, model=vision_model)
    else:
        from app.services.nebius_client import collect_observations_in_chunks
        observations = await collect_observations_in_chunks(
            client=client,
            lift=args.lift,
            images=images,
            prompts_dir=args.prompts_dir,
            model=vision_model,
            max_images_per_call=max_images,
        )
    if args.debug:
        print("=== Stage 1: Vision observations ===")
        print(observations)
        print("=== End observations ===")
    # Stage 2: judge
    j_messages = assemble_judge_messages(lift=args.lift, observations=observations, prompts_dir=args.prompts_dir)
    text = await client.create_chat_completion(messages=j_messages, model=judge_model)
    try:
        judgment = parse_and_validate_json(text)
    except (ValueError, ValidationError):
        # Retry once with repair instruction
        j_messages.append(
            {"role": "system", "content": "Your previous response was not valid JSON. Reply again with JSON only matching the schema, no markdown."}
        )
        text2 = await client.create_chat_completion(messages=j_messages, model=judge_model)
        judgment = parse_and_validate_json(text2)

    print(judgment.model_dump_json(indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

