from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

import requests
import math
import re
from pydantic import ValidationError

if TYPE_CHECKING:
    from app.schemas.judgment import LiftJudgment
    from app.schemas.segment import SegmentTriage


class ChatMessage(TypedDict, total=False):
    role: str
    content: Any


# Reasoning models (e.g. Nemotron) otherwise spend the whole token budget thinking
# and never emit the JSON object, so every call in this app asks for no thinking.
NO_THINKING_EXTRA_BODY: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}

# Headroom so the judge still completes its JSON if the provider ignores the
# no-thinking hint and reasons anyway.
DEFAULT_JUDGE_MAX_TOKENS = 2048

DEFAULT_VISION_MODEL = "openbmb/MiniCPM-V-4_5"
DEFAULT_JUDGE_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
DEFAULT_JUDGE_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b"


def vision_model() -> str:
    return os.getenv("NEBIUS_VISION_MODEL") or DEFAULT_VISION_MODEL


def judge_model() -> str:
    return os.getenv("NEBIUS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


def judge_fallback_model() -> str:
    """
    Second judge used when the primary one errors, times out, or never produces
    valid JSON. Set NEBIUS_JUDGE_FALLBACK_MODEL empty to disable the fallback.
    """
    value = os.getenv("NEBIUS_JUDGE_FALLBACK_MODEL")
    return DEFAULT_JUDGE_FALLBACK_MODEL if value is None else value.strip()


class NebiusChatClient:
    """
    Minimal OpenAI-compatible client for Nebius Token Factory.
    """

    # Verified from Nebius Token Factory docs
    DEFAULT_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None, default_model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        # default_model is optional; callers should pass model explicitly for clarity
        self.default_model = default_model

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_chat_completion(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        max_tokens: int = 800,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Call the OpenAI-compatible /chat/completions endpoint. Returns the text content.
        """
        url = f"{self.base_url}/chat/completions"
        base_payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        # First attempt: include extra_body if provided
        payload = dict(base_payload)
        if extra_body:
            payload.update(extra_body)
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=self._headers(), json=payload, timeout=60))
        if response.status_code != 200 and extra_body:
            # Fallback: retry once without extra_body
            response = await loop.run_in_executor(None, lambda: requests.post(url, headers=self._headers(), json=base_payload, timeout=60))
        if response.status_code != 200:
            raise RuntimeError(f"Nebius API error {response.status_code}: {response.text}")
        data = response.json()
        # OpenAI-compatible shape
        try:
            text = data["choices"][0]["message"]["content"]
            return strip_think_blocks(text)
        except Exception:
            raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:500]}")


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _lift_name(lift: Any) -> str:
    """
    Accept either the plain string or a LiftType member; str(LiftType.snatch) is
    "LiftType.snatch", which would corrupt prompt filenames and prompt text.
    """
    name = getattr(lift, "value", lift)
    assert name in ("snatch", "clean", "jerk")
    return name


def assemble_messages_for_lift(*, lift: str, images: List[str], prompts_dir: str) -> List[ChatMessage]:
    """
    Compose system and user prompts with image parts per OpenAI-compatible spec.
    Each image is passed as a data URL in image_url.type.
    """
    lift = _lift_name(lift)
    system_path = f"{prompts_dir}/system.txt"
    user_path = f"{prompts_dir}/user_{lift}.txt"
    system_prompt = _load_prompt(system_path)
    user_prompt = _load_prompt(user_path)

    image_items = [{"type": "image_url", "image_url": {"url": url}} for url in images]
    content = [{"type": "text", "text": user_prompt}] + image_items
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def assemble_vision_messages(*, lift: str, images: List[str], prompts_dir: str) -> List[ChatMessage]:
    """
    Vision (Stage 1): build messages for MiniCPM-V to produce factual observations only.
    """
    lift = _lift_name(lift)
    observation_prompt = _load_prompt(f"{prompts_dir}/vision_observation.txt")
    # Include lift type in the first text block to guide phase breakdown
    content = [{"type": "text", "text": f"Lift: {lift}\n\n{observation_prompt}"}]
    content += [{"type": "image_url", "image_url": {"url": url}} for url in images]
    return [
        {"role": "system", "content": "You strictly output factual observations about the images. No scores, no coaching, no speculation beyond what is visible."},
        {"role": "user", "content": content},
    ]


def assemble_triage_messages(*, images: List[str], prompts_dir: str) -> List[ChatMessage]:
    """
    Triage (Stage 0b): ask the vision model whether a candidate window from a
    session clip holds a scored lift, and which one.
    """
    triage_prompt = _load_prompt(f"{prompts_dir}/vision_triage.txt")
    content: List[Dict[str, Any]] = [{"type": "text", "text": triage_prompt}]
    # Label every frame: asked for one answer per numbered frame, the model reads
    # the sequence instead of naming the lift the whole clip reminds it of.
    for position, url in enumerate(images, start=1):
        content.append({"type": "text", "text": f"Frame {position}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {
            "role": "system",
            "content": "You classify what is visible in gym video frames and reply with JSON only. No prose, no coaching, no deliberation.",
        },
        {"role": "user", "content": content},
    ]


async def classify_segment(
    *,
    client: "NebiusChatClient",
    images: List[str],
    prompts_dir: str,
    model: str,
    max_tokens: int = 400,
) -> "SegmentTriage":
    """
    Triage one candidate window. One repair retry when the reply is not JSON;
    a second failure returns an empty triage so one bad window cannot abort a
    whole session run.
    """
    from app.schemas.segment import SegmentTriage, parse_triage_json

    messages = assemble_triage_messages(images=images, prompts_dir=prompts_dir)
    first = await client.create_chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        extra_body=NO_THINKING_EXTRA_BODY,
    )
    try:
        return parse_triage_json(first)
    except (ValueError, ValidationError):
        pass

    retry_messages = messages + [
        {
            "role": "system",
            "content": "Your previous response was not valid JSON. Reply again with JSON only matching the schema, no markdown, no reasoning.",
        }
    ]
    second = await client.create_chat_completion(
        messages=retry_messages,
        model=model,
        max_tokens=max_tokens,
        extra_body=NO_THINKING_EXTRA_BODY,
    )
    try:
        return parse_triage_json(second)
    except (ValueError, ValidationError) as exc:
        return SegmentTriage(lifts=[], activity="unclassified", notes=f"triage failed: {exc}")


def assemble_judge_messages(*, lift: str, observations: str, prompts_dir: str) -> List[ChatMessage]:
    """
    Judge (Stage 2): use rubric + per-lift prompt; evidence is the Stage-1 observations (no images).
    Instruct to lower confidence when observations mention occlusion or oblique camera angle.
    """
    lift = _lift_name(lift)
    system_prompt = _load_prompt(f"{prompts_dir}/system.txt")
    user_prompt = _load_prompt(f"{prompts_dir}/user_{lift}.txt")
    guidance = (
        "Evidence is a vision-model observation summary (not raw images). "
        "Judge bar-path efficiency ONLY from those observations. "
        "If the observations mention occlusion, darkness, or an oblique camera angle, reflect higher uncertainty and lower confidence."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": guidance},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": f"Vision observations:\n{observations}"},
    ]


async def _judge_with_model(
    *,
    client: "NebiusChatClient",
    messages: List[ChatMessage],
    model: str,
    max_tokens: int,
) -> "LiftJudgment":
    """
    One judge model, with a single repair retry when the reply is not valid JSON.
    """
    from app.schemas.judgment import parse_and_validate_json

    first = await client.create_chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        extra_body=NO_THINKING_EXTRA_BODY,
    )
    try:
        return parse_and_validate_json(first)
    except (ValueError, ValidationError):
        retry_messages = messages + [
            {
                "role": "system",
                "content": "Your previous response was not valid JSON. Reply again with JSON only matching the schema, no markdown, no reasoning.",
            }
        ]
        second = await client.create_chat_completion(
            messages=retry_messages,
            model=model,
            max_tokens=max_tokens,
            extra_body=NO_THINKING_EXTRA_BODY,
        )
        return parse_and_validate_json(second)  # Let errors bubble


async def request_judgment(
    *,
    client: "NebiusChatClient",
    lift: str,
    observations: str,
    prompts_dir: str,
    model: str,
    max_tokens: Optional[int] = None,
    fallback_model: Optional[str] = None,
) -> "LiftJudgment":
    """
    Judge (Stage 2) with one repair retry on invalid JSON, then one retry on a
    second judge model when the primary fails outright — an API error, a timeout,
    or no valid JSON even after the repair. Pass fallback_model="" to disable.
    Raises the fallback's error if that one fails too.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("NEBIUS_JUDGE_MAX_TOKENS") or DEFAULT_JUDGE_MAX_TOKENS)
    if fallback_model is None:
        fallback_model = judge_fallback_model()
    messages = assemble_judge_messages(lift=lift, observations=observations, prompts_dir=prompts_dir)

    try:
        return await _judge_with_model(client=client, messages=messages, model=model, max_tokens=max_tokens)
    except Exception:
        # No usable second model: the caller sees the primary failure as before.
        if not fallback_model or fallback_model == model:
            raise
    return await _judge_with_model(
        client=client,
        messages=messages,
        model=fallback_model,
        max_tokens=max_tokens,
    )


DEFAULT_VISION_MAX_TOKENS = 1500


def vision_max_images() -> int:
    """Images per vision call; the provider rejects more than this in one request."""
    try:
        value = int(os.getenv("NEBIUS_VISION_MAX_IMAGES") or "10")
    except ValueError:
        return 10
    return value if value > 0 else 10


async def collect_observations(
    *,
    client: "NebiusChatClient",
    lift: str,
    images: List[str],
    prompts_dir: str,
    model: str,
    max_images_per_call: Optional[int] = None,
    max_tokens: int = DEFAULT_VISION_MAX_TOKENS,
) -> str:
    """
    Vision (Stage 1) for one lift: a single call when the frames fit the
    per-request image limit, otherwise chunked calls merged into one summary.
    """
    limit = max_images_per_call if max_images_per_call and max_images_per_call > 0 else vision_max_images()
    if len(images) <= limit:
        messages = assemble_vision_messages(lift=lift, images=images, prompts_dir=prompts_dir)
        return await client.create_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            extra_body=NO_THINKING_EXTRA_BODY,
        )
    return await collect_observations_in_chunks(
        client=client,
        lift=lift,
        images=images,
        prompts_dir=prompts_dir,
        model=model,
        max_images_per_call=limit,
    )


async def collect_observations_in_chunks(
    *,
    client: "NebiusChatClient",
    lift: str,
    images: List[str],
    prompts_dir: str,
    model: str,
    max_images_per_call: int,
) -> str:
    """
    Call the vision model in chunks of <= max_images_per_call and concatenate observations.
    """
    if max_images_per_call <= 0:
        max_images_per_call = 10
    chunks: List[List[str]] = [images[i : i + max_images_per_call] for i in range(0, len(images), max_images_per_call)]
    observations_parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        messages = assemble_vision_messages(lift=lift, images=chunk, prompts_dir=prompts_dir)
        text = await client.create_chat_completion(
            messages=messages,
            model=model,
            max_tokens=DEFAULT_VISION_MAX_TOKENS,
            extra_body=NO_THINKING_EXTRA_BODY,
        )
        observations_parts.append(f"[Chunk {idx}/{len(chunks)}]\n{text}")
    return "\n\n".join(observations_parts)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_PREFIX_RE = re.compile(r"^\s*<think>.*$", re.DOTALL | re.IGNORECASE)


def strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks or an initial unclosed <think> prefix.
    """
    if not isinstance(text, str):
        return text
    # Remove all closed think blocks
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # If starts with an unclosed <think>, drop everything up to the first closing tag or end
    if cleaned.lstrip().lower().startswith("<think>") and "</think>" not in cleaned.lower():
        return ""
    return cleaned.strip()

