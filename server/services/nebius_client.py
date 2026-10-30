from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

import requests
import math
import re
from pydantic import ValidationError

if TYPE_CHECKING:
    from server.schemas.judgment import LiftJudgment
    from server.schemas.segment import SegmentTriage


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

# A healthy call answers in a few seconds (2-3 s measured for both stages), so
# a hung one should not hold a phone for ever: connect fast, read within
# NEBIUS_TIMEOUT_SECONDS. The judge has its fallback model for what fails.
#
# Previous default was 45 s; a live jerk job hit it on the primary and the
# fallback both, so judges saw "The model service took too long to answer".
# 90 s leaves headroom for the primary to actually finish, while still fitting
# inside the default JOB_TIMEOUT_SECONDS (120) with room for other stages.
CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 90.0
# Transient provider trouble (and a provider-side timeout the client could not
# detect in time) gets exactly one more try after a short pause.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_BACKOFF_SECONDS = 1.5
# Status codes that mean "I do not know that parameter": the one case where
# dropping the no-thinking hint and asking again makes sense.
UNKNOWN_PARAMETER_STATUS = frozenset({400, 422})


def read_timeout_seconds() -> float:
    try:
        value = float(os.getenv("NEBIUS_TIMEOUT_SECONDS") or DEFAULT_READ_TIMEOUT_SECONDS)
    except ValueError:
        return DEFAULT_READ_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_READ_TIMEOUT_SECONDS


class NebiusAPIError(RuntimeError):
    """A non-200 reply from Token Factory. `status` lets callers tell busy from broken."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Nebius API error {status}: {body}")
        self.status = status


def vision_model() -> str:
    return os.getenv("NEBIUS_VISION_MODEL") or DEFAULT_VISION_MODEL


def judge_model() -> str:
    return os.getenv("NEBIUS_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL


_NEMOTRON_TIER_RE = re.compile(r"^nvidia/nemotron-3-([a-z]+)", re.IGNORECASE)
_MINICPM_RE = re.compile(r"^openbmb/minicpm-v-(\d+)[_.](\d+)", re.IGNORECASE)


def model_display_name(model: str) -> str:
    """
    A model id as the page names it. The result card says which model did
    what, and the public demo may run a cheaper judge than the default, so
    the names come from what is actually configured, never from a constant.
    """
    match = _NEMOTRON_TIER_RE.match(model)
    if match:
        return f"NVIDIA Nemotron 3 {match.group(1).capitalize()}"
    match = _MINICPM_RE.match(model)
    if match:
        return f"MiniCPM-V {match.group(1)}.{match.group(2)}"
    return model


def judge_display_name(model: Optional[str] = None) -> str:
    """What the footer says the page is judged by."""
    return model_display_name(model or judge_model())


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

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        *,
        meter: Optional[Callable[[], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        # default_model is optional; callers should pass model explicitly for clarity
        self.default_model = default_model
        # Called before every HTTP attempt — retries included, since each one
        # is billed. The public demo's budget raises here to stop the call.
        self.meter = meter

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
        # First attempt: include extra_body if provided
        payload = dict(base_payload)
        if extra_body:
            payload.update(extra_body)
        timeout = (CONNECT_TIMEOUT_SECONDS, read_timeout_seconds())

        def post(body: Dict[str, Any]):
            return requests.post(url, headers=self._headers(), json=body, timeout=timeout)

        async def attempt(body: Dict[str, Any]):
            if self.meter is not None:
                self.meter()
            # The request blocks, so it runs on a worker thread. A timeout or
            # a dropped connection is returned as a (None, exc) sentinel so
            # the retry loop below treats it like a 5xx — Token Factory does
            # on-prem queueing that occasionally runs past our read timeout.
            try:
                return await asyncio.to_thread(post, body), None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                return None, exc

        response, timeout_exc = await attempt(payload)
        if response is not None and response.status_code in UNKNOWN_PARAMETER_STATUS and extra_body:
            # The provider rejected a parameter it does not know: ask again
            # without the no-thinking hint.
            payload = base_payload
            response, timeout_exc = await attempt(payload)
        retryable = timeout_exc is not None or (
            response is not None and response.status_code in RETRYABLE_STATUS
        )
        if retryable:
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
            response, timeout_exc = await attempt(payload)
        if timeout_exc is not None:
            # One retry wasn't enough; let the caller fall back to the
            # second judge model or surface a partial result.
            raise timeout_exc
        if response.status_code != 200:
            raise NebiusAPIError(response.status_code, str(response.text)[:500])
        data = response.json()
        # OpenAI-compatible shape
        try:
            text = data["choices"][0]["message"]["content"]
            return strip_think_blocks(text)
        except Exception:
            raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:500]}")

    async def create_tool_completion(
        self,
        messages: List[ChatMessage],
        *,
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 800,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """
        One turn of OpenAI-style function calling: the whole assistant
        message comes back (`content` and any `tool_calls`), not just the
        text, so the caller can run the tools and continue. Same metering,
        timeouts and single retry as create_chat_completion; thinking off.
        """
        url = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        payload.update(NO_THINKING_EXTRA_BODY)
        timeout = (CONNECT_TIMEOUT_SECONDS, read_timeout_seconds())

        def post(body: Dict[str, Any]):
            return requests.post(url, headers=self._headers(), json=body, timeout=timeout)

        async def attempt(body: Dict[str, Any]):
            if self.meter is not None:
                self.meter()
            try:
                return await asyncio.to_thread(post, body), None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                return None, exc

        response, timeout_exc = await attempt(payload)
        if response is not None and response.status_code in UNKNOWN_PARAMETER_STATUS:
            # The provider may not know the no-thinking hint: ask again without it.
            payload = {key: value for key, value in payload.items() if key != "chat_template_kwargs"}
            response, timeout_exc = await attempt(payload)
        if timeout_exc is not None or (response is not None and response.status_code in RETRYABLE_STATUS):
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
            response, timeout_exc = await attempt(payload)
        if timeout_exc is not None:
            raise timeout_exc
        if response.status_code != 200:
            raise NebiusAPIError(response.status_code, str(response.text)[:500])
        data = response.json()
        try:
            message = dict(data["choices"][0]["message"])
        except Exception:
            raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:500]}")
        if isinstance(message.get("content"), str):
            message["content"] = strip_think_blocks(message["content"])
        return message


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
    from server.schemas.segment import SegmentTriage, parse_triage_json

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


def assemble_judge_messages(
    *, lift: str, observations: str, prompts_dir: str, measurement: Optional[str] = None
) -> List[ChatMessage]:
    """
    Judge (Stage 2): rubric + per-lift prompt, with two pieces of evidence and
    no images — the Stage-1 observations, and the local tracker's measurement
    of the bar path when there is one. Instruct to lower confidence when the
    observations mention occlusion or an oblique camera angle.
    """
    lift = _lift_name(lift)
    system_prompt = _load_prompt(f"{prompts_dir}/system.txt")
    user_prompt = _load_prompt(f"{prompts_dir}/user_{lift}.txt")
    guidance = (
        "Evidence is a vision-model observation summary (not raw images)"
        + (", plus a numeric bar-path measurement from a local plate tracker. " if measurement else ". ")
        + "Judge bar-path efficiency ONLY from that evidence. "
    )
    if measurement:
        guidance += (
            "The two sources are independent: the measurement is the better guide to HOW FAR the bar "
            "strayed and WHERE in the rise; the observations are the better guide to direction (forward "
            "or back), bar-body contact and the quality of each phase. If they disagree on the size of "
            "the drift, prefer the measurement and say so in notes. Make max_horizontal_drift consistent "
            "with the measured figure. "
        )
    guidance += (
        "If the observations mention occlusion, darkness, or an oblique camera angle, reflect higher "
        "uncertainty and lower confidence."
    )
    messages: List[ChatMessage] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": guidance},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": f"Vision observations:\n{observations}"},
    ]
    if measurement:
        messages.append({"role": "user", "content": measurement})
    return messages


async def _judge_with_model(
    *,
    client: "NebiusChatClient",
    messages: List[ChatMessage],
    model: str,
    max_tokens: int,
) -> "LiftJudgment":
    """
    One judge model, with a single repair retry when the reply is not valid
    JSON. The verdict records which model produced it (`judged_by`), so the
    result can say whether the primary judge or its fallback answered.
    """
    from server.schemas.judgment import parse_and_validate_json

    first = await client.create_chat_completion(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        extra_body=NO_THINKING_EXTRA_BODY,
    )
    try:
        return parse_and_validate_json(first).model_copy(update={"judged_by": model})
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
        return parse_and_validate_json(second).model_copy(update={"judged_by": model})  # Let errors bubble


async def request_judgment(
    *,
    client: "NebiusChatClient",
    lift: str,
    observations: str,
    prompts_dir: str,
    model: str,
    max_tokens: Optional[int] = None,
    fallback_model: Optional[str] = None,
    measurement: Optional[str] = None,
) -> "LiftJudgment":
    """
    Judge (Stage 2) with one repair retry on invalid JSON, then one retry on a
    second judge model when the primary fails outright — an API error, a timeout,
    or no valid JSON even after the repair. Pass fallback_model="" to disable.
    Raises the fallback's error if that one fails too. `measurement` is the
    tracker's evidence block (see services/measurement.py), if the bar was tracked.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("NEBIUS_JUDGE_MAX_TOKENS") or DEFAULT_JUDGE_MAX_TOKENS)
    if fallback_model is None:
        fallback_model = judge_fallback_model()
    messages = assemble_judge_messages(
        lift=lift, observations=observations, prompts_dir=prompts_dir, measurement=measurement
    )

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

