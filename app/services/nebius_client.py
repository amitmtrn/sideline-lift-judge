from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, TypedDict

import requests


class ChatMessage(TypedDict, total=False):
    role: str
    content: Any


class NebiusChatClient:
    """
    Minimal OpenAI-compatible client for Nebius Token Factory.
    """

    # Verified from Nebius Token Factory docs
    DEFAULT_BASE_URL: str = "https://api.tokenfactory.nebius.com/v1"
    # Preferred default model (vision-capable; see README)
    DEFAULT_MODEL: str = "nvidia/Cosmos3-Super-Reasoner"

    def __init__(self, api_key: str, base_url: Optional[str] = None, default_model: Optional[str] = None) -> None:
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.default_model = default_model or self.DEFAULT_MODEL

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_chat_completion(self, messages: List[ChatMessage], model: Optional[str] = None, max_tokens: int = 800) -> str:
        """
        Call the OpenAI-compatible /chat/completions endpoint. Returns the text content.
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, headers=self._headers(), json=payload, timeout=60))
        if response.status_code != 200:
            raise RuntimeError(f"Nebius API error {response.status_code}: {response.text}")
        data = response.json()
        # OpenAI-compatible shape
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected response format: {json.dumps(data)[:500]}")


def _load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def assemble_messages_for_lift(*, lift: str, images: List[str], prompts_dir: str) -> List[ChatMessage]:
    """
    Compose system and user prompts with image parts per OpenAI-compatible spec.
    Each image is passed as a data URL in image_url.type.
    """
    assert lift in ("snatch", "clean", "jerk")
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

