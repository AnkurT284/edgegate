"""BYOK LLM client — user brings their own API key.

Design goals (per user request):
- Keys are NEVER stored server-side. They arrive per-request via
  `X-LLM-API-Key` header or `llm` block in the JSON body, kept in memory
  only for that call. The dashboard keeps the key in browser localStorage.
- OpenAI-compatible Chat Completions transport, so one code path serves
  OpenAI, OpenRouter, Groq, Together, DeepSeek, xAI, Gemini (OpenAI
  endpoint), Ollama (no key) and any custom base_url.
- Mock mode: when no key is supplied (and provider needs one), agents fall
  back to deterministic template reasoning so the pipeline still runs
  offline and is fully testable without spending money.

Supported providers + default base URLs:
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx

PROVIDER_PRESETS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
    "custom": "",
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek": "deepseek-chat",
    "xai": "grok-3-mini",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.1",
    "custom": "",
}

NO_KEY_PROVIDERS = {"ollama", "custom"}


@dataclass
class BYOKConfig:
    provider: str = "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.provider = (self.provider or "openai").lower()
        if not self.base_url:
            self.base_url = PROVIDER_PRESETS.get(self.provider, "")
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "gpt-4o-mini")

    @property
    def mock_mode(self) -> bool:
        if self.provider in NO_KEY_PROVIDERS and self.base_url:
            return False  # try live call; Ollama usually needs no key
        return not bool(self.api_key)

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}" if not self.mock_mode else "mock"


async def chat_json(cfg: BYOKConfig, system: str, user: str,
                    max_tokens: int = 800) -> str:
    """Call OpenAI-compatible /chat/completions, ask for a JSON object.

    Returns raw text (expected JSON). Raises on transport errors so the
    caller can fall back to mock reasoning.
    """
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if cfg.provider == "openrouter":
        headers.setdefault("HTTP-Referer", "https://localhost/edgegate")
        headers.setdefault("X-Title", "EdgeGate BYOK")
    payload: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"] or "{}"
    except (KeyError, IndexError, TypeError):
        return "{}"


def parse_llm_json(text: str) -> dict[str, Any]:
    """Best-effort parse; never throws."""
    import json

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {"text": text}
    except Exception:
        # try to salvage the first {...} block
        try:
            s, e = text.index("{"), text.rindex("}") + 1
            obj = json.loads(text[s:e])
            return obj if isinstance(obj, dict) else {"text": text}
        except Exception:
            return {"text": text}
