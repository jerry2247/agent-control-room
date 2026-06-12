"""TrueFoundry LLM Gateway client (Sponsor: TrueFoundry).

The gateway exposes a standard OpenAI-compatible /chat/completions endpoint;
guardrails, rate limiting, cost controls, and model fallback are configured
server-side in the TrueFoundry console and apply transparently to every call
made here. Docs: https://www.truefoundry.com/docs/ai-gateway/chat-completions-overview

Backends:
  truefoundry : OpenAI-compatible POST {TRUEFOUNDRY_BASE_URL}/chat/completions
  anthropic   : direct Anthropic Messages API (hackathon credits)
  local       : no LLM; deterministic template engine handles generation
"""
from __future__ import annotations

import json
import re

import httpx


class LLMClient:
    def __init__(self, settings):
        self.backend = settings.llm_backend
        self.s = settings

    @property
    def available(self) -> bool:
        return self.backend in ("anthropic", "truefoundry")

    async def chat(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if self.backend == "truefoundry":
            return await self._chat_openai_compatible(system, user, max_tokens)
        if self.backend == "anthropic":
            return await self._chat_anthropic(system, user, max_tokens)
        raise RuntimeError("No LLM backend configured (LLM_BACKEND=local).")

    async def _chat_openai_compatible(self, system: str, user: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.s.truefoundry_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.s.truefoundry_api_key}"},
                json={
                    "model": self.s.truefoundry_chat_model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _chat_anthropic(self, system: str, user: str, max_tokens: int) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.s.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": self.s.anthropic_model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            blocks = resp.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    async def chat_json(self, system: str, user: str, max_tokens: int = 1200):
        """Chat and parse the first JSON array/object in the response."""
        text = await self.chat(system, user, max_tokens)
        match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM response: {text[:200]}")
        return json.loads(match.group(1))
