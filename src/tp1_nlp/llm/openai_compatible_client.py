from __future__ import annotations

import os
import time
from typing import Any

import requests

from tp1_nlp.domain.models import ChatMessage, GenerationParams, GenerationResult, ModelConfig
from tp1_nlp.llm.base import BaseLLMClient


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if not config.api_base:
            raise ValueError(f"{config.model_id}: api_base is required")
        if not config.api_key_env:
            raise ValueError(f"{config.model_id}: api_key_env is required")
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is required for {config.model_id}"
            )

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            **params.to_dict(),
        }
        payload.pop("n_candidates", None)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.config.api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=float(self.config.extra.get("timeout", 90)),
        )
        response.raise_for_status()
        raw = response.json()
        choice = raw["choices"][0]
        text = choice.get("message", {}).get("content") or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        return GenerationResult(
            text=text,
            raw_response=raw,
            model_name=self.config.model_name,
            provider=self.config.provider,
            usage=raw.get("usage", {}),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )

