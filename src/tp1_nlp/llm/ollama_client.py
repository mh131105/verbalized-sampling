from __future__ import annotations

import time
from typing import Any

import requests

from tp1_nlp.domain.models import ChatMessage, GenerationParams, GenerationResult, ModelConfig
from tp1_nlp.llm.base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.api_base = (config.api_base or "http://localhost:11434").rstrip("/")

    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        started = time.perf_counter()
        options: dict[str, Any] = {
            key: value
            for key, value in params.to_dict().items()
            if key not in {"max_tokens", "n_candidates", "stop"}
        }
        if params.max_tokens is not None:
            options["num_predict"] = params.max_tokens
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        response = requests.post(
            f"{self.api_base}/api/chat",
            json=payload,
            timeout=float(self.config.extra.get("timeout", 180)),
        )
        response.raise_for_status()
        raw = response.json()
        text = raw.get("message", {}).get("content") or ""
        latency_ms = int((time.perf_counter() - started) * 1000)
        return GenerationResult(
            text=text,
            raw_response=raw,
            model_name=self.config.model_name,
            provider="ollama",
            usage={
                "prompt_tokens": raw.get("prompt_eval_count"),
                "completion_tokens": raw.get("eval_count"),
                "total_tokens": None,
            },
            latency_ms=latency_ms,
            finish_reason="stop" if raw.get("done") else None,
        )

