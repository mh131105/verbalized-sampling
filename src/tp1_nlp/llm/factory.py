from __future__ import annotations

from tp1_nlp.domain.models import ModelConfig
from tp1_nlp.llm.base import BaseLLMClient
from tp1_nlp.llm.fake_client import FakeLLMClient
from tp1_nlp.llm.ollama_client import OllamaClient
from tp1_nlp.llm.openai_compatible_client import OpenAICompatibleClient


def create_llm_client(config: ModelConfig) -> BaseLLMClient:
    provider = config.provider.lower()
    if provider == "fake":
        return FakeLLMClient(config)
    if provider in {"openai_compatible", "openai", "openrouter"}:
        return OpenAICompatibleClient(config)
    if provider == "ollama":
        return OllamaClient(config)
    raise ValueError(f"Unsupported provider for {config.model_id}: {config.provider}")

