from __future__ import annotations

from abc import ABC, abstractmethod

from tp1_nlp.domain.models import ChatMessage, GenerationParams, GenerationResult, ModelConfig


class BaseLLMClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self, messages: list[ChatMessage], params: GenerationParams
    ) -> GenerationResult:
        raise NotImplementedError

