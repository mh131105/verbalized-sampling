from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ChatMessage = dict[str, str]


@dataclass(frozen=True)
class MessageExample:
    message_id: str
    category: str
    formality: str | None
    input_message: str
    expected_tones: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationParams:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: list[str] | None = None
    n_candidates: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationParams":
        data = dict(data or {})
        known = {
            "temperature": data.pop("temperature", None),
            "top_p": data.pop("top_p", None),
            "max_tokens": data.pop("max_tokens", None),
            "seed": data.pop("seed", None),
            "stop": data.pop("stop", None),
            "n_candidates": data.pop("n_candidates", None),
        }
        extra = data.pop("extra_params", {})
        if data:
            extra = {**extra, **data}
        return cls(extra_params=extra, **known)

    def merged(self, *overrides: "GenerationParams") -> "GenerationParams":
        result = self
        for override in overrides:
            result = GenerationParams(
                temperature=override.temperature
                if override.temperature is not None
                else result.temperature,
                top_p=override.top_p if override.top_p is not None else result.top_p,
                max_tokens=override.max_tokens
                if override.max_tokens is not None
                else result.max_tokens,
                seed=override.seed if override.seed is not None else result.seed,
                stop=override.stop if override.stop is not None else result.stop,
                n_candidates=override.n_candidates
                if override.n_candidates is not None
                else result.n_candidates,
                extra_params={**result.extra_params, **override.extra_params},
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": self.stop,
            "n_candidates": self.n_candidates,
        }
        data.update(self.extra_params)
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    provider: str
    model_name: str
    backend: str | None = None
    revision: str | None = None
    is_local: bool = False
    api_base: str | None = None
    api_key_env: str | None = None
    quantization: str | None = None
    device: str | None = None
    generation_params: GenerationParams = field(default_factory=GenerationParams)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionConfig:
    condition_id: str
    condition_type: str
    prompt_id: str
    prompt_version: str
    prompt_path: str
    system_message: str
    n_candidates: int
    vs_probability_mode: str | None = None
    probability_threshold: float | None = None
    generation_params: GenerationParams = field(default_factory=GenerationParams)


@dataclass(frozen=True)
class PerplexityConfig:
    enabled: bool = True
    evaluator_model: str = "pierreguillou/gpt2-small-portuguese"
    max_length: int = 256


@dataclass(frozen=True)
class ExperimentConfig:
    run_id_prefix: str
    output_root: str
    baseline_repetitions: int
    dataset_path: str
    min_messages: int
    max_messages: int
    required_categories: list[str]
    generation_defaults: GenerationParams
    models: list[ModelConfig]
    conditions: list[ConditionConfig]
    perplexity: PerplexityConfig


@dataclass
class GenerationResult:
    text: str
    raw_response: dict[str, Any]
    model_name: str
    provider: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class ParsedAlternative:
    tone: str | None
    text: str
    probability: float | None
    missing_probability: bool = False


@dataclass(frozen=True)
class GenerationRecord:
    run_id: str
    message_id: str
    model_id: str
    provider: str
    condition: str
    sample_index: int
    text: str
    declared_tone: str | None
    verbalized_probability: float | None
    raw_response_path: str
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    status: str
    error_code: str | None = None
    missing_probability: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "message_id": self.message_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "condition": self.condition,
            "sample_index": self.sample_index,
            "text": self.text,
            "declared_tone": self.declared_tone,
            "verbalized_probability": self.verbalized_probability,
            "raw_response_path": self.raw_response_path,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "status": self.status,
            "error_code": self.error_code,
            "missing_probability": self.missing_probability,
        }


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    message_id: str | None
    model_id: str
    condition: str
    metric: str
    value: float | None
    direction: str
    level: str
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "message_id": self.message_id,
            "model_id": self.model_id,
            "condition": self.condition,
            "metric": self.metric,
            "value": self.value,
            "direction": self.direction,
            "level": self.level,
            "category": self.category,
        }

