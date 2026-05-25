from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tp1_nlp.domain.models import (
    ConditionConfig,
    ExperimentConfig,
    GenerationParams,
    ModelConfig,
    PerplexityConfig,
)
from tp1_nlp.storage.paths import ProjectPaths


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def load_experiment_config(path: Path, project_paths: ProjectPaths) -> ExperimentConfig:
    raw = _load_yaml(path)
    run = raw.get("run", {})
    dataset = raw.get("dataset", {})
    perplexity = raw.get("perplexity", {})

    defaults = GenerationParams.from_dict(raw.get("generation_defaults", {}))

    models = [_parse_model(item) for item in raw.get("models", [])]
    if not models:
        raise ValueError("At least one model must be configured")

    conditions = [_parse_condition(item) for item in raw.get("conditions", [])]
    if not conditions:
        raise ValueError("At least one condition must be configured")

    return ExperimentConfig(
        run_id_prefix=str(run.get("run_id_prefix", "run")),
        output_root=str(run.get("output_root", "results/runs")),
        baseline_repetitions=int(run.get("baseline_repetitions", 1)),
        dataset_path=str(dataset.get("path", "data/messages.jsonl")),
        min_messages=int(dataset.get("min_messages", 20)),
        max_messages=int(dataset.get("max_messages", 30)),
        required_categories=list(dataset.get("required_categories", [])),
        generation_defaults=defaults,
        models=models,
        conditions=conditions,
        perplexity=PerplexityConfig(
            enabled=bool(perplexity.get("enabled", True)),
            evaluator_model=str(
                perplexity.get(
                    "evaluator_model", "pierreguillou/gpt2-small-portuguese"
                )
            ),
            max_length=int(perplexity.get("max_length", 256)),
        ),
    )


def _parse_model(item: dict[str, Any]) -> ModelConfig:
    if "model_id" not in item or "provider" not in item or "model_name" not in item:
        raise ValueError(f"Invalid model config: {item}")
    known_keys = {
        "model_id",
        "provider",
        "model_name",
        "backend",
        "revision",
        "is_local",
        "api_base",
        "api_key_env",
        "quantization",
        "device",
        "generation_params",
    }
    extra = {key: value for key, value in item.items() if key not in known_keys}
    return ModelConfig(
        model_id=str(item["model_id"]),
        provider=str(item["provider"]),
        model_name=str(item["model_name"]),
        backend=item.get("backend"),
        revision=item.get("revision"),
        is_local=bool(item.get("is_local", False)),
        api_base=item.get("api_base"),
        api_key_env=item.get("api_key_env"),
        quantization=item.get("quantization"),
        device=item.get("device"),
        generation_params=GenerationParams.from_dict(item.get("generation_params", {})),
        extra=extra,
    )


def _parse_condition(item: dict[str, Any]) -> ConditionConfig:
    required = [
        "condition_id",
        "condition_type",
        "prompt_id",
        "prompt_version",
        "prompt_path",
        "system_message",
        "n_candidates",
    ]
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"Invalid condition config, missing {missing}: {item}")
    return ConditionConfig(
        condition_id=str(item["condition_id"]),
        condition_type=str(item["condition_type"]),
        prompt_id=str(item["prompt_id"]),
        prompt_version=str(item["prompt_version"]),
        prompt_path=str(item["prompt_path"]),
        system_message=str(item["system_message"]),
        n_candidates=int(item["n_candidates"]),
        vs_probability_mode=item.get("vs_probability_mode"),
        probability_threshold=item.get("probability_threshold"),
        generation_params=GenerationParams.from_dict(item.get("generation_params", {})),
    )

