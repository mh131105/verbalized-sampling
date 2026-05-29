from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tp1_nlp.config.loader import load_experiment_config
from tp1_nlp.dataset.loader import load_messages
from tp1_nlp.dataset.validator import validate_messages
from tp1_nlp.domain.models import (
    ConditionConfig,
    ExperimentConfig,
    GenerationParams,
    GenerationRecord,
    MessageExample,
    ModelConfig,
)
from tp1_nlp.generation.parser import ParseError, parse_vs_response
from tp1_nlp.llm.factory import create_llm_client
from tp1_nlp.prompts.renderer import render_messages
from tp1_nlp.prompts.template_loader import load_template
from tp1_nlp.storage.jsonl import append_jsonl
from tp1_nlp.storage.manifest import base_manifest, write_manifest
from tp1_nlp.storage.paths import ProjectPaths, validate_project_contract
from tp1_nlp.utils.hashing import file_sha256
from tp1_nlp.utils.logging import append_event
from tp1_nlp.utils.time import make_run_id, utc_now_iso


MAX_GENERATION_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (5, 15)


def run_experiment(
    *,
    config_path: Path,
    project_paths: ProjectPaths,
    run_id: str | None = None,
    only_model: str | None = None,
    only_condition: str | None = None,
) -> str:
    config = load_experiment_config(config_path, project_paths)
    _preflight(project_paths, config)

    dataset_path = project_paths.resolve(config.dataset_path)
    messages = load_messages(dataset_path)
    errors = validate_messages(
        messages,
        min_messages=config.min_messages,
        max_messages=config.max_messages,
        required_categories=config.required_categories,
    )
    if errors:
        raise ValueError("Dataset validation failed:\n" + "\n".join(errors))

    current_run_id = run_id or make_run_id(config.run_id_prefix)
    run_dir = project_paths.run_dir(current_run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = run_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    generations_path = run_dir / "generations.jsonl"
    manifest_path = run_dir / "manifest.json"

    selected_models = [
        model for model in config.models if only_model in {None, model.model_id}
    ]
    selected_conditions = [
        condition
        for condition in config.conditions
        if only_condition in {None, condition.condition_id}
    ]
    if not selected_models:
        raise ValueError(f"No configured model matches --only-model={only_model}")
    if not selected_conditions:
        raise ValueError(f"No configured condition matches --only-condition={only_condition}")

    prompt_snapshots = _prompt_snapshots(project_paths, selected_conditions)
    manifest = base_manifest(
        run_id=current_run_id,
        root=project_paths.root,
        dataset_path=dataset_path,
        dataset_hash=file_sha256(dataset_path),
        config_path=config_path,
        models=[_model_manifest(model) for model in selected_models],
        conditions=[_condition_manifest(condition) for condition in selected_conditions],
        prompts=prompt_snapshots,
        paths={
            "run_dir": str(run_dir),
            "generations": str(generations_path),
            "manifest": str(manifest_path),
            "raw_responses": str(raw_dir),
            "metrics_long": str(run_dir / "metrics_long.csv"),
            "metrics_summary": str(run_dir / "metrics_summary.csv"),
            "figures": str(run_dir / "figures"),
        },
    )
    write_manifest(manifest_path, manifest)
    append_event(log_path, "run_started", run_id=current_run_id)

    try:
        for model in selected_models:
            append_event(log_path, "model_started", model_id=model.model_id)
            client = create_llm_client(model)
            for condition in selected_conditions:
                template = load_template(project_paths.resolve(condition.prompt_path))
                append_event(
                    log_path,
                    "condition_started",
                    model_id=model.model_id,
                    condition=condition.condition_id,
                )
                for message in messages:
                    _run_message_condition(
                        client=client,
                        config=config,
                        model=model,
                        condition=condition,
                        message=message,
                        template=template,
                        run_id=current_run_id,
                        raw_dir=raw_dir,
                        generations_path=generations_path,
                        log_path=log_path,
                    )
        manifest["status"] = "completed"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append({"message": str(exc), "timestamp": utc_now_iso()})
        write_manifest(manifest_path, manifest)
        append_event(log_path, "run_failed", error=str(exc))
        raise

    manifest["completed_at"] = utc_now_iso()
    write_manifest(manifest_path, manifest)
    append_event(log_path, "run_completed", run_id=current_run_id)
    return current_run_id


def _preflight(project_paths: ProjectPaths, config: ExperimentConfig) -> None:
    project_paths.ensure_base()
    errors = validate_project_contract(project_paths)
    for condition in config.conditions:
        prompt_path = project_paths.resolve(condition.prompt_path)
        if not prompt_path.exists():
            errors.append(f"Missing prompt template: {condition.prompt_path}")
    if errors:
        raise ValueError("Project validation failed:\n" + "\n".join(errors))


def _run_message_condition(
    *,
    client: Any,
    config: ExperimentConfig,
    model: ModelConfig,
    condition: ConditionConfig,
    message: MessageExample,
    template: str,
    run_id: str,
    raw_dir: Path,
    generations_path: Path,
    log_path: Path,
) -> None:
    calls = (
        config.baseline_repetitions
        if condition.condition_type == "baseline"
        else 1
    )
    for call_index in range(1, calls + 1):
        rendered_messages = render_messages(
            template=template, condition=condition, message=message
        )
        params = _params_for_call(config, model, condition, call_index)
        raw_path = raw_dir / (
            f"{message.message_id}__{model.model_id}__"
            f"{condition.condition_id}__{call_index}.json"
        )
        append_event(
            log_path,
            "generation_started",
            model_id=model.model_id,
            condition=condition.condition_id,
            message_id=message.message_id,
            call_index=call_index,
        )
        try:
            result = _generate_with_retries(
                client=client,
                rendered_messages=rendered_messages,
                params=params,
                model=model,
                condition=condition,
                message=message,
                call_index=call_index,
                log_path=log_path,
            )
            raw_payload = {
                "run_id": run_id,
                "message_id": message.message_id,
                "model_id": model.model_id,
                "condition": condition.condition_id,
                "call_index": call_index,
                "prompt": rendered_messages,
                "params": params.to_dict(),
                "response": result.raw_response,
                "text": result.text,
            }
            _write_json(raw_path, raw_payload)
            records = _records_from_result(
                run_id=run_id,
                model=model,
                condition=condition,
                message=message,
                call_index=call_index,
                raw_path=raw_path,
                result=result,
            )
            for record in records:
                append_jsonl(generations_path, record.to_dict())
            append_event(
                log_path,
                "generation_completed",
                model_id=model.model_id,
                condition=condition.condition_id,
                message_id=message.message_id,
                call_index=call_index,
                records=len(records),
            )
        except Exception as exc:
            _write_json(
                raw_path,
                {
                    "run_id": run_id,
                    "message_id": message.message_id,
                    "model_id": model.model_id,
                    "condition": condition.condition_id,
                    "call_index": call_index,
                    "prompt": rendered_messages,
                    "params": params.to_dict(),
                    "error": str(exc),
                },
            )
            record = GenerationRecord(
                run_id=run_id,
                message_id=message.message_id,
                model_id=model.model_id,
                provider=model.provider,
                condition=condition.condition_id,
                sample_index=call_index,
                text="",
                declared_tone=None,
                verbalized_probability=None,
                raw_response_path=str(raw_path),
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                status="failed",
                error_code=_error_code(exc),
            )
            append_jsonl(generations_path, record.to_dict())
            append_event(
                log_path,
                "generation_failed",
                model_id=model.model_id,
                condition=condition.condition_id,
                message_id=message.message_id,
                call_index=call_index,
                error=str(exc),
            )


def _generate_with_retries(
    *,
    client: Any,
    rendered_messages: list[dict[str, str]],
    params: GenerationParams,
    model: ModelConfig,
    condition: ConditionConfig,
    message: MessageExample,
    call_index: int,
    log_path: Path,
) -> Any:
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        try:
            return client.generate(rendered_messages, params)
        except Exception as exc:
            if (
                attempt >= MAX_GENERATION_ATTEMPTS
                or not _is_retryable_generation_error(exc)
            ):
                raise
            delay = RETRY_DELAYS_SECONDS[min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)]
            append_event(
                log_path,
                "generation_retry",
                model_id=model.model_id,
                condition=condition.condition_id,
                message_id=message.message_id,
                call_index=call_index,
                attempt=attempt,
                max_attempts=MAX_GENERATION_ATTEMPTS,
                delay_seconds=delay,
                error=str(exc),
                error_code=_error_code(exc),
            )
            time.sleep(delay)
    raise RuntimeError("Generation retry loop exhausted without returning or raising")


def _params_for_call(
    config: ExperimentConfig,
    model: ModelConfig,
    condition: ConditionConfig,
    call_index: int,
) -> GenerationParams:
    n_candidates = 1 if condition.condition_type == "baseline" else condition.n_candidates
    call_seed = None
    base_seed = (
        condition.generation_params.seed
        or model.generation_params.seed
        or config.generation_defaults.seed
    )
    if base_seed is not None:
        call_seed = int(base_seed) + call_index - 1
    call_params = GenerationParams(seed=call_seed, n_candidates=n_candidates)
    return config.generation_defaults.merged(
        model.generation_params, condition.generation_params, call_params
    )


def _records_from_result(
    *,
    run_id: str,
    model: ModelConfig,
    condition: ConditionConfig,
    message: MessageExample,
    call_index: int,
    raw_path: Path,
    result: Any,
) -> list[GenerationRecord]:
    usage = result.usage or {}
    if condition.condition_type == "baseline":
        return [
            GenerationRecord(
                run_id=run_id,
                message_id=message.message_id,
                model_id=model.model_id,
                provider=model.provider,
                condition=condition.condition_id,
                sample_index=call_index,
                text=result.text.strip(),
                declared_tone=None,
                verbalized_probability=None,
                raw_response_path=str(raw_path),
                latency_ms=result.latency_ms,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                status="success" if result.text.strip() else "failed",
                error_code=None if result.text.strip() else "LLM_RESPONSE_ERROR",
            )
        ]

    try:
        alternatives = parse_vs_response(result.text)
    except ParseError:
        return [
            GenerationRecord(
                run_id=run_id,
                message_id=message.message_id,
                model_id=model.model_id,
                provider=model.provider,
                condition=condition.condition_id,
                sample_index=1,
                text=result.text.strip(),
                declared_tone=None,
                verbalized_probability=None,
                raw_response_path=str(raw_path),
                latency_ms=result.latency_ms,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                status="parse_failed",
                error_code="PARSE_ERROR",
            )
        ]

    records: list[GenerationRecord] = []
    for index, alternative in enumerate(alternatives, start=1):
        records.append(
            GenerationRecord(
                run_id=run_id,
                message_id=message.message_id,
                model_id=model.model_id,
                provider=model.provider,
                condition=condition.condition_id,
                sample_index=index,
                text=alternative.text,
                declared_tone=alternative.tone,
                verbalized_probability=alternative.probability,
                raw_response_path=str(raw_path),
                latency_ms=result.latency_ms,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                status="success",
                error_code=None,
                missing_probability=alternative.missing_probability,
            )
        )
    return records


def _prompt_snapshots(
    project_paths: ProjectPaths, conditions: list[ConditionConfig]
) -> list[dict[str, Any]]:
    snapshots = []
    for condition in conditions:
        template_path = project_paths.resolve(condition.prompt_path)
        snapshots.append(
            {
                "prompt_id": condition.prompt_id,
                "prompt_version": condition.prompt_version,
                "condition": condition.condition_id,
                "system_message": condition.system_message,
                "user_template_path": str(template_path),
                "user_template": load_template(template_path),
                "output_schema": "plain_text"
                if condition.condition_type == "baseline"
                else {"responses": [{"tone": "str", "text": "str", "probability": "float"}]},
            }
        )
    return snapshots


def _model_manifest(model: ModelConfig) -> dict[str, Any]:
    data = asdict(model)
    data["generation_params"] = model.generation_params.to_dict()
    data.pop("api_key_env", None)
    return data


def _condition_manifest(condition: ConditionConfig) -> dict[str, Any]:
    data = asdict(condition)
    data["generation_params"] = condition.generation_params.to_dict()
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error_code(exc: Exception) -> str:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    if "auth" in name or "unauthorized" in message:
        return "LLM_AUTH_ERROR"
    if "timeout" in name or "timeout" in message:
        return "LLM_TIMEOUT"
    if "rate" in message and "limit" in message:
        return "LLM_RATE_LIMIT"
    if isinstance(exc, ParseError):
        return "PARSE_ERROR"
    return "LLM_RESPONSE_ERROR"


def _is_retryable_generation_error(exc: Exception) -> bool:
    status_code = _http_status_code(exc)
    if status_code == 429 or (status_code is not None and 500 <= status_code <= 599):
        return True

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    retryable_fragments = [
        "timeout",
        "connection refused",
        "remote disconnected",
        "connection reset",
        "connection aborted",
        "newconnectionerror",
        "max retries exceeded",
        "temporarily unavailable",
        "service unavailable",
    ]
    haystack = f"{name} {message}"
    return any(fragment in haystack for fragment in retryable_fragments)


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None
