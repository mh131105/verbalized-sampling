from __future__ import annotations

import json
from pathlib import Path

from tp1_nlp.domain.models import (
    ConditionConfig,
    ExperimentConfig,
    GenerationParams,
    GenerationResult,
    MessageExample,
    ModelConfig,
    PerplexityConfig,
)
from tp1_nlp.generation.runner import _run_message_condition
from tp1_nlp.storage.jsonl import read_jsonl


class FlakyClient:
    def __init__(self, *, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.calls = 0

    def generate(
        self, messages: list[dict[str, str]], params: GenerationParams
    ) -> GenerationResult:
        self.calls += 1
        if self.calls <= self.failures_before_success:
            raise ConnectionError("Connection refused")
        return GenerationResult(
            text="Claro, vou verificar e retorno.",
            raw_response={"ok": True, "attempt": self.calls},
            model_name="fake",
            provider="fake",
            usage={"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
            latency_ms=10,
            finish_reason="stop",
        )


def test_retry_success_does_not_write_intermediate_failed_record(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("tp1_nlp.generation.runner.time.sleep", lambda _: None)

    client = FlakyClient(failures_before_success=1)
    generations_path, log_path = _run_single_message(tmp_path, client)

    generations = read_jsonl(generations_path)
    log_events = _read_log_events(log_path)

    assert client.calls == 2
    assert len(generations) == 1
    assert generations[0]["status"] == "success"
    assert [event["event"] for event in log_events].count("generation_retry") == 1
    assert [event["event"] for event in log_events].count("generation_failed") == 0


def test_retry_persistent_failure_writes_one_failed_record(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("tp1_nlp.generation.runner.time.sleep", lambda _: None)

    client = FlakyClient(failures_before_success=3)
    generations_path, log_path = _run_single_message(tmp_path, client)

    generations = read_jsonl(generations_path)
    log_events = _read_log_events(log_path)

    assert client.calls == 3
    assert len(generations) == 1
    assert generations[0]["status"] == "failed"
    assert generations[0]["error_code"] == "LLM_RESPONSE_ERROR"
    assert [event["event"] for event in log_events].count("generation_retry") == 2
    assert [event["event"] for event in log_events].count("generation_failed") == 1


def _run_single_message(tmp_path: Path, client: FlakyClient) -> tuple[Path, Path]:
    config = ExperimentConfig(
        run_id_prefix="pytest",
        output_root="results/runs",
        baseline_repetitions=1,
        dataset_path="data/messages.jsonl",
        min_messages=1,
        max_messages=1,
        required_categories=[],
        generation_defaults=GenerationParams(),
        models=[],
        conditions=[],
        perplexity=PerplexityConfig(enabled=False),
    )
    model = ModelConfig(model_id="fake_model", provider="fake", model_name="fake")
    condition = ConditionConfig(
        condition_id="baseline",
        condition_type="baseline",
        prompt_id="baseline",
        prompt_version="test",
        prompt_path="unused",
        system_message="system",
        n_candidates=1,
    )
    message = MessageExample(
        message_id="msg_001",
        category="pedido",
        formality="formal",
        input_message="Pode verificar?",
        expected_tones=["formal"],
    )
    generations_path = tmp_path / "generations.jsonl"
    log_path = tmp_path / "run.log"
    _run_message_condition(
        client=client,
        config=config,
        model=model,
        condition=condition,
        message=message,
        template="Mensagem recebida:\n{{input_message}}\nResposta:",
        run_id="pytest_retry",
        raw_dir=tmp_path / "raw",
        generations_path=generations_path,
        log_path=log_path,
    )
    return generations_path, log_path


def _read_log_events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
