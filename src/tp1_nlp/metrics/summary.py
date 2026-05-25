from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from tp1_nlp.dataset.loader import load_messages
from tp1_nlp.domain.models import MetricRecord, PerplexityConfig
from tp1_nlp.metrics.bleu import self_bleu
from tp1_nlp.metrics.lexical import distinct_n, mean_pairwise_jaccard
from tp1_nlp.metrics.perplexity import PerplexityEvaluator
from tp1_nlp.storage.jsonl import read_jsonl


DIRECTIONS = {
    "jaccard": "lower_is_better",
    "self_bleu": "lower_is_better",
    "distinct_1": "higher_is_better",
    "distinct_2": "higher_is_better",
    "perplexity": "lower_is_better",
    "tone_coverage": "higher_is_better",
}


def compute_metrics_for_run(
    *,
    run_dir: Path,
    dataset_path: Path,
    perplexity: PerplexityConfig,
) -> tuple[Path, Path]:
    generations = [
        record
        for record in read_jsonl(run_dir / "generations.jsonl")
        if record.get("status") == "success" and record.get("text")
    ]
    messages = {message.message_id: message for message in load_messages(dataset_path)}
    records = _message_level_metrics(generations, messages, perplexity)

    metrics_long = run_dir / "metrics_long.csv"
    _write_metric_records(metrics_long, records)

    summary_records = summarize_metrics(records, messages)
    metrics_summary = run_dir / "metrics_summary.csv"
    _write_summary_records(metrics_summary, summary_records)
    return metrics_long, metrics_summary


def _message_level_metrics(
    generations: list[dict[str, Any]],
    messages: dict[str, Any],
    perplexity: PerplexityConfig,
) -> list[MetricRecord]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in generations:
        key = (
            record["run_id"],
            record["message_id"],
            record["model_id"],
            record["condition"],
        )
        grouped[key].append(record)

    evaluator: PerplexityEvaluator | None = None
    if perplexity.enabled:
        evaluator = PerplexityEvaluator(
            perplexity.evaluator_model, max_length=perplexity.max_length
        )

    output: list[MetricRecord] = []
    for (run_id, message_id, model_id, condition), items in sorted(grouped.items()):
        texts = [item["text"] for item in items]
        values = {
            "jaccard": mean_pairwise_jaccard(texts),
            "self_bleu": self_bleu(texts),
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
            "tone_coverage": _tone_coverage(
                [item.get("declared_tone") for item in items],
                messages[message_id].expected_tones,
            ),
        }
        if evaluator is not None:
            values["perplexity"] = _mean_perplexity(texts, evaluator)
        else:
            values["perplexity"] = None

        for metric, value in values.items():
            output.append(
                MetricRecord(
                    run_id=run_id,
                    message_id=message_id,
                    model_id=model_id,
                    condition=condition,
                    metric=metric,
                    value=value,
                    direction=DIRECTIONS[metric],
                    level="message",
                    category=messages[message_id].category,
                )
            )
    return output


def summarize_metrics(
    records: list[MetricRecord], messages: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.value is None:
            continue
        run_id = record.run_id
        grouped[
            (run_id, record.model_id, record.condition, record.metric, "__all__")
        ].append(record.value)
        grouped[
            (
                run_id,
                record.model_id,
                record.condition,
                record.metric,
                record.category or "__unknown__",
            )
        ].append(record.value)

    summaries: list[dict[str, Any]] = []
    for (run_id, model_id, condition, metric, category), values in sorted(grouped.items()):
        summaries.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "condition": condition,
                "metric": metric,
                "category": category,
                "value": sum(values) / len(values),
                "count": len(values),
                "direction": DIRECTIONS[metric],
                "level": "summary",
            }
        )
    return summaries


def _tone_coverage(
    declared_tones: list[str | None], expected_tones: list[str]
) -> float | None:
    if not expected_tones:
        return None
    expected = {tone.lower() for tone in expected_tones}
    declared = {tone.lower() for tone in declared_tones if tone}
    if not declared:
        return 0.0
    return len(declared & expected) / len(expected)


def _mean_perplexity(texts: list[str], evaluator: PerplexityEvaluator) -> float | None:
    scores: list[float] = []
    for text in texts:
        try:
            scores.append(evaluator.score(text))
        except Exception:
            return None
    return sum(scores) / len(scores) if scores else None


def _write_metric_records(path: Path, records: list[MetricRecord]) -> None:
    fieldnames = [
        "run_id",
        "message_id",
        "model_id",
        "condition",
        "metric",
        "value",
        "direction",
        "level",
        "category",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def _write_summary_records(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "model_id",
        "condition",
        "metric",
        "category",
        "value",
        "count",
        "direction",
        "level",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
