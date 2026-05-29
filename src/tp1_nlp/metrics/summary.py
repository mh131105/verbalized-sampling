from __future__ import annotations

import csv
import json
import re
import unicodedata
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
    "tone_diversity_ratio": "higher_is_better",
    "parse_success_rate": "higher_is_better",
    "valid_tone_rate": "higher_is_better",
    "threshold_violation_rate": "lower_is_better",
}

ALLOWED_TONES = {
    "formal",
    "cordial",
    "objetivo",
    "direto",
    "empatico",
    "informal",
    "firme",
    "agradecido",
    "positivo",
    "justificativo",
    "curto",
    "gentil",
    "entusiasmado",
    "caloroso",
    "animado",
    "objective",
    "direct",
    "empathetic",
    "polite",
    "firm",
    "grateful",
    "explanatory",
    "brief",
    "kind",
    "enthusiastic",
    "warm",
    "upbeat",
    "justified",
    "short",
    "gentle",
}


def compute_metrics_for_run(
    *,
    run_dir: Path,
    dataset_path: Path,
    perplexity: PerplexityConfig,
) -> tuple[Path, Path]:
    all_generations = list(read_jsonl(run_dir / "generations.jsonl"))
    generations = [
        record
        for record in all_generations
        if record.get("status") == "success" and record.get("text")
    ]
    messages = {message.message_id: message for message in load_messages(dataset_path)}
    records = _message_level_metrics(generations, messages, perplexity)
    records.extend(
        _operational_metrics(
            all_generations,
            messages,
            _condition_probability_thresholds(run_dir / "manifest.json"),
        )
    )

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
            "tone_diversity_ratio": _tone_diversity_ratio(items),
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
                    formality=messages[message_id].formality,
                )
            )
    return output


def _operational_metrics(
    generations: list[dict[str, Any]],
    messages: dict[str, Any],
    probability_thresholds: dict[str, float],
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

    output: list[MetricRecord] = []
    for (run_id, message_id, model_id, condition), items in sorted(grouped.items()):
        if condition == "baseline":
            continue
        message = messages.get(message_id)
        category = message.category if message is not None else None
        formality = message.formality if message is not None else None
        success_items = [
            item for item in items if item.get("status") == "success" and item.get("text")
        ]

        values: dict[str, float | None] = {
            "parse_success_rate": 1.0 if success_items else 0.0,
            "valid_tone_rate": _valid_tone_rate(success_items),
        }

        threshold = probability_thresholds.get(condition)
        if threshold is not None:
            values["threshold_violation_rate"] = _threshold_violation_rate(
                success_items, threshold
            )

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
                    category=category,
                    formality=formality,
                )
            )
    return output


def summarize_metrics(
    records: list[MetricRecord], messages: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.value is None:
            continue
        run_id = record.run_id
        grouped[
            (
                run_id,
                record.model_id,
                record.condition,
                record.metric,
                "__all__",
                "__all__",
            )
        ].append(record.value)
        grouped[
            (
                run_id,
                record.model_id,
                record.condition,
                record.metric,
                record.category or "__unknown__",
                "__all__",
            )
        ].append(record.value)
        grouped[
            (
                run_id,
                record.model_id,
                record.condition,
                record.metric,
                "__all__",
                record.formality or "__unknown__",
            )
        ].append(record.value)

    summaries: list[dict[str, Any]] = []
    for (
        run_id,
        model_id,
        condition,
        metric,
        category,
        formality,
    ), values in sorted(grouped.items()):
        summaries.append(
            {
                "run_id": run_id,
                "model_id": model_id,
                "condition": condition,
                "metric": metric,
                "category": category,
                "formality": formality,
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
    expected = {
        normalized
        for tone in expected_tones
        if (normalized := _normalize_tone(tone)) in ALLOWED_TONES
    }
    declared = {
        normalized
        for tone in declared_tones
        if tone and (normalized := _normalize_tone(tone)) in ALLOWED_TONES
    }
    if not expected:
        return None
    if not declared:
        return 0.0
    return len(declared & expected) / len(expected)


def _tone_diversity_ratio(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    distinct_valid_tones = {
        normalized
        for item in items
        if (
            normalized := _normalize_tone(str(item.get("declared_tone") or ""))
        )
        in ALLOWED_TONES
    }
    return len(distinct_valid_tones) / len(items)


def _valid_tone_rate(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    valid = [
        item
        for item in items
        if _normalize_tone(str(item.get("declared_tone") or "")) in ALLOWED_TONES
    ]
    return len(valid) / len(items)


def _threshold_violation_rate(
    items: list[dict[str, Any]], threshold: float
) -> float | None:
    probabilities = [
        float(item["verbalized_probability"])
        for item in items
        if item.get("verbalized_probability") is not None
    ]
    if not probabilities:
        return None
    violations = [probability for probability in probabilities if probability >= threshold]
    return len(violations) / len(probabilities)


def _condition_probability_thresholds(manifest_path: Path) -> dict[str, float]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    thresholds: dict[str, float] = {}
    for condition in manifest.get("conditions_experimental", []):
        condition_id = condition.get("condition_id")
        threshold = condition.get("probability_threshold")
        if condition_id and threshold is not None:
            thresholds[str(condition_id)] = float(threshold)
    return thresholds


def _normalize_tone(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.strip().lower())
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    return re.sub(r"\s+", "_", without_accents)


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
        "formality",
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
        "formality",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
