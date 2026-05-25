from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from tp1_nlp.storage.jsonl import read_jsonl


MANUAL_FIELDS = [
    "run_id",
    "message_id",
    "model_id",
    "condition",
    "sample_index",
    "text",
    "utility",
    "coherence",
    "fidelity",
    "tone_adequacy",
    "notes",
]


def export_manual_eval_template(run_dir: Path) -> Path:
    generations = [
        record
        for record in read_jsonl(run_dir / "generations.jsonl")
        if record.get("status") == "success"
    ]
    path = run_dir / "manual_eval_template.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANUAL_FIELDS)
        writer.writeheader()
        for record in generations:
            writer.writerow(
                {
                    "run_id": record["run_id"],
                    "message_id": record["message_id"],
                    "model_id": record["model_id"],
                    "condition": record["condition"],
                    "sample_index": record["sample_index"],
                    "text": record["text"],
                    "utility": "",
                    "coherence": "",
                    "fidelity": "",
                    "tone_adequacy": "",
                    "notes": "",
                }
            )
    return path


def import_manual_eval(run_dir: Path, csv_path: Path) -> Path:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    _validate_manual_rows(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, int]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model_id"], row["condition"], row["run_id"])].append(
            {
                metric: int(row[metric])
                for metric in ["utility", "coherence", "fidelity", "tone_adequacy"]
            }
        )

    output = run_dir / "manual_metrics_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["run_id", "model_id", "condition", "metric", "value", "count"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (model_id, condition, run_id), values in sorted(grouped.items()):
            for metric in ["utility", "coherence", "fidelity", "tone_adequacy"]:
                metric_values = [item[metric] for item in values]
                writer.writerow(
                    {
                        "run_id": run_id,
                        "model_id": model_id,
                        "condition": condition,
                        "metric": metric,
                        "value": sum(metric_values) / len(metric_values),
                        "count": len(metric_values),
                    }
                )
    return output


def _validate_manual_rows(rows: list[dict[str, Any]]) -> None:
    for line_number, row in enumerate(rows, start=2):
        for field in MANUAL_FIELDS:
            if field not in row:
                raise ValueError(f"Missing field {field} in manual eval CSV")
        for metric in ["utility", "coherence", "fidelity", "tone_adequacy"]:
            try:
                value = int(row[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {metric} at line {line_number}: expected integer 1-5"
                ) from exc
            if value < 1 or value > 5:
                raise ValueError(
                    f"Invalid {metric} at line {line_number}: expected integer 1-5"
                )

