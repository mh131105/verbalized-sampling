from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from tp1_nlp.storage.jsonl import read_jsonl


def export_response_examples(run_dir: Path, limit_per_group: int = 2) -> Path:
    generations = [
        record
        for record in read_jsonl(run_dir / "generations.jsonl")
        if record.get("status") == "success"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in generations:
        key = (record["model_id"], record["condition"])
        if len(grouped[key]) < limit_per_group:
            grouped[key].append(record)

    output = run_dir / "response_examples.csv"
    with output.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "model_id",
            "condition",
            "message_id",
            "sample_index",
            "declared_tone",
            "verbalized_probability",
            "text",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rows in grouped.values():
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
    return output

