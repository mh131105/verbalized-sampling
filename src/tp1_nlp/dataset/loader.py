from __future__ import annotations

from pathlib import Path
from typing import Any

from tp1_nlp.domain.models import MessageExample
from tp1_nlp.storage.jsonl import read_jsonl


def load_messages(path: Path) -> list[MessageExample]:
    return [_message_from_dict(record, path) for record in read_jsonl(path)]


def _message_from_dict(record: dict[str, Any], path: Path) -> MessageExample:
    required = ["message_id", "category", "input_message", "expected_tones"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing fields {missing} in dataset {path}")
    metadata = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "message_id",
            "category",
            "formality",
            "input_message",
            "expected_tones",
        }
    }
    return MessageExample(
        message_id=str(record["message_id"]),
        category=str(record["category"]),
        formality=record.get("formality"),
        input_message=str(record["input_message"]),
        expected_tones=[str(tone) for tone in record["expected_tones"]],
        metadata=metadata,
    )

