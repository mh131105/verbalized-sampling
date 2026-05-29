from __future__ import annotations

from pathlib import Path

from tp1_nlp.dataset.loader import load_messages
from tp1_nlp.dataset.validator import validate_messages


def test_project_dataset_matches_required_contract() -> None:
    messages = load_messages(Path("data/messages.jsonl"))

    errors = validate_messages(
        messages,
        min_messages=20,
        max_messages=30,
        required_categories=[
            "pedido",
            "cobranca",
            "recusa",
            "agradecimento",
            "convite",
            "esclarecimento",
            "confirmacao",
            "adiamento",
        ],
    )

    assert errors == []
    for message in messages:
        assert "response_length" not in message.metadata
