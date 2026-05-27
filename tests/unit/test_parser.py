from __future__ import annotations

from tp1_nlp.generation.parser import parse_vs_response


def test_parse_vs_response_from_json_code_fence() -> None:
    raw = """```json
    {"responses": [{"tone": "formal", "text": "Claro, envio hoje.", "probability": 0.31}]}
    ```"""

    parsed = parse_vs_response(raw)

    assert len(parsed) == 1
    assert parsed[0].tone == "formal"
    assert parsed[0].text == "Claro, envio hoje."
    assert parsed[0].probability == 0.31
    assert parsed[0].missing_probability is False

