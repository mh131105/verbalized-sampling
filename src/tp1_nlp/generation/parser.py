from __future__ import annotations

import json
import re
from typing import Any

from tp1_nlp.domain.models import ParsedAlternative


class ParseError(ValueError):
    pass


def parse_vs_response(raw_text: str) -> list[ParsedAlternative]:
    data = _load_json(raw_text)
    responses = data.get("responses") if isinstance(data, dict) else None
    if not isinstance(responses, list):
        raise ParseError("VS response must contain a 'responses' list")

    alternatives: list[ParsedAlternative] = []
    for item in responses:
        if not isinstance(item, dict):
            raise ParseError("Each VS response item must be an object")
        text = str(item.get("text", "")).strip()
        if not text:
            raise ParseError("VS response item is missing non-empty text")
        probability = _parse_probability(item.get("probability"))
        alternatives.append(
            ParsedAlternative(
                tone=str(item["tone"]).strip() if item.get("tone") else None,
                text=text,
                probability=probability,
                missing_probability=probability is None,
            )
        )
    return alternatives


def _load_json(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ParseError("Could not find JSON object in VS response")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"Could not parse VS response as JSON: {exc}") from exc


def _parse_probability(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= probability <= 1:
        return None
    return probability

