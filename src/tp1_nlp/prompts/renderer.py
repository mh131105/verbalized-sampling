from __future__ import annotations

from tp1_nlp.domain.models import ChatMessage, ConditionConfig, MessageExample


DEFAULT_RESPONSE_MIN_WORDS = 8
DEFAULT_RESPONSE_MAX_WORDS = 15


def render_messages(
    *,
    template: str,
    condition: ConditionConfig,
    message: MessageExample,
) -> list[ChatMessage]:
    min_words, max_words = _response_length(message)
    values = {
        "message_id": message.message_id,
        "category": message.category,
        "formality": message.formality or "",
        "input_message": message.input_message,
        "expected_tones": ", ".join(message.expected_tones),
        "response_min_words": str(min_words),
        "response_max_words": str(max_words),
        "n_candidates": str(condition.n_candidates),
        "probability_threshold": str(condition.probability_threshold or ""),
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return [
        {"role": "system", "content": condition.system_message},
        {"role": "user", "content": rendered},
    ]


def _response_length(message: MessageExample) -> tuple[int, int]:
    value = message.metadata.get("response_length")
    if not isinstance(value, dict):
        return DEFAULT_RESPONSE_MIN_WORDS, DEFAULT_RESPONSE_MAX_WORDS

    min_words = _positive_int_or_none(value.get("min_words"))
    max_words = _positive_int_or_none(value.get("max_words"))
    if min_words is None or max_words is None or min_words > max_words:
        return DEFAULT_RESPONSE_MIN_WORDS, DEFAULT_RESPONSE_MAX_WORDS
    return min_words, max_words


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
