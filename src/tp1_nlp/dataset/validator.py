from __future__ import annotations

from collections import Counter

from tp1_nlp.domain.models import MessageExample


def validate_messages(
    messages: list[MessageExample],
    *,
    min_messages: int = 20,
    max_messages: int = 30,
    required_categories: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not (min_messages <= len(messages) <= max_messages):
        errors.append(
            f"Dataset must contain between {min_messages} and {max_messages} messages; "
            f"found {len(messages)}"
        )

    ids = [message.message_id for message in messages]
    duplicates = [key for key, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate message_id values: {duplicates}")

    for message in messages:
        if not message.message_id.strip():
            errors.append("message_id cannot be empty")
        if not message.category.strip():
            errors.append(f"{message.message_id}: category cannot be empty")
        if not message.input_message.strip():
            errors.append(f"{message.message_id}: input_message cannot be empty")
        if not message.expected_tones:
            errors.append(f"{message.message_id}: expected_tones cannot be empty")
        response_length = message.metadata.get("response_length")
        if response_length is not None:
            errors.extend(_validate_response_length(message.message_id, response_length))

    categories = {message.category for message in messages}
    missing_categories = sorted(set(required_categories or []) - categories)
    if missing_categories:
        errors.append(f"Dataset misses required categories: {missing_categories}")

    return errors


def _validate_response_length(message_id: str, value: object) -> list[str]:
    if not isinstance(value, dict):
        return [f"{message_id}: response_length must be an object"]

    min_words = value.get("min_words")
    max_words = value.get("max_words")
    if not isinstance(min_words, int) or not isinstance(max_words, int):
        return [f"{message_id}: response_length min_words and max_words must be integers"]
    if min_words <= 0 or max_words <= 0:
        return [f"{message_id}: response_length values must be positive"]
    if min_words > max_words:
        return [f"{message_id}: response_length min_words cannot exceed max_words"]
    return []
