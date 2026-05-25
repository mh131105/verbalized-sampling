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

    categories = {message.category for message in messages}
    missing_categories = sorted(set(required_categories or []) - categories)
    if missing_categories:
        errors.append(f"Dataset misses required categories: {missing_categories}")

    return errors

