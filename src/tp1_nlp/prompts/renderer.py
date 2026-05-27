from __future__ import annotations

from tp1_nlp.domain.models import ChatMessage, ConditionConfig, MessageExample


def render_messages(
    *,
    template: str,
    condition: ConditionConfig,
    message: MessageExample,
) -> list[ChatMessage]:
    values = {
        "message_id": message.message_id,
        "category": message.category,
        "formality": message.formality or "",
        "input_message": message.input_message,
        "expected_tones": ", ".join(message.expected_tones),
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

