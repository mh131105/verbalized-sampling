from __future__ import annotations

from tp1_nlp.domain.models import ConditionConfig, MessageExample
from tp1_nlp.prompts.renderer import render_messages


def test_renderer_uses_response_length_metadata() -> None:
    rendered = render_messages(
        template="{{response_min_words}}-{{response_max_words}}: {{input_message}}",
        condition=_condition(),
        message=MessageExample(
            message_id="msg_001",
            category="pedido",
            formality="formal",
            input_message="Pode verificar?",
            expected_tones=["formal"],
            metadata={"response_length": {"min_words": 6, "max_words": 12}},
        ),
    )

    assert rendered[-1]["content"] == "6-12: Pode verificar?"


def test_renderer_uses_default_response_length_without_metadata() -> None:
    rendered = render_messages(
        template="{{response_min_words}}-{{response_max_words}}: {{input_message}}",
        condition=_condition(),
        message=MessageExample(
            message_id="msg_001",
            category="pedido",
            formality="formal",
            input_message="Pode verificar?",
            expected_tones=["formal"],
        ),
    )

    assert rendered[-1]["content"] == "8-15: Pode verificar?"


def _condition() -> ConditionConfig:
    return ConditionConfig(
        condition_id="baseline",
        condition_type="baseline",
        prompt_id="baseline",
        prompt_version="test",
        prompt_path="unused",
        system_message="system",
        n_candidates=1,
    )
