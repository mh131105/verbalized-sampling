from __future__ import annotations

from pathlib import Path

import yaml


def test_official_prompts_hide_formality_and_expected_tones() -> None:
    root = Path(__file__).resolve().parents[2]

    config_versions = {
        root / "configs" / "experiment.lmstudio.yaml": "v3",
    }

    for config_path, expected_version in config_versions.items():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for condition in config["conditions"]:
            assert condition["prompt_version"] == expected_version
            template = (root / condition["prompt_path"]).read_text(encoding="utf-8")
            assert "{{category}}" in template
            assert "{{formality}}" not in template
            assert "{{expected_tones}}" not in template
            assert "Formalidade esperada" not in template
            assert "Tons esperados" not in template
            assert "Expected formality" not in template
            assert "Expected tones" not in template


def test_v3_baselines_request_final_reply() -> None:
    root = Path(__file__).resolve().parents[2]
    templates = [
        (root / "data" / "prompts" / "baseline_v3.txt").read_text(encoding="utf-8"),
    ]

    for template in templates:
        assert "{{category}}" in template
        assert "aproximadamente 15 palavras" in template
        assert "{{response_min_words}}" not in template
        assert "{{response_max_words}}" not in template
        assert "sem forcar contagem exata" in template
        assert "SOMENTE a resposta" in template or "Output ONLY the response" in template
        assert "{{formality}}" not in template
        assert "{{expected_tones}}" not in template


def test_v3_vs_prompts_keep_required_schema() -> None:
    root = Path(__file__).resolve().parents[2]

    for prompt_path in [
        root / "data" / "prompts" / "vs_intermediate_v3.txt",
        root / "data" / "prompts" / "vs_aggressive_v3.txt",
    ]:
        template = prompt_path.read_text(encoding="utf-8")
        assert "aproximadamente 15 palavras" in template
        assert "{{response_min_words}}" not in template
        assert "{{response_max_words}}" not in template
        assert "sem forcar contagem exata" in template
        assert "distribuicao completa" in template
        assert '"responses"' in template
        assert '"tone"' in template
        assert '"text"' in template
        assert '"probability"' in template
