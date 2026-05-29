from __future__ import annotations

import csv
import json
from pathlib import Path

from tp1_nlp.domain.models import PerplexityConfig
from tp1_nlp.metrics.bleu import self_bleu
from tp1_nlp.metrics.lexical import distinct_n, mean_pairwise_jaccard
from tp1_nlp.metrics.summary import (
    _threshold_violation_rate,
    _tone_coverage,
    _tone_diversity_ratio,
    _valid_tone_rate,
    compute_metrics_for_run,
)
from tp1_nlp.storage.jsonl import write_jsonl
from tp1_nlp.storage.paths import ProjectPaths


def test_lexical_metrics_are_defined_for_multiple_texts() -> None:
    texts = [
        "Claro, posso revisar o texto ate amanha.",
        "Sem problema, reviso o material e retorno cedo.",
        "Posso verificar o texto e mandar comentarios pela manha.",
    ]

    assert mean_pairwise_jaccard(texts) is not None
    assert distinct_n(texts, 1) is not None
    assert distinct_n(texts, 2) is not None
    assert self_bleu(texts) is not None


def test_tone_metrics_use_canonical_allowed_tones() -> None:
    items = [
        {"declared_tone": "formal"},
        {"declared_tone": "Empatico"},
        {"declared_tone": "polite"},
        {"declared_tone": "formal_e_cooperativo"},
    ]

    assert _tone_coverage(["formal", "empatico"], ["formal", "empatico"]) == 1.0
    assert _tone_coverage(["formal", "polite"], ["formal", "polite"]) == 1.0
    assert _tone_diversity_ratio(items) == 3 / 4
    assert _valid_tone_rate(items) == 3 / 4


def test_threshold_violation_rate_counts_tail_failures() -> None:
    items = [
        {"verbalized_probability": 0.05},
        {"verbalized_probability": 0.1},
        {"verbalized_probability": 0.2},
    ]

    assert _threshold_violation_rate(items, 0.1) == 2 / 3


def test_compute_metrics_writes_vs_operational_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "conditions_experimental": [
                    {
                        "condition_id": "vs_aggressive",
                        "probability_threshold": 0.1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        run_dir / "generations.jsonl",
        [
            {
                "run_id": "test_run",
                "message_id": "pt_msg_001",
                "model_id": "fake_model",
                "condition": "vs_aggressive",
                "sample_index": index,
                "text": f"Resposta alternativa {index}.",
                "declared_tone": tone,
                "verbalized_probability": probability,
                "status": "success",
            }
            for index, (tone, probability) in enumerate(
                [
                    ("formal", 0.05),
                    ("empatico", 0.08),
                    ("direto", 0.1),
                    ("formal_e_cooperativo", 0.12),
                    ("curto", 0.03),
                ],
                start=1,
            )
        ],
    )

    metrics_long, _ = compute_metrics_for_run(
        run_dir=run_dir,
        dataset_path=ProjectPaths.discover().resolve("data/messages.jsonl"),
        perplexity=PerplexityConfig(enabled=False),
    )

    rows = list(csv.DictReader(metrics_long.open(encoding="utf-8")))
    metrics = {row["metric"]: float(row["value"]) for row in rows if row["value"]}

    assert metrics["parse_success_rate"] == 1.0
    assert metrics["tone_diversity_ratio"] == 4 / 5
    assert metrics["valid_tone_rate"] == 4 / 5
    assert metrics["threshold_violation_rate"] == 2 / 5
