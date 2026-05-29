from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from tp1_nlp.generation.runner import run_experiment
from tp1_nlp.storage.jsonl import read_jsonl
from tp1_nlp.storage.paths import ProjectPaths


def test_fake_runner_writes_generations(tmp_path: Path) -> None:
    paths = ProjectPaths.discover()
    run_id = f"pytest_{uuid4().hex}"
    config_path = tmp_path / "experiment.fake.yaml"
    config_path.write_text(
        """
run:
  run_id_prefix: "pytest"
  output_root: "results/runs"
  baseline_repetitions: 5

dataset:
  path: "data/messages.jsonl"
  min_messages: 20
  max_messages: 30
  required_categories:
    - pedido
    - cobranca
    - recusa
    - agradecimento
    - convite
    - esclarecimento
    - confirmacao
    - adiamento

generation_defaults:
  temperature: 0.6
  top_p: 0.95
  max_tokens: 220
  seed: 42

perplexity:
  enabled: false
  evaluator_model: "pierreguillou/gpt2-small-portuguese"
  max_length: 256

models:
  - model_id: "fake_model_a"
    provider: "fake"
    model_name: "fake-diverse-a"
    is_local: true
    generation_params:
      seed: 101

conditions:
  - condition_id: "vs_intermediate"
    condition_type: "verbalized_sampling"
    prompt_id: "vs_intermediate"
    prompt_version: "v3"
    prompt_path: "data/prompts/vs_intermediate_v3.txt"
    system_message: "Voce e um assistente de escrita cuidadoso. Retorne um objeto JSON que siga o schema solicitado."
    n_candidates: 5
    vs_probability_mode: "full_distribution"
    generation_params:
      temperature: 0.6
      top_p: 0.95
      max_tokens: 700
""".strip(),
        encoding="utf-8",
    )

    completed_run_id = run_experiment(
        config_path=config_path,
        project_paths=paths,
        run_id=run_id,
        only_model="fake_model_a",
        only_condition="vs_intermediate",
    )

    generations = read_jsonl(paths.run_dir(completed_run_id) / "generations.jsonl")

    assert completed_run_id == run_id
    assert len(generations) == 24 * 5
    assert all(record["status"] == "success" for record in generations)
