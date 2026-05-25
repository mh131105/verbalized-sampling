from __future__ import annotations

from uuid import uuid4

from tp1_nlp.generation.runner import run_experiment
from tp1_nlp.storage.jsonl import read_jsonl
from tp1_nlp.storage.paths import ProjectPaths


def test_fake_runner_writes_generations() -> None:
    paths = ProjectPaths.discover()
    run_id = f"pytest_{uuid4().hex}"

    completed_run_id = run_experiment(
        config_path=paths.resolve("configs/experiment.yaml"),
        project_paths=paths,
        run_id=run_id,
        only_model="fake_model_a",
        only_condition="vs_intermediate",
    )

    generations = read_jsonl(paths.run_dir(completed_run_id) / "generations.jsonl")

    assert completed_run_id == run_id
    assert len(generations) == 24 * 5
    assert all(record["status"] == "success" for record in generations)

