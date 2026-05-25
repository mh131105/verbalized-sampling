from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.generation.runner import run_experiment
from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TP1 NLP experiment.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--only-model", default=None)
    parser.add_argument("--only-condition", default=None)
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    run_id = run_experiment(
        config_path=paths.resolve(args.config),
        project_paths=paths,
        run_id=args.run_id,
        only_model=args.only_model,
        only_condition=args.only_condition,
    )
    print(f"Run completed: {run_id}")
    print(f"Artifacts: {paths.run_dir(run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

