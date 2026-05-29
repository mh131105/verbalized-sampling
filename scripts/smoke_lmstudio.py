from __future__ import annotations

import argparse
from datetime import datetime

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.generation.runner import run_experiment
from tp1_nlp.storage.paths import ProjectPaths


MODEL_ALIASES = {
    "gemma": "lmstudio_gemma_e2b",
    "ministral": "lmstudio_ministral_14b",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small LM Studio smoke test.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ALIASES))
    parser.add_argument("--condition", default="baseline")
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    model_id = MODEL_ALIASES[args.model]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"lmstudio_smoke_{args.model}_{args.condition}_{timestamp}"
    completed = run_experiment(
        config_path=paths.resolve("configs/experiment.lmstudio.yaml"),
        project_paths=paths,
        run_id=run_id,
        only_model=model_id,
        only_condition=args.condition,
    )
    print(f"Run completed: {completed}")
    print(f"Artifacts: {paths.run_dir(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
