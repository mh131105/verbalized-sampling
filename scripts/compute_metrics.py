from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.config.loader import load_experiment_config
from tp1_nlp.metrics.summary import compute_metrics_for_run
from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for a run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/experiment.lmstudio.yaml")
    parser.add_argument("--skip-perplexity", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    config = load_experiment_config(paths.resolve(args.config), paths)
    if args.skip_perplexity:
        object.__setattr__(config.perplexity, "enabled", False)

    metrics_long, metrics_summary = compute_metrics_for_run(
        run_dir=paths.run_dir(args.run_id),
        dataset_path=paths.resolve(config.dataset_path),
        perplexity=config.perplexity,
    )
    print(f"Wrote {metrics_long}")
    print(f"Wrote {metrics_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
