from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.reporting.examples import export_response_examples
from tp1_nlp.reporting.plots import make_plots_for_run
from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate comparison plots for a run.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    run_dir = paths.run_dir(args.run_id)
    figures = make_plots_for_run(run_dir)
    examples = export_response_examples(run_dir)
    for figure in figures:
        print(f"Wrote {figure}")
    print(f"Wrote {examples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

