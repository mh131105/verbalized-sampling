from __future__ import annotations

import argparse

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.metrics.manual import export_manual_eval_template
from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Export manual evaluation template.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    output = export_manual_eval_template(paths.run_dir(args.run_id))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

