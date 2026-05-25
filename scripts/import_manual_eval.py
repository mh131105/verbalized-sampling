from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.metrics.manual import import_manual_eval
from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Import filled manual evaluation CSV.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    output = import_manual_eval(paths.run_dir(args.run_id), Path(args.file))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

