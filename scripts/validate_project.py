from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.config.loader import load_experiment_config
from tp1_nlp.dataset.loader import load_messages
from tp1_nlp.dataset.validator import validate_messages
from tp1_nlp.storage.paths import ProjectPaths, validate_project_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TP1 project structure.")
    parser.add_argument("--config", default="configs/experiment.lmstudio.yaml")
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    config_path = paths.resolve(args.config)
    errors = validate_project_contract(paths)

    try:
        config = load_experiment_config(config_path, paths)
        messages = load_messages(paths.resolve(config.dataset_path))
        errors.extend(
            validate_messages(
                messages,
                min_messages=config.min_messages,
                max_messages=config.max_messages,
                required_categories=config.required_categories,
            )
        )
        for condition in config.conditions:
            if not paths.resolve(condition.prompt_path).exists():
                errors.append(f"Missing prompt: {condition.prompt_path}")
    except Exception as exc:
        errors.append(str(exc))

    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Project validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
