from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import yaml

from _bootstrap import bootstrap

ROOT = bootstrap()

from tp1_nlp.storage.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Check LM Studio models for TP1.")
    parser.add_argument("--config", default="configs/experiment.lmstudio.yaml")
    args = parser.parse_args()

    paths = ProjectPaths.discover(ROOT)
    config_path = paths.resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = [
        item
        for item in config.get("models", [])
        if item.get("backend") == "lmstudio"
    ]
    if not models:
        print(f"No LM Studio models found in {config_path}")
        return 1

    api_base = str(models[0].get("api_base") or "http://localhost:1234/v1").rstrip("/")
    try:
        with urlopen(f"{api_base}/models", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Could not reach LM Studio at {api_base}/models: {exc}")
        print("Start the LM Studio local server and load the expected models.")
        return 1

    available = {item["id"] for item in payload.get("data", [])}
    missing = []

    print("Configured LM Studio models:")
    for model in models:
        model_name = str(model["model_name"])
        status = "OK" if model_name in available else "MISSING"
        print(f"- {model['model_id']}: {status} {model_name}")
        if model_name not in available:
            missing.append(model_name)

    if missing:
        print("\nAvailable LM Studio models:")
        for name in sorted(available):
            print(f"- {name}")
        return 1

    print("\nLM Studio configuration is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

