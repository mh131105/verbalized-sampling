from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from tp1_nlp.utils.time import utc_now_iso


def git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return None
    commit = result.stdout.strip()
    if commit.startswith("fatal:"):
        return None
    return commit or None


def dependency_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "os": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependency_versions": dependency_versions(
            ["PyYAML", "requests", "matplotlib", "transformers", "torch"]
        ),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)


def base_manifest(
    *,
    run_id: str,
    root: Path,
    dataset_path: Path,
    dataset_hash: str,
    config_path: Path,
    models: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "status": "running",
        "git_commit": git_commit(root),
        "dataset_id": dataset_path.name,
        "dataset_hash": dataset_hash,
        "config_path": str(config_path),
        "models_used": models,
        "conditions_experimental": conditions,
        "prompts_used": prompts,
        "environment": environment_snapshot(),
        "generated_files": paths,
        "warnings": [],
        "errors": [],
    }

