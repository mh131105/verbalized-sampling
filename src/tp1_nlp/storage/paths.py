from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        current = (start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            if (candidate / "scripts").exists() or (candidate / ".git").exists():
                return cls(candidate)
        return cls(current)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def ensure_base(self) -> None:
        for directory in [
            self.data_dir,
            self.scripts_dir,
            self.results_dir,
            self.runs_dir,
            self.configs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def validate_project_contract(paths: ProjectPaths) -> list[str]:
    errors: list[str] = []
    required_dirs = ["data", "scripts", "results", "results/runs"]
    required_files = ["README.md", "requirements.txt", "data/messages.jsonl"]
    for directory in required_dirs:
        if not (paths.root / directory).is_dir():
            errors.append(f"Missing required directory: {directory}")
    for file_path in required_files:
        if not (paths.root / file_path).is_file():
            errors.append(f"Missing required file: {file_path}")
    return errors

