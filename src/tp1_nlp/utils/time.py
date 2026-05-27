from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def make_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

