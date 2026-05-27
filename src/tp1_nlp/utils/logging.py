from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tp1_nlp.utils.time import utc_now_iso


def append_event(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": utc_now_iso(), "event": event, **payload}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

