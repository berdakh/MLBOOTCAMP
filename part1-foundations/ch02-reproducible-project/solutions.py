"""Reference solutions for notebook 01. Graded by tests/test_solutions.py."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import sklearn


def ex1(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def ex2(directory: Path, config: dict, metrics: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = ex1(config)
    record = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    path = directory / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
