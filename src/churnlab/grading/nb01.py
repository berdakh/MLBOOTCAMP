"""Self-checks for notebook 01 · A reproducible ML project."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from ..checks import run_check


def ex1(config_hash):
    def test():
        a = {"model": "logreg", "C": 1.0, "nested": {"x": 1, "y": [1, 2]}}
        b = {"nested": {"y": [1, 2], "x": 1}, "C": 1.0, "model": "logreg"}
        h = config_hash(a)
        assert isinstance(h, str) and len(h) == 12, f"expected a 12-character string, got {h!r}"
        assert all(c in "0123456789abcdef" for c in h), "expected lowercase hexadecimal characters"
        assert config_hash(b) == h, "same config with keys in a different order gave a different hash"
        assert config_hash({**a, "C": 0.5}) != h, "changing a value did not change the hash"
        assert config_hash({**a, "nested": {"x": 2, "y": [1, 2]}}) != h, "a nested change did not change the hash"

    return run_check("Exercise 1 · config_hash", test,
                     hint="json.dumps(config, sort_keys=True) sorts nested keys too; then hashlib.sha256(...).hexdigest()[:12]")


def ex2(record_run):
    def test():
        config = {"model": "logreg", "C": 1.0, "seed": 0}
        metrics = {"valid_pr_auc": 0.1}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(record_run(Path(tmp), config, metrics))
            assert path.exists(), f"{path} was not written"
            record = json.loads(path.read_text())
            for key in ("run_id", "config", "metrics", "created_at", "environment"):
                assert key in record, f"missing key {key!r}"
            assert path.name == f"{record['run_id']}.json", "file name must be <run_id>.json"
            assert len(record["run_id"]) == 12, "run_id should be the 12-character config hash"
            assert record["config"] == config and record["metrics"] == metrics
            datetime.fromisoformat(record["created_at"].replace("Z", "+00:00"))
            env = record["environment"]
            assert "python" in env and "sklearn" in env, "environment needs python and sklearn versions"
            again = Path(record_run(Path(tmp), dict(reversed(list(config.items()))), {"valid_pr_auc": 0.2}))
            assert again == path, "same config must map to the same file"
            assert len(list(Path(tmp).glob("*.json"))) == 1, "re-running created a second file"
            other = Path(record_run(Path(tmp), {**config, "C": 0.1}, metrics))
            assert other != path, "a different config must get a different file"

    return run_check("Exercise 2 · record_run", test,
                     hint="build the dict, then (directory / f'{run_id}.json').write_text(json.dumps(record))")
