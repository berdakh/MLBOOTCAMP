"""A file-system model registry: versions, a 'current' pointer, and rollback.

Real registries (MLflow, SageMaker, Vertex) add a UI and permissions, but the
core idea fits in a page::

    registry/
        v1/  v2/  v3/          immutable bundles (see artifacts.py)
        CURRENT                text file containing e.g. "v3"
        history.jsonl          append-only log of every promotion/rollback

Deploying a model = writing its version into CURRENT. Rolling back = writing
the previous one. Because bundles are immutable and the pointer change is a
single atomic file replace, rollback is instant and cannot half-succeed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import ArtifactError, ModelMetadata, load_bundle, load_metadata, save_bundle


class Registry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- versions -------------------------------------------------------------
    def versions(self) -> list[str]:
        vs = [p.name for p in self.root.iterdir() if p.is_dir() and p.name.startswith("v")
              and p.name[1:].isdigit() and (p / "metadata.json").exists()]
        return sorted(vs, key=lambda v: int(v[1:]))

    def next_version(self) -> str:
        vs = self.versions()
        return f"v{int(vs[-1][1:]) + 1}" if vs else "v1"

    def register(self, model, metadata: ModelMetadata) -> str:
        """Save a new immutable version. Does NOT deploy it."""
        version = self.next_version()
        metadata.version = version
        save_bundle(model, metadata, self.root / version)
        self._log("register", version)
        return version

    def metadata(self, version: str) -> dict:
        return load_metadata(self.root / version)

    # -- the pointer ----------------------------------------------------------
    @property
    def _pointer(self) -> Path:
        return self.root / "CURRENT"

    def current(self) -> str | None:
        return self._pointer.read_text().strip() if self._pointer.exists() else None

    def promote(self, version: str, reason: str = "") -> None:
        if version not in self.versions():
            raise ArtifactError(f"unknown version {version}; have {self.versions()}")
        previous = self.current()
        tmp = self._pointer.with_suffix(".tmp")
        tmp.write_text(version + "\n")
        os.replace(tmp, self._pointer)  # atomic on POSIX and Windows
        self._log("promote", version, previous=previous, reason=reason)

    def rollback(self, reason: str = "") -> str:
        """Point CURRENT back at the version that was live before this one."""
        promotions = [e for e in self.history() if e["event"] in ("promote", "rollback")]
        if len(promotions) < 2:
            raise ArtifactError("nothing to roll back to")
        target = promotions[-1].get("previous")
        if not target:
            raise ArtifactError("no previous version recorded")
        previous = self.current()
        tmp = self._pointer.with_suffix(".tmp")
        tmp.write_text(target + "\n")
        os.replace(tmp, self._pointer)
        self._log("rollback", target, previous=previous, reason=reason)
        return target

    def load_current(self, strict_versions: bool = False):
        version = self.current()
        if version is None:
            raise ArtifactError(f"no CURRENT pointer in {self.root}; promote a version first")
        return load_bundle(self.root / version, strict_versions=strict_versions)

    # -- audit log ------------------------------------------------------------
    def _log(self, event: str, version: str, **extra) -> None:
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "event": event, "version": version, **extra}
        with open(self.root / "history.jsonl", "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def history(self) -> list[dict]:
        path = self.root / "history.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
