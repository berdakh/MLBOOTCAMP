"""Load the run configuration from a TOML file (standard library only)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config as a nested dict. Fails loudly on a missing file."""
    path = Path(path) if path else DEFAULT_CONFIG
    with path.open("rb") as fh:
        return tomllib.load(fh)
