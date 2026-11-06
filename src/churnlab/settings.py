"""Configuration from environment variables - twelve-factor style.

Code reads settings from ONE object built at start-up. The same container image
then runs in dev, staging and production with different environment variables,
and nothing environment-specific is ever hard-coded or committed.

Secrets (database passwords, API keys) would come in the same way, from the
platform's secret store - never from a file in the repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    registry_dir: Path = Path("artifacts/registry")
    model_version: str | None = None  # pin a version; None = follow CURRENT
    max_batch_size: int = 1000
    log_level: str = "INFO"
    threshold_override: float | None = None
    service_name: str = "churn-api"

    @classmethod
    def from_env(cls, prefix: str = "CHURN_", environ: dict | None = None) -> Settings:
        """Build settings from ``CHURN_*`` variables, converting types explicitly."""
        env = os.environ if environ is None else environ
        values = {}
        for f in fields(cls):
            raw = env.get(prefix + f.name.upper())
            if raw is None or raw == "":
                continue
            if f.name == "registry_dir":
                values[f.name] = Path(raw)
            elif f.name == "max_batch_size":
                values[f.name] = int(raw)
            elif f.name == "threshold_override":
                value = float(raw)
                if not 0 < value < 1:
                    raise ValueError(f"{prefix}THRESHOLD_OVERRIDE must be in (0, 1), got {value}")
                values[f.name] = value
            elif f.name == "log_level":
                if raw.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
                    raise ValueError(f"bad {prefix}LOG_LEVEL {raw!r}")
                values[f.name] = raw.upper()
            else:
                values[f.name] = raw
        return cls(**values)
