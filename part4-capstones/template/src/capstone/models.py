"""A baseline you must beat, a model, and saving both with metadata."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MeanBaseline:
    """Predicts the training mean. Replace with the domain baseline from your brief
    (seasonal naive, last value, popularity, TF-IDF + logistic regression ...)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def build_model(config: dict):
    m = config["model"]
    if m["kind"] == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=m["alpha"]))])
    # TODO: gradient boosting, a PyTorch model wrapper, ...
    raise ValueError(f"unknown model kind: {m['kind']}")


@dataclass
class Metadata:
    name: str
    version: str
    features: list[str]
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    data_sha256: str = ""
    git_commit: str = ""          # TODO: fill from `git rev-parse HEAD` in the CLI
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    python: str = platform.python_version()
    sklearn: str = sklearn.__version__


def save_bundle(model, meta: Metadata, directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    (directory / "metadata.json").write_text(json.dumps(asdict(meta), indent=2, default=str))
    return directory


def load_bundle(directory: str | Path):
    """Only load bundles you created: joblib/pickle executes code on load."""
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    model = joblib.load(directory / "model.joblib")
    return model, meta


def predict_frame(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)
