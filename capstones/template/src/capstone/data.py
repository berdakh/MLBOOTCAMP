"""Getting the raw data, checking it, and splitting it.

Rules this module enforces (keep them when you fill in the TODOs):
* raw files are downloaded once, checksummed, and never edited by hand;
* every cleaning decision is a line of code with a comment saying why;
* the split happens BEFORE any statistic is fitted (scalers, encoders, means).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["entity_id", "timestamp", "x1", "x2", "target"]  # TODO: your schema


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: str | Path, expected_sha256: str = "") -> Path:
    """Download `url` to `dest` once; verify the checksum if one is pinned."""
    # TODO: implement with urllib.request or httpx (timeout!), then pin the
    # SHA-256 in configs/default.toml after the first successful download.
    raise NotImplementedError("TODO: download the dataset named in DATA_CARD.md")


def load_raw(config: dict) -> pd.DataFrame:
    """Read the raw files into one DataFrame with the columns in REQUIRED_COLUMNS."""
    # TODO: dataset-specific parsing (time zones, sentinel values, column names).
    raise NotImplementedError("TODO: parse the raw files for your dataset")


def make_toy_data(n_entities: int = 20, n_steps: int = 50, seed: int = 0) -> pd.DataFrame:
    """A tiny synthetic frame with the same schema - used by the tests only."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2019-11-01")
    for entity in range(n_entities):
        level = rng.normal(10, 2)
        for step in range(n_steps):
            x1, x2 = rng.normal(), rng.normal()
            rows.append({
                "entity_id": f"e{entity:03d}",
                "timestamp": start + pd.Timedelta(days=step),
                "x1": x1,
                "x2": x2,
                "target": level + 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.5),
            })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the data contract. Raise with a clear message rather than fix silently."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df.duplicated(["entity_id", "timestamp"]).any():
        raise ValueError("duplicate (entity_id, timestamp) rows")
    if df["target"].isna().any():
        raise ValueError("target has missing values")
    # TODO: ranges, units, allowed categories, monotone time per entity, ...
    return df


def time_split(df: pd.DataFrame, cutoff: str, time_column: str = "timestamp"):
    """Rows strictly before `cutoff` train; the rest test. No shuffling."""
    cut = pd.Timestamp(cutoff)
    train, test = df[df[time_column] < cut], df[df[time_column] >= cut]
    assert train[time_column].max() < test[time_column].min(), "time leak"
    return train.copy(), test.copy()


def group_split(df: pd.DataFrame, group_column: str = "entity_id",
                test_fraction: float = 0.2, seed: int = 0):
    """Whole entities (engines, users, subjects) go to exactly one side."""
    groups = np.array(sorted(df[group_column].unique()))
    rng = np.random.default_rng(seed)
    n_test = max(1, int(round(test_fraction * len(groups))))
    test_groups = set(rng.choice(groups, size=n_test, replace=False))
    mask = df[group_column].isin(test_groups)
    train, test = df[~mask], df[mask]
    assert not set(train[group_column]) & set(test[group_column]), "entity leak"
    return train.copy(), test.copy()


def split(df: pd.DataFrame, config: dict):
    s = config["split"]
    if s["kind"] == "time":
        return time_split(df, s["cutoff"], s["time_column"])
    if s["kind"] == "group":
        return group_split(df, s["group_column"], s["test_fraction"], config["project"]["seed"])
    raise ValueError(f"unknown split kind: {s['kind']}")
