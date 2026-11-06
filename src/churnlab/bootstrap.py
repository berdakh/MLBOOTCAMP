"""Catch-up functions: rebuild any earlier day's outputs in seconds.

Every notebook starts by calling one of these, so a notebook works in a fresh
Colab runtime even if you never ran the previous one - or ran it in a session
that has since been recycled. Each function is idempotent: it reuses what is
already on disk and only builds what is missing.

    ensure_raw()        data/raw/*.csv.gz present and checksum-verified
    load_clean()        raw tables, cleaned (Day 1, notebook 02)
    ensure_features()   the (snapshot, customer) feature table (Day 1, notebook 03)
    ensure_split()      the fixed train/valid/test time split (Day 1, notebook 04)
    ensure_registry()   a registry with a baseline (v1) and the chosen model (v2)
                        promoted to CURRENT (Day 2, notebook 08)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data as data_mod
from .artifacts import ModelMetadata, dataframe_fingerprint
from .colab import artifact_dir, find_root
from .evaluation import best_threshold_by_value, classification_report
from .features import CATEGORICAL, FEATURES, LABEL, NUMERIC, build_features
from .modeling import make_model, predict_proba
from .registry import Registry
from .splits import Split, time_split
from .validate import clean

#: The cost model used for thresholds throughout the course. These are
#: ASSUMPTIONS for teaching, stated once, not facts about any business:
#: a retention call costs $30; a retained customer is worth $600 of margin,
#: and a call retains ~25% of the would-be churners it reaches, so a true
#: positive is worth 0.25 * 600 = $150 in expectation. For a calibrated score p,
#: calling has positive expected value when p * 150 > (1 - p) * 30, i.e. p > 1/6.
#: Change the assumptions and the threshold must change too.
VALUE_TP = 150.0
COST_FP = 30.0


def raw_dir() -> Path:
    return find_root() / "data" / "raw"


def ensure_raw(verbose: bool = True) -> Path:
    """Make sure the raw extract exists and matches its checksums."""
    directory = raw_dir()
    status = data_mod.verify_raw(directory)
    for name, ok in status.items():
        if ok:
            continue
        url = data_mod.RAW_URL.format(name=name)
        if verbose:
            print(f"fetching {name} from {url}")
        data_mod.fetch(url, directory / f"{name}.csv.gz", sha256=data_mod.RAW_SHA256[name])
    if verbose:
        print(f"raw data verified in {directory}")
    return directory


def load_clean() -> dict[str, pd.DataFrame]:
    return clean(data_mod.load_raw(ensure_raw(verbose=False)))


def ensure_features(rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    path = artifact_dir() / "features.parquet"
    if path.exists() and not rebuild:
        df = pd.read_parquet(path)
        if verbose:
            print(f"loaded {len(df):,} feature rows from {path}")
        return df
    df = build_features(load_clean())
    df.to_parquet(path, index=False)
    if verbose:
        print(f"built {len(df):,} feature rows -> {path}")
    return df


def ensure_split(verbose: bool = False) -> Split:
    split = time_split(ensure_features(verbose=verbose))
    return split


def train_and_register(registry: Registry, name: str, split: Split, notes: str = "", **params) -> str:
    """Fit on train, choose a threshold on validation, register (not promote)."""
    model = make_model(name, **params)
    model.fit(split.train[FEATURES], split.train[LABEL])
    valid_scores = predict_proba(model, split.valid)
    threshold, _ = best_threshold_by_value(split.valid[LABEL], valid_scores, VALUE_TP, COST_FP)
    meta = ModelMetadata(
        model_name=name,
        version="",
        features=FEATURES,
        categorical=CATEGORICAL,
        threshold=round(threshold, 4),
        training_data={
            "fingerprint": dataframe_fingerprint(split.train[FEATURES + [LABEL]]),
            "rows": len(split.train),
            "first_snapshot": str(split.train["snapshot_date"].min().date()),
            "last_snapshot": str(split.train["snapshot_date"].max().date()),
        },
        metrics={"validation": classification_report(split.valid[LABEL], valid_scores)},
        params={k: v for k, v in params.items()},
        notes=notes,
    )
    return registry.register(model, meta)


def ensure_registry(verbose: bool = True) -> Registry:
    """A registry with v1 = logistic regression, v2 = boosted trees.

    Both are trained on the training period; the one with the higher
    *validation* PR-AUC is promoted to CURRENT. (On this data that is the
    logistic regression - see notebook 06 for why that is not a surprise.)
    """
    registry = Registry(artifact_dir() / "registry")
    if registry.current() is not None:
        if verbose:
            print(f"registry at {registry.root}: versions {registry.versions()}, CURRENT={registry.current()}")
        return registry
    split = ensure_split()
    v1 = train_and_register(registry, "logreg", split, notes="linear baseline with log/scale preprocessing")
    v2 = train_and_register(registry, "hgb", split, notes="gradient-boosted trees challenger")
    pr = {v: registry.metadata(v)["metrics"]["validation"]["pr_auc"] for v in (v1, v2)}
    best = max(pr, key=pr.get)
    registry.promote(best, reason=f"highest validation PR-AUC ({pr[best]:.3f} vs {min(pr.values()):.3f})")
    if verbose:
        print(f"built registry at {registry.root}: versions {registry.versions()}, CURRENT={registry.current()}")
    return registry


__all__ = [
    "COST_FP", "VALUE_TP", "ensure_raw", "load_clean", "ensure_features", "ensure_split",
    "ensure_registry", "train_and_register", "NUMERIC", "CATEGORICAL", "FEATURES", "LABEL",
]
