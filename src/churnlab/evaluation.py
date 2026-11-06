"""Evaluation you can defend in an interview.

Every function returns plain numbers or DataFrames so results can be printed,
asserted on in tests, and written into a model card.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def precision_at_k(y_true, scores, k: int) -> float:
    """Precision among the top-k scored rows - "if we can call k customers"."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return float(y_true[order].mean()) if k > 0 else float("nan")


def lift_at_fraction(y_true, scores, fraction: float = 0.1) -> float:
    """How many times the base rate the top ``fraction`` of rows achieves."""
    y_true = np.asarray(y_true)
    k = max(1, int(round(fraction * len(y_true))))
    base = y_true.mean()
    return precision_at_k(y_true, scores, k) / base if base > 0 else float("nan")


def classification_report(y_true, scores) -> dict[str, float]:
    """The headline metrics for an imbalanced binary problem."""
    y_true = np.asarray(y_true)
    scores = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores)),
        "lift@10%": float(lift_at_fraction(y_true, scores, 0.10)),
    }


def bootstrap_ci(
    y_true,
    scores,
    metric: Callable = average_precision_score,
    n_boot: int = 500,
    seed: int = 0,
    groups=None,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile (1-alpha) CI for ``metric``.

    Pass ``groups`` (e.g. customer ids) to resample whole customers instead of
    rows. Rows of one customer are correlated, so a row bootstrap understates
    the uncertainty - the cluster bootstrap is the honest one here.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, scores))
    stats = []
    if groups is None:
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if y_true[idx].min() == y_true[idx].max():
                continue
            stats.append(metric(y_true[idx], scores[idx]))
    else:
        groups = np.asarray(groups)
        uniq, inverse = np.unique(groups, return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([members[g] for g in pick])
            if y_true[idx].min() == y_true[idx].max():
                continue
            stats.append(metric(y_true[idx], scores[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(y_true, scores_a, scores_b, metric: Callable = average_precision_score,
                          n_boot: int = 500, seed: int = 0, groups=None) -> dict[str, float]:
    """CI for metric(B) - metric(A) on the SAME resamples: is B really better?"""
    y_true, a, b = (np.asarray(x) for x in (y_true, scores_a, scores_b))
    rng = np.random.default_rng(seed)
    if groups is None:
        members = None
        n = len(y_true)
    else:
        uniq, inverse = np.unique(np.asarray(groups), return_inverse=True)
        members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    diffs = []
    for _ in range(n_boot):
        if members is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(metric(y_true[idx], b[idx]) - metric(y_true[idx], a[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff": float(metric(y_true, b) - metric(y_true, a)),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_b_not_better": float((diffs <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def confusion_at(y_true, scores, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) >= threshold
    return {
        "tp": int((pred & y_true).sum()),
        "fp": int((pred & ~y_true).sum()),
        "fn": int((~pred & y_true).sum()),
        "tn": int((~pred & ~y_true).sum()),
    }


def threshold_table(y_true, scores, thresholds=None) -> pd.DataFrame:
    """Precision, recall and alert volume at each candidate threshold."""
    thresholds = np.linspace(0.02, 0.6, 30) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        c = confusion_at(y_true, scores, t)
        flagged = c["tp"] + c["fp"]
        rows.append({
            "threshold": round(float(t), 4),
            "flagged": flagged,
            "precision": c["tp"] / flagged if flagged else float("nan"),
            "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
            **c,
        })
    return pd.DataFrame(rows)


def expected_value(conf: dict[str, int], value_tp: float, cost_fp: float, cost_fn: float = 0.0) -> float:
    """Net value of a decision rule under a stated (assumed!) cost model."""
    return conf["tp"] * value_tp - conf["fp"] * cost_fp - conf["fn"] * cost_fn


def best_threshold_by_value(y_true, scores, value_tp: float, cost_fp: float,
                            cost_fn: float = 0.0) -> tuple[float, float]:
    """Threshold maximising :func:`expected_value`. Returns (threshold, value)."""
    candidates = np.unique(np.round(np.asarray(scores), 4))
    if len(candidates) > 400:
        candidates = np.quantile(scores, np.linspace(0, 1, 401))
    best = (0.5, -np.inf)
    for t in candidates:
        v = expected_value(confusion_at(y_true, scores, t), value_tp, cost_fp, cost_fn)
        if v > best[1]:
            best = (float(t), float(v))
    return best


def threshold_for_precision(y_true, scores, min_precision: float) -> float | None:
    """Lowest threshold whose precision is at least ``min_precision`` (max recall)."""
    precision, _, thresholds = precision_recall_curve(y_true, scores)
    ok = np.flatnonzero(precision[:-1] >= min_precision)
    return float(thresholds[ok[0]]) if len(ok) else None


def calibration_table(y_true, scores, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed rate in quantile bins of the score."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(scores)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                               observed_rate=("y", "mean"))
    return out.reset_index(drop=True)


def slice_report(df: pd.DataFrame, scores, by: str, label: str = "churned_30d",
                 threshold: float | None = None, min_rows: int = 50) -> pd.DataFrame:
    """Metrics per value of ``by``; small slices are flagged rather than hidden."""
    data = df.assign(_score=np.asarray(scores))
    rows = []
    for value, g in data.groupby(by, observed=True):
        row = {by: value, "rows": len(g), "positive_rate": g[label].mean()}
        if g[label].nunique() == 2:
            row["roc_auc"] = roc_auc_score(g[label], g["_score"])
            row["pr_auc"] = average_precision_score(g[label], g["_score"])
        if threshold is not None:
            c = confusion_at(g[label], g["_score"], threshold)
            row["recall"] = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else np.nan
            row["flag_rate"] = (c["tp"] + c["fp"]) / len(g)
        row["small_slice"] = len(g) < min_rows
        rows.append(row)
    return pd.DataFrame(rows)
