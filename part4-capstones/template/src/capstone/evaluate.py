"""Metrics, with uncertainty. Report the split name next to every number."""

from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wape(y, yhat) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y|. Safe with zeros in y."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def bootstrap_ci(y, yhat, metric=mae, n: int = 500, alpha: float = 0.05, seed: int = 0,
                 groups=None) -> tuple[float, float]:
    """Percentile bootstrap CI. Pass `groups` to resample whole entities/days,
    which is the honest choice when rows within a group are correlated."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    rng = np.random.default_rng(seed)
    stats = []
    if groups is None:
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            stats.append(metric(y[idx], yhat[idx]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        index = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([index[g] for g in pick])
            stats.append(metric(y[idx], yhat[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def evaluate(y, yhat) -> dict[str, float]:
    """TODO: add the capstone-specific metrics from your brief
    (peak-hour MAE, NASA score, Recall@10, PR-AUC, subgroup FPR/FNR ...)."""
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "wape": wape(y, yhat), "n": int(len(y))}
