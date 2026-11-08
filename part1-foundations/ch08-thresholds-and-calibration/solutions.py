"""Reference solutions for notebook 07. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(df: pd.DataFrame, scores, k: int) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=float), index=df.index)
    rank = s.groupby(df["snapshot_date"]).rank(ascending=False, method="first")
    return (rank <= k).to_numpy()


def ex2(y, scores, value_tp: float, cost_fp: float) -> tuple[float, float]:
    """Exact search: sort once, then cumulative TP/FP counts at every cut."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    # Only cut where the score changes, so "flag = score >= t" is well defined.
    last_of_run = np.r_[s[1:] != s[:-1], True]
    value = tp * value_tp - fp * cost_fp
    value = np.where(last_of_run, value, -np.inf)
    i = int(np.argmax(value))
    return float(s[i]), float(value[i])
