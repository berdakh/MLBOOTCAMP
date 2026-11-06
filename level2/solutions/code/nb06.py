"""Reference solutions for Level 2 notebook 06 (graded in CI by tests/test_solutions.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(scores, labels, groups) -> float:
    s, y, g = (np.asarray(x) for x in (scores, labels, groups))
    good, total = 0.0, 0
    for key in np.unique(g):
        m = g == key
        sg, yg = s[m], y[m]
        pairs = yg[:, None] > yg[None, :]
        d = (sg[:, None] - sg[None, :])[pairs]
        total += int(pairs.sum())
        good += float((d > 0).sum() + 0.5 * (d == 0).sum())
    return good / total if total else float("nan")


def ex2(df: pd.DataFrame, group_col: str):
    out = df.sort_values(group_col, kind="stable")
    sizes = out.groupby(group_col, sort=False).size().to_numpy()
    return out, sizes
