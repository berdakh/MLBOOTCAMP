"""Reference solutions for Level 2 notebook 05 (graded in CI by tests/test_solutions.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(random_imps: pd.DataFrame) -> np.ndarray:
    ctr = random_imps.groupby("position")["clicked"].mean().sort_index()
    return (ctr / ctr.iloc[0]).to_numpy()


def ex2(clicks, positions, propensities, clip=None) -> float:
    w = 1.0 / np.asarray(propensities, dtype=float)[np.asarray(positions)]
    if clip is not None:
        w = np.minimum(w, clip)
    return float(np.mean(np.asarray(clicks, dtype=float) * w))
