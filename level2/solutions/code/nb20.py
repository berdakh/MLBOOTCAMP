"""Reference solutions for Level 2 notebook 20 (graded in CI)."""

from __future__ import annotations

import numpy as np


def ex1(values, groups, n_boot, seed):
    v = np.asarray(values, dtype=float)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(g, weights=v)
    counts = np.bincount(g).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return float(v.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
