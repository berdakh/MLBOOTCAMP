"""Reference solutions for Level 3 notebook 15. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def ex1(y, s, subgroup):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    g = np.asarray(subgroup, dtype=bool)
    bpsn = (g & (y == 0)) | (~g & (y == 1))  # background positives + subgroup negatives
    bnsp = (g & (y == 1)) | (~g & (y == 0))  # background negatives + subgroup positives
    return float(roc_auc_score(y[bpsn], s[bpsn])), float(roc_auc_score(y[bnsp], s[bnsp]))


def ex2(per_example_grads, clip, sigma, rng):
    g = np.asarray(per_example_grads, dtype=float)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    clipped = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))
    noisy_sum = clipped.sum(axis=0) + rng.normal(0.0, sigma * clip, size=g.shape[1])
    return noisy_sum / len(g)
