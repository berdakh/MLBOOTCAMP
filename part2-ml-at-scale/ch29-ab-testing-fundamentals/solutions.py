"""Reference solutions for notebook 11. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def ex1(p0, mde_abs, alpha=0.05, power=0.8):
    p1 = p0 + mde_abs
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(math.ceil(z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / mde_abs**2))


def ex2(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    return float(stats.chisquare([n_a, n_b], f_exp=expected).pvalue)


def ex3(n_experiments, n_users_per_arm, n_looks, alpha=0.05, p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    per_look = n_users_per_arm // n_looks
    conv = rng.binomial(per_look, p, size=(n_experiments, 2, n_looks)).cumsum(axis=2)
    n = per_look * np.arange(1, n_looks + 1)
    pooled = (conv[:, 0] + conv[:, 1]) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (conv[:, 1] - conv[:, 0]) / n / se
    pvals = 2 * stats.norm.sf(np.abs(np.nan_to_num(z)))
    return float((pvals < alpha).any(axis=1).mean())
