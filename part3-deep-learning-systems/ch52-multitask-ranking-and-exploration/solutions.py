"""Reference solutions for Level 3 notebook 14. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np


def ex1(bids, pctrs, reserve):
    bids = np.asarray(bids, dtype=float)
    pctrs = np.asarray(pctrs, dtype=float)
    eligible = np.flatnonzero(bids >= reserve)
    if len(eligible) == 0:
        return None, 0.0
    ecpm = bids[eligible] * pctrs[eligible]
    order = eligible[np.argsort(-ecpm, kind="stable")]  # stable: ties keep the lower index first
    winner = int(order[0])
    if len(order) == 1:
        return winner, float(reserve)
    runner = order[1]
    price = bids[runner] * pctrs[runner] / pctrs[winner]
    return winner, float(max(reserve, price))


def ex2(true_ps, n_rounds, seed):
    p = np.asarray(true_ps, dtype=float)
    rng = np.random.default_rng(seed)
    alpha, beta = np.ones(len(p)), np.ones(len(p))
    regret = np.empty(n_rounds)
    total, best = 0.0, p.max()
    for t in range(n_rounds):
        arm = int(np.argmax(rng.beta(alpha, beta)))
        reward = rng.random() < p[arm]
        alpha[arm] += reward
        beta[arm] += 1 - reward
        total += best - p[arm]
        regret[t] = total
    return regret
