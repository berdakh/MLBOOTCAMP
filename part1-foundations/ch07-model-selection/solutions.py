"""Reference solutions for notebook 06. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.evaluation import paired_bootstrap_diff


def ex1(y, scores_a, scores_b, groups) -> str:
    d = paired_bootstrap_diff(y, scores_a, scores_b, groups=groups, n_boot=1000, seed=0)
    if d["ci_low"] > 0:
        return "B"
    if d["ci_high"] < 0:
        return "A"
    return "no clear winner"


def ex2(param_grid_list, evaluate_fn, budget: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = min(budget, len(param_grid_list))
    picks = rng.choice(len(param_grid_list), size=n, replace=False)
    rows = []
    for i in picks:
        params = dict(param_grid_list[int(i)])
        rows.append({**params, "score": float(evaluate_fn(params))})
    trials = pd.DataFrame(rows)
    best_row = trials["score"].idxmax()
    best = {k: param_grid_list[int(picks[best_row])][k] for k in param_grid_list[int(picks[best_row])]}
    return best, trials
