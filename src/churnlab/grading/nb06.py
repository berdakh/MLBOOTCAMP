"""Self-checks for notebook 06 · Model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import run_check


def _synthetic(seed: int = 7, n_groups: int = 300, rows_per_group: int = 4):
    """Clustered rows: each 'customer' has a few correlated rows."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    risk = rng.normal(size=n_groups)[groups] + rng.normal(scale=0.5, size=len(groups))
    y = (risk + rng.normal(scale=1.0, size=len(groups)) > 1.6).astype(int)
    return rng, y, groups, risk


def decide_cases():
    """(y, scores_a, scores_b, groups, expected) - comfortably separated or clearly tied."""
    rng, y, groups, risk = _synthetic()
    good = risk + rng.normal(scale=0.3, size=len(y))
    weak = rng.normal(size=len(y))  # no signal at all
    # Two equally good models (independent noise of the same size); the seed is
    # chosen so the observed PR-AUC difference is ~0, far inside any sensible CI.
    tie_rng = np.random.default_rng(101)
    tie_a = risk + tie_rng.normal(scale=0.8, size=len(y))
    tie_b = risk + tie_rng.normal(scale=0.8, size=len(y))
    return [
        (y, good, weak, groups, "A"),
        (y, weak, good, groups, "B"),
        (y, tie_a, tie_b, groups, "no clear winner"),
    ]


def ex1(decide):
    def test():
        for k, (y, a, b, g, expected) in enumerate(decide_cases()):
            got = decide(y, a, b, g)
            assert got in ("A", "B", "no clear winner"), f"return 'A', 'B' or 'no clear winner', got {got!r}"
            assert got == expected, f"case {k}: expected {expected!r}, got {got!r}"

    return run_check("Exercise 1 · decide", test,
                     hint="paired_bootstrap_diff(y, a, b, groups=groups) gives a CI for metric(B) - metric(A); "
                          "B if ci_low > 0, A if ci_high < 0, otherwise no clear winner")


def ex2(budgeted_search):
    def test():
        grid = [{"depth": d, "lr": lr} for d in (2, 3, 4, 5, 6) for lr in (0.01, 0.03, 0.1, 0.3)]
        calls = []

        def evaluate(params):
            calls.append(dict(params))
            return -abs(params["depth"] - 4) - abs(np.log10(params["lr"]) + 1.5)

        best, trials = budgeted_search(grid, evaluate, 6)
        assert len(calls) <= 6, f"budget was 6 but evaluate_fn was called {len(calls)} times"
        assert len(calls) == 6, f"with 20 candidates and budget 6, use the whole budget (got {len(calls)} calls)"
        assert isinstance(trials, pd.DataFrame), "trials must be a DataFrame (one row per evaluated config)"
        assert len(trials) == len(calls), "trials must have one row per evaluated config"
        assert "score" in trials.columns, "trials needs a 'score' column"
        assert {"depth", "lr"} <= set(trials.columns), "trials needs one column per hyperparameter"
        keys = [tuple(sorted(c.items())) for c in calls]
        assert len(set(keys)) == len(keys), "the same config was evaluated twice - sample without replacement"
        best_seen = max(calls, key=lambda c: -abs(c["depth"] - 4) - abs(np.log10(c["lr"]) + 1.5))
        assert dict(best) == best_seen, f"best_params {best} is not the best evaluated config {best_seen}"
        assert np.isclose(trials["score"].max(), -abs(best_seen["depth"] - 4) - abs(np.log10(best_seen["lr"]) + 1.5))

        calls.clear()
        best, trials = budgeted_search(grid[:3], evaluate, 10)
        assert len(calls) == 3, "with fewer candidates than the budget, evaluate each candidate exactly once"
        assert len(trials) == 3

    return run_check("Exercise 2 · budgeted_search", test,
                     hint="rng.choice(len(grid), size=min(budget, len(grid)), replace=False); "
                          "collect {**params, 'score': s} rows into a DataFrame")
