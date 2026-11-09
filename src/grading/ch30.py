"""Self-checks for Chapter 30 · Advanced experimentation."""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check


def ex1(cuped_adjust):
    def test():
        rng = np.random.default_rng(0)
        x = rng.poisson(5, 4000).astype(float)
        y = 0.8 * x + rng.normal(0, 1.5, 4000)
        got = np.asarray(cuped_adjust(y, x), dtype=float)
        assert got.shape == y.shape, f"expected an array of shape {y.shape}, got {got.shape}"
        theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
        want = y - theta * (x - x.mean())
        assert np.allclose(got, want, atol=1e-8), (
            f"values differ from y - theta (x - mean x) with theta = {theta:.4f} "
            "(same ddof in cov and var; subtract the mean of x)")
        assert abs(got.mean() - y.mean()) < 1e-9, "the adjusted metric must keep the mean of y"
        assert got.var() < 0.5 * y.var(), "variance should drop substantially for a strongly correlated x"
        const = np.asarray(cuped_adjust(y[:10], np.ones(10)), dtype=float)
        assert np.allclose(const, y[:10]), "with a constant x, return y unchanged (theta = 0, not NaN)"

    return run_check("Exercise 1 · cuped_adjust", test,
                     hint="theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1); return y - theta * (x - x.mean())")


def _formula(c, n):
    mc, mn = c.mean(), n.mean()
    cov = np.cov(c, n)
    return (cov[0, 0] / mn**2 - 2 * mc * cov[0, 1] / mn**3 + mc**2 * cov[1, 1] / mn**4) / len(c)


def ex2(delta_method_ratio_var):
    def test():
        rng = np.random.default_rng(1)
        n = 10 * np.maximum(rng.poisson(rng.lognormal(1.5, 0.8, 3000)), 1).astype(float)
        p = rng.beta(2, 12, 3000)
        c = rng.binomial(n.astype(int), p).astype(float)
        got = float(delta_method_ratio_var(c, n))
        want = _formula(c, n)
        assert got > 0, f"a variance must be positive, got {got}"
        assert abs(got - want) <= 0.01 * want, f"expected {want:.3e} (delta method, ddof=1), got {got:.3e}"
        idx = rng.integers(0, len(c), (1000, len(c)))
        boot = (c[idx].sum(1) / n[idx].sum(1)).var(ddof=1)
        assert abs(got - boot) <= 0.15 * boot, f"far from the bootstrap variance {boot:.3e}: got {got:.3e}"
        naive = (c.sum() / n.sum()) * (1 - c.sum() / n.sum()) / n.sum()
        assert got > 2 * naive, "suspiciously close to the binomial variance: did you treat impressions as independent?"

    return run_check("Exercise 2 · delta_method_ratio_var", test,
                     hint="(var_c / mn**2 - 2 * mc * cov_cn / mn**3 + mc**2 * var_n / mn**4) / N, with ddof=1")


def ex3(team_draft_interleave):
    def test():
        a = [1, 2, 3, 4, 5, 6, 7, 8]
        b = [3, 9, 1, 10, 11, 2, 12, 13]
        first_a = 0
        for seed in range(400):
            items, teams = team_draft_interleave(a, b, np.random.default_rng(seed))
            items, teams = list(items), list(teams)
            assert len(items) == len(a) == len(teams), f"expected {len(a)} items and teams, got {len(items)}/{len(teams)}"
            assert len(set(items)) == len(items), f"duplicates in {items}"
            assert set(teams) <= {"A", "B"}, f"teams must be 'A' or 'B', got {set(teams)}"
            for i, (item, team) in enumerate(zip(items, teams)):
                own = a if team == "A" else b
                best = next(x for x in own if x not in items[:i])
                assert item == best, (f"position {i}: team {team} should pick its highest-ranked unused item {best}, "
                                      f"got {item} (list {items}, teams {teams})")
                n_a, n_b = teams[:i + 1].count("A"), teams[:i + 1].count("B")
                assert abs(n_a - n_b) <= 1, f"teams unbalanced after {i + 1} picks: {teams}"
                if i % 2 == 1:
                    assert n_a == n_b, f"each round gives one pick per team; after {i + 1} picks got {teams}"
            first_a += teams[0] == "A"
        assert 160 <= first_a <= 240, f"A picked first in {first_a}/400 runs: the coin flip should be fair"
        same1 = team_draft_interleave(a, b, np.random.default_rng(7))
        same2 = team_draft_interleave(a, b, np.random.default_rng(7))
        assert list(same1[0]) == list(same2[0]) and list(same1[1]) == list(same2[1]), "use the given rng only"
        items, _ = team_draft_interleave(a, a, np.random.default_rng(0))
        assert list(items) == a, "with identical rankings the interleaved list must equal that ranking"

    return run_check("Exercise 3 · team_draft_interleave", test,
                     hint="loop until k items: pick team = fewer picks so far (coin on ties); that team adds its "
                          "first item not yet shown; append the team label")
