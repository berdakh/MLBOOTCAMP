"""Self-checks for notebook 11 · A/B testing fundamentals."""

from __future__ import annotations

import math

from churnlab.checks import run_check


def _n_formula(p0, d, alpha=0.05, power=0.8):
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return math.ceil(z**2 * (p0 * (1 - p0) + (p0 + d) * (1 - p0 - d)) / d**2)


def ex1(sample_size_per_arm):
    def test():
        cases = [(0.10, 0.01, 0.05, 0.8), (0.05, 0.0025, 0.05, 0.8), (0.30, 0.02, 0.01, 0.9), (0.5, 0.05, 0.05, 0.8)]
        for p0, d, alpha, power in cases:
            want = _n_formula(p0, d, alpha, power)
            got = sample_size_per_arm(p0, d, alpha=alpha, power=power)
            assert isinstance(got, (int,)) or float(got).is_integer(), f"return an integer, got {got!r}"
            assert int(got) == want, (
                f"p0={p0}, mde={d}, alpha={alpha}, power={power}: expected {want:,}, got {int(got):,}")
        assert sample_size_per_arm(0.10, 0.01) == _n_formula(0.10, 0.01), "defaults should be alpha=0.05, power=0.8"

    return run_check("Exercise 1 · sample_size_per_arm", test,
                     hint="z = norm.ppf(1 - alpha/2) + norm.ppf(power); "
                          "n = ceil(z**2 * (p0(1-p0) + p1(1-p1)) / mde**2) with p1 = p0 + mde")


def ex2(srm_pvalue):
    def test():
        from scipy.stats import chisquare

        for n_a, n_b, r in [(50_000, 50_000, 0.5), (50_400, 49_600, 0.5), (10_100, 9_900, 0.5),
                            (30_500, 69_500, 0.3), (1_000, 1_100, 0.5)]:
            total = n_a + n_b
            want = chisquare([n_a, n_b], f_exp=[total * r, total * (1 - r)]).pvalue
            got = srm_pvalue(n_a, n_b, r)
            assert abs(float(got) - want) <= 1e-9 + 1e-6 * want, f"(n_a={n_a}, n_b={n_b}, ratio={r}): expected {want:.6g}, got {got}"
        assert abs(srm_pvalue(50_000, 50_000) - 1.0) < 1e-9, "a perfect 50/50 split should give p = 1 (default expected_ratio=0.5)"

    return run_check("Exercise 2 · srm_pvalue", test,
                     hint="scipy.stats.chisquare([n_a, n_b], f_exp=[N * r, N * (1 - r)]).pvalue")


def ex3(aa_false_positive_rate):
    def test():
        once = aa_false_positive_rate(4000, 2000, 1, alpha=0.05, p=0.1, seed=0)
        assert 0.0 <= float(once) <= 1.0, f"return a rate between 0 and 1, got {once}"
        assert 0.035 <= once <= 0.065, f"with a single look the A/A false-positive rate should be about 0.05, got {once:.3f}"
        again = aa_false_positive_rate(4000, 2000, 1, alpha=0.05, p=0.1, seed=0)
        assert again == once, "same seed must give the same result (use np.random.default_rng(seed))"
        peek = aa_false_positive_rate(4000, 2000, 10, alpha=0.05, p=0.1, seed=1)
        assert 0.12 <= peek <= 0.28, f"with 10 looks the rate should be well above 0.05 (about 0.19), got {peek:.3f}"
        strict = aa_false_positive_rate(4000, 2000, 1, alpha=0.01, p=0.1, seed=2)
        assert strict <= 0.025, f"alpha=0.01 with one look should give about 0.01, got {strict:.3f}"

    return run_check("Exercise 3 · aa_false_positive_rate", test,
                     hint="conversions per look: rng.binomial(per_look, p, size=(n_experiments, 2, n_looks)).cumsum(axis=2); "
                          "z-test at every look; (pvalues < alpha).any(axis=1).mean()")
