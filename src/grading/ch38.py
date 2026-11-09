"""Self-checks for Level 2 notebook 20 · Final evaluation & mock loop."""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check


def _reference(values, groups, n_boot, seed):
    v = np.asarray(values, float)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums, counts = np.bincount(g, weights=v), np.bincount(g).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
    stats = sums[idx].sum(1) / counts[idx].sum(1)
    return float(v.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def ex1(bootstrap_ci_grouped):
    def test():
        rng = np.random.default_rng(1)
        n_groups, per = 80, 25
        effect = rng.normal(0, 1.0, n_groups)  # strong group effect: rows within a group are correlated
        groups = np.repeat([f"user{i}" for i in range(n_groups)], per)
        values = np.repeat(effect, per) + rng.normal(0, 0.3, n_groups * per)
        out = bootstrap_ci_grouped(values, groups, 2000, 0)
        assert isinstance(out, tuple) and len(out) == 3, f"return a tuple (mean, lo, hi), got {out!r}"
        mean, lo, hi = (float(x) for x in out)
        assert abs(mean - values.mean()) < 1e-9, f"point estimate should be the mean of all values, got {mean}"
        assert lo < mean < hi, f"expected lo < mean < hi, got {out}"
        assert bootstrap_ci_grouped(values, groups, 2000, 0) == out, "same seed must give the same interval"
        _, rlo, rhi = _reference(values, groups, 2000, 0)
        width, ref_width = hi - lo, rhi - rlo
        assert 0.75 * ref_width < width < 1.25 * ref_width, (
            f"CI width {width:.3f} vs about {ref_width:.3f} expected for a group bootstrap - "
            "are you resampling groups (not rows)?")
        iid = np.quantile([rng.choice(values, len(values)).mean() for _ in range(500)], [0.025, 0.975])
        assert width > 2 * (iid[1] - iid[0]), "the interval is as narrow as a row-level bootstrap: resample whole groups"
        # unequal group sizes: the statistic is the mean over ROWS of the resampled groups
        v2 = np.array([1.0] * 9 + [0.0])
        g2 = np.array([0] * 9 + [1])
        m2, lo2, hi2 = bootstrap_ci_grouped(v2, g2, 2000, 3)
        assert abs(m2 - 0.9) < 1e-12 and abs(lo2 - 0.0) < 1e-9 and abs(hi2 - 1.0) < 1e-9, (
            f"two groups (nine 1s, one 0): expected (0.9, 0.0, 1.0), got {(m2, lo2, hi2)}")

    return run_check("Exercise 1 · bootstrap_ci_grouped", test,
                     hint="np.unique(groups, return_inverse=True); per-group sums and counts; draw group indices "
                          "with replacement; statistic = sum of drawn sums / sum of drawn counts; 2.5/97.5 percentiles")
