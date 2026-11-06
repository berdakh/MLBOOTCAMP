"""Self-checks for Level 2 notebook 02 · Aggregation & joins at scale."""

from __future__ import annotations

from collections import Counter

import numpy as np

from churnlab.checks import run_check


class _OnePass:
    """An iterable that can be consumed once and has no len(): a real stream."""

    def __init__(self, n):
        self.n, self.used = n, False

    def __iter__(self):
        assert not self.used, "the stream was iterated twice: a stream can only be read once"
        self.used = True
        return iter(range(self.n))


def ex1(reservoir_sample):
    def test():
        out = list(reservoir_sample(iter(range(100)), 10, 0))
        assert len(out) == 10, f"expected 10 items, got {len(out)}"
        assert len(set(out)) == 10 and set(out) <= set(range(100)), "items must be distinct items from the stream"
        assert list(reservoir_sample(iter(range(100)), 10, 0)) == out, "same seed must give the same sample"
        others = {tuple(sorted(reservoir_sample(iter(range(100)), 10, s))) for s in range(1, 6)}
        assert len(others) > 1, "different seeds always give the same sample"
        assert sorted(reservoir_sample(iter(range(3)), 5, 1)) == [0, 1, 2], "a short stream returns all its items"
        assert list(reservoir_sample(iter([]), 3, 1)) == [], "an empty stream returns []"
        assert len(reservoir_sample(_OnePass(1000), 7, 3)) == 7, "must work on a one-pass iterable"
        # Uniformity: 2,000 seeds, 20-item stream, k=5 -> each item expected 500 times.
        n, k, trials = 20, 5, 2000
        counts = np.zeros(n)
        for seed in range(trials):
            for x in reservoir_sample(iter(range(n)), k, seed):
                counts[x] += 1
        expected = trials * k / n
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        # 19 degrees of freedom: P(chi2 > 50) ~ 1e-4 for a correct sampler.
        assert chi2 < 50, (f"inclusion counts are not uniform (chi-square {chi2:.0f} on 19 dof): first items "
                           f"{counts[:3].astype(int).tolist()}, last items {counts[-3:].astype(int).tolist()}, "
                           f"expected {expected:.0f} each")

    return run_check("Exercise 1 · reservoir_sample", test,
                     hint="rng = random.Random(seed); for i, x in enumerate(stream): if i < k: append; "
                          "else: j = rng.randint(0, i); if j < k: res[j] = x")


def ex2(partitioned_count):
    def test():
        rng = np.random.default_rng(5)
        uniform = rng.integers(0, 50, 2_000).tolist()
        skewed = ([7] * 5_000) + rng.integers(0, 1_000, 3_000).tolist()
        rng.shuffle(skewed)
        words = ["seller_a"] * 300 + ["seller_b"] * 20 + [f"s{i}" for i in range(50)]
        cases = [("uniform ints", uniform, 8), ("skewed ints", skewed, 16),
                 ("string keys", words, 4), ("one partition", uniform, 1), ("empty input", [], 4),
                 ("more partitions than keys", [1, 2, 2], 32)]
        for name, keys, p in cases:
            got = partitioned_count(list(keys), p)
            assert isinstance(got, dict), f"{name}: return a dict, got {type(got).__name__}"
            want = Counter(keys)
            assert dict(got) == dict(want), (f"{name}: result differs from Counter "
                                             f"({len(got)} vs {len(want)} keys)")
            assert all(isinstance(v, (int, np.integer)) for v in got.values()), f"{name}: counts must be ints"

    return run_check("Exercise 2 · partitioned_count", test,
                     hint="mappers: Counter(chunk); route each (key, c) to partitions[scale.stable_bucket(key, n)]; "
                          "each partition sums its Counters; merge the partition dicts (their keys are disjoint)")
