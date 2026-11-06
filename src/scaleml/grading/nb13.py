"""Self-checks for Level 3 notebook 13 · CTR prediction at scale."""

from __future__ import annotations

import math

from churnlab.checks import run_check


def _ne_reference(y, p):
    import numpy as np

    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    b = y.mean()
    return ll / -(b * math.log(b) + (1 - b) * math.log(1 - b))


def ex1(normalized_entropy):
    def test():
        import numpy as np

        rng = np.random.default_rng(0)
        y = (rng.random(5000) < 0.04).astype(int)
        base = np.full(5000, y.mean())
        ne = float(normalized_entropy(y, base))
        assert abs(ne - 1.0) < 1e-6, f"predicting the base rate for everyone must give NE = 1.0, got {ne:.6f}"
        p = np.clip(0.04 + 0.5 * (y - 0.04) * rng.random(5000), 0.001, 0.999)
        got, want = float(normalized_entropy(y, p)), _ne_reference(y, p)
        assert abs(got - want) < 1e-6, f"NE = {got:.6f}, expected {want:.6f} (log loss / entropy of the base rate)"
        assert got < 1.0, "an informative predictor must have NE < 1"
        y2 = np.array([0, 0, 0, 1])
        p2 = np.array([0.1, 0.2, 0.3, 0.6])
        got2 = float(normalized_entropy(y2, p2))
        assert abs(got2 - _ne_reference(y2, p2)) < 1e-6, (
            f"NE on a tiny example = {got2:.6f}, expected {_ne_reference(y2, p2):.6f}; "
            "use the base rate of THESE labels (mean of y), natural log, and the mean (not sum) log loss")
        p3 = np.array([0.0, 0.0, 0.5, 1.0])
        assert math.isfinite(float(normalized_entropy(y2, p3))), "clip predictions away from 0 and 1 (e.g. 1e-7) so NE stays finite"

    return run_check("Exercise 1 · normalized_entropy", test,
                     hint="ll = -mean(y log p + (1-y) log(1-p)) with p clipped; b = mean(y); "
                          "H = -(b log b + (1-b) log(1-b)); return ll / H")


def ex2(recalibrate_downsampled):
    def test():
        import numpy as np

        p = np.array([0.01, 0.1, 0.5, 0.9])
        got = np.asarray(recalibrate_downsampled(p, 1.0), dtype=float)
        assert got.shape == p.shape and np.allclose(got, p), "w = 1 (no downsampling) must return p unchanged"
        got = np.asarray(recalibrate_downsampled(np.array([0.5]), 0.1), dtype=float)
        assert np.allclose(got, [0.5 / (0.5 + 0.5 / 0.1)]), (
            f"p=0.5, w=0.1 should give {0.5 / 5.5:.5f} (true odds = 0.1 × downsampled odds), got {got}")
        # simulate: true rate 2%, keep 5% of negatives; a perfectly calibrated model on the
        # downsampled data predicts the downsampled rate; recalibrating must recover 2%
        rng = np.random.default_rng(1)
        y = rng.random(400_000) < 0.02
        keep = y | (rng.random(y.size) < 0.05)
        p_ds = y[keep].mean()
        q = float(np.asarray(recalibrate_downsampled(np.array([p_ds]), 0.05))[0])
        assert abs(q - y.mean()) < 0.002, f"recalibrated rate {q:.4f} should match the true rate {y.mean():.4f}"
        mono = np.asarray(recalibrate_downsampled(np.linspace(0.01, 0.99, 50), 0.2), dtype=float)
        assert np.all(np.diff(mono) > 0) and np.all(mono < np.linspace(0.01, 0.99, 50)), (
            "recalibration must be increasing in p and lower every prediction when w < 1")

    return run_check("Exercise 2 · recalibrate_downsampled", test,
                     hint="q = p / (p + (1 - p) / w); vectorise with numpy")


def ex3(hash_feature):
    def test():
        import builtins
        from collections import Counter

        n = 1000
        a = hash_feature("C1", "68fd1e64", n)
        assert isinstance(a, int) or type(a).__name__.startswith(("int", "uint")), f"return an int, got {type(a)}"
        assert 0 <= int(a) < n, f"bucket {a} outside [0, {n})"
        assert int(hash_feature("C1", "68fd1e64", n)) == int(a), "the same input must always give the same bucket"
        # detect Python's built-in (per-process salted) hash()
        real_hash = builtins.hash
        builtins.hash = lambda x: real_hash(x) ^ 0x5DEECE66D
        try:
            b = hash_feature("C1", "68fd1e64", n)
        finally:
            builtins.hash = real_hash
        assert int(b) == int(a), ("your function depends on Python's hash(), which is salted per process "
                                  "(PYTHONHASHSEED): training and serving would disagree. Use hashlib")
        # field-aware: same value in different fields should usually land in different buckets
        same = sum(int(hash_feature("C1", str(v), n)) == int(hash_feature("C2", str(v), n)) for v in range(300))
        assert same < 15, f"{same}/300 values hash to the same bucket in C1 and C2: include the field name in the hashed key"
        # roughly uniform
        counts = Counter(int(hash_feature("C7", f"v{v}", 50)) for v in range(5000))
        assert len(counts) == 50 and max(counts.values()) < 160, (
            f"buckets are far from uniform (max {max(counts.values())} of 5000 in one of 50 buckets); use a real hash")
        assert int(hash_feature("C3", 12345, 2**20)) < 2**20, "integer values must work too (convert to str)"

    return run_check("Exercise 3 · hash_feature", test,
                     hint="h = hashlib.md5(f'{field}={value}'.encode()).digest(); "
                          "return int.from_bytes(h[:8], 'little') % n_buckets")
