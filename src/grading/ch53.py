"""Self-checks for Level 3 notebook 15 · Fairness, robustness & privacy."""

from __future__ import annotations

from churnlab.checks import run_check


def ex1(bpsn_bnsp_auc):
    def test():
        import numpy as np
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(0)
        n = 2000
        y = (rng.random(n) < 0.3).astype(int)
        g = rng.random(n) < 0.2
        s = rng.random(n) + 0.8 * y + 0.4 * g  # the model scores the subgroup higher regardless of the label
        out = bpsn_bnsp_auc(y, s, g)
        assert isinstance(out, tuple) and len(out) == 2, f"return a tuple (bpsn, bnsp), got {out!r}"
        bpsn, bnsp = (float(v) for v in out)
        m_bpsn = (g & (y == 0)) | (~g & (y == 1))
        m_bnsp = (g & (y == 1)) | (~g & (y == 0))
        want_bpsn, want_bnsp = roc_auc_score(y[m_bpsn], s[m_bpsn]), roc_auc_score(y[m_bnsp], s[m_bnsp])
        if abs(bpsn - want_bnsp) < 1e-9 and abs(bnsp - want_bpsn) < 1e-9:
            raise AssertionError("BPSN and BNSP are swapped: BPSN = Background Positives + Subgroup Negatives")
        assert abs(bpsn - want_bpsn) < 1e-9, (
            f"BPSN = {bpsn:.4f}, expected {want_bpsn:.4f}: AUC on subgroup NEGATIVES together with background POSITIVES")
        assert abs(bnsp - want_bnsp) < 1e-9, (
            f"BNSP = {bnsp:.4f}, expected {want_bnsp:.4f}: AUC on subgroup POSITIVES together with background NEGATIVES")
        assert bpsn < bnsp, "a model that inflates the subgroup's scores should have BPSN < BNSP"
        g2 = np.array([True, True, False, False, True, False])
        y2 = np.array([0, 1, 0, 1, 0, 1])
        s2 = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        b1, b2 = (float(v) for v in bpsn_bnsp_auc(y2, s2, g2))
        assert b1 == 1.0 and b2 == 1.0, f"a perfect ranking gives BPSN = BNSP = 1.0, got {(b1, b2)}"
        out = bpsn_bnsp_auc(list(y2), list(s2), list(g2))
        assert len(out) == 2, "plain Python lists must work too (convert with np.asarray)"

    return run_check("Exercise 1 · bpsn_bnsp_auc", test,
                     hint="g = np.asarray(subgroup, bool); bpsn mask = (g & y==0) | (~g & y==1); "
                          "bnsp mask = (g & y==1) | (~g & y==0); roc_auc_score(y[mask], s[mask])")


def ex2(clip_and_noise):
    def test():
        import numpy as np

        g = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])  # norms 5, 0.5, 0
        out = np.asarray(clip_and_noise(g, 1.0, 0.0, np.random.default_rng(0)), dtype=float)
        assert out.shape == (2,), f"return the noisy MEAN gradient, shape (d,) = (2,), got {out.shape}"
        want = (np.array([0.6, 0.8]) + np.array([0.3, 0.4])) / 3
        assert np.allclose(out, want), (
            f"with sigma = 0 expected {want.round(4)} (row 0 scaled to norm 1, row 1 untouched, mean over 3 rows), "
            f"got {out.round(4)}")
        n, d, clip, sigma = 10, 20_000, 0.5, 2.0
        z = np.asarray(clip_and_noise(np.zeros((n, d)), clip, sigma, np.random.default_rng(1)), dtype=float)
        sd, mean = z.std(), z.mean()
        assert abs(sd - sigma * clip / n) < 0.05 * sigma * clip / n, (
            f"noise std {sd:.4f}, expected sigma * clip / n = {sigma * clip / n:.4f}: add N(0, (sigma*clip)^2) to "
            "the SUM of clipped gradients, then divide by the batch size")
        assert abs(mean) < 0.005, f"noise must have mean 0, got {mean:.4f}"
        big = np.random.default_rng(2).normal(size=(50, 30)) * 10
        out1 = np.asarray(clip_and_noise(big, 1.0, 0.0, np.random.default_rng(0)), dtype=float)
        assert np.linalg.norm(out1) <= 1.0 + 1e-9, "after clipping to 1, the mean gradient's norm cannot exceed 1"
        a = clip_and_noise(big, 1.0, 1.0, np.random.default_rng(5))
        b = clip_and_noise(big, 1.0, 1.0, np.random.default_rng(5))
        assert np.allclose(a, b), "draw the noise from the rng argument so results are reproducible"
        assert not np.allclose(clip_and_noise(big, 1.0, 1.0, np.random.default_rng(6)), a), "noise must depend on the rng"

    return run_check("Exercise 2 · clip_and_noise", test,
                     hint="norms = np.linalg.norm(g, axis=1, keepdims=True); g * np.minimum(1, clip / max(norms, 1e-12)); "
                          "(clipped.sum(0) + rng.normal(0, sigma * clip, d)) / n")
