"""Self-checks for Level 2 notebook 05 · Implicit feedback & position bias."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from churnlab.checks import run_check

from ..data import WorldConfig, make_world, simulate_logs


def _random_traffic_world():
    """A tiny world where EVERY request is exploration traffic (fast, deterministic)."""
    cfg = dataclasses.replace(WorldConfig.scale("tiny"), random_fraction=1.0, n_days=42, seed=7)
    world = make_world(cfg)
    imps = pd.concat(list(simulate_logs(world)), ignore_index=True).drop(columns=["examined"])
    return world, imps


def ex1(estimate_propensities):
    def test():
        world, imps = _random_traffic_world()
        shuffled = imps.sample(frac=1.0, random_state=0)  # any row order must work
        est = np.asarray(estimate_propensities(shuffled), dtype=float)
        assert est.shape == (10,), f"expected an array of length 10 (one value per position), got shape {est.shape}"
        assert abs(est[0] - 1.0) < 1e-9, f"position 0 must be exactly 1 after normalising, got {est[0]:.4f}"
        truth = world.examination_prob(np.arange(10))
        err = np.abs(est - truth)
        assert err.max() < 0.06, (
            f"too far from the true examination curve (max abs error {err.max():.3f} at position "
            f"{int(err.argmax())}); expected roughly {np.round(truth, 2).tolist()}")
        assert np.all(np.diff(est[:4]) < 0), "the first positions should be strictly decreasing"

    return run_check("Exercise 1 · estimate_propensities", test,
                     hint="ctr = random_imps.groupby('position')['clicked'].mean().sort_index(); "
                          "return (ctr / ctr.iloc[0]).to_numpy()")


def ex2(ips_ctr_estimate):
    def test():
        theta = np.array([1.0, 0.5, 0.25, 0.2])
        clicks = np.array([1, 0, 1, 1, 0, 1])
        positions = np.array([0, 0, 1, 2, 3, 3])
        got = float(ips_ctr_estimate(clicks, positions, theta, None))
        want = (1 + 2 + 4 + 5) / 6
        assert abs(got - want) < 1e-9, f"no clipping: expected {want:.4f}, got {got:.4f}"
        got = float(ips_ctr_estimate(clicks, positions, theta, 3.0))
        want = (1 + 2 + 3 + 3) / 6
        assert abs(got - want) < 1e-9, f"clip=3: expected {want:.4f} (weights capped at 3), got {got:.4f}"
        got = float(ips_ctr_estimate(list(clicks.astype(bool)), list(positions), list(theta), None))
        assert abs(got - 2.0) < 1e-9, "should accept lists and boolean clicks"
        # Statistical check: PBM data with relevance 0.3 everywhere -> estimate close to 0.3.
        rng = np.random.default_rng(0)
        th = (1.0 + np.arange(10)) ** -0.9
        pos = rng.integers(0, 10, 200_000)
        c = rng.random(pos.size) < th[pos] * 0.3
        est = float(ips_ctr_estimate(c, pos, th, None))
        assert abs(est - 0.3) < 0.01, f"on simulated PBM data with relevance 0.3 the estimate should be ~0.30, got {est:.3f}"

    return run_check("Exercise 2 · ips_ctr_estimate", test,
                     hint="w = 1 / np.asarray(propensities)[np.asarray(positions)]; "
                          "if clip is not None: w = np.minimum(w, clip); return float(np.mean(clicks * w))")
