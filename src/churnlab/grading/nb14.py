"""Self-checks for notebook 14 · Monitoring & rollback."""

from __future__ import annotations

import math

import numpy as np

from ..checks import run_check
from ..monitoring import psi as reference_psi


def ex1(my_psi):
    def check(ref, cur, bins, why):
        got = my_psi(ref, cur, bins=bins)
        expected = reference_psi(ref, cur, bins=bins)
        if math.isnan(expected):
            assert got is not None and math.isnan(float(got)), f"{why}: expected nan, got {got}"
            return
        assert isinstance(got, float), f"return a Python float, got {type(got).__name__}"
        assert abs(got - expected) < 1e-6, f"{why}: expected {expected:.6f}, got {got:.6f}"

    def test():
        rng = np.random.default_rng(14)
        ref = rng.normal(0, 1, 5000)
        check(ref, rng.normal(0, 1, 3000), 10, "same distribution (PSI near 0)")
        check(ref, rng.normal(0.5, 1, 3000), 10, "mean shift of half a standard deviation")
        check(ref, rng.normal(0, 2, 3000), 10, "variance doubled")
        check(ref, rng.normal(3, 1, 3000), 10, "almost no overlap (empty bins must be clipped at 1e-4)")
        check(ref, rng.normal(0.5, 1, 3000), 5, "bins=5")
        counts_ref = rng.poisson(3, 4000).astype(float)
        check(counts_ref, rng.poisson(4, 4000).astype(float), 10, "integer counts with ties (np.unique edges)")
        with_nan = ref.copy()
        with_nan[:100] = np.nan
        check(with_nan, rng.normal(0.2, 1, 2000), 10, "missing values are dropped first")
        check(np.zeros(1000), rng.normal(0, 1, 100), 10, "a constant reference has fewer than 2 edges -> 0.0")
        zero_inflated = np.where(rng.random(3000) < 0.9, 0.0, 1.0)
        check(zero_inflated, np.where(rng.random(3000) < 0.75, 0.0, 1.0), 10, "a 90%-zero feature")
        check([1.0, 2.0, 3.0], [], 10, "an empty current sample -> nan")

    return run_check("Exercise 1 · psi", test,
                     hint="edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); "
                          "edges[0], edges[-1] = -np.inf, np.inf; r = np.histogram(ref, edges)[0] / len(ref)")


POLICY = {"max_pr_auc_drop": 0.2, "max_error_rate": 0.01, "max_p95_ms": 200.0}
BASE = {"pr_auc": 0.25, "error_rate": 0.001, "p95_ms": 40.0}


def ex2(should_rollback):
    def check(candidate, expected_keys, why, baseline=BASE, policy=POLICY):
        result = should_rollback(dict(candidate), dict(baseline), dict(policy))
        assert isinstance(result, tuple) and len(result) == 2, f"return (bool, reasons), got {result!r}"
        decision, reasons = result
        assert isinstance(decision, bool), f"the decision must be a bool, got {type(decision).__name__}"
        assert isinstance(reasons, list), f"reasons must be a list, got {type(reasons).__name__}"
        assert decision == bool(expected_keys), f"{why}: expected rollback={bool(expected_keys)}, got {decision}"
        assert len(reasons) == len(expected_keys), (
            f"{why}: expected {len(expected_keys)} reason(s) ({expected_keys}), got {reasons}")
        for reason, key in zip(reasons, expected_keys):
            assert isinstance(reason, str) and key in reason, (
                f"{why}: reasons should mention {expected_keys} in that order, got {reasons}")

    def test():
        check({"pr_auc": 0.26, "error_rate": 0.001, "p95_ms": 45.0}, [], "a healthy candidate")
        check({"pr_auc": 0.15, "error_rate": 0.001, "p95_ms": 45.0}, ["pr_auc"], "PR-AUC dropped 40%")
        check({"pr_auc": 0.22, "error_rate": 0.001, "p95_ms": 45.0}, [], "PR-AUC dropped 12% (within 20%)")
        check({"pr_auc": 0.2, "error_rate": 0.01, "p95_ms": 200.0}, [], "exactly at every limit is not a breach")
        check({"pr_auc": 0.26, "error_rate": 0.05, "p95_ms": 45.0}, ["error_rate"], "error rate 5%")
        check({"pr_auc": 0.26, "error_rate": 0.0, "p95_ms": 350.0}, ["p95_ms"], "p95 350 ms")
        check({"pr_auc": 0.05, "error_rate": 0.02, "p95_ms": 250.0}, ["pr_auc", "error_rate", "p95_ms"],
              "everything breached")
        check({"pr_auc": None, "error_rate": 0.001, "p95_ms": 45.0}, [], "labels not matured: skip PR-AUC")
        check({"pr_auc": None, "error_rate": 0.02, "p95_ms": 45.0}, ["error_rate"],
              "labels not matured, but errors are high")
        check({"pr_auc": 0.30, "error_rate": 0.0, "p95_ms": 45.0}, ["pr_auc"],
              "a stricter policy and baseline", baseline={"pr_auc": 0.40, "error_rate": 0.0, "p95_ms": 30.0},
              policy={"max_pr_auc_drop": 0.2, "max_error_rate": 0.0, "max_p95_ms": 100.0})

    return run_check("Exercise 2 · should_rollback", test,
                     hint="build reasons = [] and append one string per breached rule; "
                          "return (bool(reasons), reasons)")
