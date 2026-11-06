"""Self-checks for Level 3 notebook 05 · Fine-tuning a classifier."""

from __future__ import annotations

import math

import numpy as np

from churnlab.checks import run_check

REQUIRES = {"torch"}


def _ece_reference(probs, labels, n_bins):
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels)
    conf, correct = probs.max(1), (probs.argmax(1) == labels).astype(float)
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    return sum((bins == b).mean() * abs(correct[bins == b].mean() - conf[bins == b].mean())
               for b in range(n_bins) if (bins == b).any())


def _random_probs(rng, n, c, sharp):
    z = rng.normal(size=(n, c)) * sharp
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ex1(expected_calibration_error):
    def test():
        rng = np.random.default_rng(0)
        probs = _random_probs(rng, 400, 4, 3.0)
        labels = rng.integers(0, 4, 400)
        got = expected_calibration_error(probs, labels, 10)
        assert isinstance(got, float | np.floating), f"return a float, got {type(got).__name__}"
        want = _ece_reference(probs, labels, 10)
        assert math.isclose(float(got), want, abs_tol=1e-6), f"ECE {float(got):.5f}, expected {want:.5f}"
        # perfectly confident and always right -> 0; always wrong at confidence 1 -> 1
        onehot = np.eye(3)[[0, 1, 2, 1]]
        assert math.isclose(float(expected_calibration_error(onehot, np.array([0, 1, 2, 1]), 5)), 0.0, abs_tol=1e-9), (
            "one-hot correct predictions must give ECE 0 (confidence 1.0 goes in the last bin)")
        assert math.isclose(float(expected_calibration_error(onehot, np.array([1, 2, 0, 0]), 5)), 1.0, abs_tol=1e-9), (
            "one-hot wrong predictions must give ECE 1")
        # the number of bins matters and empty bins are skipped
        got15 = float(expected_calibration_error(probs, labels, 15))
        assert math.isclose(got15, _ece_reference(probs, labels, 15), abs_tol=1e-6), "wrong result for n_bins=15"

    return run_check("Exercise 1 · expected_calibration_error", test,
                     hint="conf = probs.max(1); bin = clip(ceil(conf * n_bins) - 1, 0, n_bins - 1); "
                          "sum over non-empty bins of mean(in_bin) * |acc - conf|")


def _nll(logits, labels, t):
    z = logits / t
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return -logp[np.arange(len(labels)), labels].mean()


def _reference_temperature(logits, labels):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 2001))
    t = grid[np.argmin([_nll(logits, labels, g) for g in grid])]
    fine = np.linspace(t * 0.99, t * 1.01, 401)
    return float(fine[np.argmin([_nll(logits, labels, g) for g in fine])])


def ex2(fit_temperature):
    def test():
        rng = np.random.default_rng(1)
        for scale, label_noise in [(4.0, 0.3), (0.5, 0.0)]:  # over-confident, then under-confident
            labels = rng.integers(0, 4, 300)
            logits = rng.normal(size=(300, 4))
            logits[np.arange(300), labels] += 1.5
            flip = rng.random(300) < label_noise
            labels = np.where(flip, rng.integers(0, 4, 300), labels)
            logits = logits * scale
            got = fit_temperature(logits.copy(), labels.copy())
            got = float(got)
            assert got > 0, f"temperature must be positive, got {got}"
            want = _reference_temperature(logits, labels)
            assert abs(got - want) / want < 0.02, (
                f"T = {got:.3f}, the NLL-minimising temperature is {want:.3f} "
                f"({'over' if want > 1 else 'under'}-confident logits)")

    return run_check("Exercise 2 · fit_temperature", test,
                     hint="log_t = torch.zeros((), requires_grad=True); LBFGS on "
                          "F.cross_entropy(logits / log_t.exp(), labels); return log_t.exp().item()")
