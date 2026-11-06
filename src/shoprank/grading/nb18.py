"""Self-checks for Level 2 notebook 18 · Model optimisation for inference."""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check

REQUIRES = {"torch"}


def ex1(parity):
    def test():
        a = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = parity(a, a + np.array([0.0, 1e-7, -2e-7]), 1e-6)
        assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
        for key in ("max_abs_diff", "n_over", "ok"):
            assert key in out, f"missing key {key!r}"
        assert out["ok"] is True or out["ok"] == True, "differences of 2e-7 with atol=1e-6 should pass"  # noqa: E712
        assert abs(out["max_abs_diff"] - 2e-7) < 5e-8, f"max_abs_diff should be ~2e-7, got {out['max_abs_diff']}"
        assert out["n_over"] == 0
        out = parity([0.0, 1.0, 2.0, 3.0], [0.0, 1.5, 2.0, 2.0], 0.1)
        assert not out["ok"] and out["n_over"] == 2 and abs(out["max_abs_diff"] - 1.0) < 1e-12, (
            f"two values differ by more than atol: expected ok=False, n_over=2, max_abs_diff=1.0, got {out}")
        out = parity(np.ones((2, 3)), np.ones((2, 3)), 0.0)
        assert out["ok"] and out["max_abs_diff"] == 0.0, "identical 2-D arrays should pass with atol=0"
        out = parity([1.0, np.nan], [1.0, 1.0], 1e-3)
        assert not out["ok"] and out["n_over"] == 1, "a NaN in either array is a mismatch (n_over counts it)"
        out = parity([1.0, np.nan], [1.0, np.nan], 1e-3)
        assert not out["ok"], "NaN outputs are a bug even when both sides have them: ok must be False"
        try:
            parity(np.zeros(3), np.zeros((3, 1)), 1e-3)
        except ValueError:
            pass
        else:
            raise AssertionError("shapes (3,) and (3, 1) must raise ValueError - silent broadcasting hides bugs")

    return run_check("Exercise 1 · parity", test,
                     hint="np.asarray both; check .shape; d = np.abs(a - b); treat np.isnan(d) as over; "
                          "max with np.nanmax (or inf when NaN)")


def ex2(distillation_loss):
    def test():
        import torch
        from torch.nn import functional as F

        g = torch.Generator().manual_seed(0)
        s = torch.randn(64, generator=g, requires_grad=True)
        p = torch.rand(64, generator=g) * 0.9 + 0.05
        y = (torch.rand(64, generator=g) < 0.3).float()

        def ref(s, p, y, alpha, T):
            soft = torch.sigmoid(torch.logit(p) / T)
            return (alpha * T * T * F.binary_cross_entropy_with_logits(s / T, soft)
                    + (1 - alpha) * F.binary_cross_entropy_with_logits(s, y))

        for alpha, T in [(0.0, 1.0), (1.0, 1.0), (0.5, 2.0), (0.7, 4.0)]:
            got = distillation_loss(s, p, y, alpha, T)
            assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor (mean over the batch)"
            want = ref(s, p, y, alpha, T)
            assert torch.allclose(got, want, atol=1e-5), (
                f"alpha={alpha}, T={T}: got {got.item():.6f}, expected {want.item():.6f}")
        loss = distillation_loss(s, p, y, 0.5, 2.0)
        loss.backward()
        assert s.grad is not None and torch.isfinite(s.grad).all(), "the loss must be differentiable w.r.t. student_logits"
        assert distillation_loss(s.detach(), p, y.long(), 0.0, 1.0).isfinite(), "integer labels should work too"

    return run_check("Exercise 2 · distillation_loss", test,
                     hint="soft target = sigmoid(logit(teacher_probs) / T); soft term = T*T * "
                          "BCE-with-logits(student/T, soft target); hard term = BCE-with-logits(student, labels)")
