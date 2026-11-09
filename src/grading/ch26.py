"""Self-checks for Level 2 notebook 08 · Two-tower retrieval & ANN."""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check

REQUIRES = {"torch"}  # exercise 1 needs PyTorch; tests skip this module without it


def _reference(user_emb, item_emb, log_q=None):
    import torch

    logits = user_emb @ item_emb.T
    if log_q is not None:
        logits = logits - log_q[None, :]
    # Explicit log-softmax of the diagonal, written differently from F.cross_entropy on purpose.
    return -(torch.diagonal(logits) - torch.logsumexp(logits, dim=1)).mean()


def ex1(in_batch_softmax_loss):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        for B, d in [(8, 4), (32, 16)]:
            u = torch.randn(B, d, generator=g)
            v = torch.randn(B, d, generator=g)
            log_q = torch.log(torch.rand(B, generator=g) * 0.1 + 1e-3)
            for lq in (None, log_q):
                u1, v1 = u.clone().requires_grad_(True), v.clone().requires_grad_(True)
                got = in_batch_softmax_loss(u1, v1, lq) if lq is not None else in_batch_softmax_loss(u1, v1)
                assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor (the mean loss)"
                want = _reference(u, v, lq)
                assert torch.allclose(got.detach(), want, atol=1e-5), (
                    f"B={B}, log_q={'given' if lq is not None else 'None'}: expected {want.item():.5f}, "
                    f"got {got.item():.5f}")
                got.backward()
                u2, v2 = u.clone().requires_grad_(True), v.clone().requires_grad_(True)
                _reference(u2, v2, lq).backward()
                assert u1.grad is not None and torch.allclose(u1.grad, u2.grad, atol=1e-5), \
                    "gradients differ from the reference (did you detach something or use NumPy?)"
                assert torch.allclose(v1.grad, v2.grad, atol=1e-5), "item gradients differ from the reference"
        # Subtracting the same constant from every column must not change a softmax loss.
        u, v = torch.randn(6, 3, generator=g), torch.randn(6, 3, generator=g)
        lq = torch.randn(6, generator=g)
        assert torch.allclose(in_batch_softmax_loss(u, v, lq), in_batch_softmax_loss(u, v, lq + 3.0), atol=1e-5), \
            "shifting every log_q by a constant should not change the loss"

    return run_check("Exercise 1 · in_batch_softmax_loss", test,
                     hint="logits = user_emb @ item_emb.T; if log_q is not None: logits = logits - log_q[None, :]; "
                          "return F.cross_entropy(logits, torch.arange(len(logits)))")


def ex2(recall_vs_exact):
    def test():
        exact = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        approx = np.array([[4, 3, 9, 10], [5, 6, 7, 8]])
        got = float(recall_vs_exact(approx, exact))
        assert abs(got - 0.75) < 1e-9, f"expected (2/4 + 4/4) / 2 = 0.75, got {got}"
        approx = np.array([[1, 2, -1, -1], [8, 7, 6, 5]])
        got = float(recall_vs_exact(approx, exact))
        assert abs(got - 0.75) < 1e-9, f"order must not matter and -1 is 'no result': expected 0.75, got {got}"
        exact = np.array([[1, 2, -1, -1]])
        approx = np.array([[2, -1, -1, -1]])
        got = float(recall_vs_exact(approx, exact))
        assert abs(got - 0.5) < 1e-9, (
            f"when exact has only 2 real results, the denominator is 2 (ignore -1 padding): expected 0.5, got {got}")
        rng = np.random.default_rng(0)
        e = np.stack([rng.permutation(1000)[:50] for _ in range(20)])
        a = np.where(rng.random(e.shape) < 0.8, e, e + 5000)
        got = float(recall_vs_exact(a, e))
        want = float(np.mean([(row_a == row_e).sum() / 50 for row_a, row_e in zip(a, e)]))
        assert abs(got - want) < 1e-9, f"random case: expected {want:.4f}, got {got:.4f}"

    return run_check("Exercise 2 · recall_vs_exact", test,
                     hint="per row: e = set(exact[e >= 0]); a = set(approx[a >= 0]); len(a & e) / len(e); then the mean")
