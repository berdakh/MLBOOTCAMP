"""Self-checks for Level 3 notebook 08 · Contrastive training & re-ranking."""

from __future__ import annotations

from churnlab.checks import run_check

REQUIRES = {"torch"}


def ex1(info_nce_loss):
    def test():
        import torch
        import torch.nn.functional as F

        g = torch.Generator().manual_seed(0)
        q = F.normalize(torch.randn(6, 16, generator=g), dim=-1).requires_grad_(True)
        d = F.normalize(torch.randn(9, 16, generator=g), dim=-1)  # 6 positives + 3 hard negatives
        for t in (0.05, 1.0):
            got = info_nce_loss(q, d, t)
            assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor"
            logits = q @ d.T / t
            want = F.cross_entropy(logits, torch.arange(6))
            wrong = F.cross_entropy(logits[:, :6], torch.arange(6))
            if torch.allclose(got, wrong, atol=1e-5) and not torch.allclose(wrong, want, atol=1e-5):
                raise AssertionError("you ignored the extra rows of d: hard negatives at rows B..N-1 must be in the softmax")
            assert torch.allclose(got, want, atol=1e-5), (
                f"temperature {t}: loss {got.item():.4f}, expected {want.item():.4f} "
                "(cross-entropy of q @ d.T / temperature with targets 0..B-1)")
        got.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all(), "the loss must be differentiable w.r.t. q"
        perfect = info_nce_loss(F.normalize(torch.eye(4), dim=-1), F.normalize(torch.eye(4), dim=-1), 0.01)
        assert perfect.item() < 1e-6, "identical, orthogonal pairs at a small temperature should give ~0 loss"

    return run_check("Exercise 1 · info_nce_loss", test,
                     hint="logits = q @ d.T / temperature; F.cross_entropy(logits, torch.arange(len(q)))")


def ex2(mine_hard_negatives):
    def test():
        ranked = ["p1", "a", "b", "p2", "c", "d", "e"]
        got = list(mine_hard_negatives(list(ranked), {"p1", "p2"}, 3, 0))
        assert got == ["a", "b", "c"], f"expected ['a', 'b', 'c'], got {got}"
        got = list(mine_hard_negatives(list(ranked), {"p1", "p2"}, 2, 2))
        assert got == ["b", "c"], f"skip_top=2 skips the first two RANKS (p1, a), then excludes positives: expected ['b', 'c'], got {got}"
        got = list(mine_hard_negatives(list(ranked), ["p1"], 10, 5))
        assert got == ["d", "e"], f"fewer candidates than n: return what is left, got {got}"
        got = list(mine_hard_negatives(["x", "y"], set(), 0, 0))
        assert got == [], f"n=0 must return [], got {got}"

    return run_check("Exercise 2 · mine_hard_negatives", test,
                     hint="[d for d in ranked_ids[skip_top:] if d not in set(positives)][:n]")
