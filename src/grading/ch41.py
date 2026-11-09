"""Self-checks for Level 3 notebook 03 · A transformer from scratch."""

from __future__ import annotations

from churnlab.checks import run_check

REQUIRES = {"torch"}


def _reference_attention(x, w_qkv, w_out, n_heads):
    import torch.nn.functional as F

    B, T, C = x.shape
    q, k, v = (x @ w_qkv).split(C, dim=-1)
    q, k, v = (t.view(B, T, n_heads, C // n_heads).transpose(1, 2) for t in (q, k, v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.transpose(1, 2).reshape(B, T, C) @ w_out


def ex1(causal_self_attention):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        B, T, C, H = 2, 7, 12, 3
        x = torch.randn(B, T, C, generator=g)
        w_qkv = torch.randn(C, 3 * C, generator=g) / C**0.5
        w_out = torch.randn(C, C, generator=g) / C**0.5
        out = causal_self_attention(x, w_qkv, w_out, H)
        assert torch.is_tensor(out) and out.shape == (B, T, C), f"expected shape {(B, T, C)}, got {getattr(out, 'shape', out)}"
        ref = _reference_attention(x, w_qkv, w_out, H)
        one_head = _reference_attention(x, w_qkv, w_out, 1)
        if (out - one_head).abs().max() < 1e-4 and (out - ref).abs().max() > 1e-4:
            raise AssertionError("output equals single-head attention: split C into n_heads heads of size C // n_heads")
        # perturbation test: changing token t must not change outputs at positions < t
        x2 = x.clone()
        x2[:, 4] += 1.0
        out2 = causal_self_attention(x2, w_qkv, w_out, H)
        assert torch.allclose(out[:, :4], out2[:, :4], atol=1e-5), (
            "changing position 4 changed outputs at positions 0-3: the causal mask is missing or wrong")
        assert not torch.allclose(out[:, 4:], out2[:, 4:], atol=1e-5), "changing position 4 changed nothing after it"
        err = (out - ref).abs().max().item()
        assert err < 1e-4, (f"max difference from the reference is {err:.2e}. Check the 1/sqrt(head_dim) scale, "
                            "the order q, k, v in the split, and the head reshape (B, T, H, D) -> (B, H, T, D)")

    return run_check("Exercise 1 · causal_self_attention", test,
                     hint="q,k,v = (x @ w_qkv).split(C, -1); reshape to (B, H, T, D); scores = q @ k^T / sqrt(D); "
                          "mask j > i with -inf; softmax; @ v; back to (B, T, C); @ w_out")


def ex2(top_k_sample):
    def test():
        import torch

        V = 10
        logits = torch.tensor([[2.0, 1.0, 0.5, 3.0, -1.0, 0.0, 2.5, -2.0, 1.5, 0.2]])
        g = torch.Generator().manual_seed(0)
        greedy = top_k_sample(logits.repeat(5, 1), 1, 1.0, g)
        assert torch.is_tensor(greedy) and greedy.shape == (5,), f"expected shape (5,), got {getattr(greedy, 'shape', greedy)}"
        assert greedy.dtype == torch.long, f"expected integer token ids (torch.long), got {greedy.dtype}"
        assert (greedy == 3).all(), f"k=1 must always pick the argmax (3), got {greedy.tolist()}"
        n, k, temp = 20_000, 3, 0.7
        s = top_k_sample(logits.repeat(n, 1), k, temp, torch.Generator().manual_seed(1))
        allowed = set(torch.topk(logits[0], k).indices.tolist())
        assert set(s.tolist()) <= allowed, f"sampled ids {sorted(set(s.tolist()) - allowed)} outside the top-{k}"
        top = torch.topk(logits[0], k)
        want = torch.zeros(V)
        want[top.indices] = torch.softmax(top.values / temp, dim=0)
        freq = torch.bincount(s, minlength=V).float() / n
        tv = 0.5 * (freq - want).abs().sum().item()
        assert tv < 0.03, (f"empirical distribution differs from softmax(top-k logits / temperature) by TV {tv:.3f}; "
                           f"expected {want[top.indices].tolist()}, got {freq[top.indices].tolist()}")
        a = top_k_sample(logits.repeat(50, 1), 5, 1.0, torch.Generator().manual_seed(7))
        b = top_k_sample(logits.repeat(50, 1), 5, 1.0, torch.Generator().manual_seed(7))
        assert torch.equal(a, b), "same generator seed must give the same samples - pass generator= to torch.multinomial"

    return run_check("Exercise 2 · top_k_sample", test,
                     hint="logits / temperature; keep values >= the k-th largest (torch.topk), set the rest to -inf; "
                          "softmax; torch.multinomial(probs, 1, generator=generator).squeeze(1)")
