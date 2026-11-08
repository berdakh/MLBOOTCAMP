"""Self-checks for Level 3 notebook 04 · Pretrained transformers & tokenisation."""

from __future__ import annotations

import math
import random

from churnlab.checks import run_check

REQUIRES = {"torch"}


def padding_waste(lengths, batches) -> int:
    """Pad tokens added when every batch is padded to its longest member."""
    return sum(max(lengths[i] for i in b) * len(b) - sum(lengths[i] for i in b) for b in batches if b)


def ex1(mean_pool):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        h = torch.randn(3, 5, 4, generator=g)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
        out = mean_pool(h, mask)
        assert torch.is_tensor(out) and out.shape == (3, 4), f"expected shape (3, 4), got {getattr(out, 'shape', out)}"
        want = torch.stack([h[0, :3].mean(0), h[1].mean(0), h[2, 0]])
        if torch.allclose(out, h.mean(1), atol=1e-5):
            raise AssertionError("you averaged over padding positions too: weight by the attention mask")
        assert torch.allclose(out, want, atol=1e-5), "wrong values: sum(h * mask) / sum(mask) per sequence"
        h2 = h.clone()
        h2[0, 3:] = 1e3  # garbage at padded positions must not matter
        assert torch.allclose(mean_pool(h2, mask)[0], want[0], atol=1e-4), "padded positions changed the result"
        empty = mean_pool(h, torch.zeros(3, 5, dtype=torch.long))
        assert torch.isfinite(empty).all(), "an all-padding row gave NaN/inf: clamp the token count (e.g. min=1e-9)"

    return run_check("Exercise 1 · mean_pool", test,
                     hint="m = attention_mask.unsqueeze(-1).to(last_hidden.dtype); "
                          "(last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)")


def ex2(bucket_by_length):
    def test():
        rng = random.Random(0)
        lengths = [rng.choice([rng.randint(5, 30), rng.randint(30, 200)]) for _ in range(203)]
        bs = 16
        batches = [list(map(int, b)) for b in bucket_by_length(list(lengths), bs)]
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(len(lengths))), "every index must appear exactly once"
        assert all(0 < len(b) <= bs for b in batches), f"batches must hold 1..{bs} indices"
        assert len(batches) == math.ceil(len(lengths) / bs), (
            f"use full batches: expected {math.ceil(len(lengths) / bs)} batches, got {len(batches)} "
            "(tiny batches avoid padding but waste the hardware)")
        naive = [list(range(i, min(i + bs, len(lengths)))) for i in range(0, len(lengths), bs)]
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        sorted_chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
        got, base, best = (padding_waste(lengths, b) for b in (batches, naive, sorted_chunks))
        assert got <= best, (f"padding waste {got:,} tokens; sorting by length and chunking gives {best:,} "
                             f"(naive order: {base:,})")

    return run_check("Exercise 2 · bucket_by_length", test,
                     hint="order = sorted(range(len(lengths)), key=lengths.__getitem__); "
                          "chunk order into slices of batch_size")
