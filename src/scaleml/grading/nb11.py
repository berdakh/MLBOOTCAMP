"""Self-checks for Level 3 notebook 11 · Quantisation & distillation."""

from __future__ import annotations

from churnlab.checks import run_check

REQUIRES = {"torch"}


def _dequantize(q, scale):
    return q.float() * scale.float()[:, None]


def ex1(quantize_per_channel_int8):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        w = torch.randn(16, 32, generator=g) * torch.logspace(-3, 1, 16)[:, None]  # rows of very different scale
        w[3] = 0.0  # an all-zero row must not produce nan
        out = quantize_per_channel_int8(w.clone())
        assert isinstance(out, tuple) and len(out) == 2, "return a tuple (q, scale)"
        q, scale = out
        assert q.dtype == torch.int8, f"q should be torch.int8, got {q.dtype}"
        assert tuple(q.shape) == tuple(w.shape), f"q has shape {tuple(q.shape)}, expected {tuple(w.shape)}"
        assert tuple(scale.shape) == (w.shape[0],), f"scale should have one entry per row (shape {(w.shape[0],)}), " \
                                                    f"got {tuple(scale.shape)}"
        assert int(q.min()) >= -127 and int(q.max()) <= 127, "symmetric int8 uses the range [-127, 127]"
        for r in range(w.shape[0]):
            if r == 3:
                continue
            assert int(q[r].abs().max()) == 127, f"row {r}: the largest |value| should map to 127 (scale = max|w|/127)"
            want = w[r].abs().max() / 127
            assert torch.isclose(scale[r].float(), want, rtol=1e-5), f"row {r}: scale {scale[r]:.3e}, expected {want:.3e}"
        deq = _dequantize(q, scale)
        assert torch.isfinite(deq).all(), "dequantized values contain nan/inf (all-zero row? guard the division)"
        assert deq[3].abs().max() == 0, "an all-zero row should dequantize to zeros"
        err = (deq - w).abs()
        bound = scale.float()[:, None] / 2 + 1e-12
        assert (err <= bound * 1.0001).all(), "rounding error must be at most scale/2 per element: round, don't truncate"
        rel = (err.norm(dim=1) / w.norm(dim=1).clamp_min(1e-12))
        per_tensor_scale = w.abs().max() / 127
        rel_pt = ((torch.round(w / per_tensor_scale) * per_tensor_scale - w).norm(dim=1) / w.norm(dim=1).clamp_min(1e-12))
        assert rel[0] < rel_pt[0], "the small-magnitude row should be quantized much better per-channel than per-tensor"

    return run_check("Exercise 1 · quantize_per_channel_int8", test,
                     hint="scale = W.abs().amax(dim=1) / 127 (replace 0 by 1); q = round(W / scale[:, None])"
                          ".clamp(-127, 127).to(torch.int8)")


def ex2(kd_loss):
    def test():
        import torch
        import torch.nn.functional as F

        g = torch.Generator().manual_seed(0)
        s = torch.randn(6, 4, generator=g, requires_grad=True)
        t = torch.randn(6, 4, generator=g) * 3
        y = torch.tensor([0, 1, 2, 3, 0, 1])
        for T, alpha in [(1.0, 0.5), (2.0, 0.5), (4.0, 0.9), (2.0, 0.0), (3.0, 1.0)]:
            got = kd_loss(s, t, y, T, alpha)
            assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor (so you can call .backward())"
            soft = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean") * T * T
            want = alpha * soft + (1 - alpha) * F.cross_entropy(s, y)
            assert torch.isclose(got, want, rtol=1e-4, atol=1e-6), (
                f"T={T}, alpha={alpha}: expected {want.item():.5f}, got {got.item():.5f}. Check: KL(teacher || "
                "student) on temperature-softened distributions, mean over the batch, times T^2")
        got = kd_loss(s, t, y, 2.0, 0.5)
        got.backward()
        assert s.grad is not None and s.grad.abs().sum() > 0, "the loss must be differentiable w.r.t. student logits"
        same = kd_loss(t.clone(), t, y, 2.0, 1.0)
        assert abs(same.item()) < 1e-5, "with alpha=1, a student equal to the teacher should have ~0 loss"

    return run_check("Exercise 2 · kd_loss", test,
                     hint="alpha * T^2 * F.kl_div(F.log_softmax(s/T), F.softmax(t/T), reduction='batchmean') "
                          "+ (1 - alpha) * F.cross_entropy(s, labels)")
