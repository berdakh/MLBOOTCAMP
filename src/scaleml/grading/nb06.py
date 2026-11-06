"""Self-checks for Level 3 notebook 06 · Parameter-efficient fine-tuning."""

from __future__ import annotations

from churnlab.checks import run_check

REQUIRES = {"torch"}


def ex1(LoRALinear):  # noqa: N803 - the learner passes a class
    def test():
        import torch
        from torch import nn

        torch.manual_seed(0)
        base = nn.Linear(12, 7)
        w0, b0 = base.weight.detach().clone(), base.bias.detach().clone()
        layer = LoRALinear(base, r=4, alpha=8)
        A, B = getattr(layer, "lora_A", None), getattr(layer, "lora_B", None)
        assert isinstance(A, nn.Parameter) and isinstance(B, nn.Parameter), (
            "create self.lora_A and self.lora_B as nn.Parameter")
        assert tuple(A.shape) == (4, 12), f"lora_A must be (r, in_features) = (4, 12), got {tuple(A.shape)}"
        assert tuple(B.shape) == (7, 4), f"lora_B must be (out_features, r) = (7, 4), got {tuple(B.shape)}"
        assert torch.count_nonzero(B) == 0, "lora_B must start at zero (so the layer equals the base at init)"
        assert torch.count_nonzero(A) > 0, "lora_A must start random, otherwise no gradient ever reaches B"
        trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
        assert trainable == {"lora_A", "lora_B"}, f"only lora_A and lora_B may be trainable, found {sorted(trainable)}"
        x = torch.randn(3, 5, 12)
        out = layer(x)
        assert tuple(out.shape) == (3, 5, 7), f"output shape {tuple(out.shape)}, expected (3, 5, 7)"
        assert torch.allclose(out, base(x), atol=1e-6), "at initialisation the output must equal base(x)"
        with torch.no_grad():
            layer.lora_B.normal_()
        want = x @ w0.T + b0 + (8 / 4) * (x @ layer.lora_A.T @ layer.lora_B.T)
        assert torch.allclose(layer(x), want, atol=1e-4), "wrong output: base(x) + (alpha / r) * x @ A.T @ B.T"
        layer(x).sum().backward()
        assert base.weight.grad is None, "the base weight received a gradient: freeze it"
        assert layer.lora_A.grad is not None and layer.lora_B.grad is not None, "A and B must receive gradients"

    return run_check("Exercise 1 · LoRALinear", test,
                     hint="freeze with p.requires_grad_(False) for p in base.parameters(); "
                          "forward: self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * (alpha / r)")


def ex2(merge_lora):
    def test():
        import torch

        g = torch.Generator().manual_seed(0)
        W, A, B = torch.randn(6, 5, generator=g), torch.randn(3, 5, generator=g), torch.randn(6, 3, generator=g)
        W_before = W.clone()
        out = merge_lora(W, A, B, 6, 3)
        assert torch.is_tensor(out) and tuple(out.shape) == (6, 5), f"expected a (6, 5) tensor, got {getattr(out, 'shape', out)}"
        assert torch.equal(W, W_before), "W was modified in place: return a new tensor"
        want = W_before + 2.0 * (B @ A)
        if torch.allclose(out, W_before + B @ A, atol=1e-5):
            raise AssertionError("missing the alpha / r scaling")
        assert torch.allclose(out, want, atol=1e-5), "expected W + (alpha / r) * B @ A"
        x = torch.randn(4, 5, generator=g)
        lora_out = x @ W_before.T + 2.0 * (x @ A.T @ B.T)
        assert torch.allclose(x @ out.T, lora_out, atol=1e-4), "the merged layer must reproduce the LoRA layer's output"

    return run_check("Exercise 2 · merge_lora", test, hint="return W + (alpha / r) * (B @ A)")
