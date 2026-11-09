"""Self-checks for Level 3 notebook 02 · Training efficiency & memory."""

from __future__ import annotations

from churnlab.checks import close, run_check

REQUIRES = {"torch"}

#: bytes per parameter (weights + gradients + optimizer state [+ fp16 weight copy]); no activations
TABLE = {("sgd", False): 8, ("sgd_momentum", False): 12, ("adam", False): 16, ("adamw", False): 16,
         ("sgd", True): 10, ("sgd_momentum", True): 14, ("adam", True): 18, ("adamw", True): 18}


def ex1(training_memory_bytes):
    def test():
        n = 1_000_003
        for (opt, mp), per in TABLE.items():
            got = training_memory_bytes(n, opt, mp)
            assert isinstance(got, int), f"return an int (bytes), got {type(got).__name__}"
            assert got == per * n, (f"{opt}, mixed_precision={mp}: expected {per} bytes/param = {per * n:,}, "
                                    f"got {got:,} ({got / n:.2f} bytes/param)")
        assert training_memory_bytes(10, "Adam", False) == 160, "optimizer names should be case-insensitive"
        try:
            training_memory_bytes(10, "lion", False)
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown optimizer should raise ValueError")

    return run_check("Exercise 1 · training_memory_bytes", test,
                     hint="4 (weights) + 4 (grads) + state (sgd 0, momentum 4, adam 8) + 2 if mixed precision")


def ex2(accumulated_step):
    def test():
        import torch

        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 3))
        sizes = [3, 5, 8]  # unequal micro-batches: a plain loss / len(batches) is wrong
        xs = [torch.randn(s, 6) for s in sizes]
        ys = [torch.randint(0, 3, (s,)) for s in sizes]

        def loss_fn(m, x, y):
            return torch.nn.functional.cross_entropy(m(x), y)

        model.zero_grad()
        big = loss_fn(model, torch.cat(xs), torch.cat(ys))
        big.backward()
        want = [p.grad.clone() for p in model.parameters()]
        for p in model.parameters():  # stale gradients the student must clear
            p.grad = torch.full_like(p, 7.0)
        out = accumulated_step(model, list(zip(xs, ys)), loss_fn)
        assert isinstance(out, float), f"return the mean loss as a Python float, got {type(out).__name__}"
        assert close(out, big.item(), 1e-5), f"returned loss {out:.6f}, big-batch mean loss is {big.item():.6f}"
        for (name, p), w in zip(model.named_parameters(), want):
            assert p.grad is not None, f"{name}.grad is None - did you call backward()?"
            err = (p.grad - w).abs().max().item()
            assert err < 1e-5, (f"{name}: gradient differs from the big batch by {err:.2e}. Did you zero the "
                                "old gradients, and weight each micro-batch by its size / total size?")

    return run_check("Exercise 2 · accumulated_step", test,
                     hint="zero grads; N = sum of sizes; for each (x, y): (loss_fn(model, x, y) * len(x) / N)"
                          ".backward(); sum the weighted loss.item() values")
