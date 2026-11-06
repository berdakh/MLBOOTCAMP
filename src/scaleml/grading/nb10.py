"""Self-checks for Level 3 notebook 10 · Scaling to large models."""

from __future__ import annotations

import math

from churnlab.checks import close, run_check


def ex1(zero_memory_per_gpu):
    def test():
        psi = 1_000_000_000
        cases = [  # (world_size, stage, bytes per GPU) - ZeRO paper accounting, K = 12
            (1, 0, 16e9), (8, 0, 16e9), (8, 1, 4e9 + 12e9 / 8), (8, 2, 2e9 + 14e9 / 8), (8, 3, 16e9 / 8),
            (64, 1, 4e9 + 12e9 / 64), (64, 2, 2e9 + 14e9 / 64), (64, 3, 16e9 / 64), (1, 3, 16e9),
        ]
        for world, stage, want in cases:
            got = zero_memory_per_gpu(psi, world, stage)
            assert close(got, want, 1e-9), (f"{psi:,} params, {world} GPUs, stage {stage}: expected "
                                             f"{want / 1e9:.4f} GB, got {got / 1e9:.4f} GB")
        got7 = zero_memory_per_gpu(7_000_000_000, 64, 3)
        assert close(got7, 112e9 / 64, 1e-9), f"7B on 64 GPUs, stage 3: expected 1.75 GB, got {got7 / 1e9:.3f} GB"
        for bad in (4, -1):
            try:
                zero_memory_per_gpu(psi, 8, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"stage={bad} should raise ValueError")

    return run_check("Exercise 1 · zero_memory_per_gpu", test,
                     hint="params 2, grads 2, optimizer 12 bytes/param; stage 1 divides the 12 by N, "
                          "stage 2 also the grads, stage 3 also the params")


def ex2(train_flops_and_days):
    def test():
        out = train_flops_and_days(1e9, 20e9, 100.0, 0.5, 8)
        assert isinstance(out, tuple) and len(out) == 2, "return a tuple (flops, days)"
        flops, days = out
        assert close(flops, 1.2e20, 1e-9), f"6 x 1e9 x 20e9 = 1.2e20 FLOPs, got {flops:.3e}"
        want = 1.2e20 / (8 * 100e12 * 0.5) / 86400
        assert close(days, want, 1e-9), f"expected {want:.4f} wall-clock days, got {days:.4f}"
        f2, d2 = train_flops_and_days(175e9, 300e9, 312.0, 0.4, 1024)
        assert close(f2, 3.15e23, 1e-9), f"GPT-3-sized: expected 3.15e23 FLOPs, got {f2:.3e}"
        assert close(d2, 3.15e23 / (1024 * 312e12 * 0.4) / 86400, 1e-9), f"GPT-3-sized days wrong: {d2:.3f}"
        _, d_half = train_flops_and_days(1e9, 20e9, 100.0, 0.25, 8)
        assert close(d_half, 2 * want, 1e-9), "halving MFU should double the days"
        _, d16 = train_flops_and_days(1e9, 20e9, 100.0, 0.5, 16)
        assert close(d16, want / 2, 1e-9), "doubling the GPUs should halve the wall-clock days"

    return run_check("Exercise 2 · train_flops_and_days", test,
                     hint="flops = 6 N D; seconds = flops / (n_gpus x gpu_tflops x 1e12 x mfu); days = seconds / 86400")


def ex3(optimal_checkpoint_interval):
    def test():
        for c, m in [(60.0, 86400.0), (300.0, 3600.0 * 8), (10.0, 1000.0), (1.0, 1.0)]:
            got = optimal_checkpoint_interval(c, m)
            want = math.sqrt(2 * c * m)
            assert close(got, want, 1e-9), f"C={c} s, MTBF={m} s: expected {want:.2f} s, got {got:.2f} s"
        a = optimal_checkpoint_interval(60.0, 86400.0)
        b = optimal_checkpoint_interval(240.0, 86400.0)
        assert close(b / a, 2.0, 1e-9), "4x the checkpoint cost should double the interval (square root)"

    return run_check("Exercise 3 · optimal_checkpoint_interval", test, hint="Young (1974): tau = sqrt(2 C M)")
