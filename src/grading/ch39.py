"""Self-checks for Level 3 notebook 01 · GPU fundamentals & profiling."""

from __future__ import annotations

import time

from churnlab.checks import close, run_check

REQUIRES = {"torch"}


def ex2(tflops):
    def test():
        v = tflops(1024, 0.01)
        assert isinstance(v, float), f"return a Python float, got {type(v).__name__}"
        assert close(v, 2 * 1024**3 / 0.01 / 1e12, 1e-9), (
            f"tflops(1024, 0.01) = {v!r}; expected 2*n^3 FLOPs / seconds / 1e12 = {2 * 1024**3 / 0.01 / 1e12:.6f}")
        assert close(tflops(4096, 0.02), 2 * 4096**3 / 0.02 / 1e12, 1e-9), "wrong for n=4096"
        assert close(tflops(2048, 0.004), 8 * tflops(1024, 0.004), 1e-9), "doubling n should multiply FLOPs by 8"

    return run_check("Exercise 2 · tflops", test,
                     hint="an n x n matmul does n^2 dot products of length n: n multiplies + n adds each -> 2 n^3 FLOPs")


def ex3(arithmetic_intensity):
    def test():
        n = 1024
        ai = arithmetic_intensity(2 * n**3, 3 * n * n * 4)
        assert isinstance(ai, float), f"return a Python float, got {type(ai).__name__}"
        assert close(ai, n / 6, 1e-9), f"fp32 matmul n=1024: expected n/6 = {n / 6:.3f} FLOP/byte, got {ai}"
        assert close(arithmetic_intensity(10**6, 12 * 10**6), 1 / 12, 1e-9), "x + y on 1e6 fp32 values is 1/12"
        try:
            arithmetic_intensity(1, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("bytes <= 0 should raise ValueError (an intensity needs memory traffic)")

    return run_check("Exercise 3 · arithmetic_intensity", test,
                     hint="FLOPs divided by bytes moved; raise ValueError when bytes <= 0")


def ex1(time_fn):
    def test():
        import torch

        # warm-up calls are slow (like a first CUDA call); one timed call is an outlier
        durations = [0.06, 0.06] + [0.002, 0.002, 0.04, 0.002, 0.002]
        calls = []

        def fn():
            time.sleep(durations[min(len(calls), len(durations) - 1)])
            calls.append(1)

        out = time_fn(fn, torch.device("cpu"), 2, 5)
        assert len(calls) == 7, f"expected warmup + iters = 7 calls, fn was called {len(calls)} times"
        assert isinstance(out, float), f"return seconds as a Python float, got {type(out).__name__}"
        assert out < 0.02, (f"got {out:.4f} s: the median of the 5 timed calls is ~0.002 s. Did you time the "
                            "warm-up calls, or return the mean (one slow outlier pulls it up)?")
        assert out >= 0.0015, f"got {out:.4f} s, shorter than the calls themselves - are you timing fn()?"

    return run_check("Exercise 1 · time_fn", test,
                     hint="call fn() `warmup` times untimed; then for each iter: sync, perf_counter, fn(), sync, "
                          "perf_counter; return statistics.median of the times")
