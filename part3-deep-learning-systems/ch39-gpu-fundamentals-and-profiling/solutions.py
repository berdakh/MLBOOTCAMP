"""Reference solutions for Level 3 notebook 01. Graded by tests/test_solutions.py."""

from __future__ import annotations

import statistics
import time

from scaleml.device import sync


def ex2(n: int, seconds: float) -> float:
    return float(2 * n**3 / seconds / 1e12)


def ex3(flops: float, bytes_moved: float) -> float:
    if bytes_moved <= 0:
        raise ValueError("bytes must be positive")
    return float(flops / bytes_moved)


def ex1(fn, device, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync(device)  # wait for any queued GPU work before starting the clock
        start = time.perf_counter()
        fn()
        sync(device)  # ...and for fn's own kernels before stopping it
        times.append(time.perf_counter() - start)
    return float(statistics.median(times))
