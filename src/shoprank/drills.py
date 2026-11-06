"""Small tools for interview drills (notebooks 13-16): timing, growth, gradient checks.

* ``Timer``            - ``with Timer("label"): ...`` measures wall-clock time.
* ``time_call``        - best-of-``repeat`` time of one call (inputs built outside the clock).
* ``growth_table``     - time a function at growing sizes and estimate the exponent
                         (slope of log time vs log n) - how you *measure* complexity.
* ``check_scaling``    - what the graders use: fail if a run, or the projected next
                         run of a quadratic algorithm, exceeds a time budget.
* ``numerical_gradient`` / ``relative_error`` - central-difference gradient checks.

Timings depend on the machine; every budget here is generous (an O(n log n)
solution typically uses a few percent of it) so a slow Colab CPU still passes.
"""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Sequence

import numpy as np


class Timer:
    """Context manager: ``with Timer("sort") as t: ...`` then ``t.elapsed`` (seconds)."""

    def __init__(self, label: str | None = None, verbose: bool = True):
        self.label, self.verbose, self.elapsed = label, verbose, 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start
        if self.verbose and self.label:
            print(f"{self.label}: {self.elapsed * 1000:.1f} ms")


def time_call(fn: Callable, *args, repeat: int = 1):
    """Return ``(result, best_seconds)`` of ``fn(*args)`` over ``repeat`` calls.

    Like ``timeit``, the garbage collector is paused while timing: a collection
    triggered by someone else's objects should not count against your function.
    """
    best, result = math.inf, None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeat):
            start = time.perf_counter()
            result = fn(*args)
            best = min(best, time.perf_counter() - start)
    finally:
        if gc_was_enabled:
            gc.enable()
    return result, best


def growth_table(fn: Callable, make_args: Callable[[int], tuple], sizes: Sequence[int],
                 repeat: int = 1, verbose: bool = True) -> list[tuple[int, float]]:
    """Time ``fn(*make_args(n))`` for each n; print times and the local exponent.

    The exponent between consecutive sizes is log(t2/t1) / log(n2/n1): about 1 for
    O(n), slightly above 1 for O(n log n), about 2 for O(n^2). Tiny timings are noisy,
    so trust the exponent between the larger sizes, and the least-squares slope
    over all sizes (also printed), more than any single ratio.
    """
    rows: list[tuple[int, float]] = []
    for n in sizes:
        args = make_args(n)
        _, t = time_call(fn, *args, repeat=repeat)
        rows.append((n, t))
    if verbose:
        print(f"{'n':>10} {'time (ms)':>11} {'exponent':>9}")
        for i, (n, t) in enumerate(rows):
            if i == 0 or t <= 0 or rows[i - 1][1] <= 0:
                expo = ""
            else:
                expo = f"{math.log(t / rows[i - 1][1]) / math.log(n / rows[i - 1][0]):9.2f}"
            print(f"{n:>10} {t * 1000:>11.2f} {expo:>9}")
        ns, ts = zip(*rows)
        if len(rows) >= 2 and min(ts) > 0:
            slope = np.polyfit(np.log(ns), np.log(ts), 1)[0]
            print(f"{'fitted exponent over all sizes':>31}: {slope:.2f}")
    return rows


def check_scaling(fn: Callable, make_args: Callable[[int], tuple], sizes: Sequence[int],
                  budget: float = 2.0, exponent: float = 2.0, label: str = "your function",
                  validate: Callable | None = None) -> list[tuple[int, float]]:
    """Raise ``AssertionError`` if ``fn`` is too slow on large inputs.

    Sizes are run in increasing order. Each run must finish within ``budget``
    seconds. Before the next size is run, its time is projected from the last one
    assuming growth ``n**exponent`` (or the growth observed between the last two
    sizes, if it is far steeper - this catches exponential recursion); if that projection
    exceeds ``3 * budget`` the
    check fails without running it (so a quadratic solution fails quickly instead
    of making you wait minutes). ``validate(n, args, result)`` may assert on the
    output, so a fast-but-wrong answer cannot pass.
    """
    rows: list[tuple[int, float]] = []
    for n in sizes:
        if rows:
            n0, t0 = rows[-1]
            growth = exponent
            if len(rows) >= 2 and rows[-2][1] >= 5e-3:      # observed growth, once timings are above noise
                (n_a, t_a) = rows[-2]
                observed = math.log(max(t0, 1e-9) / t_a) / math.log(n0 / n_a)
                if observed > exponent + 1:                  # clearly super-polynomial (e.g. exponential recursion)
                    growth = observed
            projected = t0 * (n / n0) ** growth
            assert projected <= 3 * budget, (
                f"{label} took {t0:.2f} s at n={n0:,}; at that growth rate n={n:,} would take "
                f"~{projected:,.3g} s (budget {budget:.0f} s) - a solution with the expected complexity "
                "takes well under a second here")
        args = make_args(n)
        result, t = time_call(fn, *args)
        assert t <= budget, (f"{label} took {t:.2f} s at n={n:,} (budget {budget:.0f} s) - "
                             "look for a nested loop or a repeated O(n) operation inside a loop")
        if validate is not None:
            validate(n, args, result)
        rows.append((n, t))
    return rows


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar ``f`` at ``x`` (any shape, float64).

    Costs 2 * x.size evaluations of f, so use it on small problems only. ``x`` is
    perturbed in place and restored.
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a, b) -> float:
    """||a-b|| / (||a|| + ||b||): below ~1e-6 (float64) means an analytic gradient is right."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12))
