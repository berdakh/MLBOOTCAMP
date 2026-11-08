"""Self-checks for Level 2 notebook 19 · Recommendation & search system design."""

from __future__ import annotations

from churnlab.checks import close, run_check


def ex1(embedding_memory_gb):
    def test():
        cases = [((1_000_000, 64, 4), 0.256), ((300_000_000, 128, 4), 153.6), ((50_000_000, 256, 2), 25.6),
                 ((10, 8, 1), 8e-8), ((0, 128, 4), 0.0)]
        for args, want in cases:
            got = embedding_memory_gb(*args)
            assert isinstance(got, float), f"return a float, got {type(got).__name__} for {args}"
            assert close(got, want, 1e-9), f"embedding_memory_gb{args} = {got}, expected {want} (GB = 1e9 bytes)"

    return run_check("Exercise 1 · embedding_memory_gb", test,
                     hint="rows x dims x bytes per number, divided by 1e9 (decimal gigabytes)")


def ex2(qps):
    def test():
        cases = [((86_400, 1, 1.0), 1.0), ((50_000_000, 20, 2.5), 50_000_000 * 20 / 86_400 * 2.5),
                 ((1_000_000, 20, 3), 694.4444444444445), ((0, 10, 3), 0.0)]
        for args, want in cases:
            got = qps(*args)
            assert isinstance(got, float), f"return a float, got {type(got).__name__} for {args}"
            assert close(got, want, 1e-9), f"qps{args} = {got}, expected {want:.4f}"

    return run_check("Exercise 2 · qps", test,
                     hint="requests per day = dau x req_per_user; / 86,400 seconds; x peak factor")
