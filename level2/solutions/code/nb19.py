"""Reference solutions for Level 2 notebook 19 (graded in CI)."""

from __future__ import annotations


def ex1(n, dim, bytes_per):
    return float(n * dim * bytes_per / 1e9)


def ex2(dau, req_per_user, peak):
    return float(dau * req_per_user / 86_400 * peak)
