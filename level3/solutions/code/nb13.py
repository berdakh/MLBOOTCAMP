"""Reference solutions for Level 3 notebook 13. Graded by tests/test_solutions.py."""

from __future__ import annotations

import hashlib

import numpy as np


def ex1(y, p, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    b = float(np.clip(y.mean(), eps, 1 - eps))
    entropy = -(b * np.log(b) + (1 - b) * np.log(1 - b))
    return float(log_loss / entropy)


def ex2(p, w):
    p = np.asarray(p, dtype=np.float64)
    return p / (p + (1 - p) / w)


def ex3(field, value, n_buckets: int) -> int:
    digest = hashlib.md5(f"{field}={value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % n_buckets
