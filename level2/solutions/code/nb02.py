"""Reference solutions for Level 2 notebook 02. Graded by tests/test_solutions.py."""

from __future__ import annotations

import random
from collections import Counter

from shoprank.scale import stable_bucket


def ex1(stream, k: int, seed: int) -> list:
    """Algorithm R: after i items, each has probability k / i of being in the reservoir."""
    rng = random.Random(seed)
    reservoir: list = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)  # inclusive: i + 1 choices
            if j < k:
                reservoir[j] = item
    return reservoir


def ex2(keys, n_partitions: int) -> dict:
    keys = list(keys)
    n_mappers = 4
    chunk = -(-len(keys) // n_mappers) if keys else 1
    partitions: list[list[tuple]] = [[] for _ in range(n_partitions)]
    for lo in range(0, len(keys), chunk):                      # MAP + COMBINE
        for key, c in Counter(keys[lo:lo + chunk]).items():
            partitions[stable_bucket(key, n_partitions)].append((key, c))   # SHUFFLE
    result: dict = {}
    for inbox in partitions:                                   # REDUCE
        merged: Counter = Counter()
        for key, c in inbox:
            merged[key] += c
        result.update(merged)                                  # partitions hold disjoint keys
    return result
