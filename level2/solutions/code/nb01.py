"""Reference solutions for Level 2 notebook 01. Graded by tests/test_solutions.py."""

from __future__ import annotations

import hashlib

import numpy as np


def ex1(user_ids, pct: int, salt: str = "") -> np.ndarray:
    def bucket(user_id) -> int:
        digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
        return int(digest[:16], 16) % 100

    return np.array([bucket(u) < pct for u in user_ids], dtype=bool)


def ex2(impressions_dir: str) -> str:
    return f"""
    SELECT day, position,
           count(*)                AS impressions,
           sum(clicked::INT)       AS clicks,
           avg(clicked::DOUBLE)    AS ctr
    FROM read_parquet('{impressions_dir}/*/*.parquet', hive_partitioning = true)
    WHERE day BETWEEN 7 AND 13          -- on the partition column: prunes files
    GROUP BY day, position
    ORDER BY day, position
    """
