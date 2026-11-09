"""Reference solutions for Level 3 notebook 18. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math


def ex1(total_ms, stages):
    if total_ms <= 0:
        raise ValueError(f"total_ms must be positive, got {total_ms}")
    negative = {k: v for k, v in stages.items() if v < 0}
    if negative:
        raise ValueError(f"stages cannot be negative: {negative}")
    spent = sum(stages.values())
    if spent > total_ms:
        raise ValueError(f"stages total {spent} ms: over the {total_ms} ms budget by {spent - total_ms} ms")
    return {
        "remaining_ms": total_ms - spent,
        "shares": {k: v / total_ms for k, v in stages.items()},
        "largest_stage": max(stages, key=stages.get) if stages else None,
    }


def ex2(peak_qps, per_replica_qps, headroom):
    if per_replica_qps <= 0:
        raise ValueError("per_replica_qps must be positive")
    if peak_qps < 0:
        raise ValueError("peak_qps cannot be negative")
    if not 0 <= headroom < 1:
        raise ValueError(f"headroom must be in [0, 1), got {headroom}")
    return max(2, int(math.ceil(peak_qps / (per_replica_qps * (1 - headroom)))))
