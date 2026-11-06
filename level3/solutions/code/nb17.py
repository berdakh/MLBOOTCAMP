"""Reference solutions for Level 3 notebook 17. Graded by tests/test_solutions.py."""

from __future__ import annotations


def ex1(entities, features, bytes_per, versions, replication):
    if min(entities, features, bytes_per, versions, replication) < 0:
        raise ValueError("all inputs must be non-negative")
    return float(entities * features * bytes_per * versions * replication / 1e9)


def ex2(hours, gpus, price):
    if min(hours, gpus, price) < 0:
        raise ValueError("hours, gpus and price must be non-negative")
    return float(hours * gpus * price)


def ex3(hours, gpus, price, utilization):
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be a fraction in (0, 1], got {utilization}")
    total = ex2(hours, gpus, price)
    return {
        "total_cost": total,
        "useful_cost": total * utilization,
        "wasted_cost": total * (1 - utilization),
        "cost_per_useful_gpu_hour": price / utilization,
    }
