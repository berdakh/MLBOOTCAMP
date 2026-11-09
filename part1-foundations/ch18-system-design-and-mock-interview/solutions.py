"""Reference solutions for notebook 17. Graded by tests/test_solutions.py."""

from __future__ import annotations

SECONDS_PER_DAY = 86_400


def ex1(dau: float, requests_per_user_per_day: float, peak_factor: float) -> float:
    average_qps = dau * requests_per_user_per_day / SECONDS_PER_DAY
    return float(average_qps * peak_factor)


def ex2(total_ms: float, components: dict[str, float]) -> float:
    spent = sum(components.values())
    if spent > total_ms:
        raise ValueError(f"components need {spent} ms, {spent - total_ms} ms over the {total_ms} ms budget")
    return total_ms - spent
