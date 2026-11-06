"""Reference solutions for notebook 09. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math


def ex1(model, loader, optimizer, loss_fn) -> float:
    model.train()
    total, count = 0.0, 0
    for *inputs, y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(*inputs), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)


def ex2(pos_rate: float) -> float:
    p = float(pos_rate)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def ex3() -> list[str]:
    return ["C", "D", "E"]
