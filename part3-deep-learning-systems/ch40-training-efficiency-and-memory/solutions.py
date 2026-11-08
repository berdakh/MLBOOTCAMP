"""Reference solutions for Level 3 notebook 02. Graded by tests/test_solutions.py."""

from __future__ import annotations

STATE_BYTES = {"sgd": 0, "sgd_momentum": 4, "adam": 8, "adamw": 8}


def ex1(n_params: int, optimizer: str, mixed_precision: bool) -> int:
    key = optimizer.lower()
    if key not in STATE_BYTES:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    per_param = 4 + 4 + STATE_BYTES[key] + (2 if mixed_precision else 0)
    return int(per_param * n_params)


def ex2(model, batches, loss_fn) -> float:
    model.zero_grad(set_to_none=True)
    total = sum(len(x) for x, _ in batches)
    mean_loss = 0.0
    for x, y in batches:
        loss = loss_fn(model, x, y) * (len(x) / total)  # mean over ALL examples, not over micro-batches
        loss.backward()  # gradients add up in .grad
        mean_loss += loss.item()
    return float(mean_loss)
