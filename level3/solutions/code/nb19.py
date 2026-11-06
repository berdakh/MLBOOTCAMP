"""Reference solutions for Level 3 notebook 19. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math


def ex1(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    if step < 0:
        raise ValueError("step must be >= 0")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")
    if step < warmup_steps:
        return float(base_lr * (step + 1) / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return float(min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)))


def ex2():
    # A: subject leakage vs the "new users" claim; B: tuning on test; C: under-tuned baselines;
    # D: best-of-seeds, no uncertainty; E: normalisation fitted before the split.
    return {"A", "B", "C", "D", "E"}
