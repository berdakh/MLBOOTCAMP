"""Reference solutions for Level 3 notebook 10. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math


def ex1(n_params, world_size, stage):
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2 or 3")
    params, grads, optim = 2.0 * n_params, 2.0 * n_params, 12.0 * n_params
    if stage >= 1:
        optim /= world_size
    if stage >= 2:
        grads /= world_size
    if stage >= 3:
        params /= world_size
    return params + grads + optim


def ex2(n_params, tokens, gpu_tflops, mfu, n_gpus):
    flops = 6.0 * n_params * tokens
    seconds = flops / (n_gpus * gpu_tflops * 1e12 * mfu)
    return flops, seconds / 86_400


def ex3(checkpoint_cost_s, mtbf_s):
    return math.sqrt(2.0 * checkpoint_cost_s * mtbf_s)
