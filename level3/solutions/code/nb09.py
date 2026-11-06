"""Reference solutions for Level 3 notebook 09. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math


def ex1(n_params, bytes_per, world_size):
    # reduce-scatter (N-1 steps) + all-gather (N-1 steps), each step sends one 1/N chunk
    return 2 * (world_size - 1) / world_size * n_params * bytes_per


def ex2(grads_per_rank):
    import torch

    world = len(grads_per_rank)
    return [torch.stack([rank[i] for rank in grads_per_rank]).sum(0) / world
            for i in range(len(grads_per_rank[0]))]


def ex3(n, world_size, rank, epoch, seed, shuffle):
    import torch

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        indices = torch.randperm(n, generator=g).tolist()
    else:
        indices = list(range(n))
    total = math.ceil(n / world_size) * world_size
    pad = total - n
    if pad:
        indices += (indices * math.ceil(pad / n))[:pad]
    return indices[rank:total:world_size]
