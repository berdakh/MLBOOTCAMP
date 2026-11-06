"""Reference solutions for Level 3 notebook 04. Graded by tests/test_solutions.py."""

from __future__ import annotations

import torch


def ex1(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # (B, T, 1)
    return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


def ex2(lengths: list[int], batch_size: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
