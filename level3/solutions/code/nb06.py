"""Reference solutions for Level 3 notebook 06. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math

import torch
from torch import nn


class ex1(nn.Module):  # noqa: N801 - graded as ``ex1``; in the notebook it is called LoRALinear
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def ex2(W, A, B, alpha, r):
    return W + (alpha / r) * (B @ A)
