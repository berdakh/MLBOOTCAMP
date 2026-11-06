"""Reference solutions for Level 3 notebook 03. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def ex1(x: torch.Tensor, w_qkv: torch.Tensor, w_out: torch.Tensor, n_heads: int) -> torch.Tensor:
    B, T, C = x.shape
    D = C // n_heads
    q, k, v = (x @ w_qkv).split(C, dim=-1)
    q, k, v = (t.view(B, T, n_heads, D).transpose(1, 2) for t in (q, k, v))  # (B, H, T, D)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)  # (B, H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    weights = F.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    y = (weights @ v).transpose(1, 2).reshape(B, T, C)
    return y @ w_out


def ex2(logits: torch.Tensor, k: int, temperature: float, generator: torch.Generator | None = None) -> torch.Tensor:
    logits = logits / temperature
    if k < logits.size(-1):
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)
