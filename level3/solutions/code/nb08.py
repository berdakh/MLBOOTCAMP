"""Reference solutions for Level 3 notebook 08. Graded by tests/test_solutions.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ex1(q, d, temperature=0.05):
    logits = q @ d.T / temperature  # (B, N): row i's positive is column i
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


def ex2(ranked_ids, positives, n, skip_top=0):
    pos = set(positives)
    return [d for d in list(ranked_ids)[skip_top:] if d not in pos][:n]
