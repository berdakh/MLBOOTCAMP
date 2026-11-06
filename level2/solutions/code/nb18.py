"""Reference solutions for Level 2 notebook 18 (graded in CI)."""

from __future__ import annotations

import numpy as np


def ex1(a, b, atol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    over = np.isnan(diff) | (diff > atol)
    max_abs = float(np.inf) if np.isnan(diff).any() else float(diff.max(initial=0.0))
    return {"max_abs_diff": max_abs, "n_over": int(over.sum()), "ok": bool(not over.any())}


def ex2(student_logits, teacher_probs, labels, alpha, T):
    import torch
    from torch.nn import functional as F

    p = teacher_probs.clamp(1e-6, 1 - 1e-6)
    soft_target = torch.sigmoid(torch.logit(p) / T)
    soft = F.binary_cross_entropy_with_logits(student_logits / T, soft_target) * T * T
    hard = F.binary_cross_entropy_with_logits(student_logits, labels.float())
    return alpha * soft + (1 - alpha) * hard
