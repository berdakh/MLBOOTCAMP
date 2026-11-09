"""Reference solutions for Level 3 notebook 11. Graded by tests/test_solutions.py."""

from __future__ import annotations


def ex1(W):
    import torch

    scale = W.abs().amax(dim=1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # an all-zero row: any scale works
    q = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale


def ex2(student_logits, teacher_logits, labels, T, alpha):
    import torch.nn.functional as F

    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1), F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
