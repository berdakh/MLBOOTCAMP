"""Reference solutions for Level 2 notebook 08 (graded in CI by tests/test_solutions.py)."""

from __future__ import annotations

import numpy as np


def ex1(user_emb, item_emb, log_q=None):
    import torch
    import torch.nn.functional as F

    logits = user_emb @ item_emb.T
    if log_q is not None:
        logits = logits - log_q[None, :]
    return F.cross_entropy(logits, torch.arange(logits.shape[0], device=logits.device))


def ex2(approx_ids, exact_ids) -> float:
    vals = []
    for a, e in zip(np.asarray(approx_ids), np.asarray(exact_ids)):
        e_set = {int(x) for x in e if x >= 0}
        if e_set:
            vals.append(len({int(x) for x in a if x >= 0} & e_set) / len(e_set))
    return float(np.mean(vals)) if vals else float("nan")
