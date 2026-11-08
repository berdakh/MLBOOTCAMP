"""Reference solutions for Level 3 notebook 05. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import torch


def ex1(probs, labels, n_bins=15):
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)  # (lo, hi]; 1.0 -> last bin
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def ex2(logits, labels):
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)  # optimise log T: T stays > 0
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())
