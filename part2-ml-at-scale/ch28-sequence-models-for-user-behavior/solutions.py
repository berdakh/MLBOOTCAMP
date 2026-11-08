"""Reference solutions for notebook 10. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np


def ex1(df, max_len):
    d = df.sort_values(["user_id", "ts"], kind="stable")
    users = np.sort(d["user_id"].unique())
    seqs = np.zeros((len(users), max_len), dtype=np.int64)
    for r, (_, items) in enumerate(d.groupby("user_id", sort=True)["item_id"]):
        last = items.to_numpy()[-max_len:] + 1
        seqs[r, max_len - len(last):] = last
    return users, seqs


def ex2(n):
    import torch

    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


def ex3(lengths, n):
    import torch

    lengths = torch.as_tensor(lengths).reshape(-1, 1)
    return torch.arange(n).reshape(1, -1) < (n - lengths)
