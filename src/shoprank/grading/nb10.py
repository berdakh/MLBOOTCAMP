"""Self-checks for notebook 10 · Sequence models for user behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.checks import run_check

REQUIRES = {"torch"}


def ex1(build_sequences):
    def test():
        ts = pd.to_datetime(["2025-03-03 10:00", "2025-03-03 09:00", "2025-03-03 11:00",
                             "2025-03-04 08:00", "2025-03-03 09:00", "2025-03-03 12:00",
                             "2025-03-03 09:00"])
        df = pd.DataFrame({"user_id": [7, 7, 7, 3, 7, 7, 3],
                           "item_id": [10, 11, 12, 0, 13, 14, 5],
                           "ts": ts})
        # user 7 in time order: 11 (09:00), 13 (09:00, later row), 10, 12, 14; user 3: 5, 0
        users, seqs = build_sequences(df, 4)
        users = np.asarray(users)
        seqs = np.asarray(seqs)
        assert list(users) == [3, 7], f"users should be the sorted distinct ids [3, 7], got {list(users)}"
        assert seqs.shape == (2, 4), f"expected shape (2, 4), got {seqs.shape}"
        assert seqs.dtype == np.int64, f"expected dtype int64, got {seqs.dtype}"
        assert list(seqs[0]) == [0, 0, 6, 1], (
            f"user 3 should be [0, 0, 6, 1] (items 5 then 0, shifted +1, left-padded), got {list(seqs[0])}")
        assert list(seqs[1]) == [14, 11, 13, 15], (
            f"user 7 should keep its LAST 4 items in time order [14, 11, 13, 15], got {list(seqs[1])} "
            "(ties at 09:00 keep their row order; 'last' means most recent)")
        _, long = build_sequences(df, 7)
        assert list(np.asarray(long)[1]) == [0, 0, 12, 14, 11, 13, 15], "with max_len=7 user 7 should be fully kept"
        _, one = build_sequences(df.iloc[[3]], 3)
        assert list(np.asarray(one)[0]) == [0, 0, 1], "item_id 0 must become 1 (0 is reserved for padding)"

    return run_check("Exercise 1 · build_sequences", test,
                     hint="sort_values(['user_id', 'ts'], kind='stable'); per user take items[-max_len:] + 1 "
                          "and write them into the RIGHT end of a row of zeros")


def ex2(causal_mask):
    def test():
        import torch

        for n in (1, 4, 7):
            m = causal_mask(n)
            assert isinstance(m, torch.Tensor) and m.dtype == torch.bool, "return a torch.bool tensor"
            assert tuple(m.shape) == (n, n), f"expected shape ({n}, {n}), got {tuple(m.shape)}"
            want = torch.tensor([[j > i for j in range(n)] for i in range(n)])
            assert torch.equal(m, want), (
                f"for n={n} expected True strictly above the diagonal (j > i), got\n{m.int()}")

    return run_check("Exercise 2 · causal_mask", test,
                     hint="torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)")


def ex3(padding_mask):
    def test():
        import torch

        m = padding_mask([2, 4, 0, 1], 4)
        assert isinstance(m, torch.Tensor) and m.dtype == torch.bool, "return a torch.bool tensor"
        assert tuple(m.shape) == (4, 4), f"expected shape (4, 4), got {tuple(m.shape)}"
        want = torch.tensor([[True, True, False, False], [False] * 4, [True] * 4, [True, True, True, False]])
        assert torch.equal(m, want), f"LEFT padding: the last `length` positions are real. Expected\n{want.int()}\ngot\n{m.int()}"
        m2 = padding_mask(torch.tensor([3]), 5)
        assert torch.equal(m2, torch.tensor([[True, True, False, False, False]])), "lengths given as a tensor failed"

    return run_check("Exercise 3 · padding_mask", test,
                     hint="torch.arange(n)[None, :] < (n - lengths)[:, None]")
