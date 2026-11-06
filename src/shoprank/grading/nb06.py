"""Self-checks for Level 2 notebook 06 · Learning to rank."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.checks import run_check


def _slow_pairwise(scores, labels, groups):
    good = total = 0.0
    n = len(scores)
    for i in range(n):
        for j in range(n):
            if groups[i] == groups[j] and labels[i] > labels[j]:
                total += 1
                good += 1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0
    return good / total if total else float("nan")


def ex1(pairwise_accuracy):
    def test():
        # One group, perfectly ordered / reversed.
        assert abs(pairwise_accuracy(np.array([3.0, 2.0, 1.0]), np.array([2, 1, 0]), np.array([0, 0, 0])) - 1.0) < 1e-12, \
            "a perfectly ordered group should give 1.0"
        assert abs(pairwise_accuracy(np.array([1.0, 2.0, 3.0]), np.array([2, 1, 0]), np.array([0, 0, 0]))) < 1e-12, \
            "a reversed group should give 0.0"
        # Ties in score count half; equal labels are not pairs.
        got = pairwise_accuracy(np.array([1.0, 1.0, 0.0]), np.array([1, 0, 0]), np.array([5, 5, 5]))
        assert abs(got - 0.75) < 1e-12, f"scores [1,1,0], labels [1,0,0]: expected 0.75 (one tie = 0.5, one correct), got {got}"
        # Pairs across groups must not count.
        got = pairwise_accuracy(np.array([1.0, 5.0, 0.0, 6.0]), np.array([1, 0, 0, 1]), np.array([0, 1, 0, 1]))
        assert abs(got - 1.0) < 1e-12, (
            f"groups [0,1,0,1]: within each group the clicked item scores higher, so expected 1.0, got {got} "
            "(are you comparing items across groups?)")
        # No informative pair -> nan.
        got = pairwise_accuracy(np.array([1.0, 2.0]), np.array([1, 1]), np.array([0, 0]))
        assert got != got, f"no pair with different labels: expected nan, got {got}"
        # Random, interleaved groups, graded labels vs a slow reference.
        rng = np.random.default_rng(0)
        s = rng.integers(0, 4, 60).astype(float)
        y = rng.integers(0, 3, 60)
        g = rng.integers(0, 7, 60)
        want = _slow_pairwise(s, y, g)
        got = pairwise_accuracy(s, y, g)
        assert abs(got - want) < 1e-9, f"random interleaved groups with ties: expected {want:.6f}, got {got:.6f}"
        got = pairwise_accuracy(list(s), list(y), list(g))
        assert abs(got - want) < 1e-9, "should accept plain lists too"

    return run_check("Exercise 1 · pairwise_accuracy", test,
                     hint="for each group: pairs = labels[:, None] > labels[None, :]; "
                          "d = (scores[:, None] - scores[None, :])[pairs]; good += (d > 0).sum() + 0.5 * (d == 0).sum()")


def ex2(make_groups):
    def test():
        df = pd.DataFrame({"request_id": [7, 3, 7, 3, 9, 7, 3, 3],
                           "item": [0, 1, 2, 3, 4, 5, 6, 7]})
        out = make_groups(df.copy(), "request_id")
        assert isinstance(out, tuple) and len(out) == 2, "return a tuple (sorted_df, sizes)"
        sorted_df, sizes = out
        assert isinstance(sorted_df, pd.DataFrame), "first element must be the sorted DataFrame"
        sizes = np.asarray(sizes)
        assert sorted_df["request_id"].tolist() == [3, 3, 3, 3, 7, 7, 7, 9], \
            f"rows must be sorted by request_id, got {sorted_df['request_id'].tolist()}"
        assert sizes.tolist() == [4, 3, 1], f"sizes must follow the sorted order: expected [4, 3, 1], got {sizes.tolist()}"
        assert sorted_df["item"].tolist() == [1, 3, 6, 7, 0, 2, 5, 4], (
            "order inside each group must be preserved (use a stable sort, kind='stable'); "
            f"got items {sorted_df['item'].tolist()}")
        assert int(sizes.sum()) == len(df), "sizes must add up to the number of rows"
        # Larger random case, string group ids.
        rng = np.random.default_rng(1)
        big = pd.DataFrame({"q": rng.choice(list("abcdefghij"), 500), "x": np.arange(500)})
        s2, z2 = make_groups(big, "q")
        z2 = np.asarray(z2)
        assert z2.tolist() == big["q"].value_counts().sort_index().tolist(), "sizes wrong on a random 500-row frame"
        assert (s2.groupby("q")["x"].apply(lambda v: v.is_monotonic_increasing)).all(), "within-group order not preserved"

    return run_check("Exercise 2 · make_groups", test,
                     hint="out = df.sort_values(group_col, kind='stable'); "
                          "sizes = out.groupby(group_col, sort=False).size().to_numpy()")
