"""Self-checks for Level 2 notebook 07 · Candidate generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.checks import run_check


def ex1(covisitation):
    def test():
        pairs = pd.DataFrame({
            "session_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
            "item_id":    [10, 20, 30, 10, 20, 20, 30, 30, 40, 10, 50],
        })
        # Co-visit counts: (10,20): s1,s2 = 2 | (10,30): s1 = 1 | (20,30): s1,s3 = 2 | (10,40): s4 = 1
        got = covisitation(pairs.copy(), 2)
        assert isinstance(got, dict), "return a dict item -> list of neighbours"
        got = {int(k): [int(x) for x in v] for k, v in got.items()}
        assert got.get(10) == [20, 30], (
            f"item 10: neighbours by count are 20 (2 sessions), then 30 and 40 (1 each; tie -> smaller id): "
            f"expected [20, 30], got {got.get(10)}")
        assert got.get(20) == [10, 30], f"item 20: expected [10, 30] (both 2 sessions, tie -> smaller id), got {got.get(20)}"
        assert got.get(30) == [20, 10], (
            f"item 30: 20 co-visits in 2 sessions (session 3 lists 30 twice: count once), 10 in 1; "
            f"expected [20, 10], got {got.get(30)}")
        assert got.get(40) == [10], f"item 40: expected [10], got {got.get(40)}"
        assert got.get(50, []) == [], "item 50 was alone in its session: it has no neighbours"
        for k, v in got.items():
            assert k not in v, f"item {k} lists itself as a neighbour"
        got3 = {int(k): [int(x) for x in v] for k, v in covisitation(pairs, 3).items()}
        assert got3.get(10) == [20, 30, 40], f"top_n=3 for item 10: expected [20, 30, 40], got {got3.get(10)}"
        # Bigger random case against a direct count.
        rng = np.random.default_rng(0)
        big = pd.DataFrame({"session_id": rng.integers(0, 60, 400), "item_id": rng.integers(0, 25, 400)})
        res = covisitation(big, 5)
        sets = big.groupby("session_id")["item_id"].apply(set)
        for item in (0, 7, 13):
            cnt = {}
            for s in sets:
                if item in s:
                    for j in s - {item}:
                        cnt[j] = cnt.get(j, 0) + 1
            want = sorted(cnt, key=lambda j: (-cnt[j], j))[:5]
            assert [int(x) for x in res.get(item, [])] == want, f"random case, item {item}: expected {want}, got {res.get(item)}"

    return run_check("Exercise 1 · covisitation", test,
                     hint="d = pairs_df.drop_duplicates(); p = d.merge(d, on='session_id'); drop item == item_nb; "
                          "count with groupby, sort by [item, -count, neighbour], groupby(item).head(top_n)")


def ex2(recall_at_k_batch):
    def test():
        cands = {1: [5, 6, 7, 8], 2: [1, 2, 3], 3: [9]}
        truth = {1: {6, 8, 100}, 2: {3}, 4: {1, 2}, 5: set()}
        # user 1: k=2 -> {6} of 3 = 1/3 ; user 2: k=2 -> {} = 0 ; user 4: missing -> 0 ; user 5: empty -> skipped
        got = float(recall_at_k_batch(cands, truth, 2))
        want = (1 / 3 + 0 + 0) / 3
        assert abs(got - want) < 1e-9, f"k=2: expected {want:.4f}, got {got:.4f}"
        got = float(recall_at_k_batch(cands, truth, 3))
        want = (1 / 3 + 1 + 0) / 3
        assert abs(got - want) < 1e-9, f"k=3: expected {want:.4f}, got {got:.4f} (only the first k candidates count)"
        got = float(recall_at_k_batch(cands, truth, 10))
        want = (2 / 3 + 1 + 0) / 3
        assert abs(got - want) < 1e-9, f"k=10 (longer than the lists): expected {want:.4f}, got {got:.4f}"
        got = float(recall_at_k_batch({1: np.array([6, 8])}, {1: [6, 8]}, 5))
        assert abs(got - 1.0) < 1e-9, "should accept arrays for candidates and lists for truth"

    return run_check("Exercise 2 · recall_at_k_batch", test,
                     hint="loop over truth: skip empty sets; got = set(list(candidates.get(u, []))[:k]); "
                          "append len(got & set(rel)) / len(rel); return the mean")
