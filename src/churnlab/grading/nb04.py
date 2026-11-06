"""Self-checks for notebook 04 · Leakage-safe splits & a baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import run_check

_KEYS = ["snapshot_date", "customer_id"]


def _keys(df: pd.DataFrame) -> list[tuple]:
    return sorted(map(tuple, df[_KEYS].astype(str).to_numpy().tolist()))


def ex1(time_split_with_embargo):
    def test():
        from ..bootstrap import ensure_features
        from ..splits import time_split

        df = ensure_features(verbose=False)
        cases = [
            ("2024-04-01", "2024-07-01", 30),
            ("2024-03-15", "2024-07-01", 30),  # mid-month boundary: the embargo bites
            ("2024-01-01", "2024-05-01", 60),
            ("2024-01-15", "2024-05-20", 1),
        ]
        for train_end, valid_end, gap in cases:
            got = time_split_with_embargo(df, pd.Timestamp(train_end), pd.Timestamp(valid_end), gap)
            assert isinstance(got, tuple) and len(got) == 3, "return a tuple (train, valid, test)"
            ref = time_split(df, pd.Timestamp(train_end), pd.Timestamp(valid_end), gap_days=gap)
            for name, mine, theirs in zip(("train", "valid", "test"), got, (ref.train, ref.valid, ref.test)):
                assert len(mine) == len(theirs), (
                    f"train_end={train_end}, valid_end={valid_end}, gap={gap}: {name} has {len(mine)} rows, "
                    f"expected {len(theirs)}")
                assert _keys(mine) == _keys(theirs), f"{name} has the right size but different rows ({train_end}, gap={gap})"

    return run_check("Exercise 1 · time_split_with_embargo", test,
                     hint="a row belongs to an earlier split only if snapshot_date + gap <= that split's end")


def _frame(dates, customers):
    return pd.DataFrame({"snapshot_date": pd.to_datetime(dates), "customer_id": customers})


def ex2(leakage_problems):
    def test():
        train = _frame(["2024-01-01", "2024-02-01", "2024-03-01"], ["A", "B", "C"])
        good_test = _frame(["2024-04-01", "2024-05-01"], ["D", "E"])
        problems = leakage_problems(train, good_test, 30)
        assert isinstance(problems, list), f"return a list of strings, got {type(problems).__name__}"
        assert problems == [], f"a clean split was flagged: {problems}"
        assert leakage_problems(train, good_test, 30, group_col="customer_id") == [], \
            "a clean split with disjoint customers was flagged"

        # label windows of March rows reach into the test period starting 2024-03-15
        overlap = _frame(["2024-03-15", "2024-04-15"], ["D", "E"])
        problems = leakage_problems(train, overlap, 30)
        assert problems and all(isinstance(p, str) for p in problems), (
            "train's last snapshot 2024-03-01 + 30 days > test's first snapshot 2024-03-15: report a time problem")

        # train rows AFTER the test period (a random split)
        shuffled = _frame(["2024-01-01", "2024-06-01"], ["A", "B"])
        assert leakage_problems(shuffled, good_test, 30), "train contains rows after the test period - report it"

        # group overlap, but only when asked
        shared = _frame(["2024-04-01", "2024-05-01"], ["A", "E"])
        assert leakage_problems(train, shared, 30) == [], \
            "without group_col, sharing customers across periods is allowed (the next-month question)"
        problems = leakage_problems(train, shared, 30, group_col="customer_id")
        assert len(problems) == 1, f"expected exactly one problem (customer overlap), got {problems}"
        assert "A" in problems[0] or "1" in problems[0], "say which/how many groups overlap"

        # both at once
        both = leakage_problems(train, _frame(["2024-03-15"], ["A"]), 30, group_col="customer_id")
        assert len(both) == 2, f"time overlap AND group overlap should give two problems, got {both}"

    return run_check("Exercise 2 · leakage_problems", test,
                     hint="time: train.snapshot_date.max() + horizon > test.snapshot_date.min(); "
                          "group: set(train[g]) & set(test[g])")


def ex3(rule_scores):
    def test():
        from sklearn.metrics import average_precision_score

        from ..bootstrap import ensure_split
        from ..features import LABEL, LEAKY_COLUMNS

        valid = ensure_split().valid
        X = valid.drop(columns=[LABEL, *[c for c in LEAKY_COLUMNS if c in valid.columns]])
        scores = np.asarray(rule_scores(X.copy()), dtype=float)
        assert scores.shape == (len(valid),), f"expected one score per row ({len(valid)}), got shape {scores.shape}"
        assert np.isfinite(scores).all(), "scores contain NaN or inf - fill missing values first"
        assert np.unique(scores).size > 1, "every row got the same score"
        rate = float(valid[LABEL].mean())
        pr_auc = float(average_precision_score(valid[LABEL], scores))
        assert pr_auc > 1.5 * rate, (f"validation PR-AUC {pr_auc:.3f} is not above 1.5 x the positive rate "
                                     f"({1.5 * rate:.3f}); a binary rule gives few distinct scores - try a continuous one")
        print(f"    your rule: validation PR-AUC {pr_auc:.3f} (positive rate {rate:.3f})")

    return run_check("Exercise 3 · rule_scores", test,
                     hint="combine a few continuous signals: falling usage (sessions_trend), "
                          "cancellation tickets, failed payments; higher score = more likely to churn")
