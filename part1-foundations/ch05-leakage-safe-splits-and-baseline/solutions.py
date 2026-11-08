"""Reference solutions for notebook 04. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(df: pd.DataFrame, train_end, valid_end, gap_days: int):
    t = df["snapshot_date"]
    gap = pd.Timedelta(days=gap_days)
    train = df[t + gap <= pd.Timestamp(train_end)]
    valid = df[(t >= pd.Timestamp(train_end)) & (t + gap <= pd.Timestamp(valid_end))]
    test = df[t >= pd.Timestamp(valid_end)]
    return train, valid, test


def ex2(train: pd.DataFrame, test: pd.DataFrame, label_horizon_days: int, group_col=None) -> list[str]:
    problems = []
    if len(train) and len(test):
        last_train = train["snapshot_date"].max()
        first_test = test["snapshot_date"].min()
        if last_train + pd.Timedelta(days=label_horizon_days) > first_test:
            problems.append(
                f"time: last train snapshot {last_train.date()} + {label_horizon_days}d label window "
                f"reaches past the first test snapshot {first_test.date()}")
    if group_col is not None:
        shared = set(train[group_col]) & set(test[group_col])
        if shared:
            examples = ", ".join(map(str, sorted(shared)[:3]))
            problems.append(f"group: {len(shared)} {group_col} value(s) in both train and test (e.g. {examples})")
    return problems


def ex3(df: pd.DataFrame) -> np.ndarray:
    # Higher = more likely to churn. Falling usage is the strongest single signal;
    # cancellation tickets, failed payments and recent tickets add to it.
    score = (
        -np.log(df["sessions_trend"].fillna(1.0).clip(lower=1e-3))
        + 1.0 * df["cancellation_tickets_30d"].fillna(0)
        + 0.5 * df["failed_payments_90d"].fillna(0)
        + 0.3 * df["tickets_30d"].fillna(0)
    )
    return score.to_numpy(dtype=float)
