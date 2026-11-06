"""Leakage-safe splits for (snapshot, customer) data.

Two kinds of leakage matter here:

* **Temporal.** A random split puts March 2024 rows in training and January
  2024 rows in test. The model then learns from the future of the period it is
  scored on. Production never gets to do that.
* **Entity (group).** The same customer appears at many snapshots, and its rows
  are strongly correlated. If one customer is in both train and test, the test
  score partly measures memorisation of that customer.

For "score existing customers next month" - this project's use case - a
**time split** is the honest choice, and re-using a customer across periods is
realistic. For "score customers we have never seen", use a **group split**.
Pick the split that matches how the model will be used, and say which one you
used in every result you report.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: The bootcamp's fixed periods. The test period is touched ONCE, on Day 5.
TRAIN_END = pd.Timestamp("2024-04-01")  # train: snapshots before this
VALID_END = pd.Timestamp("2024-07-01")  # validation: [TRAIN_END, VALID_END)
# test: snapshots >= VALID_END


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self, label: str = "churned_30d") -> pd.DataFrame:
        rows = []
        for name, df in (("train", self.train), ("valid", self.valid), ("test", self.test)):
            rows.append({
                "split": name,
                "rows": len(df),
                "customers": df["customer_id"].nunique(),
                "first_snapshot": df["snapshot_date"].min().date() if len(df) else None,
                "last_snapshot": df["snapshot_date"].max().date() if len(df) else None,
                "positive_rate": round(float(df[label].mean()), 4) if len(df) else None,
            })
        return pd.DataFrame(rows).set_index("split")


def time_split(df: pd.DataFrame, train_end=TRAIN_END, valid_end=VALID_END,
               gap_days: int = 30) -> Split:
    """Split by snapshot date, with an embargo gap before each boundary.

    Why a gap: the label for snapshot T looks 30 days ahead. A training row at
    ``train_end - 1 month`` has a label window that overlaps the validation
    period, so its label already "knows" about validation-period events. Rows
    whose label window crosses the boundary are dropped from the earlier split.
    """
    t = df["snapshot_date"]
    gap = pd.Timedelta(days=gap_days)
    # "< train_end" as well as "+ gap <=": with gap_days=0 a snapshot dated
    # exactly train_end would otherwise land in both train and validation.
    train = df[(t + gap <= train_end) & (t < train_end)]
    valid = df[(t >= train_end) & (t + gap <= valid_end) & (t < valid_end)]
    test = df[t >= valid_end]
    return Split(train.reset_index(drop=True), valid.reset_index(drop=True),
                 test.reset_index(drop=True))


def random_split(df: pd.DataFrame, seed: int = 0, frac=(0.6, 0.2)) -> Split:
    """The WRONG split for this problem, kept to measure how wrong it is."""
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    a, b = frac[0], frac[0] + frac[1]
    return Split(df[u < a].reset_index(drop=True), df[(u >= a) & (u < b)].reset_index(drop=True),
                 df[u >= b].reset_index(drop=True))


def group_split(df: pd.DataFrame, group: str = "customer_id", seed: int = 0,
                frac=(0.6, 0.2)) -> Split:
    """Assign whole customers to one split - for the "unseen customers" question."""
    rng = np.random.default_rng(seed)
    groups = df[group].unique()
    u = pd.Series(rng.random(len(groups)), index=groups)
    a, b = frac[0], frac[0] + frac[1]
    g = df[group].map(u)
    return Split(df[g < a].reset_index(drop=True), df[(g >= a) & (g < b)].reset_index(drop=True),
                 df[g >= b].reset_index(drop=True))


def rolling_origin_folds(df: pd.DataFrame, n_folds: int = 4, valid_months: int = 2,
                         gap_days: int = 30, min_train_months: int = 6) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV over snapshot months, yielding (train_idx, valid_idx).

    Each fold trains on everything before a cut-off and validates on the next
    ``valid_months`` snapshots. This is the time-series analogue of k-fold, and
    it is what you should cross-validate with inside the training period.
    """
    months = np.sort(df["snapshot_date"].unique())
    last_start = len(months) - valid_months
    starts = np.linspace(min_train_months, last_start, n_folds).round().astype(int)
    gap = pd.Timedelta(days=gap_days)
    t = df["snapshot_date"]
    for s in starts:
        cut = pd.Timestamp(months[s])
        end = pd.Timestamp(months[min(s + valid_months, len(months) - 1)])
        train_idx = np.flatnonzero((t + gap <= cut).to_numpy())
        if s + valid_months >= len(months):
            valid_idx = np.flatnonzero((t >= cut).to_numpy())
        else:
            valid_idx = np.flatnonzero(((t >= cut) & (t < end)).to_numpy())
        yield train_idx, valid_idx


def assert_no_leakage(split: Split, label_horizon_days: int = 30, by_group: str | None = None) -> None:
    """Raise AssertionError if the split violates time order or group separation."""
    horizon = pd.Timedelta(days=label_horizon_days)
    if len(split.train) and len(split.valid):
        assert split.train["snapshot_date"].max() + horizon <= split.valid["snapshot_date"].min(), \
            "train label windows overlap the validation period"
    if len(split.valid) and len(split.test):
        assert split.valid["snapshot_date"].max() + horizon <= split.test["snapshot_date"].min(), \
            "validation label windows overlap the test period"
    if by_group:
        a, b, c = (set(s[by_group]) for s in (split.train, split.valid, split.test))
        assert not (a & b or a & c or b & c), f"{by_group} appears in more than one split"
