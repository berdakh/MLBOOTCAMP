"""Self-checks for Chapter 8 · Thresholds & calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.checks import run_check
from churnlab.evaluation import best_threshold_by_value, confusion_at, expected_value


def _snapshot_frame(seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(["2024-04-01", "2024-05-01", "2024-06-01"])
    sizes = [37, 52, 6]  # the last snapshot has fewer rows than k
    df = pd.DataFrame({
        "snapshot_date": np.repeat(dates, sizes),
        "customer_id": [f"C{i:04d}" for i in range(sum(sizes))],
    })
    scores = rng.permutation(np.linspace(0.001, 0.999, len(df)))  # distinct scores
    order = rng.permutation(len(df))  # rows NOT sorted by date
    df = df.iloc[order]
    df.index = np.arange(len(df)) * 3 + 11  # a non-default index
    return df, scores[order]


def ex1(top_k_per_snapshot):
    def test():
        df, scores = _snapshot_frame()
        k = 10
        flags = np.asarray(top_k_per_snapshot(df, scores, k))
        assert flags.shape == (len(df),), f"expected {len(df)} flags (one per row), got shape {flags.shape}"
        assert flags.dtype == bool, f"flags must be booleans, got dtype {flags.dtype}"
        for date, idx in df.reset_index(drop=True).groupby("snapshot_date").groups.items():
            idx = np.asarray(idx)
            want = min(k, len(idx))
            got = int(flags[idx].sum())
            assert got == want, f"snapshot {date.date()}: flagged {got} rows, expected {want}"
            top = idx[np.argsort(-scores[idx])[:want]]
            assert flags[top].all(), f"snapshot {date.date()}: the flagged rows are not the {want} highest scores"
        if isinstance(top_k_per_snapshot(df, scores, k), pd.Series):
            assert top_k_per_snapshot(df, scores, k).index.equals(df.index), "a returned Series must keep df's index"

    return run_check("Exercise 1 · top_k_per_snapshot", test,
                     hint="rank within snapshot: pd.Series(scores, index=df.index).groupby(df['snapshot_date'])"
                          ".rank(ascending=False, method='first') <= k")


def ex2(best_threshold):
    def test():
        rng = np.random.default_rng(5)
        for case, (value_tp, cost_fp) in enumerate([(150.0, 30.0), (150.0, 60.0), (300.0, 30.0)]):
            n = 3000
            y = (rng.random(n) < 0.06).astype(int)
            scores = np.clip(0.04 + 0.25 * y * rng.random(n) + 0.15 * rng.random(n), 0, 1)
            out = best_threshold(y, scores, value_tp, cost_fp)
            assert isinstance(out, tuple) and len(out) == 2, "return a tuple (threshold, value)"
            t, v = float(out[0]), float(out[1])
            actual = expected_value(confusion_at(y, scores, t), value_tp, cost_fp)
            assert abs(actual - v) < 1e-6, (f"case {case}: you report value {v:.1f} but threshold {t:.4f} "
                                            f"actually gives {actual:.1f}")
            _, ref_v = best_threshold_by_value(y, scores, value_tp, cost_fp)
            assert v >= ref_v - max(1.0, 0.01 * abs(ref_v)), (
                f"case {case}: value {v:.1f} is below the reference optimum {ref_v:.1f}")

    return run_check("Exercise 2 · best_threshold", test,
                     hint="try every distinct score as a threshold (flag = score >= t); "
                          "value = tp * value_tp - fp * cost_fp; keep the argmax")
