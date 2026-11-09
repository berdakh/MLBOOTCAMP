"""Reference solutions for Level 2 notebook 04. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class ex1:  # noqa: N801 - graded as a class: TumblingWindowCounter
    """Event-time tumbling counts per key, closed by a bounded-out-of-orderness watermark."""

    def __init__(self, window_seconds: float, allowed_lateness_seconds: float):
        self.window = float(window_seconds)
        self.lateness = float(allowed_lateness_seconds)
        self.watermark = -math.inf
        self.counts: dict = {}           # (key, window_start) -> count, open windows only
        self.late_events: list = []

    def _close(self, upto: float) -> list:
        ready = sorted((ws, str(k), k) for (k, ws) in self.counts if ws + self.window <= upto)
        return [(k, ws, self.counts.pop((k, ws))) for ws, _, k in ready]

    def add(self, key, event_time: float) -> list:
        ws = math.floor(event_time / self.window) * self.window
        if ws + self.window <= self.watermark:      # its window already closed: late
            self.late_events.append((key, event_time))
            return []
        self.counts[(key, ws)] = self.counts.get((key, ws), 0) + 1
        self.watermark = max(self.watermark, event_time - self.lateness)
        return self._close(self.watermark)

    def flush(self) -> list:
        return self._close(math.inf)


def ex2(batch_df: pd.DataFrame, stream_df: pd.DataFrame, key, tol: float) -> dict:
    keys = [key] if isinstance(key, str) else list(key)
    for name, df in (("batch", batch_df), ("stream", stream_df)):
        if df.duplicated(keys).any():
            raise ValueError(f"duplicated keys in {name} frame")
    m = batch_df[keys + ["value"]].merge(stream_df[keys + ["value"]], on=keys, how="outer",
                                         suffixes=("_b", "_s"), indicator=True)
    both = m[m["_merge"] == "both"]
    a, b = both["value_b"].to_numpy(float), both["value_s"].to_numpy(float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    diff = np.abs(a - b)
    match = (nan_a & nan_b) | (~nan_a & ~nan_b & (diff <= tol))
    valid = ~nan_a & ~nan_b
    n_keys = len(m)
    report = {
        "n_batch": len(batch_df),
        "n_stream": len(stream_df),
        "missing_in_stream": int((m["_merge"] == "left_only").sum()),
        "missing_in_batch": int((m["_merge"] == "right_only").sum()),
        "mismatched": int((~match).sum()),
        "max_abs_diff": float(diff[valid].max()) if valid.any() else 0.0,
        "match_rate": float(match.sum() / n_keys) if n_keys else 1.0,
    }
    report["passed"] = report["missing_in_stream"] == 0 and report["missing_in_batch"] == 0 \
        and report["mismatched"] == 0
    return report
