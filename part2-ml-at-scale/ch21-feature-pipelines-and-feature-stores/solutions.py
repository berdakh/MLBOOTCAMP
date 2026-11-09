"""Reference solutions for Level 2 notebook 03. Graded by tests/test_solutions.py."""

from __future__ import annotations

import pandas as pd


def ex1(events: pd.DataFrame, snapshots: pd.DataFrame, key: str, ts_col: str) -> pd.DataFrame:
    left = events.assign(_row=range(len(events))).sort_values(ts_col, kind="stable")
    right = snapshots.sort_values(ts_col, kind="stable")
    joined = pd.merge_asof(left, right, on=ts_col, by=key,
                           direction="backward", allow_exact_matches=False)  # strictly before
    return joined.sort_values("_row").drop(columns="_row").reset_index(drop=True)


class ex2:  # noqa: N801 - graded as a class: OnlineStore
    """Latest feature values per key, with a TTL; older writes are ignored."""

    def __init__(self, ttl_seconds: float):
        self.ttl = float(ttl_seconds)
        self._data: dict = {}

    def put(self, key, features: dict, event_ts: float) -> bool:
        current = self._data.get(key)
        if current is not None and event_ts < current[0]:
            return False
        self._data[key] = (event_ts, dict(features))
        return True

    def get(self, key, now: float):
        entry = self._data.get(key)
        if entry is None or now - entry[0] > self.ttl:
            return None
        return dict(entry[1])
