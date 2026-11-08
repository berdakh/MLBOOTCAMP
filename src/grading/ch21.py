"""Self-checks for Level 2 notebook 03 · Feature pipelines & feature stores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.checks import run_check


def _pit_data(seed=3):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-03-03")
    snaps = pd.DataFrame({
        "user_id": rng.integers(0, 6, 40),
        "ts": base + pd.to_timedelta(rng.integers(0, 10, 40), unit="D"),
        "clicks_7d": rng.integers(0, 20, 40).astype(float),
        "ctr_7d": rng.random(40).round(3),
    }).drop_duplicates(["user_id", "ts"]).reset_index(drop=True)
    events = pd.DataFrame({
        "event_id": np.arange(60),
        "user_id": rng.integers(0, 8, 60),  # users 6, 7 have no snapshots
        "ts": base + pd.to_timedelta(rng.integers(0, 12 * 24, 60), unit="h"),
        "clicked": rng.random(60) < 0.3,
    })
    # ties: some events exactly at a snapshot timestamp (must NOT match that snapshot)
    tied = snaps.sample(8, random_state=seed)
    events.loc[:7, ["user_id", "ts"]] = tied[["user_id", "ts"]].to_numpy()
    events["user_id"] = events["user_id"].astype(int)
    events["ts"] = pd.to_datetime(events["ts"])
    return events.sample(frac=1, random_state=seed).reset_index(drop=True), snaps


def _reference(events, snaps):
    import duckdb

    con = duckdb.connect()
    con.register("e", events.assign(_row=np.arange(len(events))))
    con.register("s", snaps)
    out = con.execute("""SELECT e.*, s.clicks_7d, s.ctr_7d FROM e
                         ASOF LEFT JOIN s ON e.user_id = s.user_id AND e.ts > s.ts
                         ORDER BY e._row""").df()
    con.close()
    return out.drop(columns="_row")


def ex1(point_in_time_join):
    def test():
        events, snaps = _pit_data()
        want = _reference(events, snaps)
        got = point_in_time_join(events.copy(), snaps.copy(), "user_id", "ts")
        assert isinstance(got, pd.DataFrame), "return a DataFrame"
        assert len(got) == len(events), f"expected one row per event ({len(events)}), got {len(got)}"
        for col in ("event_id", "clicked", "clicks_7d", "ctr_7d"):
            assert col in got.columns, f"missing column {col!r}; got {list(got.columns)}"
        got = got.reset_index(drop=True)
        assert (got["event_id"].to_numpy() == events["event_id"].to_numpy()).all(), \
            "events must come back in their original order (merge_asof sorts by time: restore the order)"
        for col in ("clicks_7d", "ctr_7d"):
            a, b = got[col].to_numpy(float), want[col].to_numpy(float)
            same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b)
            if not same.all():
                i = int(np.flatnonzero(~same)[0])
                raise AssertionError(
                    f"{col} differs for event_id {got.event_id[i]} (user {got.user_id[i]}, ts {got.ts[i]}): "
                    f"got {a[i]}, expected {b[i]}. Is the match strictly before (not at) the event time?")
        events2, snaps2 = _pit_data(seed=8)
        got2 = point_in_time_join(events2.copy(), snaps2.copy(), "user_id", "ts").reset_index(drop=True)
        want2 = _reference(events2, snaps2)
        a, b = got2["clicks_7d"].to_numpy(float), want2["clicks_7d"].to_numpy(float)
        assert ((np.isnan(a) & np.isnan(b)) | np.isclose(a, b)).all(), "wrong values on a second dataset"

    return run_check("Exercise 1 · point_in_time_join", test,
                     hint="e = events.assign(_row=range(len(events))).sort_values(ts_col); pd.merge_asof(e, "
                          "snapshots.sort_values(ts_col), on=ts_col, by=key, allow_exact_matches=False)"
                          ".sort_values('_row')")


def ex2(OnlineStore):
    def test():
        s = OnlineStore(ttl_seconds=100)
        assert s.get("u1", now=0) is None, "an unknown key must return None"
        assert s.put("u1", {"ctr": 0.1}, event_ts=1000) is True, "a first put must return True"
        assert s.get("u1", now=1050) == {"ctr": 0.1}, "a fresh value must be returned"
        assert s.get("u1", now=1100) == {"ctr": 0.1}, "a value exactly ttl seconds old is still valid"
        assert s.get("u1", now=1100.5) is None, "a value older than the TTL must return None"
        s.put("u2", {"ctr": 0.5}, event_ts=2000)
        assert s.put("u2", {"ctr": 0.9}, event_ts=1990) is False, "an older write must be ignored (return False)"
        assert s.get("u2", now=2001) == {"ctr": 0.5}, "an older write must not replace a newer value"
        assert s.put("u2", {"ctr": 0.7}, event_ts=2000) is True, "an equal timestamp overwrites"
        assert s.get("u2", now=2001) == {"ctr": 0.7}
        assert s.put("u2", {"ctr": 0.8}, event_ts=2050) is True
        assert s.get("u2", now=2140) == {"ctr": 0.8}, "TTL counts from the newest event_ts"
        s.put(("u3", 7), {"clicks": 3}, event_ts=0)
        assert s.get(("u3", 7), now=10) == {"clicks": 3}, "tuple keys (e.g. user x category) must work"
        assert s.get("u1", now=1100.5) is None, "an expired key must stay expired"
        s.put("u1", {"ctr": 0.2}, event_ts=1200)
        assert s.get("u1", now=1250) == {"ctr": 0.2}, "a key can be written again after it expired"
        other = OnlineStore(ttl_seconds=5)
        assert other.get("u2", now=2001) is None, "two stores must not share state"

    return run_check("Exercise 2 · OnlineStore", test,
                     hint="keep self._data[key] = (event_ts, features); on put compare with the stored "
                          "event_ts; on get return None if now - event_ts > self.ttl")
