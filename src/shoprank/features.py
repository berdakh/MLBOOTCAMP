"""Day 2: features that are correct at prediction time, offline and online.

Notebook 03 (batch) and notebook 04 (streaming) use this module.

The rule behind everything here::

    A feature value used for an event at time t may only use data from before t.

Offline, features are stored as **daily snapshots**: one row per entity per
day, stamped ``snapshot_ts = start of that day`` and computed only from
*earlier* days. A training set attaches to each impression the latest
snapshot with ``snapshot_ts < event ts`` (an as-of join). Online, the same
values sit in a key-value **online store** with a time-to-live (TTL), so that a
value that stopped being refreshed is treated as missing rather than served
forever.

The SQL lives in module-level strings so the notebook can print it.
"""

from __future__ import annotations

import heapq
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

# --- offline: daily aggregates and snapshots -------------------------------------

DAILY_SQL = """
CREATE OR REPLACE TABLE user_day AS
    SELECT user_id, day, count(*) AS imps, sum(clicked::INT) AS clicks
    FROM imps GROUP BY user_id, day;
CREATE OR REPLACE TABLE item_day AS
    SELECT item_id, day, count(*) AS imps, sum(clicked::INT) AS clicks
    FROM imps GROUP BY item_id, day;
CREATE OR REPLACE TABLE user_cat_day AS
    SELECT i.user_id, it.category, i.day, sum(i.clicked::INT) AS clicks
    FROM imps i JOIN items it USING (item_id)
    GROUP BY i.user_id, it.category, i.day HAVING sum(i.clicked::INT) > 0;
"""

# {first} / {last}: days in the window relative to the snapshot day d.
# Correct: first = -7, last = -1 (the 7 days *before* d). Leaky: first = -6, last = 0.
SNAPSHOT_SQL = """
CREATE OR REPLACE TABLE user_snap AS
    SELECT s.day AS snap_day, TIMESTAMP '{start}' + to_days(s.day::INT) AS snapshot_ts, u.user_id,
           sum(u.imps) AS user_imps_7d, sum(u.clicks) AS user_clicks_7d
    FROM snap_days s JOIN user_day u
      ON u.day BETWEEN s.day {first:+d} AND s.day {last:+d}
    GROUP BY ALL;
CREATE OR REPLACE TABLE item_snap AS
    SELECT s.day AS snap_day, TIMESTAMP '{start}' + to_days(s.day::INT) AS snapshot_ts, i.item_id,
           sum(i.imps) AS item_imps_7d, sum(i.clicks) AS item_clicks_7d
    FROM snap_days s JOIN item_day i
      ON i.day BETWEEN s.day {first:+d} AND s.day {last:+d}
    GROUP BY ALL;
CREATE OR REPLACE TABLE user_cat_snap AS
    SELECT s.day AS snap_day, TIMESTAMP '{start}' + to_days(s.day::INT) AS snapshot_ts, c.user_id, c.category,
           sum(c.clicks) AS user_cat_clicks_14d
    FROM snap_days s JOIN user_cat_day c
      ON c.day BETWEEN s.day {first:+d} - 7 AND s.day {last:+d}
    GROUP BY ALL;
"""


def build_snapshots(con: duckdb.DuckDBPyConnection, start: str, days, include_same_day: bool = False,
                    lag_days: int = 0) -> None:
    """Create ``user_snap``, ``item_snap`` and ``user_cat_snap`` for the given snapshot days.

    Expects tables ``user_day``, ``item_day`` and ``user_cat_day`` (run
    :data:`DAILY_SQL` first). ``include_same_day=True`` builds the classic
    leak: the window ends *on* the snapshot day, so a feature used during day
    ``d`` already contains day ``d``'s clicks. ``lag_days`` simulates a stale
    pipeline: the snapshot for day ``d`` only has data up to ``d - 1 - lag``.
    """
    first, last = (-6, 0) if include_same_day else (-7, -1)
    con.execute("CREATE OR REPLACE TABLE snap_days AS SELECT unnest(?::INT[]) AS day", [list(days)])
    con.execute(SNAPSHOT_SQL.format(start=start, first=first - lag_days, last=last - lag_days))


PIT_SQL = """
SELECT e.*,
       us.user_imps_7d, us.user_clicks_7d, us.snapshot_ts AS user_snapshot_ts,
       its.item_imps_7d, its.item_clicks_7d, its.snapshot_ts AS item_snapshot_ts,
       uc.user_cat_clicks_14d, uc.snapshot_ts AS user_cat_snapshot_ts
FROM events e
ASOF LEFT JOIN user_snap us     ON e.user_id = us.user_id AND e.ts > us.snapshot_ts
ASOF LEFT JOIN item_snap its    ON e.item_id = its.item_id AND e.ts > its.snapshot_ts
ASOF LEFT JOIN user_cat_snap uc ON e.user_id = uc.user_id AND e.category = uc.category
                                AND e.ts > uc.snapshot_ts
"""


def point_in_time_features(con: duckdb.DuckDBPyConnection, events: pd.DataFrame,
                           ttl: pd.Timedelta = pd.Timedelta(days=1)) -> pd.DataFrame:
    """Attach the latest snapshot strictly before each event; expire snapshots older than ``ttl``.

    Snapshots only contain entities that were active in the window, so for an
    inactive user the as-of join would find an *old* snapshot and silently
    carry its values forward. The TTL turns those into missing values (and
    missing activity counts mean 0).
    """
    con.register("events", events)
    out = con.execute(PIT_SQL + " ORDER BY e.request_id, e.position").df()
    groups = {"user_snapshot_ts": ["user_imps_7d", "user_clicks_7d"],
              "item_snapshot_ts": ["item_imps_7d", "item_clicks_7d"],
              "user_cat_snapshot_ts": ["user_cat_clicks_14d"]}
    for ts_col, cols in groups.items():
        expired = (out["ts"] - out[ts_col]) > ttl
        out.loc[expired, cols] = np.nan
    count_cols = [c for cols in groups.values() for c in cols]
    out[count_cols] = out[count_cols].fillna(0)
    return out


def model_matrix(df: pd.DataFrame, prior_ctr: float = 0.1, strength: float = 20.0) -> pd.DataFrame:
    """Turn raw counts into model inputs: log counts and smoothed CTRs (no learned state)."""
    def smoothed(clicks, imps):
        return (clicks + prior_ctr * strength) / (imps + strength)

    return pd.DataFrame({
        "log_user_imps_7d": np.log1p(df["user_imps_7d"]),
        "user_ctr_7d": smoothed(df["user_clicks_7d"], df["user_imps_7d"]),
        "log_item_imps_7d": np.log1p(df["item_imps_7d"]),
        "item_ctr_7d": smoothed(df["item_clicks_7d"], df["item_imps_7d"]),
        "log_user_cat_clicks_14d": np.log1p(df["user_cat_clicks_14d"]),
        "log_position": np.log1p(df["position"].astype(float)),
    }, index=df.index)


# --- online: a key-value store with a TTL ---------------------------------------


class OnlineStore:
    """An in-memory online feature store: latest feature values per entity key, with a TTL.

    * ``put(key, features, event_ts)`` stores the values *as of* ``event_ts``
      (seconds). A write older than what is stored is ignored, so out-of-order
      writes cannot roll a value back; ties overwrite (last write wins).
    * ``get(key, now)`` returns the values, or ``None`` when the key is unknown
      or the value is older than ``ttl_seconds`` (``now - event_ts > ttl``).
      Expired entries are evicted on read.

    A real online store (Redis, DynamoDB, Bigtable, Feast's online store) adds
    persistence, replication and batch reads; the semantics are these.
    """

    def __init__(self, ttl_seconds: float):
        self.ttl = float(ttl_seconds)
        self._data: dict = {}

    def put(self, key, features: dict, event_ts: float) -> bool:
        current = self._data.get(key)
        if current is not None and event_ts < current[0]:
            return False
        self._data[key] = (float(event_ts), dict(features))
        return True

    def get(self, key, now: float):
        entry = self._data.get(key)
        if entry is None:
            return None
        event_ts, features = entry
        if now - event_ts > self.ttl:
            del self._data[key]
            return None
        return dict(features)

    def __len__(self) -> int:
        return len(self._data)


# --- streaming (notebook 04) -----------------------------------------------------


def simulate_event_stream(df: pd.DataFrame, seed: int = 0, median_delay_s: float = 2.0,
                          late_fraction: float = 0.03, late_delay_s: tuple = (60, 3600)) -> pd.DataFrame:
    """Turn logged events into the order a stream processor would receive them.

    Each event gets ``event_id``, ``event_ts`` (seconds since the first event:
    when it happened) and ``arrival_ts`` (when it reached the processor).
    Most delays are short (log-normal around ``median_delay_s``); a
    ``late_fraction`` of events (a phone offline, a retrying client) arrive
    ``late_delay_s`` seconds late. Rows come back sorted by arrival: the stream.
    """
    rng = np.random.default_rng(seed)
    out = df.reset_index(drop=True).copy()
    out["event_id"] = np.arange(len(out))
    t0 = out["ts"].min()
    out["event_ts"] = (out["ts"] - t0).dt.total_seconds() + out["position"].to_numpy() * 0.1
    delay = rng.lognormal(np.log(median_delay_s), 0.6, len(out))
    late = rng.random(len(out)) < late_fraction
    delay[late] = rng.uniform(*late_delay_s, late.sum())
    out["arrival_ts"] = out["event_ts"] + delay
    return out.sort_values("arrival_ts", kind="stable", ignore_index=True)


class WindowAggregator:
    """Event-time tumbling-window sums per key, with a watermark and a late-data policy.

    ``late_policy``:

    * ``"drop"``: events for an already-closed window are discarded (counted in
      ``n_late``);
    * ``"side"``: like drop, but late events are kept in ``side_output`` for a
      batch job to reconcile;
    * ``"update"``: a late event re-opens its window's result and an *updated*
      value is emitted (downstream must upsert by window key).

    ``dedup=True`` keeps the event ids seen per open window and ignores
    re-deliveries: idempotent processing under at-least-once delivery.
    Emitted rows are ``(key, window_start, value, emitted_at_watermark)``.
    """

    def __init__(self, window_s: float, lateness_s: float, late_policy: str = "drop", dedup: bool = False):
        assert late_policy in ("drop", "side", "update")
        self.window, self.lateness = float(window_s), float(lateness_s)
        self.policy, self.dedup = late_policy, dedup
        self.watermark = -np.inf
        self.state: dict = defaultdict(float)       # (key, window_start) -> running sum
        self.seen: dict = defaultdict(set)          # (key, window_start) -> event ids
        self.closed: dict = {}                      # final values (needed for "update")
        self.ends: list = []                        # heap of (window_end, key, window_start)
        self.emitted: list = []
        self.side_output: list = []
        self.n_late = self.n_duplicates = 0
        self.max_open = 0

    def process(self, key, event_ts: float, value: float = 1.0, event_id=None) -> None:
        ws = np.floor(event_ts / self.window) * self.window
        wkey = (key, ws)
        if ws + self.window <= self.watermark:      # its window already closed
            self.n_late += 1
            if self.policy == "side":
                self.side_output.append((key, event_ts, value))
            elif self.policy == "update":
                self.closed[wkey] = self.closed.get(wkey, 0.0) + value
                self.emitted.append((key, ws, self.closed[wkey], self.watermark))
            return
        if self.dedup and event_id is not None:
            if event_id in self.seen[wkey]:
                self.n_duplicates += 1
                return
            self.seen[wkey].add(event_id)
        if wkey not in self.state:
            heapq.heappush(self.ends, (ws + self.window, str(key), key, ws))
        self.state[wkey] += value
        self.max_open = max(self.max_open, len(self.state))
        self.watermark = max(self.watermark, event_ts - self.lateness)
        self._close(self.watermark)

    def _close(self, upto: float) -> None:
        while self.ends and self.ends[0][0] <= upto:
            _, _, key, ws = heapq.heappop(self.ends)
            value = self.state.pop((key, ws))
            self.seen.pop((key, ws), None)
            if self.policy == "update":
                self.closed[(key, ws)] = value
            self.emitted.append((key, ws, value, self.watermark))

    def flush(self) -> None:
        """End of stream: close every open window."""
        self._close(np.inf)

    def results(self) -> pd.DataFrame:
        """Latest emitted value per (key, window_start): what an upserting sink would hold."""
        df = pd.DataFrame(self.emitted, columns=["key", "window_start", "value", "watermark"])
        return df.groupby(["key", "window_start"], as_index=False).last()[["key", "window_start", "value"]]
