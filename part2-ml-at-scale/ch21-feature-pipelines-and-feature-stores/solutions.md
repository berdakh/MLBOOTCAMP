# Solutions · Chapter 21 · Feature pipelines & feature stores

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `point_in_time_join`

✅ Reference code: [`code/nb03.py`](solutions.py) (`ex1`), graded in CI.

```python
def point_in_time_join(events, snapshots, key, ts_col):
    left = events.assign(_row=range(len(events))).sort_values(ts_col, kind="stable")
    right = snapshots.sort_values(ts_col, kind="stable")
    joined = pd.merge_asof(left, right, on=ts_col, by=key,
                           direction="backward", allow_exact_matches=False)
    return joined.sort_values("_row").drop(columns="_row").reset_index(drop=True)
```

**Why it works.** `merge_asof(direction="backward")` takes, per `by` key,
the last right row whose time is `<=` the left time;
`allow_exact_matches=False` makes it strictly `<`. It requires both inputs
sorted by the `on` column, so we remember each event's position (`_row`) and
restore it afterwards.

**Common mistakes.**

* Forgetting `allow_exact_matches=False`. An event at exactly a snapshot's
  timestamp then sees that snapshot. With daily snapshots stamped at
  midnight and built from the *previous* days this is harmless, but if the
  snapshot was computed *from* events up to and including that time, it
  leaks the event's own outcome. Strict is the safe default.
* Returning the time-sorted frame. Every downstream `concat` with labels or
  predictions in the original order is then silently misaligned.
* Mismatched dtypes (`int32` vs `int64` keys, `datetime64[ns]` vs `[us]`):
  `merge_asof` refuses; cast first.

---

## Exercise 2 · `OnlineStore`

✅ Reference code: [`code/nb03.py`](solutions.py) (`ex2`).

```python
class OnlineStore:
    def __init__(self, ttl_seconds):
        self.ttl = float(ttl_seconds)
        self._data = {}

    def put(self, key, features, event_ts):
        current = self._data.get(key)
        if current is not None and event_ts < current[0]:
            return False
        self._data[key] = (event_ts, dict(features))
        return True

    def get(self, key, now):
        entry = self._data.get(key)
        if entry is None or now - entry[0] > self.ttl:
            return None
        return dict(entry[1])
```

**Why ignore older writes?** Writers are distributed and retried: a
backfill re-materialising last Tuesday, a delayed batch, a streaming job
replaying a partition. If writes were applied in arrival order, any of them
could roll a key back to an old value, and nobody would notice. Comparing
event times makes the store converge to the newest value regardless of
arrival order (the same idea as "last-writer-wins by event time" in
distributed databases). Copying the dict on `put`/`get` keeps a caller from
mutating stored values by accident.

**TTL exactly at the boundary.** The spec says `now - event_ts > ttl` is
expired, so exactly `ttl` old is valid. The offline join used the same
comparison; a `>=` on one side and `>` on the other is precisely the kind of
one-character skew that the parity check in 3.6 is there to catch.

---

## Exercise 3 · A streaming feature joined "as it is today" (written)

✍️ A strong answer names several of these problems:

1. **Future leakage.** The table "as it is today" holds the *latest* value
   per item (or values computed after the training events). Joining it to
   old impressions gives each impression a CTR from its future, including
   its own click.
2. **Overwritten history.** A streaming job that upserts the latest value per
   item keeps no history at all, so a point-in-time join is impossible.
3. **Late data and corrections.** The stream's value at time *t* (what
   serving saw) differs from a recomputation after late events arrived. Using
   the corrected values in training is a subtle skew.
4. **Different definitions.** A re-implementation in batch SQL may differ from
   the streaming code (window alignment, event vs processing time,
   deduplication, time zone): online/offline skew.
5. **Coverage/timing.** Hourly features change within a day while the other
   features are daily snapshots; the as-of tolerance (TTL) must be about an
   hour, not a day, or stale values get used.

**How to do it right.** Log the feature values *served* with each request
(the logged-features approach): training then uses exactly what the model
saw. Or have the streaming job append every emitted value with its window
end time (an offline history table, `(item_id, window_end, ctr_1h)`), and
build training data with an as-of join `event_ts > window_end`, TTL ≈ 1-2
hours. For history before the stream existed, backfill by replaying the
event log through **the same streaming code** (Flink/Spark can run a job in
batch mode over a bounded source) rather than writing a second SQL version.

**Verify.** Assert `window_end < event_ts` for every row; compare
backfilled values with logged served values on overlapping days (parity
report, tolerance, share of mismatches); check that offline AUC with the new
feature is not suspiciously high, and confirm with an A/B test.

---

## Checkpoint answers

1. **Offline vs online store.** ✅ Offline: full history of feature values
   with timestamps, columnar and cheap, used to build training sets with
   point-in-time joins and for batch scoring. Online: only the latest value
   per entity key, in a low-latency key-value store, read at request time.
   You need both because training needs history at past event times and
   serving needs now in milliseconds. Consistency comes from one feature
   definition feeding both (materialisation copies offline → online, or one
   streaming job writes both), the same TTL semantics, and continuous parity
   checks against logged served values (our day-45 check agreed on 100% of
   4,739 requests).
2. **Point-in-time join.** ✅
   `SELECT e.*, s.f FROM events e ASOF LEFT JOIN snaps s ON e.user_id =
   s.user_id AND e.ts > s.snapshot_ts`. For each event, the latest feature
   row with a timestamp before the event. Strict because a feature stamped at
   the event's own time may include the event (or other information from that
   instant) and so its outcome.
3. **Offline lift, flat A/B.** ✅ Suspect leakage first: check every
   feature's timestamp against the event time (`snapshot_ts < ts` for all
   rows), window bounds, joins to "current" tables. In our run, a same-day
   window turned an honest 0.793 into a reported 0.849, and the leaky model
   scored 0.786 on the features production actually has. Then check
   training/serving skew (compare logged served values with offline values),
   population differences (new users, traffic mix), the metric (AUC on
   impressions vs the business metric), position bias and feedback loops
   (Day 3), and whether the A/B test is powered (Day 6).
4. **TTL.** ✅ Offline: stops the as-of join from carrying a stale snapshot
   forward for entities absent from recent snapshots (5% of our training
   rows). Online: stops serving values from a pipeline that stopped updating,
   or for entities that went inactive; expired → default value, as in
   training. Choose it from the materialisation cadence (a bit more than one
   period, to tolerate a late run) and the feature's meaning; it must be the
   same offline and online.
5. **Is real-time worth it?** 🔬 Measure the value of freshness offline first:
   rebuild features with artificial staleness (lag 0, 1, 3, 7 days) and
   compare validation metrics; in ShopStream the AUC barely moved
   (0.7933 → 0.7922), because tastes are static. On real data, look at segments
   where freshness matters (new items, trending items, session intent). Weigh
   the gain against the cost: a streaming pipeline, its on-call, parity
   risks. Often a cheaper middle ground (hourly batch, or a few session
   features computed in the request path) captures most of it.
6. **Backfill for two years.** ✍️ Correct: the same code as production,
   point-in-time windows (data before each date only), handling of late data
   consistent with production, versioned feature definition. Safe to re-run:
   deterministic, idempotent writes (overwrite each date partition, never
   append), checks such as "day 30 alone == day 30 from the full run" (true
   in our run), row counts and null rates per day. Affordable: compute daily
   aggregates once and derive rolling windows from them (sum of 30 daily rows,
   not a 30-day rescan per date), process in date-partitioned chunks in
   parallel, read only the needed columns.
