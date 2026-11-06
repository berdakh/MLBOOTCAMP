# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/04_streaming_features.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 04 · Streaming features
#
# **Day 2 · ~90 minutes · CPU only**
#
# "Clicks on this category in the last 15 minutes" cannot come from a daily
# batch job: it has to be computed from the event stream as events arrive.
# Streams deliver events late, out of order and sometimes twice, so a
# streaming feature is a set of explicit decisions about time, completeness
# and duplicates. This notebook simulates such a stream from ShopStream and
# makes those decisions visible with numbers. It also checks that the
# streaming feature matches its batch definition.
#
# By the end you will be able to:
#
# 1. distinguish event time from processing time, and measure how out of order a stream is;
# 2. implement tumbling, sliding and session windows incrementally, one event at a time;
# 3. use watermarks to decide when a window is complete, and choose a late-data policy (drop, update, side output) with its trade-offs measured;
# 4. explain at-least-once vs exactly-once delivery, and make an aggregation idempotent by deduplicating on event ids;
# 5. write an online/offline parity test between a batch (SQL) and a streaming computation of the same feature;
# 6. map these ideas onto Kafka, Flink and Spark Structured Streaming.
#
# **You will produce:** nothing on disk besides the cached dataset; this notebook is a simulation.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level2/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: the ShopStream small dataset (notebook 01) -------------------
from collections import defaultdict

import duckdb
import numpy as np
import pandas as pd

from shoprank import features as F
from shoprank import store
from shoprank.grading import nb04 as grade

world, users, items, imps = store.load("small")
imps = imps.merge(items[["item_id", "category"]], on="item_id", how="left")
day = imps[imps.day == 42]
print(f"day 42: {len(day):,} impressions, {day.clicked.sum():,} clicks, {day.request_id.nunique():,} requests")

# %% [markdown]
# **Expected:** 44,830 impressions, 6,259 clicks and 4,483 requests on day
# 42 (the first validation day).

# %% [markdown]
# ## 4.1 · Event time vs processing time
#
# * **Event time**: when the thing happened (the impression was rendered on the
#   phone). Features and labels are defined on it.
# * **Processing time**: when the event reaches your stream processor. It is
#   later, by network, batching, retries, and phones that were offline.
#
# `F.simulate_event_stream` turns the day's log into a stream: each event gets
# an `event_id`, an `event_ts` (seconds since the day's first event) and an
# `arrival_ts`. Most delays are a few seconds; 3% of events (offline phones,
# retries) arrive 1 to 60 minutes late. The stream is sorted by arrival.

# %%
stream = F.simulate_event_stream(day, seed=0)
delay = stream.arrival_ts - stream.event_ts
print("delay (s) percentiles:", delay.quantile([0.5, 0.9, 0.97, 0.99, 1.0]).round(1).to_dict())
behind = stream.event_ts < stream.event_ts.cummax()
print(f"events older than an event already received: {behind.mean():.0%}; "
      f"by more than 60 s: {(stream.event_ts.cummax() - stream.event_ts > 60).mean():.1%}")
stream[["event_id", "category", "clicked", "event_ts", "arrival_ts"]].head(4)

# %% [markdown]
# **Expected:** median delay about 2 s, p90 about 5 s, p99 in the tens of
# minutes; about 65% of events arrive after a later-timestamped event (a
# little disorder is the normal state) but only about 3% are more than a
# minute behind.
#
# The feature we will compute: **clicks per category per 15-minute window**.
# Its batch definition, on event time, in SQL:

# %%
W = 900  # 15 minutes
con = duckdb.connect()
con.register("stream", stream)
BATCH_SQL = f"""
SELECT category AS key, floor(event_ts / {W}) * {W} AS window_start, sum(clicked::INT) AS value
FROM stream GROUP BY ALL ORDER BY window_start, key
"""
batch = con.execute(BATCH_SQL).df()
print(f"{len(batch):,} (category, window) rows; total clicks {int(batch.value.sum()):,}")

# %% [markdown]
# **Expected:** 1,886 rows (20 categories × 96 windows, minus empty
# ones) and the day's 6,259 clicks.
#
# The naive streaming version puts each event into the window of its
# **arrival** time (processing-time windows: simple, and what you get if you
# timestamp events when they reach the server):

# %%
proc = stream.assign(window_start=np.floor(stream.arrival_ts / W) * W).groupby(
    ["category", "window_start"]).clicked.sum().rename("value").reset_index().rename(columns={"category": "key"})
m = batch.merge(proc, on=["key", "window_start"], how="outer", suffixes=("_batch", "_proc")).fillna(0)
print(f"processing-time windows that differ from the event-time truth: {(m.value_batch != m.value_proc).mean():.0%}")

# %% [markdown]
# **Expected:** about 15% of the windows hold different counts. Events
# near a window boundary slide into the next window, and late events land in
# whatever window is open when they arrive. A model trained on event-time
# features (the batch SQL) and served processing-time features sees skew.
# Event-time processing is the fix, at the price of having to decide when a
# window is *complete*.

# %% [markdown]
# ## 4.2 · Windows, computed incrementally
#
# A stream processor cannot `GROUP BY` the whole day; it updates state per
# event. Three window types cover almost every feature:
#
# | Window | Definition | Example feature |
# |---|---|---|
# | **Tumbling** | fixed size, no overlap: `[0, 15m)`, `[15m, 30m)`, … | clicks per category per 15 min |
# | **Sliding (hopping)** | size `S`, advancing every `h < S`; each event is in `S/h` windows | clicks in the last hour, updated every 15 min |
# | **Session** | per key, events closer than a gap `g` belong together; windows merge | session length, items viewed this session |
#
# Sliding windows first: each event is added to the 4 one-hour windows (sliding
# every 15 minutes) that contain it. With no completeness question yet (we
# process the whole stream, then read the state), the result must equal the
# batch definition exactly.

# %%
SIZE, HOP = 3600, 900
sliding = defaultdict(int)
for key, t, clicked in zip(stream.category.to_numpy(), stream.event_ts.to_numpy(), stream.clicked.to_numpy()):
    last_start = np.floor(t / HOP) * HOP
    for k in range(SIZE // HOP):  # the windows [s, s + SIZE) with s <= t < s + SIZE
        sliding[(key, last_start - k * HOP)] += int(clicked)

batch_sliding = con.execute(f"""
    WITH b AS (SELECT category AS key, floor(event_ts / {HOP}) * {HOP} AS b, sum(clicked::INT) AS c
               FROM stream GROUP BY ALL)
    SELECT key, b - k * {HOP} AS window_start, sum(c) AS value
    FROM b, range({SIZE // HOP}) AS r(k) GROUP BY ALL""").df()
stream_sliding = pd.Series(sliding).rename_axis(["key", "window_start"]).rename("value").reset_index()
both = batch_sliding.merge(stream_sliding, on=["key", "window_start"], suffixes=("_b", "_s"))
print(f"{len(stream_sliding):,} sliding windows | equal to batch: "
      f"{len(both) == len(batch_sliding) == len(stream_sliding) and (both.value_b == both.value_s).all()}")

# %% [markdown]
# **Expected:** 1,980 windows and `True`. (About as many windows as the
# tumbling version, 20 categories × 99 window starts, but each event is
# counted in 4 of them, so the state does 4× the updates.)
#
# **Session windows** are harder incrementally, because an out-of-order event
# can *bridge* two sessions that were separate so far. Per user, keep a list
# of `[start, end]` sessions; a new event at `t` starts `[t, t]`, which is
# then merged with every session within `gap` of it:

# %%
GAP = 1800  # 30 minutes


def add_to_sessions(sessions: list, t: float, gap: float) -> list:
    lo, hi, kept = t, t, []
    for s, e in sessions:
        if s - gap <= hi and lo <= e + gap:   # overlaps [lo - gap, hi + gap]: merge
            lo, hi = min(lo, s), max(hi, e)
        else:
            kept.append((s, e))
    return kept + [(lo, hi)]


per_request = stream.drop_duplicates("request_id")  # one event per feed load
sessions = defaultdict(list)
for user, t in zip(per_request.user_id.to_numpy(), per_request.event_ts.to_numpy()):
    sessions[user] = add_to_sessions(sessions[user], t, GAP)
n_stream = sum(len(v) for v in sessions.values())

ordered = per_request.sort_values(["user_id", "event_ts"])
new_session = (ordered.user_id != ordered.user_id.shift()) | (ordered.event_ts.diff() > GAP)
print(f"sessions: streaming {n_stream:,} | batch {int(new_session.sum()):,}")

# %% [markdown]
# **Expected:** the same number of sessions both ways (4,416 on day 42). Remove the merge loop's ability to merge
# *several* sessions (e.g. stop at the first match) and the counts
# diverge: that is the bridging case.

# %% [markdown]
# ## 4.3 · Watermarks: when is a window complete?
#
# A streaming job must *emit* a window's result at some point, but late events
# may still arrive. A **watermark** is the processor's claim "I have seen
# (almost) every event with event time ≤ W". The simplest and most common
# kind is *bounded out-of-orderness*:
#
# `watermark = (max event time seen so far) − allowed lateness`
#
# A tumbling window `[s, s + size)` is **closed and emitted** when
# `watermark ≥ s + size`. An event that arrives for a window that is
# already closed is **late**, and a policy decides what happens to it:
#
# * **drop** it (simplest; the feature undercounts slightly);
# * **side output**: drop it from the stream result, but keep it for a batch job
#   to reconcile;
# * **update**: emit a corrected result for the window (downstream must accept
#   upserts, and consumers may have already used the old value).
#
# The allowed lateness is a trade-off: larger lateness → fewer late events
# but results emitted later and more open windows held in memory.

# %% [markdown]
# > **Exercise 1.** Implement `TumblingWindowCounter(window_seconds,
# > allowed_lateness_seconds)` that counts events per key in event-time
# > tumbling windows:
# >
# > * `add(key, event_time)` → returns the list of windows that **closed**
# >   because of this event, as `(key, window_start, count)` tuples sorted by
# >   `(window_start, key)`;
# > * the window of `t` starts at `floor(t / window) * window`;
# > * watermark = max event time seen − allowed lateness (starts at −∞); a
# >   window closes when `watermark >= window_start + window`;
# > * an event whose window has already closed (`window_start + window <=`
# >   the watermark *before* the event) is late: append `(key, event_time)` to
# >   `self.late_events` and do not count it;
# > * `flush()` closes all open windows and returns them, sorted the same way.
# >
# > The grader replays a crafted out-of-order sequence (with late events and
# > boundary cases) and a random stream against a reference.

# %%
import math


class TumblingWindowCounter:
    def __init__(self, window_seconds: float, allowed_lateness_seconds: float):
        # YOUR CODE HERE
        raise NotImplementedError

    def add(self, key, event_time: float) -> list:
        raise NotImplementedError

    def flush(self) -> list:
        raise NotImplementedError


grade.ex1(TumblingWindowCounter)

# %% [markdown]
# ## 4.4 · Late-data policies, measured
#
# `F.WindowAggregator` is the same machinery as Exercise 1, extended with
# sums, the three late-data policies and optional deduplication (read it in
# [`src/shoprank/features.py`](../../src/shoprank/features.py)). Run the
# click-count feature through it for several allowed lateness values and
# compare with the batch truth.

# %%
def run_stream(events: pd.DataFrame, lateness: float, policy: str = "drop", dedup: bool = False):
    agg = F.WindowAggregator(W, lateness, late_policy=policy, dedup=dedup)
    for key, t, clicked, eid in zip(events.category.to_numpy(), events.event_ts.to_numpy(),
                                    events.clicked.to_numpy(), events.event_id.to_numpy()):
        agg.process(key, t, float(clicked), event_id=eid)
    agg.flush()
    return agg


def exact_share(result: pd.DataFrame) -> float:
    m = batch.merge(result, on=["key", "window_start"], how="left", suffixes=("", "_s"))
    return float((m.value == m.value_s).mean())


rows = []
for lateness in (0, 10, 60, 600, 3600):
    agg = run_stream(stream, lateness)
    rows.append({"allowed lateness (s)": lateness, "late events dropped": agg.n_late,
                 "windows exactly right": exact_share(agg.results()),
                 "max open windows (state)": agg.max_open,
                 "result available after window end (s)": lateness})
pd.DataFrame(rows).set_index("allowed lateness (s)").round(3)

# %% [markdown]
# **Expected:** with no allowed lateness about 1,100 events (about 2.5%)
# are dropped and about 92% of windows are exactly right; 10 minutes of
# lateness still drops about 900 (the long-delayed tail) and gets about 94%
# right; 1 hour drops none and matches the batch definition exactly, at the
# cost of holding about 100 open windows instead of about 20 and publishing
# each result an hour after its window ends. For a "last 15 minutes"
# feature, an hour of delay defeats the purpose. So the usual answer is a
# short lateness, plus a policy for the stragglers:

# %%
side = run_stream(stream, 60, policy="side")
upd = run_stream(stream, 60, policy="update")
corrections = len(upd.emitted) - len(upd.results())
print(f"side output: {len(side.side_output):,} late events kept for reconciliation | "
      f"windows exactly right {exact_share(side.results()):.1%}")
print(f"update:      {corrections:,} corrected results emitted after the first | "
      f"windows exactly right {exact_share(upd.results()):.1%}")

# %% [markdown]
# **Expected:** about 1,100 events in the side output (the result itself is
# the same as *drop*), and with *update* every window ends up exactly right
# (100.0%), at the price of about 1,100 corrections that downstream systems
# must apply as upserts. The online store from notebook 03 does exactly that
# (a newer write for the same key replaces the old one), but a model that
# already scored a request with the first value has used it.

# %% [markdown]
# ## 4.5 · At-least-once, exactly-once, and idempotence
#
# Delivery guarantees between a log (Kafka) and a processor:
#
# * **at-most-once**: commit the read offset before processing; a crash loses events;
# * **at-least-once**: process, then commit; a crash re-delivers events
#   processed since the last commit: **duplicates**;
# * **exactly-once (effectively-once)**: the state update and the offset commit
#   happen atomically (Flink checkpoints with transactional sinks; Kafka
#   transactions; Spark checkpoints with idempotent sinks). It is about the
#   *effect* on state and sinks, not about the network delivering once.
#
# The cheapest robust defence works under at-least-once: make the update
# **idempotent** by deduplicating on the event id. Re-deliver 5% of the
# stream a few seconds to minutes after the original:

# %%
rng = np.random.default_rng(1)
dupes = stream.sample(frac=0.05, random_state=1).copy()
dupes["arrival_ts"] = dupes["arrival_ts"] + rng.uniform(1, 120, len(dupes))
redelivered = pd.concat([stream, dupes]).sort_values("arrival_ts", kind="stable", ignore_index=True)

naive = run_stream(redelivered, 3600)
idem = run_stream(redelivered, 3600, dedup=True)
print(f"re-delivered events: {len(dupes):,}")
print(f"no dedup:   total clicks {naive.results().value.sum():,.0f} vs truth {int(batch.value.sum()):,} | "
      f"windows exactly right {exact_share(naive.results()):.1%}")
print(f"dedup by id: total clicks {idem.results().value.sum():,.0f} | duplicates ignored {idem.n_duplicates:,} | "
      f"windows exactly right {exact_share(idem.results()):.1%}")

# %% [markdown]
# **Expected:** 2,242 re-delivered events. Without dedup the click total is
# about 5% too high (6,586 vs 6,259) and about 15% of the windows are wrong;
# with dedup by event id the result matches the batch truth exactly
# (100.0%), and all 2,242 duplicates are ignored.
#
# The dedup state (event ids per open window) is freed when the window
# closes, so memory stays bounded by the allowed lateness. A duplicate that
# arrives *after* its window closed is handled by the late-data policy
# instead. Note what makes this work: a **stable event id assigned at the
# source** (the client or the first server). An id assigned by the processor
# would differ between the original and the retry.

# %% [markdown]
# ## 4.6 · Online/offline parity
#
# The feature has two implementations: the batch SQL (used to build training
# data, notebook 03) and the streaming job (used to serve). A **parity test**
# runs both over the same events and reports where they disagree. By hand,
# for the 60-second-lateness *drop* stream:

# %%
stream_result = run_stream(stream, 60).results()
joined = batch.merge(stream_result, on=["key", "window_start"], how="outer",
                     suffixes=("_batch", "_stream"), indicator=True)
print(joined["_merge"].value_counts().to_dict())
both = joined[joined["_merge"] == "both"]
diff = (both.value_batch - both.value_stream).abs()
print(f"windows in both: {len(both):,} | differing: {(diff > 0).sum():,} | max |diff| {diff.max():.0f}")

# %% [markdown]
# **Expected:** every window appears in both (`both` about 1,900, no
# `left_only`/`right_only`); about 150 windows differ, by at most 2
# clicks: exactly the late events the drop policy discarded. A parity
# test turns "the streaming feature looks fine" into a number that can gate a
# deployment or page an on-call engineer.

# %% [markdown]
# > **Exercise 2.** Package the comparison as
# > `parity_report(batch_df, stream_df, key, tol)`. Both frames have the
# > key column(s) (`key` is a column name or a list of names) and a `value`
# > column. Return a dict with:
# >
# > * `n_batch`, `n_stream`: row counts;
# > * `missing_in_stream`, `missing_in_batch`: number of keys present only in the other frame;
# > * `mismatched`: keys present in both whose values differ by more than `tol` (a NaN on exactly one side is a mismatch; NaN on both sides matches);
# > * `max_abs_diff`: largest absolute difference over keys present in both (0.0 if none, ignoring NaNs);
# > * `match_rate`: keys matching within `tol` divided by the number of distinct keys in either frame (1.0 if both are empty);
# > * `passed`: `True` only if nothing is missing and nothing is mismatched.
# >
# > Raise `ValueError` if a key appears more than once in either frame (a
# > parity test on duplicated keys silently compares the wrong rows).

# %%
def parity_report(batch_df: pd.DataFrame, stream_df: pd.DataFrame, key, tol: float) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(parity_report)

# %% [markdown]
# ## 4.7 · The real systems
#
# Nothing to install; this is how the simulation maps onto production tools.
#
# | Concept here | Kafka | Flink | Spark Structured Streaming |
# |---|---|---|---|
# | the stream, ordered by arrival | a topic, split into partitions (ordered within a partition only), retained for days | source (`KafkaSource`) | `readStream.format("kafka")` |
# | `event_ts` | record timestamp / a field in the payload | `WatermarkStrategy.withTimestampAssigner` | a column used in `withWatermark` |
# | bounded-out-of-orderness watermark | — | `forBoundedOutOfOrderness(Duration)` | `withWatermark("ts", "60 seconds")` |
# | tumbling / sliding / session windows | — | `TumblingEventTimeWindows`, `SlidingEventTimeWindows`, `EventTimeSessionWindows` | `window(ts, "15 minutes")`, `window(ts, "1 hour", "15 minutes")`, `session_window` |
# | late policy | — | `allowedLateness` (update), `sideOutputLateData` (side) | late rows dropped after the watermark; `outputMode("update")` |
# | per-window state | — | keyed state in RocksDB, checkpointed | state store, checkpointed |
# | dedup by event id | idempotent producer (`enable.idempotence`) prevents producer-side duplicates | keyed state + TTL | `dropDuplicatesWithinWatermark("event_id")` |
# | exactly-once | transactions (`read_committed`) | checkpoints + two-phase-commit sinks | checkpoints + idempotent/transactional sinks |
#
# Where the feature goes: the streaming job writes each emitted value to the
# **online store** (keyed by `(category, window)` or simply by `category`
# with the latest window) *and* appends it to an **offline log** with its
# window end time, so training sets can be built point-in-time from exactly
# the values that were served (notebook 03, Exercise 3). Teams that cannot
# afford two implementations run the streaming code in batch mode over
# historical data ("kappa" style) instead of maintaining a separate SQL
# version, and they still run parity tests, because configuration (lateness,
# dedup, time zones) differs between deployments.

# %% [markdown]
# > **Exercise 3 (written).** Your Flink job computes "clicks per item in the
# > last 10 minutes" and writes it to Redis. After a deployment, the job
# > restarts from a checkpoint taken 4 minutes earlier, so Kafka re-delivers 4
# > minutes of events. Describe what breaks (in the job's state, in Redis, in
# > the model's inputs, in any offline log of served features) under (a)
# > at-least-once processing without dedup, (b) at-least-once with
# > dedup by event id, (c) exactly-once checkpointing with a non-transactional
# > Redis sink. What would you monitor to detect each problem?
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Streaming result never emitted for the last windows | the watermark only advances with new events; the stream ended | call `flush()` at end of input; in production, idle-source watermark handling (Flink `withIdleness`) |
# | Streaming counts lower than batch | late events dropped by the watermark | raise allowed lateness, or use the update/side-output policy and measure the trade-off |
# | Streaming counts higher than batch | duplicates under at-least-once delivery | dedup by a source-assigned event id within the watermark |
# | Session counts differ from batch | an out-of-order event bridging two sessions is not merged | merge *all* sessions within the gap, not just the first |
# | Parity test passes but production values differ | it compared the wrong rows (duplicated keys) or different time zones / window alignment | assert unique keys; align windows on UTC epoch seconds |
# | Memory grows without bound in a streaming job | windows (or dedup sets) never close: event time stuck or lateness huge | check the watermark advances; bound lateness; TTL on state |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Event time vs processing time: define both, and give an example where a feature computed on processing time is wrong.
# 2. What is a watermark? What does the allowed-lateness setting trade off, and how would you choose it for a "clicks in the last 15 minutes" feature?
# 3. Compare drop, update and side-output policies for late data. Which would you choose for a fraud feature, and which for a dashboard?
# 4. Explain at-least-once and exactly-once processing. What does "exactly-once" actually guarantee in Flink or Spark, and why do you still need idempotent sinks?
# 5. How do you make sure a streaming feature in production equals the feature the model was trained on? Describe the parity test and what you log.
# 6. Design the pipeline for "number of items a user viewed in the current session" for a feed ranker: windows, state, latency budget, failure handling.
#
# ➡️ **Next:** [`05_implicit_feedback_and_position_bias.ipynb`](05_implicit_feedback_and_position_bias.ipynb) — Day 3: clicks are biased by position; learn to measure and correct it.
