# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 21 · Feature pipelines & feature stores
#
# **Part II · Chapter 21 · about 90 minutes · CPU only**
#
# A ranking model can only be as good as the features it receives *at the
# moment it scores a request*. Many of the most damaging errors in production
# machine learning are not in the model but in the feature pipeline: a training
# set that used data from the future, an online service that supplies values
# the model never saw in training, a feature that stopped updating three weeks
# ago.
#
# After this lab you should be able to:
#
# 1. explain the parts of a feature store: entities, feature views, offline and online stores, materialisation, TTL;
# 2. build user and item features as daily snapshots in SQL, with windows that end strictly before the snapshot;
# 3. build a point-in-time-correct training set with an as-of join, and handle entities missing from a snapshot with a TTL;
# 4. demonstrate a same-day leak and measure what it does to a validation metric, both offline and "in production";
# 5. reason about backfills and feature freshness with numbers;
# 6. serve the same values from an online store with TTL semantics, and test online/offline parity.
#
# **Outputs of this lab:** nothing on disk besides the cached dataset; the training sets are kept in memory.

# %% [markdown]
# Read first: [Chapter 21 text](README.md) — feature stores, point-in-time joins, time-to-live, leakage, backfills, freshness and parity, explained in full.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

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
import time

import duckdb
import numpy as np
import pandas as pd

from shoprank import features as F
from shoprank import scale, store
from grading import ch21 as grade

world, users, items, imps = store.load("small")
imps = imps.merge(items[["item_id", "category"]], on="item_id", how="left")
START = world.config.start
con = duckdb.connect()
con.execute("SET threads = 2")
con.register("imps", imps)
con.register("items", items)
print(f"{len(imps):,} impressions from {START}; train days {store.TRAIN_DAYS}, valid days {store.VALID_DAYS}")

# %% [markdown]
# **Expected:** `2,446,900 impressions from 2025-03-03; train days range(0, 42), valid days range(42, 49)`.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Entity** | the thing a feature describes, identified by a key | a user (`user_id`), an item, a user × category |
# | **Feature view** | a named group of features for one entity, with its definition and TTL | `user_snap`: impressions and clicks, last 7 days |
# | **Snapshot** | feature values computed for a cut-off time, stamped with it | the row for day `d`, `snapshot_ts = d 00:00` |
# | **Offline store** | the full history of feature values, for building training sets | DuckDB tables of daily snapshots |
# | **Online store** | only the latest value per key, readable in about a millisecond | `OnlineStore`, a dict with a TTL |
# | **Point-in-time join** | for each event, the feature values as they were just before it | `ASOF JOIN … e.ts > s.snapshot_ts` |
# | **TTL** | the maximum age before a stored value counts as missing | 1 day for daily snapshots |
# | **Materialisation** | copying the newest offline values into the online store | the daily loop in 21.6 |
# | **Backfill** | computing a new feature for past dates | recomputing day 30 alone must equal the full run |
# | **Freshness** | how old the newest data behind a served value is | lag 0, 1, 3, 7 days in 21.5 |
# | **Leakage** | a feature that uses information not available at prediction time | the same-day window `[d-6, d]` |
# | **Training/serving skew** | the model sees different feature values in production than in training | a 7-day TTL online, 1-day offline |

# %% [markdown]
# ## 21.1 · What a feature store is (and is not)
#
# A feature store is two databases and a contract between them:
#
# | Concept | Meaning | Here |
# |---|---|---|
# | **Entity** | the thing a feature describes, with a key | user (`user_id`), item (`item_id`), user × category |
# | **Feature view** | a named set of features for an entity, with its definition (SQL), its source and its TTL | `user_snap`: impressions and clicks in the last 7 days |
# | **Offline store** | *history*: every value each feature had, with a timestamp; big, cheap, columnar (Parquet, BigQuery) | DuckDB tables of daily snapshots |
# | **Online store** | *latest* value per entity key, readable in ~1 ms (Redis, DynamoDB, Bigtable) | `OnlineStore`, a dict with a TTL |
# | **Point-in-time join** | builds training rows: for each event, the feature values *as they were* just before it | `ASOF JOIN … e.ts > snapshot_ts` |
# | **Materialisation** | copies the latest offline values into the online store on a schedule | a daily loop in 21.6 |
# | **TTL** | how old a value may be before it counts as missing | 1 day for daily snapshots |
#
# The point of the contract: **one definition** of each feature, used both to
# build training data (offline, as of each past event) and to serve it
# (online, as of now). Feast, Tecton, Vertex AI and SageMaker Feature Store are
# implementations of these ideas; this lab builds a small one so that their
# behaviour is understood.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-feature-store.svg" alt="Offline store plus point-in-time join for training; online store with TTL for serving; a parity test between them" width="100%">
#
# *What to notice:* both branches start from the *same* snapshots; they differ only in *which* timestamp they read as of (each past event vs now). The dashed parity line is the test that checks that they agree.

# %% [markdown]
# ## 21.2 · Daily aggregates and snapshots
#
# Two steps, both plain SQL:
#
# 1. **Daily aggregates**: impressions and clicks per user per day, per item
#    per day, clicks per (user, category) per day.
# 2. **Snapshots**: for each snapshot day `d`, sum the daily aggregates over the
#    window `[d-7, d-1]` (14 days for categories) and stamp the row
#    `snapshot_ts = d 00:00`. Everything in a snapshot was known at its
#    timestamp.
#
# ```
# day:        d-7  d-6  ...  d-1 | d  (events being scored happen here)
# window:     [================]  ^ snapshot_ts = d 00:00
# ```

# %%
with scale.timed("daily aggregates"):
    con.execute(F.DAILY_SQL)
print(F.SNAPSHOT_SQL.split(";")[0].format(start=START, first=-7, last=-1))
with scale.timed("snapshots for all 56 days"):
    F.build_snapshots(con, START, days=range(56))
for t in ("user_day", "item_day", "user_cat_day", "user_snap", "item_snap", "user_cat_snap"):
    print(f"{t:14s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>10,} rows")
con.execute("SELECT * FROM user_snap WHERE user_id = 42 ORDER BY snap_day LIMIT 4").df()

# %% [markdown]
# **Expected:** about a second in total; about 410,000 user-snapshot rows
# (one per user and day on which the user had activity in the previous
# week), about 200,000 item-snapshot rows and 1.5 million user × category
# rows. User 42's rows show counts for the 7 days *before* each
# `snapshot_ts`, starting the day after their first activity; there is no
# snapshot for day 0 at all, because there is no history before it.

# %% [markdown]
# ## 21.3 · A point-in-time training set
#
# For every impression, attach the **latest snapshot strictly before the
# impression's timestamp**. That is an as-of join (Chapter 20) per entity,
# with a strict `>`:

# %%
print(F.PIT_SQL)

# %% [markdown]
# > **Intuition.** A point-in-time join resembles work in a newspaper archive: to explain why a stock moved on 3 March, only papers printed *before* 3 March may be used. A same-day snapshot is the evening edition, printed after the event and stamped with that morning's date. *Where it breaks down:* newspapers carry trustworthy dates; a feature table has only the timestamp the pipeline wrote, which is why `snapshot_ts < ts` should be asserted in code rather than assumed from the table's name.
#
# There is one trap. Snapshots only contain entities that were *active* in the window.
# If user 7 was inactive last week, there is no day-`d` row for them, and the
# as-of join returns their snapshot from three weeks ago, carrying
# stale counts forward as if they were current. A real feature store guards
# against that with the feature view's **TTL**: a value older than the TTL
# counts as missing (here: missing activity = 0). Our snapshots are daily, so
# TTL = 1 day. We build the training set (a 20% request sample of days 14-41, so
# every 14-day window is full) and the validation set (all of week 7), and
# count how often the TTL applies.

# %%
keep = np.array([scale.stable_bucket(r, 100, salt="nb03") < 20 for r in range(imps.request_id.max() + 1)])
train_events = imps[imps.day.between(14, 41) & keep[imps.request_id]]
valid_events = imps[imps.day.isin(store.VALID_DAYS)]

with scale.timed("point-in-time join (train + valid)"):
    train = F.point_in_time_features(con, train_events)
    valid = F.point_in_time_features(con, valid_events)

stale = (train.ts - train.user_snapshot_ts) > pd.Timedelta(days=1)
print(f"train {len(train):,} rows | valid {len(valid):,} rows")
print(f"rows whose latest user snapshot is older than the TTL: {stale.mean():.1%} "
      f"(no user snapshot at all: {train.user_snapshot_ts.isna().mean():.1%})")
leaks = (train.user_snapshot_ts >= train.ts).sum() + (train.item_snapshot_ts >= train.ts).sum()
print("rows with a snapshot at or after the event:", leaks)
train[["ts", "user_id", "user_snapshot_ts", "user_imps_7d", "user_clicks_7d", "item_imps_7d"]].head(3)

# %% [markdown]
# **Expected:** 246,570 training rows and 325,810 validation rows; about 5%
# of training rows would have carried a stale user snapshot forward without
# the TTL (another ~1.4% have no user snapshot at all); `0` rows with a
# snapshot at or after the event. Each
# `user_snapshot_ts` is the midnight starting the event's day.
#
# > **Exercise 1.** Implement the as-of join in pandas:
# > `point_in_time_join(events, snapshots, key, ts_col)` returns `events`
# > with all other columns of `snapshots` attached from the **latest snapshot
# > with the same key and a timestamp strictly before** the event's
# > timestamp (NaN when there is none). Both frames use the column names
# > `key` and `ts_col`. Return the events in their **original order**, one
# > output row per event. The grader compares with DuckDB's
# > `ASOF LEFT JOIN … ON e.key = s.key AND e.ts > s.ts`, on data with ties
# > and unsorted events.
# >
# > Hint: `pd.merge_asof` needs both frames sorted by the time column and has
# > `allow_exact_matches`. How can the original order be restored?

# %%
def point_in_time_join(events: pd.DataFrame, snapshots: pd.DataFrame, key: str, ts_col: str) -> pd.DataFrame:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(point_in_time_join)

# %% [markdown]
# ## 21.4 · The same-day leak, measured
#
# A very common bug: the snapshot for day `d` is computed over `[d-6, d]`
# instead of `[d-7, d-1]`. Each row is still "as of" day `d`, nothing looks
# wrong, and the as-of join still uses `>`. But a snapshot built at the *end*
# of day `d` and stamped at its *start* contains day `d`'s clicks, including
# the click being predicted. We train a small logistic regression (log counts,
# smoothed CTRs, log position; see `F.model_matrix`) in three settings and
# validate on week 7:

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit(df):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(F.model_matrix(df), df.clicked)


def auc(model, df):
    return roc_auc_score(df.clicked, model.predict_proba(F.model_matrix(df))[:, 1])


honest_model = fit(train)
F.build_snapshots(con, START, range(56), include_same_day=True)
leaky_train = F.point_in_time_features(con, train_events)
leaky_valid = F.point_in_time_features(con, valid_events)
leaky_model = fit(leaky_train)
F.build_snapshots(con, START, range(56))  # restore the correct snapshots

position_only = LogisticRegression().fit(np.log1p(train[["position"]].astype(float)), train.clicked)
results = pd.Series({
    "position only (baseline)": roc_auc_score(
        valid.clicked, position_only.predict_proba(np.log1p(valid[["position"]].astype(float)))[:, 1]),
    "honest features, honest validation": auc(honest_model, valid),
    "leaky features, leaky validation (what the offline report says)": auc(leaky_model, leaky_valid),
    "leaky model served honest features (what production gets)": auc(leaky_model, valid),
}, name="valid AUC").round(4)
results.to_frame()

# %% [markdown]
# **Expected:** position alone gives AUC about 0.73; honest features about
# 0.793. The leaky pipeline *reports* about 0.85, a large improvement that
# does not exist: served with features that are actually
# available at request time, the leaky model scores about 0.786, **below**
# the honest model, because it learned to trust a same-day CTR signal that
# production never has.
#
# Two lessons. A validation metric computed with the same leaky pipeline
# cannot detect the leak: train and validation leak together. And the leak
# does not cause any failure; it shows up only as a model that is excellent
# offline and disappointing in the A/B test. The defence is in code: assert
# `snapshot_ts < event ts` for every row (as in 21.3), and compare offline
# feature values with values logged at serving time (21.6).

# %% [markdown]
# ## 21.5 · Backfills and freshness
#
# A **backfill** computes a feature for past dates: a new feature needs
# history before a model can train on it. Two properties make backfills safe:
#
# * **Deterministic and idempotent.** Recomputing day 30 alone must give the
#   same rows as day 30 from the full backfill; re-running must not
#   duplicate rows.
# * **Point-in-time.** The backfilled value for day 30 must use only data that
#   would have existed on day 30, including late-arriving data that landed
#   after day 30 only if it would also arrive before serving in production
#   (this one is subtle; streaming pipelines see late data differently).

# %%
full_day30 = con.execute("SELECT * FROM user_snap WHERE snap_day = 30 ORDER BY user_id").df()
F.build_snapshots(con, START, days=[30])
alone_day30 = con.execute("SELECT * FROM user_snap ORDER BY user_id").df()
print("day 30 recomputed alone == from the full backfill:", alone_day30.equals(full_day30))
F.build_snapshots(con, START, days=range(56))

# %% [markdown]
# **Expected:** `True`.
#
# > **Intuition.** A TTL is like the use-by date on milk: after it, the value is discarded and treated as missing, even if it is still in the refrigerator. Without it, an inactive user's three-week-old counts would be served as if they were yesterday's. *Where it breaks down:* milk goes bad gradually; a TTL is a hard cliff, so a value one second past it is discarded while one second before is trusted fully.
#
# **Freshness** is how old the newest data in a served feature is. A daily
# batch pipeline that runs at 02:00 serves features 2-26 hours old; if it
# breaks, features silently age. How much is freshness worth here? We build
# snapshots that are 1, 3 and 7 days *staler* than necessary (the snapshot
# for day `d` only sees data up to `d-1-lag`), retrain and validate:

# %%
fresh_rows = {}
for lag in (0, 1, 3, 7):
    F.build_snapshots(con, START, range(56), lag_days=lag)
    tr, va = F.point_in_time_features(con, train_events), F.point_in_time_features(con, valid_events)
    fresh_rows[f"lag {lag} d"] = round(auc(fit(tr), va), 4)
F.build_snapshots(con, START, range(56))
pd.Series(fresh_rows, name="valid AUC").to_frame().T

# %% [markdown]
# **Expected:** AUC falls only from about 0.793 to 0.792 between lag 0 and
# lag 7. The interpretation: ShopStream users have *fixed* tastes and items
# have fixed quality, so last week's CTR is almost as informative as
# yesterday's. Real products differ: trends, new items, sales and news make
# fresh features valuable, which is why Chapter 22 builds streaming features.
# The method matters more than the number: before paying for a streaming
# pipeline, measure what freshness is worth on the actual data.

# %% [markdown]
# ## 21.6 · The online store
#
# At serving time there is no as-of join: the service reads the *latest*
# values per key from the online store, in about a millisecond. A daily job
# **materialises** each new snapshot into it. The online store needs the same
# TTL rule as the offline join, or the two disagree for inactive users.
#
# > **Exercise 2.** Implement `OnlineStore(ttl_seconds)`:
# >
# > * `put(key, features, event_ts)` stores a dict of feature values as of
# >   `event_ts` (seconds) and returns `True`; if the stored value is *newer*
# >   (`event_ts` smaller than the stored one) it ignores the write and returns
# >   `False`; an equal timestamp overwrites;
# > * `get(key, now)` returns the feature dict, or `None` if the key is unknown
# >   or the value is expired: `now - event_ts > ttl_seconds` (exactly at the
# >   TTL it is still valid).
# >
# > Why must an online store ignore older writes rather than apply writes in
# > arrival order?

# %%
class OnlineStore:
    def __init__(self, ttl_seconds: float):
        # YOUR CODE HERE
        raise NotImplementedError

    def put(self, key, features: dict, event_ts: float) -> bool:
        raise NotImplementedError

    def get(self, key, now: float):
        raise NotImplementedError


grade.ex2(OnlineStore)

# %% [markdown]
# The package version is `shoprank.features.OnlineStore` (read
# [`src/shoprank/features.py`](../../src/shoprank/features.py)). We use it to
# serve day 45: materialise the user snapshots of days 40-45 in order (as a
# daily job would), then read features for every day-45 request at its
# request time, and compare with the offline point-in-time features of the
# same impressions. This is an **online/offline parity test**.

# %%
online = F.OnlineStore(ttl_seconds=86_400)
snaps = con.execute("SELECT * FROM user_snap WHERE snap_day BETWEEN 40 AND 45 ORDER BY snap_day").df()
with scale.timed("materialise 6 daily snapshots"):
    for row in snaps.itertuples(index=False):
        online.put(row.user_id, {"user_imps_7d": row.user_imps_7d, "user_clicks_7d": row.user_clicks_7d},
                   event_ts=row.snapshot_ts.timestamp())
print(f"online store holds {len(online):,} users")

day45 = valid[valid.day == 45].drop_duplicates("request_id")
served = []
for r in day45.itertuples(index=False):
    feats = online.get(r.user_id, now=r.ts.timestamp())
    served.append((feats or {"user_imps_7d": 0, "user_clicks_7d": 0})["user_imps_7d"])
served = np.array(served, dtype=float)
mismatch = served != day45.user_imps_7d.to_numpy()
print(f"{len(day45):,} day-45 requests | online == offline for {1 - mismatch.mean():.2%} | "
      f"served default (missing or expired): {np.mean(served == 0):.1%}")

# %% [markdown]
# **Expected:** about 8,900 users in the store, 4,739 day-45 requests,
# about 6% of them served the default (no fresh snapshot), and **100.00%**
# agreement:
# because the online store applies the same TTL as the offline join, an
# inactive user's old value expires in both places. With `ttl_seconds=7*86_400`
# (re-run the two cells to see it), the online store serves stale counts that the
# training data never contained, and agreement drops: training/serving skew
# created by a single configuration value.
#
# In production, parity is checked continuously: the service **logs the
# feature values it served** with the request id, and a daily job joins them to
# the offline values for the same requests. That logged-features table is also
# the safest source of training data: by construction it is what the model
# will see.

# %% [markdown]
# > **Exercise 3 (written).** A team adds a feature "item CTR over the last
# > hour" computed by a streaming job, while all other features stay daily. A
# > team member builds training data by joining impressions to the streaming job's
# > output table *as it is today*. List what can go wrong (at least three
# > distinct problems), and describe how to produce a point-in-time-correct
# > training set for this feature and how to verify it.
# >
# > Write the answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · A window feature, cut off strictly before the snapshot.** In words: the value stamped at day $d$ sums the 7 *complete* days before $d$.
#
# $$
# x_u(d) = \sum_{t=d-7}^{d-1} \text{clicks}_u(t)
# $$
#
# * $x_u(d)$: user $u$'s feature in the snapshot stamped $d$ 00:00; $\text{clicks}_u(t)$: their clicks on day $t$. *Example:* clicks 0, 2, 1, 0, 0, 3, 1 on days $d-7 \dots d-1$ give $x_u(d) = 7$. The leaky version sums $t = d-6 \dots d$ and so includes today's click, the one being predicted.
#
# **2 · The point-in-time rule, with a TTL.** In words: take the latest snapshot strictly before the event, and use it only if it is young enough.
#
# $$
# s^{*}(e) = \underset{s:\; k_s = k_e,\; \tau_s < t_e}{\arg\max}\; \tau_s, \qquad \text{use } s^{*} \text{ only if } t_e - \tau_{s^{*}} \le \text{TTL}
# $$
#
# * $e$: an event with key $k_e$ and time $t_e$; $s$: a snapshot with key $k_s$ and timestamp $\tau_s$. *Example:* an impression on day 30 at 14:00; the user's snapshots exist for days 12 and 13 only. The as-of join returns day 13 (17 days old); with TTL = 1 day it counts as missing (activity = 0).
#
# **3 · A smoothed click rate.** In words: add $\alpha$ pseudo-impressions at the prior rate, so tiny counts do not produce extreme rates.
#
# $$
# \widetilde{\text{CTR}} = \frac{\text{clicks} + \alpha\, p_0}{\text{impressions} + \alpha}
# $$
#
# * $p_0$: the prior (global) click rate; $\alpha$: the prior's weight in impressions. *Example:* 1 click in 2 impressions with $p_0 = 0.137$ and $\alpha = 20$: $(1 + 2.74) / 22 = 0.17$ instead of a raw 0.50. Compute $p_0$ from past data only, or the prior itself leaks (Chapter 24 found that bug).
#
# **4 · Feature age at serving.** In words: what the model sees online is as old as the newest data behind it.
#
# $$
# \text{age} = t_\text{request} - t_\text{newest event in the value}
# $$
#
# * *Example:* a daily job that runs at 02:00 over data up to midnight serves values 2 to 26 hours old; if it fails for three days, 74 to 98 hours. Feature age should therefore be monitored.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Validation AUC suspiciously high (≫ previous model) | features use data from the event's own day or later | assert `snapshot_ts < ts` per row; check window bounds `[d-7, d-1]` |
# | `merge_asof`: `left keys must be sorted` | merge_asof needs both frames sorted by the time column | sort, join, then restore the original order (keep a row id) |
# | `merge_asof`: `incompatible merge keys` | `int32` vs `int64` key or `datetime64[ns]` vs `[us]` | cast both columns to the same dtype |
# | PIT join returns values for users inactive for weeks | missing TTL: the as-of join carries old snapshots forward | null out values whose snapshot is older than the TTL |
# | Online vs offline features disagree for some users | different TTL, a missed materialisation run, or writes applied out of order | same TTL in both; materialise in order; ignore older writes |
# | Snapshot build very slow | joining every day to every daily row without the `BETWEEN` window | keep the range join; aggregate to daily first |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. Explain the difference between an offline and an online feature store. Why are both needed, and what keeps them consistent?
# 2. What is a point-in-time join? Write the SQL for one, and say why the inequality is strict.
# 3. A new feature lifts offline AUC from 0.79 to 0.85, but the A/B test is flat or negative. Describe how to find out why.
# 4. What does a TTL on a feature view protect against, offline and online? How do you choose it?
# 5. A product manager asks for real-time features. How do you decide whether freshness is worth the cost, and what would you measure first?
# 6. You must backfill a new 30-day feature for two years of history. What makes a backfill correct, safe to re-run, and affordable?
#
# **Next chapter:** [Chapter 22 · Streaming features](../ch22-streaming-features/notebook.ipynb) — computing features from a live, out-of-order event stream, and showing that they match the batch definition.
