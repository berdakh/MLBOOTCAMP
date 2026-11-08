# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 20 · Aggregation & joins at scale
#
# **Day 1 · ~90 minutes · CPU only**
#
# A `GROUP BY` over a billion rows does not run on one machine: the engine
# splits it into map tasks, a shuffle and reduce tasks, and most slow or
# failed production jobs come down to that shuffle: too much data moved, or
# one reducer handed a "hot" key. This notebook builds the machinery by hand on
# ShopStream so that you can reason about any engine's plan, and then covers
# the joins and approximations that big pipelines rely on.
#
# By the end you will be able to:
#
# 1. explain map → shuffle → reduce, and why a combiner shrinks the shuffle, with a hand-written partitioned aggregation checked against DuckDB;
# 2. detect key skew, and fix it with salting (two-stage aggregation);
# 3. choose between broadcast, shuffle-hash and sort-merge joins, and avoid the row explosion of joining to a skewed many-rows-per-key table;
# 4. attach "the value as of the event time" with an as-of join;
# 5. deduplicate re-delivered events with `ROW_NUMBER`;
# 6. trade accuracy for speed and memory with HyperLogLog, approximate quantiles and reservoir sampling, and say when that trade is acceptable.
#
# **You will produce:** nothing on disk besides the cached ShopStream dataset; this notebook is about how queries execute.

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

from shoprank import scale, store
from grading import ch20 as grade

world, users, items, imps = store.load("small")
con = duckdb.connect()
con.execute("SET threads = 2")
con.register("imps_df", imps)
con.execute("CREATE TABLE imps AS SELECT * FROM imps_df")
con.register("items", items)
print(f"{len(imps):,} impressions, {len(items):,} items, {len(users):,} users")

# %% [markdown]
# **Expected:** `2,446,900 impressions, 4,000 items, 10,000 users` (the
# dataset is generated on first use: about 10 s here, up to a minute on Colab).

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Map task** | reads one input split and emits (key, value) pairs | 4 mappers over the impression files |
# | **Shuffle** | sends every pair over the network to the reducer that owns its key | reducer = `hash(key) % n_reducers` |
# | **Reduce task** | merges all values of the keys it owns | sums clicks per `item_id` |
# | **Combiner** | a map-side pre-aggregation that shrinks the shuffle | 2.4M rows → ~15k partial rows |
# | **Algebraic aggregate** | partial results merge exactly into a fixed-size state | count, sum, avg as (sum, count) |
# | **Holistic aggregate** | state grows with the data; cannot combine to a fixed size | count(DISTINCT), median |
# | **Key skew** | a few keys hold a large share of the rows | the top seller's (seller, user) pairs |
# | **Salting** | splitting a hot key into S sub-keys, then re-combining | `(seller, hash(user) % S)` |
# | **Broadcast join** | copy the small table to every worker; no shuffle of the big one | impressions ⋈ sellers |
# | **As-of join** | attach the latest right row with timestamp ≤ the left row's | the price when the item was shown |
# | **At-least-once** | every event arrives, some more than once | 2% re-delivered impressions |
# | **Sketch** | a small fixed-size summary with a known error | HyperLogLog, t-digest, a reservoir |

# %% [markdown]
# ## 2.1 · Map → shuffle → reduce, by hand
#
# A distributed `SELECT item_id, count(*), sum(clicked) … GROUP BY item_id`
# runs in three steps:
#
# 1. **Map.** Each mapper reads one input split (a few files) and emits
#    `(key, value)` pairs. Optionally it **combines**: pre-aggregates its own
#    pairs by key, so it emits one row per key it saw instead of one per input
#    row.
# 2. **Shuffle.** Every pair is sent over the network to reducer
#    `hash(key) % n_reducers`. All rows of a key meet on one reducer. This is
#    the expensive step: disk, network, sorting.
# 3. **Reduce.** Each reducer merges the values per key.
#
# Here is all of it in about fifteen lines, checked against DuckDB.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-map-shuffle-reduce.svg" alt="Map, combine, shuffle by hash(key), reduce: a distributed GROUP BY" width="100%">
#
# *What to notice:* the shuffle in the middle is the only step that crosses the network, and the combiner decides how much goes into it.
#
# > **Intuition.** A combiner is counting votes at each polling station and phoning in the totals, instead of trucking every ballot to the capital. *Where it breaks down:* totals add up, but "the median voter's age" does not: a polling station cannot pre-compute its share of a median, which is why holistic aggregates are expensive.

# %%
def map_shuffle_reduce(df: pd.DataFrame, key: str, n_mappers: int = 4, n_reducers: int = 8,
                       combine: bool = True):
    bounds = np.linspace(0, len(df), n_mappers + 1).astype(int)
    inboxes = [[] for _ in range(n_reducers)]           # what each reducer receives
    for lo, hi in zip(bounds[:-1], bounds[1:]):          # MAP: one input split per mapper
        split = df.iloc[lo:hi]
        out = pd.DataFrame({key: split[key].to_numpy(), "impressions": 1,
                            "clicks": split["clicked"].to_numpy().astype(int)})
        if combine:                                      # COMBINE: local pre-aggregation
            out = out.groupby(key, as_index=False).sum()
        dest = scale.partition_of(out[key].to_numpy(), n_reducers)
        for r in range(n_reducers):                      # SHUFFLE: route rows by hash(key)
            inboxes[r].append(out[dest == r])
    reduced = [pd.concat(box).groupby(key).sum() for box in inboxes]   # REDUCE
    shuffled_rows = [sum(len(part) for part in box) for box in inboxes]
    return pd.concat(reduced).sort_index(), shuffled_rows


for combine in (False, True):
    with scale.timed(f"by hand, combine={combine}"):
        by_hand, loads = map_shuffle_reduce(imps, "item_id", combine=combine)
    print(f"   rows shuffled: {sum(loads):,}   per reducer: {loads}")

ref = con.execute("SELECT item_id, count(*) AS impressions, sum(clicked::INT) AS clicks "
                  "FROM imps GROUP BY item_id ORDER BY item_id").df().set_index("item_id")
print("identical to DuckDB:", by_hand.equals(ref.astype(by_hand.dtypes)))

# %% [markdown]
# **Expected:** without a combiner all 2,446,900 rows are shuffled; with it,
# about 15,000 (4 mappers × ~3,700 items each), roughly **160× less data moved**,
# and the result is identical to DuckDB's (`True`). The reducers' loads are
# within a few percent of each other: ShopStream's item popularity is only
# mildly skewed.
#
# A combiner works because `count` and `sum` are *algebraic*: partial results
# merge exactly. `avg` works if you carry `(sum, count)`; `count(DISTINCT)`,
# medians and "collect all values" do not combine to a fixed size, which is why
# they are the slow, memory-hungry aggregates (notebook 01 hit exactly that).
#
# You will implement the same map → shuffle → reduce for plain counting in
# Exercise 2.

# %% [markdown]
# ## 2.2 · Skew, and salting
#
# Real marketplaces are dominated by a few big sellers, real search logs by a
# few queries, and real pipelines by a few junk keys (`NULL`, `-1`,
# `"unknown"`). Give each ShopStream item a **seller**, with seller sizes
# following a power law, as in real catalogues.

# %%
rng = np.random.default_rng(7)
n_sellers = 500
seller_weight = 1.0 / np.arange(1, n_sellers + 1) ** 1.1
item_seller = rng.choice(n_sellers, size=len(items), p=seller_weight / seller_weight.sum())
imps["seller_id"] = item_seller[imps.item_id.to_numpy()]
share = imps.seller_id.value_counts(normalize=True)
print(f"top seller: {share.iloc[0]:.1%} of impressions | top 5: {share.iloc[:5].sum():.1%} | "
      f"sellers with impressions: {len(share)}")

# %% [markdown]
# Counting impressions per seller is harmless: the combiner shrinks every
# mapper's output to one row per seller. Now ask a question that does **not**
# combine to a fixed size: *distinct users per seller*. The mapper can only
# drop duplicate `(seller, user)` pairs it sees itself; every remaining pair
# must travel to the seller's reducer, and the biggest seller's reducer gets
# all of its pairs.
#
# **Salting** splits a hot key across reducers. Route each pair to
# `(seller, salt)` with `salt = hash(user) % S`. Each reducer counts distinct
# users for its `(seller, salt)` sub-key; because a user always gets the same
# salt, the sub-keys hold disjoint users, and a second, tiny stage adds the
# `S` partial counts per seller: an exact answer.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-skew-salting.svg" alt="Key skew makes one reducer a straggler; salting spreads the hot key" width="100%">
#
# *What to notice:* the job's time is the *tallest* bar, not the average. Salting makes the bars even, and a tiny second stage adds the partial counts back up.
#
# > **Intuition.** Skew is a supermarket where a celebrity's fans all queue at checkout 3 while the other lanes sit idle. Salting hands each fan a ticket number (`hash(user) % S`) that sends them to one of S lanes; at closing, the S lane totals are added. Because a fan always gets the same ticket, nobody is counted twice. *Where it breaks down:* if you do not know which key is hot, salting every key multiplies the work for all the small ones.

# %%
N_REDUCERS, S = 64, 16
bounds = np.linspace(0, len(imps), 5).astype(int)  # 4 mappers
pairs = pd.concat([imps.iloc[lo:hi][["seller_id", "user_id"]].drop_duplicates()  # map-side dedup
                   for lo, hi in zip(bounds[:-1], bounds[1:])])
seller = pairs.seller_id.to_numpy()
salt = (scale.splitmix64(pairs.user_id.to_numpy()) % np.uint64(S)).astype(np.int64)


def load_report(dest):
    load = np.bincount(dest, minlength=N_REDUCERS)
    return {"rows shuffled": int(load.sum()), "max reducer": int(load.max()),
            "max / mean": round(load.max() / load.mean(), 2)}


plain_dest = scale.partition_of(seller, N_REDUCERS)
salted_dest = scale.partition_of(seller * S + salt, N_REDUCERS)
print(pd.DataFrame({"by seller": load_report(plain_dest),
                    f"by (seller, salt), S={S}": load_report(salted_dest)}).T)

# stage 1: distinct users per (seller, salt); stage 2: sum over salts
stage1 = pairs.assign(salt=salt).drop_duplicates().groupby(["seller_id", "salt"]).user_id.nunique()
salted_answer = stage1.groupby(level="seller_id").sum()
exact = imps.groupby("seller_id").user_id.nunique()
print("salted two-stage answer exact:", salted_answer.sort_index().equals(exact.sort_index()))

# %% [markdown]
# **Expected:** without salt, the busiest of 64 reducers receives several
# times the average load (it holds all of the top seller's pairs), so the
# whole job waits for it: the classic "99% of
# tasks finished an hour ago" straggler. Salting brings max/mean close to 1.
# The answer is exact (`True`).
#
# Salting costs a second stage and `S×` more partial rows for small keys; in
# practice you salt only the hot keys (found from a sample or from last
# run's statistics). Spark 3's adaptive query execution splits skewed
# partitions automatically for joins. Know the manual version anyway: it is
# the interview answer, and the fix when the automatic one is not enough.

# %% [markdown]
# ## 2.3 · Join strategies, and the skewed-join trap
#
# | Strategy | How it works | Use when | Cost |
# |---|---|---|---|
# | **Broadcast hash join** | copy the small table to every mapper; each joins its split locally | one side fits in memory (items, sellers, dimension tables) | no shuffle of the big side |
# | **Shuffle hash join** | hash-partition *both* sides by key; each reducer builds a hash table of one side | both sides large, one side's partitions fit in memory | full shuffle of both sides; skew → one huge partition |
# | **Sort-merge join** | partition and **sort** both sides by key, then merge | both sides huge (Spark's default for large-large) | shuffle + sort; robust because it spills; skew still hurts |
#
# Skew in a join is worse than in an aggregation, because nothing combines:
# if a key has `a` rows on the left and `b` on the right, the join produces
# `a × b` rows for it. Join the impressions to a **promotions** table where
# big sellers run many promotions (rows per seller grow with the seller's size):

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-join-strategies.svg" alt="Broadcast hash join versus shuffle hash join" width="100%">
#
# *What to notice:* in a broadcast join the big table never moves; in a shuffle join both tables cross the network, so a hot key lands entirely on one partition.

# %%
seller_items = np.bincount(item_seller, minlength=n_sellers)
n_promos = np.maximum(1, seller_items // 4)
promos = pd.DataFrame({"seller_id": np.repeat(np.arange(n_sellers), n_promos)})
promos["discount"] = np.round(rng.uniform(0.05, 0.5, len(promos)), 2)
con.register("promos", promos)
con.register("imps_df", imps)  # re-register: the frame now has seller_id
con.execute("CREATE OR REPLACE TABLE imps_s AS SELECT * FROM imps_df")

imp_per_seller = imps.seller_id.value_counts()
promo_per_seller = promos.seller_id.value_counts()
predicted = int((imp_per_seller * promo_per_seller.reindex(imp_per_seller.index)).sum())
print(f"promotions: {len(promos):,} rows; top seller has {promo_per_seller.max()} of them")
print(f"predicted join output: {predicted:,} rows ({predicted / len(imps):.0f}x the impressions)")

t = {}
with scale.timed("naive join, then aggregate", t):
    naive = con.execute("""
        SELECT i.seller_id, count(*) AS rows_joined, avg(p.discount) AS avg_discount
        FROM imps_s i JOIN promos p USING (seller_id) GROUP BY i.seller_id""").df()
with scale.timed("aggregate promos first, then join", t):
    smart = con.execute("""
        WITH p AS (SELECT seller_id, count(*) AS n_promos, avg(discount) AS avg_discount
                   FROM promos GROUP BY seller_id)
        SELECT i.seller_id, count(*) AS impressions, any_value(p.avg_discount) AS avg_discount
        FROM imps_s i JOIN p USING (seller_id) GROUP BY i.seller_id""").df()
print("rows joined (naive):", f"{naive.rows_joined.sum():,}", "| same avg_discount per seller:",
      np.allclose(naive.sort_values("seller_id").avg_discount, smart.sort_values("seller_id").avg_discount))

# %% [markdown]
# **Expected:** the join produces about 129 million rows (about 53× the
# 2.4M impressions), most of them from the top few sellers, and takes about
# 0.6 s against 0.02 s for aggregating the promotions table first, for the
# same per-seller answer. DuckDB survives because it never materialises the join;
# a distributed engine would shuffle those rows, and the reducer of the top
# seller would run out of memory or time.
#
# **Rules of thumb:** aggregate (or deduplicate) the many-rows-per-key side
# *before* joining; broadcast small tables (`promos` is a few thousand rows:
# Spark would broadcast it automatically below `spark.sql.autoBroadcastJoinThreshold`,
# 10 MB by default); salt only when both sides are big and skewed (salt the
# big side randomly, replicate the hot keys of the other side `S` times).
#
# In pandas, a broadcast join of a small dimension table is just a lookup:

# %%
with scale.timed("pandas merge with the items table"):
    merged = imps[["item_id"]].merge(items[["item_id", "category"]], on="item_id", how="left")
with scale.timed("broadcast as an array lookup"):
    category_of = items.set_index("item_id").category.reindex(np.arange(items.item_id.max() + 1)).to_numpy()
    looked_up = category_of[imps.item_id.to_numpy()]
print("same categories:", np.array_equal(merged.category.to_numpy(), looked_up))

# %% [markdown]
# **Expected:** `True`; the array lookup is several times faster than the
# `merge` because there is nothing to hash or shuffle. It is what a broadcast join does inside each mapper.

# %% [markdown]
# ## 2.4 · As-of joins: the value *at the time of the event*
#
# Prices change. To train a model with "price" as a feature, each impression
# needs the price **when it was shown**, not today's price. That is an
# **as-of join**: for each left row, take the right row with the same key and
# the latest timestamp `<=` the left timestamp. Build a price-change log
# (each item gets a few price changes over the eight weeks) and attach prices.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-asof-join.svg" alt="As-of join on a timeline: each event takes the latest price at or before its timestamp" width="100%">
#
# *What to notice:* each impression looks *back* to the most recent price record; nothing to its right on the timeline is visible to it.

# %%
start = pd.Timestamp(world.config.start)
n_changes = rng.poisson(2, len(items))
change_item = np.repeat(items.item_id.to_numpy(), n_changes)
price_log = pd.DataFrame({
    "item_id": np.concatenate([items.item_id.to_numpy(), change_item]),
    "valid_from": np.concatenate([np.full(len(items), start.to_datetime64()),
                                  (start + pd.to_timedelta(rng.uniform(0, 56 * 86400, len(change_item)), unit="s")).to_numpy()]),
    "price": np.concatenate([items.price.to_numpy(),
                             np.round(items.price.to_numpy()[change_item] * rng.uniform(0.7, 1.3, len(change_item)), 2)]),
}).astype({"item_id": "int32"}).sort_values(["item_id", "valid_from"], ignore_index=True)
con.register("price_log", price_log)
print(f"{len(price_log):,} price records for {price_log.item_id.nunique():,} items")

asof = con.execute("""
    SELECT i.request_id, i.position, i.item_id, i.ts, p.price AS price_at_ts
    FROM imps i ASOF JOIN price_log p
      ON i.item_id = p.item_id AND i.ts >= p.valid_from
    ORDER BY i.request_id, i.position""").df()
# merge_asof needs both keys at the same resolution. pandas 3 may store one
# as datetime64[us] and the other as [ns] and refuses to mix them, so say it.
left = imps[["request_id", "position", "item_id", "ts"]].astype({"ts": "datetime64[ns]"})
right = price_log.astype({"valid_from": "datetime64[ns]"})
check = pd.merge_asof(left.sort_values("ts"), right.sort_values("valid_from"),
                      left_on="ts", right_on="valid_from",
                      by="item_id", direction="backward").sort_values(["request_id", "position"])
latest = price_log.groupby("item_id").price.last()
print("DuckDB ASOF == pandas merge_asof:", np.allclose(asof.price_at_ts, check.price))
print(f"impressions whose price differs from the item's latest price: "
      f"{(asof.price_at_ts.to_numpy() != latest.reindex(asof.item_id).to_numpy()).mean():.0%}")

# %% [markdown]
# **Expected:** about 12,100 price records; `True`; and about **55%** of
# impressions would get the wrong price from a plain join to "current
# price". Worse than wrong: the current price is partly a *consequence* of the
# clicks you are predicting (a seller cuts the price of an item that sells
# badly), so it leaks the future. Notebook 03 builds whole feature tables
# this way.
#
# Strictness matters: `>=` lets an event see a price set at the same instant;
# for features computed from the event's own outcome you want strict `>`.
# DuckDB supports both; `pd.merge_asof` uses `allow_exact_matches=False` for
# strict.

# %% [markdown]
# ## 2.5 · Deduplicating re-delivered events
#
# Logging pipelines are usually **at-least-once**: a client retries when it
# does not get an acknowledgement, a consumer restarts and re-reads a batch.
# Duplicates are normal, not exceptional. Re-deliver 2% of one week's
# impressions and see what they do to the metrics.

# %%
week = imps[imps.day.between(42, 48)].drop(columns="seller_id")
raw_log = scale.inject_duplicates(week, frac=0.02, seed=1)
con.register("raw_log", raw_log)
print(f"clean rows {len(week):,} | raw rows {len(raw_log):,} | clicks clean {week.clicked.sum():,} "
      f"vs raw {raw_log.clicked.sum():,}")

dedup = con.execute("""
    SELECT * EXCLUDE (rn) FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY request_id, position ORDER BY ingest_ts) AS rn
        FROM raw_log)
    WHERE rn = 1
    ORDER BY request_id, position""").df()
print(f"after dedup: {len(dedup):,} rows; identical to the clean log:",
      dedup.drop(columns="ingest_ts").reset_index(drop=True).equals(week.reset_index(drop=True)))

# %% [markdown]
# **Expected:** about 326,000 clean rows, 2% more raw rows and clicks
# (44,188 → about 45,060 clicks); after
# dedup, the clean count again and `True`.
#
# Three details that matter in real pipelines:
#
# * The dedup key must identify the *event*, not the content: here
#   `(request_id, position)`. Deduplicating on all columns fails as soon as a
#   retry carries a different `ingest_ts`.
# * Keep the **first** delivery (`ORDER BY ingest_ts`) so the result is
#   deterministic. DuckDB's `QUALIFY rn = 1` is shorthand for the subquery.
# * Duplicates can cross partition boundaries (an event at 23:59 re-delivered
#   at 00:01). Dedup over a window that covers your maximum delivery delay, or
#   partition by *event* time, as here.

# %% [markdown]
# ## 2.6 · Approximate answers: HyperLogLog, quantile sketches, reservoirs
#
# Exact distinct counts and quantiles need memory that grows with the data.
# Sketches answer with a fixed, small state and a known error:
#
# * **HyperLogLog** (`approx_count_distinct`): hashes values, keeps per bucket
#   the longest run of leading zeros seen. A few KB, ~1-2% typical error, and
#   sketches **merge** (union), so they combine in the map step.
# * **Quantile sketches** (t-digest, KLL; `approx_quantile`): keep a compressed
#   summary of the distribution, accurate in the tails.
# * **Reservoir sampling**: a uniform sample of fixed size `k` from a stream of
#   unknown length, in one pass.
#
# > **Intuition (HyperLogLog).** If a friend flips coins all day and tells you only "my longest run of heads was 10", you can guess they flipped about $2^{10}$ times. HLL hashes every value to random-looking bits and remembers, per bucket, the longest run of leading zeros: a few bytes per bucket, whatever the number of values. *Where it breaks down:* one bucket is a very noisy guess (a lucky early run); HLL averages $m$ buckets, which is where the $1.04/\sqrt{m}$ error below comes from.
#
# Measure accuracy against speed on the impression table (each timing is the
# best of 3 runs).

# %%
def best_of(sql, n=3):
    times = []
    for _ in range(n):
        start_t = time.perf_counter()
        out = con.execute(sql).fetchall()
        times.append(time.perf_counter() - start_t)
    return out, min(times) * 1000


rows = []


def add(quantity, exact, approx, error, t_exact, t_approx):
    rows.append({"quantity": quantity, "exact": exact, "approximate": approx, "error": error,
                 "exact ms": round(t_exact, 1), "approx ms": round(t_approx, 1)})


pair = "(user_id::BIGINT * 100000 + item_id)"
((ex,),), t_ex = best_of(f"SELECT count(DISTINCT {pair}) FROM imps")
((ap,),), t_ap = best_of(f"SELECT approx_count_distinct({pair}) FROM imps")
add("distinct (user, item) pairs", f"{ex:,}", f"{ap:,}", f"{(ap - ex) / ex:+.1%}", t_ex, t_ap)

per_group = con.execute("""
    SELECT count(DISTINCT user_id) AS exact, approx_count_distinct(user_id) AS approx
    FROM imps GROUP BY item_id, day HAVING count(DISTINCT user_id) >= 20""").df()
_, t_ex = best_of("SELECT item_id, day, count(DISTINCT user_id) FROM imps GROUP BY item_id, day")
_, t_ap = best_of("SELECT item_id, day, approx_count_distinct(user_id) FROM imps GROUP BY item_id, day")
rel = (per_group.approx - per_group.exact).abs() / per_group.exact
add(f"distinct users per item-day ({len(per_group):,} groups with >= 20 users)",
    f"median {per_group.exact.median():.0f}", "", f"|rel| median {rel.median():.1%}, p99 {rel.quantile(0.99):.1%}",
    t_ex, t_ap)

((ex,),), t_ex = best_of("SELECT quantile_cont(policy_score, 0.99) FROM imps WHERE NOT is_random")
((ap,),), t_ap = best_of("SELECT approx_quantile(policy_score, 0.99) FROM imps WHERE NOT is_random")
add("p99 of policy_score", f"{ex:.3f}", f"{ap:.3f}", f"{ap - ex:+.3f}", t_ex, t_ap)

((ex,),), t_ex = best_of("SELECT avg(clicked::INT) FROM imps")
((ap,),), t_ap = best_of("SELECT avg(clicked::INT) FROM imps USING SAMPLE reservoir(20000 ROWS) REPEATABLE (42)")
add("CTR from a 20,000-row reservoir", f"{ex:.4f}", f"{ap:.4f}", f"{ap - ex:+.4f}", t_ex, t_ap)

pd.DataFrame(rows).set_index("quantity")

# %% [markdown]
# **Expected (our run; your times will differ):**
#
# * distinct `(user, item)` pairs: about 2.06 million exact; DuckDB's HLL
#   answered about 1.93 million, **about 6% low**, roughly 5× faster
#   (about 35 ms vs 185 ms);
# * distinct users per item-day: a median absolute relative error of about
#   6% and a p99 of about 24%. The error is *relative*, and DuckDB's sketch is
#   small (it favours memory over precision), so do not read individual small
#   groups off it;
# * p99 of the policy score: within about 0.01 of exact, but *slower* than
#   the exact quantile here;
# * CTR from a 20,000-row reservoir: within about ±0.005 of the exact 0.137
#   (binomial standard error `sqrt(0.137·0.863/20000) ≈ 0.0024`), and also
#   slower than scanning 2.4M in-memory rows.
#
# So on 2.4M rows in memory, approximation buys little speed and sometimes
# none. The case for sketches is **memory and mergeability** at scale: an
# exact 10-billion-row distinct count needs gigabytes of state per reducer,
# an HLL a few kilobytes, and daily HLL sketches can be stored and merged
# into weekly or monthly uniques without rescanning the raw log. Error
# depends on the sketch size: BigQuery's `HLL_COUNT` and Spark's
# `approx_count_distinct(col, rsd)` let you choose it (about 1% is typical).
# The case for a reservoir is when the *next* step is expensive per row
# (training, a Python loop, a human labelling). Do not use approximations
# where an exact count is a contract: billing, payments, legal reporting.

# %% [markdown]
# > **Exercise 1.** Implement `reservoir_sample(stream, k, seed)`: one pass
# > over an iterable of unknown length, memory `O(k)`, returning a list of `k`
# > items (all items if the stream has fewer than `k`), every item equally
# > likely to be in the sample. Use `random.Random(seed)` so the same seed
# > gives the same sample.
# >
# > (Algorithm R: keep the first `k`; for the `i`-th item, 0-indexed, with
# > `i >= k`, draw `j` uniformly from `0..i` and replace slot `j` if `j < k`.)
# > The grader runs your sampler with 2,000 seeds on a 20-item stream and
# > applies a chi-square test of uniformity, then checks the edge cases.

# %%
import random


def reservoir_sample(stream, k: int, seed: int) -> list:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(reservoir_sample)

# %% [markdown]
# > **Exercise 2.** Implement `partitioned_count(keys, n_partitions)`: count
# > how often each key occurs, map → shuffle → reduce style. Split `keys` into
# > 4 "mapper" chunks; in each, count keys locally (the combiner); route each
# > `(key, count)` to partition `stable_hash(key) % n_partitions`; let each
# > partition merge its counts; return one `dict` merging all partitions. It
# > must equal `collections.Counter(keys)` exactly, including for heavily
# > skewed input, string keys, an empty list and `n_partitions=1`.
# > (`scale.stable_bucket(key, n_partitions)` is a ready stable hash.)

# %%
from collections import Counter


def partitioned_count(keys, n_partitions: int) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(partitioned_count)

# %% [markdown]
# > **Exercise 3 (written).** Product asks for a dashboard of **distinct users
# > per item per day** over a 10-billion-row impression log (one year, 200M
# > users, 50M items), updated daily. Describe how you would compute it:
# > exact or approximate (and why), what you store per day so that "distinct
# > users per item over the last 7 or 30 days" does not rescan the log, how
# > you deal with a handful of items that appear in 5% of all impressions, and
# > how you would validate the numbers.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · Partitioning by hash.** In words: a key always goes to the same reducer, and keys spread evenly *if* no key is hot.
#
# $$
# \text{reducer}(k) = h(k) \bmod R
# $$
#
# * $h$: a stable hash; $R$: number of reducers. *Example:* with $R = 64$ and one seller holding 25% of the pairs, the reducer that owns it gets at least $0.25 \cdot 64 = 16\times$ the average load, whatever $h$ is. Only changing the key (salting) helps.
#
# **2 · Join output size.** In words: each key multiplies its left rows by its right rows, so the output is dominated by keys that are big on *both* sides.
#
# $$
# \lvert A \bowtie B \rvert = \sum_{k} a_k \, b_k
# $$
#
# * $a_k$, $b_k$: rows with key $k$ in $A$ and in $B$. *Example:* a seller with 100,000 impressions and 50 promotions alone contributes $5{,}000{,}000$ rows; aggregating promotions first ($b_k = 1$) contributes 100,000.
#
# **3 · HyperLogLog relative error.** In words: the standard error of a distinct count shrinks with the square root of the number of registers.
#
# $$
# \frac{\operatorname{SE}(\hat n)}{n} \approx \frac{1.04}{\sqrt{m}}, \qquad m = 2^{p}
# $$
#
# * $m$: registers (buckets), $p$: precision bits; each register is a few bits. *Example:* $m = 1{,}024$ gives about 3.3%; $m = 16{,}384$ (about 12 KB) gives about 0.8%, for 1 thousand or 1 billion distinct values alike.
#
# **4 · Reservoir sampling (Algorithm R).** In words: after reading $n$ items, every one of them is in the size-$k$ reservoir with the same probability.
#
# $$
# P(\text{item } i \text{ in sample after } n \text{ items}) = \frac{k}{n}
# $$
#
# * Item $i$ (counting from 1, $i > k$) enters with probability $k/i$; each later item $j$ evicts it with probability $\frac{k}{j} \cdot \frac{1}{k} = \frac{1}{j}$. So $\frac{k}{i} \prod_{j=i+1}^{n} \frac{j-1}{j} = \frac{k}{i} \cdot \frac{i}{n} = \frac{k}{n}$.
# * *Example:* $k = 5$, $n = 20$: every item is kept a quarter of the time, the first and the last alike.
#
# The cell below checks equation 4 by simulation (Algorithm R written out in five lines; your Exercise 1 version should behave the same).

# %%
# Check P(in reservoir) = k/n by simulation: k = 5, n = 20, 20,000 runs
import random as _random


def _algorithm_r(n, k, rnd):
    sample = list(range(min(n, k)))
    for i in range(k, n):
        j = rnd.randint(0, i)  # uniform on 0..i inclusive
        if j < k:
            sample[j] = i
    return sample


rnd = _random.Random(0)
hits = np.zeros(20)
for _ in range(20_000):
    hits[_algorithm_r(20, 5, rnd)] += 1
freq = hits / 20_000
print("k/n =", 5 / 20, "| simulated min/max over the 20 items:", freq.min().round(3), freq.max().round(3))

# %% [markdown]
# **Expected:** every item's frequency within about ±0.01 of 0.25 (roughly 0.24–0.26), the first items and the last alike.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Hand-written aggregation differs from DuckDB | dtypes differ (int64 vs int32) in `equals`, or rows not sorted by key | compare `sort_index()` results with `np.array_equal` / cast dtypes |
# | `Binder Error: … ASOF JOIN` | the inequality is not `>=`/`>`/`<`/`<=` on comparable types, or there is no equality key | `ON a.key = b.key AND a.ts >= b.valid_from`; cast timestamps to the same type |
# | `merge_asof` raises `left keys must be sorted` | merge_asof needs both frames sorted by the `on` column | `sort_values(ts)` on both, then restore the original order |
# | `MergeError: incompatible merge keys ... datetime64[us] and datetime64[ns]` | pandas 3 keeps different time resolutions per column | cast both keys with `.astype("datetime64[ns]")` before the join |
# | Dedup leaves duplicates | the dedup key includes a column that differs between deliveries (`ingest_ts`) | partition by the event's identity only (`request_id, position` / `event_id`) |
# | The skewed join runs for minutes or exhausts memory | you materialised the `a × b` rows (e.g. `.df()` without aggregation) | aggregate the many-rows side first; never fetch the raw join |
# | Reservoir sample changes between runs | global `random` or no seed | use `random.Random(seed)`; never re-seed inside the loop |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Walk through what happens, physically, when Spark runs `SELECT item_id, count(*) FROM logs GROUP BY item_id` on 1 TB. Where is the cost, and what does a combiner change?
# 2. One task of your job runs for two hours while the other 999 finished in five minutes. What is the likely cause, how do you confirm it, and what are three ways to fix it?
# 3. Broadcast, shuffle-hash or sort-merge: which join would you expect for impressions ⋈ items, impressions ⋈ clicks, and users ⋈ user_embeddings (both 1B rows)? Why?
# 4. Why does a model trained with "current price" look great offline and disappoint online? How does an as-of join fix it, and when should the inequality be strict?
# 5. Your click count jumped 3% overnight with no product change. How do you check for duplicate delivery, and how do you deduplicate correctly and deterministically?
# 6. When is HyperLogLog acceptable and when is it not? Why are HLL sketches mergeable, and what does that let you build?
#
# **Next chapter:** [`03_feature_pipelines_and_feature_stores.ipynb`](../ch21-feature-pipelines-and-feature-stores/notebook.ipynb) — use these joins to build features that are correct at prediction time, offline and online.
