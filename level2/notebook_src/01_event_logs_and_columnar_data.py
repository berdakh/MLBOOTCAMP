# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/01_event_logs_and_columnar_data.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 01 · Event logs & columnar data
#
# **Day 1 · ~90 minutes · CPU only**
#
# Every ranking model at a large company is trained on *logs*: one row per
# item shown, written by a serving system that was not designed with your
# model in mind. Whether a model can be trained, debugged and evaluated
# honestly is mostly decided before any modelling, by what the log records,
# how it is stored and how you sample it.
#
# By the end you will be able to:
#
# 1. say what a good impression/click log row must contain, and why some things (like "the user looked at it") can never be logged;
# 2. measure CSV against Parquet (size, read time, column projection) and read a Parquet file's row-group statistics;
# 3. use day partitioning, partition pruning and predicate pushdown in DuckDB, and prove they happened with `EXPLAIN ANALYZE`;
# 4. aggregate a dataset larger than your memory budget out of core, with DuckDB's `memory_limit`;
# 5. sample a log without biasing the metric you care about, reproducibly, with a stable hash.
#
# **You will produce:** the ShopStream `small` dataset, a day-partitioned copy of it, and the first week of the `large` scale, all under `artifacts/level2/`.

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
# --- Catch up: this is the first Level 2 notebook; it builds the data -------
import time

import duckdb
import numpy as np
import pandas as pd

from shoprank import scale, store
from shoprank.grading import nb01 as grade

L2 = store.level2_dir()
t0 = time.perf_counter()
world, small_dir = store.ensure_dataset("small")
print(f"small dataset ready in {time.perf_counter() - t0:.1f} s")

# %% [markdown]
# **Expected:** `generating ShopStream (small) -> .../level2/shopstream_small ...`
# the first time (about 10 s on our 4-core test machine, 30-60 s on a free
# Colab CPU), then `using cached ...` on re-runs.

# %% [markdown]
# ## 1.1 · What a good log row contains
#
# One impression row should let you answer, months later: *who saw what,
# where, when, chosen by which system, with what probability, and what
# happened next?* The fields that matter:
#
# | Field | Why you need it | ShopStream column |
# |---|---|---|
# | **event id** (unique per impression) | deduplication; joining clicks to the impression they came from | derive `request_id * 10 + position` |
# | **request id** | groups the 10 items shown together: every per-list metric (NDCG, "did the user click anything") needs it | `request_id` |
# | **entity ids** | user, item (and session, device, surface) | `user_id`, `item_id` |
# | **event time** | when it happened on the device, in UTC; the time your features must be "as of" | `ts` |
# | **position** | the single strongest driver of clicks (Day 3) | `position` |
# | **policy / model version** | which ranker produced the list; without it you cannot compare models in the log | (one policy here) |
# | **logging propensity** | P(this item shown at this position by the policy); needed for unbiased off-policy evaluation | `is_random` marks uniform exploration traffic; propensities of the main policy are *not* logged, a common real-world gap |
# | **outcomes** | click, purchase, dwell…, each with its own timestamp | `clicked`, `purchased` |
#
# What can never be in a real log: whether the user **examined** the item.
# A missing click means "not examined" *or* "examined and not interesting",
# and the log cannot tell which. ShopStream's generator knows (it decided),
# so the raw files contain a hidden `examined` column; `store.load` drops it,
# and so should you. Using it would be training on information no production
# system has.

# %%
raw = pd.read_parquet(sorted((small_dir / "impressions").glob("*.parquet"))[0])
print(raw.dtypes.to_string())
raw.head(3)

# %%
imps = store.load("small")[3]  # world, users, items, impressions; `examined` dropped
print(f"{len(imps):,} impressions | {imps.request_id.nunique():,} requests | "
      f"{imps.user_id.nunique():,} users | days {imps.day.min()}-{imps.day.max()}")
print("hidden column present after store.load?", "examined" in imps.columns)
print(f"CTR policy traffic {imps.loc[~imps.is_random, 'clicked'].mean():.3f} | "
      f"random traffic {imps.loc[imps.is_random, 'clicked'].mean():.3f}")

# %% [markdown]
# **Expected:** about 2.45 million impressions, about 245,000 requests from
# about 9,900 users over days 0-55; `False`; CTR about 0.14 on policy
# traffic and about 0.06 on random (exploration) traffic. The random traffic
# shows items in random order, so it shows worse items: that gap is the
# policy doing its job, not a logging bug.

# %% [markdown]
# ## 1.2 · Two scales: `small` and (part of) `large`
#
# `small` is 2.4M rows and fits comfortably in pandas. The `large` preset
# (200,000 users, 20,000 items) is about **46 million** impressions over 56
# days. We generate its first `LARGE_DAYS` days, written the way a warehouse
# stores logs: Parquet files partitioned by day. The generator emits requests
# in day order, so stopping early gives exactly the first days of the full
# world. Then we extrapolate how long the whole thing would take.

# %%
from shoprank.data import K, WorldConfig, make_world

LARGE_DAYS = 7  # None = all 56 days (~46M impressions; several minutes and ~1 GB)
large_cfg = WorldConfig.scale("large")
large_world = make_world(large_cfg)
large_dir = L2 / "shopstream_large"
if not (large_dir / "_SUCCESS").exists():
    info = scale.write_days_partitioned(large_world, large_dir, n_days=LARGE_DAYS)
    print(f"wrote {info['rows']:,} rows in {info['files']} files, {info['seconds']:.1f} s")
    expected_total = large_cfg.requests_per_user_day * large_world.users["activity"].sum() * large_cfg.n_days * K
    est = info["seconds"] / info["rows"] * expected_total
    print(f"full large ≈ {expected_total / 1e6:.0f}M rows -> about {est / 60:.1f} min on this machine")
else:
    print("using cached", large_dir)
print(f"large on disk: {scale.dir_size_mb(large_dir / 'impressions'):.0f} MB")
print(sorted(p.name for p in (large_dir / "impressions").iterdir()))

# %% [markdown]
# **Expected:** about 4.7 million rows for 7 days, written in about 30 s
# on our 4-core test machine (Colab is often 1.5-2× slower), and an
# extrapolated full `large` of about 46M rows taking about 5 minutes here,
# more on Colab. That is past this notebook's time budget, which is why we
# stop at a week. On disk: about 50 MB, in folders `day=0` … `day=6`.
#
# Notice the folder names: `day=3`. This *Hive-style* layout puts a column's
# value in the path, so a query engine can decide which files to open from
# the directory listing alone.

# %% [markdown]
# ## 1.3 · CSV vs Parquet
#
# CSV is text, one row after another: every read parses every character of
# every column. Parquet is **columnar** (each column stored contiguously, so
# you can read only the columns you need), **typed and compressed**
# (similar values next to each other compress well), and split into **row
# groups** that carry min/max statistics per column. Measure it.

# %%
FLAT = L2 / "formats"
FLAT.mkdir(exist_ok=True)
small_glob = str(small_dir / "impressions" / "*.parquet")
con = duckdb.connect()
con.execute("SET threads = 2")
select_logged = f"SELECT * EXCLUDE (examined) FROM read_parquet('{small_glob}') ORDER BY request_id, position"
con.execute(f"COPY ({select_logged}) TO '{FLAT / 'imps.csv'}' (HEADER)")
con.execute(f"COPY ({select_logged}) TO '{FLAT / 'imps.parquet'}' (FORMAT parquet)")

t = {}
with scale.timed("pandas read_csv (all columns)", t):
    df_csv = pd.read_csv(FLAT / "imps.csv")
with scale.timed("pandas read_parquet (all columns)", t):
    df_pq = pd.read_parquet(FLAT / "imps.parquet")
with scale.timed("pandas read_parquet (2 columns)", t):
    df_2 = pd.read_parquet(FLAT / "imps.parquet", columns=["position", "clicked"])
sizes = pd.Series({"csv": scale.dir_size_mb(FLAT / "imps.csv"),
                   "parquet": scale.dir_size_mb(FLAT / "imps.parquet")}).round(1)
print("size on disk (MB):", sizes.to_dict())
print("dtypes after read_csv:", dict(df_csv.dtypes.astype(str)))
del df_csv, df_pq, df_2  # free memory before the larger steps

# %% [markdown]
# **Expected:** the CSV is about 5-6× larger than the Parquet file
# (about 165 MB vs 25 MB), `read_csv` is roughly 10× slower than
# `read_parquet` (about 2.8 s vs 0.25 s here), and reading two columns is
# faster again (about 0.05 s). And `read_csv`
# silently returns `ts` as a plain `object` (string) column: CSV has no
# types, so every reader re-guesses them. With Parquet, the schema travels
# with the data.
#
# **Row groups and statistics.** The file was written sorted by
# `request_id`. Each row group stores min/max per column, so a filter like
# `request_id BETWEEN 100000 AND 100500` can skip every row group whose
# range does not overlap, without decompressing it. A filter on a column the
# file is *not* sorted by (`user_id`) cannot skip anything: every row group
# contains almost every user.

# %%
import pyarrow.parquet as pq

meta = pq.ParquetFile(FLAT / "imps.parquet").metadata
cols = {meta.schema.column(i).name: i for i in range(meta.num_columns)}


def row_group_ranges(column):
    rgs = [meta.row_group(r).column(cols[column]).statistics for r in range(meta.num_row_groups)]
    return [(s.min, s.max) for s in rgs]


rid = row_group_ranges("request_id")
uid = row_group_ranges("user_id")
print(f"{meta.num_row_groups} row groups of ~{meta.row_group(0).num_rows:,} rows")
print("request_id ranges, first 3:", rid[:3])
print("row groups that may contain request_id in [100000, 100500]:",
      sum(lo <= 100_500 and hi >= 100_000 for lo, hi in rid))
print("row groups that may contain user_id == 4242:", sum(lo <= 4242 <= hi for lo, hi in uid))

# %% [markdown]
# **Expected:** about 20 row groups of 122,880 rows (DuckDB's default);
# exactly 1 row group can contain the `request_id` range, while all of them
# may contain `user_id == 4242`. **Sort (or cluster) your files by the
# column you filter on most**, and the statistics do the skipping for you.

# %% [markdown]
# ## 1.4 · Partition pruning and predicate pushdown
#
# Row-group skipping still opens every file to read its footer. **Partition
# pruning** is one level higher: with a `day=` directory per day, a filter on
# `day` removes whole files from the scan *before any is opened*. Write a
# day-partitioned copy of `small` (DuckDB does it in one statement) and ask
# DuckDB what it actually scanned.

# %%
PART = L2 / "shopstream_small_by_day" / "impressions"
if not PART.exists():
    PART.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * EXCLUDE (examined) FROM read_parquet('{small_glob}')) "
                f"TO '{PART}' (FORMAT parquet, PARTITION_BY (day))")
part_glob = f"read_parquet('{PART}/*/*.parquet', hive_partitioning = true)"
print(len(list(PART.iterdir())), "partitions, e.g.", sorted(p.name for p in PART.iterdir())[:3])

query = f"""
SELECT position, count(*) AS n, avg(clicked::INT) AS ctr
FROM {part_glob}
WHERE day BETWEEN 42 AND 48
GROUP BY position ORDER BY position
"""
plan = con.execute("EXPLAIN ANALYZE " + query).fetchall()[0][1]
print("\n".join(line for line in plan.splitlines() if any(
    w in line for w in ("File Filters", "day BETWEEN", "Scanning Files", "Projections", "Total Time"))))

# %% [markdown]
# **Expected:** 56 partitions (`day=0`, `day=1`, …); in the plan,
# `File Filters: (day BETWEEN 42 AND 48)` and **`Scanning Files: 7/56`**:
# only one week's files were opened. `Projections` lists only the columns the
# query uses: column projection, pushed into the scan.
#
# How much does that save? Time the same aggregate over one week with
# pruning, and with a filter DuckDB cannot push into the file list (the day is
# computed from the timestamp, so it is only known after reading each row).

# %%
week_pruned = query
week_unpruned = query.replace(
    "WHERE day BETWEEN 42 AND 48",
    "WHERE date_diff('day', TIMESTAMP '2025-03-03', ts) BETWEEN 42 AND 48")
for name, q in [("partition filter", week_pruned), ("filter on derived ts", week_unpruned)]:
    con.execute(q).fetchall()  # warm the file cache, so we time the scan, not the disk
    start = time.perf_counter()
    for _ in range(5):
        res = con.execute(q).df()
    files = [ln for ln in con.execute("EXPLAIN ANALYZE " + q).fetchall()[0][1].splitlines() if "Scanning Files" in ln]
    scanned = files[0].strip(" │") if files else "no file filter: every file scanned"
    print(f"{name:22s} {(time.perf_counter() - start) / 5 * 1000:6.1f} ms  {scanned}"
          f"  pos0 CTR {res.ctr.iloc[0]:.3f}")

# %% [markdown]
# **Expected:** identical results (position-0 CTR about 0.45), but the
# derived-timestamp filter scans all 56 files and takes roughly 10× longer
# (about 10 ms vs 130 ms here). On a laptop that is milliseconds; on a 10 TB table it is the
# difference between a one-minute query and a one-hour, expensive one. Real
# warehouses (BigQuery, Snowflake, Spark/Hive tables) apply the same rule:
# **filter on the partition column itself**, not on an expression of
# another column.

# %% [markdown]
# ## 1.5 · Out of core: aggregating more data than your memory budget
#
# Pandas needs the whole table in RAM, plus copies during a `groupby`.
# DuckDB streams through the files and, when its `memory_limit` is reached,
# spills intermediate state to disk. Compare what pandas needs for the
# week of `large` with what DuckDB is allowed to use.

# %%
large_glob = f"read_parquet('{large_dir}/impressions/*/*.parquet', hive_partitioning = true)"
with scale.timed("pandas: load the large week"):
    big = pd.read_parquet(large_dir / "impressions")
print(f"pandas frame: {len(big):,} rows, {big.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory")
del big

ooc = duckdb.connect()
ooc.execute("SET memory_limit = '64MB'")
ooc.execute("SET threads = 2")
ooc.execute(f"SET temp_directory = '{L2 / 'duckdb_tmp'}'")
per_user_sql = f"""
SELECT user_id, count(*) AS impressions, sum(clicked::INT) AS clicks,
       approx_count_distinct(item_id) AS distinct_items_approx
FROM {large_glob} GROUP BY user_id
"""
with scale.timed("duckdb (memory_limit=64MB): per-user aggregate"):
    per_user = ooc.execute(per_user_sql).df()
print(f"{len(per_user):,} users; median impressions {per_user.impressions.median():.0f}, "
      f"median distinct items ≈ {per_user.distinct_items_approx.median():.0f}")

exact_sql = per_user_sql.replace("approx_count_distinct(item_id)", "count(DISTINCT item_id)")
try:
    ooc.execute(exact_sql).df()
    print("exact count(DISTINCT) also fit in 64MB with this DuckDB version")
except duckdb.Error as exc:
    print("exact count(DISTINCT) at 64MB ->", str(exc).splitlines()[0][:90])
ooc.close()

# %% [markdown]
# **Expected:** the pandas frame takes about 155 MB of RAM (and
# `read_parquet` needs more while building it). DuckDB computes per-user
# counts and click sums for the ~130,000 users active that week inside a **64 MB** limit in well under a
# second: a `GROUP BY` with `count`/`sum` keeps only a few numbers per group,
# and DuckDB streams the files through it. With `LARGE_DAYS = None` pandas
# would need well over 1 GB just to hold the table; the DuckDB query is
# unchanged.
#
# The exact `count(DISTINCT item_id)` is different. In our run (DuckDB 1.2.2)
# it failed with `Out of Memory Error` at 64 MB (it needed about 250 MB);
# other versions may spill it to disk and succeed slowly. Distinct counts are
# *holistic*: the state per group grows with the number of distinct values,
# not a fixed few bytes. `approx_count_distinct` (HyperLogLog) keeps a
# fixed-size sketch per group instead. Notebook 02 measures what that costs in
# accuracy. In production the same reasoning scales out (Spark, BigQuery):
# you write the aggregate, the engine streams, shuffles and spills, and
# holistic aggregates are the ones that hurt.

# %% [markdown]
# ## 1.6 · Sampling without bias
#
# You rarely analyse a full log; you sample. **What you sample decides which
# metrics stay unbiased.** Three 10% samples of `small`:
#
# * **rows**: keep each impression independently;
# * **requests**: keep all 10 rows of 10% of requests;
# * **users**: keep every row of 10% of users.
#
# Compare three metrics, each defined at a different level: impression CTR
# (per row), the share of requests with at least one click (per request), and
# the number of distinct items a user saw (per user).

# %%
def metrics(df: pd.DataFrame) -> dict:
    per_req = df.groupby("request_id")["clicked"].any()
    per_user = df.groupby("user_id")["item_id"].nunique()
    return {"rows": len(df), "impression CTR": df.clicked.mean(),
            "requests with a click": per_req.mean(), "distinct items / user": per_user.mean()}


rng = np.random.default_rng(0)
req_keep = rng.random(imps.request_id.max() + 1) < 0.10
user_keep = rng.random(imps.user_id.max() + 1) < 0.10
samples = {
    "full log": imps,
    "10% of rows": imps.sample(frac=0.10, random_state=0),
    "10% of requests": imps[req_keep[imps.request_id]],
    "10% of users": imps[user_keep[imps.user_id]],
}
pd.DataFrame({name: metrics(df) for name, df in samples.items()}).T.round(3)

# %% [markdown]
# **Expected (the pattern matters, not the third decimal):**
#
# * impression CTR ≈ 0.137 in **every** sample: a per-row metric survives any
#   random sample of rows;
# * "requests with a click" ≈ 0.76 in the full log and in the request and user
#   samples, but only about **0.20** in the row sample. Each surviving request
#   keeps about one of its 10 rows, so most of its clicks are thrown away.
#   NDCG, MRR, "did the user click anything": every per-request metric breaks
#   the same way;
# * distinct items per user ≈ 210 in the full log and the user sample, but
#   only about 25-30 in the row and request samples. A per-user metric needs
#   *all* of a user's rows: sample users.
#
# **Rule:** sample at the level of the unit your metric is computed over (or
# coarser). Requests for ranking metrics; users for anything per-user, and for
# A/B tests (Day 6), where the user is the randomisation unit.
#
# **Reproducibility.** `rng.random(...)` above gives a different sample if
# the user count changes, and it cannot be recomputed inside SQL or on
# another machine. The standard trick is a **hash of the key**:
# `keep = hash(salt + user_id) % 100 < pct`. Same user → same decision,
# every day, in every job. But *which* hash? Python's built-in `hash()` of a
# string is salted per process (a security feature, `PYTHONHASHSEED`):

# %%
cmd = [sys.executable, "-c", "print(hash('user_42'), hash(42))"]
for run in (1, 2):
    print(f"process {run}:", subprocess.run(cmd, capture_output=True, text=True).stdout.strip())

# %% [markdown]
# **Expected:** the string hash differs between the two processes (the
# integer hash is just `42` both times, which is not a hash you want either:
# `user_id % 100` keeps users 0-9, 100-109, …, and ids are often assigned in
# signup order). Use a cryptographic hash from `hashlib`: its output is
# defined by a standard, identical in every process, language and machine.

# %% [markdown]
# > **Exercise 1.** Write `user_hash_sample(user_ids, pct, salt="")` that
# > returns a boolean NumPy array, `True` where the user is in the sample.
# > Use exactly this rule, so your sample matches the one in the grader (and
# > in any other job that uses the rule):
# >
# > ```
# > h = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:16], 16)
# > keep = h % 100 < pct
# > ```
# >
# > The grader checks the exact decisions on 20,000 ids, determinism, that the
# > same user always gets the same decision, that a 5% sample is contained in
# > the 10% sample, and that a different salt gives a different sample.
# > (Why would you *want* the samples of two experiments to use different
# > salts?)

# %%
def user_hash_sample(user_ids, pct: int, salt: str = "") -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(user_hash_sample)

# %% [markdown]
# The package version of the same rule is `shoprank.scale.stable_bucket`
# (read [`src/shoprank/scale.py`](../../src/shoprank/scale.py)). Using it, a
# hash-based 10% user sample gives the same per-user metric as the random
# 10% user sample, and the same users every time it is run:

# %%
uids = np.arange(imps.user_id.max() + 1)
in_sample = np.array([scale.stable_bucket(u, 100, salt="nb01") < 10 for u in uids])
hashed = imps[in_sample[imps.user_id]]
print(f"{in_sample.sum():,} users sampled ({in_sample.mean():.1%});",
      f"distinct items / user {hashed.groupby('user_id').item_id.nunique().mean():.0f}")

# %% [markdown]
# **Expected:** about 1,030 users (10.3%) and about 209 distinct items per
# user, matching the full log. These numbers are identical on every run and
# every machine: that is the point.
#
# > **Exercise 2.** Write `daily_ctr_by_position_sql(impressions_dir)` that
# > returns a DuckDB SQL **string** computing, over the day-partitioned
# > layout (`<impressions_dir>/day=D/*.parquet`), for **days 7-13 only**:
# > `day, position, impressions, clicks, ctr` (one row per day and position,
# > `ctr = clicks / impressions` as a float), ordered by `day, position`.
# >
# > The grader runs your query on a small dataset in which **every file
# > outside days 7-13 is corrupt** (except day 0, which is valid but must not
# > appear in the result). A query that prunes partitions never opens them;
# > one that reads everything and filters later crashes. Then it compares
# > your numbers with a reference.

# %%
def daily_ctr_by_position_sql(impressions_dir: str) -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(daily_ctr_by_position_sql)

# %% [markdown]
# > **Exercise 3 (written).** A product team launches a "similar items"
# > carousel on the item page. You are asked to specify its impression log
# > before launch. List the fields you require (with types), the partitioning
# > and sort order of the table, and how analysts should sample it to compute
# > (a) carousel CTR by slot and (b) the share of sessions in which a user
# > clicked anything in the carousel. Name one field that, if forgotten now,
# > makes an offline evaluation of a *new* carousel model impossible later.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `Binder Error: Cannot compare values of type VARCHAR and INTEGER_LITERAL` on `day` | partition folders written zero-padded (`day=07`), so DuckDB infers `day` as text | write unpadded folders, cast (`CAST(day AS INT)`, which still prunes), or pass `hive_types = {'day': INTEGER}` |
# | `Scanning Files: 56/56` although you filter by day | the filter is on an expression of another column, or `hive_partitioning` is off | filter on the partition column itself; pass `hive_partitioning = true` |
# | `Out of Memory Error` in DuckDB | `memory_limit` very low and no `temp_directory` to spill to | set `temp_directory`; raise the limit; select fewer columns |
# | Kernel dies loading `large` into pandas | the frame (plus copies) exceeds RAM | aggregate in DuckDB; read only the columns you need; lower `LARGE_DAYS` |
# | Large generation takes many minutes | `LARGE_DAYS = None` writes all 46M rows | keep `LARGE_DAYS = 7`; delete `artifacts/level2/shopstream_large` to regenerate |
# | Hash sample differs between runs | Python's `hash()` on strings, or `random` without a seed | `hashlib.sha256` of `f"{salt}:{id}"`, as in Exercise 1 |
# | A cached dataset looks wrong after editing code | `_SUCCESS` marker present, so nothing is regenerated | delete the dataset folder under `artifacts/level2/` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Design the impression log for a home feed. Which fields are non-negotiable, and which one do teams most often forget that later blocks off-policy evaluation?
# 2. Why is Parquet faster than CSV for analytics? Explain columnar layout, compression, row-group statistics and schema, and say when CSV is still fine.
# 3. What is the difference between partition pruning and predicate pushdown? How would you prove that a query used them, and how do you choose a partition key (and what goes wrong with too fine a key)?
# 4. You must compute per-user 30-day engagement over 5 TB of logs on a machine with 16 GB RAM. What do you do, and why does an out-of-core engine succeed where pandas fails?
# 5. A colleague samples 1% of impression rows and reports that NDCG@10 dropped. What is wrong, and how should they sample?
# 6. Why `hash(salt + id) % 100 < k` rather than `random() < k/100`, why a salt, and why not Python's `hash()`?
#
# ➡️ **Next:** [`02_aggregation_and_joins_at_scale.ipynb`](02_aggregation_and_joins_at_scale.ipynb) — what a distributed engine does with your `GROUP BY` and `JOIN`, why skew hurts, and when an approximate answer is the right one.
