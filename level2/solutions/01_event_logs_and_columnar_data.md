# Solutions · 01 · Event logs & columnar data

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 01-1 · `user_hash_sample`

✅ Reference code: [`code/nb01.py`](code/nb01.py) (`ex1`), graded in CI.

```python
def user_hash_sample(user_ids, pct, salt=""):
    def bucket(user_id):
        digest = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
        return int(digest[:16], 16) % 100
    return np.array([bucket(u) < pct for u in user_ids], dtype=bool)
```

**Why it works.** SHA-256 is a fixed, standardised function: the same bytes
give the same digest in every process, language and machine, so the sample
can be recomputed in a Spark job, in SQL, or next year. Taking the first 16
hex digits (64 bits) before `% 100` keeps the bucket distribution uniform
(the modulo bias of 2^64 mod 100 is negligible). Because the rule is
`bucket < pct`, a 5% sample is always inside the 10% sample: you can grow a
sample without re-drawing it.

**Why salts.** Without a salt, every analysis and every experiment that takes
"bucket < 10" gets the *same* users. Two experiments would then always share
their treatment group (their effects get confounded), and one unlucky bucket
of heavy users biases every study. A salt per experiment (or per analysis)
makes the samples independent; the grader checks that the overlap between
two salts' 10% samples is about 10%, not 100%.

**Common mistakes.**

* `hash(user_id)`: for strings it changes between processes
  (`PYTHONHASHSEED`); for integers it is the identity, so `% 100` keeps ids
  ending in 00-09: often a signup cohort, not a random sample.
* `random.random() < pct / 100` with a seed: reproducible only while the list
  of users and its order stay identical.
* Hashing the user id *without* a separator: `salt="exp1"` and `user_id=23`
  give `"exp123"`, the same as `salt="exp12"`, `user_id=3`. The `:` avoids it.

**Push further.** In SQL, DuckDB has `hash()` (stable within a version, not
promised across versions) and `md5_number()`; BigQuery has
`FARM_FINGERPRINT`. Pick one, write it into your team's sampling convention,
and never mix two.

---

## 01-2 · Daily CTR by position over the partitioned log

✅ Reference code: [`code/nb01.py`](code/nb01.py) (`ex2`).

```sql
SELECT day, position,
       count(*)             AS impressions,
       sum(clicked::INT)    AS clicks,
       avg(clicked::DOUBLE) AS ctr
FROM read_parquet('<impressions_dir>/*/*.parquet', hive_partitioning = true)
WHERE day BETWEEN 7 AND 13
GROUP BY day, position
ORDER BY day, position
```

**Why it works.** `hive_partitioning = true` turns the folder name `day=7`
into a column. A filter on that column is applied to the *file list* before
any file is opened (`File Filters` / `Scanning Files: 7/N` in
`EXPLAIN ANALYZE`), so the corrupt files in the grader's dataset are never
read.

**Common mistakes.**

* Forgetting `hive_partitioning`: there is no `day` column (the files do not
  contain it), so the query fails to bind.
* Filtering on something derived from another column, such as
  `date_diff('day', start, ts)`: the engine must read every row to evaluate
  it, so it opens every file (and crashes on the corrupt ones).
* `sum(clicked) / count(*)` with integer types gives integer division in some
  engines. Cast to `DOUBLE` or use `avg(clicked::DOUBLE)`.

**Push further.** DuckDB's optimiser is good at pushing filters down: in our
tests a `HAVING day BETWEEN 7 AND 13`, or a filter in an outer query over a
`GROUP BY` subquery, were also pushed into the file list. Do not rely on it
across engines: put partition filters in the `WHERE` of the scan.

---

## 01-3 · Specifying a carousel's impression log (written)

✍️ A strong answer contains most of these.

**Fields** (one row per item shown):

| Field | Type | Why |
|---|---|---|
| `impression_id` | string/uuid | unique; dedup; click join key |
| `request_id` | string | the carousel load; groups the slots shown together |
| `session_id`, `user_id` (or anonymous id), `device`, `country` | string | units for per-session/per-user metrics, segments |
| `anchor_item_id` | int | the item page the carousel sits on (the "query") |
| `item_id`, `slot` (position) | int | what and where |
| `event_ts` (UTC, device), `server_ts`, `ingest_ts` | timestamp | event time for features; lateness and dedup diagnostics |
| `model_version`, `experiment_id`, `variant` | string | which ranker produced the list; experiment analysis |
| `score`, **`propensity`** (P(item in this slot under the policy)) | float | off-policy evaluation, debugging |
| `is_exploration` | bool | marks randomised traffic |
| `visible` / `viewport_ms` | bool / int | was the slot rendered on screen (not "examined", but the closest proxy) |
| outcome events (`click`, `add_to_cart`, `purchase`) | separate table keyed by `impression_id` with own timestamps | outcomes arrive later, and from other systems |

**Layout.** Partition by event date (`dt=YYYY-MM-DD`), cluster/sort by
`request_id` (or `user_id` if per-user reads dominate). Daily partitions keep
pruning effective without creating millions of tiny files; an hourly
partition is fine for very high volume.

**Sampling.** (a) CTR by slot is a per-impression metric: any random sample
of rows or requests works, but sample **requests** anyway so the same sample
serves per-list metrics. (b) "Share of sessions with a carousel click" is per
session: sample **sessions** (or users), never rows, and keep all of a
sampled session's rows. Use a salted stable hash of the unit id.

**The field you cannot add later:** the logging **propensity** (with
`model_version` and the exploration flag). Without it, inverse-propensity or
doubly-robust evaluation of a new carousel model on old logs is impossible,
and you are left with A/B tests for every idea. Second-best answer:
`request_id`, without which no list-level metric can be computed at all.

---

## Checkpoint · 01

1. **Feed impression log.** ✍️ Non-negotiable: unique impression id, request
   id, user/session ids, item id, position, event timestamp in UTC, model
   version and experiment variant, outcomes keyed to the impression id.
   Most often forgotten: the logging propensity (or enough to recompute it:
   policy scores and the randomisation used), plus an exploration flag. Without
   them, off-policy evaluation of a new ranker on old logs is not possible.
   Never loggable: whether the user examined the item; a non-click is
   ambiguous.
2. **Parquet vs CSV.** ✅ Columnar: a query reads only the columns it needs
   (2 of 10 columns read in ~0.05 s vs 0.25 s for all, in our run). Typed,
   compressed encodings (dictionary, run-length, delta) shrank the file ~6×
   (165 MB → 25 MB). Row-group min/max statistics let a filter skip row groups
   (1 of 20 for a `request_id` range). The schema travels with the data
   (`read_csv` returned `ts` as strings). CSV is still fine for tiny
   hand-edited files and for interchange with tools that read nothing else.
3. **Pruning vs pushdown.** ✅ Partition pruning removes whole files/folders
   using the partition column in the path, before opening anything (DuckDB:
   `File Filters`, `Scanning Files: 7/56`). Predicate pushdown pushes a filter
   into the reader, which skips row groups using footer statistics and avoids
   materialising rows that fail it. Prove it with the query plan (`EXPLAIN
   ANALYZE`, the BigQuery "bytes processed" estimate, the Spark UI's
   `PartitionFilters`/`PushedFilters`). Choose the key you filter on almost
   always (date for logs), with moderate cardinality. Too fine (user id,
   minute) creates millions of tiny files: listing and per-file overhead
   dominate, and the statistics become useless.
4. **5 TB on 16 GB.** 🔬 Use an engine that streams and spills (DuckDB on one
   machine for a few hundred GB; Spark/BigQuery for 5 TB). Read only the needed
   columns from Parquet, prune to the 30-day partitions, aggregate per user
   (fixed-size state per group: counts, sums), and write the small result.
   Pandas needs the whole table (plus copies) in RAM, so it is bounded by memory;
   a streaming hash aggregate is bounded by the *number of groups* and can
   spill partitions to disk. Watch for holistic aggregates (exact distinct
   counts, medians): in our run exact `count(DISTINCT)` failed at 64 MB where
   counts and sums succeeded. Use sketches (HLL) or two-stage distinct.
5. **1% of rows and NDCG.** ✅ Row sampling keeps ~0.1 of the 10 rows of a
   request, so a per-request metric is computed on lists that are almost
   empty: in our run "requests with a click" fell from 0.76 to 0.20 under a
   10% row sample. NDCG is just as broken. Sample whole requests (or users,
   if the analysis is per user or an A/B test), with a stable hash of the id.
6. **Why hashing.** ✅ A hash of the id is a pure function: the same user
   gets the same decision in every job, every day, in SQL or Python, with no
   state to store, and samples are nested (5% ⊂ 10%). `random()` changes with
   order, seed and population. A salt makes different studies' samples
   independent. Python's `hash()` is salted per process for strings (we saw
   two different values for `hash('user_42')`) and the identity for integers.
