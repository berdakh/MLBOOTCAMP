# Solutions · Chapter 20 · Aggregation & joins at scale

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `reservoir_sample`

[verified] Reference code: [`code/nb02.py`](solutions.py) (`ex1`), graded in CI.

```python
def reservoir_sample(stream, k, seed):
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)      # inclusive: i + 1 choices
            if j < k:
                reservoir[j] = item
    return reservoir
```

**Why it works (the interview proof).** Claim: after seeing `n >= k` items,
each is in the reservoir with probability `k/n`. True for `n = k`. Item
`n+1` enters with probability `k/(n+1)`. An item already inside stays unless
the new item enters *and* lands on its slot:
`k/n · (1 − k/(n+1) · 1/k) = k/n · n/(n+1) = k/(n+1)`. Induction done.

**Common mistakes.**

* `rng.randint(0, i - 1)` or `rng.randrange(i)`: off by one. The `i`-th
  item (0-indexed) is the `(i+1)`-th seen, so there are `i + 1` choices.
* Replacing with probability `1/i` and a random slot: early items are kept
  far too often. The grader's chi-square test shows it (first items ~1,500
  times, last ~100, where 500 is expected).
* `random.sample(list(stream), k)`: uniform, but it stores the whole stream,
  which defeats the purpose (memory `O(n)`, not `O(k)`).

**Push further.** Weighted reservoir sampling (Efraimidis–Spirakis: keep the
`k` items with the largest `u ** (1/w)`) and distributed reservoirs (sample
`k` per mapper with a random key, then keep the `k` smallest keys overall).

---

## Exercise 2 · `partitioned_count`

[verified] Reference code: [`code/nb02.py`](solutions.py) (`ex2`).

```python
def partitioned_count(keys, n_partitions):
    keys = list(keys)
    chunk = -(-len(keys) // 4) if keys else 1
    partitions = [[] for _ in range(n_partitions)]
    for lo in range(0, len(keys), chunk):                       # map + combine
        for key, c in Counter(keys[lo:lo + chunk]).items():
            partitions[stable_bucket(key, n_partitions)].append((key, c))   # shuffle
    result = {}
    for inbox in partitions:                                    # reduce
        merged = Counter()
        for key, c in inbox:
            merged[key] += c
        result.update(merged)
    return result
```

**Why it works.** Every occurrence of a key is counted in exactly one
mapper's combiner, and all partial counts for that key go to the same
partition (the hash depends only on the key), so each partition's sums are
complete and partitions hold disjoint keys. With the combiner, the skewed
case (`7` appearing 5,000 times) sends 4 rows for key 7, not 5,000.

**Common mistakes.** Routing by `hash(key)`: it still works within one
process, but the partition of a string key changes between runs, so a re-run
of a failed reducer could receive different keys. Using `len(keys) // 4` as
chunk size: for fewer than 4 keys it is 0, and `range(0, n, 0)` raises.

---

## Exercise 3 · Distinct users per item per day on 10B rows (written)

[open-ended] A strong answer:

* **Approximate, with HLL sketches.** Exact distinct counts per item-day need
  state proportional to the number of distinct (item, day, user) triples:
  billions. A dashboard tolerates ~1% error; choose a sketch precision that
  gives it (BigQuery `HLL_COUNT.INIT(user_id, 14)`, Spark/Datasketches HLL
  with `lgK = 12-14`), and show the error band on the chart.
* **Store one sketch per (item, day)** in a daily table built incrementally
  from that day's partition only. 7- and 30-day uniques are
  `HLL_COUNT.MERGE` over the last 7/30 daily sketches: no rescan of the log,
  and a late correction to one day means rebuilding one day's sketches. (Sum
  of daily distinct counts would double-count users who return; that is the
  whole reason for sketches that merge.)
* **Skew.** The hot items are fine for an *aggregation* with sketches: each
  mapper builds a partial sketch per item (combiner), so the shuffle carries
  one fixed-size sketch per item per mapper, not 5% of all rows. If computing
  exactly, salt the hot items (`(item, hash(user) % S)`, distinct per sub-key,
  sum over salts, exact because users are disjoint across salts), or let Spark
  AQE split skewed partitions.
* **Validation.** Compute exact counts for a stratified sample of items
  (head, torso, tail) for one day and compare (relative error should match the
  sketch's documented standard error); check monotonicity (30-day ≥ 7-day ≥
  1-day); reconcile totals with an independent source (e.g. the sessions
  table); alert on day-over-day jumps (duplicates, logging changes).
* **Cost.** Partition the log by day and read only `item_id, user_id` from
  Parquet: one day of a 10B-row year is ~27M rows, small.

---

## Checkpoint answers

1. **GROUP BY on 1 TB.** [verified] Mappers read input splits (Parquet row groups),
   project `item_id`, and emit `(item_id, 1)`; with a combiner (partial
   aggregation) each mapper emits one `(item_id, partial_count)` per item it
   saw. Rows are hash-partitioned by `item_id`, written to local disk,
   fetched over the network by reducers (the shuffle), merged and summed. The
   cost is the shuffle: serialisation, disk, network, sort. In our run the
   combiner cut shuffled rows from 2,446,900 to about 15,000.
2. **The straggler.** [verified] Skew: one partition holds a hot key (a top seller, a
   `NULL`/default value, a bot user). Confirm in the Spark UI (task duration and
   shuffle-read size distribution per stage; one task far above the median)
   and with a `GROUP BY key ORDER BY count DESC LIMIT 20` on a sample. Fixes:
   filter junk keys (`NULL`, `-1`) early; salt the hot keys (two-stage
   aggregation: in our run the max/mean reducer load fell from 4.3 to 1.4);
   broadcast the small side of a join; pre-aggregate before joining; enable
   AQE skew-join handling; increase partitions only helps if the skew is
   *many* moderately hot keys, not one.
3. **Join choice.** [open-ended] impressions ⋈ items: broadcast (items is small).
   impressions ⋈ clicks: both large, joined on impression id: shuffle-hash or
   sort-merge; better, bucket/partition both by date and id so the join is
   co-partitioned (no shuffle). users ⋈ user_embeddings (1B each): sort-merge
   (Spark's default for large–large; robust, spills); if both tables are
   stored bucketed by user id with the same bucket count, the shuffle is
   avoided.
4. **Current price.** [verified] "Current price" is the price at export time: for an
   impression last month it is a future value, partly caused by the outcomes
   (prices are cut for items that do not sell), so offline it leaks and online
   the model sees a different distribution. In our run 55% of impressions had a
   price different from the item's latest price. An as-of join takes the
   latest value with `valid_from <= event_ts`. Use strict `<` when the right
   side is derived from data that includes the event itself (e.g. a daily
   aggregate stamped at the event's time), so an event never sees its own
   outcome.
5. **Duplicates.** [verified] Compare `count(*)` with `count(DISTINCT event_id)` per
   day/hour; check the gap against the ingest-time distribution (retries show
   up as rows with the same event id and later `ingest_ts`), and against client
   release or pipeline incidents. Deduplicate with
   `ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY ingest_ts) = 1`: event
   identity only, deterministic tie-break, over a window covering the maximum
   redelivery delay. In our run, 2% injected duplicates were removed exactly.
6. **HLL.** [verified] Acceptable for dashboards, monitoring, capacity planning and
   exploratory analysis where a known relative error is fine; not for billing,
   payments, legal or experiment metrics whose differences are as small as the
   sketch error (DuckDB's default sketch was 6.5% off on 2M distinct pairs in
   our run; sketch size sets the error). HLL registers keep the maximum of a
   hash statistic per bucket, and `max` is associative and commutative, so two
   sketches merge into the sketch of the union. That gives incremental daily
   sketches merged into any date range, combiners in the map step, and
   distributed distinct counts without a full shuffle.
