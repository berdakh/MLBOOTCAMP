# Solutions · Chapter 35 · Serving a ranking cascade

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `LRUCache`

[verified] Reference code: [`code/nb17.py`](solutions.py) (`ex1`), graded in CI.

```python
class LRUCache:
    def __init__(self, capacity):
        self.capacity, self._data = capacity, OrderedDict()
        self.hits = self.misses = 0

    def get(self, key, default=None):
        if key in self._data:
            self._data.move_to_end(key)      # now most recent
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return default

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)   # evict least recent

    def __len__(self):
        return len(self._data)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
```

**Why it is O(1).** `OrderedDict` is a hash map plus a doubly linked list:
lookup, `move_to_end` and `popitem(last=False)` are all constant time. The
interview version without `OrderedDict` is exactly that: a `dict` from key
to node, and a doubly linked list with sentinel head/tail nodes.

**Common mistakes.** A `list` for recency order (`list.remove` is O(n): the
grader times it at 100 vs 20,000 entries and sees ~100× slowdown); counting
`put` as a miss; forgetting that `put` on an existing key must refresh its
recency. In a threaded server, add a lock (the module's `sv.LRUCache` does):
two threads interleaving `move_to_end` and `popitem` can corrupt the order.

---

## Exercise 2 · `apply_business_rules`

[verified] Reference code: [`code/nb17.py`](solutions.py) (`ex2`).

One pass in rank order with three pieces of state: `seen` (ids), `per_cat`
(counts), and `capped` (items skipped only because of the cap). Dedupe and
eligibility come first: a duplicate or unlaunched item must never be
backfilled. Stop as soon as `k` items are kept; if the pass ends short,
append `capped[:k - len(out)]`.

**Design notes.** Backfilling keeps the page full (an empty slot is worse
than a fourth item from one category), but put backfilled items *after* the
others so the cap still shapes the top of the list. Production systems use
softer versions: a penalty per repeated category (MMR-style re-ranking) rather
than a hard cap, or a cap only within the first screen.

---

## Exercise 3 · `with_fallback`

[verified] Reference code: [`code/nb17.py`](solutions.py) (`ex3`).

```python
_POOL = ThreadPoolExecutor(max_workers=8)

def with_fallback(primary_fn, fallback_fn, timeout_s):
    def wrapped(*args, **kwargs):
        future = _POOL.submit(primary_fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s), "ok"
        except TimeoutError:            # concurrent.futures.TimeoutError
            return fallback_fn(*args, **kwargs), "timeout"
        except Exception:
            return fallback_fn(*args, **kwargs), "error"
    return wrapped
```

**Points interviewers probe.**

* The pool is created **once**, not per call: creating threads per request
  costs more than the work, and a `with ThreadPoolExecutor()` block would
  *wait for the slow call on exit*, defeating the timeout.
* The timed-out call keeps running: Python cannot kill a thread. Under
  overload this wastes CPU exactly when you have none, so real services
  also bound the pool (reject when full: load shedding), make the work
  itself cancellable, or run models in a separate process/service with its
  own deadline.
* Return the *status*, so the caller can count fallbacks. A fallback nobody
  counts is a silent outage.
* With `asyncio`, the same pattern is `asyncio.wait_for(...)`; with gRPC,
  per-call deadlines propagate to downstream services.

---

## Exercise 4 · Latency budget for a 200 ms SLO (written)

[open-ended] One reasonable answer (numbers are p99 budgets; they intentionally sum to
less than 200 because percentiles do not add and you need slack):

| Stage | Budget (ms) | Why | Fallback when blown |
|---|---|---|---|
| Client ↔ edge network + TLS | 40 | mobile networks; outside our control, measured from client telemetry | none (measure, don't promise) |
| Gateway, auth, request parsing | 5 | cheap, in-process | — |
| User features (remote store) | 15 | one key, cached in-process for hot users | default user features |
| ANN retrieval, 3 shards in parallel | 25 | the slowest shard sets the tail; hedge requests after 15 ms | drop the slow shard's results; popular/cached candidates |
| Item features for ~500 candidates | 20 | batched multi-get; local cache for hot items | defaults per item (smoothed priors) |
| Ranker RPC to GPU pool | 50 | biggest share: this is where quality comes from; batching on the GPU side | pre-ranker / popularity order (timeout at 50 ms) |
| Rules, re-ranking, logging | 10 | in-process | skip optional re-rankers |
| Slack | 35 | GC pauses, retries, queueing at peak | — |

If a new model needs 40 ms more: first take it from slack only if the
current p99 has room (measure!); otherwise shrink ranker input (fewer
candidates via a better pre-ranker), distil or quantise the model
(notebook 18), or move feature fetch to overlap with retrieval (fetch user
features and start retrieval in parallel). Cutting the network budget is not
an option; cutting retrieval below ~20 ms costs recall.

---

## Checkpoint answers

1. **Cascade walkthrough.** Retrieval: whole catalogue (10⁶–10⁹) → ~1,000–2,000
   candidates from several sources (two-tower ANN, co-visitation, popularity,
   follows), ~20–30 ms in parallel. Optional pre-ranker: → ~300–500, ~10 ms.
   Feature fetch: user row + item rows, ~10–20 ms batched. Heavy ranker:
   multi-task model over ~500 rows, ~30–50 ms. Re-ranking/rules: diversity,
   filters, ads, ~5–10 ms. At 100M users the numbers scale through QPS
   (tens of thousands), not per-request work.
2. **Retrieval vs ranking bottleneck.** Score the candidates with an oracle
   (true relevance, or a strong offline model / human labels in real life).
   The gap from perfect to the oracle is retrieval loss; from the oracle to
   your ranker is ranking loss. In notebook 17 (no cap): oracle 0.99, ranker
   0.875 → ranking loss dominates; retrieval recall of the true top-10 was
   0.85. Online, also check whether the offline gain came from items the
   retrieval never surfaces in production (training/serving candidate
   mismatch) and from position bias in the offline metric.
3. **Graceful degradation.** Ranker timeout/error → candidates in a cheap
   order (pre-ranker or popularity); missing features → defaults (smoothed
   priors) and skip sources that need them; retrieval failure → a cached
   popular/trending list. Every fallback increments a counter exported to
   metrics, carried in a response header/log field, and alerted on as a rate
   (in 17.6: ranker raises → NDCG 0.38 vs 0.72 healthy, so the counter
   matters).
4. **Caching.** In-process LRU for user feature rows (hit rate bounded by
   repeat requests: 0.29 at most for one ShopStream day, 0.13 in the load
   test), for item features of hot items (high hit rate: popularity is
   skewed), for popular/trending lists (~1.0). Problems: staleness (TTL per
   feature family), memory per replica, inconsistency across replicas,
   thundering herd on expiry, and caching wrong values (an error response
   cached as "no features").
5. **p99 doubled, p50 flat.** Something affects only some requests: one
   slow dependency or shard (fan-out amplifies it), GC pauses, a cold cache
   after deploy, a code path for some users (long histories → more
   candidates), CPU throttling, thread-pool saturation/queueing. Find it with
   per-stage timings (Server-Timing / tracing) split by percentile, then by
   request attributes; compare canary vs baseline.
6. **Category cap.** It costs short-term relevance (17.5: 0.875 → 0.71 NDCG
   at cap 3) and buys diversity (1.3 → 4.2 categories per page). Measure its
   value online: A/B on long-term engagement, return rate, and purchases per
   user, not CTR on the session. It belongs in re-ranking (after the
   ranker), as a soft penalty if possible, and backfilling must never
   reintroduce filtered items.
