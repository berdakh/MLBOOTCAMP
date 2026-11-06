# Solutions · 04 · Streaming features

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 04-1 · `TumblingWindowCounter`

✅ Reference code: [`code/nb04.py`](code/nb04.py) (`ex1`), graded in CI.

```python
class TumblingWindowCounter:
    def __init__(self, window_seconds, allowed_lateness_seconds):
        self.window = float(window_seconds)
        self.lateness = float(allowed_lateness_seconds)
        self.watermark = -math.inf
        self.counts = {}            # (key, window_start) -> count, open windows only
        self.late_events = []

    def _close(self, upto):
        ready = sorted((ws, str(k), k) for (k, ws) in self.counts if ws + self.window <= upto)
        return [(k, ws, self.counts.pop((k, ws))) for ws, _, k in ready]

    def add(self, key, event_time):
        ws = math.floor(event_time / self.window) * self.window
        if ws + self.window <= self.watermark:          # window already closed
            self.late_events.append((key, event_time))
            return []
        self.counts[(key, ws)] = self.counts.get((key, ws), 0) + 1
        self.watermark = max(self.watermark, event_time - self.lateness)
        return self._close(self.watermark)

    def flush(self):
        return self._close(math.inf)
```

**Why it works.** The late check uses the watermark *before* the event:
an event cannot make its own window late. The watermark only moves forward
(`max`), so a closed window never reopens, and every event is either counted
in exactly one emitted window or reported late (the grader checks that
accounting).

**Common mistakes.**

* Setting `watermark = t - lateness` on every event instead of taking the
  `max`: an old event moves the watermark *backwards*, a window that was
  already emitted can be re-created by the next event and emitted twice.
* Closing on `>` instead of `>=`: a window `[0, 10)` is complete when the
  watermark reaches 10, because no event with `t < 10` is expected any more.
* Keeping closed windows in `counts`: memory grows forever, and a late event
  silently increments a window that was already emitted.
* `int(t // w) * w` is fine for positive times; `math.floor` also handles
  negative ones.

**Push further.** `F.WindowAggregator` scans a heap of window end times
instead of all open windows on each event: `O(log n)` per close instead of
`O(n)`. Real engines do the same with timers.

---

## 04-2 · `parity_report`

✅ Reference code: [`code/nb04.py`](code/nb04.py) (`ex2`).

The core is an outer merge with `indicator=True`:

```python
m = batch_df[keys + ["value"]].merge(stream_df[keys + ["value"]], on=keys, how="outer",
                                     suffixes=("_b", "_s"), indicator=True)
both = m[m["_merge"] == "both"]
```

then counts of `left_only` / `right_only`, and on `both` a match mask
`(nan_a & nan_b) | (|a - b| <= tol)`.

**Design choices worth defending.**

* **Missing keys are failures, not zeros.** A window the stream never emitted
  is a different bug (a dropped partition, a stuck watermark) from a window
  with a slightly different count. Report them separately.
* **Duplicated keys raise.** With duplicates, a merge creates a cross product
  and the test compares the wrong rows while still printing reassuring numbers.
* **Tolerance is explicit.** Counts should match exactly (`tol=0`); floats
  computed in different orders (sums of CTRs, averages) need a small
  tolerance; a *policy* difference (dropped late events) needs a separate
  threshold on `match_rate`, not a bigger `tol`.

---

## 04-3 · Re-delivered events after a restart (written)

✍️ A strong answer, case by case:

**(a) At-least-once, no dedup.** Four minutes of events are processed twice:
the job's window state double-counts them, so "clicks in the last 10
minutes" is inflated for every item active in those minutes (popular items
the most). Redis receives inflated values; the ranker over-promotes items
that were clicked during the replay window; the offline log of served
features records the inflated values (consistent with what was served, but
not with the batch definition, so parity tests fail). Detect: input rate vs
Kafka producer rate around restarts, counts of distinct event ids vs events
per window, parity against batch, a spike in feature values at deployment
times.

**(b) At-least-once, dedup by event id.** Correct, *if* the dedup state
survived the restart (it is part of checkpointed state) and the replay is
within the dedup horizon (the watermark/state TTL). If dedup state is
in-memory only, it is lost on restart, and you are back to (a). Duplicates
arriving after their window closed are handled by the late policy (dropped
or treated as updates: then they must be deduplicated there too). Detect:
the counter of ignored duplicates (should spike at the restart, then fall
to zero), state size, parity.

**(c) Exactly-once checkpointing, non-transactional Redis sink.** Flink
restores the state *and* the Kafka offsets of the checkpoint, so the internal
counts are correct. But the Redis writes made in the 4 minutes after the
checkpoint are re-sent. If writes are idempotent upserts of the *full*
window value (`SET item:window = value`), replaying them is harmless: the
same keys get the same values. If the sink does increments (`INCRBY`), Redis
double-counts: exactly-once state, at-least-once effects. A feature log
written by the same sink gets duplicate rows. Fix: idempotent sink writes
(set, not increment; key includes the window), or a transactional/two-phase
commit sink. Detect: compare Redis values with the job's emitted values;
duplicate rows per (key, window) in the feature log.

---

## Checkpoint · 04

1. **Event vs processing time.** ✅ Event time: when it happened at the
   source. Processing time: when the processor sees it. A "clicks in the last
   15 minutes" feature computed on arrival time puts a phone's buffered clicks
   from an hour ago into the current window: in our simulation, 15% of
   15-minute windows had different counts than the event-time definition.
   Training uses event time, so the model is served a different feature.
2. **Watermark.** ✅ A watermark is the processor's assertion that events
   with event time up to W have (almost) all arrived; windows ending at or
   before W can be emitted. Allowed lateness trades completeness against
   latency and state: in our run, 0 s lateness dropped 1,138 events (92% of
   windows exactly right, about 20 open windows), 1 hour dropped none (100%)
   but held about 100 open windows and published each result an hour late.
   For a 15-minute feature, choose a lateness from the measured delay
   distribution (our p97 was about 13 s: a lateness of 15-60 s covers the
   bulk), plus a late policy for the tail.
3. **Policies.** ✍️ Drop: simplest, slight undercount; fine where small bias
   is acceptable. Update: correct eventually (100% in our run with about 1,100
   corrections), needs upsert-capable sinks; consumers may have acted on the
   first value. Side output: keeps the streaming result fast and simple and
   lets a batch job reconcile. Fraud: decisions are made immediately, so
   favour low latency with update semantics for later features/labels and side
   outputs for audit; a dashboard: update (or a batch recomputation that
   overwrites) so the numbers become exact.
4. **Delivery semantics.** ✅ At-least-once: every event affects the result
   one or more times (duplicates after failures). Exactly-once in Flink/Spark:
   the *state* and committed offsets are consistent (each event affects the
   state once) via checkpoints; effects on external systems are exactly-once
   only with transactional or idempotent sinks, because outputs written after
   the last checkpoint are written again on recovery. Idempotent operations
   (dedup by source event id, upserts keyed by window) make replays harmless;
   in our run dedup turned a 5%-inflated click count back into the exact one.
5. **Train/serve equality for a streaming feature.** ✅ One definition (ideally
   one implementation, run in batch mode for backfills); log the served feature
   values with each request id; a daily parity job compares logged values and
   streaming outputs with the batch definition per key and window (missing
   keys, mismatches beyond tolerance, match rate), alerting on regressions;
   build training sets from the logged values or from an as-of join on window
   end times. Our parity check found 149 of 1,886 windows differing by at most
   2 clicks: exactly the dropped late events.
6. **Session feature design.** ✍️ Source: client view events on Kafka with
   a client-assigned event id and event time, partitioned by user id (order per
   user). Job: session windows per user with a 30-minute gap, merging on
   out-of-order events; state = current session start, last event time, count
   (and a small dedup set of event ids); short allowed lateness (seconds),
   because the value is needed now; the client can also send its own session
   counter to cross-check. Sink: upsert `user:session_views` in the online
   store with event time, TTL slightly above the gap. Latency budget: the
   ranker reads it in about 1 ms; end-to-end freshness target a few seconds.
   Failures: checkpointed state, idempotent upserts, default value when
   missing/expired (the same default used in training), monitoring of consumer
   lag, watermark progress, and parity against a batch sessionisation of the
   logs.
