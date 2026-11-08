# Solutions · Chapter 14 · Health, logs & latency

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `percentiles` (nearest rank)

✅ Reference code: [`code/nb13.py`](solutions.py) (`ex1`), graded in CI.

```python
import math

def percentiles(latencies, qs=(50, 95, 99)):
    xs = sorted(float(x) for x in latencies)
    if not xs:
        raise ValueError("no latencies")
    n = len(xs)
    return {q: xs[max(1, math.ceil(q * n / 100)) - 1] for q in qs}
```

**Why nearest rank.** It always returns a latency that actually happened, and
"p95 = 42 ms" then means literally "95% of requests took at most 42 ms". The
linear interpolation `np.percentile` uses by default can return a value no
request had; for large samples the two agree closely, for 20 requests they
can differ a lot. In NumPy ≥ 1.22, `np.percentile(x, q, method="inverted_cdf")`
is the nearest-rank definition.

**Common mistakes.**

* `xs[int(q / 100 * n)]`: floor instead of ceiling and 0-based confusion; it is
  off by one rank for most n.
* `math.ceil(q / 100 * n)`: mathematically the same, but `0.07 * 100` is
  `7.000000000000001` in floating point, and the ceiling jumps to rank 8.
  Multiply first: `q * n / 100`.
* Forgetting to sort, or sorting the caller's list in place (`latencies.sort()`)
  and surprising them.

---

## Exercise 2 · `summarize_logs`

✅ Reference code: [`code/nb13.py`](solutions.py) (`ex2`).

```python
def summarize_logs(records):
    df = pd.DataFrame([r for r in records if r.get("msg") == "request"])
    rows = []
    for path, group in df.groupby("path", sort=True):
        rows.append({"path": path, "count": len(group),
                     "error_rate": float((group["status"] >= 500).mean()),
                     "p95_latency_ms": percentiles(group["latency_ms"], qs=(95,))[95]})
    return pd.DataFrame(rows).set_index("path")
```

**Decisions worth saying out loud.**

* Filter on `msg == "request"` *before* building the frame. `model_loaded`
  and `model_reload` events have no `status`, and mixing them in turns the
  integer status column into floats with `NaN`.
* A 422 is not a service error: the caller sent an invalid body and would get
  the same answer again. Counting 4xx as errors makes the error rate depend on
  how careless your callers are, not on your service. Track 4xx separately,
  because a sudden rise usually means a client deployed a contract change.
* A per-path table matters because `/health` calls are fast and numerous:
  pooled with `/predict`, they make p95 look better than any user experiences.

---

## Exercise 3 · An SLO and an alert (written)

✍️ Personal. A strong answer is specific and justified. For example:

> **Who calls it:** the customer-success tool, when an account manager opens a
> customer page, plus a nightly batch job. The interactive caller needs an
> answer well under a second; the batch job needs throughput, not latency.
>
> **SLO (28-day rolling window, measured at the load balancer, i.e. close to
> the client):** 99.5% of `POST /predict` requests return a non-5xx response,
> and 95% of them complete in under 200 ms. Requests rejected with 4xx count as
> good (the service behaved correctly).
>
> **Why 200 ms:** we measured p95 ≈ 20–25 ms for a single request on a shared
> 4-vCPU Xeon (HTTP, one worker, one client) and p95 ≈ 100 ms with 4
> concurrent clients. 200 ms leaves room for a slower production network and
> moderate load, while staying invisible on a page load.
>
> **Alerts (burn-rate style):** page if the error budget is burning fast,
> e.g. more than 2% of requests failed over the last 5 minutes *and* the last
> hour, or `/ready` fails on all replicas; open a ticket if p95 > 200 ms for
> 30 minutes or if more than half of the monthly error budget is spent.

What graders look for: a metric with a measurement point, a numeric target,
a window, a definition of "bad", a page/ticket split, and a link between the
target and the measured numbers *and the machine they came from*. Weak
answers say "latency should be low" or copy `DEFAULT_RULES` without a reason.

---

## Checkpoint answers

**1. Liveness vs readiness.** Liveness: is the process alive (restart it if
not); it must not depend on the model or anything external. Readiness: should
it receive traffic (a model is loaded). If liveness checked the model, a slow
or failed model load would make the platform kill and restart the process in
a loop, and a registry outage would restart every replica at once.

**2. "Average is 6 ms".** Ask for p50, p95 and p99 with the request count, per
endpoint, from the client side, under the real load, and on which machine.
The mean hides the tail that users notice, and one slow dependency or GC
pause can make p99 ten times the mean.

**3. Log line.** Request id, timestamp, method, path, status, latency, model
version (and service version). Never raw personal data or feature values,
secrets, or tokens; log identifiers and aggregates, keep inputs (if needed)
in an access-controlled store.

**4. Server vs client latency.** The server's clock starts after the request
is read and stops before the response is sent; it misses connection set-up,
queueing before the app, network and client-side parsing. The client number is
what users experience.

**5. CPU-bound, p95 doubles at peak.** (a) More replicas/workers: costs money
and memory, and in-memory state becomes per process. (b) Batch where the
caller allows it: lower per-row cost, higher latency for each batch. (c) Make
the request path cheaper (skip pandas for one row, a lighter model, cache
repeated requests): engineering time and a risk of skew. Also: autoscaling on
p95/CPU, which cannot react faster than the cold start.

**6. `--workers 4`.** Four processes, each with its own GIL and model copy:
throughput scales with free cores, memory ×4. In-memory metrics describe one
worker, and `/admin/reload` reloads only the worker that receives it, so
after a rollback the workers can disagree about the model version. Use a
restart (or one worker per container) to switch all of them, and a shared
metrics system.

🔬 The measured numbers above (p95 ≈ 20–25 ms single-client, ~65 req/s with
one worker, per-row cost ≈ 13 ms at batch size 1 vs ≈ 0.05 ms at 1000) are from
the course's verification machine, a shared 4-vCPU Intel Xeon @ 2.10 GHz under
other load. Yours will differ.
