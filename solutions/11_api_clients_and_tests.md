# Solutions · 11 · API clients & tests

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 11-1 · `retrying_get`

✅ Reference code: [`code/nb11.py`](code/nb11.py) (`ex1`), graded in CI.

```python
RETRYABLE = {429, 502, 503, 504}


def retrying_get(http, path, retries=3, backoff=0.1):
    for attempt in range(retries + 1):
        last_try = attempt == retries
        try:
            r = http.get(path)
        except httpx.TransportError:      # connection errors and timeouts
            if last_try:
                raise
        else:
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRYABLE or last_try:
                r.raise_for_status()      # 4xx / 500: fail fast; retryable: out of attempts
        time.sleep(backoff * 2 ** attempt)
```

**The three rules, and why.**

1. **Retry only what can succeed next time.** 429/502/503/504 and transport
   errors are about the *moment*: rate limits, a restarting instance, a
   network blip. 4xx are about the *request*: retrying a 422 sends the same
   invalid body again and gets the same 422, so it only adds load. A 500 is
   usually a bug, which also fails the same way again.
2. **Bounded attempts.** `retries + 1` calls at most, then raise the last
   error. Unbounded retries turn an outage into a self-inflicted
   denial-of-service.
3. **Exponential backoff** (`backoff × 2^attempt`). In production add jitter,
   `delay * random.uniform(0.5, 1.5)`, and honour a `Retry-After` header on
   429/503.

`httpx.TransportError` is the base class of `ConnectError`, `ReadTimeout`,
`ConnectTimeout` and friends; catching it covers "the request may never have
arrived". Retrying after a *read* timeout is only safe because the GET has no
side effects.

**Common mistakes.** Retrying on `status >= 400` (retries 422s; the grader
counts 4 calls instead of 1); catching `Exception` around the whole call (also
swallows the `HTTPStatusError` you meant to raise); returning `None` after the
last failure instead of raising; sleeping after the final attempt.

---

## 11-2 · `test_rejects_negative_sessions`

✅ Reference code: [`code/nb11.py`](code/nb11.py) (`ex2`).

```python
def test_rejects_negative_sessions(client):
    r = client.post("/predict", json={**valid_payload, "sessions_4w": -1})
    assert r.status_code == 422, f"expected 422, got {r.status_code}"
    errors = r.json()["detail"]
    assert any(e["loc"][-1] == "sessions_4w" and e["type"] == "greater_than_equal" for e in errors), errors
```

**What makes it a good test.**

* It starts from a **valid** payload and changes **one** field, so the only
  possible reason for a 422 is that field. A test that sends `tenure_days=-1`
  (or an empty body) also gets a 422, but from a different rule, so it passes
  against an app that has lost the `sessions_4w` bound. The grader's first
  buggy app catches exactly that.
* It asserts the **exact** status, not "not 200". The grader's second buggy
  app crashes with a 500 for negative sessions; `assert r.status_code != 200`
  passes there, although a 500 for bad input is a bug (notebook 09).
* It checks the error **names the field**, which is part of the contract
  callers rely on.
* The assertion messages show what was received, so a CI failure is readable
  without re-running.

---

## 11-3 · CI vs staging (written)

✍️ Open-ended. The organising idea: **CI** runs on every push, must be fast
(minutes), deterministic and need no real infrastructure, so it uses small
data, temporary registries and in-process clients. **Staging** is a
production-like deployment (same container image, real network, real model
artifact, production-sized data) that runs before promotion and catches what
only appears when the pieces are wired together.

| Check | Where | Why there |
|---|---|---|
| Unit tests (`tests/`): features SQL, splits, metrics, registry | CI | fast, pure Python, catch most regressions at the source |
| API contract/behaviour/parity/ops tests with `TestClient` and a tiny fixture model (this notebook's suite) | CI | seconds, no server, no network; the fixture model need not be good, only real |
| Lint, type checks, `nbbuild --check`, dependency resolution / lock file | CI | cheap, catch mistakes before review |
| Build the Docker image; run the container and hit `/health` | CI (if the runner supports Docker) | the image itself is an artifact; a broken `Dockerfile` should fail the PR |
| Parity of the **candidate** model against offline scores on a held-out sample, in the built image | staging | needs the real artifact and the production image (library versions!) |
| Smoke test against the deployed service: `/ready`, a few known requests with expected scores, version header | staging | proves the deployment wiring: config, env vars, registry path, secrets |
| Load test: p95/p99 latency, throughput, memory at realistic batch sizes | staging | needs production-like hardware; too slow and noisy for CI |
| Client integration: the CRM's real client calling the staging URL (timeouts, retries, auth, TLS) | staging | real network paths and real callers |
| Shadow / canary comparison with the live model on real traffic | staging → production | only real inputs reveal online/offline feature skew (notebook 10, exercise 3) |
| Rollback drill: promote, then roll back, check `/ready` and the version | staging | rehearses the procedure before you need it (notebook 14) |

Credit three of each with a reason. A strong answer also says **what blocks
what**: CI must pass to merge; staging checks must pass to promote a model or
release an image; and anything that fails in staging but not CI is a missing CI
test.

---

## Checkpoint · 11

**1. What to retry.** Connection errors/timeouts (if the call is idempotent),
429, 502, 503, 504, with bounded, backed-off attempts. Never 4xx: the request
itself is wrong and will fail identically. Usually not 500: it is a bug.
✅ Observed: two 503s then success on the 3rd call; a 422 reached the server
exactly once.

**2. Backoff with jitter.** Exponential backoff spaces retries out
(`b, 2b, 4b, ...`) so a struggling server gets time to recover instead of
receiving more load. Jitter randomises each delay so that many clients that
failed at the same moment do not retry at the same moment (the
"thundering herd").

**3. Timeouts.** Without one, a hung server holds the caller's thread,
connection and user request indefinitely, and the hang propagates upstream.
✅ The 0.5 s timeout fired after 0.50 s. Choose it from measured latency (a
few times the p99, notebook 13) and from the caller's own deadline. Set
connect and read timeouts separately when they differ.

**4. Four categories.** Contract: `test_rejects_out_of_range[sessions_4w]`
(422), `test_oversize_batch_is_413`. Behaviour:
`test_flagged_at_exact_threshold`. Parity: `test_api_matches_offline`.
Operations: `test_not_ready_without_a_model`,
`test_version_header_and_request_id`.

**5. Mutation testing.** Deliberately introduce small bugs (mutants) and check
that the suite fails for each. Surviving mutants show untested behaviour.
✅ Two swapped features (`sessions_4w` ↔ `sessions_prev_4w`) still give valid
probabilities in [0, 1]; only the parity test caught it. `>` instead of `>=` was
caught only by the boundary test. The survivor (4xx not counted in
`/metrics`) shows the metrics endpoint is untested.

**6. Idempotency.** A prediction has no side effects, so receiving it twice
changes nothing. A payment executed twice charges twice. For non-idempotent
endpoints the client sends an **idempotency key** (a unique id per logical
operation); the server stores results by key and returns the stored result for
a repeat instead of executing again.
