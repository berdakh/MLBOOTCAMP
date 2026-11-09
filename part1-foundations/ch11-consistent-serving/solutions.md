# Solutions · Chapter 11 · Consistent serving

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `parity_check`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def parity_check(model, http_client, rows_df):
    threshold = float(http_client.get("/model").json()["threshold"])
    bodies = [example_payload(row) for _, row in rows_df.iterrows()]
    r = http_client.post("/predict/batch", json={"customers": bodies})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    p_api = np.array([p["churn_probability"] for p in preds], dtype=float)
    flag_api = np.array([p["flagged"] for p in preds], dtype=bool)
    p_off = model.predict_proba(rows_df[FEATURES])[:, 1]
    return {"max_abs_diff": float(np.max(np.abs(p_off - p_api))),
            "n_decision_flips": int(np.sum((p_off >= threshold) != flag_api))}
```

[expected] With the served model you should see `max_abs_diff` ≈ 5e-7 (the API rounds
to six decimals) and `0` flips. The grader then passes an offline model whose
scores are shifted up slightly; your check must report a difference of the
size of the shift and at least one flip.

**Key decisions.**

* The threshold comes from **`/model`**, not from the registry file or a
  constant. The parity test should check what the *service* believes, which
  may differ (e.g. `CHURN_THRESHOLD_OVERRIDE` is set in production).
* Flips are computed against the API's **`flagged`**, not by re-thresholding
  the API's rounded probability. That way the test covers the service's
  decision logic too (a `>` vs `>=` bug, a stale threshold).
* `rows_df[FEATURES]` selects columns explicitly; extra columns
  (`snapshot_date`, the label) never reach the model.

**Common mistakes.** Asserting `max_abs_diff == 0` (fails because of rounding);
comparing with `np.allclose` defaults (`rtol=1e-5` hides real differences on
small probabilities, since 1e-5 × 0.02 = 2e-7); posting one row at a time (it
works, but is 300 requests where one is enough).

---

## Exercise 2 · `fixed_serve_preprocess`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def fixed_serve_preprocess(customers, stats):
    return train_transform(pd.DataFrame(customers).reindex(columns=FEATURES), stats)
```

**Why this one line is the right answer, not a shortcut.** The three bugs
(alphabetical column order, zero-imputation, a forgotten `log1p`) all came
from having **two implementations** of one transform. The fix is to have one.
`pd.DataFrame(list_of_dicts)` aligns keys by name (so key order cannot
matter), turns missing keys and `None` into NaN, and `reindex(columns=...)`
guarantees every expected column exists even if no request in the batch
contained it. Then the *training* function does the rest. This is exactly what
`churnlab.api` does with the saved `Pipeline`.

If you fixed the loop instead (iterate over `NUMERIC` in training order,
impute `stats["median"][col]`, log1p every column in `LOG_FEATURES`), it
passes too, and you now maintain two copies. The next change to training
(say, a new log column) must be made twice, and the parity test is the only
thing that will tell you when someone forgets.

**Common mistakes.** Missing the `None` → median path for **omitted** keys
(`c.get(col)` returns `None`, fine; `c[col]` raises `KeyError`); one-hot by
`pd.get_dummies` on the request batch (creates only the categories present
in *that* batch, so the column count changes with the input).

---

## Exercise 3 · Skew a stored-row parity test cannot catch (written)

[open-ended] Open-ended. The parity test replays rows from the **offline** feature
table, so it can only find differences *after* the feature values exist. Any
difference in how the feature values themselves are produced online is
invisible to it. Strong answers name a mechanism and a detection/prevention:

| Source of skew | Example | Detect / prevent |
|---|---|---|
| Feature computed by different code online | the CRM computes `sessions_4w` from a live events stream; training used weekly SQL aggregates that exclude the incomplete current week | one feature definition shared by both paths (a feature store or the same SQL); log served features and compare their distributions with training (PSI per feature) |
| Stale or delayed features | the online store refreshes nightly, so `weeks_since_last_active` is up to a day old; or a pipeline failed and features are a week old | freshness timestamps on every feature, alerts on age; log `as_of` with each request |
| Point-in-time / time-zone differences | training used snapshots at 00:00 UTC; the service computes "last 30 days" in local time, so `tickets_30d` windows shift by hours | define windows in UTC everywhere; unit tests on window boundaries; compare served features against the batch features for the same customer and time |
| Upstream semantic changes | billing changes `mrr_usd` from monthly to annual for some plans; a new `plan` value appears | input monitoring (ranges, unseen-category rate), data contracts with the upstream team |
| Missing-value encoding upstream | a caller sends `0` for "no tickets yet" where training had NaN for `avg_resolution_hours_90d` | document `null` semantics in the contract; monitor missing-rate per field vs training |
| Population shift (not skew, but often confused with it) | the service is called for trial accounts that never appeared in training | monitor the input distribution; scope the contract |

Credit any three with a specific mechanism and a credible check. The deeper
point: **log the features the service actually received** (with request id
and model version). Then you can run a parity test on *live* inputs too:
recompute the same customers' features offline for the same timestamp, and
compare.

---

## Checkpoint answers

**1. Skew.** Any difference between how inputs are prepared in training and
in serving. Nothing raises: the model gets numbers of the right shape and
returns valid probabilities, just different ones. [verified] In this notebook, three
plausible bugs flipped 531 of 6,956 validation decisions with no error.

**2. Three re-implementation bugs.** Column order (positional arrays),
imputation value (0 vs median), a missing transform step (a forgotten
`log1p`). [verified] Column order was the most destructive here (≈ 590 flips, PR-AUC
0.158 → 0.068) and the hardest to see in review because every line looks
right. The forgotten `log1p` is the most *insidious*: ≈ 100 flips while PR-AUC
barely moved (0.158 → 0.157), so an aggregate metric would not reveal it.

**3. Serving the Pipeline.** The object that was fitted (imputer medians,
scaler means, categories) is the object that is served, and the
`ColumnTransformer` selects columns by name. It cannot prevent skew *before*
the model: features computed differently online, stale features, different
time windows, upstream changes (see Exercise 3).

**4. 4e-7.** Pass, *if* you know why it is not zero: this API rounds to six
decimals, so differences up to 5e-7 are expected. Without that knowledge
4e-7 is unexplained and should be investigated. A tolerance must come from
the contract (here, 6-decimal rounding), and decisions must match exactly.

**5. Loading.** `load_bundle` verifies the SHA-256 of `model.joblib` against
metadata and compares the recorded scikit-learn major.minor version with the
running one (warning by default, error in strict mode); `ModelHolder` checks
the feature list. [verified] It does **not** protect `metadata.json` itself: editing it
(e.g. the threshold) went undetected. On failure the service keeps running,
answers `/health` 200, `/ready` and `/predict` 503 with the reason, and keeps
serving the previous model if one was loaded.

**6. Investigating a complaint.** The request id (to find the log line), the
model version (to find the bundle: training data fingerprint, threshold,
metrics), the timestamp, and ideally the features the service received for
that request. Every response here carries `model_version` and the
`x-request-id` header, and every log line has both.
