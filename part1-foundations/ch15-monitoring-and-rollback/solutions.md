# Solutions · Chapter 15 · Monitoring & rollback

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `psi`

✅ Reference code: [`code/nb14.py`](solutions.py) (`ex1`), graded in CI against
`churnlab.monitoring.psi`.

```python
def my_psi(reference, current, bins=10):
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))
```

**Why each step.**

* Edges from the **reference's** quantiles, so each bin holds ~10% of the
  reference: a fixed yardstick that does not move with the current data.
* `np.unique`: with ties (counts, mostly-zero features) several quantiles are
  equal; duplicate edges would create empty, zero-width bins.
* Outer edges to ±∞: current values outside the reference range still land in
  a bin instead of being dropped by `np.histogram`.
* Clipping at `1e-4`: an empty bin would make `log(c / r)` infinite.

**Common mistakes.**

* Edges from the *combined* or the *current* sample: the result then depends
  on how much the current sample moved, which is what you are measuring.
* Using counts instead of fractions when the samples differ in size.
* `np.histogram(ref, bins=10)` (equal-width bins): a valid statistic, but a
  different one, and very sensitive to outliers.

**Push further.** The notebook found PSI's blind spot: with a 90%-zero
reference the edges collapse to one bin and PSI is 0 whatever happens. Add a
fallback for features with few distinct values: treat them as categories
(`categorical_psi`), or use fixed, domain-chosen bins (0, 1, 2, 3+).

---

## Exercise 2 · `should_rollback`

✅ Reference code: [`code/nb14.py`](solutions.py) (`ex2`).

```python
def should_rollback(candidate, baseline, policy):
    reasons = []
    cand_pr, base_pr = candidate.get("pr_auc"), baseline.get("pr_auc")
    if cand_pr is not None and base_pr is not None:
        floor = base_pr * (1 - policy["max_pr_auc_drop"])
        if cand_pr < floor:
            reasons.append(f"pr_auc {cand_pr:.3f} below {floor:.3f}")
    if candidate["error_rate"] > policy["max_error_rate"]:
        reasons.append(f"error_rate {candidate['error_rate']:.4f} above {policy['max_error_rate']}")
    if candidate["p95_ms"] > policy["max_p95_ms"]:
        reasons.append(f"p95_ms {candidate['p95_ms']:.1f} above {policy['max_p95_ms']}")
    return bool(reasons), reasons
```

**Design points.**

* Returning the **reasons** matters as much as the decision: they go into the
  rollback's `reason` in `history.jsonl` and into the incident channel. "Rolled
  back" with no reason is useless at 3 a.m.
* The drop is **relative** (`baseline * (1 - drop)`): a 0.02 drop is large
  for a PR-AUC of 0.16 and small for one of 0.8.
* `None` for PR-AUC is normal right after a deploy: labels mature 30 days
  later. Skipping the check (rather than treating `None` as 0 or crashing)
  lets the service-level checks still protect you in the meantime.
* Exactly at a limit is not a breach, so the policy's numbers mean what they
  say ("at most 1% errors").

**Common mistake.** `if candidate["pr_auc"] < ...` without the `None` check
raises `TypeError` in the first minutes after every deploy, exactly when the
check is needed.

---

## Exercise 3 · Incident postmortem for the bad deploy (written)

✍️ Personal. A strong six-sentence answer (times from the notebook run are
illustrative):

> **Timeline:** the automated monthly retrain registered `v3` and promoted it
> to `CURRENT` at 15:53:14 UTC; the service reloaded it at once, and the canary
> check rolled it back to `v1` about one second later. **Detection:** the
> canary comparison on the latest matured month (June 2025, 2,144 customers)
> fired two *page* rules (PR-AUC 0.38× the incumbent's, score spread 0.09×) while
> every service-health signal (errors, latency, readiness) stayed green.
> **Impact:** for the time `v3` was live its scores were close to the base rate
> for everyone, so the customer-success list for that period was effectively
> random; in the notebook that window was about a second, but without the
> canary it would have lasted until the next retrain or until labels matured
> 30 days later. **Root cause:** the training data join misaligned labels and
> rows (simulated by shuffling the label column), and the promotion gate that
> compares candidate and incumbent validation metrics had been disabled, so a
> model whose recorded validation PR-AUC (0.054) equalled the base rate was
> promoted. **Fix:** `registry.rollback()` restored `v1`, `/admin/reload`
> applied it, and we verified the served version and that predictions were
> identical to those before the deploy. **Prevention:** make the promotion gate
> mandatory in the retraining pipeline (refuse to promote when validation PR-AUC
> is below 0.9× the incumbent's or when the score spread collapses), add a
> unit test that the training join preserves the label rate per snapshot, and
> run every new version in shadow for a day before promotion.

What graders look for: concrete numbers and times, the signal that did *not*
catch it (service health), a root cause with both a technical and a process
part, and a prevention step someone can own. Blameless: "the gate was
disabled", not "Alex disabled the gate".

---

## Checkpoint answers

**1. Three layers.** Service health (errors, latency, readiness: seconds);
data and prediction drift (feature PSI, unseen categories, score distribution:
as soon as requests arrive); performance (PR-AUC, calibration against real
outcomes: 30 days after each prediction, the label horizon).

**2. PSI 0.4 on an input.** Not necessarily. Check, in order: is it real (look
at the two distributions; is it an artifact such as the snapshot-weekday
effect on `weeks_since_last_active`, a pipeline bug, a unit change, a new
category)? How much does the model use the feature? Did the score
distribution move? Then the matured-cohort performance and calibration when
labels arrive. Drift opens an investigation; it is not an automatic retrain.

**3. Two ways PSI misleads.** (a) It can be blind: quantile bins collapse for
zero-inflated features, so `failed_payments_90d` gave PSI 0.000 while the
share with a failed payment rose from 10% to 24%. (b) It can scream about
nothing: discrete, calendar-dependent values (`weeks_since_last_active`) gave
PSI > 3 in the *control* world with no shift at all. Also: it depends on the
bin count and is noisy for small samples.

**4. PR-AUC 0.16 → 0.26.** Not evidence of improvement: PR-AUC rises with the
positive rate, which went from about 5% to about 8.6%. Compare
prevalence-independent views: ROC-AUC (0.73 vs 0.76, overlapping intervals),
PR-AUC ÷ positive rate (≈3.1 both), lift at a fixed fraction, and calibration
(mean score vs observed rate), with confidence intervals.

**5. A rollback.** On disk: `CURRENT` is atomically rewritten to the previous
version (from `history.jsonl`), and a `rollback` record is appended; the bundles
are untouched. The service: `POST /admin/reload` re-reads `CURRENT` and swaps
the model under a lock (on failure it keeps the old one). Verify: `/ready` and
`x-model-version` show the old version on every worker/replica, and
predictions on a fixed slice equal the pre-deploy ones.

**6. Release strategy.** With a 30-day label delay a live canary cannot
measure quality quickly, so: an offline canary on the latest matured slice
(ranking quality) plus **shadow** mode for a few days (crashes, latency, score
distribution, disagreement with the incumbent, at no user risk), then a full
switch with instant rollback by pointer (blue/green-like). A traffic canary
adds little here except for service-level risk.

🔬 Numbers above are from the course's verification run (seed 2025, 3,000
simulated customers, March-July 2025 snapshots). Your run should match
closely, because the simulation and models are seeded; library versions can
move the third decimal.
