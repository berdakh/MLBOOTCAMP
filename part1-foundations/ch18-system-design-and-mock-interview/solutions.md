# Solutions · Chapter 18 · System design & mock interview

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `peak_qps`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def peak_qps(dau, requests_per_user_per_day, peak_factor):
    return float(dau * requests_per_user_per_day / 86_400 * peak_factor)
```

**Mental arithmetic.** 1M × 20 = 20M requests/day. A day is ≈ 10⁵ seconds
(86,400), so ≈ 200 QPS on average (exact: 231), × 3 ≈ **700 QPS** at peak
(exact: 694). Say "about 700" and move on; one significant figure is the
point. Rounding 86,400 to 100,000 under-estimates by ~14%, which is fine for
architecture decisions and worth saying.

**Where the peak factor comes from.** The ratio of the busiest hour (or
minute) to the daily average: 2-3 for a global consumer product, higher for a
single-time-zone or event-driven one. Ask for it or state it.

---

## Exercise 2 · `latency_budget`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def latency_budget(total_ms, components):
    spent = sum(components.values())
    if spent > total_ms:
        raise ValueError(f"components need {spent} ms, {spent - total_ms} ms over the {total_ms} ms budget")
    return total_ms - spent
```

**Design point.** Adding p95 latencies is a conservative approximation (the
p95 of a sum is usually lower than the sum of p95s, unless the steps are
correlated, e.g. all slow under the same load). In an interview say: "I budget
with p95s and keep headroom; then I measure the end-to-end p95 under load,
which is the number that counts."

---

## Exercise 3 · A full design for one practice prompt (written)

[open-ended] Open-ended. A strong answer to **prompt 1, card-fraud detection**, under the
nine headings:

1. **Clarify.** Decision: approve / decline / step-up (send a code) for each
   card authorisation, in < 100 ms end-to-end with a model share of ~20 ms.
   Scale: say 50M transactions/day. Fraud rate ~0.1%. Today: rules.
2. **Metrics.** Offline: PR-AUC; recall at a fixed false-positive rate
   (declines annoy good customers); value-weighted recall (fraud in $). Online:
   fraud losses in $ per $ processed, false-decline rate, step-up completion.
   Guardrails: latency p99, approval rate by country and merchant type.
3. **Data & labels.** Labels from chargebacks and customer reports, arriving
   days to ~90 days late; declined transactions never get a label
   (**selection bias**): keep a tiny random approve-anyway sample for
   evaluation, or use the step-up outcome as a partial label.
4. **Features.** Online velocity counters (transactions per card in the last
   10 min / 1 h / 24 h, distinct merchants, distance from the last location)
   computed by a streaming job into a low-latency store; batch features (card
   age, historical spend profile) refreshed nightly. Point-in-time training data
   by logging the feature vector **at decision time**.
5. **Model.** Rules + logistic regression baseline, then gradient-boosted trees
   (strong on tabular, fast to serve). Retrain weekly; fraud patterns drift
   adversarially.
6. **Serving.** Online only. 50M/day ≈ 580 QPS average; peak factor 3 ≈ 1,700 QPS.
   At ~200 QPS per replica at 60% utilisation: ~15 replicas. Latency budget
   (100 ms): network 20, feature store 25, model 20, rules 10, logging 5,
   headroom 20. Fallback if the feature store is down: rules only.
7. **Evaluation & experimentation.** Time-split backtests weighted by $; shadow
   mode (score but do not act) for two weeks; then an A/B test by card (not
   transaction) on fraud loss and false-decline rate.
8. **Monitoring.** Feature drift, score distribution, decline rate per hour
   (a sudden jump is an incident), latency, fallback rate; delayed-label
   PR-AUC weekly. Rollback by registry pointer.
9. **Risks.** Disparate decline rates by geography; attackers probing the
   threshold; explaining declines to customers and regulators.

**Trade-off table:** threshold at a fixed FPR vs cost-based (cost-based needs
$ values for both errors); streaming features vs batch only (freshness vs
complexity: velocity features are the signal, so streaming is required);
trees vs a sequence model over transactions (latency and explainability vs
accuracy, measured in shadow mode).

Grade your own: every heading present, at least one number per heading
where a number makes sense, at least one explicit trade-off, labels' delay
and bias addressed, a fallback named.

---

## Exercise 4 · Two STAR stories (written)

[open-ended] Personal: only real stories. A strong *shape* for prompt 1, from a
project with several recordings per participant (adapt the facts to yours,
never borrow them):

* **S (10%).** "In my thesis project, we reported cross-subject accuracy for
  an activity-recognition classifier trained on wearable-sensor data."
* **T (10%).** "I owned the evaluation pipeline and the numbers in the report."
* **A (60%).** "When a colleague's re-implementation scored lower, I traced the
  difference to my preprocessing: I had fitted the normalisation (and the
  feature selection) on all participants before the leave-one-subject-out loop, so
  the held-out subject influenced its own features. I moved every fitted step
  inside the loop as a pipeline, re-ran, and added a test that fails if any
  fitted object sees the held-out subject. I also re-checked our previous
  results with the corrected pipeline before reporting anything."
* **R (20%).** "Accuracy dropped by a few points (state your real number), we
  reported the corrected figure, and every pipeline I've built since keeps
  preprocessing inside the model object. It's exactly the training/serving-skew
  discipline I applied in the churn project."

Checks: "I" for your actions; the decision and the alternative you rejected;
an honest number; what you do differently now. Prompt 2 (non-replication) is
strong when you show how you separated noise (seed/subject variance), a bug and
a real difference. Prompt 3 (simplify) is strong when you name what you gave up
and how you measured that it was acceptable.

---

## Checkpoint answers

**1. Nine headings.** Clarify goal & constraints → success metrics (offline,
online, guardrails) → data & labels → features → model → serving →
evaluation & experimentation → monitoring & iteration → risks & ethics.
Metrics come first because the decision and how it is measured determine
everything else: the label, the split, the threshold, and whether you need a
model at all. A model chosen before the metric is optimised for the wrong thing.

**2. Arithmetic.** 4M × 10 = 40M/day ÷ 86,400 ≈ 463 QPS average; × 3 ≈
**1,390 QPS** at peak. At 100 QPS per replica run at 60%: 1,390 / 60 ≈ 23.1 →
**24 replicas** (plus spread across zones for availability).

**3. Precomputed score in the console.** The nightly batch already scores
every account; reading a stored score is one key-value lookup (single-digit
ms, no feature computation, no model dependency at request time), traffic is
tiny anyway, and the console degrades gracefully (show yesterday's score with
its date). Online recomputation adds latency, cost and a second code path for
features (skew risk), and it only pays off if same-day activity would change the
agent's action, which you should check before building it.

**4. Interventions and labels.** Acting on predictions changes outcomes: saved
customers do not churn, removed comments are never seen, declined fraud never
gets a chargeback. The labels then reflect your policy, not the underlying
risk, and naive retraining learns from that. Build: an untreated random
holdout (or a small random review sample) for unbiased evaluation, logs of every
action with model version and score, and evaluation on the holdout only.

**5. Interleaving.** For comparing two rankers, merge their result lists for
the *same* user and query (e.g. team-draft interleaving) and credit each click
to the ranker that contributed the item. Each user sees both rankers, so the
comparison is within-user, removing between-user variance. It typically needs
far less traffic than an A/B test for the same sensitivity. Its limit: it
measures preference (clicks), not long-term business metrics, so confirm
winners with an A/B test.

**6. STAR story.** [open-ended] Personal. Use your Exercise 4 answer; time it. Ninety seconds
is roughly 220 words; most of it should be the Action.
