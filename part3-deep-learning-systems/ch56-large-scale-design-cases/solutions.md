# Solutions · Chapter 56 · Large-scale design cases

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `latency_budget_split`

[verified] Reference code: [`code/nb18.py`](solutions.py) (`ex1`), graded in CI.

```python
def latency_budget_split(total_ms, stages):
    if total_ms <= 0:
        raise ValueError(f"total_ms must be positive, got {total_ms}")
    negative = {k: v for k, v in stages.items() if v < 0}
    if negative:
        raise ValueError(f"stages cannot be negative: {negative}")
    spent = sum(stages.values())
    if spent > total_ms:
        raise ValueError(f"stages total {spent} ms: over the {total_ms} ms budget by {spent - total_ms} ms")
    return {"remaining_ms": total_ms - spent,
            "shares": {k: v / total_ms for k, v in stages.items()},
            "largest_stage": max(stages, key=stages.get) if stages else None}
```

**What to say with it.** Budgets are set on p99 (or p95) per stage, and
percentiles do not add exactly: the sum of stage p99s is an upper bound for a
sequential chain (`ds.sequential_p99_upper_bound`), pessimistic when stages
are independent. Parallel stages count once, at their slowest member, and
fan-out makes that member's tail worse (`ds.fanout_tail_probability`).
`largest_stage` is where optimisation effort goes first; headroom is not
waste, it absorbs GC pauses, retries and the next feature.

---

## Exercise 2 · `serving_fleet_size`

[verified] Reference code: [`code/nb18.py`](solutions.py) (`ex2`).

```python
def serving_fleet_size(peak_qps, per_replica_qps, headroom):
    if per_replica_qps <= 0:
        raise ValueError("per_replica_qps must be positive")
    if peak_qps < 0:
        raise ValueError("peak_qps cannot be negative")
    if not 0 <= headroom < 1:
        raise ValueError(f"headroom must be in [0, 1), got {headroom}")
    return max(2, int(math.ceil(peak_qps / (per_replica_qps * (1 - headroom)))))
```

The ads case needs 3,969 replicas at a (assumed) 50 requests/s each, which is
the signal to batch requests, cascade (cheap pre-ranker, heavy ranker on
fewer ads) or distil. Say where `per_replica_qps` comes from: a load test of
the real serving image at the target p99 latency (notebook 12), not the
model's throughput in a benchmark loop. A staff answer adds zone redundancy
(`ds.replicas_needed(..., zones=3)`: 27 instead of 18 replicas for the fraud
scorer) and autoscaling lag: if scaling takes 5 minutes, headroom must cover
the growth in 5 minutes of a traffic ramp.

---

## Exercise 3 · A full 45-minute design

[open-ended] Personal. A skeleton for **Case D (fraud)** at the expected depth, using
the notebook's numbers (all assumptions):

* **Requirements.** 5,000 TPS peak, ≤ 100 ms p99 decision, approve /
  decline / step-up, reason codes, analysts can add rules same day.
* **Metrics.** Online: fraud loss (amount) at 60-day maturity, false-decline
  rate, step-up rate. Offline: amount-weighted recall at a fixed FPR, PR-AUC,
  calibration. Guardrails: p99 latency, decline rate by merchant segment,
  complaints.
* **Data & labels.** Chargebacks (weeks), analyst confirmations (days),
  customer disputes (early signal). Selective labels: declined transactions
  are unlabelled → keep a small random approval holdout below a risk cap.
* **Architecture with numbers.** In-process rules → parallel feature fetch
  (velocity counters per card/device/merchant, graph features) → GBDT →
  cost-based threshold → decision. Budget: 10 + 5 + 30 + 8 + 4 + 3 = 60 ms,
  40 ms headroom (`latency_budget_split(100, ...)`). Fleet: 18 replicas at
  400 TPS each with 30% headroom, 27 across three zones.
* **Deep dive 1 (signature): adversarial drift + label delay.** Weekly
  retrains on mature labels; early-signal features; rule layer for same-day
  response; monitoring of score and rule-hit distributions by channel to
  detect probing; red-team cadence.
* **Deep dive 2: tail latency.** 6 feature calls at 1% slow each → 5.9% of
  requests wait; fix with a consolidated low-latency feature store, per-call
  timeouts, defaults for missing features with the model trained on
  missingness, and the missing-feature rate as a paged metric.
* **Failure modes and fallbacks.** Feature store down → rules-only decision,
  fail-open below an amount threshold, step-up above it (policy signed off);
  model service down → same; bad model deployed → canary on decline rate,
  auto-rollback.
* **Staff adds.** The exploration holdout and its cost (expected fraud loss
  of the holdout) as an explicit business decision; ownership of rules vs
  model and the analyst feedback loop; fail-open/closed policy by tier with
  risk and compliance sign-off; how regulators' explainability needs shape the
  model choice.

Score yourself with the notebook-20 rubric afterwards; time each part.

---

## Checkpoint answers

1. **Calibration in ads.** [verified] (simulation) The auction ranks by bid × pCTR
   and prices by the runner-up's eCPM divided by the winner's pCTR, so a
   prediction's *scale* moves money. In 18.2, over-predicting one segment by
   1.5× raised its win share from 50% to ~69% and lowered both platform
   revenue (3,172 → 3,057) and advertiser value (5,019 → 4,893). A feed ranker
   only needs the order within one request, which a constant scale error
   preserves (it still matters if scores are blended across heads or
   thresholded). Negative down-sampling at rate *w* inflates the odds by
   1/*w*; correct with *p* = *q* / (*q* + (1 − *q*)/*w*) where *q* is the
   model's output, then verify with a reliability plot per segment.
2. **Served length doubled.** [expected] It is a problem if satisfaction did not
   rise with it: the expected-watch-time objective favours long videos by
   construction (18.3: median 27 min in the top 100 vs 6 min catalogue). Check
   satisfaction signals (surveys, likes, "not interested", return visits) by
   length bucket and in a long-term holdout; check completion and early
   abandonment. Changes: blend completion/satisfaction heads, normalise
   watch time by length (or use watch-time percentile within length bucket),
   add length-distribution guardrails; decide weights by experiment.
3. **LLM conversation cost.** [verified] (arithmetic) ~0.0114 per 6-turn
   conversation at the placeholder prices, because prompt tokens grow from
   2,360 to 3,660 as history is re-sent. Levers: summarise or truncate
   history after a few turns; cap retrieved context (better re-ranking, fewer
   passages); cache the static prefix; route easy intents to a smaller
   model. Hand-off quality dominates because a human-handled conversation
   (placeholder 4.00) costs ~350× the LLM's tokens: a wrong answer that
   triggers a re-contact costs far more than any token saving.
4. **Decline rate 1% → 4% at 2 a.m.** [expected] First check whether the *inputs*
   changed: feature-service timeouts and missing-feature rate (defaults
   pushing scores up), a feature pipeline that stopped updating, a recent
   deploy (model, rules, threshold config), a rule with a sudden hit-rate
   jump. Then check whether *traffic* changed: one merchant/channel/country
   driving the increase (a real attack or a merchant integration change).
   Mitigate first (roll back the deploy, disable the rule, switch to the
   fallback policy), then analyse. Only then consider adversarial adaptation,
   which shows up as a shift concentrated in specific segments and feature
   patterns.
5. **Selective labels.** [expected] Transactions the model declines are never
   observed as fraud or not, so the next model trains only on approved
   traffic, under-represents the risky region and can drift into
   over-confidence about it (and analysts' labels are biased toward what
   rules flagged). Fix: a small, randomised approval holdout below an
   exposure cap (an explicit, costed exploration policy), propensity-weighted
   training/evaluation, and reject-inference methods treated with caution.
6. **Staff vs senior.** [open-ended] Example for ads: a senior answer builds a correct
   calibrated multi-task ranker with delayed-feedback handling and an A/B
   test; a staff answer also treats the system as a marketplace (advertiser
   ROI and retention as long-term metrics), designs budget-split experiments
   because user-split tests leak through shared budgets, defines the
   contracts between ranking, auction and pacing teams, and says who detects
   and owns a calibration incident.
