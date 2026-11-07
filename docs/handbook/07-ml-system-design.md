# ML system design

An ML system design interview gives you a vague product problem and 35–45
minutes. It tests whether you can turn ambiguity into a system that makes good
decisions, can be evaluated honestly, and can be operated. This chapter gives a
framework and one worked sketch.


---

## What is being assessed

- **Problem framing:** do you find the decision and the success measure before
  reaching for a model?
- **Data thinking:** where labels come from, how late, how biased; what is
  available at prediction time.
- **Evaluation:** offline metrics that match the decision; an online test that
  measures the outcome.
- **Engineering judgment:** batch vs online, latency, cost, failure modes,
  monitoring, rollback.
- **Communication:** structure, signposting, checking in, trading off out loud.

Interviewers usually care more about the first and last than about which model
you name.

## A framework: eight steps

Spend roughly the time shown in a 40-minute interview. Say each step's name
as you start it so the interviewer can follow, and redirect you.

| # | Step | Minutes | Questions to answer |
|---|---|---|---|
| 1 | **Clarify the decision** | 4 | Who acts on the prediction, and how? What is the unit (user, order, transaction)? How often? What are the constraints (latency, volume, capacity, regulation)? |
| 2 | **Define success** | 3 | Business metric (what moves if this works). Model metric (what you can measure offline). Guardrails (what must not get worse). |
| 3 | **Labels and data** | 6 | What is the label, exactly? When is it known? How is it biased (only reviewed items get labels)? What features exist at prediction time? Point-in-time correctness. |
| 4 | **Baseline and model** | 5 | A rules or simple-model baseline first. Then a model family that fits the data, latency and explainability needs. Why not something simpler? |
| 5 | **Offline evaluation** | 5 | Split that matches use (time, group). Metrics for imbalance. Threshold from costs or capacity. Slices. Uncertainty. |
| 6 | **Serving** | 6 | Batch, online, or streaming. Feature computation online vs offline and how to avoid skew. Latency budget. Fallback when the model is unavailable. |
| 7 | **Monitoring and iteration** | 6 | Service, data and performance layers. Delayed labels. Feedback loops. Retraining. Rollback. Online test (A/B, canary, shadow). |
| 8 | **Risks and trade-offs** | 3 | Fairness and subgroup errors, privacy, abuse, what you would do with more time. |

Leave a couple of minutes for questions. If the interviewer pushes deeper on
one step, follow them; the framework is a map, not a script.

## Habits that score well

- **Ask before assuming, then state the assumption.** "I'll assume about a
  million transactions a day and a 100 ms budget; tell me if that's off."
- **Draw the pipeline** early: data sources → features → model → decision →
  feedback. Then talk over it.
- **Name the baseline.** It shows you know complexity has to earn its place.
- **Name the failure modes** before being asked: leakage, skew, delayed and
  biased labels, feedback loops, cold start.
- **Quantify roughly.** Rows per day, positives per day, requests per second,
  reviewer capacity. Order-of-magnitude arithmetic is expected.
- **Trade off out loud.** "A larger model might gain a little recall, but it
  would push p99 latency past the budget, so I'd start with…"

## Worked sketch: card-transaction fraud screening

*Prompt: "Design a system to detect fraudulent card transactions for a
payments company."* The numbers below are **assumptions for illustration**,
stated as you would state them in an interview.

**1. Clarify the decision.** At authorisation time, each transaction is
approved, declined, or sent for step-up verification or manual review. Assume
the scoring service must answer within a strict latency budget (say 50–100 ms
at p99) because it sits in the payment path. Manual review capacity is limited.

**2. Define success.** Business: fraud losses avoided minus the cost of
false declines (lost sales, angry customers) and review effort. Offline model
metric: precision and recall at the operating thresholds, PR-AUC for ranking,
weighted by transaction amount as a secondary view. Guardrails: false-decline
rate, review queue size, latency.

**3. Labels and data.** Labels come mostly from chargebacks and confirmed
fraud reports, which arrive **weeks** after the transaction. Labels are
**biased by the current system**: declined transactions never get a
chargeback, so we never learn whether they were fraud. Features: transaction
fields (amount, merchant category, channel), card and account history
(velocity counts over the last minute, hour, day), device and location
signals, merchant risk. Velocity features must be computed **as of the
transaction time** from an event stream, the same way online and offline.

**4. Baseline and model.** Baseline: the existing rules (velocity limits,
blocked merchants), which also give a comparison point. Model: gradient-boosted
trees on tabular features are a strong, fast default. Keep rules as a layer for
hard policy decisions and new attack patterns the model has not seen.

**5. Offline evaluation.** Time-based split with an embargo at least as long as
the label delay, otherwise recent training transactions have immature labels
(fraud not yet reported looks legitimate). Thresholds chosen from the cost of
fraud vs false declines, and from review capacity for the middle band.
Slices by merchant category, region, new versus established cards. Bootstrap
intervals, resampling by card or account, since transactions of one card are
correlated.

**6. Serving.** Online scoring service called by the authorisation system.
Velocity features from a low-latency store updated by a stream processor;
the same feature code backfills training data to avoid skew. If the model is
unavailable or too slow, **fall back to the rules** rather than blocking
payments; the fallback is part of the design, not an afterthought.

**7. Monitoring and iteration.** Service: latency percentiles, timeouts,
fallback rate. Data: velocity feature distributions, unseen merchants, score
distribution, decline and review rates per day. Performance: matured-cohort
precision and recall once chargebacks arrive. **Feedback loop:** declined
transactions produce no labels. Mitigate with a small, randomised
exploration group that is approved or reviewed regardless of score, so
unbiased labels keep arriving. Release new models in shadow first, then as a
canary, with instant rollback.

**8. Risks and trade-offs.** Adversaries adapt, so performance decays faster
than in most problems: plan for regular retraining and rule updates. False
declines may fall unevenly across customer groups or regions: monitor slice
rates. Privacy: device and location data need access controls and retention
limits. With more time: graph features linking cards, devices and merchants;
cost-sensitive training weighted by amount.

### What the sketch deliberately shares with TaskFlow

| Concern | TaskFlow churn | Fraud screening |
|---|---|---|
| Label delay | 30 days | Weeks (chargebacks) |
| Point-in-time features | Monthly snapshots in SQL | Per-transaction velocity from a stream |
| Correlated rows | Customer across snapshots | Card across transactions |
| Threshold from costs | $150 vs $30 (assumed) | Fraud loss vs false-decline cost (assumed) |
| Feedback effect | Called customers may be retained | Declined transactions get no label |
| Serving | Batch is enough | Online, strict latency, with fallback |

The same ideas recur across nearly every ML system design question. Learning
them once in depth, on a project you built, is more useful than memorising
designs for twenty products.

## Practising

- Notebook 17 has its own worked design and a scored mock interview.
- The [interview question bank](../guides/interview-questions.md#ml-system-design)
  has more prompts.
- Practise aloud, timed, and record yourself. Score with the
  [readiness rubric](../guides/readiness-rubric.md). Reading designs is not
  the same skill as producing one under time pressure.
