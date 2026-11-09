# Solutions · Chapter 55 · ML platform design

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `feature_store_storage_gb`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def feature_store_storage_gb(entities, features, bytes_per, versions, replication):
    if min(entities, features, bytes_per, versions, replication) < 0:
        raise ValueError("all inputs must be non-negative")
    return float(entities * features * bytes_per * versions * replication / 1e9)
```

**Out loud:** 200M × 400 = 8 × 10¹⁰ values; × 4 B = 3.2 × 10¹¹ B = 320 GB;
× 2 versions × 3 replicas = **1,920 GB ≈ 1.9 TB**. Say "decimal GB" (or
convert to GiB, about 7% smaller in number, and say so). In an interview,
follow the number with what it implies: ~2 TB of RAM-class storage is a
cluster, not a box, so the conversation moves to which features really need
to be online, and to cheaper tiers for dormant users. The function ignores
key and index overhead on purpose; `ds.online_store_gb` adds them
(× 1.3–2 is a common planning range; measure the actual store).

---

## Exercise 2 · `gpu_cost`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def gpu_cost(hours, gpus, price):
    if min(hours, gpus, price) < 0:
        raise ValueError("hours, gpus and price must be non-negative")
    return float(hours * gpus * price)
```

64 GPUs reserved for a month (730 h) at a placeholder 2.0 per GPU-hour is
93,440. The arithmetic is trivial; the interview value is in naming what the
price is (on-demand, reserved, spot; the provider's current list, not a
number from memory) and that reserved capacity is paid whether used or not,
which is Exercise 3.

---

## Exercise 3 · `utilization_waste`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex3`).

```python
def utilization_waste(hours, gpus, price, utilization):
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be a fraction in (0, 1], got {utilization}")
    total = gpu_cost(hours, gpus, price)
    return {"total_cost": total, "useful_cost": total * utilization,
            "wasted_cost": total * (1 - utilization),
            "cost_per_useful_gpu_hour": price / utilization}
```

`cost_per_useful_gpu_hour` is the most persuasive number: a team at 20%
utilisation pays five times the list price for every hour of real work.
"Utilisation" needs a definition before it is a KPI: allocated vs idle
(scheduler view), or SM busy / MFU (hardware view). A job can hold a GPU at
100% allocation and 15% MFU because it is input-bound (Chapter 39). State
which one is meant.

---

## Exercise 4 · Written ADR

[open-ended] Personal. A good ADR-002 for *"GPU quotas: strict per-team reservations vs
a shared preemptible pool"* looks like this (assumptions labelled):

* **Context.** 160 reserved GPUs across four teams, 48% idle in the
  illustrative table (111k of 234k a month at a placeholder price). Research
  and LLM-apps teams are bursty; ads ranking runs daily retrains on a fixed
  schedule.
* **Options.** (A) strict reservations (predictable, simple, wastes idle
  capacity); (B) one fully shared queue (high utilisation, but deadlines
  become unpredictable and one team can starve others); (C) **guaranteed
  quota + borrowable burst + preemption** of borrowed capacity, requiring
  checkpoint/resume for any job that borrows.
* **Criterion that separates them.** Can the critical scheduled retrains
  still meet their deadline at p95 queueing time while idle share falls?
* **Decision.** C, with showback first and chargeback after one quarter;
  borrowing allowed only for jobs that checkpoint at least every 30 minutes.
* **Consequences.** Every training template must checkpoint (golden path
  change); preemption adds lost work (≤ 30 min per preemption); scheduler
  configuration becomes a contract owned by the platform team.
* **Revisit when.** Idle share stays above 30% for two months; the p95 queue
  time for scheduled retrains exceeds 1 hour; preemptions exceed N per job-day.

What graders of ADRs look for: a real alternative (not a straw man), the
deciding criterion stated before the decision, at least one number with its
assumptions, and measurable revisit triggers.

---

## Checkpoint answers

1. **Feature platform for 50 teams.** [expected] The unit is a *feature group*: a
   transformation in code, an entity key, an event timestamp, an owner, a
   freshness SLO. The platform materialises the same definition to an
   offline columnar store (full history with event timestamps) and an online
   KV store (latest value); consistency comes from one definition and one
   computation path where possible, plus a **skew monitor** that compares
   logged online values with an offline recompute for a sample of requests.
   Point-in-time correctness: training sets are built by as-of joins on
   `(entity, label_time)` against the history, never against a current
   snapshot; backfills recompute history with the same code and are
   versioned. Then name the contracts: who owns a feature, how it is
   deprecated, how consumers are tracked (lineage).
2. **Online store sizing.** [verified] 200M × (400 × 4 B) = 320 GB raw; with 16-byte
   keys, 1.5× overhead and 3 replicas ≈ **1.45 TB** (section 55.1). Halving it: keep
   only features read by a production model (usage-based GC; costs a
   governance process); fp16/int8 or bucketed values (costs precision; check
   the model is insensitive); a hot tier for recently active users with
   defaults or a slower tier for the rest (costs latency/quality for dormant
   users); 2 replicas instead of 3 (costs availability).
3. **GPU bill doubled at 45% utilisation.** [expected] Measure first: spend by team
   and job type; allocation-idle vs low-MFU (different fixes); interactive
   sessions left open; queue times. Policies: showback, idle reaping,
   guaranteed quota + preemptible burst pool, right-sizing small jobs to small
   cards, checkpointing as a requirement for preemptible capacity. Who must
   agree: team leads (their quotas), finance (chargeback), the platform team
   (scheduler), leadership (priorities when the pool is contended). The
   organisational agreement is the hard part and is what makes it a staff
   question.
4. **Retrained model to 100%.** [verified] (arithmetic) Training job → candidate
   registered with snapshot id, code version, signature, eval report → CD
   gates: offline comparison against the champion on the same snapshot with
   intervals and slices, latency/memory test in the serving image → shadow →
   canary → A/B (or ramp with guardrails) → 100%, previous version kept warm
   for rollback. A 1% canary at 2,000 peak QPS sees 20 QPS; detecting a +20%
   rise in a 0.5% error rate needs ~85,900 requests per arm (~72 minutes),
   while a doubling is visible in ~4 minutes. So the canary catches crashes,
   latency blow-ups and large regressions; small quality effects need the
   A/B test.
5. **Delayed labels.** [expected] Compare cohorts at **equal label maturity** (e.g.
   recall on predictions made 30 days ago vs the same metric for the
   previous model at its own 30-day mark), plot metrics by prediction date
   with a maturity curve, and rely on leading indicators meanwhile: input and
   score drift, early proxy labels, a small human-reviewed sample. With an
   exponential delay of mean 20 days only 30% of labels exist after a week,
   so a "drop" in last week's recall is expected.
6. **Buy vs build a feature store.** [expected] Buy when the requirements fit the
   product's documented limits, the team cannot staff a stateful on-call, and
   features are not what differentiates the organisation. Put a thin API in front so the
   decision is reversible (ADR-001). Reverse it when per-unit cost at the organisation's
   scale becomes a top cost line, latency/throughput needs exceed the
   product's limits, compliance needs (residency, deletion audit) are unmet,
   or the team can staff the on-call. State the trigger as a number.
