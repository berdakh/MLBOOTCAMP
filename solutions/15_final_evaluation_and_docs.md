# Solutions · 15 · Final evaluation & documentation

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

Numbers quoted below are from the run recorded when the notebook was built
(fresh artifact directory, CPU). Yours should match to about the third
decimal with the same library versions; the intervals tell you how much
anything else matters.

---

## 15-1 · `final_report`

✅ Reference code: [`code/nb15.py`](code/nb15.py) (`ex1`), graded in CI.

```python
def final_report(y, scores, threshold, groups, seed=0):
    y, scores = np.asarray(y), np.asarray(scores)
    pr_auc, lo, hi = bootstrap_ci(y, scores, metric=average_precision_score, groups=groups, seed=seed)
    c = confusion_at(y, scores, threshold)
    flagged = c["tp"] + c["fp"]
    return {"pr_auc": pr_auc, "pr_auc_ci": (lo, hi),
            "roc_auc": float(roc_auc_score(y, scores)),
            "precision": c["tp"] / flagged if flagged else float("nan"),
            "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
            "flagged": int(flagged)}
```

**Why `groups`.** The grader's synthetic data has 120 customers × 20 rows
whose labels and scores are strongly correlated within a customer. A row
bootstrap treats 2,400 rows as independent evidence and produces an interval
far too narrow; the grader checks that your interval is clearly wider than the
row-bootstrap one. On TaskFlow the same thing happens less dramatically
(a customer appears at up to five test snapshots).

**Common mistakes.**

* Returning `pr_auc_ci` as a list or NumPy array (the spec says tuple; JSON
  will turn it into a list later anyway, which is fine *there*).
* `precision` when nothing is flagged: dividing by zero raises or returns
  `nan` silently. Decide explicitly and document it.
* Using `>` instead of `>=` at the threshold: the course convention
  (`confusion_at`) is `score >= threshold`. On real data with rounded scores this
  can change `flagged` by dozens.

---

## 15-2 · `results_markdown`

✅ Reference code: [`code/nb15.py`](code/nb15.py) (`ex2`).

```python
def results_markdown(report):
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in report.items()]
    return "\n".join(lines) + "\n"
```

with `_fmt` mapping a 2-tuple to `[lo, hi]` (3 decimals), ints to `f"{v:,}"`,
floats to `f"{v:.3f}"`. Check `bool` **before** `int` (`True` is an `int` in
Python), and the tuple before anything else.

**Why bother.** A results table generated from the report dict cannot drift
from the numbers you computed. Notebook 15's last code cell checks that
mechanically: every decimal in the README must appear in `final_report.json`.

---

## 15-3 · Your 60-second pitch (written)

✍️ Personal. A strong pitch, using the numbers from the recorded run (yours
may differ slightly):

> "I built an end-to-end churn prediction project for a fictional
> subscription company, on a **synthetic** four-table extract with planted
> defects. I validated the data first: a region code pandas reads as missing,
> duplicate usage rows, and a status column that leaks the label. Features are
> point-in-time SQL in DuckDB, the split is by time with a 30-day embargo, and
> preprocessing lives inside the model pipeline. A logistic regression beat
> gradient-boosted trees on validation, so I kept it. On a test period touched
> once under a pre-registered plan, PR-AUC was 0.197 with a customer-level 95%
> interval of 0.17 to 0.22, against a positive rate of about 6%, and clearly
> above a three-feature baseline in a paired comparison. It is served by a tested
> FastAPI service from a versioned registry, with drift monitoring and a
> rehearsed rollback. With real data, the next step is a randomised outreach
> experiment, because ranking churners is not the same as saving them."

Checks: problem and user in the first sentence; "synthetic" said early; one
metric **with** interval, base rate and baseline; one engineering sentence; one
limitation or next step. About 150 words ≈ 60 seconds.

---

## 15-4 · Three limitations and your answers (written)

✍️ Personal. A strong answer:

| Limitation the interviewer raises | What you say |
|---|---|
| "It's synthetic, so the result means nothing." | "Right, and I say that up front: the numbers describe the method, not customers. What transfers is the process: data contracts that caught real-looking defects, point-in-time features, a pre-registered single test evaluation with customer-level intervals. My capstones apply the same process to public real data." |
| "PR-AUC of 0.2 is low." | "It's 3.3× the 5.9% base rate on test, and what matters is the decision: at the validation-chosen threshold, about 29% of calls reach a real churner versus 6% at random. Whether that is worth it depends on costs I had to assume ($150/$30). With a real team I'd agree those numbers first." |
| "You never showed the model *helps*." | "Correct: offline evaluation measures ranking, not the effect of calling. I'd run a randomised holdout among flagged accounts and measure retention uplift, and log contacts, because once we intervene, saved customers look like false positives and naive retraining would learn to stop flagging them." |

Other valid ones: test positive rate higher than validation (shift, explained
in 15.8); slices (e.g. `region=unknown` weaker); calibration drifted; contacting
customers has side effects.

---

## Checkpoint · 15

**1. Pre-registration.** Model version and file hash (v1, logistic
regression), the threshold chosen on validation (≈ 0.19), primary metric
(PR-AUC) and secondary metrics, the CI method (customer bootstrap, n_boot=500,
seed 0), the baselines (dummy prior, 3-feature logistic regression, the v2
challenger) and the decision rule (paired CI of the PR-AUC difference above 0).
Order matters because every decision taken after seeing test results makes
the test number optimistic (forking paths); a plan written first, with a
timestamp and hash, makes the report an estimate rather than a selection.

**2. Test > validation.** 🔬 Test PR-AUC ≈ 0.197 vs validation ≈ 0.163, but the
test positive rate is higher (≈ 0.059 vs ≈ 0.052), and PR-AUC rises with the base
rate. Normalised by the base rate (≈ 3.3 vs ≈ 3.1) and on ROC-AUC (0.740 vs
0.733) the two periods are close, and the intervals overlap. So: performance
held up in a period with more churn; it did not "improve".

**3. Customer-level and paired.** Rows from the same customer are correlated,
so resampling rows overstates how much independent evidence there is and gives
too-narrow intervals; resampling customers respects the dependence. *Paired*:
both models are scored on the same resampled customers, so customer-to-customer
difficulty cancels in the difference, and a small but consistent gain becomes
detectable. Two separate CIs that overlap do not show "no difference".

**4. "Worth $X".** The $150 and $30 are assumptions (call cost, a 25% save
rate, $600 margin), not measurements; nobody was called, so the effect of
calling is unknown; and the data is synthetic. The honest statement is
"under assumed costs of $150/$30, the policy has positive expected value; an
experiment would measure the real value".

**5. Feedback loop.** Customers the calls retained no longer churn, so they
become negatives that the model had scored high: they look like false
positives, and a model retrained on them learns to stop flagging the people
the intervention helps. Log every contact (who, when, model version, score),
keep a random never-contacted holdout for evaluation and retraining, or model
the treatment explicitly (uplift modelling).

**6. Three limitations.** Synthetic data; offline evaluation only (no causal
effect measured); interventions have side effects and create feedback loops.
Also acceptable: assumed costs, label delay, base-rate shift between periods.
