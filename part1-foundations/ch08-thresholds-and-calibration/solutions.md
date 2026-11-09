# Solutions · Chapter 8 · Thresholds & calibration

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `top_k_per_snapshot`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def top_k_per_snapshot(df, scores, k):
    s = pd.Series(np.asarray(scores, dtype=float), index=df.index)
    rank = s.groupby(df["snapshot_date"]).rank(ascending=False, method="first")
    return (rank <= k).to_numpy()
```

**Why it works.** The scores are put on `df`'s index, so grouping by
`df["snapshot_date"]` aligns row by row whatever the order or index. `rank`
returns a value per row in the original order (unlike sorting), and
`method="first"` breaks ties deterministically so exactly k rows are chosen.
Snapshots with fewer than k rows get every row flagged automatically.

**Common mistakes.** `sort_values(...).groupby(...).head(k)` and then
returning the flags in *sorted* order; using `np.asarray(scores)` with a
`.loc` index lookup; `method="min"` (ties can push the count above k).

---

## Exercise 2 · `best_threshold`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def best_threshold(y, scores, value_tp, cost_fp):
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    last_of_run = np.r_[s[1:] != s[:-1], True]      # cut only between distinct scores
    value = np.where(last_of_run, tp * value_tp - fp * cost_fp, -np.inf)
    i = int(np.argmax(value))
    return float(s[i]), float(value[i])
```

**Why it works.** Sorting by score once, flagging the top i rows corresponds
to the threshold `s[i]`; cumulative sums give TP and FP for every i in
O(n log n). Cutting in the middle of a run of tied scores is not achievable
with `score >= t`, so those positions are excluded. This is exhaustive,
so it is at least as good as `best_threshold_by_value`, which checks 400
quantiles for speed.

**Push further.** "Call nobody" (value 0) is also a valid decision; add it
as a candidate if all values are negative.

---

## Exercise 3 · Explaining the threshold to a manager (written)

[open-ended] Personal. A good answer (numbers [expected] from our run):

> "The model gives each account a risk score. We call an account when its
> score is above about 0.19, roughly a one-in-five chance of cancelling. That
> sounds low, but a call costs about $30 and, by our current estimate,
> reaching someone who was about to leave is worth about $150 on average, so
> a call pays for itself once the odds are better than about one in six. At
> that line we would make about 64 calls a month and roughly one in four
> would reach someone who was about to cancel. If calls cost twice as much,
> the line moves up to about one-in-three odds: far fewer calls, and the
> programme only just pays. The biggest uncertainty is the $150, which
> assumes a call saves one customer in four; we should test that first,
> because the right number of calls depends on it more than on the model."

What makes it good: odds instead of jargon, the costs stated as assumptions,
the volume of calls (what the manager actually manages), and the next step.

---

## Checkpoint answers

**1. Break-even.** Call when p × VALUE_TP > (1 − p) × COST_FP, i.e.
p > COST_FP / (VALUE_TP + COST_FP) = 30/180 = 1/6. Valid only if the scores
are **calibrated** probabilities and the costs are right (and independent of
who is called).

**2. Top-k per month.** Capacity is a hard limit in calls per month; a global
threshold makes the volume float with the score distribution (seasonality,
drift), so some months overflow and some waste capacity. Top-k matches the
constraint exactly and always uses the best k.

**3. Sensitivity.** To the save rate inside VALUE_TP: halving it moved the
best threshold from ≈ 0.19 to ≈ 0.45 and value to near zero; doubling it
moved it to ≈ 0.10. Measure it with a randomised holdout: among flagged
customers, call a random half, and compare 30-day churn.

**4. Calibration and metrics.** A monotone calibration map (sigmoid) keeps
the order of scores, so PR-AUC and ROC-AUC are unchanged (isotonic creates
ties, so they can move slightly). The best *threshold value* changes (it is
on the probability scale), and so does the expected value of a rule
written as a fixed probability such as 1/6. The set of accounts flagged at a
*re-optimised* threshold can stay the same.

**5. Where to fit calibration.** On data the model was not trained on and
that comes after its training data in time (here the last three train months,
with the base model fitted on earlier months), or with time-ordered CV.
Fitting it on validation and then evaluating on validation makes the
reliability plot look perfect by construction and spends validation on a
fitted step, so it no longer estimates future performance.
