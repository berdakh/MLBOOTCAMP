# Solutions · Chapter 7 · Model selection

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `decide`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def decide(y, scores_a, scores_b, groups):
    d = paired_bootstrap_diff(y, scores_a, scores_b, groups=groups, n_boot=1000, seed=0)
    if d["ci_low"] > 0:
        return "B"
    if d["ci_high"] < 0:
        return "A"
    return "no clear winner"
```

**Why paired and clustered.** Paired: both models are scored on the same
resample each time, so noise common to both (an easy month, a lucky set of
churners) cancels in the difference. Clustered: whole customers are
resampled, so correlated rows of one customer are not counted as independent
evidence.

**Common mistakes.**

* Sign convention: `paired_bootstrap_diff` returns metric(B) − metric(A).
  Swapping it makes every verdict backwards; check on a case you can see by
  eye (a model with no signal vs one with signal).
* Comparing two separate intervals ("they overlap, so no difference"). That
  test is too conservative; the paired interval is the right one.
* Returning "B" when `diff > 0` alone. A positive point estimate is not a
  decision.

---

## Exercise 2 · `budgeted_search`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def budgeted_search(param_grid_list, evaluate_fn, budget, seed=0):
    rng = np.random.default_rng(seed)
    n = min(budget, len(param_grid_list))
    picks = rng.choice(len(param_grid_list), size=n, replace=False)
    rows = []
    for i in picks:
        params = dict(param_grid_list[int(i)])
        rows.append({**params, "score": float(evaluate_fn(params))})
    trials = pd.DataFrame(rows)
    best = dict(param_grid_list[int(picks[trials["score"].idxmax()])])
    return best, trials
```

**Key decisions.** Sampling **without replacement** (re-evaluating a config is
wasted budget); `min(budget, len(...))` handles small grids; a **fixed seed**
so the search itself is reproducible; and every trial kept, not just the
winner. In practice, add `seconds` and fold-level scores to each row, and
write the table to disk (Chapter 7 writes `artifacts/trials.csv`).

**Push further.** Random search beats grid search at a fixed budget when only
a few hyperparameters matter (Bergstra & Bengio, 2012): a grid wastes
evaluations varying unimportant ones. Successive halving
(`HalvingRandomSearchCV`) spends the budget more cleverly by dropping bad
configs after cheap evaluations.

---

## Exercise 3 · "Why didn't you use XGBoost?" (written)

[open-ended] Personal. A strong 60-second answer, with numbers from the notebook ([expected]
yours may differ slightly):

> "I did try gradient-boosted trees: sklearn's `HistGradientBoosting`, which
> is the same algorithm family as XGBoost and LightGBM. With rolling-origin
> CV on the training year, the boosted trees scored about 0.156 PR-AUC
> against 0.186 for logistic regression, and they lost on every one of the
> four folds. On the validation quarter it was 0.135 vs 0.163, and the paired
> customer-level bootstrap put the difference at about +0.03 with a 95%
> interval from roughly 0.004 to 0.05. A budgeted search of eight tree
> configurations didn't close the gap; deeper trees were worse. That fits the
> data: about 1,400 positives, features already engineered as rates and
> trends, and a risk that rises smoothly with them. XGBoost specifically
> wouldn't change that conclusion, and the linear model is also calibrated,
> faster and easier to explain. I'd revisit it if we add raw features with
> interactions, like renewal timing by plan, or once we have several times
> more churners."

What makes it strong: it reframes the question (the family *was* tried), it
cites the comparison and its uncertainty, it gives a mechanism, and it names
the condition that would change the decision. Weak answers: "trees overfit"
with no number, or "I didn't have time".

---

## Checkpoint answers

**1. B is 0.03 higher.** A paired bootstrap interval for the difference,
resampling customers; consistency across rolling folds; confirmation that
the comparison used the same rows, the same split and no tuning on the
evaluation set. Also whether 0.03 matters at the decision threshold
(precision at the calls you will actually make).

**2. Customer as the unit.** A customer contributes several correlated rows;
resampling rows treats them as independent and understates uncertainty. On
this validation set it barely mattered because each customer has at most
three rows and at most one positive (churn is terminal), so there is little
within-customer correlation in what drives PR-AUC. The tripled-rows demo
shows the mechanism: the row interval shrank by about √3 for copies that
carry no new information, the cluster interval did not.

**3. Overlapping intervals.** No. Two marginal intervals ignore that the
models are evaluated on the same rows, so shared noise is counted twice.
Compute the paired bootstrap interval of the difference.

**4. Why the linear model was better.** A smooth, roughly monotone risk; few positives (≈ 1,400),
so tree leaves estimate noisy rates; non-linearity already engineered into
the features; synthetic data generated from a smooth score. Reversing
property: strong interactions or thresholds in the true relationship (e.g.
churn spikes at renewal only for some plans), many raw features, or much
more positive data.

**5. Best of 16 is optimistic.** The maximum of noisy estimates is biased
upwards (winner's curse); the fold-to-fold spread (≈ 0.02) was larger than
most differences between configurations. Protections: a budget fixed in
advance, all trials recorded, preferring the simpler config on a plateau,
and an untouched evaluation period (validation for this choice, test once at
the end).
