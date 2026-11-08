# Solutions · Chapter 24 · Learning to rank

Attempt each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `pairwise_accuracy`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def pairwise_accuracy(scores, labels, groups):
    s, y, g = (np.asarray(x) for x in (scores, labels, groups))
    good, total = 0.0, 0
    for key in np.unique(g):
        m = g == key
        sg, yg = s[m], y[m]
        pairs = yg[:, None] > yg[None, :]          # i more relevant than j
        d = (sg[:, None] - sg[None, :])[pairs]
        total += int(pairs.sum())
        good += float((d > 0).sum() + 0.5 * (d == 0).sum())
    return good / total if total else float("nan")
```

**Why it works.** Broadcasting builds all ordered pairs of a group at once;
`yg[:, None] > yg[None, :]` keeps each unordered pair with different labels
exactly once (in the direction "more relevant first").

**Complexity.** O(Σ n_g²) time. The `g == key` mask makes this version
O(n · #groups) as well; `ltr.pairwise_accuracy` sorts once and slices instead,
which matters with 30,000 groups.

**Common mistakes.** Counting pairs across groups (the grader's
`[0, 1, 0, 1]` case); counting each pair twice; treating ties as wrong (a
constant scorer should score 0.5, not 0); returning 0 instead of NaN when no
pair exists (that would silently drag an average down).

---

## Exercise 2 · `make_groups`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def make_groups(df, group_col):
    out = df.sort_values(group_col, kind="stable")
    sizes = out.groupby(group_col, sort=False).size().to_numpy()
    return out, sizes
```

**Why `kind="stable"`.** The default quicksort may reorder rows *within* a
group. LightGBM does not depend on the order inside a group, but the
evaluation code (and any per-row arrays kept alongside, such as
positions or IPS weights) does. **Why `sort=False`.** After sorting, the
appearance order *is* the sorted order; `sort=False` makes that explicit
and cheaper.

**Common mistake.** `df.groupby(col).size()` on the *unsorted* frame, passed
together with the unsorted rows: the sizes are right, the rows are not
contiguous, and LightGBM trains on meaningless groups without an error.

---

## Exercise 3 · Clicks and purchases in one label (written)

[open-ended] One well-reasoned answer:

* **Graded label**: 0 = shown, not clicked; 1 = clicked; 3 = purchased
  (purchase implies click). LambdaMART uses gain 2^label − 1, so a purchase
  is worth 7 clicks with these grades; set the grade (or LightGBM's
  `label_gain`) so the gain ratio matches the business value ratio, e.g.
  the average margin of a purchase relative to the long-term value of a click.
* **Alternatively**, keep the click label and add a **group weight** or a
  second model: a click model times a purchase-given-click model, multiplied
  at serving (the classic CTR × CVR decomposition; it allows the
  trade-off to be tuned at serving time without retraining).
* **Choosing the parameters**: sweep the purchase gain (1, 3, 7, 15) and plot
  validation clicks-NDCG against purchases-NDCG: a trade-off curve. Pick the
  point with the product owner; the offline metric should not choose it alone.
* **Evaluating without misleading results**: purchases are also
  position-biased (they require a click), so use IPS or random traffic; use
  request-level bootstrap intervals, because purchases are rare (a few per
  hundred requests here); check the result by segment (cheap vs expensive
  categories: price-dependent purchase probability can make the model push
  cheap items); and confirm online. The offline metric is a filter, the A/B
  test is the decision.

---

## Checkpoint answers

**1. Objectives.** Pointwise: per-item loss (log loss, MSE). Pairwise:
correct order of pairs (RankNet's logistic pair loss). Listwise: a
list-level metric (NDCG) via LambdaRank gradients or a softmax over the
list. Pointwise when calibrated probabilities are needed (ads auctions, CTR ×
CVR × bid, thresholds) or when labels are absolute; pairwise/listwise when
only the order matters and lists are long with graded labels. A pointwise
model can rank well while miscalibrated because ranking only needs a
monotone transform of the truth *within* each group. Calibration matters
when scores are combined with other scores, compared across groups, or
turned into decisions.

**2. LambdaRank.** For a pair (i, j) with y_i > y_j, the RankNet gradient
is λ_ij = −σ / (1 + exp(σ (s_i − s_j))). LambdaRank multiplies it by
|ΔNDCG_ij|, the NDCG change from swapping i and j in the current ranking;
item i's gradient is the sum of its λ's. NDCG is piecewise constant in the
scores (non-differentiable), but only *a direction* in which to move the
scores is needed; empirically these "lambdas" are gradients of an implicit smooth
loss that upper-bounds NDCG loss (LambdaLoss formalises this), and they
focus effort at the top of the list.

**3. The group in a feed.** The request (impression set) shown to one user
at one time: items in it competed for the same attention. Grouping by
user-day mixes different candidate sets, contexts and times, so the model
learns to compare items that were never alternatives, and position effects
from different requests get mixed; bigger groups also make pair counts (and
training time) grow quadratically.

**4. 0.95 vs 0.70.** In order: (a) leakage, i.e. recompute features from a
truncated log and compare (6.2), check that no feature uses same-day or
label data, check the train/valid split is by time and by request; (b) is
the training-week score in-sample? (it should be evaluated out-of-sample on a
held-out part of the training period); (c) label/metric definition identical
in both weeks; (d) drift: new items/users (cold start), feature coverage
(share of rows with no history), a policy or UI change that moved
propensities; (e) noise: bootstrap intervals.

**5. Position as a feature.** (a) Train with the logged position as a
feature, serve with a constant (position 0); (b) a separate shallow
"position tower" (bias term) added to the logit in training and dropped at
serving; or no feature plus IPS weights. You know it helped only by an
unbiased evaluation: IPS/random-traffic offline metrics or an A/B test. In
6.6 the position feature took 64% of the gain yet changed the ranking by
almost nothing, because no other feature correlated with position.

**6. 0.002 apart.** Within noise (bootstrap it). Prefer the simpler model if
it is genuinely equal: logistic regression is cheaper to serve, easier to
debug and to calibrate; but check latency at the actual candidate count, the
segments (new users, cold items, long-tail categories), calibration if
scores feed other systems, robustness to missing features, and run an A/B
test (or interleaving, Chapter 30) because offline equality does not imply
online equality.
