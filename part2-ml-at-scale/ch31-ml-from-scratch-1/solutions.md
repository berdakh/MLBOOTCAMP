# Solutions · Chapter 31 · ML from scratch I

Try each exercise before reading its answer, with a timer running. Markers:
✅ verified by running · 🔬 expected, depends on your run · ✍️ personal / open-ended.

Reference code for every exercise: [`code/nb13.py`](solutions.py), graded in CI
by `tests/test_solutions.py` against the graders in
[`src/shoprank/grading/nb13.py`](../../src/grading/ch31.py).

---

## Exercise 1 · `fit_logreg` (logistic regression, batch GD + L2)

✅ Reference: `ex1` in [`code/nb13.py`](solutions.py).

```python
def fit_logreg(X, y, lr=0.1, epochs=1000, l2=0.0):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)               # stable, piecewise
        err = p - y
        w -= lr * (X.T @ err / n + l2 * w)   # intercept not penalised
        b -= lr * err.mean()
    return w, float(b)
```

**Why it matches sklearn.** sklearn minimises `C·Σᵢ logloss + ½‖w‖²`. Dividing
by `C·n` gives `mean logloss + ‖w‖²/(2Cn)`, i.e. λ = 1/(Cn). The objective is
strictly convex with L2, so both optimisers find the same unique minimum; 3,000
GD steps at lr = 0.5 on the grader's standardised features land within ~1e-8 of sklearn's coefficients (measured).

**Complexity.** O(n·d) per epoch; O(epochs·n·d) total; O(n + d) memory.

**Edge cases.** Separable data with λ = 0: the minimum is at infinity, weights
grow every epoch (with a stable sigmoid they stay finite for a finite number
of epochs; the grader checks no NaN). All labels equal: the intercept drifts
towards +∞ (or −∞); L2 does not stop it because the intercept is unpenalised.
Unscaled features: the largest feature's curvature limits the stable learning
rate; standardise.

**Common mistakes.** Penalising the intercept; using `sum` in one place and
`mean` in another (gradient off by n); `1/(1+np.exp(-z))` overflowing for
z ≪ 0 (a warning, not wrong, but say it); updating `w` in place before
computing `b`'s gradient from the old `w` (harmless here, a bug in
coordinate-wise code).

---

## Exercise 2 · `softmax_grad`

✅ Reference: `ex2` in [`code/nb13.py`](solutions.py).

```python
def softmax_grad(W, X, y):
    n = X.shape[0]
    Z = X @ W
    Z = Z - Z.max(axis=1, keepdims=True)
    logP = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    loss = -logP[np.arange(n), y].mean()
    P = np.exp(logP)
    P[np.arange(n), y] -= 1.0
    return float(loss), X.T @ P / n
```

**Why `P − Y`.** With `L = −Σ_c Y_c log softmax(z)_c` and `Σ_c Y_c = 1`,
`∂L/∂z_k = softmax(z)_k − Y_k` (derivation in 14-7). Chain rule through `Z = XW`
gives `dW = Xᵀ (P − Y) / n`.

**Complexity.** O(n·d·C) time; O(n·C) memory.

**Common mistakes.** Building one-hot with a Python loop; `log(softmax)`
instead of log-softmax (NaN for logits ≈ 1000; the grader multiplies W by 1000
to catch it); forgetting `/ n` in the gradient while the loss is a mean
(gradient check: relative error ≈ 1 − 1/n).

---

## Exercise 3 · `kmeans` with k-means++

✅ Reference: `ex3` in [`code/nb13.py`](solutions.py).

Key lines:

```python
centers = [X[rng.integers(n)]]
d2 = ((X - centers[0]) ** 2).sum(1)
for _ in range(1, k):
    idx = rng.choice(n, p=d2 / d2.sum())          # D^2 sampling
    centers.append(X[idx])
    d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(1))
# Lloyd: assign by argmin of the (n, k) distance matrix, recompute means,
# keep the old center if a cluster is empty, stop when labels do not change.
```

✅ On the grader's five blobs, uniform random init plus Lloyd gave a median
inertia of ~9,100 against sklearn's ~790 (it merged blobs); the k-means++
reference matches sklearn.

**Complexity.** Init O(n·k·d); each Lloyd iteration O(n·k·d) time and O(n·k)
memory for the distance matrix.

**Edge cases.** All points identical → `d2.sum() == 0`, fall back to a uniform
pick; empty clusters; k > distinct points (some centers coincide); the result
depends on the seed (run several seeds, keep the lowest inertia — that is
sklearn's `n_init`).

---

## Exercise 4 · `knn_predict`

✅ Reference: `ex4` in [`code/nb13.py`](solutions.py).

```python
d2 = (Xte**2).sum(1)[:, None] - 2 * Xte @ Xtr.T + (Xtr**2).sum(1)[None, :]
nn = np.argpartition(d2, k - 1, axis=1)[:, :k]          # k smallest, unordered
classes, y_idx = np.unique(y_train, return_inverse=True)
votes = np.zeros((len(Xte), len(classes)), int)
np.add.at(votes, (np.arange(len(Xte))[:, None], y_idx[nn]), 1)
return classes[votes.argmax(1)]                          # first max = smallest label
```

(`argpartition` needs `k < n_train`; for `k == n_train` take all indices.)

**Complexity.** O(m·n·d) for distances (one GEMM), O(m·n) for selection, O(m·n)
memory — chunk the test set if m·n·8 bytes does not fit.

✅ A version with two Python loops over test and training points failed the
grader's timing at 1,600 training points (3.3 s > 2 s budget).

---

## Exercise 5 · `roc_auc` with ties

✅ Reference: `ex5` in [`code/nb13.py`](solutions.py).

```python
order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]   # start of each tie run
ends = np.r_[starts[1:], len(s)]
ranks = np.empty(len(s)); ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
auc = (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

**Why it works.** The sum of the positives' ranks minus its minimum possible
value counts, for every positive, how many negatives rank below it: that is the
Mann-Whitney U statistic, and U / (n₊n₋) is the probability a random positive
outscores a random negative. Average ranks make each positive-negative tie
contribute exactly ½.

**Complexity.** O(n log n) time, O(n) memory.

✅ Ordinal ranks (no tie handling) gave 0.7761 against sklearn's 0.7765 on the
grader's rounded scores; the pairwise O(n₊n₋) version is correct but failed
the timing check at n = 16,000 (2.8 s).

---

## Exercise 6 · `ndcg_at_k`

✅ Reference: `ex6` in [`code/nb13.py`](solutions.py).

```python
order = np.argsort(-scores, kind="stable")
dcg = lambda rel: ((2.0 ** rel[:k] - 1) / np.log2(np.arange(2, min(k, rel.size) + 2))).sum()
ideal = dcg(np.sort(labels)[::-1])
return dcg(labels[order]) / ideal if ideal > 0 else 0.0
```

**Conventions that change the number** (say them before coding): gain
`2^rel − 1` vs `rel`; tie-breaking among equal scores (stable here; some
libraries break ties pessimistically); 0 vs "skip" for queries with no relevant
item (`shoprank.metrics.grouped_ranking_metrics` skips them by default and
reports how many); IDCG over the full candidate set vs the shown list.

**Complexity.** O(n log n), or O(n + k log k) with `np.argpartition`.

---

## Checkpoint answers

1. **Gradient of BCE w.r.t. the logit.** L = −y log σ(z) − (1−y) log(1−σ(z)).
   With σ'(z) = σ(1−σ): ∂L/∂z = −y(1−σ) + (1−y)σ = σ(z) − y. Softmax + CE gives
   p − y for the same reason; squared loss with an identity link gives ŷ − y.
   All three are generalised linear models with the **canonical link**, for
   which the gradient of the negative log-likelihood w.r.t. the natural
   parameter is always "mean minus observation". ✅ (13-1, 13-2 gradient checks)
2. **NaN in softmax.** (a) Overflow: `exp` of logits > ~709 → inf/inf. Fix:
   subtract the row max / use log-softmax. (b) `log(0)` in the loss when a
   probability underflows to 0. Fix: compute CE from logits
   (`logsumexp(z) − z_y`). Other real causes: a learning rate so high the
   weights diverge (fix the lr, clip gradients), NaN already in the inputs
   (validate data), mixed-precision overflow in fp16 (loss scaling). ✍️
3. **Convexity.** The logistic loss is convex in (w, b) and strictly convex with
   L2, so any minimum is global and the init does not matter (only speed). The
   k-means objective is non-convex in the centers (permutations alone give k!
   equivalent minima, and there are bad local ones), so the init matters: use
   k-means++ and several restarts, report the best inertia, and check stability
   across seeds. ✅ (13-3: uniform init ~9,100 vs ~790)
4. **High AUC, useless model.** AUC averages over all thresholds and is
   insensitive to class balance: with 0.5% positives, a model can rank most
   positives above most negatives yet have terrible precision at any threshold
   you can afford to act on (e.g. 0.92 AUC with 3% precision at the top 1%).
   Report precision/recall at the operating point, PR-AUC, and the cost-weighted
   outcome; also check calibration if scores are used as probabilities. ✍️
5. **kNN at 50M items / 20 ms.** Exact brute force is O(N·d) per query —
   50M × 128 floats is ~25 GB and seconds per query. Move to approximate
   nearest-neighbour search (IVF, HNSW, product quantisation; notebook 08),
   possibly on GPU, sharded across machines; reduce d; cache popular queries.
   You give up exact recall (measure recall@k vs exact on a sample), memory
   (index overhead), and freshness (index rebuilds/updates). ✍️
6. **Gradient check.** Central differences with ε ≈ 1e-5 in float64 on a small
   problem (few examples, few dimensions); relative error
   ‖g − g_num‖ / (‖g‖ + ‖g_num‖); < 1e-7 good, > 1e-4 a bug. Misleading when:
   the function has kinks (ReLU at 0, max, hinge) near the test point;
   float32; randomness inside the loss (dropout — fix the mask/seed);
   regularisation terms checked separately from the data term; only a subset
   of parameters checked. ✅ (13.4: 7.5e-12 vs 0.33 for a missing factor 2)
