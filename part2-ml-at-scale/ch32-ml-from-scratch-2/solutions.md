# Solutions · Chapter 32 · ML from scratch II

Try each exercise before reading its answer, with a timer running. Markers:
✅ verified by running · 🔬 expected, depends on your run · ✍️ personal / open-ended.

Reference code: [`code/nb14.py`](solutions.py), graded in CI against
[`src/shoprank/grading/nb14.py`](../../src/grading/ch32.py).

---

## Exercise 1 · `best_split` in O(n log n)

✅ Reference: `ex1` in [`code/nb14.py`](solutions.py).

```python
order = np.argsort(x, kind="stable"); xs, ys = x[order], y[order]
s1, s2 = np.cumsum(ys), np.cumsum(ys ** 2)
nl = np.arange(1, n)                              # left size for a split after position i
sl, ql = s1[:-1], s2[:-1]
sr, qr = s1[-1] - sl, s2[-1] - ql
loss = (ql - sl**2 / nl) + (qr - sr**2 / (n - nl))
loss = np.where(xs[:-1] < xs[1:], loss, np.inf)   # no split between equal x
i = np.argmin(loss)
return (xs[i] + xs[i + 1]) / 2, loss[i]
```

**Complexity.** O(n log n) time (sort), O(n) memory. Everything after the sort
is a handful of vectorised O(n) passes.

✅ A correct O(n²) version (recompute both children per split) passed every
correctness test but took 3.95 s at n = 64,000 in the grader's timing check.

**Edge cases.** Duplicates (the `xs[:-1] < xs[1:]` mask), constant `x` or n < 2
(return `(None, total SSE)`), ties in the minimum loss (any is acceptable;
`argmin` picks the leftmost), large `|y|` (cancellation in `S₂ − S₁²/m`:
subtract `y.mean()` first for robustness; the minimiser does not change).

---

## Exercise 2 · `gbm_fit` / `gbm_predict` with stumps

✅ Reference: `ex2` and `gbm_predict` in [`code/nb14.py`](solutions.py).

```python
init = y.mean(); F = np.full(len(y), init); stumps = []
for _ in range(n_estimators):
    r = y - F                                         # negative gradient of ½(y-F)²
    loss, f, t = min((best_split(X[:, j], r)[1], j, best_split(X[:, j], r)[0])
                     for j in range(d) if best_split(X[:, j], r)[0] is not None)
    left = X[:, f] <= t
    stump = (f, t, r[left].mean(), r[~left].mean())
    stumps.append(stump)
    F += lr * np.where(X[:, f] <= t, stump[2], stump[3])
```

(The reference calls `best_split` once per feature; the one-liner above is for
reading, not for copying.)

✅ On the grader's 200-point problem (variance of y ≈ 0.758), 50 stumps at
lr = 0.1 reach a training MSE of 0.2345 — identical to
`GradientBoostingRegressor(max_depth=1)` to 16 digits, and predictions on new
points agree to ~4e-16. sklearn's default `friedman_mse` criterion ranks
splits by n_L·n_R/(n_L+n_R)·(ȳ_L − ȳ_R)², which is exactly the SSE reduction,
so both choose the same stumps.

**Complexity.** O(M·d·n log n) to train with re-sorting each round; pre-sort
once per feature to get O(M·d·n); histogram binning makes it O(M·d·(n + B)).
Prediction O(M) per row.

**Why the MSE cannot increase.** Each stump is the least-squares fit to the
residuals among stumps, and the constant-zero stump is always available, so
adding `lr × stump` with 0 < lr ≤ 1 never increases the training SSE.

---

## Exercise 3 · `mlp_grads`

✅ Reference: `ex3` in [`code/nb14.py`](solutions.py).

```python
z1 = X @ W1 + b1; h = np.maximum(z1, 0); logits = h @ W2 + b2
logp = log_softmax(logits); loss = -logp[np.arange(n), y].mean()
dlogits = np.exp(logp); dlogits[np.arange(n), y] -= 1; dlogits /= n
dW2 = h.T @ dlogits;  db2 = dlogits.sum(0)
dz1 = (dlogits @ W2.T) * (z1 > 0)
dW1 = X.T @ dz1;      db1 = dz1.sum(0)
```

**Complexity.** Forward O(n(dh + hC)); backward ≈ 2× forward (two matmuls per
layer: one for the weights, one for the input gradient). Memory O(n·h) for the
cached activations.

**Common mistakes.** The ReLU mask from `h > 0` is equivalent to `z1 > 0`
except at exactly 0; the real bugs are `dlogits @ W2` (shape error) and
forgetting `/ n` (all gradients off by n).

---

## Exercise 4 · Masked `attention`

✅ Reference: `ex4` in [`code/nb14.py`](solutions.py).

```python
scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
if mask is not None:
    scores = np.where(np.broadcast_to(mask, scores.shape), scores, -np.inf)
m = scores.max(-1, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
e = np.exp(scores - m)                                  # exp(-inf) = 0
den = e.sum(-1, keepdims=True)
w = np.divide(e, den, out=np.zeros_like(e), where=den > 0)
return w @ V, w
```

✅ The common `-1e9` fill passes everything except the fully masked row, where
it silently produces uniform weights (the grader reports "attention weights
differ from the reference").

**Complexity.** O(B·H·Lq·Lk·d) time, O(B·H·Lq·Lk) memory for the score matrix.

---

## Exercise 5 · `batchnorm_forward`

✅ Reference: `ex5` in [`code/nb14.py`](solutions.py).

Training: `mu, var = x.mean(0), x.var(0)`; `out = gamma * (x - mu) / sqrt(var + eps) + beta`;
`new_mean = (1-m)·old + m·mu`; `new_var = (1-m)·old + m·var·n/(n-1)`; raise for
n < 2. Eval: the same formula with the running statistics, which are returned
unchanged.

**Complexity.** O(n·F) time; O(F) state.

**Why raise for n = 1.** The batch variance is 0, every output equals `beta`,
and the unbiased variance divides by zero. PyTorch raises
`ValueError: Expected more than 1 value per channel when training` for the
same reason.

---

## Exercise 6 · `inverted_dropout`

✅ Reference: `ex6` in [`code/nb14.py`](solutions.py).

```python
if not training or p == 0: return x.copy()
if p >= 1: return np.zeros_like(x)
keep = rng.random(x.shape) >= p
return np.where(keep, x / (1 - p), 0.0)
```

E[out] = (1 − p) · x/(1 − p) = x, so inference needs no rescaling. The
variance grows by a factor 1/(1 − p) in training — one reason dropout and batch
norm interact badly when stacked (the "variance shift" seen at eval).

---

## Exercise 7 · Backprop through softmax + cross-entropy (written)

✍️ A whiteboard-ready derivation (your wording will differ):

1. **Softmax:** pₖ = e^{zₖ} / Σⱼ e^{zⱼ}. Its Jacobian: ∂pₖ/∂zⱼ = pₖ(δₖⱼ − pⱼ).
2. **Loss:** L = −log p_y = −z_y + log Σⱼ e^{zⱼ}. Differentiate directly (easier
   than going through the Jacobian): ∂L/∂zₖ = −δₖy + e^{zₖ}/Σⱼe^{zⱼ} = pₖ − δₖy.
   In vector form: **∇_z L = p − onehot(y)**. Each component lies in (−1, 1) and
   the components sum to 0.
3. **Batch and `Z = XW`:** with the mean over n examples, dZ = (P − Y)/n
   (shape n × C). Since Z = XW, dW = Xᵀ dZ (d × C, the shape of W) and
   dX = dZ Wᵀ.
4. **Stability:** compute log p with log-sum-exp (subtract max z); never
   `log(softmax(z))`. The gradient p − y itself is bounded and safe once p is
   computed stably.
5. **Confidently wrong vs right:** confidently wrong (p_y ≈ 0, some other
   pₖ ≈ 1): the gradient is ≈ −1 on the true class and ≈ +1 on the wrong one —
   the maximum possible magnitude, so learning is fast (unlike squared error
   on a sigmoid, whose gradient vanishes when saturated). Confidently right
   (p_y ≈ 1): the gradient ≈ 0. With soft targets (label smoothing,
   distillation), replace onehot(y) by the target distribution q: ∇ = p − q.

---

## Checkpoint answers

1. **GBDT training.** Start with a constant F₀ (mean for squared loss). Each
   round: compute pseudo-residuals rᵢ = −∂L/∂F(xᵢ) (y − F for squared loss); fit
   a small regression tree to them; set leaf values (mean residual for squared
   loss; a Newton step Σg/Σh for general losses); F ← F + lr·tree. Log-loss:
   residuals y − σ(F), Newton leaf values Σ(y − p)/Σp(1 − p), init = log-odds.
   LambdaMART: pseudo-residuals are the "lambdas" — pairwise RankNet gradients
   within a query weighted by |ΔNDCG| of swapping the pair — and leaves use
   the Newton step with the matching second derivatives (notebook 06). ✅ (14-2)
2. **Histogram GBDTs.** They bucket each feature into ≤ 255 bins once; a node's
   split search builds gradient/hessian histograms in O(n) and scans B bins,
   instead of sorting (O(n log n)) per node; the sibling's histogram comes from
   parent − child for free; bins are small integers (cache-friendly, less
   memory) and parallelise well. They give up exact thresholds (only bin edges
   are candidates), which rarely matters and acts as regularisation; very
   fine-grained features may lose a little. ✍️
3. **NaN at inference with masks.** Check for queries with **no** allowed key
   (empty history after filtering, all-padding sequences, left- vs
   right-padding mismatch between training and serving); the mask's polarity
   (True = keep vs True = block, which differs between APIs); the mask dtype
   (float additive mask vs boolean); fp16 overflow of −inf/−1e9 fills; and
   reproduce with the failing users' inputs, asserting `isfinite` after each
   layer. Fix by defining the empty-row output (zeros, as in 14-4) or by
   guaranteeing at least one valid key (e.g. a "no history" token). ✍️
4. **Batch norm offline vs one-at-a-time serving.** (a) The model is served in
   training mode: outputs depend on the batch, and a batch of 1 normalises
   everything to `beta` (PyTorch would raise; ONNX exports may not). Confirm by
   scoring the same row alone and in different batches; fix with `eval()` and a
   parity test. (b) The running statistics do not match serving inputs
   (computed on a different distribution or preprocessing, or never updated
   because of a frozen/fused export). Confirm by comparing per-feature input
   mean/variance in serving with the stored running stats. ✍️
5. **Quadratic attention.** Every query scores every key: Lq·Lk dot products and
   an Lq × Lk weight matrix per head. Reducing memory/time: IO-aware exact
   kernels (FlashAttention: tiling, no materialised matrix — memory linear,
   time still quadratic); sparse/local patterns (sliding windows, block-sparse,
   Longformer/BigBird); low-rank or kernelised linear attention (Linformer,
   Performer); and for recommender histories, truncating or summarising the
   sequence. ✍️
6. **Gradient of softmax CE.** p − y (or p − q for a target distribution q).
   Because the gradient only needs a target distribution, label smoothing
   (q = (1 − ε)·onehot + ε/C) and distillation (q = teacher softmax at
   temperature T) plug in without any new derivation. ✅ (14-7, 13-2)
