# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 14 · ML from scratch II
#
# **Day 7 · ~90 minutes · CPU only**
#
# The second half of the ML coding drill covers the models you use every day
# through a library: tree splits and gradient boosting (the engine of LightGBM
# rankers in notebook 06), backpropagation, attention, batch norm and dropout.
# Interviewers ask for these because a few lines of NumPy reveal whether you
# know what the library is doing, including the parts that bite in production:
# train/eval modes, masking and numerical overflow.
#
# By the end you will be able to:
#
# 1. find the best regression-tree split in O(n log n) with sorting and prefix sums;
# 2. implement gradient boosting with stumps and explain "fitting the negative gradient";
# 3. derive and code backprop for a two-layer MLP, and verify it with a gradient check;
# 4. implement masked scaled dot-product attention that never returns NaN;
# 5. implement batch norm (train vs eval, running statistics) and inverted dropout;
# 6. walk an interviewer through the softmax-cross-entropy gradient on a whiteboard.
#
# **You will produce:** six graded NumPy implementations and one written derivation.
#
# Same rules as notebook 13: start a timer, no lookups, run the grader, then
# read the talk-through box.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level2/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: nothing from earlier notebooks is needed here ----------------
import numpy as np

from shoprank.drills import numerical_gradient, relative_error
from shoprank.grading import nb14 as grade

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

# %% [markdown]
# ## 14.1 · Prefix sums: the trick behind fast tree splits
#
# A regression tree chooses the split that minimises the summed squared error
# (SSE) of the two children. For a set with sum S₁ = Σy, sum of squares
# S₂ = Σy² and size m,
#
# $$\text{SSE} = \sum (y - \bar y)^2 = S_2 - S_1^2 / m.$$
#
# After sorting by the feature, every candidate left child is a **prefix**, so
# cumulative sums give S₁ and S₂ for all n−1 splits at once. Sorting is
# O(n log n); everything after it is O(n). Recomputing each child's mean from
# scratch is O(n) per split and O(n²) overall. Here is the identity checked on
# every prefix of a small array:

# %%
y_demo = rng.normal(size=8)
S1, S2 = np.cumsum(y_demo), np.cumsum(y_demo ** 2)
m = np.arange(1, 9)
fast = S2 - S1 ** 2 / m
slow = np.array([((y_demo[:i] - y_demo[:i].mean()) ** 2).sum() for i in range(1, 9)])
print("prefix SSE, fast:", fast)
print("max |fast - slow|:", np.abs(fast - slow).max())

# %% [markdown]
# **Expected:** the first entry is 0 (one point has no spread), the rest grow,
# and the two methods agree to ~1e-16.
#
# Two details matter in the exercise and in interviews. **You cannot split
# between equal feature values** (a threshold puts all copies on one side), so
# only positions where `x_sorted[i] < x_sorted[i+1]` are valid. And
# `S₂ − S₁²/m` subtracts two large numbers: with big `y` values it loses
# precision (catastrophic cancellation). Libraries centre `y` first or use
# Welford-style updates; say so if asked about robustness.

# %% [markdown]
# ## 14.2 · Backprop: shapes, local gradients, and the chain rule
#
# Backprop is the chain rule applied in reverse topological order, with one
# rule that removes most bugs: **the gradient of a quantity has the shape of
# that quantity.** If `W` is `(d, h)`, `dW` is `(d, h)`, and there is usually only
# one way to multiply the available arrays to get that shape.
#
# | Forward | Backward (given `dout`, the gradient of the loss w.r.t. the output) |
# |---|---|
# | `out = X @ W + b` | `dX = dout @ W.T`, `dW = X.T @ dout`, `db = dout.sum(0)` |
# | `out = relu(z)` | `dz = dout * (z > 0)` |
# | `out = sigmoid(z)` | `dz = dout * out * (1 - out)` |
# | `L = mean CE(softmax(z), y)` | `dz = (softmax(z) - onehot(y)) / n` |
# | `out = x * mask / (1-p)` (dropout) | `dx = dout * mask / (1-p)` |
#
# A one-layer check, using the helpers from notebook 13:

# %%
X_demo = rng.normal(size=(6, 4))
W_demo = rng.normal(size=(4, 3))
T_demo = rng.normal(size=(6, 3))


def loss_linear(W):                       # L = 0.5 * mean over rows of ||XW - T||^2
    return 0.5 * np.sum((X_demo @ W - T_demo) ** 2) / len(X_demo)


dout = (X_demo @ W_demo - T_demo) / len(X_demo)    # dL/d(XW): shape (6, 3)
dW = X_demo.T @ dout                                 # (4, 6) @ (6, 3) -> (4, 3), the shape of W
print("dW shape:", dW.shape)
print(f"relative error vs numerical gradient: {relative_error(dW, numerical_gradient(loss_linear, W_demo.copy())):.1e}")

# %% [markdown]
# **Expected:** `dW shape: (4, 3)` and a relative error around 1e-11 or smaller.

# %% [markdown]
# ## 14.3 · Attention and the fully-masked row
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$
#
# The `√d` keeps the scores' variance near 1 when `q` and `k` have unit-variance
# entries (a dot product of d such terms has variance d), so the softmax does not
# saturate. A **mask** says which keys each query may use: *causal* masks hide
# the future in a decoder; *padding* masks hide pad tokens in a batch of
# variable-length sequences (notebook 10 used both).
#
# The production bug: a query row whose keys are **all** masked. Setting masked
# scores to `-inf` gives `exp(-inf - (-inf)) = exp(nan)` and the NaN then flows
# into every gradient. It happens with left-padded batches, empty histories
# (a cold-start user in notebook 10) and some cross-attention layouts.

# %%
scores = np.array([[2.0, 1.0, 0.5], [0.3, 0.2, 0.1]])
mask = np.array([[True, True, False], [False, False, False]])    # row 2: nothing allowed
masked = np.where(mask, scores, -np.inf)
with np.errstate(invalid="ignore"):
    e = np.exp(masked - masked.max(axis=1, keepdims=True))
    print("naive weights:\n", e / e.sum(axis=1, keepdims=True))
big_negative = np.where(mask, scores, -1e9)
e2 = np.exp(big_negative - big_negative.max(axis=1, keepdims=True))
print("-1e9 instead of -inf:\n", e2 / e2.sum(axis=1, keepdims=True))

# %% [markdown]
# **Expected:** the naive version prints `nan` for the whole second row; the
# `-1e9` version prints a *uniform* `[0.3333 0.3333 0.3333]` for it — no NaN, but
# the query now silently attends to keys it was told to ignore. Exercise 4 asks
# for the explicit convention: zero weights and a zero output for such rows.
# (In our check, PyTorch 2.8's `F.scaled_dot_product_attention` on CPU returned
# zeros for a fully masked row — the same convention; earlier releases and some
# kernels returned NaN, so test the version and backend you deploy.)

# %% [markdown]
# ## 14.4 · Batch norm and dropout: two layers with two modes
#
# Both layers behave differently in training and inference, which makes them
# the most common source of "the offline metric does not match serving" bugs
# in deep models.
#
# | | Training | Inference (eval) |
# |---|---|---|
# | **BatchNorm** | normalise with the *batch* mean and (biased) variance; update running averages | normalise with the *running* averages; output does not depend on the rest of the batch |
# | **Dropout (inverted)** | zero each unit with prob. p, scale survivors by 1/(1-p) | identity |
#
# Why "inverted": scaling at training time keeps E[output] = input, so
# inference needs no change. (The original paper scaled at test time instead.)
# A model left in training mode at serving time gives predictions that change
# with the batch it happens to be in, and differ between calls:

# %%
x_demo = rng.normal(loc=2.0, size=(4, 3))
batch_norm_train = (x_demo - x_demo.mean(0)) / np.sqrt(x_demo.var(0) + 1e-5)
same_row_other_batch = np.vstack([x_demo[:1], rng.normal(loc=2.0, size=(3, 3))])
bn_other = (same_row_other_batch - same_row_other_batch.mean(0)) / np.sqrt(same_row_other_batch.var(0) + 1e-5)
print("row 0 normalised within batch A:", batch_norm_train[0])
print("row 0 normalised within batch B:", bn_other[0])

# %% [markdown]
# **Expected:** two different vectors for the same input row (in our run
# `[1.4692 -0.3189 0.0834]` and `[0.8546 0.2136 0.0057]`). With
# running statistics (eval mode) the row would map to the same output in any
# batch. In PyTorch, that is the whole job of `model.eval()` (notebook 09).

# %% [markdown]
# ## 14.5 · Exercises
#
# | # | Implement | Target time |
# |---|---|---|
# | 1 | best split for a regression stump, O(n log n) | 15 min |
# | 2 | gradient boosting with stumps (squared loss) | 20 min |
# | 3 | two-layer MLP forward + backward | 20 min |
# | 4 | masked scaled dot-product attention | 15 min |
# | 5 | batch norm forward, train and eval | 12 min |
# | 6 | inverted dropout | 5 min |
# | 7 | written: backprop through softmax + cross-entropy | 10 min |

# %% [markdown]
# > **Exercise 1.** *(target 15 min)* Implement `best_split(x, y)` → `(threshold, loss)`
# > for a regression stump on one feature: points with `x <= threshold` go left,
# > `loss = SSE(left) + SSE(right)` is minimal, and `threshold` is the **midpoint
# > between two consecutive distinct sorted values**. If no split is possible
# > (fewer than 2 points or constant `x`), return `(None, SSE of all points)`.
# > Run in **O(n log n)**: sort once, prefix sums for everything else.
# >
# > Checked: 20 random small cases with many duplicate `x` values against a
# > reference; the returned loss equals the SSE of splitting at the returned
# > threshold; `x=[1,1,2,2], y=[0,10,0,10]` → `(1.5, 100.0)`; one point; constant
# > `x`; timing up to n = 256,000 (2-second budget); `x` not modified.

# %%
def best_split(x, y):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(best_split)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Clarify:* the criterion (SSE/variance for regression; Gini or entropy for classification), the threshold convention, minimum leaf size, missing values.
# > 2. *Brute force:* try every threshold, recompute both SSEs: O(n²). Say it, then improve.
# > 3. *Optimise:* sort by x, prefix sums of y and y², SSE of every prefix/suffix in O(1), mask out positions between equal x values.
# > 4. *Complexity:* O(n log n) time (the sort), O(n) memory. Across d features: O(d·n log n) per node; LightGBM/XGBoost "hist" mode pre-buckets x into ≤ 255 bins so a node costs O(n + bins) per feature with no sort.
# > 5. *Edge cases:* duplicates, constant feature, n = 1, huge y values (cancellation in S₂ − S₁²/m).

# %% [markdown]
# > **Exercise 2.** *(target 20 min)* Implement `gbm_fit(X, y, n_estimators=100, lr=0.1)`
# > for **squared loss with depth-1 trees (stumps)** and the matching
# > `gbm_predict(model, X)`. The model is a dict
# > `{"init": mean(y), "lr": lr, "stumps": [(feature, threshold, left_value, right_value), ...]}`
# > and `predict(X) = init + lr · Σ stump(X)`, where a stump returns `left_value`
# > if `X[:, feature] <= threshold` else `right_value`. Each round fits a stump
# > to the current residuals `y − F` (reuse your `best_split` on each feature);
# > leaf values are the mean residual on each side.
# >
# > Checked (the grader uses its own predict on your model dict, and also
# > compares your `gbm_predict`): exactly `n_estimators` stumps; `init = mean(y)`;
# > training MSE never increases as stumps are added; final training MSE within
# > 1% of `sklearn.ensemble.GradientBoostingRegressor(max_depth=1)`; predictions on
# > new points within 1e-3 of sklearn's; a constant feature is never split.

# %%
def gbm_fit(X, y, n_estimators=100, lr=0.1):
    # YOUR CODE HERE
    raise NotImplementedError


def gbm_predict(model, X):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(gbm_fit, gbm_predict)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *The idea in one sentence:* boosting is gradient descent in function space; each tree approximates the negative gradient of the loss at the current predictions. For ½(y − F)², the negative gradient is the residual y − F.
# > 2. *Why `init = mean(y)`:* it is the constant minimising squared loss; for log-loss it would be the log-odds of the base rate.
# > 3. *Learning rate (shrinkage):* smaller lr needs more trees and generalises better; tune `n_estimators` with early stopping on a validation set.
# > 4. *Complexity:* O(M · d · n log n) for M rounds (re-sorting each round; pre-sorting once per feature gives O(M·d·n)); prediction O(M) per row.
# > 5. *Beyond squared loss:* other losses change the pseudo-residuals and the leaf values (a Newton step: Σg / Σh, which is what XGBoost/LightGBM use; LambdaMART plugs in ranking gradients — notebook 06).

# %% [markdown]
# > **Exercise 3.** *(target 20 min)* Implement `mlp_grads(params, X, y)` →
# > `(loss, grads)` for `X → Linear(W1, b1) → ReLU → Linear(W2, b2) → softmax
# > cross-entropy` (mean over the batch). `params` and `grads` are dicts with keys
# > `"W1" (d, h)`, `"b1" (h,)`, `"W2" (h, C)`, `"b2" (C,)`; `y` holds integer
# > labels. Do not modify `params`.
# >
# > Checked: the loss equals the reference; every gradient passes a numerical
# > gradient check (relative error < 1e-6); logits scaled by 1000 stay finite.

# %%
def mlp_grads(params, X, y):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(mlp_grads)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Write the forward pass with named intermediates* (`z1, h, logits`): you need them in the backward pass. Say the shapes as you go.
# > 2. *Backward in reverse:* `dlogits = (P − Y)/n` → `dW2 = hᵀ dlogits`, `db2 = Σ dlogits` → `dh = dlogits W2ᵀ` → `dz1 = dh ⊙ 1[z1 > 0]` → `dW1 = Xᵀ dz1`, `db1 = Σ dz1`.
# > 3. *Verify:* a gradient check on a tiny network; mention the ReLU kink (avoid points with z1 ≈ 0).
# > 4. *Complexity:* forward and backward are both O(n·(d·h + h·C)); the backward pass costs about 2× the forward. Memory O(n·h) for the stored activations — the reason activation checkpointing exists (Level 3).
# > 5. *Follow-ups to expect:* initialisation (He for ReLU), dead ReLUs, why not sigmoid hidden units (vanishing gradients).

# %% [markdown]
# > **Exercise 4.** *(target 15 min)* Implement `attention(Q, K, V, mask=None)` →
# > `(output, weights)`: scaled dot-product attention for inputs with any leading
# > batch dimensions — `Q (..., Lq, d)`, `K (..., Lk, d)`, `V (..., Lk, dv)`.
# > `mask` is boolean, **True = may attend**, broadcastable to `(..., Lq, Lk)`.
# > A query with no allowed key gets **all-zero weights and a zero output**.
# >
# > Checked against a reference: no mask, a causal mask `(4, 5)`, a padding mask
# > `(2, 1, 1, 5)`, a fully masked row; no NaN anywhere; scores in the thousands;
# > plain 2-D inputs.

# %%
def attention(Q, K, V, mask=None):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex4(attention)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Shapes:* scores `(..., Lq, Lk)` from `Q @ swapaxes(K, -1, -2)`; weights sum to 1 over the **key** axis (the last one).
# > 2. *Why √d,* and why masking happens **before** the softmax (masking after and renormalising is equivalent only if you renormalise; zeroing after the softmax is a bug).
# > 3. *Stability:* subtract the row max over allowed entries; handle a row max of −inf.
# > 4. *Complexity:* O(Lq·Lk·d) time and O(Lq·Lk) memory per head — quadratic in sequence length, which is what FlashAttention (tiling, no materialised score matrix) and sparse/linear attention address.
# > 5. *Multi-head:* reshape `(B, L, H·dₕ)` → `(B, H, L, dₕ)`, run this function, reshape back, output projection.

# %% [markdown]
# > **Exercise 5.** *(target 12 min)* Implement
# > `batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5)`
# > → `(out, new_running_mean, new_running_var)` for `x (n, features)`, with
# > PyTorch's conventions:
# >
# > * training: normalise with the batch mean and the **biased** batch variance;
# >   `new = (1 − momentum)·old + momentum·batch_stat`, where the running variance
# >   uses the **unbiased** (n − 1) batch variance; a batch of 1 raises `ValueError`;
# > * eval: normalise with the running statistics and return them unchanged.
# >
# > Return new arrays; do not modify the inputs.

# %%
def batchnorm_forward(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex5(batchnorm_forward)

# %% [markdown]
# > **Exercise 6.** *(target 5 min)* Implement `inverted_dropout(x, p, training, rng)`:
# > in training, zero each element independently with probability `p` (draw
# > from the `np.random.Generator` you are given) and scale the survivors by
# > `1/(1 − p)`; in eval, return `x` unchanged. Handle `p = 0` and `p = 1`
# > (all zeros, no division by zero). Do not modify `x`.

# %%
def inverted_dropout(x, p, training, rng):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex6(inverted_dropout)

# %% [markdown]
# > 🗣️ **Talk it through (Exercises 5-6).**
# > 1. *Why biased vs unbiased:* the forward pass normalises the batch you have (biased); the running variance estimates the population variance (unbiased). Mixing them up is invisible in training and shifts eval outputs slightly.
# > 2. *Momentum convention:* PyTorch's `momentum=0.1` weights the **new** batch by 0.1; Keras's `momentum=0.99` weights the **old** value by 0.99. Same idea, opposite parameter — say which you mean.
# > 3. *Failure modes:* tiny batches (noisy statistics; use GroupNorm/LayerNorm), batch-dependent outputs in serving (forgot eval mode), distribution shift between train and serve (running stats stale), dropout left on at inference (noisy predictions).
# > 4. *Complexity:* O(n·features) for both; batch norm keeps 2 × features extra state.
# > 5. *Backward (follow-up):* dropout's backward is the same mask times the same scale; batch norm's backward has three paths through the mean and variance — offer the compact formula `dx = (γ/σ)(dy − mean(dy) − x̂·mean(dy·x̂))`.

# %% [markdown]
# > **Exercise 7.** *(written, target 10 min)* Out loud or on paper, as you would at
# > a whiteboard: derive ∂L/∂z for `L = −log softmax(z)_y` with logits `z ∈ ℝ^C`.
# > Then extend to a batch of n examples with the mean loss and to `Z = XW`,
# > giving `dW`. Say where numerical stability enters, and what the gradient
# > looks like when the model is confidently wrong vs confidently right.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer:*
#
# 1. Softmax: pₖ = ...
# 2. L = ..., so ∂L/∂zₖ = ...
# 3. Batch and `Z = XW`: dW = ...
# 4. Stability: ...
# 5. Confidently wrong vs right: ...

# %% [markdown]
# ## 14.6 · Score yourself
#
# Same rubric as 13.6: 0-2 on framing, correctness, numerical care, efficiency,
# verification and communication; time against the target. The exercises that
# most often go wrong in practice interviews are 3 (a transposed gradient) and
# 4 (the mask): if those took over 1.5× their target, redo them tomorrow from
# a blank cell before moving on.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Exercise 1: "threshold is not a midpoint" | splitting between two equal x values, or returning x itself | only positions with `xs[i] < xs[i+1]`; threshold `(xs[i] + xs[i+1]) / 2` |
# | Exercise 1 times out at n = 64,000 | recomputing means per split | prefix sums: `SSE = S2 - S1**2 / m` |
# | Exercise 2: MSE close to sklearn's but predictions differ | leaf values not the mean residual, or `lr` applied twice | store raw means in the stump; multiply by `lr` once in predict |
# | Exercise 2: "a constant feature can never be split on" | `best_split` returns a threshold for constant x | return `(None, ...)` and skip that feature |
# | Exercise 3: relative error ~0.5-1 for `W1` only | ReLU mask applied to `h` instead of `z1`, or `dh` computed with `W2` not `W2.T` | `dz1 = (dlogits @ W2.T) * (z1 > 0)` |
# | Exercise 3: all gradients off by a constant factor | sum vs mean mismatch | divide `dlogits` by n once |
# | Exercise 4: NaN only with the fully masked row | `-inf - (-inf)` in the max subtraction | replace a −inf row max with 0, divide only where the row sum > 0 |
# | Exercise 4: weights wrong with the padding mask | mask applied with the wrong axis or inverted (True = blocked) | here True means *may attend*; broadcast to the scores' shape |
# | Exercise 5: "running var uses the biased variance" | `x.var(0)` in the update | `x.var(0, ddof=1)` (or `* n / (n - 1)`) for the running statistic only |
# | Exercise 6: dropped fraction ≈ 1 − p | the keep/drop condition reversed | `keep = rng.random(x.shape) >= p` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Walk through how a gradient-boosted tree model is trained for squared loss, then say what changes for log-loss and for a ranking loss like LambdaMART.
# 2. Why do histogram-based GBDTs (LightGBM, XGBoost `hist`) train much faster than exact-split implementations, and what do they give up?
# 3. Your transformer trains fine but outputs NaN for some users at inference. Masking is the suspect. What exactly would you check?
# 4. A model with batch norm scores well offline but much worse in production, where requests are scored one at a time. Give two plausible causes and how to confirm each.
# 5. Why is attention's cost quadratic in sequence length, and name two families of methods that reduce the time or memory cost.
# 6. Without code: what is the gradient of softmax cross-entropy with respect to the logits, and why does it make training with label smoothing or soft targets straightforward?
#
# ➡️ **Next:** [`15_coding_patterns_1.ipynb`](15_coding_patterns_1.ipynb) — Day 8: the classic coding round, pattern by pattern, with a protocol for talking through it.
