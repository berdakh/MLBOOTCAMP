# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 13 · ML from scratch I
#
# **Day 7 · ~90 minutes · CPU only**
#
# Many ML engineer loops include an "ML coding" round: *implement logistic
# regression*, *write k-means*, *compute AUC without a library*, in 30-45 minutes,
# in NumPy, while talking. The method is never the hard part (you know all of
# these); the round tests whether your code is **correct at the edges,
# numerically stable, vectorised, and checked**, and whether you can say why.
#
# By the end you will be able to:
#
# 1. describe how an ML coding round is usually scored, and plan your 45 minutes;
# 2. replace Python loops with broadcasting, and *measure* the difference;
# 3. make softmax, sigmoid and cross-entropy overflow-proof (log-sum-exp);
# 4. check any analytic gradient against a numerical one in five lines;
# 5. implement, under time pressure, logistic and softmax regression, k-means++, kNN, ROC-AUC with ties and NDCG@k.
#
# **You will produce:** six graded NumPy implementations (nothing is written to disk).
#
# **How to use this notebook.** Every exercise has a **target time**. Start a
# timer, write the function without looking anything up, run the grader, and
# only then read the talk-through box and the solution. Rerun the exercises on
# later days without notes: retrieval practice is what makes them stick.

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
# Every input in this notebook is small, synthetic and seeded.
import numpy as np

from shoprank.drills import Timer, growth_table, numerical_gradient, relative_error, time_call
from shoprank.grading import nb13 as grade

np.set_printoptions(precision=4, suppress=True)
print("numpy", np.__version__)

# %% [markdown]
# ## 13.1 · What the ML coding round actually scores
#
# The format varies between companies and teams, so treat this as the common
# shape rather than any company's rubric: one or two problems, 30-60 minutes, a
# shared editor or notebook, NumPy allowed, scikit-learn and PyTorch usually
# not. Interviewers tend to score along the same few axes:
#
# | Axis | Strong signal | Weak signal |
# |---|---|---|
# | **Problem framing** | states the objective (loss), shapes and conventions before coding | starts typing at once |
# | **Correctness** | right maths, right shapes, edge cases named and handled | works on the happy path only |
# | **Numerical care** | stable softmax/sigmoid, no `log(0)`, float64 for checks | NaN on the interviewer's second input |
# | **Efficiency** | vectorised; states time and memory complexity | triple Python loops; cannot say the cost |
# | **Verification** | writes a 3-line test or gradient check unprompted | "I think it works" |
# | **Communication** | narrates trade-offs, asks before assuming | silent for ten minutes |
#
# A 45-minute plan that works for "implement X":
#
# | Minutes | Do |
# |---|---|
# | 0-5 | Restate. Write the loss and the update rule as maths. Fix shapes: `X (n, d)`, `y (n,)`, `W (d, C)`. Ask about regularisation, bias, stopping. |
# | 5-25 | Code the core, vectorised. Say what each line computes. |
# | 25-35 | Test: a tiny hand-checkable case, an edge case, a gradient check or a library comparison. |
# | 35-45 | Complexity, stability, what you would change at scale (mini-batches, sparse X, early stopping). |
#
# The sections below drill the three habits that separate the strong column from
# the weak one: vectorisation, stability and gradient checking.

# %% [markdown]
# ## 13.2 · Vectorisation: measure it
#
# Pairwise squared distances between `A (m, d)` and `B (n, d)` appear in kNN,
# k-means, RBF kernels and retrieval. Three versions, from interview-weak to
# interview-strong:

# %%
def pdist_loops(A, B):
    out = np.empty((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            out[i, j] = ((A[i] - B[j]) ** 2).sum()
    return out


def pdist_broadcast(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)       # builds an (m, n, d) array


def pdist_expand(A, B):
    # |a - b|^2 = |a|^2 - 2 a.b + |b|^2 : one matrix product, (m, n) memory
    d2 = (A ** 2).sum(1)[:, None] - 2 * A @ B.T + (B ** 2).sum(1)[None, :]
    return np.maximum(d2, 0.0)                                    # rounding can give tiny negatives


rng = np.random.default_rng(0)
A, B = rng.normal(size=(300, 32)), rng.normal(size=(400, 32))
d_loop, t_loop = time_call(pdist_loops, A, B)
d_bc, t_bc = time_call(pdist_broadcast, A, B, repeat=5)       # best of 5 runs
d_ex, t_ex = time_call(pdist_expand, A, B, repeat=5)
for name, t in [("python loops", t_loop), ("broadcast", t_bc), ("expansion (BLAS)", t_ex)]:
    print(f"{name:<18}{t * 1000:8.2f} ms")
print("max |difference|:", np.abs(d_loop - d_ex).max())

# %% [markdown]
# **Expected:** in our runs the loop version took 170-310 ms, broadcasting
# 10-13 ms and the expansion usually 0.4-0.6 ms (twice ~16 ms, on a machine
# shared with other jobs, where multi-threaded BLAS waits for free cores); the
# three agree to about 6e-14. Your timings will differ; the loop will always be
# the slowest by far. Lesson for benchmarks: take the best of several runs and
# say what else was running.
#
# Why the expansion wins: it turns the work into one matrix multiply (BLAS,
# multi-threaded, cache-friendly) and never materialises the `(m, n, d)`
# intermediate. Say the memory cost out loud in an interview: broadcasting
# needs `m·n·d·8` bytes (here 300·400·32·8 ≈ 31 MB; with 10k × 10k × 128 it
# would be 102 GB). Mention the `np.maximum(…, 0)` too: catastrophic
# cancellation can make a distance of zero come out as `-1e-15`, and a
# `sqrt` of that is NaN.
#
# **Measuring complexity.** "It's O(n log n)" is a claim; a growth table is
# evidence. `growth_table` (in [`src/shoprank/drills.py`](../../src/shoprank/drills.py))
# times a function at doubling sizes and prints the local exponent
# `log(t2/t1) / log(n2/n1)`, plus the least-squares slope of log time against
# log n over all sizes: about 1 for linear, about 2 for quadratic.

# %%
def count_inversions_slow(x):          # O(n^2): pairs i < j with x[i] > x[j], two Python loops
    n, count = len(x), 0
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            count += xi > x[j]
    return count


def sort_only(x):                      # O(n log n)
    return np.sort(x)


print("two nested loops:")
quad = growth_table(count_inversions_slow, lambda n: (np.random.default_rng(n).normal(size=n).tolist(),),
                    [500, 1000, 2000, 4000], repeat=2)
print("np.sort:")
nlogn = growth_table(sort_only, lambda n: (np.random.default_rng(n).normal(size=n),),
                     [100_000, 200_000, 400_000, 800_000], repeat=5)
x_big = np.random.default_rng(0).normal(size=1_000_000)
with Timer("np.sort of 1M floats, timed with Timer"):
    np.sort(x_big)

# %% [markdown]
# **Expected:** a fitted exponent of about 2 for the nested loops (1.8-2.2
# in our runs) and about 1.1 for `np.sort` (1.11-1.14); single ratios between
# neighbouring sizes jump around (we saw 1.0 to 2.4), which is why you read the
# trend, not one ratio. The `Timer` line prints a
# few milliseconds. Every grader in notebooks 13-16 that has a performance
# part uses the same idea: it runs your function at growing sizes with a
# **2-second** budget per run, and stops early when the growth projects far
# past it. An O(n log n) solution uses a few
# percent of that budget; an O(n²) one cannot fit.

# %% [markdown]
# ## 13.3 · Numerical stability: log-sum-exp
#
# `exp(1000)` overflows float64 (max ≈ 1.8e308, i.e. `exp(709.8)`). Logits of
# a few hundred are routine: an unnormalised feature, a large learning rate, a
# temperature of 0.01. The fix uses an identity: for any constant `m`,
#
# $$\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}, \qquad m = \max_j z_j,$$
#
# so every exponent is ≤ 0 and the largest term is exactly 1.

# %%
z = np.array([[1000.0, 1001.0, 999.0], [1.0, 2.0, 3.0]])

with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(z) / np.exp(z).sum(1, keepdims=True)
print("naive softmax:\n", naive)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


print("stable softmax:\n", np.exp(log_softmax(z)))

# The same trap in binary log-loss: log(sigmoid(z)) for z = -800
zb = np.array([-800.0, 0.0, 800.0])
with np.errstate(over="ignore", divide="ignore"):
    print("naive  -log(sigmoid(z)):", -np.log(1 / (1 + np.exp(-zb))))
print("stable -log(sigmoid(z)):", np.logaddexp(0, -zb))     # log(1 + e^{-z}) without overflow

# %% [markdown]
# **Expected:** the naive softmax prints `nan` in the first row; the stable one
# prints `[0.2447 0.6652 0.09]` (the same as softmax of `[1, 2, 0]`) and
# `[0.09 0.2447 0.6652]` for the second row. The naive log-loss is `inf` at
# z = -800 (the true value is 800); `np.logaddexp(0, -z)` gives `[800, 0.6931, 0]`.
#
# Rules worth saying in the interview:
#
# * compute **log-probabilities**, not probabilities, whenever a log follows;
# * cross-entropy from logits = `logsumexp(z) - z[y]`; never `log(softmax(z))`;
# * `np.logaddexp(0, z)` is softplus `log(1 + e^z)`; `np.log1p(x)` for small x;
# * for a sigmoid on large negative inputs, use `e^z / (1 + e^z)` there (or `scipy.special.expit`).

# %% [markdown]
# ## 13.4 · Gradient checking
#
# If you derive a gradient by hand, check it numerically before you trust it.
# The central difference has error O(ε²):
#
# $$\frac{\partial f}{\partial w_i} \approx \frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}$$
#
# Compare with the **relative error** `‖g - g_num‖ / (‖g‖ + ‖g_num‖)`. In float64
# with ε = 1e-5: below ~1e-7 is right; 1e-4 or above is a bug (a wrong factor
# or a wrong transpose usually gives 0.1-1). Five lines, and interviewers notice
# when you write them unprompted.

# %%
Xg = rng.normal(size=(20, 3))
yg = rng.normal(size=20)
wg = rng.normal(size=3)


def mse(w):
    return np.mean((Xg @ w - yg) ** 2)


def mse_grad_right(w):
    return 2 * Xg.T @ (Xg @ w - yg) / len(yg)


def mse_grad_buggy(w):          # forgot the factor 2: a classic
    return Xg.T @ (Xg @ w - yg) / len(yg)


num = numerical_gradient(mse, wg.copy())
print(f"right gradient  rel. error = {relative_error(mse_grad_right(wg), num):.1e}")
print(f"buggy gradient  rel. error = {relative_error(mse_grad_buggy(wg), num):.1e}")

# %% [markdown]
# **Expected:** about 1e-11 for the right gradient, 0.33 for the buggy one
# (half the gradient: |g/2 - g| / (|g/2| + |g|) = 1/3).
#
# Practical notes: check in **float64** (float32 rounding swamps ε); use a
# handful of examples and dimensions (it costs 2 evaluations per parameter);
# avoid points where the function has a kink (ReLU at 0, `max`, `abs`), where
# the numerical gradient is meaningless.

# %% [markdown]
# ## 13.5 · Exercises
#
# Six implementations. Each states a **target time**, the signature and
# conventions, the edge cases the grader checks openly (it also checks hidden
# ones), and ends with a talk-through box: what to say, in what order, and the
# complexity you should state. NumPy only.
#
# | # | Implement | Target time |
# |---|---|---|
# | 1 | logistic regression, batch GD + L2 | 15 min |
# | 2 | softmax regression loss and gradient | 15 min |
# | 3 | k-means with k-means++ | 20 min |
# | 4 | kNN classifier, vectorised | 12 min |
# | 5 | ROC-AUC with ties (rank statistic) | 15 min |
# | 6 | NDCG@k | 10 min |

# %% [markdown]
# > **Exercise 1.** *(target 15 min)* Implement `fit_logreg(X, y, lr, epochs, l2)`
# > → `(w, b)`: binary logistic regression trained by **full-batch gradient
# > descent** on
# >
# > $$L(w, b) = \frac{1}{n}\sum_i \big[\log(1 + e^{z_i}) - y_i z_i\big] + \frac{\lambda}{2}\lVert w\rVert^2, \qquad z_i = x_i^\top w + b,$$
# >
# > starting from zeros, with `l2` = λ and **no penalty on the intercept**.
# > `X (n, d)`, `y (n,)` in {0, 1}; return `w (d,)` and a float `b`.
# >
# > The grader compares you with `sklearn.linear_model.LogisticRegression`. Its
# > objective is `C·Σ logloss + ½‖w‖²`; divide by `C·n` and you get yours with
# > `C = 1/(n·λ)`. Checked: coefficients within 0.02 of sklearn's; the loss
# > decreases with more epochs; no NaN on large-scale separable data; finite
# > output when every label is 1; your inputs are not modified.

# %%
def fit_logreg(X, y, lr=0.1, epochs=1000, l2=0.0):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(fit_logreg)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Clarify:* labels 0/1 or ±1? Intercept regularised? Mean or sum loss? Fixed epochs or a tolerance?
# > 2. *Derive aloud:* dL/dz = σ(z) − y, so ∇w = Xᵀ(p − y)/n + λw and ∂b = mean(p − y). That one line is the whole model.
# > 3. *Stability:* σ for very negative z: `np.exp(-z)` overflows → use a piecewise sigmoid or `scipy.special.expit`; compute the loss with `np.logaddexp(0, z) - y*z`, never `log(sigmoid)`.
# > 4. *Complexity:* O(n·d) per epoch, O(epochs·n·d) total; memory O(n + d). At scale: mini-batch SGD, sparse X, early stopping on a validation loss.
# > 5. *Edge cases:* perfectly separable data without L2 → weights grow forever (L2 or early stopping fixes it); all labels one class; unscaled features → lr must shrink (standardise first).

# %% [markdown]
# > **Exercise 2.** *(target 15 min)* Implement `softmax_grad(W, X, y)` →
# > `(loss, dW)` for multinomial logistic regression without a bias (append a
# > column of ones to X if you want one): `W (d, C)`, `X (n, d)`, integer labels
# > `y (n,)` in `0..C-1`. `loss` is the **mean** cross-entropy; `dW` has `W`'s
# > shape. Checked: the loss equals the reference; `dW` passes a numerical
# > gradient check (relative error < 1e-6); logits in the thousands give a finite,
# > correct loss; `W` is not modified.

# %%
def softmax_grad(W, X, y):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(softmax_grad)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Shapes first:* logits `Z = XW` is `(n, C)`; the one-hot target `Y` is `(n, C)`, built by indexing, not by a loop.
# > 2. *The key fact:* for softmax + cross-entropy, ∂L/∂Z = (softmax(Z) − Y)/n. Then dW = Xᵀ(P − Y)/n. Offer to derive it (notebook 14, Exercise 7, walks through it).
# > 3. *Stability:* subtract the row max before `exp`; loss = mean(logsumexp(Z) − Z[i, yᵢ]).
# > 4. *Verify:* say you would run a numerical gradient check on a 5×3 problem, and do it if time allows.
# > 5. *Complexity:* O(n·d·C) time, O(n·C) memory for the probabilities. With a huge C (a vocabulary or item catalogue), mention sampled softmax / negative sampling.

# %% [markdown]
# > **Exercise 3.** *(target 20 min)* Implement `kmeans(X, k, seed=0, n_iter=100)`
# > → `(centers (k, d), labels (n,))` with **k-means++ initialisation**
# > (first center uniformly at random; each next one sampled with probability
# > proportional to the squared distance to the nearest center chosen so far),
# > then Lloyd iterations until the labels stop changing or `n_iter` is reached.
# > Use `np.random.default_rng(seed)` so a seed reproduces the result.
# >
# > Checked: `labels` are each point's nearest returned center; on five
# > separated blobs the median inertia over seeds 0-4 is within 2% of
# > `sklearn.cluster.KMeans(n_init=10)`; the same seed gives the same centers;
# > `k=1` gives the mean; duplicated points do not produce NaN (empty clusters).

# %%
def kmeans(X, k, seed=0, n_iter=100):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(kmeans)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Objective:* inertia = Σ‖xᵢ − μ_{c(i)}‖². Both steps (assign, re-average) can only lower it, so Lloyd converges, but to a **local** minimum: that is why the init matters.
# > 2. *Why k-means++:* D² sampling spreads the initial centers; it is O(log k)-competitive in expectation, versus arbitrarily bad for uniform init. Production libraries also run several inits (`n_init`) and keep the best.
# > 3. *Vectorise:* the (n, k) distance matrix with the expansion trick; `argmin` over axis 1.
# > 4. *Edge cases:* an empty cluster (keep the old center, or re-seed it at the farthest point); duplicate points; k > number of distinct points; features on different scales (standardise).
# > 5. *Complexity:* O(n·k·d) per iteration; init O(n·k·d). At scale: mini-batch k-means.

# %% [markdown]
# > **Exercise 4.** *(target 12 min)* Implement `knn_predict(X_train, y_train, X_test, k)`
# > → predicted labels `(m,)` by majority vote among the `k` nearest training
# > points (Euclidean). A tie between labels goes to the **smallest** label.
# > Labels may be integers or strings. No Python loop over training points.
# >
# > Checked: agreement with a brute-force reference for k = 1, 4, 7; `k=1` on
# > the training set returns `y_train`; vote ties; `k = n_train`; string labels;
# > a performance check (up to 4,000 training × 1,000 test points in 16-d, 2-second budget).

# %%
def knn_predict(X_train, y_train, X_test, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex4(knn_predict)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Clarify:* distance metric? Tie-breaking? Memory limit (m × n distances)?
# > 2. *Plan:* distances with the expansion trick → `np.argpartition(d2, k-1, axis=1)[:, :k]` (O(n) per row instead of a full sort) → map labels to 0..C−1 with `np.unique(..., return_inverse=True)` → count votes with `np.add.at`.
# > 3. *Complexity:* O(m·n·d) for distances + O(m·n) for selection; memory O(m·n) → process the test set in chunks if it does not fit.
# > 4. *Scale story:* exact kNN does not scale to millions of items; approximate nearest neighbours (FAISS, HNSW; notebook 08) trade a little recall for orders of magnitude in latency.
# > 5. *Edge cases:* ties in distance, ties in votes, k > n, duplicate training points, unscaled features dominating the distance.

# %% [markdown]
# > **Exercise 5.** *(target 15 min)* Implement `roc_auc(y_true, scores)` without
# > scikit-learn, in O(n log n), using the **rank statistic** (Mann-Whitney U):
# >
# > $$\text{AUC} = \frac{\sum_{i \in \text{pos}} r_i - n_+(n_+ + 1)/2}{n_+\, n_-},$$
# >
# > where rᵢ is the 1-based rank of score i among all scores and **tied scores
# > share their average rank** (so a positive-negative tie counts ½). Raise
# > `ValueError` if only one class is present.
# >
# > Checked: equal to `sklearn.metrics.roc_auc_score` on continuous and on
# > heavily tied scores; all-tied → 0.5; perfect and reversed rankings; boolean
# > labels; a single class → `ValueError`; timing up to n = 256,000 (2-second budget).

# %%
def roc_auc(y_true, scores):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex5(roc_auc)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Define it two ways:* the area under TPR-vs-FPR, and the probability that a random positive outscores a random negative (ties ½). The second gives the algorithm.
# > 2. *Brute force first, in one sentence:* compare every positive with every negative, O(n₊·n₋), then improve.
# > 3. *Sort once:* average ranks for tie groups (run boundaries from `np.diff` on the sorted scores), sum the positives' ranks, subtract the minimum possible sum n₊(n₊+1)/2.
# > 4. *Complexity:* O(n log n) time for the sort, O(n) memory.
# > 5. *Edge cases:* one class (undefined, raise or return NaN and say which), ties, NaN scores (reject), labels as bool/±1. Mention: AUC ignores calibration and class imbalance; PR-AUC is more informative for rare positives (Level 1, notebook 05).

# %% [markdown]
# > **Exercise 6.** *(target 10 min)* Implement `ndcg_at_k(scores, labels, k)`
# > for **one query**: rank the items by `scores` descending (ties keep the input
# > order: use a stable sort), then
# >
# > $$\text{DCG@k} = \sum_{i=1}^{\min(k, n)} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}},$$
# >
# > where IDCG uses the labels sorted descending. Return 0.0 when no item is
# > relevant. The grader compares with `shoprank.metrics.ndcg_at_k`, the
# > implementation every Level 2 notebook uses
# > ([`src/shoprank/metrics.py`](../../src/shoprank/metrics.py)) — try it before reading that file.

# %%
def ndcg_at_k(scores, labels, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex6(ndcg_at_k)

# %% [markdown]
# > 🗣️ **Talk it through.**
# > 1. *Clarify the conventions out loud* — they change the number: exponential (2^rel − 1) or linear gain? log₂(i+1) discount with 1-based i? How are score ties broken? What is NDCG for a query with no relevant item (0, or skip the query)? Is IDCG computed over the top k of *all* candidates or of the shown list?
# > 2. *Complexity:* O(n log n) for the sort (O(n + k log k) with `argpartition`); computing many queries at once is a group-by.
# > 3. *Context:* in notebook 05 you saw that NDCG on logged clicks rewards the logging policy's own order (position bias); the formula is easy, choosing the labels is the hard part.

# %% [markdown]
# ## 13.6 · Score yourself
#
# After each exercise, give yourself 0-2 on each axis of the table in 13.1
# (framing, correctness, numerical care, efficiency, verification,
# communication). A 9-12 is a pass for most loops; below 8, redo the exercise
# tomorrow from a blank cell. Record the time you took against the target —
# over 1.5× the target means the method is not yet automatic.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `RuntimeWarning: overflow encountered in exp`, then NaN | exponentiating large logits | subtract the row max; use `np.logaddexp`; piecewise sigmoid |
# | Exercise 1: coefficients close but not within 0.02 of sklearn | regularising the intercept, or using the sum (not mean) loss | penalise `w` only; `grad_w = X.T @ (p - y) / n + l2 * w` |
# | Exercise 1: loss goes up | learning rate too large for the feature scale, or the sign of the update | `w -= lr * grad`; standardise features |
# | Gradient check: relative error ≈ 0.33 or 0.5 | a missing factor 2 or 1/n | re-derive on paper; check both sides use the same mean/sum |
# | Gradient check: relative error ≈ 1e-3 everywhere | float32 inputs | cast to float64 for the check |
# | Exercise 3: inertia far above sklearn's | uniform random init, or sampling ∝ distance instead of distance² | `p = d2 / d2.sum()` with d2 the squared distance to the nearest chosen center |
# | Exercise 3: NaN centers | an empty cluster's mean | keep the previous center when a cluster is empty |
# | `MemoryError` in kNN | broadcasting `(m, n, d)` | use the expansion trick, or chunk the test set |
# | Exercise 5 differs from sklearn only with ties | ordinal ranks instead of average ranks | give each run of equal scores the mean of its ranks |
# | A grader says "took 3.1 s at n=…" | a Python loop over pairs | vectorise, or sort once and use prefix sums |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Derive the gradient of the binary cross-entropy with respect to the logit. Why does the same "prediction minus target" form appear for softmax, and for linear regression with squared loss?
# 2. Your softmax returns NaN on some batches in training. Give two causes and the fix for each.
# 3. Why does logistic regression's loss have a unique minimum while k-means' does not? What does that imply for how you initialise and evaluate each?
# 4. ROC-AUC is 0.92 on a dataset with 0.5% positives, yet the model is useless in production. How is that possible, and what would you report instead?
# 5. kNN works well offline with 50k items. The catalogue grows to 50M items with a 20 ms latency budget. What changes, and what do you give up?
# 6. An interviewer asks how you know your gradient is right. Describe a gradient check: ε, the error measure, the threshold, and two situations where it misleads.
#
# ➡️ **Next:** [`14_ml_from_scratch_2.ipynb`](14_ml_from_scratch_2.ipynb) — trees, boosting, backprop, attention and batch norm from scratch.
