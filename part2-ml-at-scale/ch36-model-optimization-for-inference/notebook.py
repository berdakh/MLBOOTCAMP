# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 36 · Model optimisation for inference
#
# **Part II · Chapter 36 · about 90 minutes · CPU core · optional GPU extension**
#
# A ranker scores hundreds of candidates for every request, millions of times
# a day, so a model that is 10× cheaper per row is often worth more than one
# that is 1 % better. This lab takes the ranker of Chapter 35 and makes
# it cheaper in three ways (a portable runtime, lower precision and a smaller
# model taught by a bigger one), measuring every step on two axes at once:
# milliseconds and quality.
#
# After this lab you should be able to:
#
# 1. export a PyTorch scoring model to ONNX and run it with ONNX Runtime;
# 2. write a numerical parity check and know when exact parity is the wrong test;
# 3. apply dynamic int8 quantisation in PyTorch and ONNX Runtime, and measure what it actually gains;
# 4. distil a LightGBM teacher into a small MLP student with a tempered soft-label loss;
# 5. build the size / latency / quality table that decides which model is deployed, and state when optimisation is not worth it.
#
# **Outputs:** `artifacts/level2/optimize/` with `.pt`, `.onnx` and
# `.txt` model files and `optimization_table.csv`.

# %% [markdown]
# Read first: [Chapter 36 text](README.md), which explains the concepts this lab measures. Worked answers are in [solutions.md](solutions.md).

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: data, the notebook-17 ranker (teacher) and evaluation sets ----
import time

import numpy as np
import pandas as pd
import torch

from shoprank import optimize as opt
from shoprank import serving as sv
from shoprank import store
from grading import ch36 as grade

RUN_GPU_EXTENSION = False  # optional section 18.7; needs a CUDA runtime
torch.set_num_threads(2)
OUT = ART / "level2" / "optimize"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
world, users, items, imps = store.load("small")
# Same point-in-time setup as Chapter 35: features from days 0-34, labels 35-41.
_, lgbm, train_frame = sv.train_ranker(imps, items, len(users), feature_days=(0, 34), label_days=(35, 41))
state = sv.build_state(imps, items, len(users), feature_days=(0, 41))  # serving state for the valid week
X_train = train_frame[sv.FEATURES].to_numpy(np.float32)
y_train = train_frame["clicked"].to_numpy()
print(f"train rows {len(X_train):,} x {X_train.shape[1]} features | ready in {time.time() - t0:.0f}s")

# %% [markdown]
# **Expected:** `train rows 300,000 x 10 features` after 15–60 s (data
# generation plus the LightGBM ranker of Chapter 35).
#
# Two quality measures on the **validation week**, both against the hidden
# truth (`world.true_click_prob`), never against clicks:
#
# * **Cascade NDCG@10**: for 1,000 validation requests, score the ~300
#   retrieval candidates of Chapter 35's cascade and compare the top 10 with
#   the user's true best 10 in the whole catalogue (no category cap). This is
#   what the ranker is *for*, and it has a wide range (popularity ≈ 0.39,
#   perfect = 1).
# * **Re-rank NDCG@10**: re-order the 10 items users were actually shown in
#   20,000 logged requests. Narrower range (random order ≈ 0.79), but it
#   needs no retrieval.

# %%
valid = imps[imps["day"].between(store.VALID_DAYS[0], store.VALID_DAYS[-1])]
rng = np.random.default_rng(0)

# (a) cascade candidates for 1,000 requests -> one big feature matrix + offsets
cascade = sv.Cascade(state, lgbm, store=sv.OnlineStore(state, latency_s=0.0))
req = valid.groupby("request_id")[["user_id", "day"]].first().sample(1000, random_state=0)
P = sv.true_prob_matrix(world, req["user_id"].to_numpy(), req["day"].to_numpy())
blocks, cands = [], []
for u, d in zip(req["user_id"], req["day"]):
    user, _ = cascade._user(u)
    c, _ = cascade.retrieve(u, d, user)
    cands.append(c)
    blocks.append(cascade.features(u, user, c, d))
X_cand = np.vstack(blocks)
offsets = np.cumsum([0] + [len(c) for c in cands])

# (b) 20,000 logged requests, re-ranked
keep = rng.choice(valid["request_id"].unique(), 20_000, replace=False)
logged = valid[valid["request_id"].isin(keep)]  # sorted by request, position
X_logged = sv.pair_features(state, logged["user_id"].to_numpy(), logged["item_id"].to_numpy(),
                            logged["day"].to_numpy(), position=0)
rel_logged = world.true_click_prob(logged["user_id"].to_numpy(), logged["item_id"].to_numpy())


def quality(score_fn) -> dict:
    s = score_fn(X_cand)
    tops = [cands[j][np.argsort(-s[offsets[j]:offsets[j + 1]], kind="stable")][:10] for j in range(len(cands))]
    return {"cascade NDCG@10": float(sv.catalogue_ndcg(P, tops).mean()),
            "re-rank NDCG@10": float(np.nanmean(sv.ndcg_by_request(score_fn(X_logged), rel_logged)))}


print(f"cascade set: {X_cand.shape[0]:,} candidate rows for {len(cands)} requests | logged set: {len(X_logged):,} rows")
q_lgbm = quality(lambda X: sv.score_with(lgbm, X))
print("LightGBM (teacher):", {k: round(v, 3) for k, v in q_lgbm.items()})
print("random scores     :", {k: round(v, 3) for k, v in quality(lambda X: rng.random(len(X))).items()})

# %% [markdown]
# **Expected:** 317,639 candidate rows and 200,000 logged rows. LightGBM:
# cascade NDCG@10 0.875, re-rank NDCG@10 0.966. Random scores: 0.46 and 0.79
# (random order over *good candidates* is already far better than
# popularity: retrieval does real work).

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | ONNX | a portable file format for a model's computation graph | `mlp.onnx`, with a dynamic batch dimension |
# | ONNX Runtime (ORT) | a C++ engine that executes ONNX graphs without Python in the loop | about 0.006 ms per call at batch 1 |
# | Eager-mode overhead | the fixed Python cost PyTorch pays per operation, dominant for small inputs | PyTorch is 4–6× slower than ORT at batch 1 |
# | Parity check | comparing two implementations' outputs element by element within a tolerance | max difference 1.2e-6 after export |
# | Tolerance (atol) | the largest absolute difference still counted as equal | about 1e-5 for lossless steps |
# | Dynamic int8 quantisation | weights stored as 8-bit integers ahead of time, activations quantised per batch | `DynamicQuantizedLinear` layers |
# | Scale and zero-point | the two numbers that map floats to integers and back | see the figure in 36.3 |
# | Spearman correlation | the correlation of ranks: do two models order items the same way? | 0.997 between ORT int8 and fp32 |
# | Knowledge distillation | training a small student to match a large teacher's probabilities | a 193-parameter student taught by LightGBM |
# | Soft labels | the teacher's probabilities used as training targets instead of 0/1 clicks | `teacher_probs` |
# | Temperature T | a divisor on logits that softens probabilities so near-misses carry signal | `T = 2.0` |
#
# The notebook on one page:
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-inference-optimization.svg" alt="Lossless path: PyTorch to ONNX to ONNX Runtime with an exact parity test; lossy path: LightGBM teacher to distilled student to int8 with ranking-quality checks" width="100%">
#
# *What to notice:* the two rows need different tests: exact parity cannot pass for a lossy step, and ranking checks are too weak to catch a broken export.

# %% [markdown]
# ## 36.1 · A PyTorch scoring model
#
# The model to optimise: a small MLP over the same 10 features
# (`opt.RankMLP`). One design choice matters for everything that follows:
# the **input standardisation is a layer** (mean and std stored as buffers),
# so the exported model contains its own preprocessing. Training/serving skew
# from a re-implemented scaler (Chapter 11) becomes impossible.

# %%
t0 = time.time()
mlp = opt.train_mlp(X_train, y_train, hidden=(128, 64), epochs=3, seed=0)
print(f"trained in {time.time() - t0:.1f}s | {opt.n_params(mlp):,} parameters")
q_mlp = quality(lambda X: opt.torch_scores(mlp, X))
print("MLP 128-64:", {k: round(v, 3) for k, v in q_mlp.items()})

# %% [markdown]
# **Expected:** loss falling from about 0.326 to 0.311 over three epochs in a
# few seconds; 9,729 parameters; cascade NDCG@10 about 0.86 and re-rank
# NDCG@10 0.961 — a little below LightGBM (0.875 / 0.966).

# %% [markdown]
# ## 36.2 · ONNX and ONNX Runtime
#
# **ONNX** is a file format for computation graphs; **ONNX Runtime (ORT)** is
# a C++ inference engine for it. Why use them when PyTorch can already predict?
#
# * **Overhead.** PyTorch's eager mode dispatches each op through Python;
#   ORT runs a fused, pre-planned graph. For small models and small batches,
#   that overhead *is* the latency.
# * **Portability.** The serving stack (C++, Java, Go, mobile) does not need
#   Python or PyTorch, and the model file is self-contained.
# * **Optimisation hooks.** Graph fusions, quantisation, and hardware
#   providers (CUDA, TensorRT, OpenVINO, CoreML) behind the same API.
#
# The model is exported with a *dynamic* batch dimension so one file serves batch 1 and
# batch 1,024.
#
# > **Intuition.** PyTorch eager is a chef reading the cookbook aloud at every step:
# > negligible for a banquet (batch 1,024), most of the time for one sandwich (batch 1).
# > ONNX export prints the recipe as a card that any kitchen can follow without the
# > narration. *Where it breaks down:* the card records only the path taken while tracing;
# > Python `if`s that depend on the data are frozen at export time.

# %%
onnx_path = opt.export_onnx(mlp, OUT / "mlp.onnx", n_features=X_train.shape[1])
session = opt.ort_session(onnx_path, threads=1)
print("inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
print("outputs:", [(o.name, o.shape) for o in session.get_outputs()])
s_torch = opt.torch_scores(mlp, X_logged)
s_ort = opt.ort_scores(session, X_logged)
print(f"max |torch - onnxruntime| over {len(s_torch):,} rows: {np.abs(s_torch - s_ort).max():.2e}")

# %% [markdown]
# **Expected:** input `features` with shape `['batch', 10]`, output `logit`
# with shape `['batch']`, and a maximum difference of about 1.2e-6: float32
# rounding from a different order of operations, not a bug.
#
# A parity check belongs in the deployment pipeline, as a test that blocks
# the release. Write the comparison function it calls.
#
# > **Exercise 1.** Write `parity(a, b, atol)` that returns a dict with
# > `max_abs_diff` (float), `n_over` (number of elements with
# > `|a − b| > atol`, counting any NaN as over) and `ok` (True only if
# > `n_over == 0`). Accept lists or arrays; raise `ValueError` if the shapes
# > differ (no broadcasting: `(n,)` vs `(n, 1)` is a classic export bug).
# > If any NaN is present, `max_abs_diff` may be `inf` or `nan`, but `ok`
# > must be False.

# %%
def parity(a, b, atol):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(parity)

# %% [markdown]
# ## 36.3 · Dynamic int8 quantisation
#
# *Dynamic* quantisation stores the weights of linear layers as 8-bit
# integers (4× smaller than float32) and quantises activations on the fly,
# per batch. No calibration data is needed, which is why it is the first
# thing to try. Two implementations:

# %%
int8_onnx_path = opt.quantize_onnx_int8(onnx_path, OUT / "mlp_int8.onnx")
session_int8 = opt.ort_session(int8_onnx_path, threads=1)
mlp_int8 = opt.torch_int8(mlp)
print(mlp_int8.net)

s_ort8 = opt.ort_scores(session_int8, X_logged)
s_torch8 = opt.torch_scores(mlp_int8, X_logged)
top10 = lambda s: set(np.argsort(-s)[:1000])  # noqa: E731 - the 1,000 highest-scored rows
for name, s in [("ORT int8", s_ort8), ("torch int8", s_torch8)]:
    print(f"{name:<10} max |diff| vs fp32 {np.abs(s - s_torch).max():.3f} | "
          f"Spearman {pd.Series(s).corr(pd.Series(s_torch), method='spearman'):.4f} | "
          f"top-1000 overlap {len(top10(s) & top10(s_torch)) / 1000:.3f}")
q_ort8 = quality(lambda X: opt.ort_scores(session_int8, X))
q_torch8 = quality(lambda X: opt.torch_scores(mlp_int8, X))
print("ORT int8  :", {k: round(v, 3) for k, v in q_ort8.items()})
print("torch int8:", {k: round(v, 3) for k, v in q_torch8.items()})

# %% [markdown]
# **Expected:** `Linear` layers replaced by `DynamicQuantizedLinear`. Maximum
# logit differences versus float32 of about 0.47 (ORT) and 0.94 (PyTorch) —
# the parity check of Exercise 1 would *fail* at any sensible `atol` — yet
# Spearman correlations of 0.997 and 0.992, top-1000 overlaps of about 0.94
# and 0.88, re-rank NDCG unchanged (0.961) and cascade NDCG down by only
# 0.003–0.007.
#
# The conclusion is that for a lossy optimisation **exact parity is the wrong
# test**. Test what the model is used for: ranking agreement (Spearman,
# top-k overlap) and the offline metric, with thresholds agreed in advance.
# Keep exact parity (`atol` ~1e-5) for *lossless* steps such as the ONNX
# export itself.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-quantization.svg" alt="A float number line from -1.2 to 2.3 mapped onto int8 values from -128 to 127, with the scale and zero-point formulas and a worked example" width="100%">
#
# *What to notice:* zero maps to an integer exactly, and every other value moves by at most half a step (s/2 ≈ 0.007 here).
#
# > **Intuition.** Quantisation is measuring every weight with a ruler that has 256 marks
# > instead of an almost continuous one: individual readings move (here the logits moved by
# > up to 0.47), but which item is longer than which mostly does not (Spearman 0.997).
# > *Where it breaks down:* one huge outlier stretches the ruler, so all the ordinary
# > values crowd onto a few marks; per-channel scales and outlier handling exist for that.

# %% [markdown]
# ## 36.4 · Distillation: a small student taught by a big teacher
#
# The LightGBM teacher is accurate but costs milliseconds per 1,000 rows (measured
# below). A tiny MLP with one hidden layer of 16 units costs
# almost nothing, but trained on clicks alone it learns from noisy 0/1
# labels. **Knowledge distillation** trains the student on the teacher's
# *probabilities* as well: soft labels carry "this item is a 0.3, that one a
# 0.05" information that a click/no-click never does.
#
# For a binary task with temperature `T` (Hinton et al., 2015, adapted to a
# sigmoid):
#
# $$\mathcal{L} = \alpha\,T^2\,\mathrm{BCE}\big(\sigma(z_s/T),\ \sigma(z_t/T)\big) + (1-\alpha)\,\mathrm{BCE}\big(\sigma(z_s),\ y\big)$$
#
# where $z_s$ are student logits, $z_t = \mathrm{logit}(p_t)$ the teacher's
# logits and $y$ the clicks. $T > 1$ softens both sides so the student also
# learns the teacher's ordering among low-probability items; $T^2$ keeps the
# gradient scale comparable across temperatures.
#
# > **Intuition.** A teacher who marks only right or wrong teaches less than one who says
# > "this answer is 70% right, and that wrong one was a reasonable guess". Soft labels are
# > the second teacher; temperature makes the teacher talk more about the near-misses.
# > *Where it breaks down:* the student can only learn what its own inputs can express; a
# > teacher that sees more features passes on more than one that sees the same ten.
#
# > **Exercise 2.** Write `distillation_loss(student_logits, teacher_probs, labels, alpha, T)`
# > (all tensors of shape `(n,)`) that returns the scalar mean loss above. Use
# > `F.binary_cross_entropy_with_logits` (numerically stable). Clamp
# > `teacher_probs` to `[1e-6, 1 − 1e-6]` before `torch.logit`.

# %%
from torch.nn import functional as F  # noqa: F401 - for your solution


def distillation_loss(student_logits, teacher_probs, labels, alpha, T):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(distillation_loss)

# %% [markdown]
# `opt.train_mlp(..., teacher_probs=...)` uses the module's version,
# `opt.distillation_loss`. Train the same 16-unit student twice, on clicks
# alone and distilled from LightGBM, with the same seed and epochs:

# %%
teacher_probs = lgbm.predict_proba(train_frame[sv.FEATURES])[:, 1]
student_hard = opt.train_mlp(X_train, y_train, hidden=(16,), epochs=3, seed=0, verbose=False)
student_kd = opt.train_mlp(X_train, y_train, hidden=(16,), epochs=3, seed=0, verbose=False,
                           teacher_probs=teacher_probs, alpha=0.7, T=2.0)
q_hard = quality(lambda X: opt.torch_scores(student_hard, X))
q_kd = quality(lambda X: opt.torch_scores(student_kd, X))
print(f"student: {opt.n_params(student_kd)} parameters ({opt.n_params(mlp) / opt.n_params(student_kd):.0f}x fewer than the 128-64 MLP)")
print("student, clicks only :", {k: round(v, 4) for k, v in q_hard.items()})
print("student, distilled   :", {k: round(v, 4) for k, v in q_kd.items()})
print("teacher (LightGBM)   :", {k: round(v, 4) for k, v in q_lgbm.items()})

# %% [markdown]
# **Expected:** 193 parameters (50× fewer). Cascade NDCG@10: student on
# clicks 0.843, distilled student 0.858, teacher 0.875 — distillation
# recovers almost half of the gap. Re-rank NDCG barely differs (0.958 vs
# 0.959). We re-ran seeds 0, 1 and 2 outside the lab: the distilled
# student was ahead by 0.015–0.021 every time. Such a check is needed before
# claiming that "distillation helps": one seed is an anecdote.
#
# Why not more? The features are few and dense, and 300,000 rows is plenty
# for 193 parameters; the student is limited mostly by its capacity. Distillation pays most when
# the teacher sees something the student cannot (more features, a long
# history, a cross-encoder in search) or when the student's own labels are
# scarce.

# %% [markdown]
# ## 36.5 · The table that decides
#
# Size on disk, median latency per call (single thread: the serving
# configuration where many requests run in parallel) for batches of 1, 64
# and 1,024 rows, and quality. Batch 64 is the row to read for a ranker that
# scores a few hundred candidates per request in chunks; batch 1 shows fixed
# overhead.

# %%
torch.set_num_threads(1)
lgbm_path = OUT / "lightgbm.txt"
lgbm.booster_.save_model(str(lgbm_path))
torch.save(mlp.state_dict(), OUT / "mlp.pt")
torch.save(mlp_int8.state_dict(), OUT / "mlp_int8.pt")
student_path = opt.export_onnx(student_kd, OUT / "student.onnx", n_features=X_train.shape[1])
session_student = opt.ort_session(student_path, threads=1)
torch.save(student_kd.state_dict(), OUT / "student.pt")

candidates = {
    "LightGBM teacher (150 trees)": (lambda X: sv.score_with(lgbm, X), lgbm_path, q_lgbm),
    "MLP 128-64, PyTorch fp32": (lambda X: opt.torch_scores(mlp, X), OUT / "mlp.pt", q_mlp),
    "MLP 128-64, PyTorch int8": (lambda X: opt.torch_scores(mlp_int8, X), OUT / "mlp_int8.pt", q_torch8),
    "MLP 128-64, ONNX Runtime fp32": (lambda X: opt.ort_scores(session, X), onnx_path, quality(lambda X: opt.ort_scores(session, X))),
    "MLP 128-64, ONNX Runtime int8": (lambda X: opt.ort_scores(session_int8, X), int8_onnx_path, q_ort8),
    "student 16, distilled, PyTorch": (lambda X: opt.torch_scores(student_kd, X), OUT / "student.pt", q_kd),
    "student 16, distilled, ONNX Runtime": (lambda X: opt.ort_scores(session_student, X), student_path,
                                            quality(lambda X: opt.ort_scores(session_student, X))),
}
rows = []
for name, (fn, path, q) in candidates.items():
    ms = opt.bench(fn, X_logged, batch_sizes=(1, 64, 1024))
    rows.append({"model": name, "size KB": opt.size_kb(path), **{f"ms @ batch {b}": v for b, v in ms.items()}, **q})
table = pd.DataFrame(rows).set_index("model")
table.to_csv(OUT / "optimization_table.csv")
torch.set_num_threads(2)
table.round(3)

# %% [markdown]
# **Expected (our runs; milliseconds will differ on other machines, the ordering
# should not):**
#
# * **Size:** LightGBM's text file about 517 KB; the fp32 MLP about 40 KB;
#   int8 about 15 KB (the 4× applies to weights only; biases, buffers and
#   graph metadata stay); the student 1.6 KB (ONNX) to 3.5 KB (`.pt`).
# * **Latency at batch 1,024:** LightGBM 5–6.5 ms; the PyTorch MLP
#   0.24–0.34 ms and ONNX Runtime about 0.18 ms; int8 *not* faster (PyTorch
#   int8 is slower, 0.42–0.49 ms); the student 0.035–0.055 ms.
# * **Latency at batch 1:** ONNX Runtime about 0.006 ms, 4–6× faster than
#   PyTorch eager (0.025–0.034 ms) — fixed per-call overhead dominates tiny
#   inputs. LightGBM is just as fast as PyTorch at batch 1: tree cost grows
#   with rows, not with a fixed dispatch overhead.
# * **Quality:** every MLP within 0.02 of the teacher on cascade NDCG, and
#   within 0.007 on re-rank NDCG.
#
# Two findings deserve a sentence each in a design review.
#
# 1. **int8 did not speed this model up.** With 10 inputs and ~10 K weights,
#    the matrix multiplies are trivial and quantising/dequantising activations
#    costs as much as it saves. Quantisation pays when weights dominate memory
#    bandwidth: large embedding tables, wide layers, transformers.
# 2. **The largest gain for the least effort came from changing the model class.** Moving from 150
#    trees to the 128-64 MLP cut batch-1,024 latency 20–35×, and to the
#    distilled student about 100×, for 0.014–0.017 of cascade NDCG — a cost
#    the teacher partly recovered. Whether that trade is worth it is a
#    product question: what is 0.017 NDCG worth versus 5 ms of budget?

# %% [markdown]
# ## 36.6 · When is optimisation worth it?
#
# A back-of-envelope estimate comes before engineering: if the ranker is 30 % of a
# 60 ms p99 and its time is halved, 9 ms of p99 are saved, which might allow scoring
# 2× the candidates in the same budget (a quality gain) or running 30 % fewer
# machines (a cost gain). If the ranker is 2 % of the latency, a 10×
# speed-up is invisible to users. **Profile first** (Chapter 35's stage
# timings); optimise the stage that dominates.
#
# | Technique | Typical gain | Typical cost | Worth it when |
# |---|---|---|---|
# | ONNX / a compiled runtime | 1.2–5× on small models | an export step, parity tests | Python overhead dominates; non-Python serving stack |
# | Dynamic int8 | 2–4× smaller; 1–3× faster on big linear layers | small accuracy loss; per-op support | memory-bound models (embeddings, transformers) |
# | Static int8 / fp16 on GPU | 2–4× throughput | calibration data, hardware support | large batch, GPU serving |
# | Distillation | 10–100× fewer FLOPs | training pipeline for the student, teacher upkeep | the teacher is too slow for the latency budget |
# | Pruning / smaller features | varies | feature-pipeline work | features dominate cost (fetch time!) |
# | Caching scores | ~∞ for hits | staleness | the same (user, item) pairs repeat |
#
# > **Exercise 3 (written).** A ranker is a 6-layer transformer over the
# > user's last 50 actions plus 300 candidates; it takes 45 ms of a 100 ms
# > p99 budget on CPU, and the product team wants to add 200 more candidates.
# > Using this lab's table as evidence of *how* each technique behaves,
# > propose an optimisation plan: what to try first, second and
# > third, what to measure at each step (name the parity and quality
# > checks and their thresholds), and what would justify stopping.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 36.7 · Optional: GPU inference notes
#
# Not run by default (`RUN_GPU_EXTENSION = False`) and not needed later.
# Install `onnxruntime-gpu` instead of `onnxruntime` and pick a GPU runtime.
# What to expect, stated as expectations because our verification runs are
# CPU only: on a T4, a model this small is *slower* per request on the GPU
# than on the CPU (kernel-launch and host-to-device copy overhead dominate
# below batch sizes of a few thousand); GPUs win for large batches and large
# models, especially with fp16 (`CUDAExecutionProvider`) or TensorRT
# (`TensorrtExecutionProvider`). Not measured in our verification runs.

# %%
if RUN_GPU_EXTENSION and torch.cuda.is_available():
    import onnxruntime as ort

    print("ORT providers:", ort.get_available_providers())
    gpu_session = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("GPU ORT:", opt.bench(lambda X: gpu_session.run(None, {"features": X})[0], X_logged, (1, 64, 1024, 65536)))
    mlp_gpu = mlp.to("cuda").half()

    def gpu_torch(X):
        with torch.inference_mode():
            return mlp_gpu(torch.from_numpy(X).to("cuda").half()).float().cpu().numpy()

    print("GPU torch fp16:", opt.bench(gpu_torch, X_logged, (1, 64, 1024, 65536)))
else:
    print("GPU extension skipped (RUN_GPU_EXTENSION = False or no CUDA device)")

# %% [markdown]
# ## Key equations
#
# **1. Affine (asymmetric) int8 quantisation**
#
# $$
# s = \frac{x_{\max} - x_{\min}}{255}, \qquad z = \operatorname{round}\!\left(-128 - \frac{x_{\min}}{s}\right), \qquad q = \operatorname{clip}\!\left(\operatorname{round}(x/s) + z,\ -128,\ 127\right), \qquad \hat x = s\,(q - z)
# $$
#
# *In words:* a scale turns float units into integer steps, a zero-point shifts so that 0.0 lands exactly on an integer; the rounding error per value is at most $s/2$.
# *Example:* range [−1.2, 2.3]: $s = 0.0137$, $z = -41$; $1.0 \to 73 - 41 = 32 \to 0.0137 \times 73 = 1.002$.
#
# **2. Distillation loss with temperature**
#
# $$
# \mathcal{L} = \alpha\, T^2\, \operatorname{BCE}\big(\sigma(z_s/T),\ \sigma(z_t/T)\big) + (1 - \alpha)\, \operatorname{BCE}\big(\sigma(z_s),\ y\big)
# $$
#
# *In words:* match the teacher's softened probabilities and the real labels. The soft term's gradient with respect to $z_s$ is $(\sigma(z_s/T) - \sigma(z_t/T))/T$, which shrinks roughly like $1/T^2$ for large $T$; multiplying by $T^2$ keeps the two terms' balance independent of the temperature.
# *Example:* $z_s = 1$, $z_t = 2$: the soft-term gradient is −0.150 at T = 1, −0.054 at T = 2, −0.015 at T = 4; times $T^2$: −0.150, −0.217, −0.241 (tending to $\sigma'(0)(z_s - z_t) = -0.25$).
#
# **3. Parity check**
#
# $$
# \max_i \lvert a_i - b_i \rvert \le \text{atol} \quad (\text{and } \lvert a_i - b_i \rvert \le \text{atol} + \text{rtol}\,\lvert b_i \rvert \text{ in } \texttt{np.allclose})
# $$
#
# *In words:* exact-within-tolerance agreement; right for lossless steps (export), wrong for lossy ones (int8, distillation), which are judged by ranking agreement instead.
#
# **4. Spearman rank correlation** (no ties)
#
# $$
# \rho_s = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}
# $$
#
# *In words:* the Pearson correlation of the ranks; 1 means both models order every item identically.
# *Symbols:* $d_i$ = the difference between item $i$'s two ranks.
# *Example:* ranks (1, 2, 3, 4) vs (1, 3, 2, 4): $\sum d^2 = 2$, $\rho_s = 1 - 12/60 = 0.8$.
#
# **5. Amdahl's law: what a speed-up is worth end to end**
#
# $$
# \text{overall speed-up} = \frac{1}{(1 - f) + f / k}
# $$
#
# *Symbols:* $f$ = the share of latency in the optimised stage, $k$ = that stage's speed-up.
# *Example:* the ranker is 30% of a 60 ms p99 and becomes 2× faster: $1 / (0.7 + 0.15) = 1.18$×, i.e. 9 ms saved, as in 36.6. Made infinitely fast, the best possible is $1/0.7 = 1.43$×.
#
# **6. Model size**
#
# $$
# \text{bytes} \approx \text{parameters} \times \text{bytes per parameter}
# $$
#
# *Example:* 9,729 parameters × 4 B = 38.9 KB in fp32 (the file is about 40 KB); int8 weights need about a quarter of that, but biases, buffers and graph metadata stay, so the int8 file is about 15 KB.

# %%
# --- Check the key equations numerically -------------------------------------
lo, hi = -1.2, 2.3
s_q = (hi - lo) / 255
z_q = int(np.round(-128 - lo / s_q))
x_q = np.array([-1.2, 0.0, 1.0, 2.3])
q_q = np.clip(np.round(x_q / s_q) + z_q, -128, 127)
x_hat = s_q * (q_q - z_q)
print(f"1. s = {s_q:.4f}, z = {z_q} | q = {q_q.astype(int)} | x_hat = {x_hat.round(4)} | "
      f"max error {np.abs(x_hat - x_q).max():.4f} <= s/2 = {s_q / 2:.4f}")
torch_q = torch.quantize_per_tensor(torch.tensor(x_q, dtype=torch.float32), s_q, z_q, torch.qint8)
print("   torch.quantize_per_tensor gives q =", torch_q.int_repr().numpy())
for T in (1.0, 2.0, 4.0):
    z_s = torch.tensor([1.0], requires_grad=True)
    F.binary_cross_entropy_with_logits(z_s / T, torch.sigmoid(torch.tensor([2.0]) / T)).backward()
    print(f"2. T = {T:.0f}: soft-term gradient {z_s.grad.item():+.3f} | times T^2 {z_s.grad.item() * T**2:+.3f}")
print(f"4. Spearman (1,2,3,4) vs (1,3,2,4): {pd.Series([1, 2, 3, 4]).corr(pd.Series([1, 3, 2, 4]), method='spearman'):.2f}")
print(f"5. Amdahl: {1 / (0.7 + 0.3 / 2):.2f}x overall, {60 * (1 - (0.7 + 0.3 / 2)):.0f} ms saved of 60; ceiling {1 / 0.7:.2f}x")
print(f"6. fp32 MLP weights: {opt.n_params(mlp) * 4 / 1000:.1f} KB")

# %% [markdown]
# **Expected:** 1. s = 0.0137, z = −41, q = `[-128  -41   32  127]` both by hand and from PyTorch, a
# maximum error of about 0.006 (below s/2 = 0.0069); 2. gradients −0.150, −0.054, −0.015, which
# times T² are −0.150, −0.217, −0.241; 4. 0.80; 5. 1.18×, 9 ms, ceiling 1.43×; 6. 38.9 KB.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: onnxscript` during export | newer PyTorch defaults to the dynamo exporter | `opt.export_onnx` passes `dynamo=False`; a direct call to `torch.onnx.export` should do the same (or `pip install onnxscript`) |
# | ORT: `INVALID_ARGUMENT ... Unexpected input data type` | float64 input | cast with `X.astype(np.float32)` |
# | ORT: `Got invalid dimensions for input` at batch ≠ 2 | exported without `dynamic_axes` | re-export with a dynamic batch dimension |
# | `RuntimeError: Didn't find engine for operation quantized::linear_prepack` | no quantisation backend (e.g. some ARM builds) | `torch.backends.quantized.engine = "qnnpack"`, or skip the PyTorch int8 row |
# | "Please consider to run pre-processing before quantization" | ORT's advice for conv nets | harmless for this MLP |
# | Latency numbers jump between runs | shared CPU, frequency scaling, other notebooks | re-run 36.5; compare ratios, not absolute values |
# | Quantised model fails the parity test | lossy by design | compare ranking metrics with agreed thresholds instead (36.3) |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Why might an ONNX Runtime model be faster than the same PyTorch model, and in which regime (model size, batch size) does the difference matter most?
# 2. What does dynamic quantisation quantise, what does static quantisation add, and why did int8 not speed up this model?
# 3. An exported model's outputs differ from PyTorch by 3e-6. Deploy or not? And for the int8 version that differs by 0.5 in logits?
# 4. Explain knowledge distillation for a ranking model: what the soft labels carry that clicks do not, what the temperature does, and when it should help most.
# 5. A ranker takes 45 % of the latency budget. List four ways to cut it, ordered by expected effort, and the risk of each.
# 6. How would you roll out an optimised model safely (offline checks, shadow traffic, A/B), and which metrics would trigger a rollback?
#
# **Next chapter:** [Chapter 37 · Recommendation & search system design](../ch37-recsys-and-search-system-design/notebook.ipynb) — combining cascades, latency budgets and model costs in two complete system-design answers.
