# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 57 · ML depth & research discussion
#
# **Day 10 · ~90 minutes · T4 GPU (runs on CPU; every experiment here is CPU-sized)**
#
# The depth round ("ML breadth/depth", "ML fundamentals", "research
# discussion") is where an interviewer asks *why* three times in a row until
# you reach the edge of what you know. Candidates from research usually know
# more theory than they can show; this notebook turns four classic depth
# questions into small experiments you have run yourself, gives you a
# question bank with the answers strong candidates give (and the wrong ones
# that sound right), and practises the other half of the round: critiquing a
# paper and telling your own research-to-production story.
#
# By the end you will be able to:
#
# 1. answer depth questions on optimisation, generalisation, transformers, recsys, evaluation and LLMs at the level of "what a strong answer covers";
# 2. show with your own runs why warmup widens the stable learning-rate range, how the learning rate should scale with batch size and where that rule breaks;
# 3. show how deep networks memorise noisy labels, and why test error can *rise then fall* as models grow (double descent);
# 4. critique a paper with a structured template and find planted flaws (leakage, test-set tuning, weak baselines, no uncertainty);
# 5. tell a two-minute research-to-production story from your EEG/BCI background.
#
# **You will produce:** `artifacts/level3/nb19/depth_demos.json` (every number
# the demos printed) and two written answers (Exercises 3–4).
#
# **Measured here vs on a T4.** All experiments are small and were verified on
# CPU (2 threads); each prints its own timing. They use fixed sizes (not the
# budget) so that CPU and GPU runs show the same effects. On a T4 they run
# faster; exact losses can differ slightly because GPU kernels are not
# bit-identical to CPU ones (not measured in our verification runs).

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# Level 3 is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: the Shakespeare corpus from notebook 03 (cached download) -----
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import make_classification

from scaleml import tinygpt as tg
from scaleml.design import table  # an aligned-table printer
from grading import ch57 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb19"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {}

text = tg.load_tiny_shakespeare()
tok = tg.CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
train_data, val_data = data[: int(0.9 * len(data))], data[int(0.9 * len(data)):]
print(f"dataset: https://huggingface.co/datasets/{tg.SHAKESPEARE_REPO} (revision {tg.SHAKESPEARE_REVISION[:7]}); "
      "repo licence MIT (char-rnn), text public domain")
print(f"{len(text):,} characters, vocab {tok.vocab_size}")

# %% [markdown]
# **Expected:** 1,115,394 characters, vocab 65 (the same corpus as notebook 03).

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Depth round | An interview round that drills one topic with "why?" until the edge of your knowledge | 19.0 |
# | Warmup | Starting with a small learning rate and raising it linearly over the first steps | 100 warmup steps in 19.2 |
# | Cosine decay | Lowering the learning rate along half a cosine to a minimum | `warmup_cosine_lr` (Exercise 1) |
# | Linear scaling rule | Multiplying the learning rate by k when the batch grows k times | batch 32 → 512, LR 0.01 → 0.16 |
# | Critical batch size | The batch beyond which larger batches stop buying proportionally faster progress | linear scaling breaks at 2,048 |
# | Memorisation | Fitting training labels that carry no signal (e.g. flipped ones) exactly | 100% train accuracy at 40% noise |
# | Early stopping | Keeping the checkpoint with the best validation score instead of the last | keeps ~0.93 at 20% noise |
# | Interpolation threshold | The model size at which it can just fit every training point (p ≈ n) | p = 100 for 100 points |
# | Double descent | Test error that falls, peaks at the interpolation threshold, then falls again | 0.51 → ~190 → 0.30 |
# | Minimum-norm solution | Among all weight vectors that fit the data exactly, the one with the smallest length | `pinv(X) @ y` |
# | Data leakage | Test information reaching training (split, preprocessing or tuning) | flaws A, B, E in the abstract |

# %% [markdown]
# ## 19.0 · How a depth round works
#
# The interviewer picks a topic you claimed (your CV is the syllabus) and
# drills: *what* → *why* → *what happens if* → *how would you know*. They are
# looking for (generic signals, not a specific company's rubric):
#
# * **mechanism, not vocabulary**: "warmup helps because Adam's second-moment
#   estimate is noisy in the first steps, so the effective step size is
#   large" beats "warmup stabilises training";
# * **evidence**: a number from something you ran, or the experiment you
#   would run to find out;
# * **calibrated uncertainty**: "I am not sure; my guess is X because Y; I
#   would check Z" scores better than a confident wrong answer;
# * **connection to practice**: what this changes in a production system.
#
# ## 19.1 · The question bank
#
# Read a row, answer aloud *before* reading the right-hand columns, then compare.
#
# **Optimisation**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | Adam vs SGD with momentum: when would you pick each? | Adam adapts per-parameter step sizes (first and second moment estimates, bias correction); robust to badly scaled gradients and sparse features, default for transformers; SGD+momentum often generalises as well or better on vision with a tuned schedule, uses less memory (no second moment: 8 vs 4 extra bytes/param for fp32 states); AdamW decouples weight decay from the adaptive step | "Adam is always better because it is adaptive"; "Adam's weight decay is the same as L2" (not with adaptive steps; hence AdamW) |
# | Why does warmup help? | Early on, Adam's second-moment estimate is based on few samples and the network is far from a good region, so large steps can blow up activations (especially with post-LayerNorm transformers); a linear ramp keeps updates small until statistics settle; it widens the range of usable peak learning rates (19.2) | "Warmup avoids local minima" |
# | How should the learning rate change with batch size? | Larger batches reduce gradient noise; for SGD, a linear scaling rule (k× batch → k× LR) with warmup works up to a *critical batch size*, beyond which returns diminish and training can become unstable; for Adam, square-root scaling is a common heuristic; always re-tune (19.3) | "Keep the LR, it is a property of the model"; "scale linearly forever" |
# | What does a learning-rate schedule (cosine, step, linear decay) do? | Large steps early explore and make fast progress; decaying reduces the noise floor so the model settles into a lower-loss region; cosine and linear decay are smooth versions of step decay; the end LR and the schedule length matter more than the shape | "Schedules are only for speed" |
#
# **Generalisation**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | Explain double descent. | Test error vs model size can fall, rise to a peak near the *interpolation threshold* (parameters ≈ training points, where the fit is forced through noise with a badly conditioned solution), then fall again as the minimum-norm interpolating solution gets smoother; regularisation (ridge, early stopping, more data) flattens the peak (19.5) | "Bigger models always overfit more" / "bigger is always better" |
# | Your network reaches 100% training accuracy on noisy labels. Is it broken? | No: deep networks can memorise random labels; clean structure is learned first and noise later, which is why early stopping on a validation set helps; noisy validation labels still rank checkpoints reasonably; robust losses or label cleaning for heavy noise (19.4) | "100% train accuracy proves overfitting, so the test accuracy must be bad" |
# | How do L2, dropout and data augmentation differ? | L2 shrinks weights (a prior); dropout trains an implicit ensemble and discourages co-adaptation; augmentation encodes invariances you believe in (the most powerful when correct); all trade bias for variance, measured on validation | "They are interchangeable" |
#
# **Transformers**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | What is the complexity of self-attention, and what dominates at long context? | O(T²·d) compute and O(T²) memory for scores per head (unless fused kernels avoid materialising them, e.g. FlashAttention); MLP is O(T·d²); attention dominates when T ≳ d; at inference the KV cache grows linearly in T (notebook 03) | "Attention is O(T)"; forgetting the KV cache |
# | Absolute, learned, sinusoidal, relative, rotary positions: why does it matter? | Attention is permutation-invariant, so position must be injected; learned absolute positions do not extrapolate beyond the trained length; relative schemes (e.g. rotary embeddings, ALiBi-style biases) encode distances and extrapolate better; the choice affects long-context behaviour | "Position does not matter for attention" |
# | Pre-LayerNorm vs post-LayerNorm? | Post-LN (original Transformer) normalises after the residual add; deep post-LN stacks have large gradients near the output at init and need warmup; pre-LN keeps an identity path through the residual stream, trains stably with less warmup, and is the common choice in GPT-style models (notebook 03's `TinyGPT` is pre-LN) | "LayerNorm placement is cosmetic" |
#
# **Recommender systems**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | What is position bias and how do you correct it? | Clicks depend on examination, which falls with position; training on raw clicks learns the old ranking; estimate propensities from randomised traffic or interventions, then IPS-weight training/evaluation, or add position as a feature and set it to a constant at serving (Level 2 notebooks 05 and 20) | "Use more data" |
# | How do you choose negatives for a two-tower model? | In-batch negatives are cheap but popularity-biased (popular items appear as negatives more often), so correct with a log-Q (sampling-probability) term; mix in random and hard negatives; too-hard negatives can be false negatives (Level 2 notebook 08, Level 3 notebook 08) | "Random negatives are always fine" |
# | Limits of two-tower models? | No early interaction between user and item features (only a dot product at the end), so fine-grained crosses are lost; that is why they retrieve and a cross-feature ranker ranks; embeddings stale for new items; ANN recall loss | "They are as accurate as rankers" |
#
# **Evaluation and causal reasoning**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | Offline NDCG improved but the A/B test is flat. Why? | Logged-data bias (only items the old system showed have labels), proxy metric not aligned with the online goal, distribution shift between log period and test period, novelty or interaction effects, test underpowered; check each with data | "The A/B test is wrong" |
# | What is counterfactual (off-policy) evaluation? | Estimating how a new policy would perform from logs of an old one: IPS (unbiased with known propensities, high variance), self-normalised IPS, doubly robust estimators; needs the logging policy to give non-zero probability to the new policy's actions (exploration) | "Replay the logs and count clicks for the new ranking" |
#
# **LLMs**
#
# | Question | What a strong answer covers | Common wrong answer |
# |---|---|---|
# | Fine-tuning vs RAG for a domain assistant? | RAG for knowledge that changes or must be cited and access-controlled; fine-tuning (or PEFT, notebook 06) for format, style, task behaviour and latency (shorter prompts); often both; evaluate both against a strong prompted baseline first | "Fine-tuning teaches the model new facts reliably" |
# | How do you evaluate hallucination? | Define it operationally (claims not supported by sources, or factually wrong); a labelled set with source passages; claim-level groundedness checks by humans and calibrated LLM judges (agreement measured, notebook 16); separate retrieval failures from generation failures; track in production on a sample | "Ask the model whether it hallucinated" |
#
# The four experiments below give you *your own numbers* for four of these rows.

# %% [markdown]
# ## 19.2 · Experiment: warmup widens the stable learning-rate range
#
# A 2-layer, 64-wide `TinyGPT` (notebook 03) trained with plain Adam (no
# gradient clipping, so instability is visible) for 300 steps at three peak
# learning rates, with and without a 100-step linear warmup. One seed per
# cell; this takes about a minute on a CPU.

# %%
def train_tinygpt(lr: float, warmup: int, steps: int = 300, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = tg.TinyGPT(tg.GPTConfig(tok.vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=2)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step, batch in enumerate(tg.RandomWindows(train_data, 64, 32, steps, seed=seed)):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup) if warmup else lr
        _, loss = model(batch["x"].to(DEVICE), batch["y"].to(DEVICE))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        val = tg.estimate_loss(model, tg.RandomWindows(val_data, 64, 32, 10, seed=1), 10, DEVICE)
    return {"train_loss_last20": float(np.mean(losses[-20:])), "val_loss": float(val)}


t0 = time.time()
warm_rows = []
for lr in (3e-3, 1e-2, 3e-2):
    for warmup in (0, 100):
        warm_rows.append({"peak_lr": lr, "warmup_steps": warmup, **train_tinygpt(lr, warmup)})
for r in warm_rows:
    print(f"lr={r['peak_lr']:<6} warmup={r['warmup_steps']:>3}  val loss {r['val_loss']:.3f}")
print(f"({time.time() - t0:.0f}s)")
RESULTS["warmup"] = warm_rows

# %% [markdown]
# **Expected (CPU run, seed 0):** at 3e-3 both runs reach a validation loss of
# ~2.27–2.31 (warmup makes no difference, or costs a little because the
# first 100 steps are slower). At **1e-2** no-warmup ends ~**2.53** against
# ~**2.24** with warmup; at **3e-2** ~**2.75** against ~**2.26**. Without
# warmup the best usable LR is ~3e-3; with it, 10× larger rates train as well.
# The mechanism to say aloud: in the first steps Adam divides by a
# second-moment estimate built from very few gradients, so the effective step
# is large and erratic exactly when the network is least robust. (A single
# seed per cell: the 0.03-sized differences at 3e-3 are within seed noise;
# the 0.3–0.5 gaps at high LR are not.) Runtime ~1 minute on 2 idle CPU threads
# (up to ~6 minutes when our build machine was heavily shared).
#
# > 💡 **Intuition.** Adam divides each step by a running estimate of the gradient's size. In the
# > first few steps that estimate is built from a handful of gradients, so it can be badly wrong and
# > the effective step erratic, while the freshly initialised network is at its most fragile. Warmup
# > is driving slowly out of the car park until the instruments have warmed up. *Where the analogy
# > breaks down:* warmup also helps plain SGD with large batches and post-LayerNorm transformers for
# > reasons unrelated to Adam's estimates; "why" depends on the setting.
#
# Production schedules combine warmup with a decay. `scaleml.train.linear_warmup_cosine`
# (used by `train()`) is one; write it yourself as a function of the step:
#
# > **Exercise 1.** Write `warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0)`.
# > For `step < warmup_steps` return `base_lr * (step + 1) / warmup_steps`
# > (linear warmup reaching `base_lr` on the last warmup step). Afterwards
# > decay along a half cosine from `base_lr` to `min_lr`:
# > `progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)`,
# > clipped to at most 1, and
# > `lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))`.
# > Steps beyond `total_steps` stay at `min_lr`. Raise `ValueError` if
# > `step < 0` or `warmup_steps > total_steps`. With `min_lr=0` this matches
# > `scaleml.train.linear_warmup_cosine` step for step.

# %%
def warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(warmup_cosine_lr)

# %% [markdown]
# ## 19.3 · Experiment: learning rate vs batch size
#
# A 2-hidden-layer MLP on a synthetic 40-feature classification task (16k
# training rows, 8k test), SGD with momentum 0.9, a **fixed budget of 3
# epochs** (so larger batches mean fewer steps). The reference is batch 32 at
# LR 0.01. For each larger batch we try: the same LR, √k scaling, linear (k×)
# scaling, and linear scaling with a one-epoch warmup. Three seeds each.

# %%
X, y = make_classification(n_samples=24_000, n_features=40, n_informative=20, n_redundant=10,
                           class_sep=0.8, flip_y=0.0, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)
Xtr, ytr, Xte, yte = X[:16_000], y[:16_000], X[16_000:], y[16_000:]


def mlp(width: int = 128, d_in: int = 40) -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.ReLU(),
                               torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 2))


def sgd_run(batch: int, lr: float, warmup_epoch: bool = False, epochs: int = 3, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = mlp()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    g = torch.Generator().manual_seed(seed)
    steps_per_epoch, step = math.ceil(len(Xtr) / batch), 0
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), batch):
            if warmup_epoch:
                for pg in opt.param_groups:
                    pg["lr"] = lr * min(1.0, (step + 1) / steps_per_epoch)
            idx = perm[i:i + batch]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
    with torch.no_grad():
        return float((model(Xte).argmax(1) == yte).float().mean())


t0 = time.time()
BASE_BATCH, BASE_LR, SEEDS = 32, 0.01, (0, 1, 2)
bs_rows = []
for batch in (32, 128, 512, 2048):
    k = batch / BASE_BATCH
    rules = {"same LR": (BASE_LR, False), "sqrt(k) LR": (BASE_LR * math.sqrt(k), False),
             "linear k LR": (BASE_LR * k, False), "linear + warmup": (BASE_LR * k, True)}
    row = {"batch": batch, "steps": 3 * math.ceil(len(Xtr) / batch)}
    for name, (lr, warm) in rules.items():
        accs = [sgd_run(batch, lr, warm, seed=s) for s in SEEDS]
        row[name] = f"{np.mean(accs):.3f} [{min(accs):.3f}-{max(accs):.3f}]"
    bs_rows.append(row)
table(bs_rows)
print(f"test accuracy, mean [min-max] over 3 seeds ({time.time() - t0:.0f}s)")
RESULTS["batch_scaling"] = bs_rows

# %% [markdown]
# **Expected (CPU run):** at batch 32 every rule is the same run (~0.97). With
# the **same LR**, accuracy falls as the batch grows (~0.92 at 512 and ~0.77
# at 2048: too few, too small steps). **Linear** scaling holds ~0.97 up to
# batch 512 (warmup narrows its seed spread there). At **2048** (LR 0.64, 24
# steps in total) linear scaling becomes unstable and seed-dependent (0.55–0.75
# in our run), warmup rescues some seeds but not reliably (0.61–0.94), and √k
# scaling (~0.92, tight across seeds) is the most reliable of the four. That
# is the *critical batch size* idea: beyond some batch, more
# examples per step stop buying proportionally bigger steps. The published
# reference for linear scaling with warmup is *"Accurate, Large Minibatch SGD:
# Training ImageNet in 1 Hour"* (Goyal et al., 2017); what you measured is a
# toy version of its argument and of its limits.
#
# > 💡 **Intuition.** A batch k times larger gives a gradient estimate with about k times less
# > variance, so you can afford k times bigger steps, but you also take k times fewer of them in a
# > fixed number of epochs. Linear scaling keeps the distance travelled per epoch the same. *Where the
# > analogy breaks down:* once the gradient noise is already small (past the critical batch size),
# > bigger steps overshoot curvature rather than average out noise, and the rule fails, as at 2,048.
#
# ## 19.4 · Experiment: noisy labels and memorisation
#
# A wide MLP (2 × 384 hidden units, ~160k parameters) on 2,000 training
# examples, with 0%, 20% or 40% of training labels flipped at random. Test
# labels are clean. We track training accuracy (against the noisy labels the
# model sees), clean test accuracy at the end, and test accuracy at the
# epoch chosen by early stopping on a (equally noisy) validation set.

# %%
Xn, yn = make_classification(n_samples=6_000, n_features=20, n_informative=10, class_sep=1.0,
                             flip_y=0.0, random_state=1)
Xn, yn = torch.tensor(Xn, dtype=torch.float32), torch.tensor(yn)
Xa, ya, Xv, yv, Xt, yt = Xn[:2000], yn[:2000], Xn[2000:3000], yn[2000:3000], Xn[3000:], yn[3000:]


def noisy_run(noise: float, epochs: int = 100, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    y_train = torch.where(torch.rand(len(ya), generator=g) < noise, 1 - ya, ya)
    y_val = torch.where(torch.rand(len(yv), generator=g) < noise, 1 - yv, yv)
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(20, 384), torch.nn.ReLU(), torch.nn.Linear(384, 384),
                                torch.nn.ReLU(), torch.nn.Linear(384, 2))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_val, best_test, best_epoch = -1.0, 0.0, 0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(Xa), generator=g)
        for i in range(0, len(Xa), 100):
            idx = perm[i:i + 100]
            loss = F.cross_entropy(model(Xa[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            with torch.no_grad():
                acc = lambda X_, y_: float((model(X_).argmax(1) == y_).float().mean())  # noqa: E731
                tr, va, te = acc(Xa, y_train), acc(Xv, y_val), acc(Xt, yt)
            if va > best_val:
                best_val, best_test, best_epoch = va, te, epoch
    return {"label_noise": noise, "train_acc_noisy_labels": tr, "test_acc_final": te,
            "test_acc_early_stopped": best_test, "early_stop_epoch": best_epoch}


t0 = time.time()
noise_rows = [noisy_run(p) for p in (0.0, 0.2, 0.4)]
table(noise_rows, floatfmt="{:.3f}")
print(f"({time.time() - t0:.0f}s)")
RESULTS["label_noise"] = noise_rows

# %% [markdown]
# **Expected (CPU run, seed 0):** training accuracy reaches **1.000 at every
# noise level**: the network memorises the flipped labels. Clean test
# accuracy after 100 epochs falls from ~0.96 (no noise) to ~**0.81** (20%)
# and ~**0.62** (40%), while the early-stopped model (picked on the *noisy*
# validation set, at epoch 5–15) keeps ~**0.93** at 20% noise and ~0.67 at 40%.
# The interview points: memorisation is not a bug in the code; clean patterns
# are fitted before noise, so early stopping and regularisation are
# noise-robustness tools; and noisy validation labels still rank checkpoints
# usefully, though they understate true accuracy. For EEG, where labels
# come from cues the subject may not have followed, this is directly relevant.
#
# > 💡 **Intuition.** A student cramming an answer key with 20% typos first learns the subject
# > (patterns shared by many questions), then memorises the typos one by one. Stop them early and
# > you get the subject without the typos. *Where the analogy breaks down:* "early" is chosen on a
# > validation set that has the same typos, so it ranks checkpoints correctly but understates their
# > true accuracy.
#
# ## 19.5 · Experiment: double descent with random features
#
# Regression with 100 training points, 20 inputs and a non-linear target.
# The model is a random ReLU feature map with *p* features followed by the
# **minimum-norm least-squares** fit (`pinv`), plus a ridge-regularised fit
# for comparison. Median test MSE over 5 random feature draws. This takes a
# second.

# %%
rng = np.random.default_rng(0)
d, n_train, n_test = 20, 100, 2_000
w_true = rng.normal(size=d) / np.sqrt(d)


def make_data(m):
    Xd = rng.normal(size=(m, d))
    return Xd, Xd @ w_true + 0.5 * np.sin(2 * Xd[:, 0]) + 0.3 * rng.normal(size=m)


Xd_tr, yd_tr = make_data(n_train)
Xd_te, yd_te = make_data(n_test)
dd_rows = []
for p in (10, 25, 50, 75, 90, 100, 110, 150, 200, 400, 1000, 3000):
    mse_minnorm, mse_ridge = [], []
    for s in range(5):
        W = np.random.default_rng(100 + s).normal(size=(d, p)) / np.sqrt(d)
        Ftr, Fte = np.maximum(Xd_tr @ W, 0), np.maximum(Xd_te @ W, 0)
        beta = np.linalg.pinv(Ftr) @ yd_tr
        mse_minnorm.append(np.mean((Fte @ beta - yd_te) ** 2))
        beta_r = np.linalg.solve(Ftr.T @ Ftr + 1.0 * np.eye(p), Ftr.T @ yd_tr)
        mse_ridge.append(np.mean((Fte @ beta_r - yd_te) ** 2))
    dd_rows.append({"features_p": p, "p / n": p / n_train, "test_mse_min_norm": float(np.median(mse_minnorm)),
                    "test_mse_ridge": float(np.median(mse_ridge))})
table(dd_rows, floatfmt="{:.3f}")
RESULTS["double_descent"] = dd_rows

# %% [markdown]
# **Expected:** the min-norm test MSE falls from ~0.89 (p=10) to ~0.51
# (p=50), **explodes** near p = n = 100 (to ~190 in our run: the system is
# exactly determined and badly conditioned, so the fit chases noise), then
# **descends again** to ~0.30 at p=3,000, better than any under-parameterised
# model. The ridge column has no spike: it falls almost
# monotonically from ~0.88 to ~0.30. The interview
# version: the peak sits at the interpolation threshold, not at "too many
# parameters"; past it, among the many interpolating solutions the
# minimum-norm one gets smoother as *p* grows; and explicit or implicit
# regularisation (ridge, early stopping, SGD's implicit bias, more data)
# removes the peak. Published references: *"Reconciling modern machine
# learning practice and the bias-variance trade-off"* (Belkin et al., 2019)
# and *"Deep Double Descent"* (Nakkiran et al., 2019).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-double-descent.svg" alt="Test error against model size on a log scale: the minimum-norm fit falls, spikes at p = n, then falls again below its earlier minimum; the ridge fit has no peak" width="100%">
#
# *What to notice:* the peak is at p = n exactly, not at 'large p'; past it, more features *help* because the minimum-norm interpolant gets smoother, and ridge removes the peak altogether.
#
# **A fifth depth question, by arithmetic: attention memory.** The attention
# score matrix is T × T per head per layer. For a 12-layer, 12-head model at
# fp16, materialising all of them during a forward pass costs:

# %%
for T in (512, 2_048, 8_192, 32_768):
    gb = 12 * 12 * T * T * 2 / 1e9
    print(f"T={T:>6}: {gb:8.1f} GB of attention scores per sequence (if materialised)")

# %% [markdown]
# **Expected:** 0.1, 1.2, 19.3 and 309.2 GB. Quadratic growth is why fused
# attention kernels that never materialise the full matrix (FlashAttention,
# PyTorch's `scaled_dot_product_attention`) and KV-cache management matter
# for long context; the T4's 16 GB runs out between 2k and 8k tokens per
# sequence if you materialise scores.

# %%
(OUT / "depth_demos.json").write_text(json.dumps(RESULTS, indent=2, default=float))
print("wrote", (OUT / "depth_demos.json").relative_to(ART), "sections:", list(RESULTS))

# %% [markdown]
# **Expected:** `wrote level3/nb19/depth_demos.json sections: ['warmup', 'batch_scaling', 'label_noise', 'double_descent']`.
#
# ## 19.6 · Critiquing a paper
#
# Research discussions often include "tell me about a paper you read
# recently" or "here is an abstract: what would you ask the authors?". Use a
# fixed template so you do not stop at the first flaw:
#
# | # | Question | Looking for |
# |---|---|---|
# | 1 | **Claim** | What exactly is claimed, and for which population (new users? new sites? new time period?) |
# | 2 | **Split** | Does the split match the claim? Group (subject, user, session) and time leakage |
# | 3 | **Selection** | Were hyperparameters, checkpoints, seeds or features chosen on the test set? |
# | 4 | **Baselines** | Strong, tuned with the same budget, same data and preprocessing? |
# | 5 | **Uncertainty** | Intervals, multiple seeds, paired tests, number of independent units (subjects, not trials) |
# | 6 | **Preprocessing** | Anything fitted on all data before splitting (normalisation, feature selection, ICA)? |
# | 7 | **Metric** | Does it match the use (balanced classes? cost of errors? calibration?) |
# | 8 | **Ablations and mechanism** | Which component causes the gain? |
# | 9 | **Practicality** | Latency, data needs, calibration time, robustness to shift |
# | 10 | **The decisive experiment** | The one experiment that would most change your belief |
#
# ### A FICTIONAL abstract with planted flaws
#
# > ⚠️ **Fictional.** This abstract was written for this exercise. No such
# > paper, model or dataset exists; any resemblance to real work is accidental.
#
# > *SpectraFormer: calibration-free motor-imagery decoding for new BCI users.*
# > We propose SpectraFormer, a transformer over time–frequency patches of
# > EEG, and show that it decodes four-class motor imagery for users it has
# > never seen. On a 20-subject dataset (12,000 balanced trials) we z-score
# > every channel using the mean and standard deviation of the full dataset,
# > then split trials at random into 80% training, 10% validation and 10%
# > test. Learning rate, depth and frequency band were selected from 60
# > configurations by test accuracy. SpectraFormer reaches **94.1%** test
# > accuracy, compared with 71.3% for CSP+LDA (toolbox defaults) and 78.0% for
# > EEGNet (trained for 10 epochs). We report the best of five random seeds.
# > SpectraFormer therefore enables plug-and-play BCIs that need no
# > calibration session for new users.
#
# **Options.** Which of these are real flaws in the abstract?
#
# * **A.** The random trial split puts trials from the same subjects (and sessions) in training and test, so the claim about *new users* is not tested.
# * **B.** Hyperparameters were selected by test accuracy, so 94.1% is an optimistically biased estimate.
# * **C.** The baselines received far less tuning than the proposed model (toolbox defaults; 10 epochs vs 60 configurations).
# * **D.** "Best of five seeds" with no spread or interval: the reported number is a maximum, not an estimate.
# * **E.** Normalisation statistics were computed on the full dataset before splitting, leaking test information into training.
# * **F.** Accuracy is meaningless for four balanced classes; the paper must report AUC instead.
# * **G.** Transformers cannot be applied to EEG because EEG is not a sequence of tokens.
# * **H.** Twenty subjects are too few to report any result at all.
# * **I.** Comparing a transformer with CSP+LDA is invalid because they have different numbers of parameters.
#
# > **Exercise 2.** Write `critique_flags()` returning the **set of letters**
# > of the real flaws, e.g. `{"A", "Q"}`. The grader stores only hashes of the
# > answer and tells you how many you missed and how many you picked wrongly.

# %%
def critique_flags():
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(critique_flags)

# %% [markdown]
# > **Exercise 3 (written: the full critique).** Apply all ten rows of the
# > template to the fictional abstract. For each flaw, say how large you
# > expect its effect to be and in which direction; then design the
# > *decisive experiment* (hint: the split that matches the claim, with
# > tuning inside the training subjects, equal tuning budgets for baselines,
# > and intervals over subjects) and say what result would make you believe
# > the claim. Finish with the one question you would ask the authors first.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 19.7 · Research to production: your EEG story
#
# Interviewers ask researchers some version of *"how would your research
# work in production?"* or *"tell me about taking a model from a paper to a
# product"*. They are probing whether you know what changes. A structure
# that works in two minutes:
#
# 1. **The research result** (20 s): task, data, metric, your number, *with
#    its split named* ("leave-subjects-out across N subjects").
# 2. **What would break in production** (40 s): new users (cross-subject
#    shift), new sessions on the same user (electrode placement, impedance,
#    fatigue), different hardware and sampling rates, artefacts (blinks, EMG,
#    line noise), latency limits for closed-loop use, labels that do not exist
#    online.
# 3. **What you would build** (40 s): a calibration-light adaptation path
#    (a short calibration block, re-centring of covariance or feature
#    statistics per session, or fine-tuning a small head; PEFT from notebook 06),
#    a data-quality gate (impedance, artefact rate) that refuses to predict on
#    bad signal, a streaming inference loop with a latency budget (18.1),
#    monitoring of input drift per session, and a fallback (ask the user to
#    repeat; default to a non-BCI control).
# 4. **How you would know it works** (20 s): online evaluation with new users
#    (not new trials), pre-registered, intervals over users; what you would
#    monitor after launch.
#
# Rules from Level 1 notebook 17 still apply: real projects only, your own
# role explicit, numbers only if you measured them, and no claims about
# products you did not build.
#
# > **Exercise 4 (written).** Write your two-minute research-to-production
# > story for one of your EEG/BCI projects (or another research project) in
# > the four-part structure above. Use only numbers you actually measured,
# > name the split each number comes from, and say what you would do
# > differently now that you have done Levels 1–3. Record yourself; cut to
# > 2:15 at most.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🧮 Key equations
#
# **1 · Warmup + cosine schedule** (Exercise 1).
#
# $$
# \eta(t) = \begin{cases} \eta_{\max} \dfrac{t + 1}{t_w} & t < t_w \\[2ex] \eta_{\min} + \tfrac12 (\eta_{\max} - \eta_{\min}) \big(1 + \cos(\pi\, \pi_t)\big) & t \ge t_w \end{cases}, \qquad \pi_t = \min\!\left(1, \frac{t - t_w}{t_{\text{total}} - t_w}\right)
# $$
#
# *Symbols:* $t$ the step, $t_w$ warmup steps, $\pi_t$ the decay progress in $[0, 1]$. *Example:*
# $\eta_{\max} = 10^{-3}$, $t_w = 100$, $t_{\text{total}} = 1{,}000$, $\eta_{\min} = 0$: $\eta(49) = 5 \times 10^{-4}$,
# $\eta(550) = 5 \times 10^{-4}$ (halfway down the cosine), $\eta(1{,}000) = 0$.
#
# **2 · Learning rate vs batch size.**
#
# $$
# \eta(kB) = k\, \eta(B) \ \ \text{(linear, SGD)}, \qquad \eta(kB) = \sqrt{k}\, \eta(B) \ \ \text{(square-root, more conservative)}
# $$
#
# *Example:* batch 32 at 0.01 → batch 2,048 ($k = 64$): linear 0.64 (unstable in 19.3), √k 0.08 (stable).
#
# **3 · Minimum-norm least squares and ridge.**
#
# $$
# \hat{w}_{\text{min-norm}} = X^{+} y = \arg\min_w \lVert w \rVert_2 \ \text{ s.t. } Xw = y \ \ (p > n), \qquad \hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
# $$
#
# *In words:* when there are more features than points, infinitely many $w$ fit exactly; the
# pseudo-inverse picks the shortest. Near $p = n$ the smallest singular value $\sigma_{\min}$ of $X$ is
# tiny and $\lVert \hat{w} \rVert$ scales like $1/\sigma_{\min}$: the peak. Ridge's $\lambda$ keeps
# $1/(\sigma^2 + \lambda)$ bounded. *Example:* in 19.5, p = 100 = n gives test MSE ~190.
#
# **4 · Self-attention cost.** Compute $O(T^2 d)$ per layer; materialised score memory
# $= L \cdot H \cdot T^2 \cdot b$ bytes per sequence. *Example:* $12 \cdot 12 \cdot 2{,}048^2 \cdot 2 = 1.2$ GB.
#
# The cell below checks 1, 3 and 4 numerically.

# %%
# Key equations, checked numerically (instant).
from scaleml.train import linear_warmup_cosine


def schedule_by_formula(t, peak=1e-3, tw=100, total=1_000):
    if t < tw:
        return peak * (t + 1) / tw
    return 0.5 * peak * (1 + math.cos(math.pi * min(1.0, (t - tw) / (total - tw))))


opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
sched = linear_warmup_cosine(opt, 100, 1_000)
lib = {}
for t in range(1_001):
    lib[t] = opt.param_groups[0]["lr"]
    opt.step()
    sched.step()
print("lr at steps 0, 49, 99, 550, 1000:", [f"{lib[t]:.2e}" for t in (0, 49, 99, 550, 1_000)],
      "| max |formula - library|", f"{max(abs(lib[t] - schedule_by_formula(t)) for t in lib):.1e}")

rng_chk = np.random.default_rng(0)
Xw, yw = rng_chk.normal(size=(20, 50)), rng_chk.normal(size=20)  # 20 points, 50 features
w_mn = np.linalg.pinv(Xw) @ yw
null = np.linalg.svd(Xw)[2][-1]  # a direction X cannot see: adding it keeps the fit exact
w_other = w_mn + 0.5 * null
print(f"min-norm fit residual {np.abs(Xw @ w_mn - yw).max():.1e}, ||w|| = {np.linalg.norm(w_mn):.3f} | "
      f"another exact fit: residual {np.abs(Xw @ w_other - yw).max():.1e}, ||w|| = {np.linalg.norm(w_other):.3f}")
print(f"attention scores at T = 2,048: {12 * 12 * 2048**2 * 2 / 1e9:.2f} GB")

# %% [markdown]
# **Expected:** learning rates `1.00e-05, 5.00e-04, 1.00e-03, 5.00e-04, 0.00e+00` and a formula–library
# difference of 0 (to rounding, ~1e-19); both fits have a residual near 1e-15, but the other exact fit
# is longer (`||w||` 0.780 vs 0.598, i.e. √(0.598² + 0.5²): the added direction is orthogonal to the
# minimum-norm solution); `1.21` GB.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Warmup table differs from the Expected line by a few hundredths | different hardware/threads; one seed per cell | compare the *gaps* at high LR (0.3–0.5), not the third decimal |
# | `Could not download Trelis/tiny-shakespeare ... using a 10-line excerpt` | no Hub access | the experiments still run, but losses are meaningless; reconnect and re-run the catch-up cell |
# | Batch-scaling cell takes minutes | many threads contending on a shared CPU | `torch.set_num_threads(2)`; it runs 48 small trainings |
# | Double-descent peak at a slightly different *p* or height | random features and data draws | the peak is always near p = n_train; its height is very sensitive (division by tiny singular values) |
# | Exercise 1 off by one step at the warmup boundary | used `step / warmup` instead of `(step + 1) / warmup` | the schedule must reach `base_lr` on step `warmup_steps - 1` |
# | Exercise 2 says "picked wrongly" | chose a plausible-sounding non-flaw | re-read F–I: is it a flaw *in this study*, or a general opinion? |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Why does learning-rate warmup help Adam-trained transformers? Show the numbers from your warmup run and give the mechanism.
# 2. You move training from 1 GPU (batch 64) to 32 GPUs (batch 2,048) with data parallelism. What do you do with the learning rate and schedule, and how would you know if you had passed the critical batch size?
# 3. A model reaches 100% training accuracy and 81% test accuracy, and you suspect 20% of labels are wrong. What is happening, and what do you try first?
# 4. Explain double descent: where is the peak, why is it there, and why can a model with 30× more parameters than data points generalise better than one with 0.5×?
# 5. What is the memory and compute complexity of self-attention in sequence length, and what do FlashAttention-style kernels and a KV cache each change?
# 6. Critique the fictional SpectraFormer abstract in 90 seconds, flaws in order of how much they inflate the headline number.
#
# ➡️ **Next:** [`20_full_interview_loop_and_portfolio.ipynb`](../ch58-full-interview-loop-and-portfolio/notebook.ipynb) — run a full, timed loop against leveling rubrics and package your Level 1–3 work as a portfolio.
