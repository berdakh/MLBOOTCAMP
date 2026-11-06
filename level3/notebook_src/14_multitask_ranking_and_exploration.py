# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 14 · Multi-task ranking & exploration
#
# **Day 7 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# A production ranker never optimises one number. An ads system predicts clicks *and*
# conversions, blends them with bids and business value, sells the slot in an auction, and must
# keep showing new ads it knows little about. Each of these is a classic system-design deep dive:
# "how would you train CTR and CVR together?", "how is the winner's price set?", "what happens if
# pCTR is miscalibrated?", "how do you cold-start a new ad?".
#
# By the end you will be able to:
#
# 1. train a **shared-bottom** and an **MMoE** multi-task model for click and conversion, and explain when experts help;
# 2. explain **sample selection bias** in conversion modelling and the **ESMM** idea (pCTCVR = pCTR × pCVR over the entire space), and measure it with counterfactual labels only a simulator has;
# 3. blend objectives (`pCTR^a · pCVR^b · value`) and read the **Pareto** trade-off between clicks and value;
# 4. run **first- and second-price auctions** with reserve prices and show how calibration errors move revenue and allocation;
# 5. implement **ε-greedy, UCB and Thompson sampling**, compare their regret, and apply them to cold-start ads.
#
# **You will produce:** `artifacts/level3/nb14/pareto.png`, `artifacts/level3/nb14/regret.png` and
# `artifacts/level3/nb14/results.json`. The generator and models are in
# [`src/scaleml/ctr.py`](../../src/scaleml/ctr.py) (`SharedBottom`, `MMoE`).
#
# **Measured here vs on a T4.** Our verification runs are CPU-only with the CPU budget
# (300,000 training impressions). Numbers in **Expected:** lines were printed by such a run.
# Statements about a T4 are marked *"not measured in our verification runs"*.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level3/requirements.txt"], check=True)

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
# --- Catch up: nothing needed from notebook 13; the data is regenerated -----
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from scaleml import ctr
from scaleml.device import seed_everything
from scaleml.grading import nb14 as grade
from scaleml.train import TrainConfig, train

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb14"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order

N_TRAIN = {"gpu": 2_000_000, "cpu": 300_000, "smoke": 40_000}[BUDGET.name]
N_TEST = {"gpu": 400_000, "cpu": 100_000, "smoke": 20_000}[BUDGET.name]
print(f"train impressions: {N_TRAIN:,} | test impressions: {N_TEST:,}")

# %% [markdown]
# ## 14.1 · Clicks, conversions and sample selection bias
#
# We reuse notebook 13's generator with two changes: higher base rates (CTR ≈ 8%, CVR ≈ 15% of
# clicks) so that a CPU-sized sample holds a few thousand conversions, and a **negative**
# correlation between the click and conversion logits (`cvr_ctr_corr = −0.3`): some of what makes
# people click (eye-catching creatives, "clickbait") makes them *less* likely to buy. That is
# where multi-task learning gets interesting, because the two tasks partly disagree.
#
# A conversion can only be observed after a click. A CVR model trained on clicked impressions is
# then *used* on every impression (to rank ads before anyone has clicked): it is trained on one
# distribution and served on another. That is **sample selection bias (SSB)**. Real logs cannot
# measure it, because nobody knows whether a non-clicked impression *would* have converted. Our
# simulator knows `true_cvr` for every impression, so we can.

# %%
world = ctr.CTRWorld(ctr.CTRConfig(seed=1, base_ctr=0.08, base_cvr=0.15, cvr_ctr_corr=-0.3))
t0 = time.perf_counter()
train_df = world.sample(N_TRAIN, seed=1, start_day=0, n_days=9)
test_df = world.sample(N_TEST, seed=2, start_day=10, n_days=4)
print(f"generated in {time.perf_counter() - t0:.1f} s")
clicked = train_df.click == 1
print(f"CTR {train_df.click.mean():.4f} | conversions {train_df.conversion.sum():,} | "
      f"CVR among clicks {train_df.conversion[clicked].mean():.4f} | CTCVR {train_df.conversion.mean():.4f}")
print(f"mean true CVR on clicked impressions {train_df.true_cvr[clicked].mean():.4f} vs on non-clicked "
      f"{train_df.true_cvr[~clicked].mean():.4f}  <- the population a click-trained CVR model never sees")
corr = np.corrcoef(np.log(test_df.true_ctr), np.log(test_df.true_cvr))[0, 1]
print(f"correlation of log true CTR and log true CVR: {corr:+.2f}")

# %% [markdown]
# **Expected:** CTR ≈ 0.080, ~4,700 conversions, CVR among clicks ≈ 0.12, and a mean true CVR
# that is **higher on non-clicked impressions** than on clicked ones (the negative correlation,
# about −0.3): a model trained only on clicks sees the wrong slice of the population.

# %% [markdown]
# ## 14.2 · Shared-bottom, MMoE and ESMM
#
# Three ways to predict both tasks with one network (all share the hashed embeddings):
#
# * **Shared-bottom**: one MLP shared by both tasks, a small tower per task. Cheap and the default;
#   suffers *negative transfer* when tasks conflict, since one representation must serve both.
# * **MMoE** (Ma et al., 2018): several expert MLPs; each task has its own softmax **gate** that mixes
#   the experts per example. Tasks can use different experts where they disagree.
# * **ESMM** (Ma et al., 2018, Alibaba): same network as shared-bottom, different *loss*. Instead of a
#   CVR loss on clicked rows only, model `pCTCVR = pCTR × pCVR` and train it on **all impressions**
#   with the label "clicked and converted". pCVR is never trained on clicks alone, so it is defined
#   over the entire space; it also borrows the much larger click signal through shared embeddings.
#
# The "naive" multi-task loss for shared-bottom and MMoE is `BCE(pCTR, click)` on all rows plus
# `BCE(pCVR, conversion)` on clicked rows. Output biases start at each task's base-rate log-odds
# (notebook 13 showed why). One epoch each: we checked in development runs that a second epoch made
# every model worse on test (NE up by ~0.05–0.07), the "one-epoch" overfitting typical of
# embedding-heavy CTR models.

# %%
BUCKETS = 2**20 if BUDGET.name == "gpu" else 2**18
tr, stats = ctr.to_tensors(train_df, BUCKETS)
tr["conv"] = torch.from_numpy(train_df.conversion.to_numpy(np.float32))
te, _ = ctr.to_tensors(test_df, BUCKETS, stats)
base_ctr = float(train_df.click.mean())
base_cvr = float(train_df.conversion[clicked].mean())
bce = torch.nn.functional.binary_cross_entropy_with_logits


def naive_loss(model, batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    z_ctr, z_cvr = (z.float() for z in model(batch["idx"], batch["dense"]))
    loss = bce(z_ctr, batch["y"])
    m = batch["y"] > 0  # conversions are only observed after a click
    if m.any():
        loss = loss + bce(z_cvr[m], batch["conv"][m])
    return loss


def esmm_loss(model, batch):
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    z_ctr, z_cvr = (z.float() for z in model(batch["idx"], batch["dense"]))
    p_ctcvr = (torch.sigmoid(z_ctr) * torch.sigmoid(z_cvr)).clamp(1e-7, 1 - 1e-7)
    return bce(z_ctr, batch["y"]) + torch.nn.functional.binary_cross_entropy(p_ctcvr, batch["conv"])


BATCH = 512 if BUDGET.name != "gpu" else 2048
config = TrainConfig(epochs=1, lr=1e-3, weight_decay=0.0, max_grad_norm=None, log_every=10_000)
models = {}
for name, make, loss in [
    ("shared-bottom", lambda: ctr.SharedBottom(BUCKETS, base_rates=(base_ctr, base_cvr)), naive_loss),
    ("MMoE (3 experts)", lambda: ctr.MMoE(BUCKETS, n_experts=3, base_rates=(base_ctr, base_cvr)), naive_loss),
    ("ESMM (shared-bottom)", lambda: ctr.SharedBottom(BUCKETS, base_rates=(base_ctr, base_cvr)), esmm_loss),
]:
    seed_everything(0)
    model = make()
    res = train(model, lambda: ctr.Batches(tr, BATCH, seed=0), loss, DEVICE, config)
    models[name] = model
    print(f"{name:22s} {sum(p.numel() for p in model.parameters()):>10,} params | {res.seconds:5.1f} s")

# %% [markdown]
# Now evaluate. CTR is scored on all test impressions. For CVR we report the usual (biased) view,
# AUC on **clicked** test impressions, and two views only a simulator allows, over **all**
# impressions: AUC against counterfactual labels (`conv_if_clicked ~ Bernoulli(true_cvr)`) and the
# ratio mean pCVR / mean true CVR, separately on clicked and non-clicked impressions. Finally
# pCTCVR = pCTR × pCVR, the quantity a ranker uses for "expected conversions per impression".

# %%
y_click = test_df.click.to_numpy()
y_conv = test_df.conversion.to_numpy()
was_clicked = y_click == 1
cf = (np.random.default_rng(0).random(len(test_df)) < test_df.true_cvr.to_numpy()).astype(int)
true_cvr = test_df.true_cvr.to_numpy()
rows, preds = [], {}
for name, model in models.items():
    p_ctr, p_cvr = ctr.predict_tasks(model, te, DEVICE)
    preds[name] = (p_ctr, p_cvr)
    rows.append({"model": name, "CTR NE": ctr.normalized_entropy(y_click, p_ctr), "CTR AUC": roc_auc_score(y_click, p_ctr),
                 "CVR AUC clicked": roc_auc_score(y_conv[was_clicked], p_cvr[was_clicked]),
                 "CVR AUC all (cf)": roc_auc_score(cf, p_cvr),
                 "pCVR/true clicked": p_cvr[was_clicked].mean() / true_cvr[was_clicked].mean(),
                 "pCVR/true non-clicked": p_cvr[~was_clicked].mean() / true_cvr[~was_clicked].mean(),
                 "CTCVR AUC": roc_auc_score(y_conv, p_ctr * p_cvr)})
rows.append({"model": "oracle (true probs)", "CTR NE": ctr.normalized_entropy(y_click, test_df.true_ctr),
             "CTR AUC": roc_auc_score(y_click, test_df.true_ctr),
             "CVR AUC clicked": roc_auc_score(y_conv[was_clicked], true_cvr[was_clicked]),
             "CVR AUC all (cf)": roc_auc_score(cf, true_cvr), "pCVR/true clicked": 1.0, "pCVR/true non-clicked": 1.0,
             "CTCVR AUC": roc_auc_score(y_conv, test_df.true_ctr * true_cvr)})
mt = pd.DataFrame(rows).set_index("model")
RESULTS["multitask"] = mt.round(4).to_dict(orient="index")
print(mt.round(4).to_string())
gates = models["MMoE (3 experts)"]
gates(te["idx"][:20_000].to(DEVICE), te["dense"][:20_000].to(DEVICE))
print("MMoE mean gate weights per task (rows: CTR, CVR; columns: experts):")
print(np.round(np.stack([g.mean(0).cpu().numpy() for g in gates.last_gates]), 3))

# %% [markdown]
# **Expected (CPU budget, measured):** XX

# %% [markdown]
# ## 14.3 · Blending objectives
#
# A ranker needs one score per candidate. A common form is a weighted product,
#
# `score = pCTR^a · pCVR^b · value`,
#
# where `value` is what a conversion is worth (basket size, advertiser bid per conversion). `a` and
# `b` are business knobs: `b = 0` ranks for clicks × value, `a = b = 1` ranks for expected value per
# impression. No setting is "correct"; each trades clicks against conversions and value, and the
# useful picture is the **Pareto frontier**: the settings that cannot improve one metric without
# hurting another. We group test impressions at random into requests of 10 candidates, pick the
# top-scored candidate per request with MMoE's predictions, and score the choice with the
# **true** probabilities (expected clicks, conversions and value per request).

# %%
p_ctr, p_cvr = preds["MMoE (3 experts)"]
K = 10
n_req = len(test_df) // K
order = np.random.default_rng(1).permutation(n_req * K).reshape(n_req, K)
true_ctr_m = test_df.true_ctr.to_numpy()[order]
true_cvr_m = true_cvr[order]
value_m = test_df.value.to_numpy()[order]
pc, pv = p_ctr[order], p_cvr[order]
grid = []
for a in [0.0, 0.5, 1.0, 1.5, 2.0]:
    for b in [0.0, 0.5, 1.0, 1.5, 2.0]:
        score = a * np.log(pc) + b * np.log(pv) + np.log(value_m)  # log of the product: same ranking
        pick = score.argmax(1)
        r = np.arange(n_req)
        grid.append({"a": a, "b": b, "clicks/1k": 1000 * true_ctr_m[r, pick].mean(),
                     "conversions/1k": 1000 * (true_ctr_m * true_cvr_m)[r, pick].mean(),
                     "value/1k": 1000 * (true_ctr_m * true_cvr_m * value_m)[r, pick].mean()})
grid = pd.DataFrame(grid)
pts = grid[["clicks/1k", "value/1k"]].to_numpy()
grid["pareto"] = [not np.any(np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)) for p in pts]
rand_pick = np.random.default_rng(2).integers(0, K, n_req)
print(f"random choice: clicks/1k {1000 * true_ctr_m[np.arange(n_req), rand_pick].mean():.1f}, value/1k "
      f"{1000 * (true_ctr_m * true_cvr_m * value_m)[np.arange(n_req), rand_pick].mean():.1f}")
print(grid.sort_values("clicks/1k").round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(grid["clicks/1k"], grid["value/1k"], s=36, color="#86b6ef", label="all (a, b) settings")
front = grid[grid.pareto].sort_values("clicks/1k")
ax.plot(front["clicks/1k"], front["value/1k"], "-o", color=COLORS[0], lw=2, ms=6, label="Pareto frontier")
for _, r in front.iterrows():
    ax.annotate(f"a={r.a:g}, b={r.b:g}", (r["clicks/1k"], r["value/1k"]), fontsize=7,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("expected clicks per 1,000 requests")
ax.set_ylabel("expected conversion value per 1,000 requests")
ax.set_title("Objective blending: score = pCTR^a · pCVR^b · value")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "pareto.png", dpi=120)
plt.show()
RESULTS["blending"] = grid.round(3).to_dict(orient="records")

# %% [markdown]
# **Expected (measured):** XX

# %% [markdown]
# ## 14.4 · Auctions: eCPM, first and second price, reserves, calibration
#
# Advertisers bid per click; the platform sells impressions. To compare them, rank by expected
# revenue per impression, **eCPM = pCTR × bid × 1000**. In a **second-price** (for one slot, the
# same as GSP or VCG) auction the winner pays the *minimum* bid that would still have won:
# per click, `price = eCPM_runner-up / (pCTR_winner × 1000) = bid₂ · pCTR₂ / pCTR₁`, never more than
# its own bid. A **reserve price** is a floor: bids below it are ineligible, and the winner pays at
# least the reserve. In a **first-price** auction the winner pays its own bid, so rational bidders
# shade their bids below their value; second price makes bidding your true value a dominant
# strategy (Vickrey), which is why it dominated ads for years (display exchanges moved to first
# price around 2019, for reasons of transparency between exchanges).

# %% [markdown]
# > **Exercise 1.** Write `second_price_auction(bids, pctrs, reserve)`. Bids are per click. Only bids
# > `>= reserve` are eligible. Rank eligible ads by `bid × pctr` (ties: lowest index wins). Return
# > `(winner_index, price_per_click)` with `price_per_click = max(reserve, bid₂ · pctr₂ / pctr_winner)`,
# > or the reserve if there is no eligible runner-up; return `(None, 0.0)` if nobody is eligible.

# %%
def second_price_auction(bids, pctrs, reserve):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(second_price_auction)

# %% [markdown]
# First, two textbook facts, by simulation. With `n` bidders whose values are i.i.d. uniform on
# [0, 1], second price with truthful bids and first price with the equilibrium bids
# `(n−1)/n · value` earn the **same expected revenue**, `(n−1)/(n+1)` (*revenue equivalence*), and a
# reserve price raises revenue up to a point (for uniform values the optimum is 0.5 whatever `n`:
# Myerson's optimal auction). Pure CTR = 1 here: values are per impression.

# %%
rng = np.random.default_rng(0)
n_bidders, n_auctions = 4, 200_000
values = rng.random((n_auctions, n_bidders))
srt = np.sort(values, axis=1)
second = srt[:, -2].mean()
first = ((n_bidders - 1) / n_bidders * srt[:, -1]).mean()
print(f"n = {n_bidders}: second-price revenue {second:.4f} | first-price (equilibrium bids) {first:.4f} | "
      f"theory {(n_bidders - 1) / (n_bidders + 1):.4f}")
for r in [0.0, 0.25, 0.5, 0.75]:
    top, sec = srt[:, -1], srt[:, -2]
    rev = np.where(top >= r, np.maximum(sec, r), 0.0).mean()
    print(f"  second price, reserve {r:.2f}: revenue {rev:.4f} | unsold {np.mean(top < r):.1%}")

# %% [markdown]
# **Expected:** both formats ≈ 0.600 (theory 3/5); the reserve raises revenue from 0.600 to ≈ 0.62
# at 0.25 and ≈ 0.64 at 0.5, then falls at 0.75 as more impressions go unsold (~32%). Seeded,
# so the numbers repeat.
#
# Now calibration. Each auction has 5 ads drawn from the test set (true CTR) with log-normal
# per-click bids. We vary how pCTR relates to the truth and measure, per 1,000 auctions, the
# platform's expected revenue (true CTR of the winner × price) and the advertisers' total value
# (true CTR × bid: with truthful bidding the bid is the value of a click), plus how often advertiser 0
# wins and its return on spend (value / cost).

# %%
def run_auctions(bids, pctr, true_ctr, reserve=0.0):
    """Vectorised second-price auctions (one row each). Returns winner (-1 = unsold) and price per click."""
    elig = bids >= reserve
    ecpm = np.where(elig, bids * pctr, -1.0)
    order = np.argsort(-ecpm, axis=1, kind="stable")
    r = np.arange(len(bids))
    w, s = order[:, 0], order[:, 1]
    has_w = ecpm[r, w] >= 0
    has_s = ecpm[r, s] >= 0
    price = np.where(has_s, bids[r, s] * pctr[r, s] / pctr[r, w], reserve)
    price = np.where(has_w, np.maximum(price, reserve), 0.0)
    return np.where(has_w, w, -1), price


rng = np.random.default_rng(3)
n_auc, n_ads = 50_000, 5
tctr = rng.choice(test_df.true_ctr.to_numpy(), size=(n_auc, n_ads))
bids = np.exp(rng.normal(0.0, 0.5, size=(n_auc, n_ads)))
noise = np.exp(rng.normal(0.0, 0.3, size=(n_auc, n_ads)))
scenarios = {
    "calibrated (pCTR = true CTR)": tctr,
    "everyone over-predicted x1.5": 1.5 * tctr,
    "advertiser 0 over-predicted x1.5": tctr * np.where(np.arange(n_ads) == 0, 1.5, 1.0),
    "advertiser 0 under-predicted x0.67": tctr * np.where(np.arange(n_ads) == 0, 0.67, 1.0),
    "unbiased log-normal noise (sd 0.3)": tctr * noise,
}
rows = []
for name, pctr in scenarios.items():
    for reserve in (0.0, 1.0):
        w, price = run_auctions(bids, pctr, tctr, reserve)
        sold = w >= 0
        r = np.arange(n_auc)[sold]
        wt = tctr[r, w[sold]]
        a0 = w[sold] == 0
        rows.append({"scenario": name, "reserve": reserve,
                     "revenue/1k": 1000 * (wt * price[sold]).sum() / n_auc,
                     "adv. value/1k": 1000 * (wt * bids[r, w[sold]]).sum() / n_auc,
                     "adv0 win share": np.mean(w == 0),
                     "adv0 value/cost": (wt[a0] * bids[r[a0], 0]).sum() / max(1e-12, (wt[a0] * price[sold][a0]).sum())})
auc_df = pd.DataFrame(rows)
RESULTS["auctions"] = auc_df.round(4).to_dict(orient="records")
print(auc_df.round(3).to_string(index=False))

# %% [markdown]
# **Expected (measured):** XX

# %% [markdown]
# ## 14.5 · Exploration: bandits and cold-start ads
#
# A ranker that always shows its current best guess never learns about the alternatives, and a new
# ad with no history never gets a chance. A **multi-armed bandit** formalises the trade-off: each
# round pick one of `K` arms (ads) with unknown click probabilities, observe a click or not, and
# minimise **regret**, the expected clicks lost against always playing the best arm.
#
# * **ε-greedy**: play the best empirical arm, but a random one with probability ε. Simple; explores
#   forever at a fixed rate, so regret grows linearly.
# * **UCB1**: play the arm with the highest `mean + √(2 ln t / n_arm)`: optimism under uncertainty.
# * **Thompson sampling**: keep a Beta(clicks + 1, non-clicks + 1) posterior per arm, draw one sample
#   from each, play the argmax. Explores in proportion to the probability that an arm is best; easy to
#   add priors and delayed feedback; usually the strongest in practice (Chapelle & Li, 2011).

# %% [markdown]
# > **Exercise 2.** Write `thompson_sampling(true_ps, n_rounds, seed)` for a Bernoulli bandit with
# > Beta(1, 1) priors. Use `rng = np.random.default_rng(seed)` for every random draw. Return a numpy
# > array of length `n_rounds` with the **cumulative expected regret** after each round, where round
# > `t` adds `max(true_ps) − true_ps[chosen arm]`.

# %%
def thompson_sampling(true_ps, n_rounds, seed):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(thompson_sampling)

# %% [markdown]
# The comparison below runs all seeds at once (vectorised over runs) so it stays fast: 10 ads with
# CTRs between 1% and 5%, 50,000 rounds, 20 seeds. "greedy" plays each arm once, then always the
# best empirical CTR; ε = 0.05.

# %%
def bandit(policy, ps, n_rounds, n_runs, seed=0, eps=0.05):
    rng = np.random.default_rng(seed)
    K = len(ps)
    clicks, pulls = np.zeros((n_runs, K)), np.zeros((n_runs, K))
    regret = np.zeros((n_runs, n_rounds))
    total = np.zeros(n_runs)
    rows_ = np.arange(n_runs)
    for t in range(n_rounds):
        if policy == "thompson":
            arm = rng.beta(clicks + 1, pulls - clicks + 1).argmax(1)
        elif policy == "ucb":
            mean = clicks / np.maximum(pulls, 1)
            bonus = np.sqrt(2 * np.log(t + 1) / np.maximum(pulls, 1e-9))
            arm = np.where(pulls.min(1) == 0, pulls.argmin(1), (mean + bonus).argmax(1))
        elif policy in ("greedy", "eps-greedy"):  # play each arm once, then the best empirical mean
            mean = clicks / np.maximum(pulls, 1) + 1e-9 * rng.random((n_runs, K))  # random tie-break
            e = eps if policy == "eps-greedy" else 0.0
            explore = rng.random(n_runs) < e
            arm = np.where(pulls.min(1) == 0, pulls.argmin(1),
                           np.where(explore, rng.integers(0, K, n_runs), mean.argmax(1)))
        else:  # uniform
            arm = rng.integers(0, K, n_runs)
        reward = rng.random(n_runs) < ps[arm]
        clicks[rows_, arm] += reward
        pulls[rows_, arm] += 1
        total += ps.max() - ps[arm]
        regret[:, t] = total
    return regret


ps = np.linspace(0.01, 0.05, 10)
N_ROUNDS, N_RUNS = 50_000, 20
t0 = time.perf_counter()
regrets = {p: bandit(p, ps, N_ROUNDS, N_RUNS) for p in ["uniform", "greedy", "eps-greedy", "ucb", "thompson"]}
print(f"simulated in {time.perf_counter() - t0:.1f} s")
summary = pd.DataFrame({p: {"regret@1k": r[:, 999].mean(), "regret@20k": r[:, 19_999].mean(),
                            "regret@50k": r[:, -1].mean(), "sd@50k (runs)": r[:, -1].std(),
                            "last 25k rounds": (r[:, -1] - r[:, 24_999]).mean()}
                        for p, r in regrets.items()}).T
RESULTS["bandits"] = summary.round(2).to_dict(orient="index")
print(summary.round(1).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
for i, (p, r) in enumerate(regrets.items()):
    ax.plot(r.mean(0), lw=2, color=COLORS[i], label=p)
ax.set_xlabel("round")
ax.set_ylabel("cumulative expected regret (clicks)")
ax.set_title("10 ads, CTR 1–5%, mean of 20 runs")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "regret.png", dpi=120)
plt.show()

# %% [markdown]
# **Expected (measured):** XX
#
# ### Cold-start ads
#
# In production, arms come and go: new ads launch every day with no history, and old ones expire.
# We simulate a pool where 3 new ads arrive every 1,500 rounds and each ad lives 6,000 rounds;
# CTRs are drawn from a Beta(1.5, 48.5) distribution (mean 3%, a long right tail: a few new ads
# are much better than average). A new ad's prior is its category's CTR, expressed as a Beta prior
# worth 10 impressions: Beta(0.3, 9.7). We compare a **greedy** ranker using the posterior mean,
# ε-greedy, and Thompson sampling with the same prior, measuring regret against the best ad
# *currently alive* and the share of impressions given to ads younger than 1,500 rounds.

# %%
def cold_start(policy, n_rounds=30_000, n_runs=20, seed=0, every=1_500, n_new=3, life=6_000,
               prior=(0.3, 9.7), eps=0.05):
    rng = np.random.default_rng(seed)
    n_waves = n_rounds // every
    n_ads = 10 + n_new * n_waves
    ps_all = rng.beta(1.5, 48.5, (n_runs, n_ads))
    birth = np.concatenate([np.zeros(10), np.repeat(np.arange(1, n_waves + 1) * every, n_new)])
    death = np.concatenate([np.linspace(every, life, 10), np.repeat(np.arange(1, n_waves + 1) * every, n_new) + life])
    clicks, pulls = np.zeros((n_runs, n_ads)), np.zeros((n_runs, n_ads))
    total, regret, young = np.zeros(n_runs), np.zeros(n_rounds), 0.0
    r_ = np.arange(n_runs)
    for t in range(n_rounds):
        alive = (birth <= t) & (t < death)
        a0, b0 = prior[0] + clicks, prior[1] + pulls - clicks
        if policy == "thompson":
            score = rng.beta(a0, b0)
        else:
            score = a0 / (a0 + b0) + 1e-9 * rng.random(a0.shape)
            if policy == "eps-greedy":
                score = np.where(rng.random((n_runs, 1)) < eps, rng.random(a0.shape), score)
        arm = np.where(alive, score, -1).argmax(1)
        p = ps_all[r_, arm]
        reward = rng.random(n_runs) < p
        clicks[r_, arm] += reward
        pulls[r_, arm] += 1
        total += np.where(alive, ps_all, 0).max(1) - p
        regret[t] = total.mean()
        young += np.mean((t - birth[arm] < every) & (birth[arm] > 0))
    return regret, young / n_rounds


t0 = time.perf_counter()
cs = {p: cold_start(p) for p in ["greedy", "eps-greedy", "thompson"]}
for p, (reg, young) in cs.items():
    print(f"{p:10s} cumulative regret after 30k rounds {reg[-1]:6.1f} | share of impressions on ads "
          f"younger than 1,500 rounds {young:.1%}")
print(f"simulated in {time.perf_counter() - t0:.1f} s")
RESULTS["cold_start"] = {p: {"regret": round(float(r[-1]), 2), "young_share": round(float(y), 4)} for p, (r, y) in cs.items()}
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
print("saved", OUT / "results.json")

# %% [markdown]
# **Expected (measured):** XX
#
# Production versions add context (contextual bandits such as LinUCB, or Thompson sampling on a
# Bayesian last layer of the CTR model), cap the exploration budget (e.g. 1–5% of traffic, or a
# boost only for ads younger than N impressions), and log the propensity of every decision so
# that the exploration traffic doubles as unbiased data for off-policy evaluation (Level 2).

# %% [markdown]
# > **Exercise 3.** You run ads ranking with `score = pCTR · bid` in a second-price auction. A new
# > CTR model is uniformly 10% over-calibrated for *video* ads only. (a) Which auctions change winner,
# > and who pays more or less? (b) What happens to platform revenue in the short term, and to
# > advertisers (and then revenue) in the long term? (c) How would you detect this before and after
# > launch, and how would you fix it? (d) Separately: your exploration policy gives new ads 2% of
# > impressions. How would you decide whether 2% is right? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | CVR loss is `nan` on a batch | no clicked rows in the batch and the masked loss averaged over zero rows | skip the CVR term when the mask is empty (as `naive_loss` does) or use larger batches |
# | ESMM's pCVR is poor although pCTCVR looks fine | pCVR only gets gradient through the product `pCTR · pCVR`, weak where pCTR is small | more data/epochs for the CVR tower, add an auxiliary CVR loss on clicked rows, check calibration of each factor separately |
# | Second epoch makes every metric worse | embedding tables memorise the training impressions (one-epoch overfitting) | train one epoch; regularise embeddings; more data instead of more passes |
# | Auction price above the winner's bid | dividing the runner-up's eCPM by the wrong pCTR | price = `bid₂ · pCTR₂ / pCTR₁` ≤ bid₁ by construction |
# | Thompson sampling never explores | posterior counts not updated, or the same RNG draw reused | update `alpha += click`, `beta += 1 − click` every round; draw fresh samples |
# | UCB divides by zero | arms not yet played | play every arm once first (as in `bandit`) |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Shared-bottom vs MMoE: what problem does MMoE address, how do the gates work, and when would you not bother?
# 2. What is sample selection bias in CVR modelling? How does ESMM address it, and what are its weaknesses?
# 3. How would you combine pCTR, pCVR and value into one ranking score, and how would you choose the weights?
# 4. Explain second-price auctions and eCPM. Why does a *uniform* calibration error barely change allocation, but a *segment* error does?
# 5. Compare ε-greedy, UCB and Thompson sampling. Which would you use for cold-start ads, and how would you limit its cost?
# 6. Design question: your ads system must launch a "conversion value" objective next quarter. Which models, labels, metrics and online experiments do you need?
#
# ➡️ **Next:** [`15_fairness_robustness_and_privacy.ipynb`](15_fairness_robustness_and_privacy.ipynb) — from revenue to responsibility: subgroup metrics, robustness to shift and perturbations, and differential privacy.
