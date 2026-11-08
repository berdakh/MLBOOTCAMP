# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 51 · CTR prediction at scale
#
# **Part III · Chapter 51 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Click-through-rate (CTR) prediction underlies the advertising revenue of much of the consumer
# internet, and "design the ads ranking model" is a standard ML system design question. The model
# itself is comparatively simple; the difficult questions concern sparse features with millions of
# values, tables that do not fit on one GPU, metrics that reward *calibrated* probabilities, and
# labels that arrive late.
#
# By the end of this lab you should be able to:
#
# 1. describe what a Criteo-style log looks like (dense counts, high-cardinality categoricals, power-law popularity) and why most values are rarely seen;
# 2. compute log loss, **normalised entropy (NE)**, AUC and calibration, and say which one ads teams base launch decisions on;
# 3. implement a **stable** hashing trick and measure accuracy against table size and collisions;
# 4. do embedding-table memory arithmetic and explain table sharding at production scale;
# 5. train logistic regression and a **Deep & Cross Network (DCN-v2)** and compare them with the true click probability;
# 6. downsample negatives and **re-calibrate** with `p / (p + (1 − p)/w)`;
# 7. show how **delayed conversions** bias a model downwards and correct it two ways.
#
# **You will produce:** `artifacts/level3/nb13/results.json` (every metric measured below) and
# `artifacts/level3/nb13/dcn.pt` (the trained DCN). The data generator and models are in
# [`src/scaleml/ctr.py`](../../src/scaleml/ctr.py): read it alongside this notebook.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only with the CPU budget
# (500,000 training impressions). Numbers in **Expected:** lines were printed by such a run.
# Statements about a T4 are marked *"not measured in our verification runs"*.

# %% [markdown]
# Read first: [Chapter 51 text](README.md), which explains sparse features, normalised entropy, calibration and delayed feedback in more detail than this lab.

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
# --- Catch up: nothing needed from earlier notebooks (the data is synthetic) -
import inspect
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from scaleml import ctr
from scaleml.device import count_parameters, seed_everything
from grading import ch51 as grade
from scaleml.train import TrainConfig, train

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb13"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {}

# Impressions per split, by budget. CTR models need far more rows than text classifiers:
# a 3% click rate means 500k impressions hold only ~15k positives.
N_TRAIN = {"gpu": 2_000_000, "cpu": 500_000, "smoke": 40_000}[BUDGET.name]
N_TEST = {"gpu": 400_000, "cpu": 100_000, "smoke": 20_000}[BUDGET.name]
print(f"train impressions: {N_TRAIN:,} | test impressions: {N_TEST:,}")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | CTR / pCTR | Click-through rate: clicks ÷ impressions; pCTR is a model's predicted click probability | base CTR ≈ 3% |
# | Impression | One showing of an ad (one row of the log) | 500,000 training impressions |
# | Cardinality | How many distinct values a categorical field can take | 4 to 1,000,000 per field |
# | Log loss | Average negative log-probability the model gave to what actually happened | the training objective |
# | Normalised entropy (NE) | Log loss divided by the log loss of always predicting the average CTR; lower is better | constant model NE = 1.00 |
# | Calibration (pred/obs) | Whether predicted probabilities match observed rates on average | pred/obs ≈ 1.00 per test day |
# | Hashing trick | Mapping each (field, value) to a bucket with a hash, instead of a vocabulary | `hash(field, value) mod n` |
# | Collision | Two different keys landing in the same hash bucket and sharing a weight | 1,024 buckets: every key collides |
# | Embedding table | A learned vector per id (row), looked up by index | 2^18 rows × 8 dims |
# | Cross feature | A feature that fires for a *combination* of values (this ad on this site) | what DCN cross layers learn |
# | Negative downsampling | Keeping all clicks but only a fraction `w` of non-clicks for training | `w = 0.1` in section 51.7 |
# | Delayed feedback | Labels (conversions) that arrive long after the event | exponential delays, mean 2 days |

# %% [markdown]
# ## 51.1 · A Criteo-like log with a known truth
#
# The public [Criteo display-ads](https://ailab.criteo.com/ressources/) logs have 13 integer
# features (counts) and 26 categorical features (hashed strings), one row per impression and a
# click label. We use a **synthetic generator with the same shape** (`ctr.CTRWorld`), for two
# reasons: no download or licence question, and, more importantly, it knows each impression's
# **true** click probability, so we can say how far every model is from the best possible one.
# A real log never provides that.
#
# The hidden logit contains a bias (solved so that CTR ≈ 3%), non-linear effects of the dense
# counts, a time-of-day effect, per-value effects in about half of the categorical fields,
# and **feature crosses** (pairs of fields, and dense × categorical) that a linear model cannot
# represent. We split **by time**: train on days 0–9, validate on day 9 (used only for calibration
# in section 51.6), test on days 10–13, as production does (train on the past, serve the future).

# %%
t0 = time.perf_counter()
world = ctr.CTRWorld(ctr.CTRConfig(seed=0))
train_df = world.sample(N_TRAIN, seed=1, start_day=0, n_days=9)
valid_df = world.sample(N_TEST // 2, seed=3, start_day=9, n_days=1)
test_df = world.sample(N_TEST, seed=2, start_day=10, n_days=4)
print(f"generated in {time.perf_counter() - t0:.1f} s")
print(f"train CTR {train_df.click.mean():.4f} ({train_df.click.sum():,} clicks) | test CTR {test_df.click.mean():.4f}")
print(train_df[["day", *ctr.DENSE[:3], *ctr.CATEGORICAL[:4], "true_ctr", "click"]].head(4).to_string())

# %%
rows = []
for c in ctr.CATEGORICAL:
    vc = train_df[c].value_counts()
    seen = set(vc.index)
    rows.append({"field": c, "true_cardinality": int(world.cardinalities[ctr.CATEGORICAL.index(c)]),
                 "distinct_in_train": len(vc), "top1%_share": vc.iloc[:max(1, len(vc) // 100)].sum() / len(train_df),
                 "singletons": (vc == 1).mean(), "test_unseen": (~test_df[c].isin(seen)).mean()})
card = pd.DataFrame(rows).sort_values("true_cardinality")
print(card.round(3).to_string(index=False))
print(f"distinct (field, value) pairs in train: {card.distinct_in_train.sum():,}")

# %% [markdown]
# **Expected:** CTR ≈ 0.030 in both splits (about 15,000 training clicks). Cardinalities range from
# 4 to 1,000,000, but the largest field shows only ~68,000 distinct values in 500k impressions. In
# the high-cardinality fields the top 1% of values cover ~70% of impressions, 50–74% of the
# distinct values seen in training are seen **once**, and 4–10% of test impressions carry a value
# never seen in training: the long tail that no model can learn from this much data. 425,677
# distinct (field, value) pairs appear in training (the data are seeded, so these counts repeat).

# %% [markdown]
# ## 51.2 · Metrics ads teams use
#
# * **Log loss** `−mean(y log p + (1−y) log(1−p))`: the training objective, sensitive to
#   calibration as well as ranking.
# * **Normalised entropy (NE)** (He et al., 2014, *Practical lessons from predicting clicks on ads
#   at Facebook*): log loss divided by the entropy of the base rate, i.e. by the log loss of a
#   model that predicts the average CTR for everyone. NE = 1 means "no better than the average";
#   lower is better. Normalising makes numbers comparable across days and products whose base
#   CTR differs. Teams track *relative* NE changes: a 0.5% NE gain is usually significant.
# * **AUC**: ranking quality only; blind to calibration (multiply every prediction by 2 and AUC
#   does not change).
# * **Calibration**: mean predicted / observed ratio, overall and per prediction bucket.
#   Auctions multiply pCTR by the bid (section 51.6), so a model can improve AUC and still reduce revenue.
#
# > **Intuition.** NE answers "how much better than a weather forecaster who always says the
# > average?". Log loss alone is hard to compare across products: predicting a 3% CTR is easier than
# > predicting a 30% one, so raw log losses differ even for equally good models. Dividing by the
# > "always the average" loss puts both on one scale. *Where the analogy breaks down:* NE still mixes
# > ranking and calibration; a lower NE can come from better calibration alone, so read it with AUC
# > and pred/obs.

# %% [markdown]
# > **Exercise 1.** Write `normalized_entropy(y, p)`: the mean log loss of predictions `p` (clip them
# > to `[1e-7, 1 − 1e-7]`) divided by the entropy `−(b log b + (1−b) log(1−b))` of the base rate
# > `b = mean(y)`. Natural logarithms. Return a float.

# %%
def normalized_entropy(y, p):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(normalized_entropy)

# %%
def report(name, y, p):
    """One row of metrics; also stored in RESULTS."""
    row = {"model": name, "log_loss": ctr.log_loss(y, p), "NE": ctr.normalized_entropy(y, p),
           "AUC": roc_auc_score(y, p), "pred/obs": float(np.mean(p) / np.mean(y))}
    RESULTS[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
    return row


y_test = test_df.click.to_numpy()
table = [report("constant (train CTR)", y_test, np.full(len(y_test), train_df.click.mean())),
         report("oracle (true_ctr)", y_test, test_df.true_ctr.to_numpy())]
print(pd.DataFrame(table).round(4).to_string(index=False))

# %% [markdown]
# **Expected:** the constant predictor has NE ≈ 1.00 and AUC 0.5. The **oracle**, the generator's
# true probability, reaches NE ≈ 0.73 and AUC ≈ 0.88: a ceiling, because clicks are random
# even when their probability is known. Every model below sits between these two rows.

# %% [markdown]
# ## 51.3 · The hashing trick
#
# A one-hot encoding needs a vocabulary: every value of every field mapped to a row. With millions
# of values that appear and disappear daily, maintaining that vocabulary (and agreeing on it
# between training and serving) is a system of its own. The **hashing trick** skips it:
# `bucket = hash(field, value) mod n_buckets`. Unseen values need no special case, memory is fixed
# in advance, and training and serving agree as long as they use the same **stable** hash.
# The price is **collisions**: unrelated values share a weight.
#
# Two details that matter in practice (and in interviews):
#
# * Python's built-in `hash()` of a string is **salted per process** (`PYTHONHASHSEED`): the trainer
#   and the server would hash the same value to different buckets. Use a real hash function
#   (MurmurHash, xxHash, CityHash, or `hashlib` in a pinch).
# * Include the **field** in the key, so that value `123` in `C1` and in `C2` are different features
#   that can share one big table.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-hashing-trick.svg" alt="Four field-value keys go through a stable hash into eight buckets; two unrelated keys share bucket 5" width="100%">
#
# *What to notice:* putting the field in the key keeps `C1 = nike` and `C2 = nike` apart; the collision is unavoidable but its rate is predictable.
#
# > **Intuition.** A cloakroom with a fixed number of hooks and a rule "hook = a function of your
# > name": no registration desk (vocabulary) is needed, and newcomers just work, but now and then two
# > strangers share a hook. *Where the analogy breaks down:* in a model the shared hook (weight) is
# > learned from both keys' clicks, and if one of them is frequent and the other rare, the rare one
# > mostly inherits the frequent one's weight, which is often harmless.

# %% [markdown]
# > **Exercise 3.** Write `hash_feature(field, value, n_buckets)` returning an `int` in `[0, n_buckets)`
# > that is the same in every process and on every machine (no built-in `hash()`), depends on the
# > field as well as the value, and spreads values uniformly. `value` may be a `str` or an `int`.
# > (Exercise 2 comes in section 51.7.)

# %%
def hash_feature(field, value, n_buckets):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(hash_feature)

# %% [markdown]
# Hashing millions of values one Python call at a time is slow, so the pipeline uses a
# vectorised 64-bit hash over whole columns: `ctr.hash_columns` (SplitMix64 of field and value,
# then `mod n_buckets`; read it in `ctr.py`). Same idea, numpy speed.
#
# How many collisions should we expect? Hashing `k` distinct keys into `n` buckets, a given key
# shares its bucket with at least one other key with probability `1 − (1 − 1/n)^(k−1) ≈ 1 − e^(−k/n)`.
# We measure it, and train hashed logistic regression (`ctr.HashedLogReg`: one weight per bucket,
# a dim-1 embedding, plus a linear layer on the dense features) with **Adagrad**, the classic
# optimiser for sparse CTR models (per-coordinate learning rates: rare features keep large steps).
# One detail matters more than it looks: the bias starts at the **log-odds of the base rate**
# (log(0.03/0.97) ≈ −3.5) rather than 0 (p = 0.5). Adagrad's shrinking steps would need most of
# an epoch just to walk the bias down to −3.5, leaving the model badly calibrated.

# %%
def fit_logreg(data, n_buckets, epochs=1, lr=0.05, batch_size=1024, seed=0, loss_fn=None):
    """Hashed logistic regression with Adagrad. ``loss_fn(logits, batch)`` overrides plain BCE."""
    seed_everything(seed)
    model = ctr.HashedLogReg(n_buckets, base_rate=data["y"].mean().item()).to(DEVICE)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in ctr.Batches(data, batch_size, seed=seed):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["idx"], batch["dense"])
            loss = (loss_fn(logits, batch) if loss_fn else
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["y"]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


n_distinct = int(card.distinct_in_train.sum())
sweep = []
for n_buckets in [2**10, 2**13, 2**16, 2**19, 2**22]:
    tr, stats = ctr.to_tensors(train_df, n_buckets)
    te, _ = ctr.to_tensors(test_df, n_buckets, stats)
    collide = ctr.collision_rate(train_df[ctr.CATEGORICAL].to_numpy(), n_buckets)  # measured
    t0 = time.perf_counter()
    model = fit_logreg(tr, n_buckets)
    secs = time.perf_counter() - t0
    p = ctr.predict_proba(model, te, DEVICE)
    row = report(f"LR {n_buckets:>9,} buckets", y_test, p)
    row.update({"buckets": n_buckets, "keys/buckets": n_distinct / n_buckets, "collide (measured)": collide,
                "collide (theory)": 1 - np.exp(-n_distinct / n_buckets), "train_s": secs,
                "table_MB": ctr.table_bytes(n_buckets, 1) / 1e6})
    sweep.append(row)
sweep = pd.DataFrame(sweep)
print(sweep[["buckets", "keys/buckets", "collide (measured)", "collide (theory)", "NE", "AUC", "pred/obs", "train_s", "table_MB"]]
      .round(4).to_string(index=False))

# %% [markdown]
# **Expected (CPU budget, measured):** with 1,024 buckets (~500 keys per bucket, every key
# collides; measured and theoretical collision rates agree) LR reaches NE ≈ 0.92 and AUC ≈ 0.74.
# Quality improves quickly up to ~65k–500k buckets (NE ≈ 0.88, AUC ≈ 0.78) and then flattens: at 4M buckets (`keys/buckets` ≈ 0.1, ~10% of keys
# still collide) the remaining gap to the oracle is not collisions but the long tail of values
# seen too rarely to learn, and the crosses a linear model cannot express. Each fit takes a few
# seconds; the 4M-row table takes longer because Adagrad here updates the whole (dense) table
# every step: production systems use sparse updates.
#
# The lesson: size the table for the number of keys that actually *matter* (frequent enough to
# learn), not for the raw cardinality. Doubling buckets far past the useful range costs memory
# for nothing.

# %% [markdown]
# ## 51.4 · Embedding tables and memory arithmetic
#
# Deep CTR models replace each one-hot feature with a learned vector: an **embedding table**
# of `rows × dim` floats. Memory = `rows × dim × bytes_per_value`, plus the optimiser state
# (Adagrad keeps one accumulator per value, Adam two) during training.

# %%
def gb(x):
    return f"{x / 1e9:,.2f} GB"


DCN_BUCKETS = 2**20 if BUDGET.name == "gpu" else 2**18
cases = [(f"this notebook's DCN ({DCN_BUCKETS:,} rows, dim 8)", DCN_BUCKETS, 8),
         ("one big production feature (1e8 ids, dim 64)", 10**8, 64),
         ("a whole ads model (1e9 hashed rows, dim 64)", 10**9, 64),
         ("same, dim 128", 10**9, 128)]
for name, rows_, dim in cases:
    print(f"{name:48s} fp32 weights {gb(ctr.table_bytes(rows_, dim)):>12} | "
          f"+ Adam state {gb(ctr.table_bytes(rows_, dim, optimizer_slots=2)):>12} | "
          f"fp16 weights {gb(ctr.table_bytes(rows_, dim, bytes_per_value=2)):>12}")
print("T4 memory: 16 GB. A 1e9 x 64 fp32 table alone is 16x that.")

# %% [markdown]
# **Expected:** our CPU-budget table is 0.01 GB (2^18 rows; 2^20 on the GPU budget); a billion rows × 64 dimensions is 256 GB of weights and
# 768 GB with Adam's state. Production ads models really are this size: the embeddings are
# almost all of the parameters, while the dense layers are a few hundred MB. So they are
# trained with **model parallelism for the embeddings and data parallelism for the dense part**
# (DLRM, TorchRec, HugeCTR):
#
# * **Table-wise sharding**: whole tables placed on different GPUs (simple; unbalanced if one table is huge).
# * **Row-wise sharding**: each GPU holds a range of rows of a big table (balances memory; every lookup
#   becomes an all-to-all exchange of ids and vectors).
# * **Column-wise sharding**: each GPU holds some of the dimensions (balances compute for very wide tables).
#
# A training step does an all-to-all to send ids to the shards that own them, the lookups,
# another all-to-all to return vectors, then the dense forward/backward with an all-reduce
# (Chapter 47). Other tools: fp16/int8 tables, row-wise Adagrad (one accumulator per row instead
# of per value), frequency-based admission (only give a row to ids seen ≥ k times) and
# eviction of stale rows, and caching the hot rows in GPU memory while the long tail
# lives in host memory or SSD. The power law in section 51.1 is what makes that cache effective.

# %% [markdown]
# ## 51.5 · Deep & Cross Network (DCN-v2)
#
# Logistic regression adds up per-feature weights; it cannot learn that *this* ad works for
# *this* site unless someone hand-crafts the cross feature. Factorisation machines and DeepFM
# learn pairwise crosses as dot products of embeddings; **DCN-v2** (Wang et al., 2021) learns them
# with *cross layers*: `x_{l+1} = x0 ⊙ (W_l x_l + b_l) + x_l`. Each layer raises the polynomial
# degree of the interactions by one, and the residual keeps lower-order terms. An MLP runs in
# parallel for implicit interactions.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-ctr-dcn.svg" alt="Sparse ids are hashed and embedded, dense counts are log-transformed, both are concatenated into x0, which feeds cross layers and a deep MLP in parallel, then a sigmoid" width="100%">
#
# *What to notice:* the embedding table holds almost all parameters; the cross layers multiply by `x0` at every layer, which is how they build explicit feature crosses.
#
# > **Intuition.** Logistic regression can learn "sports sites get clicks" and "this sneaker ad gets
# > clicks" but not "this sneaker ad *on sports sites* gets clicks": it can only add effects. A cross
# > layer multiplies the current representation by the original features `x0`, so after one layer it
# > holds pairwise products, after two triple products, like expanding a polynomial one degree at a
# > time. *Where the analogy breaks down:* the crosses are low-rank and learned, not enumerated, so
# > rare combinations are still learned poorly.
#
# Here is the whole model in code:

# %%
print(inspect.getsource(ctr.DCNv2))

# %%
tr, stats = ctr.to_tensors(train_df, DCN_BUCKETS)
va, _ = ctr.to_tensors(valid_df, DCN_BUCKETS, stats)
te, _ = ctr.to_tensors(test_df, DCN_BUCKETS, stats)
seed_everything(0)
dcn = ctr.DCNv2(DCN_BUCKETS, emb_dim=8, base_rate=train_df.click.mean())
n_emb = dcn.emb.weight.numel()
print(f"DCN parameters: {count_parameters(dcn):,} ({n_emb:,} = {n_emb / count_parameters(dcn):.0%} in the embedding table)")


def bce_loss(model, batch):
    logits = model(batch["idx"].to(DEVICE), batch["dense"].to(DEVICE))
    return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), batch["y"].to(DEVICE))


BATCH = 1024 if BUDGET.name != "gpu" else 4096
config = TrainConfig(epochs=1, lr=1e-3 if BUDGET.name != "gpu" else 2e-3, weight_decay=0.0,
                     max_grad_norm=None, log_every=10_000)
result = train(dcn, lambda: ctr.Batches(tr, BATCH, seed=0), bce_loss, DEVICE, config)
p_dcn = ctr.predict_proba(dcn, te, DEVICE)
print(f"trained {result.steps} steps in {result.seconds:.1f} s ({result.examples_per_second:,.0f} impressions/s on {DEVICE})")
torch.save(dcn.state_dict(), OUT / "dcn.pt")

lr_model = fit_logreg(tr, DCN_BUCKETS)
p_lr = ctr.predict_proba(lr_model, te, DEVICE)
table = [RESULTS["constant (train CTR)"], report("LR (same buckets)", y_test, p_lr),
         report("DCN-v2", y_test, p_dcn), RESULTS["oracle (true_ctr)"]]
print(pd.DataFrame(table).round(4).to_string(index=False))

# %% [markdown]
# **Expected (CPU budget, measured):** 2,245,767 parameters, 93% of them in the embedding table
# (2^18 rows × 8 on the CPU budget); one epoch of 489 steps in ~10–20 s (~25,000–50,000
# impressions/s on 2 CPU threads). DCN reaches NE ≈ 0.854 and AUC ≈ 0.80 against LR's NE ≈ 0.878 /
# AUC ≈ 0.78 on the same hashed features: it recovers some of the crosses. Both remain far from the
# oracle (NE 0.728), mostly because of the unlearnable tail.
#
# **On a T4 you should see** the GPU budget (2M impressions, batch 4096) train in well under a
# minute and close more of the gap to the oracle, since more data means more of the tail is
# seen often enough to learn (not measured in our verification runs). More epochs do **not**
# help much in CTR: models overfit after one pass (the "one-epoch phenomenon" reported for deep
# CTR models), which is also why production models train on each day's data once, in order.

# %% [markdown]
# ## 51.6 · Calibration, and why auctions care
#
# An ad auction ranks by **expected value per impression**, `eCPM = pCTR × bid × 1000`, and with
# second-price rules the winner's price also depends on the runner-up's pCTR (Chapter 52). If
# pCTR is 20% too high for one advertiser, that advertiser wins auctions it should lose and pays
# for clicks that do not happen; if it is too high for everyone, the platform's comparisons against
# other demand (organic content, reserve prices, other ad types bidding per impression) are
# skewed. So ads teams monitor **pred/obs** overall and per segment and usually treat a calibration
# shift as a reason not to launch, even with better AUC.

# %%
cal = ctr.calibration_table(y_test, p_dcn, n_bins=10)
cal["LR pred/obs"] = ctr.calibration_table(y_test, p_lr, n_bins=10)["pred/obs"]
print(cal.rename(columns={"pred/obs": "DCN pred/obs"}).round(4).to_string(index=False))
by_day = pd.DataFrame({"day": test_df.day, "y": y_test, "p": p_dcn}).groupby("day").agg(
    obs=("y", "mean"), pred=("p", "mean"))
by_day["pred/obs"] = by_day.pred / by_day.obs
print(by_day.round(4).to_string())

# %% [markdown]
# **Expected (measured):** both models are well calibrated: pred/obs ≈ 1.00 overall, per bucket
# within ~±10% except the lowest buckets (only ~20–40 clicks each, so the ratio is noisy), and
# per test day within ~±3%. That is not automatic: initialising the output bias at the base rate's
# log-odds mattered (without it, our first runs were under-confident: pred/obs ≈ 1.5–3 in the
# lowest buckets, ≈ 0.7 in the top one). With a time split, drift and noise show up exactly as
# this kind of daily wobble, which is why production systems monitor it continuously.
#
# When calibration is off, the standard fix is a small **calibration layer** fitted on recent
# held-out data, after the model: *Platt scaling* fits `q = σ(a · logit(p) + b)` (two parameters),
# *isotonic regression* fits any monotone map (more flexible, needs more data). Both keep the
# ranking (AUC unchanged) and fix the probabilities. Production systems refit it often, since it is
# cheap, and fit it per segment when segments drift differently. We fit Platt scaling on the
# validation day (day 9) and apply it to the test days.

# %%
from sklearn.linear_model import LogisticRegression


def fit_platt(p_valid, y_valid):
    """Platt scaling: a 1-feature logistic regression on logit(p). Returns a function p -> calibrated p."""
    logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))  # noqa: E731
    lr = LogisticRegression(C=1e6).fit(logit(p_valid)[:, None], y_valid)
    print(f"  Platt: a = {lr.coef_[0, 0]:.3f}, b = {lr.intercept_[0]:.3f} "
          "(a = 1, b = 0: already calibrated; a > 1: under-confident; a < 1: over-confident)")
    return lambda p: lr.predict_proba(logit(p)[:, None])[:, 1]


platt_dcn = fit_platt(ctr.predict_proba(dcn, va, DEVICE), valid_df.click.to_numpy())
p_dcn_cal = platt_dcn(p_dcn)
print(pd.DataFrame([RESULTS["DCN-v2"], report("DCN-v2 + Platt", y_test, p_dcn_cal)]).round(4).to_string(index=False))
print(ctr.calibration_table(y_test, p_dcn_cal, n_bins=5).round(4).to_string(index=False))

# %% [markdown]
# **Expected (measured):** Platt parameters close to `a = 1, b = 0` (ours: a ≈ 1.01, b ≈ 0.04): there
# was little to fix, so NE and AUC are unchanged and pred/obs stays ≈ 1.0. The layer becomes useful
# in section 51.7.

# %% [markdown]
# ## 51.7 · Negative downsampling and re-calibration
#
# With a 3% CTR, 97% of the training rows are negatives that teach the model little. Keep every
# click and only a fraction `w` of the non-clicks: training is ~`1/(CTR + w)` times cheaper with
# little loss in ranking quality. But the model now sees a click rate of roughly `CTR / (CTR + w(1−CTR))`,
# so its predictions are too high. Because negatives were kept at random with probability `w`,
# the odds are inflated by exactly `1/w`, so the fix is exact:
#
# `q = p / (p + (1 − p) / w)`  (He et al., 2014).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-negative-downsampling.svg" alt="The full log is 3% clicks; after keeping 10% of non-clicks it is 24% clicks; the raw model over-predicts and the recalibration formula restores calibrated probabilities" width="100%">
#
# *What to notice:* downsampling changes the *odds* by exactly 1/w, so the fix is exact, and the ranking (AUC) is never affected.
#
# > **Intuition.** If we photograph a crowd but delete 9 out of every 10 people who did *not*
# > click, the clickers look 10× more common in the photo than in the crowd. Knowing it was exactly
# > 1 in 10 allows it to be undone: multiply the odds back by `w`. *Where the analogy breaks down:* the
# > correction is only exact if the deletion was random with a known, logged `w`; any other filtering
# > (by time, by user) needs its own correction.

# %% [markdown]
# > **Exercise 2.** Write `recalibrate_downsampled(p, w)` that maps predictions `p` (numpy array) of a
# > model trained with negatives kept at rate `w` back to calibrated probabilities.

# %%
def recalibrate_downsampled(p, w):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(recalibrate_downsampled)

# %%
W = 0.1
rng = np.random.default_rng(0)
keep = (train_df.click.to_numpy() == 1) | (rng.random(len(train_df)) < W)
ds_df = train_df[keep]
print(f"kept {keep.mean():.1%} of rows; CTR in the downsampled data {ds_df.click.mean():.3f}")
LR_BUCKETS = 2**19
tr_full, stats = ctr.to_tensors(train_df, LR_BUCKETS)
tr_ds, _ = ctr.to_tensors(ds_df, LR_BUCKETS, stats)
te_lr, _ = ctr.to_tensors(test_df, LR_BUCKETS, stats)
t0 = time.perf_counter()
m_full = fit_logreg(tr_full, LR_BUCKETS, batch_size=1024)
t_full = time.perf_counter() - t0
# same number of optimiser steps on 1/8 of the data: scale the batch by the keep rate
ds_batch = max(32, round(1024 * keep.mean()))
t0 = time.perf_counter()
m_ds = fit_logreg(tr_ds, LR_BUCKETS, batch_size=ds_batch)
t_ds = time.perf_counter() - t0
p_full = ctr.predict_proba(m_full, te_lr, DEVICE)
p_ds = ctr.predict_proba(m_ds, te_lr, DEVICE)
p_ds_recal = ctr.recalibrate_downsampled(p_ds, W)
rows = [report("LR full data", y_test, p_full) | {"train_s": t_full},
        report(f"LR downsampled w={W}, raw", y_test, p_ds) | {"train_s": t_ds},
        report(f"LR downsampled w={W}, recalibrated", y_test, p_ds_recal) | {"train_s": t_ds}]
print(pd.DataFrame(rows).round(4).to_string(index=False))
# diagnostic: is the downsampled model calibrated *on its own (downsampled) distribution*?
keep_te = (y_test == 1) | (np.random.default_rng(1).random(len(y_test)) < W)
print(f"downsampled model on a downsampled test set: pred/obs = {p_ds[keep_te].mean() / y_test[keep_te].mean():.3f}")
# the usual last step: a calibration layer fitted on recent, NOT downsampled data (the validation day)
va_lr, _ = ctr.to_tensors(valid_df, LR_BUCKETS, stats)
platt_ds = fit_platt(ctr.recalibrate_downsampled(ctr.predict_proba(m_ds, va_lr, DEVICE), W), valid_df.click.to_numpy())
print(pd.DataFrame([report(f"LR downsampled w={W}, recalibrated + Platt", y_test, platt_ds(p_ds_recal))])
      .round(4).to_string(index=False))

# %% [markdown]
# **Expected (measured):** 12.7% of rows kept and a training CTR of 0.236. The raw downsampled
# model over-predicts massively (pred/obs ≈ 6.4, NE ≈ 1.85, worse than the constant predictor), yet
# its AUC is as good as the full-data model's (ranking survives; here even slightly better, 0.804
# vs 0.783, because the same number of steps spends more of its updates on clicks). After
# `recalibrate_downsampled`, NE drops to ≈ 0.86 and pred/obs to ≈ 0.91, not 1.00: the formula
# undoes the *sampling* exactly, but this model is itself slightly under-predicting on its own
# downsampled distribution (pred/obs ≈ 0.97 there), and the mapping carries that error over. The
# calibration layer fitted on the (not downsampled) validation day fixes the rest: Platt `a` ≈ 1.26
# (> 1: under-confident), and afterwards pred/obs ≈ 1.01 and NE ≈ 0.853.
# Training on the downsampled data took about half the time here (the same number of steps with
# 8× smaller batches; on a GPU, where the per-step cost is mostly overhead, one would instead keep
# the batch size and take 8× fewer steps). The correction only holds if the `w` used in training is
# the one used to correct, so `w` must be stored with the model.

# %% [markdown]
# ## 51.8 · Delayed feedback
#
# Clicks arrive within seconds; **conversions** (a purchase, an install) can arrive days later. A
# conversion-rate (CVR) model trained at time `T` sees recent clicks whose conversions have *not
# happened yet* and labels them negative. In our generator conversion delays are exponential with a
# mean of 2 days, so a click from 12 hours ago has had time to show only ~22% of its eventual
# conversions (1 − e^(−12/48)). The model learns a CVR that is too low, and the bias is worst for exactly the
# freshest data.
#
# We train CVR models on clicks from days 0–14 with the labels **as seen at the end of day 14**, and
# evaluate on clicks from days 14–16 with their **eventual** (mature) labels. Three fixes compared
# with the naive model:
#
# * **oracle labels**: wait for every conversion (impossible in production; the reference);
# * **wait window**: train only on clicks at least `W` days old (labels mostly mature; the model is
#   `W` days staler and loses the freshest data);
# * **delay-aware likelihood** (Chapelle, 2014, simplified): model the observed label as
#   `P(y_obs = 1 | x, elapsed e) = pCVR(x) · F(e)`, where `F` is the delay CDF estimated from an
#   older, fully matured batch of clicks, and maximise that likelihood. The model predicts `pCVR(x)`
#   itself; recent clicks count as "probably not yet" rather than "no".
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-delayed-feedback.svg" alt="A timeline of clicks and conversions around a training cutoff; a conversion after the cutoff makes its click look negative" width="100%">
#
# *What to notice:* the clicks closest to the cutoff are the most likely to be mislabelled, so the bias is worst on the freshest data.

# %%
N_CLICKS = {"gpu": 60_000, "cpu": 40_000, "smoke": 4_000}[BUDGET.name]
t0 = time.perf_counter()
clicks_tr = ctr.sample_clicks(world, N_CLICKS, seed=11, start_day=0, n_days=14)
clicks_te = ctr.sample_clicks(world, N_CLICKS // 4, seed=12, start_day=14, n_days=2)
history = ctr.sample_clicks(world, N_CLICKS // 4, seed=13, start_day=-40, n_days=10)  # long matured
print(f"sampled {len(clicks_tr):,} training clicks, {len(clicks_te):,} test clicks in {time.perf_counter() - t0:.1f} s")

T_CUTOFF_H = 14 * 24.0
elapsed_h = T_CUTOFF_H - clicks_tr.ts.to_numpy() / 3600.0
observed = (clicks_tr.conversion.to_numpy() == 1) & (clicks_tr.conv_delay_h.to_numpy() <= elapsed_h)
print(f"eventual CVR {clicks_tr.conversion.mean():.4f} | observed at cutoff {observed.mean():.4f} "
      f"({1 - observed.sum() / clicks_tr.conversion.sum():.1%} of conversions not seen yet)")
by_age = pd.DataFrame({"age_days": np.floor(elapsed_h / 24).astype(int), "eventual": clicks_tr.conversion,
                       "observed": observed}).groupby("age_days")[["eventual", "observed"]].mean()
print(by_age.iloc[:6].round(4).T.to_string())

# the delay CDF F(e), estimated from the matured history (empirical CDF of delays among converters)
delays = np.sort(history.conv_delay_h[history.conversion == 1].to_numpy())


def delay_cdf(e_hours):
    return np.searchsorted(delays, e_hours, side="right") / len(delays)


print(f"F(24h) = {delay_cdf(24.0):.3f}, F(96h) = {delay_cdf(96.0):.3f} (exponential, mean 48 h: "
      f"{1 - np.exp(-0.5):.3f}, {1 - np.exp(-2):.3f})")

# %%
CVR_BUCKETS = 2**18
WAIT_DAYS = 4


def cvr_data(df, labels, stats=None):
    data, stats = ctr.to_tensors(df, CVR_BUCKETS, stats, label="conversion")
    data["y"] = torch.as_tensor(np.asarray(labels, dtype=np.float32))
    return data, stats


def delay_aware_loss(logits, batch):
    # P(observed) = sigmoid(logit) * F(elapsed); Bernoulli log-likelihood of the observed label
    p_obs = (torch.sigmoid(logits) * batch["F"]).clamp(1e-6, 1 - 1e-6)
    return torch.nn.functional.binary_cross_entropy(p_obs, batch["y"])


y_cvr_test = clicks_te.conversion.to_numpy()
_, cvr_stats = ctr.to_tensors(clicks_tr, CVR_BUCKETS, label="conversion")
te_cvr, _ = cvr_data(clicks_te, y_cvr_test, cvr_stats)
old = elapsed_h >= WAIT_DAYS * 24
variants = {
    "oracle labels (eventual)": (clicks_tr, clicks_tr.conversion.to_numpy(), None),
    "naive (labels at cutoff)": (clicks_tr, observed, None),
    f"wait window ({WAIT_DAYS} days)": (clicks_tr[old], observed[old], None),
    "delay-aware likelihood": (clicks_tr, observed, delay_aware_loss),
}
rows = []
for name, (df_, labels, loss) in variants.items():
    data, _ = cvr_data(df_, labels, cvr_stats)
    if loss is not None:
        data["F"] = torch.as_tensor(delay_cdf(elapsed_h).astype(np.float32)).clamp_min(1e-3)
    m = fit_logreg(data, CVR_BUCKETS, epochs=3, lr=0.05, loss_fn=loss)
    p = ctr.predict_proba(m, te_cvr, DEVICE)
    rows.append(report(f"CVR {name}", y_cvr_test, p) | {"train_rows": len(df_)})
print(pd.DataFrame(rows).round(4).to_string(index=False))
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
print("saved", OUT / "results.json")

# %% [markdown]
# **Expected (measured):** 42,087 training clicks; the eventual CVR is 0.101 but only 0.087 is
# visible at the cutoff (14.5% of conversions missing); for clicks less than a day old only ~17% of
# their conversions are visible (0.018 vs 0.105). The estimated delay CDF matches the true exponential
# (F(24 h) ≈ 0.37 vs 0.39). On the test clicks, the oracle-label model has pred/obs ≈ 1.04 (our
# reference: a small finite-sample bias); the naive model is ~15% lower (≈ 0.88); the wait window
# (≈ 1.06, on 71% of the rows) and the delay-aware model (≈ 1.02) are both within ~2–3% of the
# reference. AUCs are similar across variants (0.71–0.72): delayed feedback is mostly a
# calibration problem, which is exactly what auctions punish.
#
# Real systems go further: the delay distribution depends on the features (Chapelle's DFM learns
# both jointly), labels are corrected as conversions arrive (fake-negative weighting, Ktena et al.,
# 2019, re-ingests the click as a positive), and streaming trainers pick the attribution window
# (e.g. 7 days) as a business decision.

# %% [markdown]
# > **Exercise 4.** An offline comparison of a new CTR model against production on the same
# > test day: AUC +0.4% (relative), NE −0.3% (better), but **pred/obs moved from 1.00 to 1.08**, and
# > the new model was trained with negatives downsampled at `w = 0.05` (production used `w = 0.1`).
# > (a) What most likely caused the calibration change, and how would you check? (b) What happens to
# > advertisers and to platform revenue if it is deployed as is? (c) What would you change before an A/B
# > test, and which online metrics would you watch? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 4):* ...

# %% [markdown]
# ## Key equations
#
# **1 · Logistic CTR model and log loss.**
#
# $$
# p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \mathrm{LL} = -\frac{1}{N} \sum_{i=1}^{N} \big[\, y_i \log p_i + (1 - y_i) \log (1 - p_i) \,\big]
# $$
#
# *In words:* the model outputs a logit $z$ (a weighted sum of features, or a DCN's output), the
# sigmoid turns it into a probability, and log loss charges $-\log p$ for each click and
# $-\log(1-p)$ for each non-click. *Example:* predicting 0.03 costs $-\ln 0.03 = 3.51$ on a click and
# $-\ln 0.97 = 0.030$ on a non-click.
#
# **2 · Normalised entropy.**
#
# $$
# \mathrm{NE} = \frac{\mathrm{LL}}{H(\bar{p})}, \qquad H(\bar{p}) = -\big[\, \bar{p} \log \bar{p} + (1 - \bar{p}) \log (1 - \bar{p}) \,\big]
# $$
#
# *Symbols:* $\bar{p}$ the base rate (mean of $y$). *Example:* $\bar{p} = 0.03$ gives
# $H = 0.03 \cdot 3.507 + 0.97 \cdot 0.0305 = 0.1347$; a model with $\mathrm{LL} = 0.115$ has
# $\mathrm{NE} = 0.854$ (about the DCN in section 51.5).
#
# **3 · Hash collisions.**
#
# $$
# P(\text{a key shares its bucket}) = 1 - \left(1 - \tfrac{1}{n}\right)^{k - 1} \approx 1 - e^{-k/n}
# $$
#
# *Symbols:* $k$ distinct keys, $n$ buckets. *Example:* $k = 425{,}677$, $n = 2^{19}$: $1 - e^{-0.81} = 0.56$.
#
# **4 · Embedding memory.** $\text{bytes} = \text{rows} \times \text{dim} \times \text{bytes per value} \times (1 + \text{optimiser slots})$.
# *Example:* $10^9 \times 64 \times 4$ B = 256 GB; with Adam's two slots, 768 GB.
#
# **5 · Re-calibration after negative downsampling.**
#
# $$
# \frac{q}{1 - q} = \frac{1}{w} \cdot \frac{p}{1 - p} \quad\Longrightarrow\quad p = \frac{q}{q + (1 - q)/w}
# $$
#
# *In words:* keeping a random fraction $w$ of negatives divides their count by $1/w$, so the odds the
# model learns are $1/w$ times the true odds; multiply them back by $w$. *Symbols:* $q$ the
# downsampled model's prediction, $p$ the calibrated probability. *Example:* $q = 0.236$, $w = 0.1$:
# $p = 0.236 / (0.236 + 7.64) = 0.030$.
#
# **6 · Delayed feedback.**
#
# $$
# P(y_{\text{obs}} = 1 \mid x, e) = \mathrm{pCVR}(x) \cdot F(e), \qquad F(e) = 1 - e^{-e/\tau}\ \ \text{(exponential delays)}
# $$
#
# *Symbols:* $e$ time elapsed since the click, $F$ the delay CDF, $\tau$ the mean delay. *Example:*
# $e = 12$ h, $\tau = 48$ h: $F = 1 - e^{-0.25} = 0.22$, so only 22% of that click's eventual conversions are visible.
#
# The cell below checks 1, 2, 3 and 5 numerically against `sklearn` and `scaleml.ctr`.

# %%
# Key equations, checked numerically (seconds on a CPU).
from sklearn.metrics import log_loss as sk_log_loss

ll_by_hand = -np.mean(y_test * np.log(p_dcn) + (1 - y_test) * np.log(1 - p_dcn))
b = y_test.mean()
H = -(b * np.log(b) + (1 - b) * np.log(1 - b))
print(f"log loss by hand {ll_by_hand:.5f} | sklearn {sk_log_loss(y_test, p_dcn):.5f}")
print(f"NE by hand {ll_by_hand / H:.4f} | ctr.normalized_entropy {ctr.normalized_entropy(y_test, p_dcn):.4f}")
print(f"H(0.03) = {-(0.03 * np.log(0.03) + 0.97 * np.log(0.97)):.4f}")

k, n = 425_677, 2**19
keys = np.random.default_rng(0).integers(0, n, k)  # a uniform hash = uniform random buckets
counts = np.bincount(keys, minlength=n)
print(f"collision rate: simulated {np.mean(counts[keys] > 1):.3f} | 1 - exp(-k/n) {1 - np.exp(-k / n):.3f}")

q_, w_ = np.array([0.236, 0.5, 0.9]), 0.1
p_ = q_ / (q_ + (1 - q_) / w_)
print(f"recalibrated {p_.round(4)} | ctr.recalibrate_downsampled {ctr.recalibrate_downsampled(q_, w_).round(4)} "
      f"| odds ratio q:p = {((q_ / (1 - q_)) / (p_ / (1 - p_))).round(3)} (= 1/w)")

# %% [markdown]
# **Expected:** the hand-written log loss equals `sklearn`'s, NE equals the library's (≈ 0.854 for the
# DCN), `H(0.03) = 0.1347`; simulated and theoretical collision rates both ≈ 0.556; the three
# re-calibrated values `[0.03, 0.0909, 0.4737]` from both implementations, and an odds ratio of 10 for each.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | Training and serving predictions differ for the same request | Python `hash()` (salted per process) or a different bucket count at serving | stable hash (`hashlib`, MurmurHash); version `n_buckets` and the seed with the model |
# | NE > 1 on the test set | predictions not calibrated (downsampling not undone, delayed labels) or the model overfits | check pred/obs first; re-calibrate; train one epoch |
# | AUC fine, pred/obs ≫ 1 | negatives were downsampled | `recalibrate_downsampled(p, w)` with the training `w` |
# | `RuntimeError: CUDA out of memory` on a T4 with a big table | `rows × dim × 4 B` × (1 + optimiser slots) exceeds 16 GB | fewer buckets, smaller dim, fp16 table, or sparse optimiser |
# | DCN worse than LR on NE but better on AUC | overfitting the embeddings (several epochs, LR too high) | one epoch, lower LR, weight decay on the dense part only |
# | CVR model systematically too low on recent traffic | delayed conversions counted as negatives | wait window or delay-aware likelihood (section 51.8) |
# | The notebook is slow on CPU | the CPU budget still trains on 500k rows | expected (~1–2 min in our CPU runs); `BOOTCAMP_BUDGET=smoke` only proves the code runs |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Define normalised entropy. Why do ads teams prefer it (and calibration) to AUC as the headline offline metric?
# 2. Hashing trick: what does it provide, what does it cost, and how do you pick the number of buckets? Why must the hash be stable?
# 3. Estimate the memory of the embedding tables for a model with 1 billion ids at dimension 64, with Adam. How would you train it across GPUs?
# 4. What does a DCN cross layer compute, and why can it learn interactions logistic regression cannot? How does it relate to factorisation machines / DeepFM?
# 5. You downsample negatives at rate `w`. Derive the re-calibration formula. What breaks if you forget it?
# 6. Your conversion model under-predicts on the freshest traffic. Explain the cause and two ways to fix it, with their trade-offs.
#
# **Next chapter:** [Chapter 52 · Multi-task ranking & exploration](../ch52-multitask-ranking-and-exploration/notebook.ipynb) — from one probability to many: multi-task models for clicks and conversions, objective blending, auctions and exploration.
