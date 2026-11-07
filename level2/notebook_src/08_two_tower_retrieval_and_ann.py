# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/08_two_tower_retrieval_and_ann.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 08 · Two-tower retrieval & ANN
#
# **Day 4 · ~90 minutes · CPU core (training takes under a minute)**
#
# The two-tower model is the default first-stage retriever at large
# companies (YouTube, Google Play, Pinterest, Instagram, TikTok all describe
# variants): one network embeds the user, another embeds the item, and
# retrieval is a nearest-neighbour search in the item index. It generalises
# matrix factorisation with features, which is what makes it work for new
# users and items, but it has a well-known popularity bias that you must
# correct. Then the index has to answer in a millisecond, which is an
# approximate-search problem with its own trade-offs.
#
# By the end you will be able to:
#
# 1. build a two-tower model in PyTorch with id, history and categorical features;
# 2. train it with in-batch softmax negatives and explain, then correct, their sampling bias (logQ correction);
# 3. compare it with notebook 07's ALS on recall@50/100, popularity bias and cold-start users;
# 4. index the item embeddings with FAISS: exact (`IndexFlatIP`), `IndexIVFFlat` and `IndexHNSWFlat`, and read a recall–latency–parameter table;
# 5. sketch the serving cascade around it: what is precomputed, what runs per request, and the latency budget.
#
# **You will produce:** `artifacts/level2/two_tower_item_emb.npy` (item embeddings) and `artifacts/level2/ann_benchmark.csv`.

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
# --- Catch up: clicks, evaluation sets and notebook 07's ALS baseline --------
import time

import numpy as np
import pandas as pd
import torch

from shoprank import retrieval as rt
from shoprank import store
from shoprank.grading import nb08 as grade
from shoprank.metrics import catalog_coverage

torch.set_num_threads(2)  # Colab CPUs have 2 cores; more threads only add overhead here
t0 = time.time()
world, users, items, imps = store.load("small", columns=["request_id", "day", "user_id", "item_id", "position", "clicked"])
clicks = imps.loc[imps["clicked"], ["request_id", "day", "user_id", "item_id"]]
train_clicks = clicks[clicks["day"].isin(store.TRAIN_DAYS)].sort_values(["user_id", "day", "request_id"])
valid_clicks = clicks[clicks["day"].isin(store.VALID_DAYS)]
N_USERS, N_ITEMS = len(users), len(items)

truth_clicks = rt.truth_sets(valid_clicks)
eval_users = np.array(sorted(truth_clicks))
truth_true = rt.true_top_items(world, eval_users, 20, items["created_day"].to_numpy() <= max(store.VALID_DAYS))

user_hist = rt.click_matrix(train_clicks, N_USERS, N_ITEMS)
X_als, Y_als = rt.implicit_als(user_hist, factors=32, reg=0.1, alpha=20.0, iterations=8)
cands_als = rt.score_topk(X_als, Y_als, eval_users, 100)
pop = rt.popularity_scores(train_clicks, N_ITEMS)
cold_users = np.array([u for u in eval_users if user_hist[u].nnz == 0])
print(f"train clicks {len(train_clicks):,} | eval users {len(eval_users):,} (cold {len(cold_users)}) | "
      f"ALS ready | {time.time() - t0:.0f}s")

# %% [markdown]
# **Expected:** about 244,000 training clicks, about 7,700 evaluation users
# of whom about 270 are cold (no training clicks), in under a minute. The
# ALS model is notebook 07's, retrained here (a few seconds) so this
# notebook does not depend on 07 having run.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Two-tower model** | two networks embed users and items separately; the score is a dot product | `TwoTower` in 8.1 |
# | **Embedding** | a learned vector for an id or category | 32 numbers per user id |
# | **Temperature τ** | divides the scores before the softmax; small τ sharpens it | τ = 0.05 |
# | **In-batch negatives** | the other items in the batch serve as negatives for each user | a 1024 × 1024 logit matrix |
# | **Sampling probability q_i** | how likely item i is to appear in a batch | proportional to its clicks |
# | **logQ correction** | subtract log q_i from item i's logit during training only | recall 0.24 → 0.41 |
# | **Accidental hit** | the user's own positive appearing again as a negative | masked in `rt.in_batch_softmax_loss` |
# | **ANN search** | approximate nearest-neighbour search: fast, slightly incomplete | FAISS |
# | **IVF** | k-means cells; search only the `nprobe` closest cells | `IndexIVFFlat` |
# | **HNSW** | a layered neighbour graph searched greedily with a beam of `efSearch` | `IndexHNSWFlat` |
# | **Product quantisation** | compressing vectors into a few bytes of codebook ids | IVF-PQ, 32 floats → 16 bytes |
# | **Recall vs exact** | share of the exact top-k that the approximate search returns | 0.92 at `nprobe = 4` |

# %% [markdown]
# ## 8.1 · The model
#
# ```
#  user tower                                   item tower
#  ──────────                                   ──────────
#  user id ─► emb(32) ─┐                        item id ─► emb(32) ─┐
#  last 50 clicks ─► mean of item emb(32) ─┤    category ─► emb(16) ─┼─► MLP ─► L2-norm ─► v_i
#  segment, device, country ─► embs ──────┼─► MLP ─► L2-norm ─► u     price decile ─► emb(8) ┘
#                                          ┘
#                      score(u, i) = u · v_i / τ      (τ = 0.05)
# ```
#
# The towers never see each other's inputs, so **item vectors can be
# precomputed and indexed**, and per request only the user tower runs
# (one small forward pass), followed by a nearest-neighbour search. That
# separation is the point of the architecture, and its main limitation (no
# user×item feature crosses; those belong in the ranker).
#
# Design choices worth defending in an interview:
#
# * **history mean** shares the item id embedding table between the towers
#   (fewer parameters, new users with a few clicks get a meaningful vector);
#   when training on a click (u, i), item i is removed from u's history,
#   otherwise the model learns to "retrieve what is in the history";
# * **L2 normalisation + temperature τ**: cosine scores keep training stable;
#   a small τ sharpens the softmax;
# * **small initialisation of id embeddings** (std 0.05): an id never seen in
#   training (a cold user or item) then contributes almost nothing, and the
#   feature embeddings carry the vector.

# %%
import torch.nn as nn
import torch.nn.functional as F

D, HIST_LEN, PAD = 32, 50, N_ITEMS          # PAD: an extra "no item" row in the item table

hist = np.full((N_USERS, HIST_LEN), PAD, dtype=np.int64)  # last 50 training clicks per user
for u, g in train_clicks.groupby("user_id")["item_id"]:
    v = g.to_numpy()[-HIST_LEN:]
    hist[u, :len(v)] = v
HIST = torch.from_numpy(hist)
SEG = torch.tensor(users["segment"].to_numpy(), dtype=torch.long)
DEV = torch.tensor(pd.Categorical(users["device"]).codes, dtype=torch.long)
CTY = torch.tensor(pd.Categorical(users["country"]).codes, dtype=torch.long)
CAT = torch.tensor(items["category"].to_numpy(), dtype=torch.long)
PRICE_BIN = torch.tensor(pd.qcut(np.log(items["price"]), 10, labels=False).to_numpy(), dtype=torch.long)


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_id = nn.Embedding(N_USERS, D)
        self.item_id = nn.Embedding(N_ITEMS + 1, D, padding_idx=PAD)
        for emb in (self.user_id, self.item_id):
            nn.init.normal_(emb.weight, 0.0, 0.05)
        with torch.no_grad():
            self.item_id.weight[PAD] = 0.0
        self.segment, self.device, self.country = nn.Embedding(5, 8), nn.Embedding(3, 4), nn.Embedding(5, 4)
        self.category, self.price = nn.Embedding(int(CAT.max()) + 1, 16), nn.Embedding(10, 8)
        self.user_mlp = nn.Sequential(nn.Linear(D + D + 16, 64), nn.ReLU(), nn.Linear(64, D))
        self.item_mlp = nn.Sequential(nn.Linear(D + 16 + 8, 64), nn.ReLU(), nn.Linear(64, D))

    def user(self, u, exclude_item=None):
        h = HIST[u]
        keep = h != PAD
        if exclude_item is not None:              # do not let the target leak through the history
            keep &= h != exclude_item[:, None]
        h_mean = (self.item_id(h) * keep[..., None]).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)
        x = torch.cat([self.user_id(u), h_mean, self.segment(SEG[u]), self.device(DEV[u]), self.country(CTY[u])], 1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item(self, i):
        x = torch.cat([self.item_id(i), self.category(CAT[i]), self.price(PRICE_BIN[i])], 1)
        return F.normalize(self.item_mlp(x), dim=-1)


print(f"parameters: {sum(p.numel() for p in TwoTower().parameters()):,}")

# %% [markdown]
# **Expected:** about 460,000 parameters, almost all of them in the two id
# tables (10,000 users and 4,001 items × 32).

# %% [markdown]
# ## 8.2 · In-batch softmax, and its bias
#
# Training data: (user, clicked item) pairs. For a batch of B pairs, score
# every user against every item in the batch: a B × B matrix of logits. The
# diagonal holds the positives; the other B − 1 items in each row act as
# negatives, for free (no extra item-tower passes). The loss is a softmax
# cross-entropy per row: "pick your clicked item out of the batch".
#
# The catch: the negatives are not uniform over the catalogue. An item
# appears in a batch in proportion to how often it is clicked, its sampling
# probability **q_i**. So popular items are used as negatives far more often,
# and the model learns to push them down. Mathematically, in-batch softmax
# estimates p(i | u) / q_i rather than p(i | u). The **logQ correction**
# (Yi et al., 2019, "Sampling-bias-corrected neural modeling") subtracts
# log q_i from each logit during training (only during training), which
# turns the estimate back into p(i | u) (up to a constant).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-two-tower-in-batch.svg" alt="Two-tower model and the in-batch softmax: diagonal positives, off-diagonal free negatives, logQ correction" width="100%">
#
# *What to notice:* only the dot product connects the two towers, so item vectors can be computed once and indexed. Every off-diagonal cell is a negative that costs nothing extra, and a popular item fills many of those cells.
#
# > 💡 **Intuition.** In-batch training is a speed-dating evening: each person must pick their date out of everyone who happens to be in the room. A celebrity turns up at every event, so they are "not picked" by almost everyone and the model learns that celebrities are unattractive. Subtracting $\log q_i$ is telling the model "this person was in the room 50 times more often than average; discount the rejections accordingly". *Where it breaks down:* $q_i$ is estimated from click counts, so a brand-new item has a noisy $q_i$, and its correction is noisy too.
#
# > **Exercise 1.** Write `in_batch_softmax_loss(user_emb, item_emb, log_q=None)`:
# > tensors of shape [B, d] and [B, d] (row b of each is a positive pair),
# > and optionally `log_q` of shape [B] (log sampling probability of each
# > batch item). Return the mean cross-entropy of the logits
# > `user_emb @ item_emb.T` (minus `log_q[j]` in column j, if given), with
# > the diagonal as the target. The grader compares with a reference on
# > random tensors, including gradients.

# %%
def in_batch_softmax_loss(user_emb, item_emb, log_q=None):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(in_batch_softmax_loss)

# %% [markdown]
# The training loop uses the library version, `rt.in_batch_softmax_loss`,
# which also masks *accidental hits*: when a popular item appears twice in a
# batch, its second copy should not count as a negative for the first.

# %%
U_TR = torch.tensor(train_clicks["user_id"].to_numpy(), dtype=torch.long)
I_TR = torch.tensor(train_clicks["item_id"].to_numpy(), dtype=torch.long)
counts = np.bincount(train_clicks["item_id"].to_numpy(), minlength=N_ITEMS).astype(float)
LOG_Q = torch.tensor(np.log(np.maximum(counts, 1.0) / counts.sum()), dtype=torch.float32)
TAU = 0.05


def train_two_tower(logq_correction: bool, epochs: int = 3, batch: int = 1024, lr: float = 3e-3, seed: int = 0):
    torch.manual_seed(seed)
    model = TwoTower()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, t_start = [], time.time()
    for epoch in range(epochs):
        perm, total = torch.randperm(len(U_TR)), 0.0
        for lo in range(0, len(perm), batch):
            b = perm[lo:lo + batch]
            u, i = U_TR[b], I_TR[b]
            loss = rt.in_batch_softmax_loss(model.user(u, exclude_item=i) / TAU, model.item(i),
                                            log_q=LOG_Q[i] if logq_correction else None, item_ids=i)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / len(perm))
    print(f"logQ={logq_correction}: loss per epoch {np.round(losses, 3).tolist()} | {time.time() - t_start:.0f}s")
    return model


@torch.no_grad()
def embed(model):
    model.eval()
    return model.user(torch.arange(N_USERS)).numpy(), model.item(torch.arange(N_ITEMS)).numpy()


tt_plain = train_two_tower(logq_correction=False)
tt_logq = train_two_tower(logq_correction=True)

# %% [markdown]
# **Expected:** three epochs per model, each model in about 10–40 s on a
# 2-core CPU (the variation is the machine, not the code). The loss falls
# from about 6.24 to about 5.92 for the plain model and from about 6.35 to
# 5.93 with logQ; the two losses are on different scales (the logQ logits
# include −log q), so do not compare them with each other. For scale:
# log(1024) ≈ 6.93 is the loss of a model that guesses uniformly. A GPU is not needed at this size: the batch is a 1024 ×
# 1024 matrix.

# %% [markdown]
# ## 8.3 · Evaluation against ALS

# %%
new_items = set(np.flatnonzero(items["created_day"].to_numpy() >= min(store.VALID_DAYS)).tolist())
log_pop = np.log1p(pop)
results, cands = {}, {"ALS (notebook 07)": cands_als}
for name, model in [("two-tower, no correction", tt_plain), ("two-tower + logQ", tt_logq)]:
    ue, ie = embed(model)
    cands[name] = rt.score_topk(ue, ie, eval_users, 100)
for name, c in cands.items():
    lists = [c[int(u)][:100] for u in eval_users]
    results[name] = {
        "recall@50 clicks": rt.recall_at_k_batch(c, truth_clicks, 50),
        "recall@100 clicks": rt.recall_at_k_batch(c, truth_clicks, 100),
        "recall@50 true top-20": rt.recall_at_k_batch(c, truth_true, 50),
        "recall@100 true top-20": rt.recall_at_k_batch(c, truth_true, 100),
        "recall@100 cold users": rt.recall_at_k_batch(c, {int(u): truth_clicks[int(u)] for u in cold_users}, 100),
        "coverage@100": catalog_coverage(lists, N_ITEMS),
        "mean log-popularity@100": float(np.mean([log_pop[lst].mean() for lst in lists])),
        "new items per list": float(np.mean([len(set(lst) & new_items) for lst in lists])),
    }
table = pd.DataFrame(results).T
table.round(3)

# %% [markdown]
# **Expected (our runs; ±0.01 between runs):**
#
# | | recall@100 clicks | recall@100 true top-20 | cold users | mean log-pop. |
# |---|---|---|---|---|
# | ALS | about 0.335 | about 0.59 | about 0.02 | about 5.15 |
# | two-tower, no correction | about 0.24 | about 0.42 | about 0.04 | about 3.6 |
# | two-tower + logQ | **about 0.41** | **about 0.74** | **about 0.26** | about 5.3 |
#
# Three lessons:
#
# 1. **Without the correction the two-tower model is worse than ALS**, and
#    the popularity column shows why: it retrieves unpopular items (mean
#    log-popularity about 3.6 vs 5.15) and covers the whole catalogue
#    (coverage 1.0). It over-penalised popular items, exactly as predicted.
# 2. **With logQ it beats ALS by a wide margin** on both truths (about +0.07
#    recall on clicks, +0.15 on the true top-20): features and the history
#    tower add information the click matrix alone does not have.
# 3. **Cold users** get a real vector from their segment, device and country:
#    recall about 0.25 vs ALS's 0.02 (and above global popularity's 0.18 in
#    notebook 07).
#
# New items, on the other hand, are still rare in the logQ model's lists
# (about 0.45 per list of 100): their id embedding is untrained and the
# category/price embeddings alone do not lift them into the top 100. A
# content-based fallback and an exploration quota (notebook 07) are still
# needed; richer content features (text, image) would help the item tower.

# %%
ue_logq, ie_logq = embed(tt_logq)
np.save(ART / "level2" / "two_tower_item_emb.npy", ie_logq)
print("saved two_tower_item_emb.npy", ie_logq.shape)

# %% [markdown]
# ## 8.4 · Approximate nearest-neighbour search with FAISS
#
# Serving = "find the 100 item vectors with the largest dot product with
# u". With 4,000 items, exact search is a 4,000 × 32 matrix–vector product:
# microseconds. With 100 million items it is 3.2 billion multiply-adds per
# request, far too slow, so we trade a little recall for a lot of speed:
#
# * **`IndexFlatIP`**: exact brute force. The reference for recall.
# * **`IndexIVFFlat`**: k-means the vectors into `nlist` cells; at query time
#   scan only the `nprobe` cells closest to the query. Recall and latency
#   both grow with `nprobe`.
# * **`IndexHNSWFlat`**: a navigable small-world graph; greedy search from an
#   entry point. Recall and latency grow with `efSearch`; build is slower,
#   memory higher (graph links), queries very fast.
#
# Our catalogue is too small for ANN to matter, so we build a synthetic
# **100,000-item** index: each vector is a real learned item embedding plus
# noise, re-normalised (so its structure resembles the real one), and query
# with 2,000 real user embeddings. Timings are from this machine; on Colab's
# CPU expect the same *ratios* but different absolute numbers.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-ann-tradeoff.svg" alt="ANN recall against latency for IVF (nprobe) and HNSW (efSearch), and exact search" width="100%">
#
# *What to notice:* every index is a curve, not a point: the knob (`nprobe`, `efSearch`) moves you along it. The shape matches the table you are about to compute; read your own numbers off the cell below.
#
# > 💡 **Intuition.** IVF is a library with sections: walk to the 4 sections whose signs best match your topic and search only those shelves (`nprobe = 4`). HNSW is asking for directions: take the motorway to roughly the right town, then smaller and smaller streets (`efSearch` is how many routes you keep open). *Where it breaks down:* a good book can sit in a neighbouring section that you did not visit; that missing share is exactly the "recall vs exact" you will measure.
#
# > **Exercise 2.** Write `recall_vs_exact(approx_ids, exact_ids)`: two
# > integer arrays of shape [n_queries, k] (FAISS output, where −1 means "no
# > result"). Return the mean over queries of |approx ∩ exact| / |exact|,
# > ignoring −1 entries. This is how every ANN benchmark reports recall.

# %%
def recall_vs_exact(approx_ids, exact_ids) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(recall_vs_exact)

# %%
import faiss

faiss.omp_set_num_threads(2)
rng = np.random.default_rng(0)
N_BIG, K_NN = 100_000, 100
base = ie_logq[rng.integers(0, N_ITEMS, N_BIG)] + 0.1 * rng.normal(0, 1 / np.sqrt(D), (N_BIG, D))
base = (base / np.linalg.norm(base, axis=1, keepdims=True)).astype(np.float32)
queries = np.ascontiguousarray(ue_logq[eval_users[:2000]], dtype=np.float32)


def bench(index, label, param, value):
    t = time.perf_counter()
    _, ids = index.search(queries, K_NN)
    ms = (time.perf_counter() - t) / len(queries) * 1000
    return {"index": label, "param": param, "value": value,
            "recall@100 vs exact": rt.recall_vs_exact(ids, exact_ids), "ms/query": ms}


rows = []
t = time.perf_counter()
flat = faiss.IndexFlatIP(D)
flat.add(base)
build_flat = time.perf_counter() - t
t = time.perf_counter()
_, exact_ids = flat.search(queries, K_NN)
rows.append({"index": "Flat (exact)", "param": "-", "value": 0, "recall@100 vs exact": 1.0,
             "ms/query": (time.perf_counter() - t) / len(queries) * 1000})

t = time.perf_counter()
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, 512, faiss.METRIC_INNER_PRODUCT)
ivf.train(base[:30_000])
ivf.add(base)
build_ivf = time.perf_counter() - t
for nprobe in [1, 4, 16, 64]:
    ivf.nprobe = nprobe
    rows.append(bench(ivf, "IVFFlat nlist=512", "nprobe", nprobe))

t = time.perf_counter()
hnsw = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 80
hnsw.add(base)
build_hnsw = time.perf_counter() - t
for ef in [32, 128, 512]:
    hnsw.hnsw.efSearch = ef
    rows.append(bench(hnsw, "HNSW M=32", "efSearch", ef))

ann = pd.DataFrame(rows)
print(f"build: flat {build_flat:.2f}s | IVF {build_ivf:.1f}s | HNSW {build_hnsw:.1f}s")
ann.round(4)

# %% [markdown]
# **Expected (shape, not exact numbers):** exact search costs about 0.4 ms
# per query at 100,000 vectors (it grows linearly with the catalogue).
# IVF with `nprobe=1` is about 30× faster but finds only about half of the
# true top 100; recall climbs to about 0.92 at `nprobe=4` and about 0.997 at
# `nprobe=16`, still about 15× faster than exact. HNSW is fast too but needs
# a large `efSearch` here (about 0.63 → 0.90 → 0.99 for 32 → 128 → 512), and
# it is the slowest to build (a few seconds against under half a second for
# IVF). At `efSearch=512` it is slower than IVF at a similar recall.
#
# Why is HNSW comparatively weak here, when it wins most public benchmarks?
# Those benchmarks query with vectors from the *same distribution* as the
# index. Two-tower **user** vectors are not item vectors: they live in a
# different region of the sphere. Greedy graph search, built on item–item
# neighbourhoods, needs a wider beam to find the right region from such
# queries. Check it:

# %%
item_queries = np.ascontiguousarray(base[rng.integers(0, N_BIG, 2000)])
_, exact_items = flat.search(item_queries, K_NN)
hnsw.hnsw.efSearch = 128
_, approx_items = hnsw.search(item_queries, K_NN)
ivf.nprobe = 16
_, approx_items_ivf = ivf.search(item_queries, K_NN)
print(f"best cosine: user->item {np.max(queries @ base[:20_000].T, axis=1).mean():.2f} | "
      f"item->item {np.sort(item_queries[:200] @ base.T, axis=1)[:, -2].mean():.2f}")
print(f"item->item queries: HNSW efSearch=128 recall {rt.recall_vs_exact(approx_items, exact_items):.3f} | "
      f"IVF nprobe=16 recall {rt.recall_vs_exact(approx_items_ivf, exact_items):.3f}")

# %% [markdown]
# **Expected:** the best user→item cosine is only about 0.6, while items have
# neighbours at about 0.99. With item→item queries ("more like this") HNSW
# reaches about 0.998 recall at `efSearch=128`, against about 0.90 for
# user→item queries at the same setting (IVF is at about 0.998 for both). Lesson for design rounds: **benchmark ANN with your real query
# distribution**. For user→item retrieval, IVF-style partitioning (or
# ScaNN, or graph indexes built for out-of-distribution queries) is often the
# safer choice; and an IVF index's `nprobe` gives a clean recall–latency knob
# to set per surface.
#
# Real deployments also compress vectors (IVF-PQ: product quantisation, e.g.
# 32 floats → 16 bytes) to fit hundreds of millions of items in RAM, at a
# further recall cost, then **re-rank** the top few hundred by exact dot
# product.

# %%
ann.to_csv(ART / "level2" / "ann_benchmark.csv", index=False)
print("saved ann_benchmark.csv")

# %% [markdown]
# ## 8.5 · The serving cascade
#
# ```
# OFFLINE (daily / hourly)                       ONLINE (per request, p99 budget ~150 ms)
# ────────────────────────                       ─────────────────────────────────────────
# train two-tower on clicks (logQ) ───┐          request ─► features: user history, context        ~5 ms
# item tower over the catalogue  ──► ANN index   user tower forward pass (CPU, ~0.2 ms)             ~2 ms
#   (new items: embed on upload,     (IVF/HNSW)  ANN search: top 500                                ~5 ms
#    incremental add)                      ▲     + co-visitation, fresh, popularity (07), merge    ~5 ms
# co-visitation tables, popularity ────────┘     ranker features for 500 candidates (feature store) ~20 ms
# ranker (LambdaMART / DNN, notebook 06)  ──────► rank 500 → top 50                                 ~20 ms
#                                                business rules, diversity, ads blending             ~5 ms
#                                                log impressions WITH positions + exploration flag  (async)
# ```
#
# Points interviewers listen for:
#
# * **Precompute everything that does not depend on the request**: item
#   embeddings, the index, co-visitation tables, item features.
# * **Version together**: the user tower and the index must come from the same
#   training run. A user vector from model v7 searched in an index of v6 item
#   vectors is silent garbage. Swap index and model atomically.
# * **Logging closes the loop**: positions, the candidate source and
#   exploration flags are what notebook 05's debiasing needs next month.
# * **Degrade gracefully**: if the ANN service times out, serve popularity +
#   co-visitation candidates; never an empty feed (notebook 17).

# %% [markdown]
# > **Exercise 3.** Explain in your own words, with a small numeric example,
# > why in-batch negatives over-penalise popular items, why subtracting
# > log q_i during training (and not at serving) fixes it, and one situation
# > where you might *want* to keep part of the popularity penalty. Write
# > your answer in the cell below.

# %% [markdown]
# *Your answer:*

# %% [markdown]
# ## 🧮 Key equations
#
# **1 · The two-tower score.** In words: the cosine similarity of the two normalised vectors, divided by a temperature.
#
# $$
# s(u, i) = \frac{\mathbf{u}^\top \mathbf{v}_i}{\tau}, \qquad \lVert \mathbf{u} \rVert = \lVert \mathbf{v}_i \rVert = 1
# $$
#
# * *Example:* cosine 0.6 with $\tau = 0.05$ gives a logit of 12; cosine 0.5 gives 10, so the softmax prefers the first by $e^{2} \approx 7.4\times$.
#
# **2 · In-batch softmax (InfoNCE) with the logQ correction.** In words: for each user in the batch, cross-entropy of picking their own item out of the batch's items, with every item's logit lowered by the log of how often it is sampled.
#
# $$
# L = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\!\big(s(u_b, i_b) - \log q_{i_b}\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_b, i_j) - \log q_{i_j}\big)}
# $$
#
# * $B$: batch size; $q_i$: item $i$'s sampling probability (its share of training clicks). Without the correction, in-batch softmax learns $\log \frac{p(i \mid u)}{q_i}$ + const, which penalises popular items; at serving, use $s(u, i)$ alone. *Example:* $B = 1024$ and a model that knows nothing: $L = \log 1024 = 6.93$.
#
# **3 · Recall of an approximate index.** In words: the share of the exact top-$k$ that the approximate search also returned.
#
# $$
# \operatorname{Recall@}k = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{\lvert \text{ANN}_k(q) \cap \text{Exact}_k(q) \rvert}{k}
# $$
#
# * *Example:* IVF returns 92 of the exact top 100 → 0.92 for that query.
#
# **4 · Cost of exact search.** In words: one dot product per item per query.
#
# $$
# \text{cost} = N \cdot d \ \text{multiply-adds per query}
# $$
#
# * *Example:* $N = 10^8$ items, $d = 32$: $3.2 \times 10^9$ per request, far beyond a 5 ms budget; IVF with $n_\text{list} = 4{,}096$ cells and $n_\text{probe} = 16$ scans about $16/4096 = 0.4\%$ of them.
#
# The cell below checks equation 2 against `F.cross_entropy`.

# %%
# In-batch softmax with logQ written out, checked against F.cross_entropy
import torch.nn.functional as F

g = torch.Generator().manual_seed(0)
B, d, tau = 6, 8, 0.05
u = F.normalize(torch.randn(B, d, generator=g), dim=1)
v = F.normalize(torch.randn(B, d, generator=g), dim=1)
log_q = torch.log(torch.tensor([0.30, 0.20, 0.20, 0.15, 0.10, 0.05]))
logits = u @ v.T / tau - log_q  # column j shifted by -log q_j
by_hand = -(logits.diag() - torch.logsumexp(logits, dim=1)).mean()
library = F.cross_entropy(logits, torch.arange(B))
print(f"by hand {by_hand.item():.5f} | F.cross_entropy {library.item():.5f}")
print(f"a model that knows nothing, batch 1024: log(1024) = {np.log(1024):.3f}")

# %% [markdown]
# **Expected:** the two losses agree to five decimals, and `log(1024) = 6.931`, the ceiling the training loss in 8.2 started just below.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Loss stuck at log(batch size) ≈ 6.93 | temperature too high (τ = 1 with normalised vectors), or learning rate too low | divide scores by τ ≈ 0.05; check `lr` |
# | Recall far below ALS and retrieved items are obscure | no logQ correction (in-batch popularity bias) | subtract `log_q` of the batch items from the logits in training |
# | Great training loss, poor validation recall | the target item leaks through the history tower | exclude the target from the history (`exclude_item`) |
# | Training very slow on Colab | too many threads, or Python-level work per example | `torch.set_num_threads(2)`; build batches with tensor indexing as here |
# | `RuntimeError` / wrong results from FAISS | float64 or non-contiguous arrays | `np.ascontiguousarray(x, dtype=np.float32)` |
# | IVF warns about too few training points | fewer than ~39 × `nlist` training vectors | train on more vectors or lower `nlist` |
# | ANN recall fine on item queries, poor on user queries | out-of-distribution queries for the index | benchmark with real user vectors; use IVF/ScaNN or larger `efSearch` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Draw the two-tower architecture. Why can't it use user×item cross features, and where do those go instead?
# 2. In-batch negatives: why are they cheap, what bias do they introduce, and how does the logQ correction fix it? Name one alternative negative-sampling strategy and its trade-off.
# 3. Your two-tower model's recall@100 is below plain matrix factorisation. List the first five things you check.
# 4. IVF vs HNSW vs product quantisation: how does each work, and which knobs trade recall for latency or memory?
# 5. How do you get a brand-new item into the index within minutes, and what must stay consistent between the user tower and the index?
# 6. Your ANN index loses 3% recall@100 vs exact search. How do you decide whether that matters?
#
# ➡️ **Next:** [`09_pytorch_training_done_right.ipynb`](09_pytorch_training_done_right.ipynb) — Day 5: the training loop itself, done properly: data loaders, checkpoints and resume, determinism, schedules, and debugging a model that will not learn.
