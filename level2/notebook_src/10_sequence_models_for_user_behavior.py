# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/10_sequence_models_for_user_behavior.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 10 · Sequence models for user behaviour
#
# **Day 5 · ~90 minutes · CPU (a GPU only speeds up the optional parts)**
#
# Feeds, "up next" rails and session recommendations are often framed as
# **next-item prediction**: given what a user clicked, in order, what will
# they click next? The model is a small causal transformer, and the
# engineering is in the details interviewers ask about: masking, padding,
# and an evaluation that respects time and does not flatter the model.
#
# By the end you will be able to:
#
# 1. turn an impression log into per-user, time-ordered, left-padded click sequences;
# 2. build the two baselines every sequence model must beat: most-popular and last-item co-visitation;
# 3. write causal and padding masks and **test** that they do what you think (no peeking, no NaNs);
# 4. train a SASRec-style model on CPU in under two minutes;
# 5. evaluate with leave-last-out inside the validation week, and explain why **sampled** HR/NDCG can mislead.
#
# **You will produce:** `artifacts/level2/nb10/metrics.csv`, a table of full-catalogue
# and sampled HR@10/NDCG@10 for every model, and the reusable module
# [`src/shoprank/seqrec.py`](../../src/shoprank/seqrec.py).

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
# --- Catch up: ShopStream (rebuilt in ~30 s if this runtime has never seen it)
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from shoprank import seqrec as sr
from shoprank import store
from shoprank import torchkit as tk
from shoprank.grading import nb10 as grade

torch.set_num_threads(2)
OUT = ART / "level2" / "nb10"
OUT.mkdir(parents=True, exist_ok=True)

world, users, items, imps = store.load("small")
N_ITEMS = len(items)
print(f"{len(imps):,} impressions | {N_ITEMS:,} items | {len(users):,} users")

# %% [markdown]
# **Expected:** about 2.45 million impressions, 4,000 items, 10,000 users.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Next-item prediction** | given a user's past events in order, predict the next one | the last validation click |
# | **Left padding** | fill short sequences with 0s at the *start* | `[0, 0, 18, 5, 10]` |
# | **Padding token** | the id that means "no item here" | 0, so item ids are shifted by +1 |
# | **Causal mask** | blocks attention from position i to any later position j > i | `causal_mask(n)`, Exercise 2 |
# | **Padding mask** | blocks attention to padding positions | `padding_mask(lengths, n)`, Exercise 3 |
# | **Self-attention** | each position builds its output as a weighted mix of the others | 2 blocks in `sr.SASRec` |
# | **SASRec** | a causal transformer over item ids for next-item recommendation | Kang & McAuley, 2018 |
# | **Tied embeddings** | the same item table embeds inputs and scores outputs | hidden state · item embedding |
# | **Leave-last-out** | hold out each user's last event as the test target | here inside the validation week only |
# | **HR@10** | share of users whose target is in the top 10 | about 0.07 for SASRec |
# | **Sampled metric** | rank the target against a sample of negatives, not the whole catalogue | 100 uniform negatives in 10.6 |

# %% [markdown]
# ## 10.1 · From impressions to sequences
#
# A sequence model sees each user's **clicks**, in time order. Three
# decisions hide in that sentence:
#
# * **What counts as an event.** Clicks here; purchases, add-to-carts or dwell
#   time in a real system. Impressions without a click are not in the sequence
#   (they are useful as negatives, which we do not use here).
# * **Order.** By event time, ties broken deterministically (same request: by position).
# * **Shape.** Models want a fixed-size integer array. Keep the last `max_len`
#   events, shift item ids by +1 so that **0 can mean padding**, and pad on
#   the **left**, so the most recent item is always in the last column,
#   whatever the history length.

# %% [markdown]
# > 💡 **Intuition.** Left padding right-aligns every history like the end of a receipt: the most recent purchase is always on the last line, whether the receipt is 3 lines or 30 long, so "predict from the last column" works for everyone. *Where it breaks down:* receipts have no blank lines at the top that a reader might try to read; a model does, and must be told (masked) not to.

# %%
train_clicks = sr.click_events(imps, store.TRAIN_DAYS)
valid_clicks = sr.click_events(imps, store.VALID_DAYS)
per_user = train_clicks.groupby("user_id").size()
print(f"train clicks {len(train_clicks):,} from {len(per_user):,} users | valid clicks {len(valid_clicks):,} "
      f"from {valid_clicks['user_id'].nunique():,} users")
print("clicks per user (train weeks), quantiles 10/50/90/99%:",
      per_user.quantile([0.1, 0.5, 0.9, 0.99]).round().astype(int).tolist())
train_clicks.head(3)

# %% [markdown]
# **Expected:** about 244,000 training clicks from about 9,300 users, about
# 44,000 validation clicks from about 7,700 users. Clicks per user are
# heavy-tailed: a median around 18, a 99th percentile in the low hundreds. A
# `max_len` of 30 covers the whole history of most users and the recent
# history of the heaviest ones.

# %% [markdown]
# > **Exercise 1.** Write `build_sequences(df, max_len)`. `df` has columns `user_id`, `item_id`,
# > `ts` (rows in any order). Return `(users, seqs)`: `users` is a NumPy array of the distinct
# > user ids in ascending order; `seqs` is an `int64` array of shape `(len(users), max_len)`
# > whose row `r` holds the **last** `max_len` items of `users[r]` in time order, as
# > `item_id + 1`, left-padded with `0`. Rows with equal `ts` keep their order in `df`
# > (use a stable sort).

# %%
def build_sequences(df, max_len):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(build_sequences)

# %% [markdown]
# The package version is `sr.build_sequences` (vectorised, no Python loop over
# users); the rest of the notebook uses it.
#
# ## 10.2 · An evaluation that respects time
#
# The common *leave-one-out* protocol holds out each user's last event and
# trains on everything else, including other users' events from *after*
# that moment. Here time is respected:
#
# * the model is trained only on **training-week** clicks (days 0–41);
# * every user with a click in the **validation week** (days 42–48) gets
#   one test case: the **target** is their *last* validation click; the
#   **history** is every earlier click (training weeks plus earlier validation clicks).
#
# Using earlier validation clicks as *input* is legitimate: at serving
# time you know everything the user did before the moment you predict for.

# %%
MAX_LEN = 30
target_rows = valid_clicks.groupby("user_id").tail(1)
history = pd.concat([train_clicks, valid_clicks.drop(target_rows.index)])
hist_users, hist_seqs = sr.build_sequences(history, MAX_LEN)

eval_users = np.intersect1d(target_rows["user_id"].to_numpy(), hist_users)  # need >= 1 earlier click
H = hist_seqs[np.searchsorted(hist_users, eval_users)]                       # (n_eval, MAX_LEN) inputs
T = target_rows.set_index("user_id").loc[eval_users, "item_id"].to_numpy() + 1  # shifted targets
repeat = np.mean([t in h for t, h in zip(T, H)])
print(f"{len(T):,} test users | {len(target_rows) - len(T)} users skipped (no earlier click) "
      f"| target already in the last {MAX_LEN} clicks: {repeat:.1%}")

# %% [markdown]
# **Expected:** about 7,600 test users, a few dozen skipped, and about 6% of
# targets already in the user's recent history (re-clicks). Whether to allow
# already-clicked items as recommendations is a product decision; we keep
# them and say so.
#
# **Metrics.** For each test user, rank all 4,000 items and find the target's
# rank `r` (0 = top). **HR@10** = share of users with `r < 10`;
# **NDCG@10** = mean of `1/log2(r + 2)` when `r < 10`, else 0 (one relevant
# item, so the ideal DCG is 1). Ties count *against* the target, so a constant
# scorer gets no free hits.

# %%
results = {}
neg_rng = np.random.default_rng(0)
NEG = sr.sample_negatives(T, N_ITEMS, n_neg=100, rng=neg_rng)  # used in 10.6


def evaluate_scores(name, scores):
    """scores: (n_eval, N_ITEMS + 1) array, column 0 = padding (-inf)."""
    full = sr.hr_ndcg(sr.rank_of_target(scores, T))
    sampled = sr.hr_ndcg(sr.rank_of_target(scores, T, NEG))
    results[name] = {**{f"full {k}": v for k, v in full.items()},
                     **{f"sampled {k}": v for k, v in sampled.items()}}
    return {k: round(v, 4) for k, v in results[name].items()}


# Baseline 1 - most popular in the training weeks (the same list for everyone)
pop = np.bincount(train_clicks["item_id"] + 1, minlength=N_ITEMS + 1).astype(float)
pop_score = np.log1p(pop)
pop_score[0] = -np.inf
print("popularity  :", evaluate_scores("popularity", np.tile(pop_score, (len(T), 1))))

# %% [markdown]
# **Baseline 2: last-item co-visitation.** Count, over all training sequences,
# how often item `b` is clicked right after item `a`. Score items by the row of
# the user's **last** item, breaking ties by popularity. This is "people who
# clicked this next clicked…", the backbone of many production "related items" rails.

# %%
from scipy.sparse import csr_matrix

_, full_seqs = sr.build_sequences(train_clicks, max_len=int(per_user.max()))
a, b = full_seqs[:, :-1].ravel(), full_seqs[:, 1:].ravel()
keep = (a > 0) & (b > 0)
covis = csr_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(N_ITEMS + 1, N_ITEMS + 1))
covis.sum_duplicates()
print(f"co-visitation pairs: {keep.sum():,} transitions, {covis.nnz:,} distinct (a -> b) pairs")

covis_scores = covis[H[:, -1]].toarray() + 1e-3 * pop_score / pop_score[1:].max()
covis_scores[:, 0] = -np.inf
print("co-visitation:", evaluate_scores("co-visitation", covis_scores))

# %% [markdown]
# **Expected:** popularity reaches a full-catalogue HR@10 of about 0.027 and
# co-visitation about 0.037: a 30–40% relative gain from one line of
# personalisation. Absolute numbers are small because there are 4,000
# candidates and one target per user. Ignore the "sampled" columns until 10.6.
#
# ## 10.3 · Masks: what each position may look at
#
# Training a next-item model on a sequence trains *every* position at once:
# the output at position `t` predicts the item at `t + 1`. That is only
# honest if position `t` cannot see positions after `t`: a **causal
# mask**. And padding positions must not be attended to as if they were
# items: a **padding mask**. We follow PyTorch's convention for boolean
# attention masks: **`True` means blocked**.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-causal-padding-mask.svg" alt="Causal plus padding attention mask for a left-padded sequence, with the unblocked diagonal that avoids NaN" width="100%">
#
# *What to notice:* the green triangle is everything a real position may read (itself and the past). The amber diagonal in the PAD rows is the fix you are about to see: without it those rows are entirely blocked and the softmax returns NaN.
#
# > 💡 **Intuition.** A causal mask is reading a mystery novel with a card held under the current line: each prediction may use everything above the card and nothing below it. Training slides the card down every line at once, which is why one forward pass trains every position. *Where it breaks down:* a reader can peek; a model without the mask peeks *always*, and its training loss looks wonderful.

# %% [markdown]
# > **Exercise 2.** Write `causal_mask(n)`: a `torch.bool` tensor of shape `(n, n)` where entry
# > `[i, j]` is `True` when query position `i` must **not** attend to key position `j` (i.e. `j > i`).

# %%
def causal_mask(n):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(causal_mask)

# %% [markdown]
# > **Exercise 3.** Write `padding_mask(lengths, n)`: given the true lengths of `B` sequences
# > that are **left-padded** to length `n`, return a `torch.bool` tensor of shape `(B, n)` that is
# > `True` at padding positions. `lengths` may be a list or a tensor.

# %%
def padding_mask(lengths, n):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(padding_mask)

# %% [markdown]
# Combining them has a trap. With **left** padding, a padding *query* at
# position 0 may only look at key 0 (causal), which is padding, so its whole
# row is blocked. A softmax over a row of `-inf` is `0/0 = NaN`, and NaNs
# spread: one NaN in a matrix multiply contaminates every output in the layer above.

# %%
pad = sr.padding_mask([2, 4], 4)           # sequences of length 2 and 4, padded to 4
print("padding mask:\n", pad.int().numpy())
naive = sr.attention_mask(pad, fix_all_masked_rows=False)
scores = torch.zeros(2, 4, 4).masked_fill(naive, float("-inf"))
print("softmax rows of sequence 0 with the naive mask:\n", torch.softmax(scores, -1)[0].numpy().round(2))
fixed = sr.attention_mask(pad)             # diagonal unblocked
scores = torch.zeros(2, 4, 4).masked_fill(fixed, float("-inf"))
print("with the diagonal unblocked:\n", torch.softmax(scores, -1)[0].numpy().round(2))

# %% [markdown]
# **Expected:** with the naive mask, the first two rows of sequence 0 (its padding
# positions) are `nan`; with the fix they attend only to themselves, and
# the real positions 2 and 3 attend to positions 2..3 only. The padding rows'
# outputs are never used in the loss or the prediction, so the fix changes
# nothing else. (Right-padding avoids this particular trap, at the price of the
# last real item sitting in a different column for each user.)
#
# ## 10.4 · A small SASRec, and tests that it cannot cheat
#
# `sr.SASRec` (Kang & McAuley, 2018) is: item embedding + learned positional embedding →
# 2 pre-LayerNorm transformer blocks with the combined mask → a final
# LayerNorm. The score of item `j` after position `t` is the dot product of
# the hidden state with item `j`'s embedding (**tied** input and output
# embeddings). Read the 80 lines of
# [`seqrec.py`](../../src/shoprank/seqrec.py): the attention is written out by hand.
#
# Before training anything, three tests that would catch the usual masking bugs:

# %%
torch.manual_seed(0)
model = sr.SASRec(N_ITEMS, MAX_LEN, dim=48, n_heads=2, n_layers=2, dropout=0.1).eval()
print(f"{tk.count_parameters(model):,} parameters")
x = torch.as_tensor(H[:8].copy())
with torch.no_grad():
    base = model(x)

    # 1. Causality: changing the LAST item must not change any earlier position's output.
    x2 = x.clone()
    x2[:, -1] = torch.randint(1, N_ITEMS + 1, (len(x2),))
    out2 = model(x2)
    causal_ok = torch.allclose(base[:, :-1], out2[:, :-1], atol=1e-6) and not torch.allclose(base[:, -1], out2[:, -1])

    # 2. Padding: garbage in padding slots must not change any real position's output.
    is_pad = x == sr.PAD
    x3 = x.clone()
    x3[is_pad] = torch.randint(1, N_ITEMS + 1, (int(is_pad.sum()),))
    out3 = model(x3, pad=is_pad)
    padding_ok = torch.allclose(base[~is_pad], out3[~is_pad], atol=1e-6)

    # 3. No NaNs anywhere, even for a user with a single click.
    one = torch.zeros(1, MAX_LEN, dtype=torch.long)
    one[0, -1] = 7
    no_nan = not torch.isnan(model(one)).any() and not torch.isnan(base).any()

print(f"causal ok: {causal_ok} | padding ok: {padding_ok} | no NaN: {no_nan} | "
      f"padding slots in this batch: {int(is_pad.sum())}")

# %% [markdown]
# **Expected:** about 230,000 parameters (most of them the 4,001 x 48 item
# embedding) and `True` for all three tests. Write tests like these for any
# attention model you build. A model that can see the future trains to a
# beautiful loss and is useless in production, and only a test like #1 tells
# you before the A/B test does.
#
# ## 10.5 · Training
#
# Inputs are the training-week sequences of length `MAX_LEN + 1`: positions
# `0..MAX_LEN-1` are the input, `1..MAX_LEN` the targets (shifted by one).
# The loss is a **full softmax** over all 4,000 items at every non-padding
# target position. With 4,000 items that is affordable; with millions you
# would sample negatives (as in notebook 08) or use a sampled softmax.
# The loop is notebook 09's, with warmup + cosine.

# %%
train_users, train_seqs = sr.build_sequences(train_clicks, MAX_LEN + 1)
X = torch.as_tensor(train_seqs)
EPOCHS, BATCH = 8, 128
steps_per_epoch = (len(X) + BATCH - 1) // BATCH

tk.seed_everything(0)
model = sr.SASRec(N_ITEMS, MAX_LEN, dim=48, n_heads=2, n_layers=2, dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = tk.warmup_cosine(optimizer, warmup_steps=steps_per_epoch, total_steps=EPOCHS * steps_per_epoch)
gen = tk.make_generator(0)
print(f"{len(X):,} training sequences | {int((X[:, 1:] > 0).sum()):,} target positions per epoch | "
      f"loss at init should be about log({N_ITEMS}) = {np.log(N_ITEMS):.2f}")


@torch.no_grad()
def sasrec_scores(model, seqs, batch=1024):
    model.eval()
    out = [model.logits(model(torch.as_tensor(seqs[i:i + batch]))[:, -1]) for i in range(0, len(seqs), batch)]
    return torch.cat(out).numpy()


start = time.time()
for epoch in range(EPOCHS):
    model.train()
    total, count = 0.0, 0
    for idx in torch.randperm(len(X), generator=gen).split(BATCH):
        inp, tgt = X[idx, :-1], X[idx, 1:]
        real = tgt > 0                                    # loss only where there is a next item
        hidden = model(inp)
        loss = nn.functional.cross_entropy(hidden[real] @ model.item_emb.weight.T, tgt[real])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item() * int(real.sum())
        count += int(real.sum())
    if epoch in (0, EPOCHS // 2 - 1, EPOCHS - 1):
        hr = sr.hr_ndcg(sr.rank_of_target(sasrec_scores(model, H), T))["HR@10"]
        print(f"epoch {epoch}: train loss {total / count:.3f} | full HR@10 {hr:.4f} | {time.time() - start:.0f} s")
    else:
        print(f"epoch {epoch}: train loss {total / count:.3f} | {time.time() - start:.0f} s")

S = sasrec_scores(model, H)
print("SASRec      :", evaluate_scores("SASRec", S))

# %% [markdown]
# **Expected:** a training loss of about 8.0 in epoch 0 (still warming up; log 4000 ≈ 8.29 is
# the uniform guess) down to about 6.8; full-catalogue HR@10 of about 0.07, roughly twice
# co-visitation's 0.037 and 2.5x popularity's. Training takes 60–150 s on a 2-core CPU.
#
# The loss stays high in absolute terms because next-click prediction among
# 4,000 items is genuinely uncertain. Compare it with baselines on the
# metric, not with zero.
#
# ## 10.6 · Full-catalogue vs sampled metrics
#
# Many papers (and many internal dashboards) rank the target against **100
# randomly sampled negatives** instead of the whole catalogue, because it
# is cheaper. The "sampled" columns did exactly that, with the same 100
# uniform negatives for every model. Add one more model, SASRec plus a mild
# popularity prior, and compare:

# %%
evaluate_scores("SASRec + 0.5·log(pop)", S + 0.5 * np.where(np.isfinite(pop_score), pop_score, 0.0))
table = pd.DataFrame(results).T
table.to_csv(OUT / "metrics.csv")
print(table.round(4).to_string())

# %% [markdown]
# **Expected (read the columns separately):**
#
# * **Compression.** Co-visitation beats popularity by 30–40% on full HR@10 but
#   by only about 1% on sampled HR@10 (0.42 vs 0.42). Uniform negatives are
#   mostly obscure items that *any* reasonable model ranks below the target, so
#   sampled metrics are high and squeezed together.
# * **Disagreement.** In our runs the popularity prior left full-catalogue
#   metrics unchanged or slightly better (HR@10 0.070 → 0.070 in one run,
#   0.069 → 0.072 in another) while *lowering* sampled HR@10 in both (by about
#   0.01). The full-catalogue differences are tiny, so treat this pair as an illustration,
#   not a finding. The mechanism is clear, though: a popularity prior matters against
#   the popular items that crowd a real top-10, and those are rarely among 100 uniform
#   negatives, so the sampled metric cannot see the benefit, only the cost.
#
# > 💡 **Intuition.** A sampled metric is a race against 100 people pulled off the street instead of against the actual field: almost anyone trained beats them, so every runner looks good and the gaps between runners shrink. *Where it breaks down:* a race against the street can still rank two runners correctly; it just is not *guaranteed* to, and the full field is cheap to run here.
#
# This is a known result, not an artefact of our data: Krichene & Rendle
# (KDD 2020, "On Sampled Metrics for Item Recommendation") show that sampled
# HR/NDCG are not guaranteed to preserve the ordering of models under the
# full-catalogue metric. Practical rule: report full-catalogue metrics when
# you can afford them (scoring 4,000 or even 1 million items with a matrix multiply
# is cheap offline); if you must sample, say so, use the same negatives for every
# model, and do not compare your sampled numbers with someone else's.

# %% [markdown]
# ## 10.7 · Does the *order* matter here?
#
# A sequence model earns its complexity only if the order of events carries
# signal. Test it directly: shuffle each test user's history (keeping the same
# items and the padding in place) and evaluate the same trained model.

# %%
rng = np.random.default_rng(1)
H_shuf = H.copy()
for r in range(len(H_shuf)):
    real = np.flatnonzero(H_shuf[r])
    H_shuf[r, real] = rng.permutation(H_shuf[r, real])
shuffled = sr.hr_ndcg(sr.rank_of_target(sasrec_scores(model, H_shuf), T))
print(f"SASRec full HR@10: original order {results['SASRec']['full HR@10']:.4f} | shuffled history {shuffled['HR@10']:.4f}")

# %% [markdown]
# **Expected:** shuffling costs very little (in our run HR@10 went from 0.0700 to
# 0.0687), and almost all of the gain over popularity survives. That is
# honest information about **ShopStream**: its users have fixed tastes, so a
# history is mostly a bag of evidence about *who the user is*. The model is
# effectively a user-taste model built from recent clicks, which is useful in itself:
# it needs no per-user parameters and adapts the moment a click arrives. On real
# data with sessions, intents that change within minutes, and sequential
# dependencies (phone → phone case), order matters much more. Run this test before
# you claim a sequence model "understands sequences".

# %% [markdown]
# > **Exercise 4 (written).** Your team's new sequence model improves *sampled* HR@10
# > (100 uniform negatives) from 0.58 to 0.63 over the production model, and the
# > author proposes an A/B test. Write the three checks you would ask for before
# > approving the test, and for each, what result would change your mind.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🧮 Key equations
#
# **1 · Scaled dot-product attention with a mask.** In words: each query position takes a weighted average of the values, with weights from a softmax over query–key similarities; blocked pairs get $-\infty$, so weight 0.
#
# $$
# \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}} + M\right) V, \qquad M_{ij} = \begin{cases} 0 & \text{allowed} \\ -\infty & \text{blocked } (j > i \text{ or key } j \text{ is padding}) \end{cases}
# $$
#
# * $Q, K, V$: $n \times d$ query, key and value matrices; $d$: head dimension. *Example:* query 2 with scores $(1, 2, 5)$ over keys 0–2 and key 3 blocked: weights $\operatorname{softmax}(1, 2, 5) = (0.02, 0.05, 0.94)$, 0 on key 3. A row with *every* entry blocked is $0/0$ = NaN.
#
# **2 · Next-item loss over the catalogue.** In words: at every non-padding position $t$, softmax cross-entropy of the next item over all $N$ items, scored by the dot product with the (tied) item embeddings.
#
# $$
# L = -\frac{1}{\lvert \mathcal{T} \rvert} \sum_{t \in \mathcal{T}} \log \frac{\exp(\mathbf{h}_t^\top \mathbf{e}_{y_{t+1}})}{\sum_{j=1}^{N} \exp(\mathbf{h}_t^\top \mathbf{e}_j)}
# $$
#
# * $\mathbf{h}_t$: the hidden state at position $t$; $\mathbf{e}_j$: item $j$'s embedding; $\mathcal{T}$: non-padding positions. *Example:* uniform guessing over $N = 4{,}000$ items: $\log 4000 = 8.29$, the ceiling the training loss starts under.
#
# **3 · HR@10 and NDCG@10 with one target.** In words: a hit if the target ranks in the top 10; NDCG gives a hit at rank $r$ (0 = top) the discount $1/\log_2(r + 2)$.
#
# $$
# \operatorname{HR@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \mathbf{1}[r_u < 10], \qquad \operatorname{NDCG@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \frac{\mathbf{1}[r_u < 10]}{\log_2(r_u + 2)}
# $$
#
# * *Example:* three users with target ranks 0, 3 and 12: HR@10 = 2/3; NDCG@10 = $(1 + 0.431 + 0)/3 = 0.477$.
#
# The cell below checks equation 1 against PyTorch's built-in attention, and that the causal mask really hides the future.

# %%
# Hand-written causal attention vs torch's scaled_dot_product_attention
import numpy as np
import torch
import torch.nn.functional as tnf

g = torch.Generator().manual_seed(0)
n, d = 5, 8
q, k, v = (torch.randn(1, n, d, generator=g) for _ in range(3))
blocked = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)  # True = blocked (j > i)
weights = torch.softmax((q @ k.transpose(-2, -1) / d**0.5).masked_fill(blocked, float("-inf")), dim=-1)
out_hand = weights @ v
out_lib = tnf.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"max |hand - library| = {(out_hand - out_lib).abs().max().item():.1e}")

k2, v2 = k.clone(), v.clone()
k2[0, -1] += 10.0  # change only the LAST position's key and value
v2[0, -1] += 10.0
out_changed = tnf.scaled_dot_product_attention(q, k2, v2, is_causal=True)
print("earlier positions unchanged:", torch.allclose(out_lib[0, :-1], out_changed[0, :-1]))
ranks = np.array([0, 3, 12])
print(f"HR@10 {np.mean(ranks < 10):.3f} | NDCG@10 {np.where(ranks < 10, 1 / np.log2(ranks + 2), 0).mean():.3f}"
      f" | log 4000 = {np.log(4000):.2f}")

# %% [markdown]
# **Expected:** a maximum difference around 1e-7 (float32 rounding), `earlier positions unchanged: True`, and HR@10 0.667, NDCG@10 0.477, log 4000 = 8.29.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `shoprank` | setup cell not run | run the first code cell |
# | Loss is `nan` from the first step | a fully blocked attention row (left padding + key-padding mask) | unblock the diagonal, as `sr.attention_mask` does |
# | Training loss far below the baselines' level, test metrics poor | the model sees the future: causal mask missing or transposed | run test 1 in 10.4 |
# | Initial loss much larger than `log(n_items)` | output logits too large at init (e.g. N(0,1) embeddings with tied weights) | smaller embedding init, as in `SASRec.__init__` |
# | Epochs take several minutes | full softmax over padding positions, or a large `max_len`/`dim` | compute logits only where `target > 0`; keep `dim`≈48, `max_len`≈30 on CPU |
# | Item ids off by one in predictions | forgot the +1 shift (0 = padding) | `seq_id - 1` is the ShopStream `item_id` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why does a next-item transformer need a causal mask during training but not a separate model per position? What would happen if you removed it?
# 2. Left vs right padding for a causal model: what changes in the masks, the positional encoding and the prediction position?
# 3. Explain leave-one-out evaluation and how it can leak future information. How did this notebook avoid that?
# 4. Why can HR@10 with 100 sampled negatives rank two models differently from the full-catalogue metric? When is sampling acceptable?
# 5. Your catalogue grows from 4,000 to 10 million items. What changes in training (the softmax) and in serving (retrieval)?
# 6. How would you show that your sequence model uses order, rather than just the set of past items?
#
# ➡️ **Next:** [`11_ab_testing_fundamentals.ipynb`](11_ab_testing_fundamentals.ipynb) — offline metrics only nominate a model; an experiment decides.
