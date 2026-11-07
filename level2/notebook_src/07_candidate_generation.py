# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 07 · Candidate generation
#
# **Day 4 · ~90 minutes · CPU only**
#
# Notebook 06's ranker re-ordered ten items that someone else had already
# chosen. In a real system, *choosing* those items, a few hundred out of
# millions in a few milliseconds, is a separate stage with its own models,
# metrics and failure modes. If the right item is not retrieved, no ranker
# can show it.
#
# By the end you will be able to:
#
# 1. explain the two-stage (retrieve → rank) architecture and why retrieval is judged by recall@K, not NDCG;
# 2. build four classic candidate generators: global popularity, per-category popularity, item-to-item co-visitation and implicit-feedback matrix factorisation (ALS);
# 3. evaluate them on next-week clicks *and* on the true most-relevant items, with catalogue coverage;
# 4. show how each behaves for cold-start users and items, and what fills the gap;
# 5. blend generators with quotas and measure whether the blend helps.
#
# **You will produce:** `artifacts/level2/retrieval_results.csv` and the ALS factors `artifacts/level2/als_factors.npz` (reused as a baseline in notebook 08).

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
# --- Catch up: clicks from the ShopStream logs --------------------------------
import time

import numpy as np
import pandas as pd

from shoprank import retrieval as rt
from shoprank import store
from shoprank.grading import nb07 as grade
from shoprank.metrics import catalog_coverage

t0 = time.time()
world, users, items, imps = store.load("small", columns=["request_id", "day", "user_id", "item_id", "position", "clicked"])
clicks = imps.loc[imps["clicked"], ["request_id", "day", "user_id", "item_id"]].reset_index(drop=True)
train_clicks = clicks[clicks["day"].isin(store.TRAIN_DAYS)]
valid_clicks = clicks[clicks["day"].isin(store.VALID_DAYS)]
N_USERS, N_ITEMS = len(users), len(items)
print(f"{len(clicks):,} clicks | train {len(train_clicks):,} | valid {len(valid_clicks):,} | {time.time() - t0:.0f}s")

# %% [markdown]
# **Expected:** about 335,000 clicks: about 244,000 in the six training weeks
# and 44,000 in the validation week. Retrieval models learn from
# interactions only (here, clicks). Impressions without clicks are mostly
# ignored at this stage.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Candidate generation** | picking a few hundred items out of the catalogue, cheaply | retrieve 100 of 4,000 |
# | **Recall@K** | share of a user's relevant items found among K candidates | 0.34 for ALS at K = 100 |
# | **Coverage** | share of the catalogue that appears in anyone's candidates | 0.025 for global popularity |
# | **Exposure bias** | clicks exist only for items the old system chose to show | next-week clicks vs true top-20 |
# | **Co-visitation** | how often two items are clicked in the same session | "people who clicked X also clicked Y" |
# | **Session** | a group of one user's events close in time | one user's clicks on one day |
# | **Matrix factorisation** | approximating the user × item matrix by user vectors times item vectors | `rt.implicit_als` |
# | **Confidence weight** | how much a cell of the matrix counts in implicit ALS | c_ui = 1 + α·r_ui |
# | **ALS** | alternating least squares: fix items, solve users; fix users, solve items | a few seconds here |
# | **Cold start** | a user or item with no interaction history | about 270 cold users, 170 new items |
# | **Blending with quotas** | merging generators with a fixed number of slots each | 70 ALS + 30 category popularity |
# | **Popularity bias** | a generator's tendency to return already-popular items | the mean log-popularity column |

# %% [markdown]
# ## 7.1 · Why two stages
#
# ```
#   catalogue (10^6-10^9 items)
#        │  retrieval: several cheap generators, ~1-10 ms, recall-oriented
#        ▼
#   candidates (10^2-10^3)           ← this notebook and 08
#        │  ranking: rich features, GBDT / deep model, ~10-50 ms, precision-oriented
#        ▼
#   top 10-50 shown                  ← notebook 06
# ```
#
# The ranker's features (notebook 06) cost microseconds per item; times a
# million items, per request, is far beyond any latency budget. So the first
# stage uses models whose scores can be **precomputed or indexed**
# (popularity lists, item→item tables, dot products of embeddings) and is
# judged only by whether the good items survive: **recall@K**, the share of
# a user's relevant items found in the K candidates. Order inside the K does
# not matter much; the ranker re-orders anyway.
#
# Our catalogue has 4,000 items, so "retrieve 100" is a 40× cut. The ratios
# and the lessons carry over; the latency problem is notebook 08's.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-retrieval-cascade.svg" alt="Catalogue, retrieval, candidates, ranking, rules, shown: the two-stage cascade" width="100%">
#
# *What to notice:* each stage cuts the item count by one or more orders of magnitude, so each can afford a more expensive model per item. Retrieval is judged by recall@K because the ranker can only reorder what it is given.
#
# > 💡 **Intuition.** Hiring: a recruiter skims 10,000 CVs for keywords and keeps 100 (retrieval, judged by "did we keep the good people?"), then a panel interviews those 100 carefully and ranks them (ranking, judged by "is the best person on top?"). *Where it breaks down:* recruiters read every CV; real retrieval never scores every item exactly; it relies on precomputed lists and approximate search.

# %% [markdown]
# ## 7.2 · What counts as a hit
#
# Two truths, for each user active in the validation week:
#
# * **next-week clicks**: what you can measure in production. Limited to
#   items the *logging policy* chose to show (exposure bias): a generator that
#   finds great items the old system never showed gets no credit.
# * **true top-20**: the 20 eligible items with the highest true click
#   probability for that user, available only in the simulation. It tells us
#   whether a generator finds *relevant* items, not just *previously shown* ones.

# %%
truth_clicks = rt.truth_sets(valid_clicks)
eval_users = np.array(sorted(truth_clicks))
eligible = items["created_day"].to_numpy() <= max(store.VALID_DAYS)
truth_true = rt.true_top_items(world, eval_users, 20, eligible)
n_truth = np.array([len(v) for v in truth_clicks.values()])
print(f"evaluation users: {len(eval_users):,} | valid clicks per user: median {np.median(n_truth):.0f}, "
      f"mean {n_truth.mean():.1f}")
overlap = np.mean([len(truth_clicks[u] & truth_true[u]) / len(truth_clicks[u]) for u in eval_users])
print(f"share of next-week clicks that are in the user's true top-20: {overlap:.1%}")

# %% [markdown]
# **Expected:** about 7,700 users with at least one validation click, median
# 4 clicks each (mean about 5.6), and only about 12% of next-week clicks in
# the user's true top-20. Clicks are a noisy, exposure-limited sample of relevance: the two
# truths will rank generators similarly but not identically.
#
# > **Exercise 2.** *(Do this one before Exercise 1: everything below uses it.)* Write
# > `recall_at_k_batch(candidates, truth, k)`: `candidates` maps user →
# > ranked list of item ids, `truth` maps user → set of relevant items.
# > Return the **mean over users with a non-empty truth set** of
# > |first k candidates ∩ truth| / |truth|. A user missing from `candidates`
# > scores 0 (the generator returned nothing for them); users only in
# > `candidates` are ignored.

# %%
def recall_at_k_batch(candidates: dict, truth: dict, k: int) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(recall_at_k_batch)

# %% [markdown]
# The library version is `rt.recall_at_k_batch`. One more helper: an
# evaluation row for any candidate dict.

# %%
log_pop = np.log1p(rt.popularity_scores(train_clicks, N_ITEMS))
results = {}


def evaluate(name: str, cands: dict, users=eval_users) -> dict:
    lists = [cands.get(int(u), [])[:100] for u in users]
    row = {f"recall@{k} clicks": rt.recall_at_k_batch(cands, {int(u): truth_clicks[int(u)] for u in users}, k)
           for k in (50, 100)}
    row.update({f"recall@{k} true top-20": rt.recall_at_k_batch(cands, {int(u): truth_true[int(u)] for u in users}, k)
                for k in (50, 100)})
    row["coverage@100"] = catalog_coverage(lists, N_ITEMS)
    row["mean log-popularity@100"] = float(np.mean([log_pop[lst].mean() for lst in lists if len(lst)]))
    results[name] = row
    return row

# %% [markdown]
# ## 7.3 · Global popularity
#
# The baseline every system has: the most-clicked items of the training
# period, the same list for everyone. It is also the fallback when every
# other generator has nothing to say.

# %%
pop = rt.popularity_scores(train_clicks, N_ITEMS)
global_top = rt.top_items(pop, 200)
cands_pop = {int(u): global_top.tolist() for u in eval_users}
evaluate("global popularity", cands_pop)

# %% [markdown]
# **Expected:** recall on clicks about 0.12 at K=50 and 0.20 at K=100;
# against the true top-20 about 0.19 and 0.29; coverage 0.025 (the same 100
# items for everyone). Not bad for a list with zero personalisation: the
# logging policy shows popular items a lot, and popularity in ShopStream is
# partly driven by item quality, which every user values.

# %% [markdown]
# ## 7.4 · Per-category popularity
#
# Personalise just a little: split the K slots across the categories the
# user clicked in training, in proportion to their clicks, and fill each
# share with that category's most popular items (`rt.category_popularity`).
# Users without history get the global list.

# %%
user_hist = rt.click_matrix(train_clicks, N_USERS, N_ITEMS)
item_cat = items["category"].to_numpy()
cands_catpop = rt.category_popularity(user_hist, pop, item_cat, eval_users, 200)
evaluate("category popularity", cands_catpop)

# %% [markdown]
# **Expected:** recall on clicks about 0.16 / 0.33 at 50 / 100, and a big
# jump on the true top-20 (about 0.28 / 0.57), with coverage about 0.46.
# Users in ShopStream have strong category preferences, so knowing "which
# categories" already captures much of relevance. In a real catalogue,
# replace category with any coarse attribute (brand, topic, price band).

# %% [markdown]
# ## 7.5 · Item-to-item co-visitation
#
# "People who clicked X also clicked Y": count, for each pair of items, the
# number of **sessions** in which both were clicked. Here a session is one
# user's clicks on one day (real systems use time windows, e.g. 30 minutes or
# 24 hours, and often weight by time gap and by the type of event). At
# serving time, the user's recent clicks are looked up in this table and
# their neighbours merged. It is the generator behind most "related items"
# carousels: cheap, fresh (recomputed hourly), and explainable.
#
# > 💡 **Intuition.** Co-visitation is reading supermarket receipts: if crisps and salsa appear on the same receipt again and again, show salsa to whoever just picked up crisps. *Where it breaks down:* milk is on almost every receipt, so it "co-occurs" with everything; that is why production tables down-weight very popular items.
#
# > **Exercise 1.** Write `covisitation(pairs_df, top_n)`. `pairs_df` has one
# > row per click with columns `session_id` and `item_id`. Return a dict
# > item → list of its `top_n` co-visited items: two items co-visit **once
# > per session** in which both appear (repeats in a session count once), an
# > item is never its own neighbour, and neighbours are sorted by count
# > descending, ties broken by the smaller item id. Items with no neighbour
# > may be left out.

# %%
def covisitation(pairs_df: pd.DataFrame, top_n: int) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(covisitation)

# %% [markdown]
# Now the library version on the real sessions, and a sanity check you
# should always run on a co-visitation table: do neighbours make sense?
# (Here: do they share the item's category more often than chance?)

# %%
t0 = time.time()
sessions = train_clicks.assign(session_id=train_clicks["user_id"].astype(np.int64) * 100 + train_clicks["day"])
neighbours = rt.covisitation(sessions[["session_id", "item_id"]], top_n=50)
n_sessions = sessions["session_id"].nunique()
same_cat = np.mean([np.mean(item_cat[nb] == item_cat[i]) for i, nb in neighbours.items()])
chance = float(((np.bincount(item_cat) / len(item_cat)) ** 2).sum())
print(f"{n_sessions:,} sessions | {len(neighbours):,} items with neighbours | {time.time() - t0:.1f}s")
print(f"neighbours in the same category: {same_cat:.1%} (chance: {chance:.1%})")

recent = (train_clicks.sort_values(["day", "request_id"]).groupby("user_id")["item_id"]
          .apply(lambda s: s.to_numpy()[::-1][:20].tolist()))
recent = {int(u): recent.get(u, []) for u in eval_users}
cands_covis = rt.covisitation_candidates(neighbours, recent, 200, fallback=global_top)
evaluate("co-visitation", cands_covis)

# %% [markdown]
# **Expected:** about 102,000 sessions and about 3,700 items with
# neighbours, built in well under a second; about a third of the neighbours
# (about 34%) share the item's category, against about 7% by chance. Recall on clicks
# about 0.21 / 0.30, on the true top-20 about 0.36 / 0.50, and the highest
# coverage so far (about 0.75): co-visitation reaches deep into the catalogue.
#
# The pair table grows with the *square* of session length. Production jobs
# cap session length, drop very popular items from the pair generation (or
# down-weight them, e.g. by 1/log(popularity)), and keep only the top-N
# neighbours per item.

# %% [markdown]
# ## 7.6 · Matrix factorisation (implicit ALS)
#
# Learn a vector x_u per user and y_i per item so that x_u · y_i is high for
# items the user clicked. With implicit feedback there are no negatives,
# only "not clicked" (which may mean "never seen"). Hu, Koren & Volinsky
# (2008) treat every cell as a preference p_ui ∈ {0, 1} with confidence
# c_ui = 1 + α·r_ui and minimise
#
# $$\sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\left(\sum_u \lVert x_u\rVert^2 + \sum_i \lVert y_i\rVert^2\right)$$
#
# Fix Y and each x_u has a closed form (a small linear solve); then fix X.
# The trick that makes it fast: Yᵀ C_u Y = YᵀY + Yᵀ(C_u − I)Y, and C_u − I is
# non-zero only on the user's clicked items. Read `rt.implicit_als`: about
# 25 lines of NumPy. (BPR, a pairwise alternative trained by SGD, is the
# other classic; the two-tower model of notebook 08 generalises both.)

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-matrix-factorization.svg" alt="Matrix factorisation: the click matrix approximated by user vectors times item vectors" width="100%">
#
# *What to notice:* the blank cells in R are not zeros you know; they are mostly items the user never saw. Implicit ALS still fits them (with low confidence), which is what makes the problem a cheap weighted least squares.
#
# > 💡 **Intuition.** ALS is fitting a lock and key by alternating: hold the key still and file the lock to fit it, then hold the lock still and file the key. Each step is easy (a closed-form least-squares solve); together they converge. *Where it breaks down:* a lock and key converge to one fit; ALS converges to *a* good solution, and many rotations of the vectors fit equally well, so individual dimensions mean nothing on their own.

# %%
t0 = time.time()
X_als, Y_als = rt.implicit_als(user_hist, factors=32, reg=0.1, alpha=20.0, iterations=8)
print(f"ALS: {X_als.shape} users x {Y_als.shape} items in {time.time() - t0:.1f}s")
cands_als = rt.score_topk(X_als, Y_als, eval_users, 200)
evaluate("ALS (32 factors)", cands_als)

# %% [markdown]
# **Expected:** a few seconds to train, then the best single generator on
# both truths: recall on clicks about 0.23 / 0.34, on the true top-20 about
# 0.42 / 0.59, coverage about 0.57.
#
# The comparison so far:

# %%
pd.DataFrame(results).T.round(3)

# %% [markdown]
# **Expected:** ALS > co-visitation > category popularity > global
# popularity on clicks at K=50; at K=100 category popularity overtakes
# co-visitation on clicks (about 0.33 vs 0.30) and on the true top-20 (about
# 0.57 vs 0.50). The mean log-popularity column shows each generator's popularity
# bias: global popularity highest by construction, co-visitation lowest.
# The two truths mostly agree on the order, but not on the size of the gaps,
# which is why "recall on logged clicks" alone can mislead a retrieval
# comparison.

# %% [markdown]
# ## 7.7 · Cold start
#
# Two kinds of newcomers:
#
# * **cold users**: active in the validation week, no training clicks;
# * **cold items**: launched during the validation week, so no generator
#   trained on the past can know them.

# %%
cold_users = np.array([u for u in eval_users if user_hist[u].nnz == 0])
warm_users = np.array([u for u in eval_users if user_hist[u].nnz > 0])
rows = {}
for name, c in [("global popularity", cands_pop), ("category popularity", cands_catpop),
                ("co-visitation", cands_covis), ("ALS (32 factors)", cands_als)]:
    rows[name] = {f"recall@100 clicks, cold users (n={len(cold_users)})":
                  rt.recall_at_k_batch(c, {int(u): truth_clicks[int(u)] for u in cold_users}, 100),
                  f"warm users (n={len(warm_users)})":
                  rt.recall_at_k_batch(c, {int(u): truth_clicks[int(u)] for u in warm_users}, 100)}
display(pd.DataFrame(rows).T.round(3))

new_items = set(np.flatnonzero(items["created_day"].to_numpy() >= min(store.VALID_DAYS)).tolist())
share_new = valid_clicks["item_id"].isin(new_items).mean()
in_cands = {name: float(np.mean([len(set(c[int(u)][:100]) & new_items) for u in eval_users]))
            for name, c in [("popularity", cands_pop), ("category pop.", cands_catpop),
                            ("co-visitation", cands_covis), ("ALS", cands_als)]}
print(f"items launched in the validation week: {len(new_items)} | share of validation clicks on them: {share_new:.1%}")
print("new items per 100 candidates:", {k: round(v, 2) for k, v in in_cands.items()})

# %% [markdown]
# **Expected:** about 270 cold users (about 3.5% of evaluation users). For
# them, ALS recall is about **0.02** (an all-zero user vector scores every
# item 0, so the "top" items are arbitrary), while co-visitation and
# category popularity fall back to global popularity (about 0.18). About 170
# items launched in the validation week and took about 1.2% of its clicks.
# Popularity and co-visitation retrieve **none** of them; category popularity
# and ALS retrieve a fraction of one per list, and only by accident (a
# never-clicked item scores 0 and can fill the tail of a list behind items
# with negative scores). All four learn from past clicks only.
#
# Standard fixes, which you will see in every design interview:
#
# * cold users: popularity by context (country, device, time), onboarding
#   questions, then switch to personal generators after a few clicks;
#   in the two-tower model (notebook 08) features other than the id
#   (device, country) give a non-empty user vector;
# * cold items: a **content-based** generator (category, text or image
#   embeddings: notebook 08's item tower uses category and price, so a new
#   item gets a usable vector from day one) and a small **exploration
#   quota** of new items in every candidate set. Without it, a new item never
#   gets the clicks it would need to become retrievable: the rich-get-richer
#   loop.
#
# A minimal content-based generator: the newest items in the user's top
# categories, with a quota of 10 in the blend below.

# %%
created = items["created_day"].to_numpy()
cat_rank = np.asarray((user_hist @ np.eye(item_cat.max() + 1)[item_cat]))  # user x category click counts
newest_by_cat = {c: np.flatnonzero(item_cat == c)[np.argsort(-created[item_cat == c], kind="stable")]
                 for c in range(item_cat.max() + 1)}
cands_new = {}
for u in eval_users:
    top_cats = np.argsort(-cat_rank[u], kind="stable")[:3]
    cands_new[int(u)] = [int(i) for c in top_cats for i in newest_by_cat[c][:10]]
print("new items per list:", round(np.mean([len(set(v) & new_items) for v in cands_new.values()]), 2))

# %% [markdown]
# **Expected:** about 22 of the 30 items per list are new arrivals. (We let
# this generator see the catalogue as of the end of the validation week, as
# an index refreshed daily would; that is a simplification, but item
# metadata, unlike clicks, is available the moment an item is listed.)

# %% [markdown]
# ## 7.8 · Blending generators
#
# Production systems run several generators and merge them, typically with
# **quotas** (e.g. 50 from the main model, 30 from co-visitation, ...),
# de-duplicating as they go (`rt.blend`). The ranker then sees one list and
# features such as "which generator(s) proposed this item".
#
# Does blending help at a fixed total of 100?

# %%
gens = [cands_als, cands_covis, cands_catpop, cands_pop, cands_new]
names = ["ALS", "co-vis", "cat-pop", "pop", "new"]
blend_rows = {}
for quotas in [(100, 0, 0, 0, 0), (60, 25, 15, 0, 0), (50, 30, 20, 0, 0), (40, 20, 30, 0, 10), (70, 0, 30, 0, 0)]:
    b = rt.blend(gens, list(quotas), eval_users)
    label = " + ".join(f"{q} {n}" for q, n in zip(quotas, names) if q)
    blend_rows[label] = {"recall@100 clicks": rt.recall_at_k_batch(b, truth_clicks, 100),
                         "recall@100 true top-20": rt.recall_at_k_batch(b, truth_true, 100),
                         "new items per list": np.mean([len(set(v) & new_items) for v in b.values()]),
                         "coverage@100": catalog_coverage(b.values(), N_ITEMS)}
blend_table = pd.DataFrame(blend_rows).T
blend_table.round(3)

# %% [markdown]
# **Expected:** ALS alone gets about 0.335 on clicks and 0.59 on the true
# top-20, and **no blend beats it**: replacing ALS slots with co-visitation
# and category popularity costs up to about 0.006 recall on clicks and 0.03
# on the true top-20, in exchange for coverage (about 0.57 → 0.65). "70 ALS +
# 30 category popularity" ties ALS on clicks. The blend with 10 new-item
# slots puts about 7 new items in every list for about 0.01 of recall on
# clicks, a cost you pay deliberately: without it new items never collect
# the clicks that would make them retrievable, and offline recall (computed
# on clicks the *old* system produced) cannot see that benefit.
#
# Blending pays when generators are *complementary*: different signals
# (social, content, location), different users (cold vs warm) or different
# goals (freshness, diversity). Here ALS already absorbs most of what the
# other click-based generators know. Quotas are hyperparameters: tune them on
# validation recall and then on the *ranked* outcome, since the ranker decides
# what users see. Larger systems learn the merge with a light "pre-ranker"
# instead of fixed quotas.

# %%
pd.DataFrame(results).T.to_csv(ART / "level2" / "retrieval_results.csv")
np.savez_compressed(ART / "level2" / "als_factors.npz", users=X_als, items=Y_als)
print("saved:", sorted(p.name for p in (ART / "level2").glob("*") if p.suffix in {".csv", ".npz"}))

# %% [markdown]
# > **Exercise 3.** You are designing retrieval for a marketplace with 50
# > million listings, 20% of them new each week, and a 15 ms budget for the
# > whole retrieval stage. Which generators do you run, how many candidates
# > from each, how do you evaluate them offline, and what do you monitor
# > online? Write your answer in the cell below (about 10 bullet points).

# %% [markdown]
# *Your answer:*

# %% [markdown]
# ## 🧮 Key equations
#
# **1 · Recall@K.** In words: of the items that were relevant for a user, what share did the K candidates contain?
#
# $$
# \operatorname{Recall@}K(u) = \frac{\lvert \text{top}_K(u) \cap T_u \rvert}{\lvert T_u \rvert}
# $$
#
# * $T_u$: user $u$'s relevant items (next-week clicks, or the true top-20); averaged over users with $T_u \neq \emptyset$. *Example:* 5 next-week clicks, 2 of them among the 100 candidates → 0.4. Order inside the K does not matter.
#
# **2 · Catalogue coverage.** In words: the share of the catalogue that appears in at least one user's list.
#
# $$
# \operatorname{Coverage} = \frac{\left\lvert \bigcup_{u} \text{top}_K(u) \right\rvert}{\lvert \mathcal{I} \rvert}
# $$
#
# * *Example:* global popularity gives every user the same 100 items: $100 / 4{,}000 = 0.025$.
#
# **3 · Co-visitation counts.** In words: count the sessions in which both items appear (once per session).
#
# $$
# C(a, b) = \sum_{s} \mathbf{1}[a \in s]\, \mathbf{1}[b \in s], \quad a \neq b
# $$
#
# * *Example:* sessions {1, 2, 3}, {2, 3}, {3, 4} → $C(2, 3) = 2$, $C(1, 3) = C(3, 4) = 1$. A session of length $L$ produces $L(L-1)/2$ pairs: cap $L$.
#
# **4 · Implicit ALS objective and its closed-form step.** In words: weighted least squares on every cell, with weight $c_{ui}$; with item vectors fixed, each user vector is a small linear solve.
#
# $$
# \min_{X, Y} \sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2\Big), \qquad x_u = (Y^\top C_u Y + \lambda I)^{-1} Y^\top C_u\, p_u
# $$
#
# * $p_{ui} = 1$ if $u$ clicked $i$, else 0; $c_{ui} = 1 + \alpha r_{ui}$ with $r_{ui}$ the click count; $C_u$: the diagonal matrix of $u$'s confidences. *Example:* $\alpha = 20$: a cell with 2 clicks has confidence 41, an unclicked cell 1.
#
# **5 · BPR, the pairwise alternative.** In words: for a clicked item $i$ and a sampled unclicked item $j$, make the user score $i$ above $j$.
#
# $$
# L_\text{BPR} = -\sum_{(u, i, j)} \log \sigma\!\left(x_u^\top y_i - x_u^\top y_j\right) + \lambda \lVert \Theta \rVert^2
# $$
#
# * $\sigma$: the logistic function; $\Theta$: all parameters. *Example:* scores 2.0 (clicked) and 0.5 (sampled): $-\log \sigma(1.5) = 0.20$; reversed (0.5 vs 2.0): $-\log \sigma(-1.5) = 1.70$. It is RankNet (notebook 06) applied to (clicked, unclicked) pairs.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Kernel dies computing the true top items | a users × items × dims array (7,700 × 4,000 × 16 floats) | use `rt.true_top_items`, which works in batches |
# | Co-visitation is slow or runs out of memory | long sessions: pairs grow with the square of session length | cap clicks per session; drop or down-weight the most popular items; keep top-N per item |
# | ALS recall near zero for everyone | user/item ids misaligned with the matrix, or `alpha` far too small/large | check `click_matrix` shapes; try `alpha` in 5–40; inspect a user's top items vs their history |
# | Recall is NaN | empty truth sets (users with no valid clicks) | evaluate only users with non-empty truth; report how many |
# | A generator looks great on clicks but poor on the true top-20 | it follows what the old policy showed (exposure bias) | compare on both truths; on real data use random-traffic or interleaving evaluation |
# | New items never appear in candidates | every generator is trained on past clicks | add a content-based generator and an exploration quota |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why is retrieval evaluated with recall@K and ranking with NDCG? What would go wrong if you tuned retrieval on NDCG of the final list?
# 2. Explain implicit-feedback ALS: what are p_ui and c_ui, why is it "alternating", and what makes one step cheap?
# 3. How do you build an item-to-item co-visitation table for 100 million sessions a day? Which steps are map, shuffle and reduce, and where is the skew?
# 4. Your new retrieval model has +8% recall@100 on next-week clicks, but the A/B test is flat. Give three explanations.
# 5. Design the cold-start path for a brand-new item and for a brand-new user. When does each "graduate" to the main generators?
# 6. You blend four generators with fixed quotas. How do you choose the quotas, and what would you replace them with at larger scale?
#
# ➡️ **Next:** [`08_two_tower_retrieval_and_ann.ipynb`](08_two_tower_retrieval_and_ann.ipynb) — learn user and item embeddings with a two-tower model, correct its popularity bias, and serve it with approximate nearest-neighbour search.
