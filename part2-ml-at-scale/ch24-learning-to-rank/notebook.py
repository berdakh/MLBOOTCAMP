# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 24 · Learning to rank
#
# **Part II · Chapter 24 · about 90 minutes · CPU only**
#
# A ranker is not a classifier whose outputs are sorted: only the *order
# within a request* matters, the data come in groups, and the metric (NDCG)
# weights the top far more than the bottom. This lab builds the standard
# industrial approach, point-in-time features with LightGBM LambdaMART, and
# evaluates it as Chapter 23 recommended: on clicks, and against the truth.
#
# After this lab you should be able to:
#
# 1. frame ranking as learning over request groups, and say what a "query" is in a feed;
# 2. build point-in-time (user, item) features from logs (prior days only) and test that they do not leak;
# 3. explain pointwise, pairwise (RankNet) and listwise (LambdaRank/LambdaMART) objectives and what each optimises;
# 4. train `LGBMRanker` with correct group sizes on a request-grouped time split;
# 5. evaluate NDCG@10 and MRR on clicks *and* on true relevance, read feature importance, and decide what to do with position.
#
# **Outputs of this lab:** `artifacts/level2/ltr_lambdamart.txt` (the trained LightGBM model) and `artifacts/level2/ltr_results.csv` (the comparison table).

# %% [markdown]
# Read first: [Chapter 24 text](README.md) — ranking metrics, point-in-time features, pointwise/pairwise/listwise objectives, LightGBM groups and the treatment of position, explained in full.

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
# --- Catch up: logs, propensities (notebook 05) and the hidden truth --------
import time

import numpy as np
import pandas as pd

from shoprank import bias, ltr, store
from grading import ch24 as grade

t0 = time.time()
world, users, items, imps = store.load("small")
imps = imps[imps["day"] < max(store.VALID_DAYS) + 1].reset_index(drop=True)  # never load the test week here
K = 10
THETA_TRUE = world.examination_prob(np.arange(K))
eta_hat, theta_hat = bias.power_law_propensities(imps[imps["is_random"] & imps["day"].isin(store.TRAIN_DAYS)])
imps["true_p"] = world.true_click_prob(imps["user_id"].to_numpy(), imps["item_id"].to_numpy())  # grading only
print(f"{len(imps):,} impressions (days 0-48) | fitted eta {eta_hat:.3f} | {time.time() - t0:.0f}s")

# %% [markdown]
# **Expected:** about 2.11 million impressions and a fitted η of about 0.88
# (Chapter 23 explains both). `true_p` is used only to grade rankings.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Group (query)** | the set of items ranked together; scores are compared only inside it | one request of 10 items |
# | **Pointwise loss** | scores each item on its own, like a classifier | log loss on `clicked` |
# | **Pairwise loss** | penalises each mis-ordered pair inside a group | RankNet |
# | **Listwise loss** | optimises a list metric (smoothly) | LambdaRank / LambdaMART |
# | **RankNet** | logistic loss on the score difference of a pair | log(1 + e^−(s_i − s_j)) |
# | **LambdaMART** | RankNet gradients weighted by |ΔNDCG|, fitted with boosted trees | `LGBMRanker(objective="lambdarank")` |
# | **NDCG@k** | discounted gain of the top k, divided by the best possible | 0.81 for the logged order (truth) |
# | **MRR** | mean of 1 / rank of the first relevant item | 0.73 for the logged order (clicks) |
# | **Pairwise accuracy** | share of differently-labelled pairs ordered correctly | 0.53 logged vs 0.85 learned |
# | **Group sizes** | how LightGBM learns which rows form a group: counts of contiguous rows | `[10, 10, 10, …]` |
# | **Gain importance** | total loss reduction from splits on a feature | `u_cat_clicks` leads LambdaMART |

# %% [markdown]
# ## 6.1 · The problem: rank within a request
#
# In search, the group is a **query**: rank documents for "red shoes". In a
# feed, there is no typed query; the group is the **request** (one user,
# one moment, one candidate set). Learning to rank (LTR) learns a score
# s(u, i) such that, *within each group*, more relevant items score higher.
# Scores are never compared across groups, so a model may be badly
# calibrated and still rank perfectly.
#
# > **Intuition.** A ranker is a judge at a swimming gala who only places swimmers *within their heat*. Whether heat 3 was faster than heat 7 does not matter, so a score of 0.9 in one request and 0.2 in another can both mean "first place". *Where it breaks down:* some systems do compare across groups (an ads auction mixes candidates from several models, a threshold decides whether to show anything); then calibration matters again.
#
# Each logged request here has exactly 10 shown items and binary click
# labels. Many requests have no click at all:

# %%
req = imps.groupby("request_id").agg(day=("day", "first"), is_random=("is_random", "first"),
                                     clicks=("clicked", "sum"))
print(req["clicks"].value_counts().sort_index().head(6).to_string())
print(f"requests with no click: {(req['clicks'] == 0).mean():.1%}")

# %% [markdown]
# **Expected:** about 50,000 requests with no click (about 24%), about
# 76,000 with one and 53,000 with two; a long tail after that. A request with no click (or with every item clicked) contributes
# nothing to a ranking loss: there is no pair to order. Pairwise and listwise
# trainers skip them anyway, so dropping them from training saves time
# without losing information. They still matter for *pointwise* calibration,
# and they must stay in any metric that reports "share of requests with a
# click".

# %% [markdown]
# ## 6.2 · Point-in-time features
#
# Features for a (user, item) pair at request time, all computed from logs
# **strictly before the request's day** (a daily batch job; the point-in-time
# rule of Chapter 21):
#
# | Feature | Meaning |
# |---|---|
# | `u_cat_ctr`, `u_cat_imps`, `u_cat_clicks` | the user's history with this item's category: smoothed CTR, log impressions, log clicks |
# | `u_ctr`, `u_imps` | the user's overall click rate and activity |
# | `i_ctr`, `i_imps` | the item's smoothed CTR and log impressions (popularity) |
# | `i_coec` | item clicks over *expected* clicks given the positions it was shown at (debiased CTR, Chapter 23) |
# | `i_age` | days since the item launched |
# | `log_price`, `price_vs_cat` | price, and price relative to the category median |
# | `category`, `segment`, `device` | categorical |
#
# Read `ltr.build_features` in `src/shoprank/ltr.py`: per-(key, day) sums,
# cumulative sum over days, **minus the current day**. That one subtraction
# is the difference between a valid feature and a leak.

# %%
t0 = time.time()
F = ltr.build_features(imps, items, users, theta_hat)
print(f"features: {F.shape} in {time.time() - t0:.1f}s")
F.describe().T[["mean", "min", "max"]].round(3)

# %% [markdown]
# **Expected:** a (2,110,650, 14) frame in a few seconds. Smoothed CTRs
# lie between 0 and 1; `i_age` ranges from 0 to 48.
#
# Two quick leakage tests. A point-in-time feature for the very first day
# must be "no history", and features must not change when all rows from the
# day being scored onward are deleted:

# %%
first_day = F[imps["day"] == 0]
print("day-0 rows with any user/item history:", int((first_day[["u_imps", "i_imps"]] > 0).any(axis=1).sum()))

cut = 30
F_cut = ltr.build_features(imps[imps["day"] <= cut], items, users, theta_hat)
rows_cut = imps.index[imps["day"] == cut]
same = np.allclose(F_cut.loc[rows_cut].to_numpy(float), F.loc[rows_cut].to_numpy(float))
print(f"features of day {cut} unchanged when later days are removed: {same}")

# %% [markdown]
# **Expected:** `0` and `True`. The second test is the useful one in real
# pipelines: recompute features for a past day from a truncated log and
# compare. If anything differs, a feature is peeking at the future. (It
# caught a real bug while this lab was being written: the smoothing prior was
# the click rate of *all* rows passed in, future days included. Now it is a
# fixed constant, `prior_ctr`.)
#
# ### What about position?
#
# Position is *not* in the feature list. At serving time it does not exist:
# position is the **output** of the ranker. Two options for using it anyway:
#
# * leave it out, and correct with weights (Chapter 23, IPS);
# * put it in at **training** time, so the model can attribute clicks at the
#   top to examination rather than to the item, then set it to a **constant
#   (position 0)** at serving. Under the PBM that recovers the relevance
#   ordering *if* the model separates the two effects (with trees, the
#   position effect is not forced to be multiplicative, so it is not
#   guaranteed). We test it in 6.6.

# %% [markdown]
# ## 6.3 · Three ways to learn a ranking
#
# * **Pointwise**: predict each item's label on its own (logistic regression,
#   a GBDT classifier). Simple, calibrated, and blind to the group: it spends
#   capacity on telling a dull request from a lively one, which ranking does
#   not need.
# * **Pairwise** (RankNet, 2005): for every pair (i, j) in a group with
#   y_i > y_j, minimise log(1 + exp(−(s_i − s_j))). It optimises the share
#   of correctly ordered pairs, treating a swap at positions 9/10 like one at 1/2.
# * **Listwise** (LambdaRank → **LambdaMART**): RankNet's pair gradients,
#   each multiplied by |ΔNDCG|, the change in NDCG if i and j were swapped.
#   Mistakes at the top get big gradients, mistakes at the bottom small ones,
#   so the model optimises (a smooth stand-in for) NDCG. LambdaMART is
#   LambdaRank with gradient-boosted trees, the industry workhorse for
#   feature-based ranking (LightGBM `objective="lambdarank"`).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-ltr-objectives.svg" alt="Pointwise versus pairwise (RankNet) versus listwise (LambdaMART), and why LambdaRank weights pairs by the change in NDCG" width="100%">
#
# *What to notice:* the three columns differ in *what is compared* (one item, a pair, a pair weighted by the list metric). The numbers under the rule are the central idea of LambdaRank: the same mistake costs 30× more at the top.
#
# > **Intuition.** Consider grading a ranked shortlist of job candidates: swapping the 1st and 2nd choice changes who receives the offer; swapping 9th and 10th changes nothing anyone acts on. RankNet charges both swaps the same; LambdaRank charges each by how much the *metric* would change. *Where it breaks down:* NDCG's discount is a modelling choice, not a law; if users scroll deep (a long product grid), the bottom matters more than the metric says.
#
# The quantity RankNet optimises is easy to state and is a good diagnostic.
#
# > **Exercise 1.** Write `pairwise_accuracy(scores, labels, groups)`: among
# > all pairs of items **in the same group** with **different labels**, the
# > fraction where the item with the higher label has the higher score. A tie
# > in score counts as half. Pairs across groups never count. Return `nan` if
# > there are no such pairs. Inputs are 1-D arrays of equal length; groups are
# > not necessarily contiguous.

# %%
def pairwise_accuracy(scores, labels, groups) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(pairwise_accuracy)

# %% [markdown]
# ## 6.4 · Groups for LightGBM
#
# `LGBMRanker.fit(X, y, group=sizes)` takes **group sizes**, not group ids:
# `[10, 10, 7, ...]` means "the first 10 rows are one group, the next 10
# another...". So the rows *must* be contiguous by group, in the same order
# as the sizes. If the rows are shuffled (or two frames concatenated) without
# re-sorting, LightGBM silently trains on meaningless groups.
#
# > **Exercise 2.** Write `make_groups(df, group_col)` returning
# > `(sorted_df, sizes)`: the frame sorted by `group_col` (a **stable** sort,
# > so the row order inside each group is kept), and a NumPy array of group
# > sizes in that same order. The input may be unsorted, with groups
# > interleaved.

# %%
def make_groups(df: pd.DataFrame, group_col: str):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(make_groups)

# %% [markdown]
# ## 6.5 · A request-grouped time split, sampled
#
# * **Train:** policy-traffic requests from days 21–41 (the first three weeks
#   only feed the features) that have at least one click. Sample 40,000
#   requests to keep training under a minute on a CPU.
# * **Validate:** *every* request of days 42–48, policy and random.
#
# Splitting by time keeps whole requests together (a request lives on one
# day) and mimics production: train on the past, serve the future. A random
# row-level split would put items of the same request on both sides.

# %%
rng = np.random.default_rng(0)
tr_req = req[req["day"].between(21, 41) & ~req["is_random"] & (req["clicks"] > 0)].index.to_numpy()
tr_req = rng.choice(tr_req, size=40_000, replace=False)
is_tr = imps["request_id"].isin(tr_req).to_numpy()
is_va = imps["day"].isin(store.VALID_DAYS).to_numpy()

train_df, tr_sizes = ltr.make_groups(imps.loc[is_tr].assign(**F.loc[is_tr]), "request_id")
valid_df = imps.loc[is_va].assign(**F.loc[is_va])
X_tr, y_tr = train_df[ltr.FEATURES], train_df["clicked"].astype(int).to_numpy()
X_va = valid_df[ltr.FEATURES]
print(f"train: {len(tr_sizes):,} requests, {len(X_tr):,} rows | valid: {valid_df['request_id'].nunique():,} requests")
print("same request on both sides:", len(set(train_df["request_id"]) & set(valid_df["request_id"])))

# %% [markdown]
# **Expected:** 40,000 training requests (400,000 rows), about 32,600
# validation requests, and `0` shared requests.
#
# The evaluation helper below reports, per model:
#
# * **NDCG@10 truth** and **true exp. clicks**: graded against the hidden
#   relevance (notebook 05). These decide which model is better.
# * **NDCG@10 / MRR on clicks, policy traffic**: what a team without the
#   truth would report. Biased towards the logged order (notebook 05).
# * **NDCG@10 on clicks, random traffic**: much less biased *between two new
#   rankers* (the logged order is random there), but small and noisy.

# %%
T = bias.matrix(valid_df["true_p"])
Cv = bias.matrix(valid_df["clicked"]).astype(float)
rnd = bias.matrix(valid_df["is_random"])[:, 0]


def mrr(scores, labels):
    lab = np.take_along_axis(labels, np.argsort(-scores, axis=1, kind="stable"), axis=1)
    hit = lab.max(1) > 0
    return float((1.0 / (lab[hit].argmax(1) + 1)).mean())


def evaluate(scores) -> dict:
    s = bias.matrix(scores)
    return {"NDCG@10 truth": bias.ndcg_matrix(s, T).mean(),
            "true exp. clicks": bias.expected_clicks(s, T, THETA_TRUE),
            "NDCG@10 clicks (policy)": bias.ndcg_matrix(s[~rnd], Cv[~rnd]).mean(),
            "MRR clicks (policy)": mrr(s[~rnd], Cv[~rnd]),
            "NDCG@10 clicks (random)": bias.ndcg_matrix(s[rnd], Cv[rnd]).mean()}


results = {"logged policy order": evaluate(-valid_df["position"].to_numpy(float)),
           "true relevance (oracle)": evaluate(valid_df["true_p"].to_numpy())}
pd.DataFrame(results).T.round(4)

# %% [markdown]
# **Expected:** the logged order has NDCG@10 truth about 0.81, about 1.37
# true expected clicks (policy and random requests together), NDCG on policy
# clicks about 0.76 and MRR about 0.73; the oracle 1.00 and about 1.80, but
# only about 0.68 / 0.60 on policy clicks
# (the paradox of Chapter 23 again). On random-traffic clicks the logged order
# still scores higher than the oracle (about 0.67 vs 0.66): the logged
# positions *are* where clicks happen, even when they were assigned at
# random. Only compare **new** rankers with each other on that column.

# %% [markdown]
# ## 6.6 · Five rankers
#
# Same features, same rows, four LightGBM models plus a linear baseline:
#
# 1. **logistic regression** (pointwise, linear; one-hot categoricals, scaled numerics);
# 2. **GBDT pointwise** (`LGBMClassifier`, log loss);
# 3. **GBDT pointwise + IPS** (clicked rows weighted 1/θ̂_p, Chapter 23);
# 4. **LambdaMART** (`LGBMRanker`, `objective="lambdarank"`, NDCG-driven);
# 5. **LambdaMART + position feature** (trained with the logged position, served with position 0).

# %%
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GBDT = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=2, verbose=-1)
numeric = [c for c in ltr.FEATURES if c not in ltr.CATEGORICAL]
pos_tr = train_df["position"].to_numpy()
w_ips = np.where(y_tr == 1, bias.ips_weights(pos_tr, theta_hat, clip=10.0), 1.0)
timings = {}

t0 = time.time()
logreg = make_pipeline(ColumnTransformer([("num", StandardScaler(), numeric),
                                          ("cat", OneHotEncoder(handle_unknown="ignore"), ltr.CATEGORICAL)]),
                       LogisticRegression(max_iter=1000))
logreg.fit(X_tr, y_tr)
results["logistic regression"] = evaluate(logreg.decision_function(X_va))
timings["logistic regression"] = time.time() - t0

t0 = time.time()
point = lgb.LGBMClassifier(**GBDT).fit(X_tr, y_tr, categorical_feature=ltr.CATEGORICAL)
results["GBDT pointwise"] = evaluate(point.predict_proba(X_va)[:, 1])
timings["GBDT pointwise"] = time.time() - t0

t0 = time.time()
point_ips = lgb.LGBMClassifier(**GBDT).fit(X_tr, y_tr, sample_weight=w_ips, categorical_feature=ltr.CATEGORICAL)
results["GBDT pointwise + IPS"] = evaluate(point_ips.predict_proba(X_va)[:, 1])
timings["GBDT pointwise + IPS"] = time.time() - t0

t0 = time.time()
ranker = lgb.LGBMRanker(objective="lambdarank", **GBDT)
ranker.fit(X_tr, y_tr, group=tr_sizes, categorical_feature=ltr.CATEGORICAL)
results["LambdaMART"] = evaluate(ranker.predict(X_va))
timings["LambdaMART"] = time.time() - t0

t0 = time.time()
ranker_pos = lgb.LGBMRanker(objective="lambdarank", **GBDT)
ranker_pos.fit(X_tr.assign(position=pos_tr), y_tr, group=tr_sizes, categorical_feature=ltr.CATEGORICAL)
results["LambdaMART + position (serve 0)"] = evaluate(ranker_pos.predict(X_va.assign(position=0)))
timings["LambdaMART + position (serve 0)"] = time.time() - t0

table = pd.DataFrame(results).T
table["fit+predict s"] = pd.Series(timings)
table.round(4)

# %% [markdown]
# **Expected (our run; differences in the third decimal are noise):**
#
# * every learned ranker reaches NDCG@10 truth of about **0.975–0.977** and
#   about **1.755–1.76** true expected clicks per request, against 0.81 / 1.37
#   for the logged order: about **+29%** expected clicks from re-ranking the
#   very same items;
# * logistic regression ties the pointwise GBDT (both about 0.977), and
#   LambdaMART is slightly *lower* (about 0.976), as is LambdaMART with the
#   position feature; IPS weighting changes nothing measurable;
# * on policy clicks every learned ranker scores about 0.67–0.68 NDCG and
#   0.58–0.60 MRR, *below* the logged order (0.76 / 0.73), the paradox of Chapter 23;
#   on random-traffic clicks the learned rankers sit at about 0.63, with
#   differences of the order of the noise;
# * each model fits and predicts in about 3–12 s on 400,000 rows.
#
# Two lessons follow. First, the objective matters less than the features
# when the label is binary and lists are short (10 items, one or two
# clicks): LambdaMART's advantage grows with long lists and graded labels
# (search relevance grades, purchase > click). Second, IPS and the position
# feature do little *here* because none of these features encodes the logged
# position (in contrast with Section 5.6 of the Chapter 23 lab, where the
# policy score did). This should be checked before assuming that debiasing
# will change a metric.

# %% [markdown]
# ### How much do the models agree?
#
# Similar averages can hide different rankings. Pairwise accuracy (Exercise 1,
# library version `ltr.pairwise_accuracy`) against the **true**
# relevance, plus the rank correlation of LambdaMART with the other models:

# %%
from scipy.stats import spearmanr

sample_req = valid_df["request_id"].drop_duplicates().sample(3_000, random_state=0)
m = valid_df["request_id"].isin(sample_req).to_numpy()
scores = {"logged policy order": -valid_df["position"].to_numpy(float),
          "logistic regression": logreg.decision_function(X_va),
          "GBDT pointwise": point.predict_proba(X_va)[:, 1],
          "LambdaMART": ranker.predict(X_va)}
rows = {}
for name, s in scores.items():
    rows[name] = {"pairwise acc. vs truth": ltr.pairwise_accuracy(s[m], valid_df["true_p"].to_numpy()[m],
                                                                  valid_df["request_id"].to_numpy()[m]),
                  "Spearman with LambdaMART": spearmanr(s, scores["LambdaMART"]).statistic}
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# **Expected:** pairwise accuracy against the truth of about 0.53 for the
# logged order (barely better than a coin flip) and about 0.84–0.85 for the
# learned models; Spearman correlations with LambdaMART of about 0.96 for the
# GBDT pointwise model and about 0.94 for logistic regression, against about
# 0.09 for the logged order. (Spearman here is computed across all rows, so
# it also reflects how scores compare *across* requests, which ranking does
# not need.) The learned rankers agree with each other far more than with
# production.

# %% [markdown]
# ## 6.7 · Feature importance
#
# Gain importance: the total loss reduction from splits on each feature.
# Useful for debugging ("why is `i_age` at the top?"), not proof of
# causality, and biased towards high-cardinality numeric features.

# %%
imp = pd.DataFrame({"LambdaMART": ranker.booster_.feature_importance("gain"),
                    "GBDT pointwise": point.booster_.feature_importance("gain")}, index=ltr.FEATURES)
imp = (imp / imp.sum()).sort_values("LambdaMART", ascending=False)
pos_share = ranker_pos.booster_.feature_importance("gain")
print(f"position's share of gain in the position-feature model: {pos_share[-1] / pos_share.sum():.1%}")
imp.round(3)

# %% [markdown]
# **Expected:** for LambdaMART the user×category history leads
# (`u_cat_clicks` about 0.22 and `u_cat_ctr` about 0.13 of the gain), then
# `category` and `i_ctr` (about 0.12 each) and the user activity features;
# price features are small. The pointwise GBDT leans on `u_cat_ctr` instead
# (about 0.4): correlated features share credit arbitrarily, one reason not
# to over-read importances. In the position-feature model, position takes
# about **64%** of the gain: the model uses it heavily to explain clicks. Yet
# serving with position 0 changes the *ranking* by almost nothing (6.6),
# because here position's effect is close to additive in the trees' score;
# it would matter for calibrated click probabilities, and with features that
# correlate with position.
#
# Both pointwise and LambdaMART rank `category` highly. That is the world's
# structure (users have favourite categories), and `segment` matters through
# its interaction with category. Trees find that interaction; logistic
# regression cannot without crossed features, which is part of why it
# trails slightly.

# %%
ranker.booster_.save_model(str(ART / "level2" / "ltr_lambdamart.txt"))
table.to_csv(ART / "level2" / "ltr_results.csv")
print("saved:", sorted(p.name for p in (ART / "level2").glob("ltr_*")))

# %% [markdown]
# > **Exercise 3.** Purchases are rarer than clicks but closer to what the
# > business values. Propose a label (or label + weight) scheme for
# > LambdaMART that uses both, explain how to choose its parameters, and
# > how to evaluate it offline without misleading results. Write the
# > answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · DCG and NDCG.** In words: add up each item's gain, discounted by the log of its rank, and divide by the best possible order's value.
#
# $$
# \operatorname{DCG@}k = \sum_{i=1}^{k} \frac{2^{\,\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \operatorname{NDCG@}k = \frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}
# $$
#
# * $\text{rel}_i$: relevance of the item at rank $i$ (1 = top); IDCG: the DCG of the ideal order. *Example:* clicks at ranks 2 and 5: $\text{DCG} = 1/\log_2 3 + 1/\log_2 6 = 0.631 + 0.387 = 1.018$; ideal (ranks 1, 2): $1 + 0.631 = 1.631$; NDCG = 0.624.
#
# **2 · MRR.** In words: the average, over requests, of one over the rank of the first relevant item.
#
# $$
# \operatorname{MRR} = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\text{rank}_q}
# $$
#
# * *Example:* first clicks at ranks 1, 2 and 4 in three requests: $(1 + 0.5 + 0.25) / 3 = 0.583$.
#
# **3 · RankNet's pairwise loss and its gradient.** In words: a logistic loss on the score difference of a pair where $i$ should beat $j$.
#
# $$
# L_{ij} = \log\!\left(1 + e^{-\sigma (s_i - s_j)}\right), \qquad \lambda_{ij} = \frac{\partial L_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma (s_i - s_j)}}
# $$
#
# * $s_i, s_j$: model scores; $\sigma$: a scale (1 here). *Example:* $s_i = 0.2$, $s_j = 0.5$ (wrong order): $L = \log(1 + e^{0.3}) = 0.854$, $\lambda = -0.574$: push $s_i$ up.
#
# **4 · LambdaRank.** In words: use RankNet's gradient, scaled by how much NDCG would change if $i$ and $j$ swapped places.
#
# $$
# \lambda^{\text{LR}}_{ij} = \lambda_{ij} \cdot \left\lvert \Delta \operatorname{NDCG}_{ij} \right\rvert
# $$
#
# * *Example:* one relevant item at rank 2 vs rank 1: $\lvert\Delta\rvert = 1 - 1/\log_2 3 = 0.369$; at rank 10 vs 9: $1/\log_2 10 - 1/\log_2 11 = 0.012$. LambdaMART fits boosted trees to these $\lambda$ "gradients"; they are not the gradient of an explicitly stated loss, and the method works nonetheless.
#
# The cell below checks 1, 3 and 4 numerically.

# %%
# Check NDCG, RankNet's gradient and the LambdaRank weight numerically
from sklearn.metrics import ndcg_score

from shoprank.metrics import ndcg_at_k


def dcg(rels, k=10):
    rels = np.asarray(rels, dtype=float)[:k]
    return ((2 ** rels - 1) / np.log2(np.arange(2, rels.size + 2))).sum()


labels = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])  # clicks, in the model's rank order
scores = np.linspace(1, 0, 10)                      # scores that produce that order
print(f"1. NDCG@10 by hand {dcg(labels) / dcg(np.sort(labels)[::-1]):.4f} | "
      f"sklearn {ndcg_score([labels], [scores], k=10):.4f} | shoprank {ndcg_at_k(labels, 10):.4f}")

s_i, s_j, eps = 0.2, 0.5, 1e-6


def ranknet(si, sj):
    return np.log1p(np.exp(-(si - sj)))


lam = -1 / (1 + np.exp(s_i - s_j))
finite_diff = (ranknet(s_i + eps, s_j) - ranknet(s_i - eps, s_j)) / (2 * eps)
print(f"3. RankNet loss {ranknet(s_i, s_j):.4f} | dL/ds_i {lam:.4f} vs finite difference {finite_diff:.4f}")
swap = lambda a, b: abs(1 / np.log2(a + 1) - 1 / np.log2(b + 1))  # noqa: E731
print(f"4. |dNDCG| rank 2 vs 1: {swap(2, 1):.3f} | rank 10 vs 9: {swap(10, 9):.3f}")

# %% [markdown]
# **Expected:** NDCG@10 = 0.6241 three times (sklearn uses linear gain, identical for 0/1 labels); RankNet loss 0.8544 and a gradient of −0.5744 both ways; |ΔNDCG| 0.369 vs 0.012.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `LightGBMError: ... sum of group counts ... not equal to number of data` | `group` sizes do not add up to the rows | build sizes with `make_groups` from the same frame that is passed as `X` |
# | Ranker trains but NDCG is barely above random | rows not contiguous by group (shuffled or concatenated after computing sizes) | sort by group (stable) right before `fit`; never shuffle afterwards |
# | Offline NDCG suspiciously high (0.95+ on clicks) | a feature computed with same-day data (label leakage) | recompute features from a truncated log and compare (6.2) |
# | `ValueError: ... categorical_feature ... must be int` | float or string categoricals | cast to `int32` codes (as `build_features` does) |
# | Kernel slow or killed while building features | too many rows in memory at once | restrict to the days needed; drop unused columns before `build_features` |
# | New model loses to production on click NDCG | position bias in the metric (Chapter 23) | compare on truth (here), IPS estimates, or random traffic between new models |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. Pointwise, pairwise, listwise: what does each optimise, and when is each appropriate? Why can a pointwise model rank well while being badly calibrated, and when does calibration matter?
# 2. Derive (or describe) the LambdaRank gradient. Why multiply by |ΔNDCG|, and why does it work although NDCG is not differentiable?
# 3. In a home feed there is no query. What is the "group" for LTR, and what changes when grouping by user-day instead of request?
# 4. A ranker's offline NDCG is 0.95 on the training week and 0.70 on the next week. List the leakage and drift checks to run, in order.
# 5. Should position be a feature? Describe two ways to use it in training, what is set at serving, and how to know whether it helped.
# 6. A LambdaMART model and a logistic regression are within 0.002 NDCG offline. Which should be deployed, and what else should be measured before deciding?
#
# **Next chapter:** [Chapter 25 · Candidate generation](../ch25-candidate-generation/notebook.ipynb) — the ranker can only order what it is given: building and measuring the retrieval stage that selects a few hundred candidates from the whole catalogue.
