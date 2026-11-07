# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/20_final_evaluation_and_mock_loop.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 20 · Final evaluation & mock loop
#
# **Day 10 · ~90 minutes · CPU only**
#
# For nineteen notebooks the test week (days 49–55) has stayed closed. Today
# it opens, once, under a plan written down *before* the first test number is
# computed, so that nothing you see can change what you claim. Then you turn
# the same discipline on yourself: a timed, scored mock interview loop with a
# progress log you keep adding to until interview day.
#
# By the end you will be able to:
#
# 1. pre-register an evaluation (model, metrics, sample, uncertainty method, decision rule) and show it was saved first;
# 2. evaluate a ranking system on held-out time with several lenses: true relevance, clicks, IPS-corrected clicks, and retrieval recall;
# 3. estimate position propensities from randomised traffic and see IPS reverse a conclusion that raw clicks get wrong;
# 4. put grouped bootstrap intervals on every number, resampling users rather than rows;
# 5. run a four-round mock loop against a rubric and log your progress.
#
# **You will produce:** `artifacts/level2/final/preregistration.json`,
# `artifacts/level2/final/test_results.json`, and
# `artifacts/level2/mock_loop/` (`plan.md`, `rubric.json`, `progress_log.csv`).

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
# --- Catch up: data and the notebook-17 ranker, trained on TRAIN days only ---
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shoprank import serving as sv
from shoprank import store
from shoprank.grading import nb20 as grade

FINAL = ART / "level2" / "final"
FINAL.mkdir(parents=True, exist_ok=True)

t0 = time.time()
world, users, items, imps = store.load("small")
RANKER_CONFIG = {"feature_days": [0, 34], "label_days": [35, 41], "n_estimators": 150, "seed": 0}
_, ranker, _ = sv.train_ranker(imps, items, len(users), feature_days=(0, 34), label_days=(35, 41),
                               n_estimators=150, seed=0)
print(f"{len(imps):,} impressions loaded, ranker trained ({time.time() - t0:.0f}s)")

# %% [markdown]
# **Expected:** `2,446,900 impressions loaded, ranker trained` after 15–60 s.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Validation vs test set | data you may use to choose, versus data opened once for the final claim | days 42–48 versus days 49–55 |
# | Pre-registration | a written, saved plan of exactly what will be measured and what counts as success, made before the results exist | `preregistration.json` |
# | Hash | a short fingerprint of a file; changing one byte changes it | the 16-character SHA-256 prefix stored with the results |
# | Propensity (examination probability) | the chance a user even looks at a given position | a power law with η ≈ 0.88 |
# | Exploration (randomised) traffic | requests shown in random order so that position effects can be measured | 5% of ShopStream's requests |
# | IPS | inverse propensity scoring: weight each click by 1 / (chance its position was examined) | IPS-DCG reverses the raw-click verdict |
# | Grouped bootstrap | resampling whole users rather than rows, so a user's correlated requests move together | 1,000 replicates by user |
# | Paired difference | comparing two systems on the same requests, so shared noise cancels | ranker − policy: +0.29 IPS-DCG |
# | Percentile interval | the 2.5th to 97.5th percentiles of the bootstrap replicates | cascade 0.715 [0.709, 0.721] |
# | Mock loop | a timed rehearsal of the interview rounds, scored against a rubric | four rounds, 165 minutes |
# | Rubric anchors | short descriptions of what each score means, so scores are comparable over time | 1 = missing or wrong … 4 = strong |

# %% [markdown]
# ## 20.1 · Choose on validation, not on test
#
# Everything that involves *choosing* happens on the validation week: which
# system, which configuration. Here the candidates are the systems of
# notebook 17, all with the product's category cap of 3 (a product decision
# fixed before this evaluation) and, for reference, without it.

# %%
valid = imps[imps["day"].between(store.VALID_DAYS[0], store.VALID_DAYS[-1])]
state_valid = sv.build_state(imps, items, len(users), feature_days=(0, 41))


def evaluate_systems(state, requests, P, cap):
    """Catalogue NDCG@10 per request for each candidate system on the given requests."""
    casc = sv.Cascade(state, ranker, store=sv.OnlineStore(state, latency_s=0.0), max_per_category=cap)
    out = {"cascade (LightGBM)": [], "ANN only": [], "popularity": []}
    for u, d in zip(requests["user_id"], requests["day"]):
        out["cascade (LightGBM)"].append(casc.recommend(u, d)["items"])
        user, _ = casc._user(u)
        ann = casc.index.search(user["emb"][None, :], 300)[1][0] if user["emb"] is not None else state.popular[:300]
        out["ANN only"].append(sv.business_rules(np.concatenate([ann, state.popular[:300]]), state.item_cat,
                                                 state.item_created, d, cap, 10)[0])
        out["popularity"].append(sv.business_rules(state.popular[:300], state.item_cat, state.item_created,
                                                   d, cap, 10)[0])
    return {name: sv.catalogue_ndcg(P, lists) for name, lists in out.items()}


req_valid = valid.groupby("request_id")[["user_id", "day"]].first().sample(1000, random_state=0)
P_valid = sv.true_prob_matrix(world, req_valid["user_id"].to_numpy(), req_valid["day"].to_numpy())
valid_results = {f"cap {cap}": {k: round(float(v.mean()), 4) for k, v in
                                evaluate_systems(state_valid, req_valid, P_valid, cap).items()}
                 for cap in (3, 10)}
pd.DataFrame(valid_results)

# %% [markdown]
# **Expected (validation, catalogue NDCG@10):** with cap 3, the cascade
# 0.71, ANN-only 0.63, popularity 0.40; without the cap 0.875, 0.79 and 0.39.
# The cascade numbers match notebook 17 exactly, as they should. (ANN-only
# with the cap is higher than notebook 17's 0.58 because here it takes 300
# neighbours, so the cap can backfill with more ANN items.) The cascade wins
# either way, so it is the system we take to the test week.

# %% [markdown]
# ## 20.2 · Pre-register, then look
#
# A pre-registration is a short file written **before** the test data is
# scored. It fixes: the system and its configuration, the data window, every
# metric and how it is computed, the sample, the uncertainty method, and what
# result would count as success. Its hash goes into the results file, so a
# reviewer can check the plan predates the numbers. (In a company this is the
# experiment design doc or the launch review's "success criteria" section.)
#
# Decisions fixed here that are easy to fudge later:
#
# * **Feature refresh:** for the test week the serving state is rebuilt from
#   days 0–48 (train + validation), as production would refresh it; the
#   ranker's weights stay frozen (trained on days 35–41 labels).
# * **Primary metric:** catalogue NDCG@10 against the true relevance, cap 3,
#   2,000 test requests sampled with seed 20; the comparison that matters is
#   cascade vs popularity and ANN-only.
# * **Secondary metrics:** re-rank NDCG@10 on clicks; IPS-corrected DCG@10 on
#   clicks with propensities estimated from *train-day* random traffic;
#   retrieval recall@100.
# * **Uncertainty:** 95 % percentile intervals from a bootstrap that resamples
#   **users** (1,000 replicates, seed 0).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-x-preregistration.svg" alt="Five steps: choose on validation, pre-register and hash the plan, open the test week once, bootstrap by user, report with the plan's hash; a dashed forbidden path leads from the test back to choosing" width="100%">
#
# *What to notice:* the dashed path is the one thing the procedure forbids; everything else exists to make it detectable.
#
# > 💡 **Intuition.** Pre-registration is calling your shot in pool: "eight ball, corner
# > pocket" before you hit. Sinking a ball you did not call does not count, however good
# > it looked. *Where it breaks down:* a plan cannot anticipate everything; when you must
# > deviate, write a new, dated plan with the reason, and report both.

# %%
prereg = {
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "system": "cascade: FAISS(MF) + co-visitation + popularity -> LightGBM -> rules (cap 3)",
    "ranker": RANKER_CONFIG,
    "serving_state_days": [0, 48],
    "chosen_on": "validation days 42-48, catalogue NDCG@10, 1,000 requests (seed 0)",
    "validation_results": valid_results,
    "test_days": [store.TEST_DAYS[0], store.TEST_DAYS[-1]],
    "primary_metric": "catalogue NDCG@10 vs world.true_click_prob, cap 3, 2,000 test requests (seed 20)",
    "secondary_metrics": [
        "re-rank NDCG@10 on clicks, policy (non-random) test requests",
        "IPS-DCG@10 on clicks, propensities = power-law fit to position CTR on train-day random traffic",
        "retrieval recall@100 (50 ANN + 25 co-visitation + 25 popular) of the true top-10 and of clicked items",
    ],
    "uncertainty": {"method": "percentile bootstrap resampling users", "n_boot": 1000, "seed": 0, "level": 0.95},
    "success_criterion": "cascade beats popularity and ANN-only on the primary metric with non-overlapping "
                         "user-bootstrap intervals; secondary metrics reported whatever they show",
    "rule": "no change to the system, metrics or sample after this file is written",
}
prereg_path = FINAL / "preregistration.json"
if prereg_path.exists():
    print("pre-registration already on file (kept, not overwritten):", prereg_path)
else:
    prereg_path.write_text(json.dumps(prereg, indent=2))
    print("pre-registration written:", prereg_path)
prereg_sha = hashlib.sha256(prereg_path.read_bytes()).hexdigest()[:16]
print("sha256:", prereg_sha)

# %% [markdown]
# **Expected:** `pre-registration written: .../level2/final/preregistration.json`
# and a 16-character hash (different every run: the file contains its
# creation time). On a re-run in the same runtime it says "already on file":
# a plan is never silently rewritten.

# %% [markdown]
# ## 20.3 · Propensities from randomised traffic
#
# The IPS metric needs P(examined | position). Only the **random** requests
# (5 % of traffic, items in uniformly random order) give it without bias: in
# random lists, the average relevance is the same at every position, so CTR
# by position is proportional to examination. The estimator is two lines,
# implemented here rather than imported. Using *train-day* random traffic
# keeps the test week untouched.
#
# > 💡 **Intuition.** IPS is a survey where people near the door are asked far more often
# > than people at the back: to estimate the whole room, weight each answer by 1 / (the
# > chance that person was asked). A click at a rarely examined position speaks for many
# > unseen ones. *Where it breaks down:* a tiny propensity means a huge weight, so one
# > lucky click can dominate; clip the weights and report that you did.

# %%
rnd = imps[(imps["day"] <= store.TRAIN_DAYS[-1]) & imps["is_random"]]
by_pos = rnd.groupby("position")["clicked"].agg(["mean", "sum"])
raw_prop = (by_pos["mean"] / by_pos["mean"].iloc[0]).to_numpy()
# Smooth with a power law: CTR(k) = c * (1 + k)^-eta, fitted in log-log space over all
# positions, each weighted by sqrt(clicks) (log CTR is noisier where clicks are few).
k = np.arange(10)
eta_hat = -np.polyfit(np.log1p(k), np.log(by_pos["mean"].to_numpy()), 1, w=np.sqrt(by_pos["sum"].to_numpy()))[0]
propensity = (1.0 + k) ** -eta_hat
print(f"{len(rnd):,} random-traffic impressions in train days | fitted eta = {eta_hat:.2f} "
      f"(the simulator's hidden value: {world.config.eta})")
pd.DataFrame({"raw CTR ratio": raw_prop, "power-law fit": propensity,
              "hidden truth": world.examination_prob(np.arange(10))}).round(3).T

# %% [markdown]
# **Expected:** 89,480 random impressions (8,948 per position); raw ratios
# falling from 1 to about 0.1, bumpy in the tail (positions 6 and 7 are out of
# order: only ~250 clicks each); a fitted eta of about 0.88 against the hidden
# 0.9; the fit within about 0.02 of the truth at every position. On real data there is no
# "hidden truth" row, which is why you smooth and why you keep exploration
# traffic running.

# %% [markdown]
# ## 20.4 · Open the test week
#
# From here on, nothing is tuned. The cell refuses to run without the
# pre-registration file.

# %%
assert prereg_path.exists(), "write the pre-registration first (20.2)"
test = imps[imps["day"].between(store.TEST_DAYS[0], store.TEST_DAYS[-1])]
state_test = sv.build_state(imps, items, len(users), feature_days=(0, 48))  # refreshed features, frozen ranker
print(f"test week: {len(test):,} impressions, {test['request_id'].nunique():,} requests, "
      f"{test['user_id'].nunique():,} users | CTR {test['clicked'].mean():.3f}")

# (a) primary metric: catalogue NDCG@10 on 2,000 sampled test requests, cap 3
req_test = test.groupby("request_id")[["user_id", "day"]].first().sample(2000, random_state=20)
P_test = sv.true_prob_matrix(world, req_test["user_id"].to_numpy(), req_test["day"].to_numpy())
primary = evaluate_systems(state_test, req_test, P_test, cap=3)
shown = test[test["request_id"].isin(req_test.index)].sort_values(["request_id", "position"])
primary["logged policy (uncapped)"] = sv.catalogue_ndcg(
    P_test, shown.groupby("request_id")["item_id"].apply(np.array).loc[req_test.index].tolist())

# (b) retrieval recall@100 with a 100-candidate budget
small = sv.Cascade(state_test, ranker, store=sv.OnlineStore(state_test, latency_s=0.0), n_ann=50, n_covis=25, n_pop=25)
clicked_by_req = test[test["clicked"]].groupby("request_id")["item_id"].apply(set)
rec_true, rec_click = [], []
for j, (rid, u, d) in enumerate(zip(req_test.index, req_test["user_id"], req_test["day"])):
    user, _ = small._user(u)
    cands, _ = small.retrieve(u, d, user)
    cands = set(cands[:100].tolist())
    rec_true.append(len(cands & set(np.argsort(-P_test[j])[:10].tolist())) / 10)
    if rid in clicked_by_req.index:
        rec_click.append(len(cands & clicked_by_req[rid]) / len(clicked_by_req[rid]))
print(f"retrieval recall@100: true top-10 {np.mean(rec_true):.3f} | clicked items {np.mean(rec_click):.3f} "
      f"({len(rec_click)} requests with clicks)")
pd.Series({k: v.mean() for k, v in primary.items()}, name="catalogue NDCG@10 (test, cap 3)").round(4)

# %% [markdown]
# **Expected:** 336,250 test impressions in 33,625 requests from 8,457 users
# (CTR 0.138). Retrieval recall@100: 0.63 of the true top 10, but only 0.30
# of the items users actually clicked (1,524 of the 2,000 requests had a
# click). Catalogue NDCG@10 (cap 3): cascade 0.715, ANN-only 0.635,
# popularity 0.38, logged policy 0.41 — the validation ordering holds.
#
# The two recall numbers disagree, and both are right. Clicked items were
# chosen by the *logging policy* from a random pool of 200 eligible items; a
# good retriever does not reproduce that pool, so "recall of clicked items"
# measures agreement with the old system as much as quality. Recall against
# logged clicks is the metric you usually have; know its bias.

# %% [markdown]
# Now the logged lists of the test week: re-rank the 10 items each
# **policy** request showed (random-order requests are excluded: they have no
# "policy order" to compare with) by three orderings, and score each with
# three lenses: NDCG@10 on raw clicks, IPS-DCG@10 on clicks (each click
# weighted by 1 / propensity of the position where it was *logged*,
# discounted by its *new* rank), and, because ShopStream can, the true
# expected DCG.

# %%
pol = test[~test["is_random"]]
X_pol = sv.pair_features(state_test, pol["user_id"].to_numpy(), pol["item_id"].to_numpy(),
                         pol["day"].to_numpy(), position=0)
clicks = pol["clicked"].to_numpy(float).reshape(-1, 10)
logged_pos = pol["position"].to_numpy().reshape(-1, 10)
truth = world.true_click_prob(pol["user_id"].to_numpy(), pol["item_id"].to_numpy()).reshape(-1, 10)
req_user = pol["user_id"].to_numpy()[::10]


def new_ranks(scores):
    order = np.argsort(-scores.reshape(-1, 10), axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(10)[None, :].repeat(len(order), 0), axis=1)
    return ranks


def lenses(scores):
    disc = 1.0 / np.log2(new_ranks(scores) + 2)
    return {"NDCG@10 on clicks": sv.ndcg_by_request(scores, clicks.ravel()),  # NaN when no click
            "IPS-DCG@10 on clicks": (clicks / propensity[logged_pos] * disc).sum(1),
            "true DCG@10": (truth * disc).sum(1)}


orderings = {"logging policy (as shown)": -pol["position"].to_numpy(float),
             "LightGBM ranker": sv.score_with(ranker, X_pol),
             "random": np.random.default_rng(0).random(len(pol))}
per_request = {name: lenses(s) for name, s in orderings.items()}
pd.DataFrame({name: {lens: np.nanmean(v) for lens, v in d.items()} for name, d in per_request.items()}).round(4)

# %% [markdown]
# **Expected (test week, ~32,000 policy requests):**
#
# | | NDCG@10 on clicks | IPS-DCG@10 | true DCG@10 |
# |---|---|---|---|
# | logging policy | 0.760 | 1.92 | 1.95 |
# | LightGBM ranker | 0.664 | 2.22 | 2.26 |
# | random | 0.529 | 1.86 | 1.90 |
#
# This is Level 2's central lesson in one table. **Raw clicks say the old
# policy is best** by a wide margin; that is position bias — the items the
# policy put on top got examined, so they got the clicks. **IPS reverses the
# conclusion** and agrees with the truth: the ranker's lists are better.
# IPS is not perfect (it runs a little below the true DCG here because the
# propensity estimate is not exact, and its variance is higher), but it is
# pointed in the right direction, and raw clicks are not.

# %% [markdown]
# ## 20.5 · Uncertainty, by user
#
# Requests of one user are correlated (same taste, same activity), so the
# unit of resampling is the **user**. A row-level bootstrap would give
# intervals that are too narrow.
#
# > 💡 **Intuition.** Five answers from one household are mostly one household's opinion.
# > Resample households (users), not answers (requests), or the interval pretends you asked
# > five times as many people. *Where it breaks down:* if users interact (a marketplace, a
# > social graph), even users are not independent, and the cluster must be bigger still.
#
# > **Exercise 1.** Write `bootstrap_ci_grouped(values, groups, n_boot, seed)`
# > returning `(mean, lo, hi)`: `mean` is the mean of all `values`; `lo`/`hi`
# > the 2.5th/97.5th percentiles of `n_boot` bootstrap replicates, where each
# > replicate draws as many groups as there are, **with replacement**, and
# > computes the mean over all *rows* of the drawn groups (a group drawn twice
# > counts twice). Use `np.random.default_rng(seed)`. `groups` may be any
# > hashable labels. Vectorise it: no Python loop over rows.

# %%
def bootstrap_ci_grouped(values, groups, n_boot, seed):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(bootstrap_ci_grouped)

# %% [markdown]
# The results below use the module's version, `sv.grouped_bootstrap` (same
# algorithm, chunked). Intervals for every system and lens, then the **paired
# differences** (ranker − policy on the same requests), which are much
# tighter than comparing two separate intervals:

# %%
rows = []
for name, v in primary.items():
    rows.append({"metric": "catalogue NDCG@10 (primary)", "system": name,
                 **dict(zip(["mean", "lo", "hi"], sv.grouped_bootstrap(v, req_test["user_id"], 1000, 0)))})
for name, d in per_request.items():
    for lens, v in d.items():
        ok = ~np.isnan(v)
        rows.append({"metric": lens, "system": name,
                     **dict(zip(["mean", "lo", "hi"], sv.grouped_bootstrap(v[ok], req_user[ok], 1000, 0)))})
for lens in per_request["LightGBM ranker"]:
    diff = per_request["LightGBM ranker"][lens] - per_request["logging policy (as shown)"][lens]
    ok = ~np.isnan(diff)
    rows.append({"metric": lens, "system": "ranker − policy (paired)",
                 **dict(zip(["mean", "lo", "hi"], sv.grouped_bootstrap(diff[ok], req_user[ok], 1000, 0)))})
results = pd.DataFrame(rows)
results_path = FINAL / "test_results.json"
results_path.write_text(json.dumps({"preregistration_sha256_16": prereg_sha,
                                    "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "retrieval_recall@100": {"true_top10": float(np.mean(rec_true)),
                                                             "clicked_items": float(np.mean(rec_click))},
                                    "propensity_eta_hat": float(eta_hat),
                                    "results": results.round(4).to_dict(orient="records")}, indent=2))
print("saved", results_path.name, "linked to pre-registration", prereg_sha)
results.round(3)

# %% [markdown]
# **Expected:** primary metric intervals about ±0.006–0.011: cascade 0.715
# [0.709, 0.721], ANN-only 0.635 [0.629, 0.641], popularity 0.38
# [0.37, 0.39] — no overlap, so the pre-registered success criterion is met.
# Paired differences, ranker − policy: NDCG on clicks −0.096 [−0.100,
# −0.092], entirely below zero; IPS-DCG +0.29 [0.28, 0.31] and true DCG
# +0.31 [0.309, 0.315], both entirely above zero.
#
# How to report this in one sentence (practise it): *"On the held-out test
# week, the cascade's catalogue NDCG@10 was 0.715 (95 % CI 0.709–0.721,
# bootstrap over users) versus 0.635 for embedding retrieval alone; on logged
# traffic raw clicks favour the incumbent policy, but after inverse-propensity
# correction with propensities from randomised traffic, the ranker is ahead by
# 0.29 DCG (0.28–0.31), in line with the simulator's ground truth of 0.31."*

# %% [markdown]
# ## 20.6 · The mock loop
#
# A big-tech ML engineer loop is usually four to six interviews in one day.
# This notebook's mock covers the four core rounds, back to back, timed:
#
# | Round | Minutes | Format | Material |
# |---|---|---|---|
# | 1 · Coding | 45 | 1–2 data-structure/algorithm problems, live, talk while you code | Level 2 notebooks 15–16 |
# | 2 · ML coding | 45 | implement an ML method or metric from scratch, test it | notebooks 13–14, 05–06 |
# | 3 · ML system design | 45 | one open prompt, whiteboard-style | notebooks 17–19 |
# | 4 · Behavioural | 30 | 3–4 STAR stories, with follow-ups | Level 1 notebook 17 (STAR) |
#
# Rules that make a mock useful: a real timer, no notes, a partner or a
# recording, and **scores written down within 10 minutes** of finishing, in the
# log below. The prompts are drawn at random each time you run the next cell.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-x-mock-loop.svg" alt="Four timed rounds in a row, then score within ten minutes, log, practise the lowest column and run the next mock, looping back" width="100%">
#
# *What to notice:* the loop around the rounds is the point: a mock that is not scored and logged cannot tell you what to practise next.

# %%
import random

MOCK = ART / "level2" / "mock_loop"
MOCK.mkdir(parents=True, exist_ok=True)

ROUNDS = {
    "coding": {"minutes": 45, "prompts": [
        "Top-k most frequent item ids in a stream, then over a sliding window of the last W events",
        "Merge k sorted candidate lists (score, item_id) into one top-n list without duplicates",
        "Minimum number of meeting rooms for a list of intervals; then return the assignment",
        "Course schedule: can all courses be finished given prerequisites? Return an order",
        "Longest substring with at most k distinct characters",
        "Median of a data stream, with add() and median() in O(log n)",
        "Number of connected components after each edge insertion (union-find)",
        "Cheapest path with at most k stops in a weighted directed graph",
    ]},
    "ml_coding": {"minutes": 45, "prompts": [
        "Logistic regression with mini-batch SGD and L2 in NumPy; verify the gradient numerically",
        "k-means with k-means++ initialisation; handle empty clusters",
        "NDCG@k and MRR with ties and empty lists; test against hand-computed cases",
        "Scaled dot-product attention with a padding mask; check shapes and masking",
        "Best split of a decision stump by Gini impurity on one numeric feature, O(n log n)",
        "IPS-weighted click metric for a re-ranked list, given logged positions and propensities",
        "AUC from scratch in O(n log n) with tied scores",
        "A two-tower model's in-batch softmax loss with a log-popularity correction (PyTorch)",
    ]},
    "ml_design": {"minutes": 45, "prompts": [
        "Home-feed recommendations for a 50M-DAU marketplace",
        "Product search ranking for the same marketplace",
        "Short-video 'For You' feed",
        "Ads click-through prediction for a feed",
        "People you may know",
        "Push-notification recommendations",
        "Query autocomplete",
        "'Similar items' on a product page",
    ]},
    "behavioural": {"minutes": 30, "prompts": [
        "A time you disagreed with a technical decision. What did you do?",
        "A project that failed or missed its goal. What did you learn?",
        "Influencing a team or stakeholder without authority",
        "A trade-off between speed and quality under a deadline",
        "The hardest bug or data problem you have tracked down",
        "Feedback you received that changed how you work",
    ]},
}

RUBRIC = {
    "coding": ["problem_solving", "code_correctness", "testing", "complexity", "communication"],
    "ml_coding": ["correctness", "numerical_care", "testing", "explanation", "communication"],
    "ml_design": ["structure", "metrics_and_labels", "quantification", "trade_offs", "operations"],
    "behavioural": ["star_structure", "ownership", "impact_with_numbers", "reflection"],
}
ANCHORS = {1: "missing or wrong; needed heavy hints", 2: "partly there; gaps under follow-up",
           3: "solid, independent, handles follow-ups", 4: "excellent; would raise the bar"}

draw = {name: random.choice(r["prompts"]) for name, r in ROUNDS.items()}  # deliberately unseeded
for name, prompt in draw.items():
    print(f"{name:>12} ({ROUNDS[name]['minutes']} min): {prompt}")

plan = ["# Mock loop plan", "", f"Generated {datetime.now().date().isoformat()}. Total 165 minutes plus breaks.", ""]
for name, r in ROUNDS.items():
    plan += [f"## {name} · {r['minutes']} min", "", f"Today's prompt: **{draw[name]}**", "",
             "Score 1-4 on: " + ", ".join(RUBRIC[name]), ""]
plan += ["## Anchors", ""] + [f"* {k}: {v}" for k, v in ANCHORS.items()]
(MOCK / "plan.md").write_text("\n".join(plan) + "\n")
(MOCK / "rubric.json").write_text(json.dumps({"rubric": RUBRIC, "anchors": ANCHORS}, indent=2))
print("wrote", sorted(p.name for p in MOCK.iterdir()))

# %% [markdown]
# **Expected:** one prompt per round (different every run: the one
# deliberately unseeded cell, as in Level 1) and `wrote ['plan.md', 'rubric.json']`
# (plus `progress_log.csv` once you have logged a round).
#
# Time each round with these two helpers: `start_round` when you begin,
# `finish_round` with your scores when the timer stops. The log is a CSV that
# only ever grows.

# %%
LOG = MOCK / "progress_log.csv"
_running: dict = {}


def start_round(name: str) -> None:
    _running[name] = time.time()
    print(f"{name}: started; {ROUNDS[name]['minutes']} minutes. Prompt: {draw[name]}")


def finish_round(name: str, scores: dict, notes: str = "") -> pd.DataFrame:
    missing = set(RUBRIC[name]) - set(scores)
    assert not missing, f"score every dimension of {name}: missing {sorted(missing)}"
    bad = {k: v for k, v in scores.items() if v not in (1, 2, 3, 4)}
    assert not bad, f"scores must be whole numbers 1-4, got {bad}"
    minutes = round((time.time() - _running.pop(name)) / 60, 1) if name in _running else None
    row = pd.DataFrame([{"date": datetime.now().date().isoformat(), "round": name, "prompt": draw[name],
                         "minutes": minutes, "mean_score": round(float(np.mean(list(scores.values()))), 2),
                         **{f"score_{k}": v for k, v in scores.items()}, "notes": notes}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return pd.read_csv(LOG)


# --- After a round, fill in and run, e.g.:
# start_round("ml_coding")
# ... 45 minutes later ...
# finish_round("ml_coding", {"correctness": 3, "numerical_care": 2, "testing": 3, "explanation": 3,
#                            "communication": 3}, notes="forgot the log-sum-exp trick")
history = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
print(f"{len(history)} round(s) logged in {LOG.name}")

# %% [markdown]
# **Expected:** on a fresh run `0 round(s) logged in progress_log.csv`.
# After your first loop, four rows; after a few weeks, a history you can plot.
#
# Here is how to read the log, shown with a clearly labelled **example**
# history when yours is empty (nothing is saved):

# %%
show = history if len(history) else pd.DataFrame([
    {"date": "EXAMPLE week 1", "round": r, "mean_score": s} for r, s in
    [("coding", 2.0), ("ml_coding", 2.6), ("ml_design", 2.2), ("behavioural", 3.0)]] + [
    {"date": "EXAMPLE week 3", "round": r, "mean_score": s} for r, s in
    [("coding", 2.6), ("ml_coding", 3.0), ("ml_design", 3.0), ("behavioural", 3.25)]])
progress = show.pivot_table(index="date", columns="round", values="mean_score", aggfunc="mean")
progress

# %% [markdown]
# **Expected:** a date × round table (two EXAMPLE rows if your log is empty).
# Next week's practice goes to the lowest column. Aim for 3+ in every round,
# on fresh prompts, before booking real interviews.
#
# ## 20.7 · Levelling expectations
#
# What distinguishes a **mid-level** from a **senior** answer, phrased
# generically (every company's ladder differs; use this to calibrate, not to
# label yourself):
#
# | Round | Mid-level answer | Senior answer adds |
# |---|---|---|
# | Coding | correct, working solution with complexity stated, after some hints | optimal approach found independently, clean code, edge cases and tests unprompted, crisp trade-off discussion |
# | ML coding | correct implementation that runs; basic tests | numerical stability (log-sum-exp, clipping), vectorisation, tests against a reference, explains failure modes and how it would behave at scale |
# | ML system design | a sensible cascade, standard metrics, a model per stage | drives the conversation; numbers (QPS, latency, memory); label bias and feedback loops; experimentation and rollout; prioritises what to build first and why; names what would change the design |
# | Behavioural | clear STAR stories about own tasks | scope beyond own tasks (influence, mentoring, cross-team), measurable impact, honest reflection on mistakes, judgment under ambiguity |
# | Project deep-dive | can explain what was built and the metric | can defend every decision against alternatives, knows the system's limits, the evaluation's biases and what they would do next |
#
# Your Level 2 project is designed to give you senior-style material for the
# deep-dive: a cascade with measured stage latencies (17), an optimisation
# table (18), a design with numbers (19), and a pre-registered held-out
# evaluation in which position bias reverses a naive conclusion (20).
#
# > **Exercise 2 (written).** Write the script of a **3-minute walkthrough**
# > of your Level 2 system, as you would give it at the start of a project
# > deep-dive: the problem and metric, the data and its main bias, the
# > cascade with one number per stage (latency or recall), how you evaluated
# > it (pre-registration, IPS, grouped bootstrap), the headline result with its
# > interval, and one thing you would do next. Then record yourself delivering
# > it and time it. Over 3:30? Cut.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🧮 Key equations
#
# **1. Position-based click model and propensities**
#
# $$
# P(\text{click} \mid \text{item}, k) = \theta_k \cdot P(\text{relevant} \mid \text{item}), \qquad \theta_k = (1 + k)^{-\eta}
# $$
#
# *In words:* a click needs both an examination (which depends only on the position $k$) and relevance (which depends only on the item). In randomised traffic relevance is the same on average at every position, so CTR by position estimates $\theta_k$ up to a constant.
# *Example:* η = 0.9: $\theta_0 = 1$, $\theta_4 = 5^{-0.9} = 0.235$: position 5 (index 4) is examined about a quarter as often as the top.
#
# **2. Inverse propensity scoring**
#
# $$
# \widehat{\text{DCG}}_{\text{IPS}} = \sum_{i \,\in\, \text{clicked}} \frac{1}{\theta_{k_i^{\text{logged}}}} \cdot \frac{1}{\log_2(1 + r_i^{\text{new}})}, \qquad \mathbb{E}\!\left[\frac{c_i}{\theta_{k_i}}\right] = P(\text{relevant}_i)
# $$
#
# *In words:* each click is up-weighted by how unlikely it was to be seen where it was logged, then discounted by where the new ranker would put it. On average the weighting undoes position bias.
# *Example:* a click logged at index 4 ($\theta = 0.235$) counts as $1/0.235 = 4.26$ clicks; if the new ranker puts that item first ($r = 1$), it adds 4.26 to the DCG.
#
# **3. Grouped bootstrap and the design effect**
#
# $$
# \operatorname{Var}_{\text{grouped}} \approx \operatorname{Var}_{\text{iid}} \cdot \big(1 + (m - 1)\,\rho_{\text{ICC}}\big)
# $$
#
# *In words:* with $m$ correlated rows per user, a row-level bootstrap underestimates the variance by this design effect; resampling users gets it right automatically.
# *Symbols:* $m$ = rows per user, $\rho_{\text{ICC}}$ = the correlation between two rows of the same user.
# *Example:* 4 requests per user with ICC 0.2: variance × 1.6, so the honest interval is $\sqrt{1.6} = 1.26$× wider than the row-level one.
#
# **4. Paired differences**
#
# $$
# \operatorname{Var}(A - B) = \operatorname{Var}(A) + \operatorname{Var}(B) - 2\operatorname{Cov}(A, B)
# $$
#
# *In words:* scoring two systems on the same requests makes their errors correlated, and the covariance term removes the shared part.
# *Example:* unit variances with correlation 0.9: $1 + 1 - 1.8 = 0.2$ instead of 2, an interval $\sqrt{10} \approx 3.2$× narrower than comparing two separate intervals.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `AssertionError: write the pre-registration first` | ran 20.4 before 20.2 | run the notebook top to bottom |
# | "pre-registration already on file" but you changed the system | a plan from an earlier run in this runtime | that is the point: write a *new* file with a reason, never edit the old one |
# | IPS-DCG intervals much wider than the others | IPS variance: large weights at low-propensity positions | expected; clip weights (e.g. at 1/0.05) and report that you did |
# | Fitted eta far from 0.9 | too few random requests (e.g. a subsampled log) | use more days of random traffic; check clicks per position |
# | `finish_round` asserts on scores | a score like 3.5 or a missing dimension | whole numbers 1–4 for every rubric dimension |
# | Progress log lost | Colab runtime recycled | `USE_DRIVE = True` in the setup cell |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Why must the test set be touched once, and what exactly goes into a pre-registration for a ranking model?
# 2. On logged data, the old policy has higher NDCG on clicks than your new ranker. Explain why that can happen and how you would decide which is better before an A/B test.
# 3. How do you estimate position propensities, what does IPS do with them, and what are its weaknesses?
# 4. Why bootstrap by user rather than by request or impression? What happens to the interval if you get the unit wrong?
# 5. Your retrieval recall@100 is 0.63 against true relevance but 0.30 against logged clicks. Which do you report, and why do they differ?
# 6. Give your 3-minute project walkthrough. Then: what is the weakest part of your evaluation?
#
# ➡️ **Next:** Level 2 is complete. Continue with
# [Level 3 — Deep learning systems at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/README.md):
# GPUs, mixed precision, transformers and fine-tuning, with the same habits of
# measuring every claim.
