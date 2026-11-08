# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 23 · Implicit feedback & position bias
#
# **Day 3 · ~90 minutes · CPU only**
#
# Every ranking system at a large company learns from clicks, and clicks are
# not labels of relevance: users click what they *see*, and they see the top
# of the list. If you train or evaluate on raw clicks you mostly learn, and
# reward, whatever the old ranker already did. This notebook measures that
# bias on ShopStream, where the hidden truth is available, and removes most
# of it with exploration traffic and inverse propensity scoring.
#
# By the end you will be able to:
#
# 1. state the examination hypothesis and the position-based click model, and show where they bite in the logs;
# 2. explain why the logging policy's own order "wins" an offline click metric against a better ranking;
# 3. estimate position propensities from randomised (exploration) traffic and check them against the truth;
# 4. use inverse propensity scoring (IPS) to evaluate a new ranking offline, and reason about its variance and clipping;
# 5. train a click model with IPS weights and show, against ground truth, when it beats naive training and when it does not matter.
#
# **You will produce:** `artifacts/level2/propensities.json` (estimated examination probabilities by position).

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
# --- Catch up: the ShopStream logs (built and cached on first use) ----------
import time

import numpy as np
import pandas as pd

from shoprank import bias, store
from grading import ch23 as grade

t0 = time.time()
world, users, items, imps = store.load("small")
K = 10
THETA_TRUE = world.examination_prob(np.arange(K))  # hidden truth: only for checking our estimates
# Hidden truth, used ONLY to grade methods - a real log never has this column.
imps["true_p"] = world.true_click_prob(imps["user_id"].to_numpy(), imps["item_id"].to_numpy())
train = imps[imps["day"].isin(store.TRAIN_DAYS)]
valid = imps[imps["day"].isin(store.VALID_DAYS)]
print(f"{len(imps):,} impressions | {imps['request_id'].nunique():,} requests | "
      f"{imps['is_random'].mean():.1%} exploration | loaded in {time.time() - t0:.0f}s")
print(f"train {len(train):,} rows | valid {len(valid):,} rows (test week untouched)")

# %% [markdown]
# **Expected:** about 2.45 million impressions in about 245,000 requests, 5%
# of them exploration traffic; about 1.78M train and 0.33M validation rows.
# The first run generates the data (10–60 s).
#
# The column `true_p` is P(click | examined) for that (user, item): the
# simulation's ground truth. We use it the way you would use a perfect A/B
# test: to *grade* methods, never as a training feature or label (except for
# one clearly labelled "oracle" in 5.6).

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Implicit feedback** | behaviour (clicks, purchases) used as a label instead of explicit ratings | `clicked` |
# | **Examination hypothesis** | a click needs the item to be examined *and* found relevant | $P(\text{click}) = P(\text{exam}) \cdot P(\text{rel})$ |
# | **Position-based model (PBM)** | examination depends only on the position | θ_p = (1 + p)^−0.9 in ShopStream |
# | **Propensity θ_p** | the probability that position p is examined | θ_0 = 1, θ_9 ≈ 0.13 |
# | **Exploration traffic** | requests shown in random order, so relevance is flat over positions | `is_random`, 5% of requests |
# | **Inverse propensity scoring (IPS)** | weight each click by 1/θ_p to undo examination | a click at position 9 counts about 8 |
# | **Clipping** | capping IPS weights at M: less variance, more bias | cap 4, 6 or none in 5.5 |
# | **Self-normalised IPS** | divide the IPS sum by the sum of the weights | a standard variance reducer |
# | **Off-policy evaluation** | estimating a new policy's value from another policy's logs | IPS value of a re-ranking |
# | **Support problem** | items the old policy never showed cannot be evaluated | no clicks to re-weight |
# | **Oracle** | the simulation's ground truth, used only to grade | `true_p` |

# %% [markdown]
# ## 5.1 · Clicks are not relevance
#
# The **examination hypothesis**: a user clicks an item only if they examine
# it *and* find it relevant. The simplest model built on it, the
# **position-based model (PBM)**, says examination depends only on the
# position:
#
# $$P(\text{click} \mid u, i, p) = \underbrace{\theta_p}_{\text{examined?}} \cdot \underbrace{r_{u,i}}_{\text{relevant?}}$$
#
# ShopStream follows the PBM exactly (real users only approximately: they
# also stop after a satisfying click, look at images, skip ads). Look at
# click rate by position on the two kinds of traffic:
#
# * **policy** traffic: the production ranker chose the order (95% of requests);
# * **random** (exploration) traffic: the same kind of candidates, in random order.

# %%
by_pos = bias.ctr_by_position(imps)
by_pos["true_rel_policy"] = imps[~imps["is_random"]].groupby("position")["true_p"].mean()
by_pos["true_rel_random"] = imps[imps["is_random"]].groupby("position")["true_p"].mean()
by_pos["theta_true"] = THETA_TRUE
print(f"overall CTR: policy {imps.loc[~imps['is_random'], 'clicked'].mean():.3f} | "
      f"random {imps.loc[imps['is_random'], 'clicked'].mean():.3f}")
by_pos.round(3)

# %% [markdown]
# **Expected:** CTR about 0.14 on policy traffic and 0.06 on random traffic.
# Policy CTR falls from about 0.47 at position 0 to about 0.05 at position 9;
# random CTR falls from about 0.19 to about 0.02. On policy traffic the mean
# *true* relevance also falls with position (about 0.47 → 0.38: the ranker
# is not useless), but on random traffic it is flat (about 0.19 everywhere).
#
# Read the random columns together: relevance is flat, yet CTR drops by a
# factor of about 8. That drop is pure **examination**. On policy traffic the
# drop is examination *and* relevance, mixed, and from clicks alone you
# cannot tell how much is which. That is the whole problem.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-position-bias-ips.svg" alt="Examination probability by position, and the inverse-propensity weights that correct it" width="100%">
#
# *What to notice:* the left panel is the drop you just measured on random traffic (pure examination); the right panel is its mirror image, the weight IPS gives a click at each position. The dashed tops are what a clip at 4 throws away.
#
# > 💡 **Intuition.** A shop window: items at eye level get looked at, items near the floor rarely do. Counting sales per shelf tells you mostly *where* things were, not how good they are. *Tiny example:* an item with relevance 0.4 at position 0 gets clicked 40% of the time; the same item at position 9 ($\theta_9 = 0.13$) only $0.13 \times 0.4 = 5\%$. *Where it breaks down:* real shoppers also stop after finding what they wanted (a cascade), so examination of position 5 depends on what was at positions 0–4, which the PBM ignores.

# %% [markdown]
# ## 5.2 · The paradox: the old ranker wins on logged clicks
#
# The standard offline evaluation is: take logged requests, re-rank the shown
# items with your model, compute NDCG@10 against the clicks. Try it with the
# best possible "model", the true relevance itself, against the logging
# policy's own order. `bias.ndcg_matrix` is a vectorised NDCG (same formula
# as `shoprank.metrics.ndcg_at_k`; requests are rows of a 10-column matrix).

# %%
vp = valid[~valid["is_random"]]                       # validation, policy traffic
C = bias.matrix(vp["clicked"]).astype(float)           # clicks, one row per request
T = bias.matrix(vp["true_p"])                          # true relevance
P = bias.matrix(vp["position"])                        # logged positions
logged_order = -P.astype(float)                        # "score" that reproduces the logged order

rows = {}
for name, score in [("logged policy order", logged_order), ("true relevance order", T),
                    ("random order", np.random.default_rng(0).random(T.shape))]:
    rows[name] = {"NDCG@10 on clicks": bias.ndcg_matrix(score, C).mean(),
                  "NDCG@10 on true relevance": bias.ndcg_matrix(score, T).mean(),
                  "true expected clicks / request": bias.expected_clicks(score, T, THETA_TRUE)}
paradox = pd.DataFrame(rows).T
paradox.round(3)

# %% [markdown]
# **Expected:** on clicks, the logged order scores about **0.76** and the
# true-relevance order only about **0.68** (a random order about 0.53). On
# true relevance, the order is reversed (about 0.81 vs 1.00), and the true
# expected clicks per request are about 1.41 for the logged order and about
# 1.84 for the true order: a ranking by truth would earn roughly 30% more
# clicks *online*, while the offline click metric says it is worse.
#
# Why: every logged click happened *at its logged position*, and positions
# 0–2 get most of the examination. Re-ranking moves clicked items away from
# the top, so any re-ranking is penalised for disagreeing with the logging
# policy, regardless of how good it is. NDCG on random traffic is no cure
# either if you compare against the logged order (try it: logged order still
# wins there, because clicks still concentrate at the top). Two practical
# rules follow:
#
# > 💡 **Intuition.** Judging a new seating plan for a concert by the applause recorded under the old plan: the people who clapped loudest sat in the front rows, so any plan that moves them back "loses applause", however good it is. *Where it breaks down:* applause has no exact correction; clicks do, if you know how much each seat (position) is examined, which is what the rest of this notebook estimates.
#
# * never compare a new ranker against the logging policy on that policy's clicks;
# * NDCG on clicks is only (roughly) fair between two *new* rankers, and only
#   if the propensity effects are removed. That is what IPS does.

# %% [markdown]
# ## 5.3 · Estimating examination propensities from exploration traffic
#
# On random traffic, the item at position p is a uniformly random draw from
# the candidates, so E[relevance] is the same at every position. Under the
# PBM:
#
# $$\frac{\text{CTR}_\text{random}(p)}{\text{CTR}_\text{random}(0)} = \frac{\theta_p \, \bar r}{\theta_0 \, \bar r} = \frac{\theta_p}{\theta_0}$$
#
# Only *relative* propensities are identifiable (scale θ and r by inverse
# constants and nothing observable changes), so we normalise to position 0.
# IPS only needs ratios anyway.
#
# > **Exercise 1.** Write `estimate_propensities(random_imps)`: given
# > impressions from exploration traffic (columns include `position` and
# > `clicked`; rows in any order), return a NumPy array of length 10 with the
# > click rate at each position divided by the click rate at position 0.
# > The grader simulates a small world and checks your estimate against the
# > true examination probabilities.

# %%
def estimate_propensities(random_imps: pd.DataFrame) -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(estimate_propensities)

# %% [markdown]
# The notebook continues with the library version, `bias.estimate_propensities`
# (read it in `src/shoprank/bias.py`: three lines). We estimate from
# **training-week** random traffic only and compare with the truth, and with
# the naive estimate from policy traffic.

# %%
rand_train = train[train["is_random"]]
theta_raw = bias.estimate_propensities(rand_train)
naive_ctr = train[~train["is_random"]].groupby("position")["clicked"].mean().to_numpy()
theta_naive = naive_ctr / naive_ctr[0]

# Bootstrap over REQUESTS (the unit that was randomised), not impressions.
Cr = bias.matrix(rand_train["clicked"]).astype(float)
boot_rng = np.random.default_rng(0)
boot = np.array([(lambda m: m / m[0])(Cr[boot_rng.integers(0, len(Cr), len(Cr))].mean(0)) for _ in range(300)])
lo, hi = np.percentile(boot, [5, 95], axis=0)

prop = pd.DataFrame({"true": THETA_TRUE, "random traffic": theta_raw, "90% CI low": lo, "90% CI high": hi,
                     "policy traffic (wrong)": theta_naive})
prop["truth inside CI"] = (prop["true"] >= lo) & (prop["true"] <= hi)
print(f"random-traffic impressions used: {len(rand_train):,} ({len(Cr):,} requests)")
print(f"signed error, positions 1-9: random {np.round(theta_raw - THETA_TRUE, 3)[1:].tolist()}")
print(f"                             policy {np.round(theta_naive - THETA_TRUE, 3)[1:].tolist()}")
prop.round(3)

# %% [markdown]
# **Expected:** about 89,000 exploration impressions (about 8,900 requests).
# The two estimates fail in different ways:
#
# * **policy traffic** is *biased*: every position is below the truth (by
#   about 0.02–0.03 here), because the ranker puts more relevant items on top
#   and the naive ratio credits that to examination. More data will not fix
#   it. (In ShopStream the logging policy is weak, so the bias is modest; a
#   strong ranker makes it much larger.)
# * **random traffic** is *unbiased but noisy*: errors of up to about ±0.04.
#   Look at the signs: most errors are positive, and the 90% bootstrap
#   interval misses the truth at several positions. That is not a bias. Every
#   ratio shares the same denominator, CTR(0), which in this sample came out
#   about 2% low, so all the ratios moved up together. Six "misses" here are
#   essentially *one* unlucky draw of one number.
#
# The fix for the shared-denominator problem, and for the noise: fit a smooth
# curve to all ten positions at once. Examination usually decays like a power
# law, θ_p ∝ (1 + p)^(−η): fit log CTR(p) = a − η·log(1 + p) by weighted
# least squares (weights ∝ √clicks), then set θ_p = (1 + p)^(−η) so that θ_0 = 1.

# %%
x, yv = np.log1p(np.arange(K)), np.log(Cr.mean(0))
eta_hat = -np.polyfit(x, yv, 1, w=np.sqrt(Cr.sum(0)))[0]
theta_hat = (1.0 + np.arange(K)) ** -eta_hat
print(f"eta: fitted {eta_hat:.3f} | true {world.config.eta}")
print(f"max abs error: raw ratios {np.abs(theta_raw - THETA_TRUE).max():.3f} | "
      f"power-law fit {np.abs(theta_hat - THETA_TRUE).max():.3f}")
assert np.isclose(bias.power_law_propensities(rand_train)[0], eta_hat)  # the library version, used in 06-08

# %% [markdown]
# **Expected:** a fitted η of about 0.88 (true: 0.9) and a maximum error that
# drops from about 0.044 (raw ratios) to about 0.01 (fit).
# A one-parameter curve is a strong assumption, and here it happens to be the
# true model, so this is the best case: in practice check the residuals
# before trusting it. We use the fitted `theta_hat` from here on.
#
# In production you re-estimate periodically: examination depends on the UI
# (a redesign changes θ), the device and the query type, so propensities are
# often estimated per segment.
#
# Other ways to get propensities, for your toolbox (no need to implement):
# **RandPair / swap interventions** (swap two positions instead of shuffling
# all ten: cheaper for users), **intervention harvesting** (reuse the natural
# variation when several rankers were live, no extra exploration), and
# **regression EM** (fit θ and r jointly from ordinary logs, which needs
# modelling assumptions instead of randomisation).

# %%
import json

out_path = ART / "level2" / "propensities.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({"normalised_to": "position 0",
                                "estimated_from": "exploration traffic, TRAIN_DAYS; power-law fit",
                                "eta": round(float(eta_hat), 4), "theta_raw": [round(float(x), 5) for x in theta_raw],
                                "theta": [round(float(x), 5) for x in theta_hat]}, indent=2))
print("wrote", out_path.relative_to(ART))

# %% [markdown]
# ## 5.4 · Inverse propensity scoring: debiased click rates
#
# If a click at position p happens with probability θ_p · r, then
# click / θ_p has expectation r. Weighting every click by 1/θ_p turns
# "clicks we happened to observe" into "clicks we would have seen if
# everything had been examined". That is **inverse propensity scoring**.
# Its price: the rare clicks at low positions get large weights (here up to
# 1/0.126 ≈ 8), which adds variance. **Clipping** the weight at some cap M
# reduces variance and adds bias (towards under-counting low-position clicks).
#
# > 💡 **Intuition.** A survey where young people answer only 1 time in 8 and older people every time: weight each young answer by 8 and the totals describe everyone again. IPS does the same with "examined 1 time in 8". *Where it breaks down:* if a group answers 1 time in 1,000, one answer carries a weight of 1,000 and a single respondent swings the result; that is the variance clipping trades away.
#
# > **Exercise 2.** Write `ips_ctr_estimate(clicks, positions, propensities, clip)`:
# > the mean over impressions of `click * min(1 / propensities[position], clip)`
# > (no cap when `clip is None`). It estimates the click rate those
# > impressions would have had if every one had been examined like position 0.

# %%
def ips_ctr_estimate(clicks, positions, propensities, clip=None) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(ips_ctr_estimate)

# %% [markdown]
# Apply the library version to a question an analyst might ask: *how
# relevant are the items the policy shows at the bottom of the feed,
# compared with the top?* Raw CTR says the bottom is terrible; IPS says...

# %%
vt = valid[~valid["is_random"]]
rows = []
for name, part in [("positions 0-2", vt[vt["position"] <= 2]), ("positions 7-9", vt[vt["position"] >= 7])]:
    rows.append({"slice": name, "raw CTR": part["clicked"].mean(),
                 "IPS CTR": bias.ips_ctr_estimate(part["clicked"], part["position"], theta_hat),
                 "IPS CTR clip 4": bias.ips_ctr_estimate(part["clicked"], part["position"], theta_hat, clip=4),
                 "true mean relevance": part["true_p"].mean()})
pd.DataFrame(rows).set_index("slice").round(3)

# %% [markdown]
# **Expected:** raw CTR about 0.29 at the top vs 0.055 at the bottom (a
# factor of five). IPS brings both close to the true mean relevance (about
# 0.44 vs 0.45 at the top, 0.38 vs 0.385 at the bottom): the bottom items
# are worse, but by about 15%, not 80%. Clipping at 4 badly under-estimates
# the bottom slice (about 0.22), because every weight there (about 6–8)
# gets cut to 4: clipping bias is largest exactly where the weights are large.

# %% [markdown]
# ## 5.5 · IPS for offline evaluation of a new ranking
#
# Now the evaluation problem from 5.2. To estimate how many clicks a new
# ranking of the logged items would get, move each logged click to its new
# position: a click observed at logged position p, re-ranked to position q,
# counts θ_q / θ_p clicks. Summed over a request:
#
# $$\hat V(\text{new}) = \sum_{i \in \text{request}} \frac{\theta_{q_i}}{\theta_{p_i}} \, c_i$$
#
# Under the PBM this is unbiased for the true expected clicks per request,
# *for re-orderings of the same shown items*. It cannot evaluate items the
# old policy never showed (no clicks to re-weight): that is the support
# problem, and the reason two-stage systems also log exploration *candidates*.
#
# Let us evaluate five rankings of the validation requests three ways: naive
# NDCG on clicks, the IPS estimate, and the truth.

# %%
rng = np.random.default_rng(0)
item_ctr_feat = bias.click_features(train, vp, items)["item_ctr"].to_numpy()
rankings = {
    "logged policy order": logged_order,
    "reversed logged order": P.astype(float),
    "random order": rng.random(T.shape),
    "item CTR (popularity)": bias.matrix(item_ctr_feat),
    "true relevance (oracle)": T,
}
rows = {}
for name, s in rankings.items():
    rows[name] = {"naive NDCG@10 on clicks": bias.ndcg_matrix(s, C).mean(),
                  "IPS est. clicks/request": bias.ips_policy_value(s, C, P, theta_hat).mean(),
                  "TRUE expected clicks/request": bias.expected_clicks(s, T, THETA_TRUE)}
ope = pd.DataFrame(rows).T
ope["IPS error %"] = 100 * (ope["IPS est. clicks/request"] / ope["TRUE expected clicks/request"] - 1)
ope.round(3)

# %% [markdown]
# **Expected:** the IPS estimates land within about 1% of the true expected
# clicks for every ranking (errors around −0.5% to −1%), and order the five
# rankings correctly: oracle (about 1.84) > item CTR (about 1.60) > logged
# (about 1.41) > random (about 1.34) > reversed (about 1.29). Naive NDCG on
# clicks puts the **logged order first**, above item CTR and the oracle, for
# the reasons in 5.2. (Also note how weak the logging policy is: only about
# 5% more true clicks than a random order. Common, and invisible on its own
# click logs.)
#
# Note the logged order's IPS estimate equals its observed clicks per request
# exactly: all its weights are 1. IPS is a correction *relative to the
# logging policy*: the further a ranking moves from it, the bigger the weights
# and the noisier the estimate. Next, how noisy.

# %% [markdown]
# ### Variance, and the clipping trade-off
#
# Real evaluation sets are smaller than 24,000 requests (one country, one
# surface, one day). Draw 300 random subsets of 1,000 requests and estimate
# the oracle ranking's value each time, with different caps on the weight
# θ_q / θ_p.

# %%
n_req, n_rep = 1_000, 300
true_v = bias.expected_clicks(T, T, THETA_TRUE)
per_req = {clip: bias.ips_policy_value(T, C, P, theta_hat, clip=clip)
           for clip in [None, 6.0, 4.0, 2.0, 1.0]}
idx = rng.integers(0, len(T), size=(n_rep, n_req))
rows = []
for clip, v in per_req.items():
    est = v[idx].mean(axis=1)
    rows.append({"clip": "none" if clip is None else clip,
                 "mean estimate": est.mean(), "bias": est.mean() - true_v,
                 "std across subsets": est.std(), "RMSE": np.sqrt(((est - true_v) ** 2).mean())})
print(f"true value of the oracle ranking: {true_v:.3f} clicks/request")
pd.DataFrame(rows).set_index("clip").round(4)

# %% [markdown]
# **Expected:** without clipping the bias is small (about −0.02, from the
# slightly-off θ̂) and the standard deviation about 0.06 per 1,000 requests.
# A cap of 6 trims the spread a little and doubles the bias (about −0.04); a
# cap of 4 adds a bias of about −0.13; a cap of 1 ("never give a click more
# credit than it had") cuts the spread by half and is useless (bias about
# −0.8). RMSE is lowest with no clip here, because the weights are at most
# about 8. With more positions, steeper decay, or propensities near zero the
# variance term dominates and a cap pays for itself. Choose the cap on RMSE
# measured like this (simulation or bootstrap), not by habit.
#
# Two other standard variance reducers worth naming in an interview:
# **self-normalised IPS** (divide by the mean weight) and **doubly robust**
# estimators (a reward model plus an IPS correction of its residuals).

# %% [markdown]
# ## 5.6 · Training on clicks: naive vs IPS vs oracle
#
# Evaluation is half the story. A model trained on clicks also learns
# position bias, *through any feature correlated with position*. A very
# common such feature: the **previous ranker's score**, fed into the new
# model because it is available at serving time and usually helps. On policy
# traffic the policy score decides the position, so a naive click model
# learns "high policy score → clicks", partly because of examination.
#
# Setup (point-in-time, no leakage):
#
# * features for training rows (days 35–41) come from days 0–34 only;
#   features for validation rows from days 0–41 (`bias.click_features`);
# * four features: log item CTR, log user×category CTR, log price, policy score;
# * **naive**: logistic regression on clicks;
# * **IPS**: the same, with clicked rows weighted 1/θ̂_p (capped at 10);
#   un-clicked rows keep weight 1;
# * **oracle labels**: the same features, labels drawn from the *true*
#   relevance (what clicks would be if every position were examined). This
#   is the best these features can do, the ceiling for any debiasing.
#
# Why positives-only weighting? The fully unbiased pointwise loss gives a
# clicked row weight 1/θ on the positive term and 1 − 1/θ (negative!) on the
# negative term; that loss is unbounded below and unstable. Weighting only
# the positives is the common practical compromise: it removes most of the
# position effect, not all of it.

# %%
from sklearn.linear_model import LogisticRegression

pol = imps[~imps["is_random"]]
hist, fit_rows = pol[pol["day"] < 35], pol[pol["day"].between(35, 41)]
hist_all = pol[pol["day"] < 42]


X_fit = bias.click_features(hist, fit_rows, items)    # features from days 0-34 only
X_val = bias.click_features(hist_all, vp, items)      # features from days 0-41 only
print(f"fit rows {len(X_fit):,} | validation rows {len(X_val):,}")


def fit_and_score(cols, label, weight=None):
    m = LogisticRegression(max_iter=1000).fit(X_fit[cols], label, sample_weight=weight)
    return m, bias.matrix(m.decision_function(X_val[cols]))


y = fit_rows["clicked"].to_numpy()
w_ips = np.where(y, bias.ips_weights(fit_rows["position"], theta_hat, clip=10.0), 1.0)
y_oracle = np.random.default_rng(1).random(len(fit_rows)) < fit_rows["true_p"].to_numpy()

results, coefs = {}, {}
for feats_name, cols in [("with policy_score", ["item_ctr", "ucat_ctr", "log_price", "policy_score"]),
                         ("without policy_score", ["item_ctr", "ucat_ctr", "log_price"])]:
    for model_name, label, weight in [("naive", y, None), ("IPS", y, w_ips), ("oracle labels", y_oracle, None)]:
        m, s = fit_and_score(cols, label, weight)
        key = f"{model_name} | {feats_name}"
        results[key] = {"NDCG@10 truth": bias.ndcg_matrix(s, T).mean(),
                        "true exp. clicks": bias.expected_clicks(s, T, THETA_TRUE),
                        "NDCG@10 clicks (naive)": bias.ndcg_matrix(s, C).mean()}
        coefs[key] = dict(zip(cols, m.coef_[0].round(3)))
results["logged policy order"] = {"NDCG@10 truth": paradox.iloc[0, 1], "true exp. clicks": paradox.iloc[0, 2],
                                  "NDCG@10 clicks (naive)": paradox.iloc[0, 0]}
display(pd.DataFrame(results).T.round(4))
pd.DataFrame(coefs).T

# %% [markdown]
# **Expected (the punchline):** with the policy score as a feature, the
# naive click model reaches NDCG@10 on truth of about **0.90** and about 1.63
# true expected clicks, the IPS model about **0.94** and 1.72, the oracle
# labels about **0.95** and 1.73 (the logged order: 0.81 and 1.41). IPS
# recovers about 80% of the gap between naive and oracle. The coefficient
# table shows why: the naive model puts a weight of about 0.84 on
# `policy_score`, IPS about 0.28, the oracle about 0.18. The naive model has
# learned "position", not relevance.
#
# Without the policy score, all three are within 0.0002 of each other
# (about 0.948): the remaining features are only weakly tied to position,
# so the reweighting changes the coefficients but not the ordering. (Note
# also that the policy score does not help even the oracle here: this
# logging policy knows little the other features do not.)
# **Position bias hurts most through features that encode the old ranking.**
# And, once more, the last column: on naive click-NDCG the *naive* model
# looks best (about 0.77, above even the logged order) and the oracle-label
# model looks worst of the three. Evaluate a debiased model with a debiased
# metric.
#
# Production alternatives to sample weights, for your interview vocabulary:
# **position as a training feature** (set to a constant at serving; notebook
# 06), a separate **"position tower"** whose output is added during training
# and dropped at serving (the YouTube "shallow tower" pattern), and unbiased
# pairwise/listwise losses (IPS-weighted LambdaMART).

# %% [markdown]
# > **Exercise 3.** Exploration traffic shows 5% of requests in random order.
# > Using this notebook's numbers, estimate what it costs (clicks lost per
# > day relative to policy traffic), and write the short case you would make
# > to a product manager for keeping it. Name at least one way to make it
# > cheaper. Write your answer in the cell below.

# %% [markdown]
# *Your answer:*

# %% [markdown]
# ## 🧮 Key equations
#
# **1 · The position-based click model.** In words: a click happens only if the position is examined *and* the item is relevant, and the two are independent.
#
# $$
# P(c = 1 \mid u, i, p) = \underbrace{\theta_p}_{P(\text{exam} \mid p)} \cdot \underbrace{r_{u,i}}_{P(\text{rel} \mid u, i)}
# $$
#
# * $c$: click; $p$: position (0 = top); $\theta_p$: examination propensity; $r_{u,i}$: relevance. *Example:* $r = 0.4$, $\theta_0 = 1$ → 0.40; $\theta_9 = 10^{-0.9} = 0.126$ → 0.05.
#
# **2 · Propensities from random traffic.** In words: with random order, mean relevance is the same at every position, so CTR ratios are examination ratios.
#
# $$
# \frac{\theta_p}{\theta_0} = \frac{\text{CTR}_\text{rand}(p)}{\text{CTR}_\text{rand}(0)}
# $$
#
# * *Example:* random-traffic CTR 0.19 at position 0 and 0.024 at position 9 → $\theta_9 / \theta_0 \approx 0.126$.
#
# **3 · The IPS click-rate estimator, and clipping.** In words: divide each click by the chance that its position was examined; optionally cap the weight at $M$.
#
# $$
# \hat r_\text{IPS} = \frac{1}{n} \sum_{j=1}^{n} \frac{c_j}{\theta_{p_j}}, \qquad \hat r_\text{clip} = \frac{1}{n} \sum_{j=1}^{n} c_j \min\!\left(\frac{1}{\theta_{p_j}}, M\right)
# $$
#
# * Unbiased under the PBM, because $\mathbb{E}[c / \theta_p] = \theta_p r / \theta_p = r$. Clipping lowers variance and biases the estimate down where $1/\theta_p > M$. *Example:* 5 clicks in 100 impressions at position 9: raw CTR 0.05, IPS $0.05 / 0.126 = 0.40$, clipped at 4: $0.05 \times 4 = 0.20$.
#
# **4 · Off-policy value of a re-ranking.** In words: a click observed at logged position $p$ that the new ranking moves to position $q$ counts $\theta_q / \theta_p$ clicks.
#
# $$
# \hat V(\text{new}) = \sum_{i \in \text{request}} \frac{\theta_{q_i}}{\theta_{p_i}}\, c_i
# $$
#
# * *Example:* a click at logged position 3 ($\theta = 0.287$) moved to position 0 counts $1 / 0.287 = 3.5$ clicks; moved to position 9 it counts $0.126 / 0.287 = 0.44$. Items never shown have no $c_i$: IPS cannot evaluate them.
#
# The cell below simulates the PBM for one item and checks equation 3: the raw click rate depends on where the item was shown, IPS does not.

# %%
# Simulate the PBM: one item with relevance r = 0.3 shown at random positions
rng_pbm = np.random.default_rng(0)
theta_true = (1.0 + np.arange(10)) ** -0.9
pos = rng_pbm.integers(0, 10, size=200_000)
clicks_sim = rng_pbm.random(pos.size) < theta_true[pos] * 0.3
w = 1 / theta_true[pos]
print(f"raw CTR            {clicks_sim.mean():.3f}   (true relevance 0.300)")
print(f"IPS                {(clicks_sim * w).mean():.3f}")
print(f"IPS clipped at 4   {(clicks_sim * np.minimum(w, 4)).mean():.3f}")

# %% [markdown]
# **Expected:** raw CTR about 0.097 (the mean of θ_p is 0.32, so examination hides two thirds of the relevance), IPS about 0.300, clipped about 0.24 (biased down). With 200,000 impressions the IPS estimate is within about ±0.005 of the truth; rerun with 2,000 and watch its spread grow.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `estimate_propensities` gives values above 1 or a non-monotone curve | estimated on policy traffic, or on too few requests | filter `is_random`; use all training days; bootstrap to see the noise |
# | IPS estimates explode or flip sign between runs | tiny propensities (huge weights), or too few requests | clip weights; pool more data; use self-normalised IPS |
# | `ValueError: cannot reshape array` in `bias.matrix` | the frame is not exactly 10 rows per request, or not sorted by request and position | filter whole requests (by `request_id`), keep the order `store.load` returns |
# | Your "debiased" model is no better on clicks NDCG | the metric is biased too | evaluate with IPS estimates or (here) the true relevance |
# | NDCG on truth barely moves between models | features without position-correlated signal: reweighting cannot change the ordering much | look at coefficients/feature importance; check which features correlate with position |
# | `KeyError: 'true_p'` | the catch-up cell was not run | run the catch-up cell |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. State the examination hypothesis and the position-based model. Name two ways real user behaviour violates the PBM, and a click model that handles one of them.
# 2. Your new ranker loses to production on offline NDCG computed from production's click logs, then wins the A/B test. Explain how both can be true.
# 3. How would you estimate position propensities at a company that has never run exploration traffic, and what does each option cost?
# 4. Write the IPS estimator for the clicks a new ranking would get. When is it unbiased, and what can it never evaluate?
# 5. Clipping IPS weights: what does it trade, and how would you choose the cap?
# 6. A colleague adds "the previous model's score" as a feature and offline AUC on clicks jumps. What is your concern, and how would you check it?
#
# ➡️ **Next:** [`06_learning_to_rank.ipynb`](../ch24-learning-to-rank/notebook.ipynb) — from pointwise click models to learning-to-rank objectives (LambdaMART), with point-in-time features and request-grouped evaluation.
