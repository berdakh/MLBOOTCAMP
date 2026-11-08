# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 7 · Model selection
#
# **Day 2 · ~45 minutes · CPU core · optional GPU extension**
#
# "Model B scores 0.14 and model A scores 0.16" is not a finding until you know
# how much those numbers move when the sample moves. This notebook compares a
# baseline, a linear model and boosted trees with rolling-origin CV and
# bootstrap intervals, runs a hyperparameter search on a fixed budget, and
# ends with a written selection decision you could defend in a review.
#
# By the end you will be able to:
#
# 1. compare candidates with rolling-origin CV on train and then once on validation;
# 2. put a bootstrap confidence interval on a metric, and choose rows or customers as the resampling unit for a reason;
# 3. decide between two models with a *paired* bootstrap of the difference;
# 4. run a budgeted random search and keep every trial as an experiment record;
# 5. explain why boosted trees do not win on this data, and when they would.
#
# **You will produce:** `artifacts/trials.csv` (every search trial) and
# `artifacts/selection.json` (the selection memo).
#
# Everything here uses **train** and **validation** only. The test period is
# reserved for notebook 15.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: the feature table and the fixed time split (notebooks 03-04) --
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_validate

from churnlab import bootstrap
from churnlab.evaluation import bootstrap_ci, paired_bootstrap_diff
from churnlab.features import FEATURES, LABEL
from grading import ch07 as grade
from churnlab.modeling import make_model, predict_proba
from churnlab.splits import rolling_origin_folds, time_split

T0 = time.time()
split = time_split(bootstrap.ensure_features())
train, valid = split.train, split.valid
folds = list(rolling_origin_folds(train, n_folds=4))
print(f"train {len(train):,} rows, {int(train[LABEL].sum())} positives | "
      f"valid {len(valid):,} rows, {int(valid[LABEL].sum())} positives | {len(folds)} rolling folds")

# %% [markdown]
# **Expected:** about 1,380 positives in train and about 360 in validation.
# Keep the second number in mind: every validation metric below is driven by
# a few hundred positive rows.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Candidate** | one model configuration being compared | `dummy`, `logreg`, `hgb`, `hgb_deeper` |
# | **Bootstrap** | resample the evaluation rows with replacement many times and recompute the metric | 500 resamples of the validation set |
# | **Confidence interval (CI)** | a range that would contain the true value in about 95% of repeated experiments | PR-AUC 0.163, CI ≈ [0.137, 0.197] |
# | **Resampling unit** | what the bootstrap draws: rows, or whole groups (customers) | customers are the default here |
# | **Cluster bootstrap** | draws whole groups with all their rows | `groups=customer_id` |
# | **Paired comparison** | both models scored on the same resample, then the difference is taken | `paired_bootstrap_diff` |
# | **Hyperparameter** | a setting chosen before fitting, not learned from data | `C`, `max_leaf_nodes` |
# | **Search budget** | the number of configurations you allow yourself, fixed in advance | 8 per model family |
# | **Plateau** | a region where many settings score the same within noise | logreg for `C` from 0.003 to 10 |
# | **Selection memo** | the written decision, with the evidence that supports it | `artifacts/selection.json` |

# %% [markdown]
# ## 6.1 · Four candidates, two estimates each
#
# * `dummy`: predicts the base rate for everyone. Its PR-AUC *is* the positive
#   rate, the floor every model must clear.
# * `logreg`: the logistic-regression pipeline from notebook 05.
# * `hgb`: `HistGradientBoostingClassifier` with the course's conservative
#   defaults (8 leaves, at least 200 rows per leaf, learning rate 0.03).
# * `hgb_deeper`: the more typical out-of-the-box setting: 31 leaves, 50 rows
#   per leaf, faster learning, no L2 penalty.
#
# Each gets a rolling-origin CV score on train (for choosing) and one
# validation score (for checking the choice).

# %%
CANDIDATES = {
    "dummy": ("dummy", {}),
    "logreg": ("logreg", {}),
    "hgb": ("hgb", {}),
    "hgb_deeper": ("hgb", dict(max_leaf_nodes=31, min_samples_leaf=50, learning_rate=0.05,
                               max_iter=300, l2_regularization=0.0)),
}

rows, valid_scores, fitted = [], {}, {}
for key, (name, params) in CANDIDATES.items():
    t = time.time()
    cv = cross_validate(make_model(name, **params), train[FEATURES], train[LABEL],
                        cv=folds, scoring="average_precision")["test_score"]
    fitted[key] = make_model(name, **params).fit(train[FEATURES], train[LABEL])
    valid_scores[key] = predict_proba(fitted[key], valid)
    rows.append({"candidate": key, "cv_mean": cv.mean(), "cv_std": cv.std(),
                 "cv_folds": np.round(cv, 3).tolist(),
                 "valid_pr_auc": average_precision_score(valid[LABEL], valid_scores[key]),
                 "seconds": round(time.time() - t, 1)})
comparison = pd.DataFrame(rows).set_index("candidate").round(4)
comparison

# %% [markdown]
# **Expected (our run):** dummy ≈ 0.054 CV / 0.052 validation; logreg ≈ 0.186 /
# 0.163; hgb ≈ 0.156 / 0.135; hgb_deeper ≈ 0.145 / 0.11-0.12 (tree numbers
# move in the third decimal between sklearn versions). The linear model is
# ahead on **every** rolling fold, and the deeper trees are worse than the
# regularised ones, which is what overfitting looks like with ~1,400 positives.

# %% [markdown]
# ## 6.2 · How uncertain is a validation PR-AUC?
#
# `churnlab.evaluation.bootstrap_ci` resamples the validation set with
# replacement, recomputes the metric each time, and reports the 2.5th and
# 97.5th percentiles. Two choices of resampling unit:
#
# * **rows**: treats the 6,956 rows as independent;
# * **customers** (`groups=customer_id`): resamples whole customers with all
#   their rows. If a customer's rows are correlated, this is the honest unit.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-bootstrap-units.svg" alt="Row bootstrap draws individual rows and mixes customers; cluster bootstrap draws whole customers with all their rows" width="100%">
#
# *What to notice:* in the customer bootstrap a resample can contain customer A twice and B not at all, exactly as a new sample of customers would; in the row bootstrap, A's three near-identical rows count as three independent pieces of evidence.
#
# > **Intuition.** The bootstrap asks "if I had collected a different
# > validation set of the same size from the same world, how much would my
# > number move?", and answers by pretending the sample *is* the world.
# > *Where it breaks down:* it only sees variation that is in the sample. It
# > cannot tell you how the score moves next quarter, when the world changes.

# %%
y = valid[LABEL].to_numpy()
cust = valid["customer_id"].to_numpy()
ci_rows = []
for key in ["logreg", "hgb"]:
    for unit, groups in [("rows", None), ("customers", cust)]:
        point, lo, hi = bootstrap_ci(y, valid_scores[key], n_boot=500, seed=0, groups=groups)
        ci_rows.append({"model": key, "unit": unit, "pr_auc": point, "ci_low": lo, "ci_high": hi,
                        "width": hi - lo})
ci_table = pd.DataFrame(ci_rows).round(4)
ci_table

# %% [markdown]
# **Expected (our run):** logreg 0.163 with a customer interval of roughly
# [0.137, 0.197]; hgb 0.135 with roughly [0.115, 0.164]. The intervals are
# about **0.05–0.07 wide**, a third of the metric itself.
#
# Now look at the `width` column: on **this** validation set the customer
# interval is **not** noticeably wider than the row interval (on our run they
# differed by a few thousandths, in both directions across seeds). Measure,
# don't assume. Why so little difference here?

# %%
per_customer = valid.groupby("customer_id").agg(rows=(LABEL, "size"), positives=(LABEL, "sum"))
print("rows per customer in validation:", per_customer["rows"].value_counts().sort_index().to_dict())
print("positives per customer        :", per_customer["positives"].value_counts().sort_index().to_dict())

# %% [markdown]
# **Expected:** at most 3 rows per customer (three monthly snapshots), and no
# customer with more than one positive. Churn is terminal, so the positives,
# which drive PR-AUC, are already one per customer; there is little
# within-customer correlation left for the cluster bootstrap to account for.
#
# The cluster bootstrap matters when a unit contributes **many correlated
# rows**. A quick way to see the mechanism: pretend every validation row was
# recorded three times (as if we had weekly snapshots of nearly identical
# customers).

# %%
rep = 3
y_rep, s_rep = np.repeat(y, rep), np.repeat(valid_scores["logreg"], rep)
g_rep = np.repeat(np.arange(len(y)), rep)  # each original row is one "customer"
_, lo_r, hi_r = bootstrap_ci(y_rep, s_rep, n_boot=300, seed=0)
_, lo_c, hi_c = bootstrap_ci(y_rep, s_rep, n_boot=300, seed=0, groups=g_rep)
print(f"rows tripled - row bootstrap width    : {hi_r - lo_r:.4f}   (pretends we have 3x the evidence)")
print(f"rows tripled - cluster bootstrap width: {hi_c - lo_c:.4f}   (same as the original data)")

# %% [markdown]
# **Expected:** the row interval shrinks by about √3 (to ≈ 0.04), the cluster
# interval stays at the original ≈ 0.06–0.07. Copies are not new evidence; the
# row bootstrap cannot tell. On the train period (13 snapshots per customer)
# or with weekly snapshots, this is the difference between "clearly better"
# and "can't tell". So the course **reports the customer-level interval by
# default**: it is never the wrong unit here, and you do not have to argue
# about whether the correlation is small this time.
#
# > **Research bridge.** Rows are trials, customers are subjects. You
# > would not report a cross-subject EEG result with a confidence interval that
# > resamples trials, for exactly this reason.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 2.6))
cust_ci = ci_table[ci_table["unit"] == "customers"].reset_index(drop=True)
for i, r in cust_ci.iterrows():
    ax.errorbar(r["pr_auc"], i, xerr=[[r["pr_auc"] - r["ci_low"]], [r["ci_high"] - r["pr_auc"]]],
                fmt="o", capsize=4)
ax.axvline(y.mean(), color="grey", ls="--", label=f"positive rate {y.mean():.3f}")
ax.set_yticks(range(len(cust_ci)))
ax.set_yticklabels(cust_ci["model"])
ax.set_ylim(-0.6, len(cust_ci) - 0.4)
ax.set_xlabel("validation PR-AUC (95% customer-bootstrap CI)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.3 · Is logreg really better? Ask the paired question
#
# The two intervals above overlap. That does **not** mean "no difference":
# both models are scored on the *same* rows, so most of the sampling noise is
# shared and cancels in the difference.
#
# > **Intuition.** Two students sit the same exam. Whether the exam was easy
# > moves both their marks; the *difference* between them barely depends on
# > it. Comparing their marks on the same paper is the paired question;
# > comparing their two separate "how good is this student" ranges is not.
# > *Where it breaks down:* pairing removes shared noise only; if both models
# > share a bias (e.g. the same leaky feature), the paired test cannot see it. `paired_bootstrap_diff` computes
# `metric(B) - metric(A)` on each resample and gives an interval for the
# difference itself.

# %%
paired = {
    unit: paired_bootstrap_diff(y, valid_scores["hgb"], valid_scores["logreg"],  # A = hgb, B = logreg
                                n_boot=500, seed=0, groups=groups)
    for unit, groups in [("rows", None), ("customers", cust)]
}
pd.DataFrame(paired).T.round(4)

# %% [markdown]
# **Expected (our run):** logreg − hgb ≈ +0.028 PR-AUC (+0.029 with sklearn
# 1.6), 95% interval roughly [+0.004, +0.052] with customers as the unit; about 1% of customer
# resamples (2% of row resamples) have hgb ahead or level. The interval excludes zero, but its lower end is close to it.
# Honest wording: *"on validation, logistic regression ranks better than the
# boosted trees by about 0.03 PR-AUC (customer-bootstrap 95% CI 0.004 to
# 0.05); it also won all four rolling-origin folds on train."* Two independent
# pieces of evidence pointing the same way are worth more than either alone.

# %% [markdown]
# > **Exercise 1.** Write `decide(y, scores_a, scores_b, groups)` that returns
# > `"A"`, `"B"` or `"no clear winner"` using a **paired, cluster** bootstrap 95%
# > interval for PR-AUC(B) − PR-AUC(A): `"B"` if the whole interval is above 0,
# > `"A"` if it is entirely below 0, otherwise `"no clear winner"`. You may call
# > `paired_bootstrap_diff` or write the loop yourself. The grader uses
# > synthetic clustered data with one clear winner each way and one tie.

# %%
def decide(y, scores_a, scores_b, groups):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(decide)

# %% [markdown]
# ## 6.4 · A search on a budget, recorded as experiments
#
# Hyperparameter search is where most compute is wasted and most
# self-deception happens: try 200 configurations, report the best one, and
# the "improvement" is partly the maximum of 200 noisy numbers. Two
# disciplines:
#
# 1. a **fixed budget**, decided before you look (here 8 configurations per
#    model family, under a minute in total);
# 2. **every trial recorded** (parameters, fold scores, time), not just the
#    winner. That table *is* experiment tracking; MLflow stores the same thing.
#
# Scored with rolling-origin CV on train only. Validation is not used for the
# search.

# %%
from sklearn.model_selection import ParameterSampler

search_folds = list(rolling_origin_folds(train, n_folds=3))  # 3 folds keeps the budget < 1 min


def cv_pr_auc(name, params):
    s = cross_validate(make_model(name, **params), train[FEATURES], train[LABEL],
                       cv=search_folds, scoring="average_precision")["test_score"]
    return s.mean(), s.std()


hgb_space = {
    "learning_rate": [0.02, 0.03, 0.05, 0.1],
    "max_leaf_nodes": [4, 8, 16, 31],
    "min_samples_leaf": [50, 100, 200, 400],
    "l2_regularization": [0.0, 1.0, 5.0],
    "max_iter": [100, 200],
}
logreg_space = {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}

trials = []
t_search = time.time()
for family, space, n_iter in [("hgb", hgb_space, 8), ("logreg", logreg_space, 8)]:
    for params in ParameterSampler(space, n_iter=n_iter, random_state=0):
        t = time.time()
        mean, std = cv_pr_auc(family, params)
        trials.append({"family": family, **params, "cv_pr_auc": mean, "cv_std": std,
                       "seconds": round(time.time() - t, 2)})
trials_df = pd.DataFrame(trials)
trials_df.to_csv(ART / "trials.csv", index=False)
print(f"{len(trials_df)} trials in {time.time() - t_search:.0f} s -> {ART / 'trials.csv'}")
cols = ["family", "C", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization",
        "max_iter", "cv_pr_auc", "cv_std", "seconds"]
trials_df = trials_df[cols]
trials_df.sort_values("cv_pr_auc", ascending=False).groupby("family").head(4).round(4)

# %% [markdown]
# **Expected:** 16 trials in under a minute (22 s on our run). Every
# logistic-regression trial scores between ≈ 0.191 and ≈ 0.193, for `C` from
# 0.003 to 10: the linear model is insensitive to regularisation here (a
# plateau). The best tree configuration lands around 0.166, below every `C`.
# These numbers are higher than the 4-fold numbers in 6.1 because the three
# search folds are different months; compare trials only with each other.

# %%
best = trials_df.loc[trials_df.groupby("family")["cv_pr_auc"].idxmax()]
best.round(4)

# %% [markdown]
# Note the `cv_std` column: the spread across folds (≈ 0.02) is larger than the
# differences between most configurations.
#
# > **Intuition.** The best of 16 noisy scores is like the best of 16 dice
# > rolls: it is high partly because you rolled 16 times. Expect the winner
# > to score a little lower on fresh data. *Where it breaks down:* if one
# > configuration really is better by more than the noise, it will still win
# > on fresh data; the optimism shrinks the gap, it does not invent it. Picking the single top trial is
# fine; *claiming* it is better than its neighbours is not.

# %% [markdown]
# > **Exercise 2.** Write `budgeted_search(param_grid_list, evaluate_fn, budget)`
# > that evaluates at most `budget` configurations drawn at random **without
# > replacement** from the list `param_grid_list` (each a dict), calling
# > `evaluate_fn(params) -> score` (higher is better) once per configuration.
# > Return `(best_params, trials_df)` where `trials_df` has one row per
# > evaluated configuration: one column per hyperparameter plus `score`. Use a
# > fixed seed. The grader counts how many times `evaluate_fn` is called.

# %%
def budgeted_search(param_grid_list, evaluate_fn, budget):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(budgeted_search)

# %% [markdown]
# ## 6.5 · Why the trees lose here, and when they would not
#
# This is the honest finding of the day: **gradient-boosted trees, the default
# winner on tabular benchmarks, lose to logistic regression on this data**, on
# every rolling fold and on validation. The reasons are structural, not luck:
#
# 1. **The signal is smooth and roughly monotone.** Churn risk rises steadily
#    as usage falls, tickets rise, payments fail. A linear model in
#    well-chosen transforms (`log1p`, ratios such as `sessions_trend` and
#    `seat_utilization_4w`) already has the right shape. Trees approximate a
#    smooth slope with steps and need data to place every step.
# 2. **Few positives.** About 1,400 positive rows in train. Every leaf of a
#    tree estimates a rate from the rows that land in it; with rare positives,
#    those estimates are noisy and deeper trees overfit (compare `hgb` with
#    `hgb_deeper`).
# 3. **The features are already engineered.** The non-linearity a tree would
#    discover (trends, utilisation, recency) was put into the SQL in notebook
#    03. What is left is mostly additive.
# 4. **The data is synthetic.** It was generated from a smooth risk score,
#    which is the friendliest possible setting for a linear model. Real churn
#    data usually has more interactions.
#
# When would you still choose trees? With **interactions** that matter (a
# usage drop means something different on the enterprise plan), **thresholds**
# in the response (a cliff at contract renewal), many raw un-engineered
# features, missing values with meaning, and **more data**, especially more
# positives. Then re-run this comparison; do not assume the result.
#
# And the linear model has practical advantages that count in a tie:
# coefficients a product team can read, monotone behaviour that is easy to
# test, tiny artifacts, microsecond latency, and calibrated probabilities out
# of the box (notebook 07 checks that).

# %% [markdown]
# ## 6.6 · The selection memo
#
# A decision nobody wrote down gets re-litigated every month. The memo below is
# short on purpose, and it is saved next to the trials.

# %%
best_lr = trials_df[trials_df["family"] == "logreg"].sort_values("cv_pr_auc", ascending=False).iloc[0]
logreg_ci = ci_table.query("model == 'logreg' and unit == 'customers'").iloc[0]
selection = {
    "selected_model": "logreg",
    "params": {"C": 1.0},
    "decided_on": "rolling-origin CV on the train period (2023-03..2024-03), checked once on validation (2024-04..2024-06)",
    "evidence": {
        "cv_pr_auc": {k: round(float(comparison.loc[k, "cv_mean"]), 4) for k in comparison.index},
        "valid_pr_auc": {k: round(float(comparison.loc[k, "valid_pr_auc"]), 4) for k in comparison.index},
        "valid_pr_auc_ci_customers": [round(float(logreg_ci["ci_low"]), 4), round(float(logreg_ci["ci_high"]), 4)],
        "paired_diff_logreg_minus_hgb": {k: round(float(v), 4) for k, v in paired["customers"].items()},
        "search": {"trials": int(len(trials_df)),
                   "best_logreg_C": float(best_lr["C"]),
                   "best_logreg_cv": round(float(best_lr["cv_pr_auc"]), 4),
                   "best_hgb_cv": round(float(trials_df.query("family == 'hgb'")["cv_pr_auc"].max()), 4)},
    },
    "reasoning": ("Linear model wins every rolling fold and validation; the paired customer-bootstrap CI of the "
                  "difference excludes zero. C is kept at 1.0 because the search found no C that beats it by more "
                  "than the fold-to-fold spread. Simpler, faster, interpretable, calibrated."),
    "revisit_when": "more positives, new raw features with interactions, or a drop in monitored PR-AUC",
    "test_set_used": False,
}
(ART / "selection.json").write_text(json.dumps(selection, indent=2))
print(json.dumps(selection, indent=2))

# %% [markdown]
# **Expected:** `selected_model: logreg` with `C = 1.0`, and every number in
# the memo traceable to a cell above. Keeping `C = 1.0` rather than the
# search's top `C` is deliberate: the search found a plateau, not a peak, and a
# change smaller than the noise is not worth a new version.
#
# `test_set_used: false` is a promise to your future self. Notebook 15 reads
# the test period exactly once.

# %% [markdown]
# ## 6.7 · Optional GPU extension: a small MLP
#
# *Not required, and nothing later depends on it.* A two-layer PyTorch MLP on
# the same preprocessed features, to check whether a neural network changes
# the picture. It runs on CPU too (about a minute). **It was not executed in
# our verification runs** (PyTorch is not installed in either test
# environment), so no result is claimed here. If you run it, compare its
# validation PR-AUC against logreg with the paired bootstrap from 6.3, and add
# it to the trials table. Expect it to tie at best, for the reasons in 6.5.

# %%
RUN_GPU_EXTENSION = False  # set True to run; needs `pip install torch`

if RUN_GPU_EXTENSION:
    import torch
    from torch import nn

    from churnlab.modeling import make_preprocessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    pre = make_preprocessor("linear").fit(train[FEATURES])

    def to_tensor(df):
        a = pre.transform(df[FEATURES])
        a = a.toarray() if hasattr(a, "toarray") else a
        return torch.tensor(np.asarray(a, dtype=np.float32), device=device)

    Xt, Xv = to_tensor(train), to_tensor(valid)
    yt = torch.tensor(train[LABEL].to_numpy(np.float32), device=device)
    net = nn.Sequential(nn.Linear(Xt.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(30):
        net.train()
        for idx in torch.randperm(len(Xt), device=device).split(512):
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
    print("device:", device, "| MLP validation PR-AUC:", round(float(average_precision_score(y, p_mlp)), 4))
    print(paired_bootstrap_diff(y, p_mlp, valid_scores["logreg"], groups=cust))
else:
    print("GPU extension skipped (RUN_GPU_EXTENSION = False)")

# %% [markdown]
# > **Exercise 3 (written).** In an interview: *"Why didn't you use XGBoost?"*
# > Write the answer you would give in 60 seconds. Use numbers from this
# > notebook, say what you *did* try, and say what would make you change your
# > mind. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %%
print(f"notebook compute time: {time.time() - T0:.0f} s")

# %% [markdown]
# **Expected:** about a minute on a laptop; up to two on a free Colab CPU.

# %% [markdown]
# ## Key equations
#
# **Bootstrap percentile interval.** Draw $B$ resamples (of rows or of whole
# customers) with replacement, recompute the metric $\hat\theta^{*}_b$ on each:
#
# $$
# \text{CI}_{95\%} = \left[\, Q_{0.025}\big(\hat\theta^{*}_{1..B}\big),\; Q_{0.975}\big(\hat\theta^{*}_{1..B}\big) \,\right]
# $$
#
# *In words:* the middle 95% of the metric across resamples. *Symbols:* $Q_q$ =
# the $q$-quantile, $B$ = number of resamples (500 here). *Example:* if the
# sorted 500 resampled PR-AUCs have 0.137 at position 13 and 0.197 at
# position 488, the interval is [0.137, 0.197].
#
# **Paired difference.** On each resample $b$ compute
# $\Delta^{*}_b = \text{PR-AUC}_B - \text{PR-AUC}_A$ on the *same* rows, then
# take its percentile interval. *Example:* logreg − hgb ≈ +0.028 with
# interval [+0.004, +0.052]: excludes 0, so "logreg ranks better", narrowly.
#
# **Standard error of a proportion** (precision, a churn rate):
#
# $$
# \operatorname{SE}(\hat p) = \sqrt{\frac{\hat p\,(1 - \hat p)}{n}},
# \qquad
# n \approx \frac{1.96^2\, p\,(1-p)}{m^2}
# $$
#
# *In words:* the noise in a rate shrinks with $\sqrt{n}$; the second formula
# is the sample size for a 95% margin of $\pm m$. *Symbols:* $\hat p$ =
# observed rate, $n$ = number of trials (calls, rows), $m$ = target margin.
# *Example:* precision 0.27 over $n = 192$ calls: $\text{SE} = 0.032$, so
# roughly $0.27 \pm 0.063$. To pin a 5% churn rate to $\pm 0.02$ you need
# $n \approx 3.84 \times 0.0475 / 0.0004 \approx 456$ rows.
#
# **Why copies do not help.** Repeating every row $k$ times divides the
# row-bootstrap SE by $\sqrt{k}$ (a fake gain) and leaves the customer-bootstrap
# SE unchanged. *Example:* $k = 3$: the row interval shrinks by
# $\sqrt{3} = 1.73$, from ≈ 0.06 to ≈ 0.035-0.04.

# %%
# The SE formula vs a bootstrap of the same 192 calls, and a percentile CI by hand.
rng_toy = np.random.default_rng(0)
calls = (rng_toy.random(192) < 0.27).astype(int)          # one sample of 192 call outcomes
p_hat = calls.mean()
boot = np.array([rng_toy.choice(calls, size=calls.size, replace=True).mean() for _ in range(4000)])
print(f"p_hat {p_hat:.3f} | SE formula {np.sqrt(p_hat * (1 - p_hat) / calls.size):.4f} | bootstrap sd {boot.std():.4f}")
print("95% percentile CI:", np.round(np.percentile(boot, [2.5, 97.5]), 3))
print(f"rows for +/-0.02 on a 5% rate: {1.96**2 * 0.05 * 0.95 / 0.02**2:.0f}")

# %% [markdown]
# **Expected:** a bootstrap standard deviation within a few thousandths of the
# formula (both ≈ 0.03), an interval about ±0.06 wide around $\hat p$, and 456 rows.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | The search cell takes minutes | boosted trees fighting over CPU threads on a shared machine | set `OMP_NUM_THREADS=1` before starting Python, or lower `n_iter`; the budget is the point, not the size |
# | `ValueError: quantiles` / empty list inside `bootstrap_ci` | every resample had a single class (a tiny slice with almost no positives) | do not bootstrap slices with a handful of positives; report "too small" |
# | Customer and row intervals are identical to many decimals | you passed `groups=None` both times | pass `groups=valid["customer_id"]` for the cluster version |
# | Your `decide` says "A" when B is clearly better | the difference is computed as A − B, or A and B swapped in `paired_bootstrap_diff` (it returns B − A) | check the sign convention on a case you can verify by eye |
# | Different trial results each run | `ParameterSampler` without `random_state` | fix the seed; record it in the trials table |
# | `ModuleNotFoundError: torch` | you set `RUN_GPU_EXTENSION = True` without installing PyTorch | `pip install torch`, or leave the extension off |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Model B's validation PR-AUC is 0.03 higher than model A's. What do you need before you call B better?
# 2. Why is the customer, not the row, the default resampling unit here, and why did it barely change the interval on this validation set?
# 3. Two 95% intervals overlap. Does that mean there is no significant difference? What do you compute instead?
# 4. Why did logistic regression beat gradient boosting on this data? Name a dataset property that would reverse it.
# 5. You ran 16 configurations and report the best CV score. Why is that number optimistic, and what protects you from fooling yourself?
#
# **Next chapter:** [`07_thresholds_and_calibration.ipynb`](../ch08-thresholds-and-calibration/notebook.ipynb) — turn the selected model's scores into a decision: a threshold from stated costs, a call capacity, and probabilities you can trust.
