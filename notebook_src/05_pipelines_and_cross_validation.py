# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 05 · Pipelines & cross-validation
#
# **Day 2 · ~50 minutes · CPU only**
#
# A model that is evaluated with one preprocessing path and served with
# another is two different models, and only one of them was measured. This
# notebook puts every learned transformation inside one `Pipeline` object,
# measures how much the classic leaks actually cost on this data (one is tiny,
# one is enormous), and replaces shuffled k-fold with cross-validation that
# respects time.
#
# By the end you will be able to:
#
# 1. explain why imputers, scalers and encoders belong *inside* the model object;
# 2. measure a preprocessing leak and a target-encoding leak instead of guessing their size;
# 3. read a `ColumnTransformer`: which columns go where, what comes out, what happens to an unseen category;
# 4. cross-validate inside the training period with rolling-origin folds and compare that estimate with shuffled k-fold and with the real validation score.
#
# **You will produce:** `artifacts/cv_comparison.csv`, the three estimates side by side.
#
# All work here uses the **train** and **validation** periods only. The test
# period is reserved for notebook 15 and is not touched.

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
import time

import numpy as np
import pandas as pd

from churnlab import bootstrap
from churnlab.features import CATEGORICAL, FEATURES, LABEL, NUMERIC
from churnlab.grading import nb05 as grade
from churnlab.splits import time_split

T0 = time.time()
features = bootstrap.ensure_features()
split = time_split(features)
train, valid = split.train, split.valid
split.summary().loc[["train", "valid"]]

# %% [markdown]
# **Expected:** train 25,278 rows (2023-03-01 to 2024-03-01, positive rate
# ≈ 0.055) and validation 6,956 rows (2024-04-01 to 2024-06-01, ≈ 0.052).
# The test row is deliberately not shown.

# %% [markdown]
# ## 5.1 · One object from raw columns to probability
#
# In research code the usual pattern is: clean the whole array, scale it,
# *then* split and fit. Two things go wrong with that in production.
#
# 1. **Leakage.** Anything *learned* from data (a median, a mean and standard
#    deviation, the list of categories, a target rate) learned on rows you
#    later evaluate on has seen those rows.
# 2. **Skew.** The serving code has to repeat every step exactly, from memory.
#    One forgotten `log1p` and the API scores a different model from the one
#    you validated. Notebook 10 shows that failure.
#
# The fix for both is the same: the fitted object *is* the whole recipe.
# `churnlab.modeling.make_model` returns such a pipeline:

# %%
from churnlab.modeling import make_model, make_preprocessor, predict_proba
from sklearn.metrics import average_precision_score, roc_auc_score

model = make_model("logreg")
print(model)

# %%
model.fit(train[FEATURES], train[LABEL])
p_valid = predict_proba(model, valid)
honest = {
    "valid_pr_auc": round(float(average_precision_score(valid[LABEL], p_valid)), 4),
    "valid_roc_auc": round(float(roc_auc_score(valid[LABEL], p_valid)), 4),
}
honest

# %% [markdown]
# **Expected:** validation PR-AUC ≈ 0.163 and ROC-AUC ≈ 0.733. This is the
# honest reference number for the rest of the notebook. The positive rate
# (≈ 0.052) is what a random ranking would score on PR-AUC.

# %% [markdown]
# ## 5.2 · Leak (a): fitting the imputer and scaler on everything
#
# The textbook leak: fit the preprocessing on train **and** validation rows,
# then fit the classifier on train only. Here it is, measured rather than
# assumed:

# %%
from sklearn.linear_model import LogisticRegression

both = pd.concat([train, valid], ignore_index=True)
pre_all = make_preprocessor("linear").fit(both[FEATURES])  # <- has seen validation rows
clf = LogisticRegression(max_iter=2000).fit(pre_all.transform(train[FEATURES]), train[LABEL])
p_leaky_pre = clf.predict_proba(pre_all.transform(valid[FEATURES]))[:, 1]

print("honest pipeline      :", honest["valid_pr_auc"])
print("preprocessing on all :", round(float(average_precision_score(valid[LABEL], p_leaky_pre)), 4))

# %% [markdown]
# **Expected:** the two numbers agree to about three decimals (0.1633 vs
# 0.1632 on our run). On this data the leak is **tiny**.
#
# That is worth saying out loud, because it is the opposite of what people
# often claim. A median and a standard deviation estimated on 25,000 rows
# barely move when you add 7,000 more from a similar distribution, so the
# classifier sees almost the same inputs. The leak grows when the data are
# small, when the distribution drifts between periods, or when the "fitted
# preprocessing" is something with many parameters (feature selection on all
# rows, PCA, target encoding). The reason to use a pipeline anyway is that
# you do not want to re-measure this every time, and the same object
# protects serving.

# %% [markdown]
# ## 5.3 · Leak (b): a target encoding that includes the answer
#
# A feature that *uses the label* is in a different league. A tempting one:
# "this customer's historical churn rate". Three versions:
#
# | Version | Computed from | Honest? |
# |---|---|---|
# | `cust_rate_full` | mean label per customer over the whole train + validation table, current row included | no: contains the current label and the future |
# | `cust_rate_train` | the same, but from training labels only, current row included | no inside train; looks harmless for validation |
# | `past_churn_labels` | number of the customer's **earlier** labels whose 30-day window had **closed** before this snapshot | yes |

# %%
work = both[["snapshot_date", "customer_id", LABEL]].copy()
is_train = np.r_[np.ones(len(train), bool), np.zeros(len(valid), bool)]

# (1) full table, current row included
work["cust_rate_full"] = work.groupby("customer_id")[LABEL].transform("mean")

# (2) training labels only, mapped onto validation customers (unseen -> global rate)
train_rate = work[is_train].groupby("customer_id")[LABEL].mean()
work["cust_rate_train"] = work["customer_id"].map(train_rate).fillna(work.loc[is_train, LABEL].mean())

# (3) honest: only labels whose window [t, t + 30 days) closed before this snapshot
pairs = work[["customer_id", "snapshot_date"]].merge(
    work[["customer_id", "snapshot_date", LABEL]], on="customer_id", suffixes=("", "_past"))
pairs = pairs[pairs["snapshot_date_past"] + pd.Timedelta(days=30) <= pairs["snapshot_date"]]
past = pairs.groupby(["customer_id", "snapshot_date"])[LABEL].sum().rename("past_churn_labels")
work = work.merge(past, on=["customer_id", "snapshot_date"], how="left")
work["past_churn_labels"] = work["past_churn_labels"].fillna(0)

work[["cust_rate_full", "cust_rate_train", "past_churn_labels"]].describe().round(3)

# %% [markdown]
# Now add each candidate to the honest pipeline and score it two ways:
# shuffled 5-fold CV **inside train** (how a leak usually gets "validated"),
# and the real validation period.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def with_extra(extra: str) -> Pipeline:
    pre = ColumnTransformer([("base", make_preprocessor("linear"), FEATURES),
                             ("extra", StandardScaler(), [extra])])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


X_all = pd.concat([both[FEATURES], work.drop(columns=["snapshot_date", "customer_id", LABEL])], axis=1)
rows = []
for extra in ["cust_rate_full", "cust_rate_train", "past_churn_labels"]:
    cols = FEATURES + [extra]
    kfold = cross_validate(with_extra(extra), X_all.loc[is_train, cols], both.loc[is_train, LABEL],
                           cv=KFold(5, shuffle=True, random_state=0), scoring="average_precision")
    fitted = with_extra(extra).fit(X_all.loc[is_train, cols], both.loc[is_train, LABEL])
    p = fitted.predict_proba(X_all.loc[~is_train, cols])[:, 1]
    rows.append({"feature": extra,
                 "train_shuffled_5fold_pr_auc": kfold["test_score"].mean(),
                 "valid_pr_auc": average_precision_score(valid[LABEL], p)})
rows.append({"feature": "(none: honest pipeline)", "train_shuffled_5fold_pr_auc": np.nan,
             "valid_pr_auc": honest["valid_pr_auc"]})
pd.DataFrame(rows).round(3)

# %% [markdown]
# **Expected (our run):**
#
# * `cust_rate_full`: ≈ 0.58 in shuffled CV **and** ≈ 0.58 on validation, more
#   than three times the honest 0.163. Both numbers are fake; the feature
#   contains the label of the row being scored. When the leak sits in the
#   feature table itself, *no split can catch it*: validation is fooled too.
# * `cust_rate_train`: ≈ 0.59 in shuffled CV inside train, then ≈ 0.11 on
#   validation, **below** the honest 0.163. The classic signature: a CV score
#   that collapses on a later period. (It is worse than useless there: the
#   model learned to lean on a feature that, for validation rows, no longer
#   contains their label.)
# * `past_churn_labels`: exactly the honest score, because the feature is
#   **zero for every row**. The reason is instructive: **churn is terminal**.
#   A customer with an earlier positive label has left the table by the time
#   that label's window closes.
#
# So "per-customer historical churn rate" is not a feature here at all: its
# entire apparent value was the current row's label.
#
# > 🧠 **Research bridge.** This is the tabular cousin of normalising EEG
# > epochs with statistics computed over all subjects (small leak), versus
# > selecting channels or frequency bands by their correlation with the label
# > on the full dataset before cross-validating (large leak). The second kind
# > is the one that turns a 60% decoder into a "90%" one.

# %%
print("rows with past_churn_labels > 0:", int((work["past_churn_labels"] > 0).sum()), "of", len(work))
print("customers with more than one positive label:",
      int((work.groupby("customer_id")[LABEL].sum() > 1).sum()))

# %% [markdown]
# **Expected:** 0 rows with a non-zero honest value, and only a handful (3
# on our run) of customers with two positive labels at all, from snapshots
# less than 30 days apart. The honest version of the feature carries no
# information.

# %% [markdown]
# ## 5.4 · Anatomy of a `ColumnTransformer`
#
# Here is the preprocessor the logistic-regression pipeline uses
# (`src/churnlab/modeling.py`, `make_preprocessor("linear")`):

# %%
pre = make_preprocessor("linear")
pre

# %% [markdown]
# Three branches, each a small pipeline over a named list of columns:
#
# * `log`: heavy-tailed counts → median impute (+ a missing-indicator column) → `log1p` → standardise;
# * `num`: the other numeric columns → median impute (+ indicator) → standardise;
# * `cat`: one-hot encoding, `handle_unknown="ignore"`, rare levels (< 20 rows) pooled.
#
# Fit it on the train period only and look at what comes out:

# %%
pre.fit(train[FEATURES])
names = pre.get_feature_names_out()
print(len(names), "output columns from", len(FEATURES), "input columns")
print(list(names[:4]), "...")
print([n for n in names if "missingindicator" in n])
print([n for n in names if n.startswith("cat__region")])

# %% [markdown]
# **Expected:** 33 output columns from 19 inputs: the 15 numeric columns, two
# missing-indicator columns for numeric features that had NaNs in training
# (`avg_resolution_hours_90d`, empty for customers with no tickets, and
# `features_used_4w`, empty without recent usage), and one column per
# region/plan/channel/size level (regions: APAC, EU, LATAM, NA, unknown). Names are
# prefixed by branch (`log__`, `num__`, `cat__`), so you can always trace an
# output back to its source.
#
# **Why `log1p`?** Counts such as `sessions_4w` and `mrr_usd` are heavily
# right-skewed. A linear model on the raw value lets a few huge accounts
# dominate the coefficient. `log1p` makes the effect "per doubling" rather than
# "per unit" and handles zeros:

# %%
skew = pd.DataFrame({
    "raw_skew": train[["sessions_4w", "mrr_usd", "tickets_90d"]].skew(),
    "log1p_skew": np.log1p(train[["sessions_4w", "mrr_usd", "tickets_90d"]]).skew(),
}).round(2)
skew

# %% [markdown]
# **Expected:** raw skewness between about 1.7 and 2.7; after `log1p` it is
# between about −0.9 and 0.5. (`sessions_4w` overshoots to a left skew because
# of its many zeros; still far better behaved for a linear model.)
#
# **Unseen categories.** In production a new region or plan *will* arrive.
# `handle_unknown="ignore"` turns it into all-zeros in that block, i.e. "none
# of the known levels", instead of an exception at 3 a.m.:

# %%
probe = valid.head(3).copy()
probe.loc[probe.index[0], "region"] = "Antarctica"  # never seen in training
out = pre.transform(probe[FEATURES])
out = out.toarray() if hasattr(out, "toarray") else out
region_cols = [i for i, n in enumerate(names) if n.startswith("cat__region")]
print("shape:", out.shape)
print("region block for the unseen row:", out[0, region_cols])
print("region block for a normal row  :", out[1, region_cols])
print("scores:", model.predict_proba(probe[FEATURES])[:, 1].round(4))

# %% [markdown]
# **Expected:** the same output width as before; the unseen row's region
# block is all zeros, a normal row has exactly one 1; the model still scores
# all three rows. Whether "ignore" is the *right* behaviour is a product
# decision. It keeps the service up, and monitoring (Day 4) should count how
# often it happens.

# %% [markdown]
# > **Exercise 1.** Write `build_preprocessor(numeric, categorical)` that returns
# > an **unfitted** `ColumnTransformer` which median-imputes and standardises the
# > numeric columns and one-hot encodes the categorical ones, ignoring unseen
# > categories. The grader fits it on a slice of the train period only, then
# > checks: no NaNs after transforming train or validation, unseen categories
# > keep the output width, `get_feature_names_out()` matches the width, and the
# > numeric outputs are standardised on the data they were fitted on.

# %%
def build_preprocessor(numeric, categorical):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(build_preprocessor)

# %% [markdown]
# ## 5.5 · Cross-validation that respects time
#
# Shuffled k-fold assumes rows are exchangeable. Here they are not: a
# training fold contains March 2024 while a test fold contains April 2023, and
# the same customer sits on both sides. The time-series analogue is
# **rolling-origin** (expanding-window) CV: train on everything before a
# cut-off, validate on the next couple of months, move the cut-off forward.
# `churnlab.splits.rolling_origin_folds` yields `(train_idx, valid_idx)` pairs,
# which is exactly what scikit-learn's `cv=` accepts.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/splits.svg" alt="Random, group and time splits, and rolling-origin folds" width="100%">

# %%
from churnlab.splits import rolling_origin_folds

folds = list(rolling_origin_folds(train, n_folds=4))
d = train["snapshot_date"]
pd.DataFrame([{
    "fold": k,
    "train_rows": len(tr), "train_last": d.iloc[tr].max().date(),
    "valid_rows": len(va), "valid_first": d.iloc[va].min().date(), "valid_last": d.iloc[va].max().date(),
} for k, (tr, va) in enumerate(folds)])

# %% [markdown]
# **Expected:** four folds; training windows grow (≈ 10k to ≈ 21k rows), each
# fold validates on the snapshots right after its training window, and every
# index is inside the **train** period. The validation period plays no part.
#
# The indices are **positions**, so the frame passed to `cross_validate` must
# be the same frame (same row order) the folds were computed from.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 2.4))
months = np.sort(train["snapshot_date"].unique())
for k, (tr, va) in enumerate(folds):
    ax.scatter(d.iloc[tr].unique(), [k] * d.iloc[tr].nunique(), color="tab:blue", s=25)
    ax.scatter(d.iloc[va].unique(), [k] * d.iloc[va].nunique(), color="tab:orange", s=25)
ax.set_yticks(range(len(folds)))
ax.set_ylabel("fold")
ax.set_xlabel("snapshot month (train period only)")
ax.set_title("Rolling-origin folds: blue = fit, orange = score")
plt.tight_layout()
plt.show()

# %% [markdown]
# Now the comparison the whole section is about: three estimates of the same
# model's quality.

# %%
from churnlab.modeling import make_model

estimates = []
for name in ["logreg", "hgb"]:
    t = time.time()
    shuffled = cross_validate(make_model(name), train[FEATURES], train[LABEL],
                              cv=KFold(5, shuffle=True, random_state=0), scoring="average_precision")
    rolling = cross_validate(make_model(name), train[FEATURES], train[LABEL],
                             cv=folds, scoring="average_precision")
    fitted = make_model(name).fit(train[FEATURES], train[LABEL])
    estimates.append({
        "model": name,
        "shuffled_5fold": shuffled["test_score"].mean(),
        "rolling_origin": rolling["test_score"].mean(),
        "rolling_fold_std": rolling["test_score"].std(),
        "actual_validation": average_precision_score(valid[LABEL], predict_proba(fitted, valid)),
        "seconds": round(time.time() - t, 1),
    })
cv_comparison = pd.DataFrame(estimates).round(4)
cv_comparison.to_csv(ART / "cv_comparison.csv", index=False)
cv_comparison

# %% [markdown]
# **Expected (our run):** logistic regression ≈ 0.196 shuffled, ≈ 0.186
# rolling, 0.163 on validation; boosted trees ≈ 0.159, ≈ 0.156 and 0.135. The
# fold-to-fold standard deviation is about 0.02.
#
# Read it honestly:
#
# * **Both** CV estimates are above the validation score here. Validation is a
#   later period with a slightly lower positive rate, and three months of
#   positives (≈ 360) is a noisy sample. CV inside train is a tool for
#   *choosing*, not a forecast of the next period.
# * The rolling estimate is **closer** to what happened next, and its
#   fold-to-fold spread shows how much one period's score moves on its own.
# * On this data the gap between shuffled and rolling is modest (≈ 0.01). Do
#   not generalise that: with features that drift, or with strong
#   per-customer effects, shuffled CV can be far more optimistic. The rolling
#   scheme is the one that matches how the model is used, so it is the default
#   whatever the gap turns out to be.
# * Both schemes rank the two models the same way. Notebook 06 asks whether
#   the gap between them is larger than the noise.

# %% [markdown]
# > **Exercise 2.** Write `rolling_cv_scores(make_model_fn, df, n_folds)` that
# > returns a list with one validation PR-AUC per rolling-origin fold.
# > `make_model_fn()` returns a fresh, unfitted model; fit it on the fold's
# > training rows (`FEATURES` → `LABEL`) and score `average_precision_score` on
# > the fold's validation rows. Use `rolling_origin_folds(df, n_folds=n_folds)`.
# > The grader passes a **shuffled** copy of the train period with a
# > non-default index, so use positions, not labels.

# %%
def rolling_cv_scores(make_model_fn, df, n_folds):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(rolling_cv_scores)

# %% [markdown]
# > **Exercise 3 (written).** A colleague's notebook reports validation PR-AUC
# > 0.41 with this code. Find **every** leak or evaluation mistake (there are at
# > least four), say which one you expect to be largest and why, and write the
# > corrected version in words. Write your answer in the cell below.
# >
# > ```python
# > df = features.copy()
# > df["sessions_z"] = (df["sessions_4w"] - df["sessions_4w"].mean()) / df["sessions_4w"].std()
# > df["plan_churn_rate"] = df.groupby("plan")["churned_30d"].transform("mean")
# > df["cust_churn_rate"] = df.groupby("customer_id")["churned_30d"].transform("mean")
# > train = df[df["snapshot_date"] < "2024-04-01"]
# > valid = df[df["snapshot_date"] >= "2024-04-01"]
# > cols = ["sessions_z", "plan_churn_rate", "cust_churn_rate", "tenure_days"]
# > model = LogisticRegression().fit(train[cols], train["churned_30d"])
# > print(average_precision_score(valid["churned_30d"], model.predict_proba(valid[cols])[:, 1]))
# > ```

# %% [markdown]
# *Your answer here.*

# %%
print(f"notebook compute time: {time.time() - T0:.0f} s")

# %% [markdown]
# **Expected:** well under a minute on a laptop; a free Colab CPU may take
# about twice as long.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ValueError: Found unknown categories ['...']` at predict time | an encoder with `handle_unknown="error"` (the default) | `OneHotEncoder(handle_unknown="ignore")`, or `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)` |
# | `ValueError: Input X contains NaN` from `LogisticRegression` | a numeric column reached the classifier without an imputer | add `SimpleImputer` in that branch of the `ColumnTransformer` |
# | `IndexError` / wrong rows when using `rolling_origin_folds` | folds computed on one frame, applied to another (filtered or re-sorted) | compute the folds on exactly the frame you pass to `cross_validate`; use `.iloc` |
# | Rolling CV much slower than expected | boosted trees on a busy machine fighting over threads | set the environment variable `OMP_NUM_THREADS=1` before starting Python, or use fewer folds |
# | A new feature lifts CV PR-AUC from 0.19 to 0.5+ | it almost certainly contains the label (target encoding with the current row, a post-snapshot column) | recompute it from data strictly before the snapshot; check the validation score too, and distrust any jump you cannot explain |
# | `get_feature_names_out` raises | a custom `FunctionTransformer` without `feature_names_out` | pass `feature_names_out="one-to-one"` as `make_preprocessor` does |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why should the scaler live inside the pipeline if, as you measured, fitting it on all the data barely changed the score here?
# 2. A per-customer churn-rate feature doubles your PR-AUC. What do you check first, and why did the honest version of that feature turn out to be useless on this data?
# 3. Explain rolling-origin cross-validation to someone who only knows k-fold. When is shuffled k-fold acceptable?
# 4. Your rolling CV says 0.186 and the next period scores 0.163. Is the model broken? What would you look at?
# 5. What does `handle_unknown="ignore"` do, and what should you monitor because of it?
#
# ➡️ **Next:** [`06_model_selection.ipynb`](06_model_selection.ipynb) — compare models with confidence intervals and a search budget, and find out whether the trees really lose.
