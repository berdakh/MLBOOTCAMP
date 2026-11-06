# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/04_leakage_safe_splits_and_baseline.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 04 · Leakage-safe splits & a baseline
#
# **Day 1 · ~45 minutes · CPU only**
#
# A validation score is an answer to a question, and the split decides the
# question. This notebook picks the split that matches how the churn model will
# be used, measures what leakage does to the score, and sets honest baselines
# that every later model has to beat.
#
# By the end you will be able to:
#
# 1. build a time split with an embargo gap and explain why the gap exists;
# 2. check a split for time and group leakage automatically;
# 3. measure leakage empirically instead of arguing about it, on validation only;
# 4. explain which question a time split, a group split and a random split each answer;
# 5. report baselines (prior, hand rule, small linear model) with the split named.
#
# **You will produce:** `artifacts/baselines.json`, validation PR-AUC and ROC-AUC for three baselines.
#
# **One rule for the whole course:** the **test** period (July-November 2024)
# is touched **once**, in notebook 15. Everything here is measured on
# **validation**.

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
# --- Catch up: the feature table from notebook 03 --------------------------
import numpy as np
import pandas as pd

from churnlab import bootstrap
from churnlab.features import LABEL
from churnlab.grading import nb04 as grade

features = bootstrap.ensure_features()

# %% [markdown]
# ## 4.1 · The time split, and why it has a gap
#
# The model will run on the first of each month and score the accounts active
# that day. So the honest rehearsal is: **train on earlier months, validate on
# later months**, exactly as it will happen in production.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/splits.svg" alt="Time split with embargo gaps, compared with random and group splits" width="100%">

# %%
from churnlab.splits import TRAIN_END, VALID_END, assert_no_leakage, time_split

split = time_split(features)  # train < 2024-04-01 <= valid < 2024-07-01 <= test
print("TRAIN_END", TRAIN_END.date(), " VALID_END", VALID_END.date())
split.summary()

# %% [markdown]
# **Expected:** train 25,278 rows (2023-03-01 to 2024-03-01, positive rate
# ≈ 0.055), valid 6,956 rows (2024-04-01 to 2024-06-01, ≈ 0.052), test 12,127
# rows (2024-07-01 to 2024-11-01, ≈ 0.059). The test row is shown so you know it
# exists; its rows are not used until notebook 15.
#
# **The embargo gap.** The label of a row at snapshot T looks at [T, T + 30d).
# If T is less than 30 days before the end of training, that label is decided by
# events **inside the validation period**: the training set then "knows" part of
# what validation is supposed to test. `time_split` therefore keeps a training
# row only if `T + gap <= train_end`.
#
# With monthly snapshots on the 1st and boundaries on the 1st, the March
# snapshot's window [Mar 1, Mar 31) ends just before April 1, so the gap
# happens to drop nothing. Move the boundary to mid-month and it bites:

# %%
rows = []
for train_end, gap in [("2024-04-01", 30), ("2024-03-15", 0), ("2024-03-15", 30)]:
    s = time_split(features, train_end=pd.Timestamp(train_end), gap_days=gap)
    rows.append({"train_end": train_end, "gap_days": gap, "train_rows": len(s.train),
                 "last_train_snapshot": s.train["snapshot_date"].max().date()})
pd.DataFrame(rows)

# %% [markdown]
# **Expected:** with the default boundary, training ends at the 2024-03-01
# snapshot (25,278 rows). With `train_end = 2024-03-15` and no gap, the same
# 2024-03-01 snapshot stays in training although its labels run to March 31,
# inside the next period; with a 30-day gap it is dropped (23,032 rows, 2,246
# fewer). The rule costs nothing when it is not needed, and saves you when it is.
#
# (Keep `gap_days >= 1` with `time_split`: with `gap_days=0` and a boundary
# that is itself a snapshot date, `snapshot_date + 0 <= train_end` puts the
# boundary snapshot in **both** train and validation.)
#
# `assert_no_leakage` turns the rule into a check you can put in a test or at
# the top of a training script:

# %%
from churnlab.splits import random_split

assert_no_leakage(split, label_horizon_days=30)
print("time split: no leakage")

try:
    assert_no_leakage(random_split(features), label_horizon_days=30)
except AssertionError as exc:
    print("random split:", exc)

# %% [markdown]
# **Expected:** the time split passes; the random split fails with
# `train label windows overlap the validation period`.

# %% [markdown]
# > **Exercise 1.** Write `time_split_with_embargo(df, train_end, valid_end, gap_days)`
# > returning a tuple `(train, valid, test)` of DataFrames. Train: snapshots
# > whose label window ends by `train_end`; valid: snapshots from `train_end`
# > whose label window ends by `valid_end`; test: snapshots from `valid_end` on.
# > The grader compares your rows with `churnlab.splits.time_split` for four
# > boundary/gap combinations, without looking at its source first, please.

# %%
def time_split_with_embargo(df: pd.DataFrame, train_end, valid_end, gap_days: int):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(time_split_with_embargo)

# %% [markdown]
# > **Exercise 2.** Write `leakage_problems(train, test, label_horizon_days, group_col=None)`
# > that returns a **list of human-readable strings**, empty when the split is
# > clean. Report (a) a **time** problem when the last train snapshot plus the
# > label horizon reaches past the first test snapshot, and (b) when `group_col`
# > is given, a **group** problem if any value of that column is in both frames,
# > saying how many. Without `group_col`, shared customers are fine (that is the
# > next-month question).

# %%
def leakage_problems(train: pd.DataFrame, test: pd.DataFrame, label_horizon_days: int,
                     group_col=None) -> list:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(leakage_problems)

# %% [markdown]
# ## 4.2 · Measuring leakage (on validation, never test)
#
# Arguments about leakage are settled by measurement. One small model is used
# throughout so that only the split or the features change: a scaled logistic
# regression on five point-in-time features. Metrics:
#
# * **PR-AUC** (average precision): how good the ranking is **at the top**,
#   where the retention team works. A random ranking scores about the positive
#   rate (≈ 0.05 here), not 0.5. It is the headline metric this course uses.
# * **ROC-AUC**: the probability a random churner is ranked above a random
#   non-churner. Useful, but flattered by the many easy negatives at a 5% rate.
#
# Day 2 goes deeper (curves, confidence intervals, thresholds).

# %%
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# scikit-learn 1.6 with SciPy >= 1.15 warns about a solver option it still passes; harmless.
warnings.filterwarnings("ignore", message="Unknown solver options")

BASE_FEATURES = ["sessions_4w", "tenure_days", "failed_payments_90d", "weeks_since_last_active", "tickets_30d"]


def fit_and_score(train: pd.DataFrame, valid: pd.DataFrame, cols: list) -> dict:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[cols], train[LABEL])
    scores = model.predict_proba(valid[cols])[:, 1]
    return {"pr_auc": round(float(average_precision_score(valid[LABEL], scores)), 4),
            "roc_auc": round(float(roc_auc_score(valid[LABEL], scores)), 4),
            "valid_positive_rate": round(float(valid[LABEL].mean()), 4)}


honest = fit_and_score(split.train, split.valid, BASE_FEATURES)
honest

# %% [markdown]
# **Expected:** validation PR-AUC ≈ 0.12 and ROC-AUC ≈ 0.72, against a
# positive rate of 0.052.
#
# ### (a) Leaky features
#
# `build_features(include_leaky=True)` adds two export-time columns from
# notebook 02: `status_cancelled` (status as of the export) and
# `days_until_last_active` (from `last_active_week`, which is in the future of
# every snapshot). Same split, same model:

# %%
from churnlab.features import build_features

leaky_table = build_features(bootstrap.load_clean(), include_leaky=True)
leaky_split = time_split(leaky_table)
leaky = fit_and_score(leaky_split.train, leaky_split.valid,
                      BASE_FEATURES + ["status_cancelled", "days_until_last_active"])
pd.DataFrame({"honest features": honest, "+ leaky columns": leaky}).T

# %% [markdown]
# **Expected:** PR-AUC jumps from ≈ 0.12 to ≈ 0.99 and ROC-AUC to ≈ 0.999. No
# real churn model is that good; a number like this is a **symptom**, and the
# right reaction is to look for the leak, not to celebrate. The split was
# perfectly time-ordered: leakage through **features** is invisible to any
# split check.

# %% [markdown]
# ### (b) Random rows vs time split
#
# The same five honest features, the same model, but a random 60/20/20 split
# of rows (three seeds, because a random split is itself random):

# %%
from churnlab.splits import group_split

comparison = [{"split": "time (train < Apr 2024, valid Apr-Jun)", **honest}]
for seed in range(3):
    r = random_split(features, seed=seed)
    comparison.append({"split": f"random rows, seed {seed}", **fit_and_score(r.train, r.valid, BASE_FEATURES)})
for seed in range(3):
    g = group_split(features, seed=seed)
    comparison.append({"split": f"group by customer, seed {seed}", **fit_and_score(g.train, g.valid, BASE_FEATURES)})
comparison = pd.DataFrame(comparison).set_index("split")
comparison

# %% [markdown]
# **Expected (what this run showed):** the random-row splits score PR-AUC
# ≈ 0.13-0.15 depending on the seed: a little above the time split's ≈ 0.12,
# but by less than the random split's own seed-to-seed spread - not the
# dramatic inflation textbooks warn about. The group splits land around
# 0.14-0.16. ROC-AUC is ≈ 0.72-0.74 everywhere. Note that each split's
# validation set is a **different population** (different months, different
# positive rates), so these numbers are not strictly comparable with each other
# either - one more reason to name the split next to every number.
#
# Report that honestly instead of the textbook story. Why so little inflation
# here?
#
# * The worst leak was already removed **in SQL**: every feature is
#   point-in-time, so a random split cannot pull future feature values into a
#   row.
# * A five-feature linear model has almost no capacity to memorise individual
#   customers, which is how a random split usually cheats (rows of the same
#   customer in train and validation).
# * The synthetic business is fairly stationary over two years, so
#   "a random past month" and "the next months" look alike.
#
# And yet the random split is still wrong for this project, because it answers
# a question nobody asked ("how well do we score a random month we have
# partly trained on?"), it fails `assert_no_leakage`, and it would hide drift
# that a time split exposes. Note also the **seed-to-seed spread** of the random
# split (several hundredths of PR-AUC): that is the size of difference you
# cannot claim from one split. A high-capacity model, entity-specific features or
# a drifting business would widen the gap; the habit has to be in place before
# you find out which case you are in.
#
# ### (c) Two honest splits, two different questions
#
# * **Time split** answers *"how well will we score our existing customers next
#   month?"* That is this project's use, so it is the headline split.
# * **Group split** (whole customers in one split) answers *"how well will we
#   score customers we have never seen?"* - e.g. a brand-new market. It mixes
#   periods, so it says nothing about the future.
#
# Neither is "more correct" in general. Choose the one that matches the
# deployment, and **name the split next to every number you report**.
#
# > 🧠 **Research bridge.** You already know this from EEG/BCI: within-subject
# > vs cross-subject evaluation answer different questions, and a
# > leave-one-subject-out score is not comparable to a within-session one. The
# > translation is in Exercise 4.

# %% [markdown]
# ## 4.3 · Baselines
#
# A model is only good relative to something simpler. Three baselines, all
# trained on train and measured on **validation**:
#
# 1. **Prior** (`DummyClassifier`): the same score for everyone. PR-AUC equals
#    the positive rate; this is the floor.
# 2. **Hand rule**: flag accounts with zero sessions in the last 4 weeks or a
#    failed payment in the last 90 days - what an analyst would write first.
# 3. **Logistic regression** on the five features above.

# %%
from sklearn.dummy import DummyClassifier

valid = split.valid
y = valid[LABEL]

prior = DummyClassifier(strategy="prior").fit(split.train[BASE_FEATURES], split.train[LABEL])
prior_scores = prior.predict_proba(valid[BASE_FEATURES])[:, 1]

rule_flag = ((valid["sessions_4w"] == 0) | (valid["failed_payments_90d"] > 0)).astype(float).to_numpy()


def report(name, scores):
    return {"baseline": name, "split": "time · validation",
            "pr_auc": round(float(average_precision_score(y, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y, scores)), 4)}


baselines = pd.DataFrame([
    report("prior (DummyClassifier)", prior_scores),
    report("hand rule: no sessions 4w OR failed payment 90d", rule_flag),
    {"baseline": "logistic regression, 5 features", "split": "time · validation", **{
        k: honest[k] for k in ("pr_auc", "roc_auc")}},
]).set_index("baseline")
print(f"validation positive rate: {y.mean():.4f}, accounts flagged by the rule: {rule_flag.mean():.1%}")
baselines

# %% [markdown]
# **Expected:** prior PR-AUC = 0.052 (exactly the positive rate) and ROC-AUC
# 0.5; the hand rule only ≈ 0.06 PR-AUC while flagging about 12% of accounts;
# logistic regression ≈ 0.12.
#
# The hand rule is barely above the floor. Two reasons: a yes/no rule gives one
# operating point instead of a ranking, and "zero sessions" is rare among active
# accounts. The useful lesson is not "rules are bad" - see Exercise 3 - but
# that **you only know after measuring**.

# %%
import json

record = {
    "split": "time_split(train_end=2024-04-01, valid_end=2024-07-01, gap_days=30)",
    "evaluated_on": "validation",
    "valid_rows": int(len(valid)),
    "valid_positive_rate": round(float(y.mean()), 4),
    "features": BASE_FEATURES,
    "baselines": baselines.reset_index().to_dict(orient="records"),
    "leakage_check": {"honest_pr_auc": honest["pr_auc"], "with_leaky_columns_pr_auc": leaky["pr_auc"]},
}
(ART / "baselines.json").write_text(json.dumps(record, indent=2))
print(f"saved {ART / 'baselines.json'}")

# %% [markdown]
# > **Exercise 3.** Write `rule_scores(df)` returning a numeric array with one
# > score per row (higher = more likely to churn), using **only** hand-chosen
# > combinations of feature columns - no fitting. The grader passes the
# > validation rows **without** the label, and passes if your validation PR-AUC
# > exceeds 1.5 × the positive rate (≈ 0.078). Hint: a continuous score ranks
# > better than a yes/no flag.

# %%
def rule_scores(df: pd.DataFrame) -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(rule_scores)

# %% [markdown]
# **Expected (once solved):** a validation PR-AUC around 0.10-0.12 is
# achievable with three or four signals - close to the logistic regression,
# with no training at all. That is worth knowing before anyone proposes a deep
# model.

# %% [markdown]
# > **Exercise 4 (written).** The EEG analogy. For a cross-subject BCI study,
# > say which evaluation corresponds to (a) this project's **time split**,
# > (b) the **group split**, and (c) the **random row split**, and give one
# > situation where each of (a) and (b) is the right choice. Then say which
# > EEG equivalent of "label window crossing the boundary" you would need an
# > embargo for. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | A validation PR-AUC above ~0.5 on this data | a leaky column in the features (`status`, `churn_date`, `last_active_week` or something derived) | list the features you used; rebuild with `include_leaky=False` |
# | PR-AUC ≈ 0.05 for every model | the scores are constant, or reversed (low = churn) | check `np.unique(scores)`; flip the sign |
# | `ValueError: Input X contains NaN` | a feature such as `features_used_4w` has NaN for accounts with no usage | impute inside a pipeline (notebook 05) or pick columns without NaN |
# | Your split's row counts differ from `time_split` by one snapshot | `<` vs `<=` at a boundary, or the gap applied to the wrong side | a row is in an earlier split only if `snapshot_date + gap <= end` |
# | Results change every run | a random split without a seed | pass `seed=`; better, use the time split |
# | Tempted to "just check" the test set | - | don't: notebook 15 does it once, with everything decided beforehand |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why does the churn project use a time split, and what question would a group split answer instead?
# 2. What is the embargo gap, and why did it drop no rows with the default boundaries here?
# 3. A colleague reports validation PR-AUC 0.97 on this data. What do you check first, and why do split checks not catch it?
# 4. On this data the random split did not inflate the score much. Why, and why is it still the wrong split?
# 5. Why is the prior's PR-AUC equal to the positive rate, and why report a hand rule at all?
#
# ➡️ **Next:** [`05_pipelines_and_cross_validation.ipynb`](05_pipelines_and_cross_validation.ipynb) — put preprocessing inside the model and cross-validate in time, so Day 2's comparisons are trustworthy.
