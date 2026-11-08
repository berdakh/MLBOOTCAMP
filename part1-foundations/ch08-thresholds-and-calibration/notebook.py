# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 8 · Thresholds & calibration
#
# **Part I · Chapter 8 · about 40 minutes · CPU only**
#
# A churn model does not call customers; a person does, when a score crosses
# a line someone chose. This notebook chooses that line from stated costs and
# a call capacity, shows how much it moves when the assumptions move, and
# checks whether the scores can be read as probabilities at all.
#
# By the end you will be able to:
#
# 1. read a precision-recall curve and a threshold table in terms of calls made and churners reached;
# 2. derive a threshold from an explicit (assumed) cost model, and say how it moves when an assumption changes;
# 3. turn a fixed call capacity into a top-k-per-month rule;
# 4. measure calibration (reliability table, Brier score) and recalibrate without leaking;
# 5. explain why calibration leaves PR-AUC unchanged but moves thresholds and expected value.
#
# **You will produce:** `artifacts/threshold_policy.json`, the chosen rule
# with the assumptions it depends on.
#
# Scores and decisions here are on the **validation** period; models are fitted
# on **train**. The test period is reserved for Chapter 16.

# %% [markdown]
# Read first: [Chapter 8 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: features, split, and the model selected in notebook 06 ------
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from churnlab import bootstrap
from churnlab.evaluation import (
    best_threshold_by_value,
    calibration_table,
    confusion_at,
    expected_value,
    threshold_for_precision,
    threshold_table,
)
from churnlab.features import FEATURES, LABEL
from grading import ch08 as grade
from churnlab.modeling import make_model, predict_proba
from churnlab.splits import time_split

T0 = time.time()
split = time_split(bootstrap.ensure_features())
train, valid = split.train, split.valid

selection_path = ART / "selection.json"
if selection_path.exists():
    selection = json.loads(selection_path.read_text())
    print("using notebook 06's selection:", selection["selected_model"], selection["params"])
else:
    selection = {"selected_model": "logreg", "params": {"C": 1.0}}
    print("no selection.json in this artifact dir; using the course's selection: logreg, C=1.0 (notebook 06)")

model = make_model(selection["selected_model"], **selection["params"]).fit(train[FEATURES], train[LABEL])
y = valid[LABEL].to_numpy()
scores = predict_proba(model, valid)
print(f"validation: {len(y):,} rows, {y.sum()} churners, PR-AUC {average_precision_score(y, scores):.4f}")

# %% [markdown]
# **Expected:** logistic regression, validation PR-AUC ≈ 0.163 with 362
# churners among 6,956 rows (three monthly snapshots).

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Threshold** | the score above which an account is flagged for a call | 0.19 |
# | **TP / FP / FN / TN** | flagged churner / flagged non-churner / missed churner / correctly ignored | 64 calls a month, ≈ 27% of them TPs |
# | **Precision** | share of flagged accounts that were going to churn | ≈ 0.27 at the cost threshold |
# | **Recall** | share of all churners that were flagged | ≈ 0.14 at the cost threshold |
# | **Expected value (EV)** | money gained by a rule under stated costs: TP × value − FP × cost | ≈ $3,400 over three months |
# | **Break-even probability $p^{*}$** | the probability above which a call pays for itself | $30 / (150 + 30) = 1/6$ |
# | **Capacity / top-k** | a fixed number of calls per period, filled with the highest scores | 40 calls a month |
# | **Calibrated** | among accounts scored $p$, a share $p$ actually churns | plain logreg is, balanced logreg is not |
# | **Reliability diagram** | plot of mean predicted vs observed rate, per score bin | the diagonal is perfect calibration |
# | **Brier score** | mean squared error of the probabilities; lower is better | 0.0467 vs 0.0494 for the base rate |
# | **Log loss** | average surprise of the true labels under the predicted probabilities | what logistic regression minimises |
# | **Recalibration** | a monotone map from score to probability, learned on held-out later rows | sigmoid (Platt) or isotonic |

# %% [markdown]
# ## 8.1 · The precision-recall curve is a menu
#
# Each point on the curve is a threshold: flag everyone above it, and you get
# that precision (share of calls that reach a would-be churner) and that
# recall (share of churners you reach):
#
# $$
# \text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}
# $$
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-pr-vs-roc.svg" alt="ROC curve with a random diagonal at 0.5 next to a precision-recall curve whose random baseline is the positive rate, 0.05" width="100%">
#
# *What to notice:* the ROC plot looks healthy (0.73) while the PR plot shows that precision drops towards the 5% floor quickly; the PR view is the one that matches "how many of our calls are useful".
#
# > **Intuition.** The PR curve is a restaurant menu: every dish (threshold)
# > comes with a price (calls made) and a portion (churners reached). You do
# > not pick the "best" dish, you pick the one your budget and appetite fit.
# > *Where it breaks down:* a menu has fixed prices; here the precision of a
# > given threshold drifts from month to month.

# %%
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y, scores)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rec, prec, label=f"logreg (PR-AUC {average_precision_score(y, scores):.3f})")
ax.axhline(y.mean(), color="grey", ls="--", label=f"random ({y.mean():.3f})")
for t in (0.1, 0.17, 0.3):
    c = confusion_at(y, scores, t)
    p_, r_ = c["tp"] / max(1, c["tp"] + c["fp"]), c["tp"] / (c["tp"] + c["fn"])
    ax.scatter(r_, p_, zorder=3)
    ax.annotate(f"t={t}", (r_, p_), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("recall (share of churners reached)")
ax.set_ylabel("precision (share of calls that matter)")
ax.set_title("Validation precision-recall curve")
ax.legend()
plt.tight_layout()
plt.show()

# %%
table = threshold_table(y, scores, thresholds=[0.05, 0.08, 0.10, 0.12, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40])
table["flagged_per_month"] = (table["flagged"] / valid["snapshot_date"].nunique()).round(0)
table.round(3)

# %% [markdown]
# **Expected (our run):** as the threshold rises from 0.05 to 0.30, flags fall
# from ≈ 750 to ≈ 17 a month, precision climbs from ≈ 0.10 to ≈ 0.34, and
# recall falls from ≈ 0.63 to ≈ 0.05. At 0.40 the precision of 0.52 is 12
# churners out of 23 flags: too few rows to mean much. High precision is not
# available at any useful volume. This is a weak-signal problem, and the
# business case has to work with that.

# %% [markdown]
# ## 8.2 · A threshold from costs (stated as assumptions)
#
# The course's cost model lives in `churnlab/bootstrap.py`. These are
# **teaching assumptions, not facts** about any business:
#
# * a retention call costs **$30** (`COST_FP`, paid whether or not the customer would have churned);
# * a would-be churner who is called is worth **$150** in expectation
#   (`VALUE_TP` = a $600 margin × an assumed 25% save rate);
# * a missed churner costs nothing *extra* relative to today (no call, no save).
#
# For a **calibrated** probability p, calling is worth it when
# p × 150 > (1 − p) × 30, i.e. **p > 30 / (150 + 30) = 1/6 ≈ 0.167**.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-threshold-value.svg" alt="Expected value against threshold: negative when too many low-probability accounts are called, a broad maximum near the break-even probability 1/6, falling towards zero at high thresholds" width="100%">
#
# *What to notice:* the curve is flat near its top, so the exact threshold matters less than the ratio of costs that put the top there.
#
# > **Intuition.** Each call is a $30 lottery ticket that pays $150 with
# > probability p. Buy the ticket when its expected payout beats its price:
# > $150p > 30(1 - p)$. *Where it breaks down:* "p" must be a real
# > probability (calibrated, section 8.6), and the $150 hides a save rate
# > nobody has measured yet (section 8.3). That is
# the theory. `best_threshold_by_value` finds the empirical optimum on
# validation:

# %%
VALUE_TP, COST_FP = bootstrap.VALUE_TP, bootstrap.COST_FP
t_best, v_best = best_threshold_by_value(y, scores, VALUE_TP, COST_FP)
c_best = confusion_at(y, scores, t_best)
print(f"assumed VALUE_TP=${VALUE_TP:.0f}, COST_FP=${COST_FP:.0f} -> break-even p = {COST_FP / (VALUE_TP + COST_FP):.3f}")
print(f"best threshold on validation: {t_best:.3f}, expected value ${v_best:,.0f} over 3 months")
print("confusion at that threshold:", c_best)
print(f"calls per month: {(c_best['tp'] + c_best['fp']) / 3:.0f}, "
      f"precision {c_best['tp'] / (c_best['tp'] + c_best['fp']):.2f}, recall {c_best['tp'] / y.sum():.2f}")

# %% [markdown]
# **Expected (our run):** best threshold ≈ 0.19, expected value ≈ $3,400 over
# the three validation months, about 64 calls a month at precision ≈ 0.27 and
# recall ≈ 0.14.
#
# The empirical optimum lands near the theoretical 1/6. That agreement is
# itself evidence that the scores are roughly calibrated (section 8.6). If
# they were not, the theory would point at the wrong place and only the
# empirical search would be right.
#
# A caution about this number: the threshold is chosen **on** validation, so
# its expected value on validation is optimistic. The honest check of the
# whole rule happens once, on the test period, in Chapter 16.

# %%
grid = np.round(np.arange(0.05, 0.45, 0.01), 2)
ev = [expected_value(confusion_at(y, scores, t), VALUE_TP, COST_FP) for t in grid]
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(grid, ev)
ax.axvline(1 / 6, color="grey", ls="--", label="theory: 1/6")
ax.axvline(t_best, color="tab:red", ls=":", label=f"empirical best {t_best:.3f}")
ax.axhline(0, color="black", lw=0.5)
ax.set_xlabel("threshold")
ax.set_ylabel("expected value on validation ($)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Expected:** a curve with a broad, noisy top between roughly 0.15 and 0.25.
# A flat top is good news: the exact threshold matters less than the order of
# magnitude, and a reviewer arguing 0.17 vs 0.19 is arguing about noise.

# %% [markdown]
# ## 8.3 · How fragile is the threshold? A sensitivity table
#
# The save rate (25%) is the shakiest assumption: nobody has measured it yet.
# Halve it, double it, and double the call cost:

# %%
scenarios = [
    ("save rate halved (12.5%)", VALUE_TP / 2, COST_FP),
    ("as assumed (25%)", VALUE_TP, COST_FP),
    ("save rate doubled (50%)", VALUE_TP * 2, COST_FP),
    ("call cost doubled ($60)", VALUE_TP, COST_FP * 2),
]
sens = []
for label, vt, cf in scenarios:
    t, v = best_threshold_by_value(y, scores, vt, cf)
    c = confusion_at(y, scores, t)
    sens.append({"scenario": label, "value_tp": vt, "cost_fp": cf, "theory_threshold": cf / (vt + cf),
                 "best_threshold": t, "calls_per_month": (c["tp"] + c["fp"]) / 3,
                 "precision": c["tp"] / max(1, c["tp"] + c["fp"]), "expected_value": v})
sensitivity = pd.DataFrame(sens).round(3)
sensitivity

# %% [markdown]
# **Expected (our run):** halving the save rate pushes the best threshold to
# ≈ 0.45 (about 6 calls a month) and the expected value from $3,420 to $615;
# doubling it drops the threshold to ≈ 0.10, multiplies the calls by ≈ 3.6
# (≈ 230 a month) and the value fivefold. Doubling the call cost picks the
# **same** threshold as halving the save rate, because only the ratio
# `cost_fp / value_tp` sets the threshold (theory: 0.286 for both); the value
# differs. The decision is **far more sensitive to the business
# assumptions than to anything we did in model selection**. The most valuable
# next step is not a better model; it is measuring the save rate (an A/B test of
# calls vs no calls among flagged customers).

# %% [markdown]
# ## 8.4 · A capacity constraint: top-k per month
#
# In practice the team can make a fixed number of calls per month. Suppose
# they can call **40 accounts per month**, fewer than the ≈ 64 the cost-based
# threshold asks for. Two ways to honour that:
#
# * a **global threshold** chosen so that *on average* 40 a month are flagged:
#   the volume then drifts with the score distribution, and some months
#   overflow the team;
# * **top-k per snapshot**: each month, call the 40 highest scores. Volume is
#   exact by construction. This is the usual production rule.
#
# > **Intuition.** A threshold is "everyone taller than 1.90 m makes the
# > team"; top-k is "the 40 tallest make it". With a fixed squad size only the
# > second fills it every season. *Where it breaks down:* top-k always fills
# > the quota, even in a month where nobody is really at risk; pair it with a
# > minimum score if a wasted call is expensive.

# %%
CAPACITY = 40
months = valid["snapshot_date"]
per_month_at_tbest = pd.Series(scores >= t_best).groupby(months.to_numpy()).sum()
t_global = np.quantile(scores, 1 - CAPACITY * months.nunique() / len(scores))
per_month_global = pd.Series(scores >= t_global).groupby(months.to_numpy()).sum()
pd.DataFrame({"flagged @ cost threshold": per_month_at_tbest,
              "flagged @ global threshold for 40/month": per_month_global}).rename_axis("snapshot")

# %% [markdown]
# **Expected:** at the cost-based threshold, 60-70 flags per month (more than
# the team can handle); a single global threshold tuned to 40 a month on average
# still varies month to month. Validation months are similar, so the drift is
# small here; it is not small when the score distribution shifts (Chapter 15).

# %% [markdown]
# > **Exercise 1.** Write `top_k_per_snapshot(df, scores, k)` returning a boolean
# > array (one per row of `df`, in `df`'s row order) that is `True` for the `k`
# > highest scores **within each** `snapshot_date` (all rows if a snapshot has
# > fewer than `k`). The grader shuffles the rows and uses a non-default index.

# %%
def top_k_per_snapshot(df, scores, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(top_k_per_snapshot)

# %% [markdown]
# How good are the 40 calls a month? `precision_at_k` per snapshot, and the
# value under the same cost assumptions:

# %%
from churnlab.evaluation import precision_at_k

rows = []
for month, idx in valid.groupby("snapshot_date").indices.items():
    p_k = precision_at_k(y[idx], scores[idx], CAPACITY)
    rows.append({"snapshot": pd.Timestamp(month).date(), "precision@40": p_k,
                 "churners_reached": round(p_k * CAPACITY),
                 "value": round(p_k * CAPACITY) * VALUE_TP - (CAPACITY - round(p_k * CAPACITY)) * COST_FP})
capacity_df = pd.DataFrame(rows)
capacity_df.round(3)

# %% [markdown]
# **Expected (our run):** precision@40 between 0.20 and 0.35 per month, i.e.
# 8 to 14 would-be churners reached out of 40 calls, still positive value each
# month. The average (≈ 0.275) is about the same as the precision at the cost
# threshold (≈ 0.27): on this weak-signal data the very top of the ranking is
# not much purer than the next 24 accounts. So the capacity limit costs
# roughly the share of calls dropped; that number is the business case for
# a bigger team, and you can now put a dollar figure on it.

# %% [markdown]
# > **Exercise 2.** Back to the cost-based rule of 8.2: write your own
# > `best_threshold(y, scores, value_tp, cost_fp)` returning
# > `(threshold, value)`, where flagging means `score >= threshold` and
# > value = TP × `value_tp` − FP × `cost_fp`. Search every distinct score as a
# > candidate (sorting once and using cumulative sums makes it fast). The
# > grader checks your reported value is what your threshold actually achieves,
# > and that it is at least as good as `best_threshold_by_value`.

# %%
def best_threshold(y, scores, value_tp, cost_fp):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(best_threshold)

# %% [markdown]
# A third kind of constraint is a **quality** floor: "we will only call if at
# least 30% of calls reach a would-be churner".

# %%
t_p30 = threshold_for_precision(y, scores, 0.30)
c = confusion_at(y, scores, t_p30)
print(f"lowest threshold with precision >= 0.30: {t_p30:.3f} -> {(c['tp'] + c['fp']) / 3:.0f} calls/month, "
      f"recall {c['tp'] / y.sum():.2f}")

# %% [markdown]
# **Expected:** a threshold around 0.25 and roughly 30-40 calls a month. Note
# that "lowest threshold that reaches 30%" is fragile on a noisy curve: precision
# is not monotone in the threshold, so a few rows can move it.

# %% [markdown]
# ## 8.5 · Write the policy down

# %%
policy = {
    "model": selection["selected_model"],
    "rule": f"each month, call the top {CAPACITY} accounts by score",
    "cost_based_threshold": round(t_best, 4),
    "assumptions": {"value_tp_usd": VALUE_TP, "cost_fp_usd": COST_FP,
                    "derivation": "VALUE_TP = $600 margin x 25% save rate (assumed, unmeasured)",
                    "capacity_per_month": CAPACITY},
    "validation_evidence": {
        "calls_per_month_at_threshold": round((c_best["tp"] + c_best["fp"]) / 3, 1),
        "precision_at_threshold": round(c_best["tp"] / (c_best["tp"] + c_best["fp"]), 3),
        "mean_precision_at_capacity": round(float(capacity_df["precision@40"].mean()), 3),
    },
    "sensitivity": "threshold moves 0.10-0.45 when the save rate is doubled/halved (section 7.3)",
    "chosen_on": "validation (optimistic); confirm once on test in notebook 15",
}
(ART / "threshold_policy.json").write_text(json.dumps(policy, indent=2))
print(json.dumps(policy, indent=2))

# %% [markdown]
# ## 8.6 · Can the scores be read as probabilities?
#
# The 1/6 rule assumed **calibrated** scores: among accounts scored 0.2, about
# 20% churn. Check it: sort the validation rows into ten equal-sized score
# bins and compare the mean score with the observed churn rate in each bin.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-calibration.svg" alt="Reliability diagram: a calibrated model follows the diagonal; a class-weighted model sits far below it, predicting several times the true churn rate, with almost the same ranking" width="100%">
#
# *What to notice:* the two models rank almost identically (PR-AUC 0.163 vs 0.161), yet only one of them can be used with the 1/6 rule.
#
# > **Intuition.** Calibration is a weather forecaster's honesty: on the
# > days they say "20% rain", it should rain on about one in five. Ranking is
# > their skill at telling wetter days from drier ones. A forecaster can be
# > skilled and dishonest (always doubling the number). *Where it breaks
# > down:* calibration is checked per bin on average; a model can be
# > calibrated overall and badly off for one slice of customers.

# %%
cal = calibration_table(y, scores, n_bins=10)
print(f"Brier score: {brier_score_loss(y, scores):.5f}   "
      f"(always predicting the base rate: {brier_score_loss(y, np.full_like(scores, train[LABEL].mean())):.5f})")
cal.round(4)

# %% [markdown]
# **Expected:** the two columns track each other closely, from ≈ 0.01 in the
# lowest bin to ≈ 0.17-0.18 in the top bin, and the Brier score (≈ 0.0467) is
# below the constant-base-rate Brier (≈ 0.0494). Logistic regression trained
# on the natural class balance is usually close to calibrated: its loss *is*
# the log-likelihood of the probabilities.
#
# The interesting case is a model that is *not* calibrated. A very common
# way to get one is `class_weight="balanced"`, often added "because the
# classes are imbalanced":

# %%
balanced = make_model("logreg", class_weight="balanced").fit(train[FEATURES], train[LABEL])
scores_bal = predict_proba(balanced, valid)
print(f"balanced: mean score {scores_bal.mean():.3f} vs churn rate {y.mean():.3f}; "
      f"Brier {brier_score_loss(y, scores_bal):.4f}; PR-AUC {average_precision_score(y, scores_bal):.4f}")
print("its 'best' threshold under the same costs:",
      round(best_threshold_by_value(y, scores_bal, VALUE_TP, COST_FP)[0], 3), "(vs the theory's 0.167)")

# %% [markdown]
# **Expected:** a mean score around 0.4 against a churn rate of 0.05, a much
# worse Brier score (≈ 0.19), almost the same PR-AUC (0.161), and a "best"
# threshold of ≈ 0.83, far above 1/6. Re-weighting changed the *probabilities*, barely the
# *ranking*. Anyone who applied the 1/6 rule to these scores would call almost
# everyone.

# %% [markdown]
# ### Recalibrating without leaking
#
# `CalibratedClassifierCV` learns a monotone map from score to probability.
# That map must be learned on rows the model was **not** trained on, and in a
# time-ordered problem on rows that come **after** the model's training rows.
# Here: fit the base model on train snapshots before 2024-01-01, and learn the
# calibration on the last three train months (2024-01..2024-03). Validation is
# only used to score the result.

# %%
from sklearn.calibration import CalibratedClassifierCV

cut = pd.Timestamp("2024-01-01")
early = train[train["snapshot_date"] + pd.Timedelta(days=30) <= cut]  # embargo, as in time_split
late = train[train["snapshot_date"] >= cut]
print(f"base model on {len(early):,} rows (..{early['snapshot_date'].max().date()}), "
      f"calibrator on {len(late):,} rows ({late['snapshot_date'].min().date()}..{late['snapshot_date'].max().date()})")


def calibrate(base, X, y_cal, method):
    """sklearn >= 1.6: FrozenEstimator; older versions: cv='prefit'."""
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(FrozenEstimator(base), method=method).fit(X, y_cal)
    except ImportError:
        return CalibratedClassifierCV(base, method=method, cv="prefit").fit(X, y_cal)


rows = []
for name, kw in [("logreg", {}), ("logreg balanced", {"class_weight": "balanced"})]:
    base = make_model("logreg", **kw).fit(early[FEATURES], early[LABEL])
    p_base = predict_proba(base, valid)
    rows.append({"model": name, "calibration": "none (fit on early train)", "brier": brier_score_loss(y, p_base),
                 "pr_auc": average_precision_score(y, p_base), "mean_score": p_base.mean(),
                 "best_threshold": best_threshold_by_value(y, p_base, VALUE_TP, COST_FP)[0]})
    for method in ["sigmoid", "isotonic"]:
        cal_model = calibrate(base, late[FEATURES], late[LABEL], method)
        p_cal = cal_model.predict_proba(valid[FEATURES])[:, 1]
        rows.append({"model": name, "calibration": method, "brier": brier_score_loss(y, p_cal),
                     "pr_auc": average_precision_score(y, p_cal), "mean_score": p_cal.mean(),
                     "best_threshold": best_threshold_by_value(y, p_cal, VALUE_TP, COST_FP)[0]})
calib = pd.DataFrame(rows).round(4)
calib

# %% [markdown]
# **Expected (our run):**
#
# * **Balanced logreg:** sigmoid calibration brings the Brier score from ≈ 0.2
#   down to ≈ 0.047, the mean score from ≈ 0.4 to ≈ 0.055, and the best
#   threshold back near 1/6. **PR-AUC does not change at all** with sigmoid:
#   a monotone map keeps the order of the scores, and PR-AUC depends only on
#   the order.
# * **Plain logreg:** calibration changes almost nothing (Brier 0.0468 before
#   and after). It was already calibrated; there was nothing to fix.
# * **Isotonic** is a step function: it merges nearby scores into ties, which
#   *does* change PR-AUC (here down by ≈ 0.015) and is noisy with only
#   ≈ 6,600 calibration rows and a few hundred positives. Prefer sigmoid at
#   this scale.
#
# Also compare with the model fitted on **all** train (Brier ≈ 0.0467, PR-AUC
# 0.163) against the early-train base model (PR-AUC 0.162): holding out three
# months to calibrate costs a little ranking quality because the base model
# sees less data. For this model the honest
# choice is **not to recalibrate at all** and to keep monitoring calibration
# (Chapter 15).

# %%
p_bal_cal = calibrate(make_model("logreg", class_weight="balanced").fit(early[FEATURES], early[LABEL]),
                      late[FEATURES], late[LABEL], "sigmoid").predict_proba(valid[FEATURES])[:, 1]
fig, ax = plt.subplots(figsize=(5, 5))
for label, p in [("logreg", scores), ("logreg balanced", scores_bal), ("balanced + sigmoid", p_bal_cal)]:
    t = calibration_table(y, p, n_bins=10)
    ax.plot(t["mean_predicted"], t["observed_rate"], marker="o", label=label)
ax.plot([0, 0.8], [0, 0.8], color="grey", ls="--", label="perfect calibration")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("mean predicted probability (bin)")
ax.set_ylabel("observed churn rate (bin)")
ax.set_title("Reliability diagram, validation")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Expected:** plain and recalibrated models hug the diagonal; the balanced
# model sits far to the right of it (it predicts ≈ 5-10× the true rate).
# Log axes keep the many low-score bins readable.
#
# > **Research bridge.** A BCI decoder with good balanced accuracy can still
# > produce confidences that mean nothing, and an adaptive system that acts on
# > "p > 0.9" then misbehaves. Same lesson: ranking quality and probability
# > quality are separate properties, measured separately.

# %% [markdown]
# > **Exercise 3 (written).** Your manager asks: *"Why is the threshold 0.19?
# > That seems low. And what happens if the cost of a call doubles?"* Write a
# > reply of five or six sentences with no jargon (no "precision", no
# > "calibration"), using the numbers you observed in this notebook. Write your
# > answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %%
print(f"notebook compute time: {time.time() - T0:.0f} s")

# %% [markdown]
# **Expected:** under a minute.

# %% [markdown]
# ## Key equations
#
# **Precision, recall, F1** at a threshold:
#
# $$
# P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}
# $$
#
# *In words:* how pure the flagged set is, how much of the target it covers,
# and their harmonic mean. *Example:* 3 months, 192 calls, 52 reach churners,
# 362 churners in total: $P = 52/192 = 0.27$, $R = 52/362 = 0.14$,
# $F_1 = 0.19$. (F1 weighs a missed churner and a wasted call equally, which
# is rarely the business's view; that is why this notebook uses costs.)
#
# **Expected value of a threshold and the break-even probability:**
#
# $$
# \text{EV}(t) = TP(t)\cdot V - FP(t)\cdot C,
# \qquad
# p^{*} = \frac{C}{V + C}
# $$
#
# *In words:* calling an account with probability $p$ is worth
# $pV - (1-p)C$, positive exactly when $p > p^{*}$. *Symbols:* $V$ = value of
# reaching a churner (\$150), $C$ = cost of a call (\$30). *Example:*
# $p^{*} = 30/180 = 0.167$; halving the save rate makes $V = 75$ and
# $p^{*} = 30/105 = 0.286$, the same as doubling $C$ to \$60
# ($60/210 = 0.286$).
#
# **Lift at k:** $\text{lift@}k = \text{precision@}k \,/\, \pi$. *Example:*
# top-40 precision 0.275 at a base rate $\pi = 0.052$ gives lift ≈ 5.3.
#
# **Log loss (binary cross-entropy):**
#
# $$
# \text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]
# $$
#
# *In words:* the average surprise of the true labels; a confident wrong
# answer is punished hard. It is what logistic regression minimises, which is
# why it tends to come out calibrated.
#
# **Brier score:**
#
# $$
# \text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2
# $$
#
# *In words:* mean squared error of the probabilities; the base-rate
# constant scores $\pi(1-\pi)$. *Example:* $\pi = 0.052$:
# $0.052 \times 0.948 = 0.0493$, the "constant-base-rate Brier" in 8.6.
#
# *Worked example for both* (five accounts): $y = (1, 0, 1, 0, 0)$,
# $p = (0.8, 0.6, 0.4, 0.2, 0.1)$.
# Log loss $= -\tfrac{1}{5}[\ln 0.8 + \ln 0.4 + \ln 0.4 + \ln 0.8 + \ln 0.9] = 2.384/5 = 0.477$.
# Brier $= (0.04 + 0.36 + 0.36 + 0.04 + 0.01)/5 = 0.162$.
# At $t = 0.5$: TP = 1, FP = 1, FN = 1, so $P = R = F_1 = 0.5$ and
# $\text{EV} = 150 - 30 = \$120$.

# %%
# Check every formula above against scikit-learn on the five accounts.
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

y5 = np.array([1, 0, 1, 0, 0])
p5 = np.array([0.8, 0.6, 0.4, 0.2, 0.1])
flag = p5 >= 0.5
tp, fp, fn = int((flag & (y5 == 1)).sum()), int((flag & (y5 == 0)).sum()), int((~flag & (y5 == 1)).sum())
P, R = tp / (tp + fp), tp / (tp + fn)
by_hand = {
    "precision": P, "recall": R, "f1": 2 * P * R / (P + R),
    "log_loss": -np.mean(y5 * np.log(p5) + (1 - y5) * np.log(1 - p5)),
    "brier": np.mean((p5 - y5) ** 2),
}
library = {
    "precision": precision_score(y5, flag), "recall": recall_score(y5, flag), "f1": f1_score(y5, flag),
    "log_loss": log_loss(y5, p5), "brier": brier_score_loss(y5, p5),
}
for k in by_hand:
    print(f"{k:<10} by hand {by_hand[k]:.4f}   sklearn {library[k]:.4f}")
print(f"EV at t=0.5: ${tp * VALUE_TP - fp * COST_FP:.0f};  p* = {COST_FP / (VALUE_TP + COST_FP):.4f}")

# %% [markdown]
# **Expected:** precision, recall and F1 all 0.5000, log loss 0.4768, Brier
# 0.1620, identical both ways; EV \$120 and $p^{*} = 0.1667$.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `best_threshold_by_value` returns 0.5 and value `-inf` or 0 | no threshold has positive value (costs too high, or no positives) | check the cost inputs and that `y` has positives; "call nobody" may genuinely be the answer |
# | `TypeError` / `InvalidParameterError` about `cv="prefit"` | sklearn ≥ 1.8 removed it | wrap the fitted model in `sklearn.frozen.FrozenEstimator` (sklearn ≥ 1.6), as `calibrate` does |
# | Calibrated Brier is *worse* than uncalibrated | the calibrator was learned on too few rows, or isotonic overfitted | use `method="sigmoid"`, more calibration rows, or skip calibration if the model was already calibrated |
# | Calibration looks perfect but test performance is poor | the calibrator was fitted on validation and evaluated on validation | learn the calibration on a slice that is neither the model's training rows nor the evaluation rows |
# | Top-k flags the wrong rows after you sorted `df` | flags returned in sorted order, not in `df`'s row order | compute ranks with `groupby(...).rank()` on a Series aligned to `df.index` |
# | `ValueError: Bin edges must be unique` from `calibration_table` | many identical scores (e.g. a dummy model) | fewer bins; `pd.qcut(..., duplicates="drop")` is already used |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Derive the break-even probability for calling a customer from VALUE_TP and COST_FP. What assumption makes the derived threshold valid?
# 2. The team can call 40 accounts a month. Why use top-k per month rather than a fixed threshold?
# 3. Which assumption is the threshold most sensitive to, and how would you measure it?
# 4. Does calibration change PR-AUC? ROC-AUC? The best threshold? Why?
# 5. Where must the calibration map be fitted in a time-ordered problem, and what goes wrong if you fit it on validation?
#
# **Next chapter:** [Chapter 9 · Error analysis & artifacts](../ch09-error-analysis-and-artifacts/notebook.ipynb) — find where the model fails, then package it with metadata, a registry and a model card.
