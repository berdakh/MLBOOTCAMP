# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 16 · Final evaluation & documentation
#
# **Part I · Chapter 16 · about 50 minutes · CPU only**
#
# Since Chapter 5 the test period (July-November 2024) has been locked away.
# In this chapter it is opened **once**, with every decision written down
# beforehand, so that the reported number is an estimate of future performance
# and not the best of many attempts. The chapter then turns that number into
# the documents that a reviewer, a hiring manager or a future maintainer will
# actually read.
#
# After this chapter you should be able to:
#
# 1. pre-register an evaluation (model version, threshold, metrics, baselines, decision rule) before scoring;
# 2. report a held-out result with customer-level uncertainty and a paired comparison against baselines;
# 3. explain honestly why test differs from validation, without re-tuning;
# 4. generate a README "Results" section and a model card whose every number comes from code;
# 5. give a 60-second pitch and a 5-minute walkthrough of the project, including its limitations.
#
# **Lab output:** `artifacts/preregistration.json`, `artifacts/final_report.json`,
# `artifacts/README_results.md`, `artifacts/model_card.md`.

# %% [markdown]
# **Read first:** [Chapter 16 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: the registry from Day 2 (v1 = logistic regression, v2 = boosted
# trees, the better on VALIDATION promoted to CURRENT). Builds in ~10-30 s if missing.
from churnlab import bootstrap
from grading import ch16 as grade

registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()
print(split.summary().drop(columns=["positive_rate"]))

# %% [markdown]
# **Expected:** a registry with versions `['v1', 'v2']` and `CURRENT=v1`, and
# three periods: train 25,278 rows, valid 6,956, test 12,127. The positive rate
# column is dropped on purpose: nothing about the test period, not even its
# base rate, is inspected before the plan is written.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Held-out test set** | data used exactly once, after every decision is fixed | Jul-Nov 2024 |
# | **Pre-registration** | writing the analysis plan, with its decision rule, before seeing the data | `test_plan.json` and its hash |
# | **Garden of forking paths** | many small, reasonable choices made after looking, which add up to optimism | changing a threshold after seeing test |
# | **Exploratory result** | anything decided after looking; reported, but labelled as such | test-period slices in 16.7 |
# | **Paired bootstrap difference** | the interval for model B minus model A, on the same resampled customers | ≈ +0.07 PR-AUC vs the 3-feature baseline |
# | **Operating point** | the threshold or top-k rule actually used to act | the validation threshold, applied to test |
# | **Precision@k** | share of the k flagged accounts that churn | per month, at the team's capacity |
# | **Validation-test gap** | how much the score changes on the untouched period | 0.163 on validation vs ≈ 0.197 on test |
# | **Model card** | intended use, evidence, limits, in one page | updated with test numbers |
# | **Elevator pitch** | the project in 60 seconds: problem, approach, result with uncertainty, limit | section 16.10 |

# %% [markdown]
# ## 16.1 · Why "once"
#
# Every time you look at test results and then change something (a feature, a
# threshold, which model to report), the test set becomes a little more like a
# validation set, and its number a little more optimistic. Nobody intends to
# cheat; it happens through many small, reasonable decisions (statisticians
# call it the *garden of forking paths*).
#
# > **Intuition.** The test set is a sealed envelope with the exam answers.
# > Open it once, at the end, in front of witnesses (the pre-registered plan).
# > Peek and re-seal it, and every later "grade" is partly self-marked.
# > *Where it breaks down:* even an honest single look is one sample of one
# > period: it says how the model did on those months, not on every future
# > month, which is what monitoring is for.
#
# The defence is the one clinical trials use: **write down the analysis before
# seeing the data**, then do exactly that analysis and report it whatever it
# says. Anything you decide to look at afterwards is allowed, but it is labelled
# *exploratory*.
#
# > **Connection to other fields.** The same rule holds in any study with
# > held-out subjects: once a held-out subject's accuracy has influenced a
# > design choice, that subject is no longer held out. Here the "held-out subject" is a future time period.
#
# What has already been fixed on train/validation (Chapters 2–9), and must not move:
#
# | Decision | Fixed where |
# |---|---|
# | Features and their point-in-time SQL | chapter 4 (`churnlab/features.py`) |
# | Split boundaries and the 30-day embargo | chapter 5 (`churnlab/splits.py`) |
# | Model family and hyperparameters, chosen by validation PR-AUC | chapters 6–7, `ensure_registry` |
# | Threshold, chosen by expected value on validation | chapter 8, stored in the bundle metadata |
#
# ## 16.2 · Candidate and baselines, on validation only
#
# First gather everything the pre-registration needs. Nothing here reads the
# test period.

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churnlab.artifacts import load_bundle
from churnlab.evaluation import best_threshold_by_value, classification_report
from churnlab.features import LABEL
from churnlab.modeling import predict_proba

CURRENT = registry.current()
model, meta = registry.load_current()
challengers = [v for v in registry.versions() if v != CURRENT]
challenger_version = challengers[-1]
challenger, challenger_meta = load_bundle(registry.root / challenger_version)

# The simple baseline from Chapter 2: 3 raw features, logistic regression, train period only.
BASELINE_COLS = ["sessions_4w", "tenure_days", "failed_payments_90d"]
baseline = LogisticRegression(max_iter=1000).fit(split.train[BASELINE_COLS], split.train[LABEL])
baseline_valid = baseline.predict_proba(split.valid[BASELINE_COLS])[:, 1]
baseline_threshold, _ = best_threshold_by_value(split.valid[LABEL], baseline_valid,
                                                bootstrap.VALUE_TP, bootstrap.COST_FP)

valid_summary = pd.DataFrame({
    f"{CURRENT} ({meta['model_name']}) CURRENT": meta["metrics"]["validation"],
    f"{challenger_version} ({challenger_meta['model_name']})": challenger_meta["metrics"]["validation"],
    "3-feature baseline": classification_report(split.valid[LABEL], baseline_valid),
}).T[["n", "positive_rate", "pr_auc", "roc_auc", "brier"]]
print(f"CURRENT = {CURRENT} ({meta['model_name']}), threshold from validation = {meta['threshold']}")
valid_summary.round(4)

# %% [markdown]
# **Expected:** CURRENT is `v1` (logistic regression) with validation PR-AUC
# ≈ 0.163 and ROC-AUC ≈ 0.733; `v2` (boosted trees) ≈ 0.135; the 3-feature
# baseline lower still. CURRENT's threshold is ≈ 0.19 (the expected-value
# optimum on validation; the theoretical break-even for a calibrated score
# is 1/6 ≈ 0.167).

# %% [markdown]
# ## 16.3 · Pre-register
#
# This cell writes the analysis plan to disk **before** the test period is
# read. It records the exact model file hash, so "which model?" can never be
# ambiguous later, and a hash of the plan itself, which the final report will
# quote.

# %%
import hashlib
import json
from datetime import datetime, timezone

N_BOOT, SEED = 500, 0

plan = {
    "question": "How well does the CURRENT churn model rank and flag accounts in an unseen later period?",
    "test_period": {"first_snapshot": "2024-07-01", "rule": "snapshot_date >= splits.VALID_END"},
    "model": {"version": CURRENT, "name": meta["model_name"], "model_sha256": meta["model_sha256"],
              "trained_on": meta["training_data"]},
    "threshold": {"value": meta["threshold"], "chosen_on": "validation",
                  "method": "maximise expected value with VALUE_TP and COST_FP"},
    "cost_assumptions": {"value_tp": bootstrap.VALUE_TP, "cost_fp": bootstrap.COST_FP,
                         "status": "teaching assumptions, not measured business values"},
    "primary_metric": "pr_auc",
    "secondary_metrics": ["roc_auc", "brier", "precision@threshold", "recall@threshold",
                          "flagged@threshold", "expected_value@threshold"],
    "uncertainty": {"method": "cluster bootstrap by customer_id", "n_boot": N_BOOT, "seed": SEED,
                    "interval": "95% percentile"},
    "baselines": {
        "dummy_prior": "constant score = training positive rate (PR-AUC equals the test positive rate)",
        "three_feature_logreg": {"features": BASELINE_COLS, "threshold": round(baseline_threshold, 4)},
        "challenger": {"version": challenger_version, "name": challenger_meta["model_name"],
                       "threshold": challenger_meta["threshold"]},
    },
    "decision_rule": ("CURRENT is reported as better than a baseline only if the 95% paired cluster-bootstrap "
                      "CI of PR-AUC(CURRENT) - PR-AUC(baseline) lies entirely above 0. All numbers are "
                      "reported whatever the outcome. Nothing is re-tuned on test."),
    "exploratory_after_scoring": ["per-snapshot operating point", "calibration", "slices by plan, size, region"],
}
plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]

prereg_path = ART / "preregistration.json"
if prereg_path.exists():
    previous = json.loads(prereg_path.read_text())
    if previous.get("plan_sha256") != plan_sha:
        archived = prereg_path.with_name(f"preregistration.superseded-{previous.get('plan_sha256')}.json")
        prereg_path.rename(archived)
        print(f" the plan CHANGED since {previous.get('created_at')}; old plan kept as {archived.name}. "
              "If you had already seen test results, say so in the report.")
prereg_path.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "plan_sha256": plan_sha, "plan": plan}, indent=2, default=str))
print("pre-registered:", prereg_path.name, "plan_sha256 =", plan_sha)
print(plan["decision_rule"])

# %% [markdown]
# **Expected:** `pre-registered: preregistration.json plan_sha256 = <16 hex>`.
# In a real team this file is committed (or posted in the team channel) before
# anyone runs the next cell; the timestamp is the point.
#
# ## 16.4 · Score the test period
#
# The test rows are read through one function that refuses to work without a
# pre-registration and logs every access. Re-running the *same* plan is fine
# (deterministic, same numbers); the log makes it visible if anything else
# happened.

# %%
def load_test_once(split, prereg_path=prereg_path, log_path=ART / "test_access.log"):
    if not prereg_path.exists():
        raise RuntimeError("no pre-registration on disk: write the plan before touching test")
    sha = json.loads(prereg_path.read_text())["plan_sha256"]
    with open(log_path, "a") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} plan={sha}\n")
    accesses = len(log_path.read_text().splitlines())
    print(f"test access #{accesses} under plan {sha}")
    return split.test


test = load_test_once(split)
y = test[LABEL].to_numpy()
groups = test["customer_id"].to_numpy()

scores = {
    f"{CURRENT} CURRENT": predict_proba(model, test),
    f"{challenger_version} challenger": predict_proba(challenger, test),
    "3-feature baseline": baseline.predict_proba(test[BASELINE_COLS])[:, 1],
    "dummy prior": np.full(len(test), split.train[LABEL].mean()),
}
cur_key = f"{CURRENT} CURRENT"

test_table = pd.DataFrame({name: classification_report(y, s) for name, s in scores.items()}).T
test_table[["n", "positive_rate", "pr_auc", "roc_auc", "brier", "lift@10%"]].round(4)

# %% [markdown]
# **Expected:** `test access #1`, then on 12,127 test rows with positive rate
# ≈ 0.059: CURRENT PR-AUC ≈ 0.20 and ROC-AUC ≈ 0.74; the challenger ≈ 0.15;
# the 3-feature baseline ≈ 0.12; the dummy prior's PR-AUC equals the positive
# rate and its ROC-AUC is 0.5. The order of the models is the same as on
# validation.

# %% [markdown]
# ## 16.5 · Uncertainty, and the pre-registered decision
#
# Point estimates first, intervals always. Rows of one customer are correlated
# (the same account appears at up to five test snapshots), so resampling
# **customers** is the honest bootstrap. The paired version resamples the same
# customers for both models, which is what makes a small difference detectable.

# %%
from sklearn.metrics import average_precision_score, roc_auc_score

from churnlab.evaluation import bootstrap_ci, paired_bootstrap_diff

s_cur = scores[cur_key]
pr_point, pr_lo, pr_hi = bootstrap_ci(y, s_cur, average_precision_score, n_boot=N_BOOT, seed=SEED, groups=groups)
roc_point, roc_lo, roc_hi = bootstrap_ci(y, s_cur, roc_auc_score, n_boot=N_BOOT, seed=SEED, groups=groups)
print(f"PR-AUC  {pr_point:.3f}  95% CI [{pr_lo:.3f}, {pr_hi:.3f}]   (customer bootstrap, n_boot={N_BOOT})")
print(f"ROC-AUC {roc_point:.3f}  95% CI [{roc_lo:.3f}, {roc_hi:.3f}]")

comparisons = {}
for name in ("3-feature baseline", f"{challenger_version} challenger"):
    d = paired_bootstrap_diff(y, scores[name], s_cur, average_precision_score,
                              n_boot=N_BOOT, seed=SEED, groups=groups)
    d["current_better_by_rule"] = d["ci_low"] > 0
    comparisons[name] = d
comparisons["dummy prior"] = {"diff": pr_point - float(y.mean()), "ci_low": pr_lo - float(y.mean()),
                              "ci_high": pr_hi - float(y.mean()), "current_better_by_rule": pr_lo > y.mean()}
pd.DataFrame(comparisons).T.round(4)

# %% [markdown]
# **Expected:** PR-AUC ≈ 0.197 with a customer-level 95% CI of roughly
# [0.17, 0.22]; ROC-AUC ≈ 0.74 [0.72, 0.76]. Against the 3-feature baseline the
# paired PR-AUC difference is ≈ +0.07 with a CI clear of zero; against the
# challenger ≈ +0.045, also clear of zero. So by the pre-registered rule,
# CURRENT is reported as better than all three baselines **on this test
# period**. (The dummy row is a sanity floor, not a paired bootstrap: the
# whole CI should sit far above the positive rate.)
#
# Notice what the interval says: PR-AUC is known to about ±0.03 from five months
# of data on ~2,500 customers. A future "improvement" of 0.01 would be invisible
# without a paired test.

# %% [markdown]
# ## 16.6 · The operating point, month by month
#
# A ranking metric is not a decision. The decision is "call the accounts above
# the threshold", so report what that does each month: how many calls, what
# fraction were real churners (precision), what fraction of churners were
# reached (recall). The threshold is the **pre-registered** one.

# %%
from churnlab.evaluation import confusion_at, expected_value

THRESHOLD = meta["threshold"]

per_snapshot = []
for snap, g in test.assign(_s=s_cur).groupby("snapshot_date"):
    c = confusion_at(g[LABEL], g["_s"], THRESHOLD)
    flagged = c["tp"] + c["fp"]
    per_snapshot.append({"snapshot": str(snap.date()), "rows": len(g), "positive_rate": g[LABEL].mean(),
                         "flagged": flagged, "precision": c["tp"] / flagged if flagged else np.nan,
                         "recall": c["tp"] / (c["tp"] + c["fn"]), **c})
per_snapshot = pd.DataFrame(per_snapshot)
overall = confusion_at(y, s_cur, THRESHOLD)
print(f"threshold {THRESHOLD}: flagged {overall['tp'] + overall['fp']:,} of {len(y):,} rows, "
      f"precision {overall['tp'] / (overall['tp'] + overall['fp']):.3f}, "
      f"recall {overall['tp'] / (overall['tp'] + overall['fn']):.3f}")
per_snapshot.round(3)

# %% [markdown]
# **Expected:** about 60 accounts flagged per month (≈ 310 in total), precision
# ≈ 0.29 and recall ≈ 0.13 overall, with month-to-month variation (a precision
# of 0.2 one month and 0.35 the next is normal at these volumes: 60 calls is a
# small sample).
#
# Expected value **under the course's stated assumptions** ($150 per true
# positive reached, $30 per call that was not needed; these are teaching
# assumptions on synthetic data, not facts about any business):

# %%
VALUE_TP, COST_FP = bootstrap.VALUE_TP, bootstrap.COST_FP

policies = {
    f"{CURRENT} at pre-registered threshold": confusion_at(y, s_cur, THRESHOLD),
    "3-feature baseline at its validation threshold": confusion_at(y, scores["3-feature baseline"],
                                                                     baseline_threshold),
    f"{challenger_version} at its validation threshold": confusion_at(
        y, scores[f"{challenger_version} challenger"], challenger_meta["threshold"]),
    "call everyone": confusion_at(y, s_cur, -1.0),
    "call no one": confusion_at(y, s_cur, 2.0),
}
ev = pd.DataFrame([{"policy": k, "calls": c["tp"] + c["fp"], "tp": c["tp"], "fp": c["fp"],
                    "expected_value_usd (ASSUMED costs)": expected_value(c, VALUE_TP, COST_FP)}
                   for k, c in policies.items()]).set_index("policy")
ev

# %% [markdown]
# **Expected:** CURRENT at its threshold has a positive assumed value (a few
# thousand dollars over the five test months: ≈ 90 true positives × $150 −
# ≈ 220 unneeded calls × $30); "call everyone" is strongly negative; "call no
# one" is 0. In an interview, state it in exactly those terms: *"under assumed costs of
# $150 and $30, which the business would have to confirm"*. Never present
# it as money the model made: nobody was called, and the 25% "save rate"
# behind $150 is an assumption too.

# %% [markdown]
# ## 16.7 · Calibration and slices on test (exploratory)
#
# These were listed as *exploratory* in the plan: worth looking at, not part of
# the headline claim.

# %%
import matplotlib.pyplot as plt

from churnlab.evaluation import calibration_table, slice_report

cal = calibration_table(y, s_cur, n_bins=10)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot([0, cal["mean_predicted"].max() * 1.1], [0, cal["mean_predicted"].max() * 1.1], "--", color="grey",
        label="perfect calibration")
ax.plot(cal["mean_predicted"], cal["observed_rate"], "o-", label=f"{CURRENT} on test")
ax.set_xlabel("mean predicted probability (decile)")
ax.set_ylabel("observed churn rate")
ax.set_title("Calibration on the test period")
ax.legend()
plt.show()
print(f"mean predicted {s_cur.mean():.4f} vs observed {y.mean():.4f}; "
      f"Brier {test_table.loc[cur_key, 'brier']:.4f}")
cal.round(4)

# %% [markdown]
# **Expected:** points close to the diagonal at low scores, and mostly
# **above** it in the upper-middle deciles: the model under-predicts a little
# on test (mean predicted ≈ 0.05 vs observed ≈ 0.059). That is the signature of
# a base rate that rose after training, not of a broken model. It matters
# because the threshold was derived from calibrated probabilities: if the
# shift persists, recalibrating on recent labelled months (chapter 8) is the
# fix, done on a *new* validation window, not on this test set.

# %%
slices = {by: slice_report(test, s_cur, by, threshold=THRESHOLD).round(3)
          for by in ("plan", "company_size", "region")}
for by, table in slices.items():
    print(f"\n--- by {by} ---")
    print(table.to_string(index=False))

# %% [markdown]
# **Expected:** churn and flag rates differ strongly by plan (basic accounts
# churn most and receive most of the flags); recall is low in every slice.
# No slice is under 50 rows here. The weakest slice is `region=unknown` (the
# ~200 rows whose region is truly missing): lower PR-AUC and almost nobody
# flagged. With real data that would be the first thing to investigate (is
# "missing region" a sign of a different kind of account?). On synthetic data it
# tells you how the generator was written, so treat it as practice for the
# method.

# %% [markdown]
# ## 16.8 · Validation vs test, honestly
#
# Test PR-AUC is **higher** than validation. Before interpreting this, we line
# up the two periods:

# %%
val = meta["metrics"]["validation"]
tst = test_table.loc[cur_key]
compare = pd.DataFrame({
    "validation (2024-04..06)": [val["n"], val["positive_rate"], val["pr_auc"], val["pr_auc"] / val["positive_rate"],
                                 val["roc_auc"], val["brier"]],
    "test (2024-07..11)": [tst["n"], tst["positive_rate"], tst["pr_auc"], tst["pr_auc"] / tst["positive_rate"],
                           tst["roc_auc"], tst["brier"]],
}, index=["rows", "positive_rate", "pr_auc", "pr_auc / positive_rate", "roc_auc", "brier"])
compare["test - validation"] = compare.iloc[:, 1] - compare.iloc[:, 0]

# The validation number has its own uncertainty: same customer bootstrap, same settings.
valid_scores = predict_proba(model, split.valid)
_, v_lo, v_hi = bootstrap_ci(split.valid[LABEL].to_numpy(), valid_scores, average_precision_score,
                             n_boot=N_BOOT, seed=SEED, groups=split.valid["customer_id"].to_numpy())
print(f"PR-AUC 95% CI  validation [{v_lo:.3f}, {v_hi:.3f}]   test [{pr_lo:.3f}, {pr_hi:.3f}]")
compare.round(4)

# %% [markdown]
# **Expected:** PR-AUC intervals of roughly [0.13, 0.20] on validation and
# [0.17, 0.22] on test; positive rate ≈ 0.052 → 0.059; PR-AUC ≈ 0.163 → 0.197;
# PR-AUC divided by the positive rate ≈ 3.1 → 3.3; ROC-AUC ≈ 0.733 → 0.740.
#
# How to say it:
#
# * **PR-AUC depends on the base rate.** A random ranking scores the positive
#   rate, so a period with more churners raises PR-AUC with no change in the
#   model. Normalised by the base rate, and on ROC-AUC (which does not depend on
#   it), validation and test are close.
# * **The two intervals overlap.** Validation's point estimate sits just
#   below the test interval, but validation has its own interval, as wide or
#   wider (fewer rows), and the two overlap substantially. "Test is better" is not a
#   finding; "performance held up in a later period with a higher churn rate" is.
# * **Why the rate rose is unknown** from this data alone: seasonality, a price
#   change, or a generator setting. With real data you would ask the business
#   what happened in July-August 2024 before anything else.
# * **What you must not do:** pick a new threshold or model because of these
#   numbers. If test had been *worse*, the same rule would apply: report it,
#   explain it, and fix it on the next validation window.

# %% [markdown]
# > **Exercise 1.** Write `final_report(y, scores, threshold, groups, seed)`
# > returning a dict with keys `pr_auc`, `pr_auc_ci` (a `(low, high)` tuple from
# > a **customer-level** bootstrap, use `churnlab.evaluation.bootstrap_ci` with
# > its default `n_boot` and the given `seed`), `roc_auc`, `precision`, `recall`
# > and `flagged` (rows with `score >= threshold`). The grader checks it against
# > `churnlab.evaluation` on a fixed synthetic dataset, including that your
# > interval really resamples groups.

# %%
def final_report(y, scores, threshold: float, groups, seed: int = 0) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(final_report)

# %% [markdown]
# > **Exercise 2.** Write `results_markdown(report)` that turns a report dict
# > into a two-column markdown table `| metric | value |`, one row per key **in
# > the dict's order**, first cell = the key. Floats with 3 decimals, integers as
# > integers, and a `(low, high)` tuple as `[low, high]` with 3 decimals each.
# > The grader parses your table, so it must be a real markdown table (header,
# > `|---|---|` separator, rows).

# %%
def results_markdown(report: dict) -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(results_markdown)

# %% [markdown]
# ## 16.9 · Write it down: final report, README results, model card
#
# Everything below is generated from one dictionary. **No number is typed by
# hand**: a hand-typed number in a README is a number that will eventually be
# wrong. If you re-run with a different model, the documents update.

# %%
def r(x, nd=6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


final = {
    "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "preregistration_sha256": plan_sha,
    "data": "SYNTHETIC TaskFlow extract (churnlab/data.py); not real customers",
    "model": {"version": CURRENT, "name": meta["model_name"], "model_sha256": meta["model_sha256"],
              "threshold": THRESHOLD, "training_data": meta["training_data"]},
    "test": {
        "period": f"{test['snapshot_date'].min().date()}..{test['snapshot_date'].max().date()}",
        "rows": int(len(test)), "customers": int(test["customer_id"].nunique()),
        "positive_rate": r(y.mean()),
        "pr_auc": r(pr_point), "pr_auc_ci": [r(pr_lo), r(pr_hi)],
        "roc_auc": r(roc_point), "roc_auc_ci": [r(roc_lo), r(roc_hi)],
        "brier": r(tst["brier"]),
        "flagged": int(overall["tp"] + overall["fp"]),
        "precision": r(overall["tp"] / (overall["tp"] + overall["fp"])),
        "recall": r(overall["tp"] / (overall["tp"] + overall["fn"])),
        "expected_value_usd_assumed": float(ev.iloc[0, -1]),
        "ci_method": f"cluster bootstrap by customer, n_boot={N_BOOT}, seed={SEED}",
    },
    "validation": {k: r(val[k]) for k in ("positive_rate", "pr_auc", "roc_auc", "brier")}
    | {"rows": val["n"], "pr_auc_ci": [r(v_lo), r(v_hi)]},
    "baselines_test_pr_auc": {name: r(test_table.loc[name, "pr_auc"]) for name in scores if name != cur_key},
    "paired_vs_current": {name: {k: (r(v) if not isinstance(v, (bool, np.bool_)) else bool(v))
                                 for k, v in d.items()} for name, d in comparisons.items()},
    "per_snapshot": per_snapshot[["snapshot", "rows", "positive_rate", "flagged", "precision", "recall"]]
    .round(4).to_dict(orient="records"),
    "cost_assumptions": plan["cost_assumptions"],
}
(ART / "final_report.json").write_text(json.dumps(final, indent=2, default=str))
print(json.dumps(final["test"], indent=2))

# %% [markdown]
# **Expected:** the `test` block printed with the same numbers as sections
# 16.4–16.6, and `final_report.json` written. Now the documents. A tiny
# formatter keeps them consistent:

# %%
def fmt(x, nd=3):
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.{nd}f}"
    return str(x)


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in rows]
    return "\n".join(out)


T, V = final["test"], final["validation"]
headline_rows = [
    {"metric": "PR-AUC (primary)", "validation": V["pr_auc"], "test": T["pr_auc"], "test 95% CI": T["pr_auc_ci"]},
    {"metric": "ROC-AUC", "validation": V["roc_auc"], "test": T["roc_auc"], "test 95% CI": T["roc_auc_ci"]},
    {"metric": "Brier score", "validation": V["brier"], "test": T["brier"], "test 95% CI": "-"},
    {"metric": "positive rate", "validation": V["positive_rate"], "test": T["positive_rate"], "test 95% CI": "-"},
]
baseline_rows = [{"compared with": name, "test PR-AUC": final["baselines_test_pr_auc"][name],
                  "paired diff (CURRENT - it)": d["diff"],
                  "95% CI": [d["ci_low"], d["ci_high"]], "better by pre-registered rule": d["current_better_by_rule"]}
                 for name, d in final["paired_vs_current"].items()]
ARCH = "https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/architecture.svg"

readme = f"""## Results

> All data is **synthetic** (generated by `src/churnlab/data.py`). These numbers describe
> the method on a realistic-shaped problem, not any real company's customers.

Evaluation of model **{final['model']['version']}** ({final['model']['name']}) on the held-out
test period **{T['period']}** ({fmt(T['rows'])} account-months, {fmt(T['customers'])} customers),
scored once under a pre-registered plan (`preregistration.json`, sha256 `{plan_sha}`).
Intervals: {T['ci_method']}.

{md_table(headline_rows)}

**Against baselines** (paired customer bootstrap on the same test rows):

{md_table(baseline_rows)}

**Operating point.** At the threshold chosen on validation ({fmt(final['model']['threshold'], 4)}),
the model flags {fmt(T['flagged'])} account-months over the test period
(precision {fmt(T['precision'])}, recall {fmt(T['recall'])}). Under the course's
**assumed** costs (value of a reached churner ${fmt(final['cost_assumptions']['value_tp'], 0)},
cost of an unnecessary call ${fmt(final['cost_assumptions']['cost_fp'], 0)}) that policy has an
expected value of ${fmt(T['expected_value_usd_assumed'], 0)}; the assumptions are not measured.

**Validation vs test.** Validation PR-AUC was {fmt(V['pr_auc'])} (95% CI {fmt(V['pr_auc_ci'])}).
The test positive rate ({fmt(T['positive_rate'])}) is higher than validation's
({fmt(V['positive_rate'])}), which raises PR-AUC by itself. ROC-AUC, which does not depend on the
base rate, moved from {fmt(V['roc_auc'])} to {fmt(T['roc_auc'])}. Nothing was re-tuned on test.

## Architecture

<img src="{ARCH}" alt="Raw extract, validation, SQL features, split, pipeline, registry, FastAPI service, monitoring" width="100%">

Raw extract (checksummed) → validation and cleaning → point-in-time SQL features (DuckDB)
→ time split with a 30-day embargo → scikit-learn pipeline → versioned registry with a
CURRENT pointer → FastAPI service → monitoring (drift, delayed labels) with rollback.

## Limitations

* **Synthetic data.** The generator decides what is learnable; no claim transfers to real customers.
* **Offline evaluation only.** Ranking quality on past data is not the effect of calling customers.
  Only a randomised experiment (call a random subset of flagged accounts) measures that.
* **Contacting customers has side effects.** A call can annoy, or remind someone to cancel.
* **Feedback loops.** Once the team acts on scores, labels change: a saved customer looks like a
  false positive. Future training data must record who was contacted.
* **Cost assumptions** behind the threshold and the expected value are placeholders to be agreed
  with the business.
"""

slice_rows = [{"slice": f"{by}={row[by]}", "rows": int(row["rows"]), "positive_rate": row["positive_rate"],
               "pr_auc": row.get("pr_auc", float("nan")), "recall": row["recall"], "flag_rate": row["flag_rate"]}
              for by, table in slices.items() for row in table.to_dict(orient="records")]

model_card = f"""# Model card · TaskFlow churn · {final['model']['version']}

*Generated by Chapter 16 on {final['generated_at']}. Every number below is computed, not typed.*

## Model
* **Type:** {final['model']['name']} in a scikit-learn Pipeline (preprocessing included)
* **Version / file hash:** {final['model']['version']} / `{final['model']['model_sha256'][:16]}...`
* **Decision threshold:** {fmt(final['model']['threshold'], 4)} (expected-value optimum on validation)

## Intended use
Monthly ranking of **active** TaskFlow accounts by probability of cancelling in the next 30 days,
so a customer-success team can prioritise outreach. Out of scope: automated actions without a
human, pricing decisions, accounts younger than one snapshot, any real company's data.

## Training data
{fmt(final['model']['training_data']['rows'])} account-months, snapshots
{final['model']['training_data']['first_snapshot']} to {final['model']['training_data']['last_snapshot']}
(fingerprint `{final['model']['training_data']['fingerprint']}`). **Synthetic.**

## Evaluation
Validation: {fmt(V['rows'])} rows. Test (touched once, pre-registered): {fmt(T['rows'])} rows, {T['period']}.

{md_table(headline_rows)}

Slices on test (exploratory):

{md_table(slice_rows)}

## Caveats
* Mean predicted probability on test was {fmt(float(s_cur.mean()), 4)} against an observed rate of
  {fmt(T['positive_rate'], 4)}: slightly under-confident after the base rate rose. Recalibrate on recent
  labelled data before trusting the probabilities for cost calculations.
* Labels arrive 30 days late; monitoring must use proxies (score and feature drift) in between.
* Once interventions start, outcomes are affected by them (feedback loop); log every contact.
* Cost assumptions: ${fmt(final['cost_assumptions']['value_tp'], 0)} per reached churner,
  ${fmt(final['cost_assumptions']['cost_fp'], 0)} per unnecessary call: **assumed**, not measured.
"""

(ART / "README_results.md").write_text(readme)
(ART / "model_card.md").write_text(model_card)
print(readme[:1500])

# %% [markdown]
# **Expected:** the start of the Results section, with the headline table
# filled from the dictionary, and two files written. Open
# `artifacts/model_card.md` and read it as a stranger would. If a sentence
# would not survive a reviewer asking "how do you know?", fix the generator,
# not the file.
#
# Prove the "no hand-typed numbers" claim mechanically: every 3-decimal number
# in the README must be traceable to the report.

# %%
import re

allowed = set()


def _collect(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v)
    elif isinstance(obj, (float, int, np.floating, np.integer)) and not isinstance(obj, bool):
        allowed.add(f"{float(obj):.3f}")
        allowed.add(f"{float(obj):.4f}")


_collect(final)
numbers = set(re.findall(r"\d+\.\d{3,4}", readme))
untraced = sorted(n for n in numbers if n not in allowed)
print(f"{len(numbers)} decimal numbers in the README; not traceable to final_report.json: {untraced or 'none'}")

# %% [markdown]
# **Expected:** `not traceable to final_report.json: none`.

# %% [markdown]
# ## 16.10 · Saying it: the 60-second pitch and the 5-minute walkthrough
#
# The documents are for readers. Interviews are spoken, and the first minute
# decides how the rest goes. Two outlines to adapt (use your **own** numbers from
# `final_report.json`, and say "synthetic" early):
#
# **60-second pitch** (one breath per line):
#
# 1. *Problem:* which active accounts of a subscription business will cancel in the next 30 days, so a team with limited time can call the right ones.
# 2. *Data:* a synthetic four-table extract with planted defects; I validated it first and found a region code pandas reads as missing and a leaking status column.
# 3. *Method:* point-in-time SQL features, a time split with an embargo, a pipeline whose preprocessing ships with the model, threshold from stated costs.
# 4. *Result:* on a test period touched once under a pre-registered plan, PR-AUC ≈ X with a customer-level interval, clearly above a simple baseline; logistic regression beat boosted trees.
# 5. *Engineering:* a versioned registry, a tested FastAPI service, drift monitoring and a rehearsed rollback.
# 6. *What I'd do with real data:* a randomised outreach experiment to measure the effect of calling, not just the ranking.
#
# **5-minute walkthrough** (about 40 seconds each, with the architecture diagram on screen):
#
# | # | Beat | The one thing to make clear |
# |---|---|---|
# | 1 | Decision & metric | Who acts on the score and why PR-AUC + a cost-based threshold, not accuracy |
# | 2 | Data & validation | The planted defects you found and the contract that now catches them |
# | 3 | Features & leakage | "A feature for snapshot T uses only rows before T"; the embargo; the leaky columns you refused |
# | 4 | Model selection | Validation with intervals; why the linear model won; what you did not tune |
# | 5 | Test result | Pre-registration, the interval, the baseline comparison, why test > validation is not good news |
# | 6 | Serving | One pipeline object, versions in every response, schema validation, tests |
# | 7 | Operations | Health vs readiness, latency, PSI drift with delayed labels, rollback by pointer |
# | 8 | Limitations & next | Synthetic, offline only, side effects, feedback loops; the experiment you'd run |
#
# Record yourself doing both (a phone is sufficient). The recorded walkthrough
# is the most useful single piece of evidence from Part I.
#
# ## 16.11 · Limitations a skeptical interviewer will raise
#
# These are already in the README, because raising them first is more
# convincing than answering them later:
#
# * **Synthetic data.** The model can only find what the generator put there.
# * **Offline evaluation.** A good ranking of who *would* churn is not evidence
#   that *calling them* changes anything. Uplift needs an experiment.
# * **Side effects.** Contacting an account is an intervention with possible harm.
# * **Feedback loops.** Once you intervene, labels change: a retained customer
#   becomes a "false positive", and a model retrained naively on that data learns
#   to stop flagging exactly the people the calls helped.

# %% [markdown]
# > **Exercise 3 (written).** Write your 60-second pitch for this project
# > (or your own capstone) using **your** numbers from `final_report.json`. Then
# > read it aloud with a timer. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# > **Exercise 4 (written).** Name three limitations a skeptical interviewer
# > would raise about this project, and for each give the answer you would say
# > out loud (what you did about it, or what you would do with real data). Write
# > your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **Pre-registered decision rule.** With a paired customer bootstrap of
# $\Delta = \text{PR-AUC}_{\text{CURRENT}} - \text{PR-AUC}_{\text{baseline}}$:
#
# $$
# \text{report "better"} \iff Q_{0.025}\big(\Delta^{*}_{1..B}\big) > 0
# $$
#
# *In words:* the whole 95% interval of the difference must sit above zero.
# *Example:* $\Delta \approx +0.07$ with an interval clear of zero → "better
# than the 3-feature baseline on this test period".
#
# **Precision at the team's capacity, per month $m$:**
#
# $$
# \text{precision@}k_m = \frac{\#\{\text{churners among the top } k \text{ scores in month } m\}}{k}
# $$
#
# *Example:* 11 churners among 40 calls → 0.275; lift = $0.275 / \pi_m$.
#
# **Interval width from sample size.** A bootstrap interval narrows roughly as
# $1 / \sqrt{n_{\text{customers}}}$. *Example:* ±0.03 on ≈ 2,500 customers
# means about ±0.06 on 625, and you would need ≈ 10,000 customers for ±0.015.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `the plan CHANGED since ...` | you edited the plan cell after an earlier run | fine while learning; in real work, report both plans and which one you saw results under |
# | `test access #2`, `#3`, ... | the notebook was re-run in the same artifact directory | harmless if the plan hash is unchanged; that is what the log is for |
# | Registry build takes minutes | first run in a fresh runtime trains both models; a busy machine slows the tree model | wait (normally 10-30 s); it is cached in `artifacts/registry` afterwards |
# | `UserWarning: model trained with sklearn X, running Y` | registry built under another sklearn version | delete `artifacts/registry` and re-run; never compare numbers across mixed versions |
# | Your test numbers differ slightly from the Expected lines | different library versions | fine within the stated ranges; the CI width tells you how much is noise |
# | README check lists untraced numbers | you hand-typed a value into the template | replace it with a field from `final` |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. What did you pre-register, and why does the order (plan first, scores second) matter?
# 2. Test PR-AUC is higher than validation PR-AUC. Is the model better than you thought? Explain with numbers.
# 3. Why a customer-level bootstrap rather than a row bootstrap, and why a *paired* bootstrap for the baseline comparison?
# 4. The expected value at the threshold is positive. Why can't you tell the business "the model is worth $X"?
# 5. After deployment the team calls every flagged account. What happens to next quarter's training labels, and what would you log to handle it?
# 6. Give the three limitations you would raise yourself before an interviewer does.
#
# **Next chapter:** [Chapter 17 · Python & SQL drills](../ch17-python-and-sql-drills/notebook.ipynb) — timed Python and SQL drills, which train the fluency part of a technical interview.

