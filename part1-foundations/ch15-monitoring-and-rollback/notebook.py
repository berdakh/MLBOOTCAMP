# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 15 · Monitoring & rollback
#
# **Part I · Chapter 15 · about 60 minutes · CPU only**
#
# A deployed model is scored on data it has never seen, by a world that keeps
# changing, and its mistakes become visible only when the outcomes arrive, here
# 30 days later. This notebook simulates the next half-year of production
# traffic with a deliberate shift in it, measures what monitoring can and
# cannot reveal, then deploys a broken model on purpose, detects it with a
# canary check, and rolls it back.
#
# After this chapter you should be able to:
#
# 1. name the three layers of monitoring and when each becomes available;
# 2. compute and *interpret* PSI for numeric and categorical features, including where PSI is blind or misleading;
# 3. measure score drift and, once labels mature, real performance, and say honestly whether the model degraded;
# 4. turn measurements into alert rules;
# 5. detect a bad deploy with a canary comparison, roll it back in one step, and show the audit trail;
# 6. choose between shadow, canary and blue/green releases, and between retraining triggers.
#
# **Lab output:** `artifacts/registry_nb14/` (a sandbox copy of the registry, with a broken
# `v3`, its promotion and its rollback in `history.jsonl`) and `artifacts/monitoring/` (drift and
# performance tables as CSV).

# %% [markdown]
# **Read first:** [Chapter 15 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: the registry (Chapter 9) and the fixed split (Chapter 5) --
import shutil

from churnlab import bootstrap
from grading import ch15 as grade
from churnlab.registry import Registry

base_registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()

# This notebook deploys a broken model on purpose. It works on a COPY of the
# registry, so re-running it always starts from the same state and the course
# registry that Chapter 16 uses is never touched.
OPS_DIR = ART / "registry_nb14"
if OPS_DIR.exists():
    shutil.rmtree(OPS_DIR)
shutil.copytree(base_registry.root, OPS_DIR)
ops = Registry(OPS_DIR)
MON = ART / "monitoring"
MON.mkdir(exist_ok=True)
print("sandbox registry:", ops.root, "| versions", ops.versions(), "| CURRENT", ops.current())

# %% [markdown]
# **Expected:** `versions ['v1', 'v2'] | CURRENT v1`.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Monitoring** | measuring a live system continuously so someone can act | service health, drift, performance |
# | **Data (input) drift** | the distribution of a feature changes | sessions fall by a fifth |
# | **Prediction drift** | the distribution of scores changes | mean score ratio above 1.5 |
# | **Concept drift** | the relation between features and churn changes | the same usage now means a different risk |
# | **PSI** | Population Stability Index: one number for how far a distribution moved from a reference | 0.1-0.25 = moderate |
# | **Label delay** | outcomes arrive only after the label window closes | 30 days |
# | **Matured cohort** | snapshots whose labels are all known | the latest month older than 30 days |
# | **Alert severity** | *page* wakes someone now; *warn* opens a ticket | error rate → page; drift → warn |
# | **Shadow / canary / blue-green** | ways to expose a new model gradually or switch back fast | canary = 1-10% of traffic |
# | **Rollback** | point `CURRENT` back at the previous version | seconds, no retraining |
# | **Holdout** | a random group deliberately not acted on, so the model stays measurable | customers who are never called |
# | **Postmortem** | a blameless write-up of an incident and how to prevent it | Exercise 3 |

# %% [markdown]
# ## 15.1 · Three layers, three clocks
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/monitoring.svg" alt="Monitoring layers: service health is available immediately, drift is available immediately, performance arrives after the 30-day label delay" width="100%">
#
# | Layer | Examples | Available | Tells you |
# |---|---|---|---|
# | **Service health** | error rate, p95 latency, readiness | seconds | the service works (chapter 14) |
# | **Data & prediction drift** | feature PSI, unseen categories, score distribution, share flagged | as soon as requests arrive | the *inputs* or *outputs* changed: a symptom, not a verdict |
# | **Performance** | PR-AUC, ROC-AUC, calibration against real outcomes | **30 days later** (the label horizon) | whether the model is still right |
#
# The order matters: the cheap signals arrive first and can only say "look
# here"; the signal that decides arrives last.

# %% [markdown]
# ## 15.2 · Simulating the next half-year
#
# The raw data was produced by `churnlab.data.generate`. Run it again for a
# later period (January to early August 2025), with a different seed and two
# knobs turned: **failed payments 2.5× as frequent** (say, a payment provider
# change) and **engagement 0.3 lower** (say, a competitor's launch). The
# monitoring code is not told about either.
#
# As a **control**, generate the same period with the same seed and *no*
# shift. Anything that differs between control and the training data cannot
# be the injected shift; it is how the simulation (or the calendar) works.

# %%
import numpy as np
import pandas as pd

from churnlab.data import GeneratorConfig, generate
from churnlab.features import CATEGORICAL, LABEL, NUMERIC, build_features
from churnlab.validate import clean

NEW_SNAPSHOTS = pd.date_range("2025-03-01", "2025-07-01", freq="MS")


def simulate(**knobs) -> pd.DataFrame:
    cfg = GeneratorConfig(n_customers=3000, start="2025-01-06", n_weeks=30, seed=2025,
                          inject_quality_issues=False, **knobs)
    return build_features(clean(generate(cfg)), snapshots=NEW_SNAPSHOTS)


prod = simulate(failed_payment_multiplier=2.5, engagement_shift=-0.3)
control = simulate()

reference = split.train  # what the model learned from
by_month = pd.DataFrame({
    "prod_rows": prod.groupby("snapshot_date").size(),
    "prod_churn_rate": prod.groupby("snapshot_date")[LABEL].mean(),
    "control_churn_rate": control.groupby("snapshot_date")[LABEL].mean(),
}).round(4)
by_month.index = by_month.index.date
print(f"training reference: {len(reference):,} rows, churn rate {reference[LABEL].mean():.4f}")
by_month

# %% [markdown]
# **Expected:** about 2,000-2,400 rows per monthly snapshot (fewer each month as
# accounts cancel), a production churn rate of about **0.08-0.09** per month
# against **0.04-0.06** in the control world and 0.0545 in training. The shift
# is real and large: roughly 60% more churn.
#
# What we observed about the generator (check it on your run):
#
# * It simulates only the requested weeks, so the first snapshot (2025-03-01)
#   has eight weeks of history, exactly like the training data's first snapshot
#   (2023-03-01, eight weeks after its start). Features that look back 90 days
#   are slightly truncated in the first two snapshots, in both periods.
# * Sign-up dates are drawn relative to the *end* of the simulated period, over
#   a 900-day window. This short period therefore contains many accounts
#   that are already old, and tenure is much longer than in training. Some of
#   that is also realistic: a customer base ages.
#
# ## 15.3 · Input drift with PSI
#
# The **Population Stability Index** compares a current sample with a
# reference over bins cut at the reference's deciles:
# PSI = Σ (cur% − ref%) · ln(cur% / ref%). Common bands (conventions, not laws):
# below 0.1 stable, 0.1-0.25 moderate, above 0.25 large.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-psi-bins.svg" alt="Reference and current shares in five quintile bins; each bin contributes (a - e) ln(a / e) and the terms sum to a PSI of 0.135, a moderate shift" width="100%">
#
# *What to notice:* every bin's term is positive (both factors have the same sign), and the outer bins, where the share moved most in relative terms, contribute most.
#
# > **Intuition.** PSI is a census that compares how many people live in
# > each neighbourhood this year versus the reference year. Big relative moves
# > in any neighbourhood raise it. *Where it breaks down:* it only sees the
# > bins it was given: a feature that changes *within* a bin (or is mostly one
# > value, like `failed_payments_90d`) can shift a lot with a PSI near 0.
#
# Compute it for every
# feature, for the shifted production data **and** for the control world, and
# for the validation period as a third comparison that has no shift at all:

# %%
from churnlab.monitoring import categorical_psi, drift_report, psi

drift = drift_report(reference, prod, NUMERIC, CATEGORICAL)
psi_control = drift_report(reference, control, NUMERIC, CATEGORICAL).set_index("feature")["psi"]
psi_valid = drift_report(reference, split.valid, NUMERIC, CATEGORICAL).set_index("feature")["psi"]
drift["psi_control"] = drift["feature"].map(psi_control)
drift["psi_valid"] = drift["feature"].map(psi_valid)
drift.to_csv(MON / "drift_report.csv", index=False)
drift[["feature", "kind", "psi", "status", "psi_control", "psi_valid", "ref_mean", "cur_mean"]].round(3)

# %% [markdown]
# **Expected** (each row teaches something different):
#
# * `weeks_since_last_active` has an enormous PSI (above 3) in production **and
#   in control**, and a large one even for validation. It is not drift. The
#   feature is "days since the last completed active week ÷ 7", so its values
#   sit on a few points whose fractional part depends on the **weekday of the
#   snapshot date** (`0.142857`, `0.571429`, ...). The decile bins of 2023-24
#   snapshots do not line up with the 2025 ones. A calendar artifact.
# * `tenure_days` is large in production and control alike: the population is
#   older (see above). Real, but not the injected shift.
# * The sessions features show the engagement drop as PSI of about **0.09-0.10**:
#   "stable" by the convention, though their means fell by about a fifth.
# * `failed_payments_90d`, the feature we multiplied by 2.5, has **PSI = 0**.
#
# That last one deserves a closer look:

# %%
fp_ref = reference["failed_payments_90d"].clip(upper=3).value_counts(normalize=True).sort_index()
fp_cur = prod["failed_payments_90d"].clip(upper=3).value_counts(normalize=True).sort_index()
print(pd.DataFrame({"reference": fp_ref, "production": fp_cur}).round(3).rename_axis("failed payments (3 = 3+)"))
print("reference deciles:", np.quantile(reference["failed_payments_90d"], np.linspace(0, 1, 11)))
print(f"quantile-binned PSI: {psi(reference['failed_payments_90d'], prod['failed_payments_90d']):.3f}")
print(f"PSI on the values as categories: "
      f"{categorical_psi(reference['failed_payments_90d'].clip(upper=3), prod['failed_payments_90d'].clip(upper=3)):.3f}")

# %% [markdown]
# **Expected:** the share of accounts with at least one failed payment rises
# from about **10% to about 24%**, yet the quantile-binned PSI is **0.000**:
# 90% of the reference is zero, so every decile edge but the maximum is 0, the
# edges collapse to a single bin, and one bin always has PSI 0. Treated as
# categories (0, 1, 2, 3+), the same data gives a PSI of about **0.16**:
# a moderate shift, and the only one in the table that is really there.
#
# **Lesson:** a monitoring number is only as good as its assumptions. Quantile
# PSI suits continuous features; for counts and zero-inflated features use
# category shares (or fixed bins), and always look at the distributions. A
# check that silently returns "0, stable" for the feature that changed most
# is worse than no check.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
width = 0.4
axes[0].bar(fp_ref.index - width / 2, fp_ref.values, width, label="training reference")
axes[0].bar(fp_cur.index + width / 2, fp_cur.values, width, label="production 2025")
axes[0].set_xticks([0, 1, 2, 3], labels=["0", "1", "2", "3+"])
axes[0].set_xlabel("failed payments in the last 90 days")
axes[0].set_ylabel("share of rows")
axes[0].legend()
bins = np.linspace(0, np.quantile(reference["sessions_4w"], 0.99), 40)
axes[1].hist(reference["sessions_4w"], bins=bins, density=True, alpha=0.5, label="training reference")
axes[1].hist(prod["sessions_4w"], bins=bins, density=True, alpha=0.5, label="production 2025")
axes[1].set_xlabel("sessions in the last 4 weeks")
axes[1].set_ylabel("density")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# > **Research bridge.** This is covariate shift between recording sessions
# > or subjects in EEG: amplitude statistics move, and you check whether
# > the decoder still works on the new session. The same caution applies:
# > a shift in the inputs is a reason to *check* performance, not proof that
# > it dropped.

# %% [markdown]
# > **Exercise 1.** Write your own `psi(reference, current, bins=10)`, matching
# > `churnlab.monitoring.psi` to within 1e-6. The definition, exactly:
# >
# > 1. drop missing values from both samples (if either is then empty, return `nan`);
# > 2. edges = the unique values of the reference's quantiles at 0, 1/bins, ..., 1
# >    (`np.quantile` default method); if fewer than 2 edges remain, return `0.0`;
# > 3. replace the first edge with −∞ and the last with +∞;
# > 4. r, c = the fraction of reference / current values in each bin
# >    (`np.histogram` with those edges);
# > 5. clip both at a minimum of `1e-4` (so empty bins do not divide by zero);
# > 6. return Σ (c − r) · ln(c / r) as a Python float.

# %%
def my_psi(reference, current, bins: int = 10) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(my_psi)

# %% [markdown]
# ## 15.4 · Prediction drift
#
# The model's **scores** are a one-dimensional summary of everything it sees,
# available without labels. Compare the live model's scores on production
# with its scores on the validation period (where the threshold was chosen):

# %%
from churnlab.modeling import predict_proba

model, meta = ops.load_current()
threshold = meta["threshold"]
valid_scores = predict_proba(model, split.valid)
prod = prod.assign(score=predict_proba(model, prod))
control = control.assign(score=predict_proba(model, control))

score_drift = pd.DataFrame({
    "mean_score": [valid_scores.mean(), prod["score"].mean(), control["score"].mean()],
    "share_flagged": [(valid_scores >= threshold).mean(), (prod["score"] >= threshold).mean(),
                      (control["score"] >= threshold).mean()],
    "score_psi_vs_valid": [0.0, psi(valid_scores, prod["score"]), psi(valid_scores, control["score"])],
}, index=["validation 2024", "production 2025", "control 2025"]).round(4)
score_mean_ratio = prod["score"].mean() / valid_scores.mean()
print(f"model {meta['version']} ({meta['model_name']}), threshold {threshold}")
print(f"score_mean_ratio (production / validation) = {score_mean_ratio:.2f}")
score_drift

# %% [markdown]
# **Expected:** the mean score rises by about half (ratio about **1.5**), the
# share of customers flagged nearly triples (about 3% → 8%), and the score PSI
# is moderate in production (about 0.2) but near zero in control. The model *noticed* the riskier world:
# more customers look like churners to it. That is a useful early warning, a
# month before any label arrives. It does not yet say whether the model is
# *right*.

# %% [markdown]
# ## 15.5 · Performance, once the labels arrive
#
# A prediction made at snapshot T can be judged at T + 30 days. Suppose today
# is 15 June 2025:

# %%
def matured(snapshots, today, horizon_days: int = 30) -> pd.DataFrame:
    snaps = pd.to_datetime(pd.Series(sorted(set(snapshots))))
    known = snaps + pd.Timedelta(days=horizon_days)
    return pd.DataFrame({"snapshot": snaps.dt.date, "label_known_on": known.dt.date,
                         "matured": known <= pd.Timestamp(today)})


matured(prod["snapshot_date"], today="2025-06-15")

# %% [markdown]
# **Expected:** March, April and May have matured; June and July have not.
# On 15 June you can judge **three of five** months; the most recent, most
# relevant predictions are exactly the ones you cannot judge yet.
#
# Now jump to the end of the simulation, when all five have matured, and
# compare with validation. PR-AUC is the headline metric, but it depends on
# the **positive rate**, which changed a lot, so look at ROC-AUC (independent
# of the positive rate), PR-AUC ÷ positive rate (how many times better than
# random), and calibration (mean score versus observed rate) too.

# %%
from sklearn.metrics import average_precision_score, roc_auc_score

from churnlab.evaluation import bootstrap_ci


def performance(y, s) -> dict:
    return {"rows": len(y), "positive_rate": y.mean(), "pr_auc": average_precision_score(y, s),
            "roc_auc": roc_auc_score(y, s), "pr_auc / rate": average_precision_score(y, s) / y.mean(),
            "mean_score": s.mean()}


perf = pd.DataFrame({
    "validation 2024": performance(split.valid[LABEL], valid_scores),
    **{f"prod {d.date()}": performance(g[LABEL], g["score"]) for d, g in prod.groupby("snapshot_date")},
    "prod all": performance(prod[LABEL], prod["score"]),
    "control all": performance(control[LABEL], control["score"]),
}).T.round(3)
perf.to_csv(MON / "performance.csv")
perf

# %%
ci_valid = bootstrap_ci(split.valid[LABEL], valid_scores, n_boot=200, groups=split.valid["customer_id"])
ci_prod = bootstrap_ci(prod[LABEL], prod["score"], n_boot=200, groups=prod["customer_id"])
ci_roc_valid = bootstrap_ci(split.valid[LABEL], valid_scores, roc_auc_score, n_boot=200,
                            groups=split.valid["customer_id"])
ci_roc_prod = bootstrap_ci(prod[LABEL], prod["score"], roc_auc_score, n_boot=200, groups=prod["customer_id"])
print("PR-AUC  validation %.3f [%.3f, %.3f]   production %.3f [%.3f, %.3f]" % (*ci_valid, *ci_prod))
print("ROC-AUC validation %.3f [%.3f, %.3f]   production %.3f [%.3f, %.3f]" % (*ci_roc_valid, *ci_roc_prod))
pr_auc_ratio = ci_prod[0] / ci_valid[0]
print(f"pr_auc_ratio = {pr_auc_ratio:.2f}")

# %% [markdown]
# **Expected (95% customer-level bootstrap intervals):** production PR-AUC
# about **0.27** against about **0.16** on validation, so `pr_auc_ratio` is
# about **1.6**: a naive "PR-AUC dropped" alert would never fire, because the
# metric went *up*. But the positive rate went up by about as much, and
# PR-AUC ÷ rate is about **3.1** in both periods. ROC-AUC is about 0.73 on
# validation and 0.76 in production, with overlapping intervals; month by
# month it moves between about 0.73 and 0.79. Overall, the mean score (about
# 0.079) is a little below the observed rate (about 0.086).
#
# **The honest summary:** on this simulated shift the model did **not**
# measurably degrade. Its ranking is as good as on validation, and its
# probabilities are only slightly low (under by roughly a tenth overall, not
# in every month). That is not luck: the simulation changed the *inputs*
# (more failed payments, less engagement) but not *how* they lead to churn,
# and the model already knew that relationship. Covariate shift without
# concept drift. The input-drift alerts were loud; the damage was small. That
# is the difference between drift and damage, measured rather than assumed.
# Had the generator changed how failed payments drive cancellation, the same
# drift report would have looked similar while performance fell; only this
# section could have told the two apart.

# %% [markdown]
# ## 15.6 · From measurements to alerts
#
# Monitoring is useful only if somebody acts on it. `churnlab.monitoring`
# holds a small rule set: each rule has a metric, a threshold and a severity
# (*page* = wake someone up, *warn* = a ticket for working hours). To fill in
# the service-health metrics, replay the latest production snapshot through
# the real app:

# %%
from fastapi.testclient import TestClient

from churnlab.api import create_app, example_payload
from churnlab.logs import ListHandler, get_logger
from churnlab.monitoring import DEFAULT_RULES, AlertRule, evaluate_rules
from churnlab.settings import Settings

api_logs = ListHandler()
app = create_app(Settings(registry_dir=ops.root), logger=get_logger("nb14-api", "INFO", handler=api_logs))
api = TestClient(app)

latest = prod[prod["snapshot_date"] == prod["snapshot_date"].max()]
latest_payloads = [example_payload(row) for _, row in latest.iterrows()]
for i in range(0, len(latest_payloads), 250):
    api.post("/predict/batch", json={"customers": latest_payloads[i:i + 250]})
service = api.get("/metrics").json()
n_requests = sum(service["requests"].values())
n_errors = sum(service["errors"].values())

values = {
    "error_rate": n_errors / n_requests,
    "latency_p95_ms": service["latency_ms"]["p95"],
    "max_psi": drift["psi"].max(),
    "score_mean_ratio": score_mean_ratio,
    "pr_auc_ratio": pr_auc_ratio,
}
print({k: round(float(v), 3) for k, v in values.items()})
alerts = evaluate_rules(values)
alerts

# %% [markdown]
# **Expected:** no service errors and a p95 of tens of milliseconds per batch
# of 250 (machine-specific), so the two *page* rules stay quiet. Two *warn*
# rules fire: **feature drift large** (`max_psi` above 3, from the calendar
# artifact) and **score mean shifted** (ratio above 1.5, the real shift). The
# **PR-AUC dropped** rule stays quiet because PR-AUC rose with the positive rate.
#
# Read that critically:
#
# * One warning fired for the wrong reason (an artifact), one for the right
#   reason, and neither means the model is wrong (15.5 showed it is not). The
#   rule set also has no calibration check; a good next rule is
#   `mean_score / observed_rate` on matured cohorts outside [0.8, 1.25].
# * A drift alert that fires every month for a known artifact teaches people
#   to ignore the channel. Fix the feature check (exclude or re-bin it) rather
#   than living with the noise.
# * Normalise `pr_auc_ratio` by the change in positive rate, or alert on
#   ROC-AUC or lift, so that a base-rate change cannot hide a real drop.

# %% [markdown]
# ## 15.7 · A bad deploy, caught and rolled back
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-canary-rollback.svg" alt="A retrained model is registered, gated against the current model, run in shadow or canary mode, compared on PR-AUC, error rate and p95 latency, then promoted or rolled back by moving CURRENT" width="100%">
#
# *What to notice:* in this section the gate was switched off, so the bad model reaches CURRENT; the canary comparison and the pointer rollback are the remaining safeguards.
#
# > **Intuition.** A canary is a taste test on one table before the dish
# > goes to the whole restaurant; rollback is taking the dish off the menu,
# > not re-cooking it. *Where it breaks down:* with 30-day label delay the
# > "taste" of ranking quality is only available on old, matured data, so a
# > canary on live traffic can only catch crashes, latency and distribution
# > collapse.
#
# Monthly retraining is automated. This month the training job had a bug: a
# join misaligned rows and labels, which here we imitate by **shuffling the
# label column** of the training data. The job registers the model and, because
# someone disabled the promotion gate "temporarily", promotes it.

# %%
from churnlab.splits import Split

broken_train = split.train.copy()
broken_train[LABEL] = np.random.default_rng(14).permutation(broken_train[LABEL].to_numpy())
v_bad = bootstrap.train_and_register(ops, "logreg", Split(broken_train, split.valid, split.test),
                                     notes="monthly retrain (labels misaligned: deliberately broken in nb14)")
good_version = ops.current()
for v in (good_version, v_bad):
    m = ops.metadata(v)
    print(f"{v}: validation PR-AUC {m['metrics']['validation']['pr_auc']:.3f}, threshold {m['threshold']}")

# %% [markdown]
# **Expected:** the new version (`v3`) has a validation PR-AUC near the
# positive rate (about 0.05: random ranking), against about 0.16 for `v1`. The
# registry *recorded* the problem; a promotion gate that compared these two
# numbers would have stopped it. Without the gate, it is deployed:

# %%
canary = prod[prod["snapshot_date"] == pd.Timestamp("2025-06-01")]  # latest matured month
canary_payloads = [example_payload(row) for _, row in canary.iterrows()]


def score_via_api(bodies) -> tuple[np.ndarray, set]:
    scores, versions = [], set()
    for i in range(0, len(bodies), 500):
        r = api.post("/predict/batch", json={"customers": bodies[i:i + 500]})
        r.raise_for_status()
        scores += [p["churn_probability"] for p in r.json()["predictions"]]
        versions.add(r.json()["model_version"])
    return np.array(scores), versions


baseline, served = score_via_api(canary_payloads)
print("before:", served, "| GET /model ->", api.get("/model").json()["model_version"])

ops.promote(v_bad, reason="monthly retrain")
print("reload:", api.post("/admin/reload").json())
candidate, served = score_via_api(canary_payloads)
print("after :", served)

# %% [markdown]
# **Expected:** before, `{'v1'}`; the reload answers `{'status': 'reloaded',
# 'model_version': 'v3', ...}`; after, `{'v3'}`. No error, no latency change:
# **every service-health signal is green**. Only a model-level check can see
# this.
#
# The **canary check** compares the candidate with the model it replaces on
# the most recent slice whose labels have matured (June 2025, about 2,100
# rows), plus label-free checks on the score distribution:

# %%
y = canary[LABEL].to_numpy()
canary_values = {
    "pr_auc_ratio": average_precision_score(y, candidate) / average_precision_score(y, baseline),
    "score_std_ratio": candidate.std() / baseline.std(),
    "score_psi": psi(baseline, candidate),
    "score_mean_ratio": candidate.mean() / baseline.mean(),
}
CANARY_RULES = [
    AlertRule("canary PR-AUC dropped", "pr_auc_ratio", "<", 0.9, "page"),
    AlertRule("score spread collapsed", "score_std_ratio", "<", 0.5, "page"),
    AlertRule("score distribution moved", "score_psi", ">", 0.25, "warn"),
    AlertRule("score mean collapsed", "score_mean_ratio", "<", 0.5, "warn"),
]
print({k: round(float(v), 3) for k, v in canary_values.items()})
canary_alerts = evaluate_rules(canary_values, CANARY_RULES)
canary_alerts

# %% [markdown]
# **Expected:** PR-AUC ratio far below 0.9 (the candidate ranks no better than
# chance), the score spread collapsed (std ratio well below 0.5: a model that
# learned nothing predicts nearly the base rate for everyone), and a large
# score PSI. The **mean** ratio is about 0.65, so the default "mean collapsed"
# rule (< 0.5) stays quiet: a broken model can keep a plausible average.
# Spread and ranking are what give it away.
#
# **Roll back:** move the pointer back and reload. Bundles are immutable, so
# nothing is rebuilt.

# %%
if canary_alerts.query("fires and severity == 'page'").shape[0]:
    reasons = "; ".join(canary_alerts.query("fires")["rule"])
    restored = ops.rollback(reason=f"canary failed: {reasons}")
    print("rolled back to", restored, "| reload:", api.post("/admin/reload").json())

after, served = score_via_api(canary_payloads)
print("serving:", served, "| CURRENT:", ops.current())
print("predictions identical to before the bad deploy:", bool(np.allclose(after, baseline)))

# %% [markdown]
# **Expected:** `rolled back to v1`, serving `{'v1'}`, and `True`: the
# predictions are exactly the ones from before the deploy. The registry's
# append-only history is the audit trail, and the service logged each reload:

# %%
history = pd.DataFrame(ops.history())
history.to_csv(MON / "registry_history.csv", index=False)
print(history.to_string(index=False))
print()
print([r for r in api_logs.parsed() if r["msg"] == "model_reload"])

# %% [markdown]
# **Expected:** the two original registrations and the promotion of `v1`
# (copied from the course registry), then `register v3`, `promote v3
# (previous v1, reason "monthly retrain")` and `rollback v1 (previous v3,
# reason "canary failed: ...")`, each with a UTC timestamp. The service logs
# show two `model_reload` events: v1 → v3 and v3 → v1.
#
# Two caveats from chapters 13 and 14 apply here. `/admin/reload` has no
# authentication in this teaching app (in production it sits behind auth or
# is replaced by a restart), and with several uvicorn workers a reload reaches
# only one of them; a rollback must restart or reload **every** replica, and
# then check, as above, that each serves the expected version.

# %% [markdown]
# ## 15.8 · Release strategies and retraining triggers
#
# The canary above ran *after* the promotion, on a slice we chose. Real
# release strategies decide how much traffic a new model sees before anyone
# trusts it:
#
# | Strategy | How | Catches | Cost |
# |---|---|---|---|
# | **Shadow** | the candidate scores live traffic in parallel; only the old model's answers are used | crashes, latency, score-distribution changes, disagreement rate, all with **zero user impact** | double compute; no outcome feedback on the candidate's *decisions* |
# | **Canary** | route a small share (1-10%) of traffic to the candidate, compare, then ramp up | everything shadow catches, plus real-world behaviour on a slice | some users get the bad model for a while |
# | **Blue/green** | two full environments; switch all traffic at once, switch back to roll back | deploy mistakes (config, dependencies); rollback in seconds | double infrastructure during the switch; no gradual exposure |
#
# For a model whose labels arrive 30 days later, the **offline canary on the
# latest matured slice** (what we did) plus **shadow mode** for a few days is
# the practical combination: shadow catches crashes and distribution collapse
# at once; the matured slice measures ranking quality.
#
# **When to retrain:**
#
# | Trigger | Pros | Cons |
# |---|---|---|
# | **Schedule** (monthly) | simple, predictable, easy to staff | retrains when nothing changed; may wait too long when something did |
# | **Drift** (PSI, score shift) | reacts early, no labels needed | drift ≠ damage (our `weeks_since_last_active` artifact would trigger it every month) |
# | **Performance** (matured PR-AUC, calibration) | reacts to actual damage | always at least one label horizon late |
#
# A common answer: retrain **on a schedule**, *investigate* on drift alerts,
# and retrain **early** when matured performance or calibration crosses a
# threshold. Every retrain goes through the same gate (validation metrics
# versus the current model) and the same canary. And with delayed labels,
# acting on predictions changes outcomes: if the success team calls flagged
# customers and retains some, those labels no longer show what the model
# predicted. A small random **holdout** that is not contacted is needed to keep
# the model measurable at all.

# %% [markdown]
# > **Exercise 2.** Write `should_rollback(candidate, baseline, policy)` that
# > decides whether to roll a candidate back. `candidate` and `baseline` are
# > dicts with keys `"pr_auc"`, `"error_rate"`, `"p95_ms"` (`candidate["pr_auc"]`
# > may be `None`: labels not matured yet). `policy` is a dict with
# > `"max_pr_auc_drop"` (a fraction, e.g. 0.1 = 10%), `"max_error_rate"` and
# > `"max_p95_ms"`. Return `(rollback: bool, reasons: list[str])`:
# >
# > * roll back if `candidate["pr_auc"] < baseline["pr_auc"] * (1 - max_pr_auc_drop)`
# >   (skip this check when either PR-AUC is `None`);
# > * roll back if `candidate["error_rate"] > max_error_rate`;
# > * roll back if `candidate["p95_ms"] > max_p95_ms`.
# >
# > Exactly at a limit is **not** a breach. Give one reason string per breached
# > rule, in the order above, each containing the metric's key (`"pr_auc"`,
# > `"error_rate"` or `"p95_ms"`). `rollback` is True exactly when `reasons`
# > is non-empty.

# %%
def should_rollback(candidate: dict, baseline: dict, policy: dict) -> tuple[bool, list[str]]:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(should_rollback)

# %% [markdown]
# > **Exercise 3 (written).** Write a six-sentence **incident postmortem** for
# > the bad deploy in section 15.7, one sentence each: **timeline** (what
# > happened when), **detection** (which signal caught it, which did not),
# > **impact** (who got what, for how long; estimate it), **root cause** (the
# > technical cause *and* the process gap), **fix** (what was done), and
# > **prevention** (one concrete change, with an owner-able action). Keep it
# > blameless: describe systems and decisions, not people. Write your answer
# > in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **Population Stability Index.** Cut the reference into bins (deciles here),
# then compare the shares:
#
# $$
# \operatorname{PSI} = \sum_{b=1}^{B} (a_b - e_b)\, \ln\frac{a_b}{e_b}
# $$
#
# *In words:* for each bin, the change in share times the log of the ratio of
# shares; both factors have the same sign, so every term is $\ge 0$.
# *Symbols:* $e_b$ = reference share in bin $b$, $a_b$ = current share, $B$ =
# number of bins (a tiny $\varepsilon$ replaces empty shares). *Example
# (figure above):* $e = 0.2$ in each of 5 bins, $a = (0.30, 0.25, 0.20, 0.15,
# 0.10)$: terms $0.041 + 0.011 + 0 + 0.014 + 0.069 = 0.135$, "moderate".
#
# **Score-mean ratio** (prediction drift): $r = \bar{s}_{\text{current}} / \bar{s}_{\text{reference}}$.
# *Example:* reference mean 0.055, current 0.090 → $r = 1.64 > 1.5$: warn.
#
# **Calibration on a matured cohort:** $\bar{s} / \bar{y}$ (mean predicted over
# observed rate). *Example:* 0.055 predicted vs 0.060 observed → 0.92, inside
# a [0.8, 1.25] band.
#
# **Rollback rule** (Exercise 2): roll back if
# $\text{PR-AUC}_{\text{cand}} < (1 - \delta)\,\text{PR-AUC}_{\text{base}}$,
# or the error rate or p95 exceeds its limit. *Example:* $\delta = 0.1$,
# baseline 0.163: a candidate below 0.147 is rolled back; exactly 0.147 is not.

# %%
# The PSI worked example, by hand and with churnlab.monitoring.psi on samples with those shares.
e = np.full(5, 0.2)
a = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
terms = (a - e) * np.log(a / e)
print("terms by hand:", np.round(terms, 4), "| PSI", round(float(terms.sum()), 4))

reference_toy = np.arange(1000) + 0.5                                   # uniform: quintiles hold 20% each
current_toy = np.repeat([100.0, 300.0, 500.0, 700.0, 900.0], [300, 250, 200, 150, 100])
print("churnlab psi :", round(psi(reference_toy, current_toy, bins=5), 4))

# %% [markdown]
# **Expected:** terms 0.0405, 0.0112, 0, 0.0144, 0.0693, and PSI 0.1354 both ways.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Different rows or churn rates than the Expected lines | another seed or `n_customers`, or a changed generator | the shift should still show: production churn well above control |
# | `ArtifactError: unknown version` on promote | the sandbox registry was rebuilt between cells | re-run from the catch-up cell; it recreates `registry_nb14` from the course registry |
# | `ArtifactError: nothing to roll back to` | `rollback()` needs a previous promotion in `history.jsonl` | promote the candidate first; never edit `history.jsonl` by hand |
# | After rollback the API still serves v3 | the reload was not called, or (with several workers) reached only one worker | `POST /admin/reload` on every worker, or restart them; then check `x-model-version` |
# | `UserWarning: model trained with sklearn ...` | the registry was built with another scikit-learn version | delete `artifacts/registry` and re-run; the catch-up cell rebuilds it |
# | PSI is `0.000` for a feature you know changed | quantile bins collapsed (a mostly-zero feature) | compare category shares (`categorical_psi`) or use fixed bins |
# | Huge PSI on `weeks_since_last_active` every month | calendar artifact of the snapshot weekday | exclude it from drift alerts or bin it by whole weeks |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. Name the three layers of model monitoring and when each becomes available for this churn model.
# 2. PSI on an input feature is 0.4. Is the model broken? What do you check next, in what order?
# 3. Give two ways PSI can mislead, using what you saw in this notebook.
# 4. Production PR-AUC went from 0.16 to 0.26. Did the model get better? How do you find out?
# 5. Walk through a rollback in this design: what changes on disk, what the service does, and how you verify it worked.
# 6. Shadow, canary or blue/green for the next model version of this service? Justify, given the 30-day label delay.
#
# **Next chapter:** [Chapter 16 · Final evaluation & documentation](../ch16-final-evaluation-and-docs/notebook.ipynb) — evaluating on the test set once, reporting with uncertainty, and writing the README and model card.

