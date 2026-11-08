# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 9 · Error analysis & artifacts
#
# **Day 2 · ~45 minutes · CPU only**
#
# A single PR-AUC hides where the model fails and for whom. And a model that
# exists only as a variable in a notebook cannot be served, audited or rolled
# back. This notebook does both halves of "finishing" a model: slice and error
# analysis, then a versioned bundle with metadata, a registry and a model card.
#
# By the end you will be able to:
#
# 1. produce a slice report, flag slices too small to read, and put an interval on the worst one;
# 2. compare false negatives with true positives and recognise errors that no model on these features can fix;
# 3. save a model as an immutable bundle with metadata, and detect a tampered file;
# 4. register versions, promote one, and read the audit history;
# 5. write a model card that states intended use, evidence and limits.
#
# **You will produce:** `artifacts/registry/` (v1, v2, `CURRENT`,
# `history.jsonl`) and `artifacts/model_card.md`.
#
# All numbers are on the **validation** period; models are fitted on **train**.
# The test period is reserved for notebook 15.

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
# --- Catch up: features, split, the selected model and its threshold --------
import json
import time

import numpy as np
import pandas as pd

from churnlab import bootstrap
from churnlab.evaluation import best_threshold_by_value, bootstrap_ci, classification_report, slice_report
from churnlab.features import CATEGORICAL, FEATURES, LABEL, NUMERIC
from grading import ch09 as grade
from churnlab.modeling import make_model, predict_proba
from churnlab.splits import time_split

T0 = time.time()
split = time_split(bootstrap.ensure_features())
train, valid = split.train, split.valid

model = make_model("logreg").fit(train[FEATURES], train[LABEL])  # notebook 06's selection
scores = predict_proba(model, valid)
y = valid[LABEL].to_numpy()
threshold, _ = best_threshold_by_value(y, scores, bootstrap.VALUE_TP, bootstrap.COST_FP)  # notebook 07
report = classification_report(y, scores)
print(f"threshold {threshold:.3f} | " + ", ".join(f"{k} {v:.3f}" for k, v in report.items() if k != "n"))

# %% [markdown]
# **Expected:** threshold ≈ 0.19, PR-AUC ≈ 0.163, ROC-AUC ≈ 0.733, Brier
# ≈ 0.047, lift@10% ≈ 3.2 (the top 10% of scores churn at about three times
# the base rate).

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Slice** | a subset of rows sharing a value | `plan = basic`, `tenure 0-30d` |
# | **Lift** | how many times the base rate a group churns at | lift@10% ≈ 3.2 |
# | **Irreducible error** | mistakes no model can fix with the available features | healthy-looking customers who leave anyway |
# | **Model bundle** | a folder with the model file and the metadata needed to trust it | `model.joblib` + `metadata.json` |
# | **Fingerprint** | a hash of a table's values, identifying the exact training data | 16 hex characters |
# | **Pickle / joblib** | Python's way to save objects; loading one can execute code | why the hash is checked first |
# | **Registry** | a store of immutable model versions plus a pointer to the live one | `v1/`, `v2/`, `CURRENT` |
# | **Promote / roll back** | point `CURRENT` at a new version / at the previous one | `registry.promote("v1")` |
# | **Audit log** | an append-only record of who changed what and why | `history.jsonl` |
# | **Model card** | a short document of intended use, evidence and limits | `artifacts/model_card.md` |

# %% [markdown]
# ## 8.1 · Slices: where is the model weak?
#
# `evaluation.slice_report` computes per-slice metrics and **flags** slices
# with fewer than `min_rows` rows instead of hiding them. Tenure needs
# bucketing first.

# %%
v = valid.copy()
v["tenure_bucket"] = pd.cut(v["tenure_days"], bins=[0, 180, 365, 730, np.inf],
                            labels=["<6m", "6-12m", "1-2y", "2y+"], right=False)

slices = []
for col in ["plan", "region", "acquisition_channel", "company_size", "tenure_bucket"]:
    r = slice_report(v, scores, by=col, threshold=threshold, min_rows=300)
    r.insert(0, "feature", col)
    slices.append(r.rename(columns={col: "slice"}))
slices = pd.concat(slices, ignore_index=True)
slices["slice"] = slices["slice"].astype(str)
slices["positives"] = (slices["rows"] * slices["positive_rate"]).round().astype(int)
slices.round(3)

# %% [markdown]
# **Expected (our run):** 20 slices. Positive rates vary by plan (≈ 0.07 on
# basic vs ≈ 0.04 on pro and team) and company size (≈ 0.03 for 200+). PR-AUC
# ranges from ≈ 0.09 to ≈ 0.27, **but** read the `positives` column before any
# PR-AUC: a slice with 30-50 positives has a PR-AUC that can move by ±0.05 or
# more on resampling. We used `min_rows=300` (only ≈ 15 positives at a 5%
# rate) and two slices are flagged: region `unknown` (130 rows, the truly
# missing regions from notebook 02) and tenure `2y+` (47 rows).
#
# Before reading anything into the worst slice, put an interval on it:

# %%
readable = slices[~slices["small_slice"]].dropna(subset=["pr_auc"])
worst = readable.sort_values("pr_auc").iloc[0]
mask = (v[worst["feature"]].astype(str) == worst["slice"]).to_numpy()
pt, lo, hi = bootstrap_ci(y[mask], scores[mask], n_boot=500, seed=0, groups=v.loc[mask, "customer_id"])
pt_all, lo_all, hi_all = bootstrap_ci(y, scores, n_boot=500, seed=0, groups=v["customer_id"])
print(f"worst readable slice: {worst['feature']} = {worst['slice']} "
      f"({int(worst['rows'])} rows, {int(worst['positives'])} positives)")
print(f"  slice PR-AUC {pt:.3f}  95% customer CI [{lo:.3f}, {hi:.3f}]")
print(f"  overall      {pt_all:.3f}  95% customer CI [{lo_all:.3f}, {hi_all:.3f}]")

# %% [markdown]
# **Expected (our run):** the worst readable slice is `company_size = 1-10`
# (PR-AUC 0.113, interval ≈ [0.086, 0.154]), and its interval **overlaps** the
# overall one (≈ [0.136, 0.197]). Also note that PR-AUC depends on the
# slice's positive rate: a slice with fewer churners has a lower PR-AUC
# baseline, so compare `pr_auc / positive_rate` (lift) or ROC-AUC before
# calling a slice "badly served". The defensible statement is usually
# *"we see no slice where the model is clearly worse, at this sample size"*,
# not a story about why one region is hard.
#
# > **Intuition.** A slice with 40 churners is a poll with 40 respondents:
# > its PR-AUC can swing several hundredths on who happened to answer. Read
# > the `positives` column before the metric. *Where it breaks down:* a
# > consistent gap across many months *is* evidence, even if each month's
# > slice is small; pooling over time is the fix.

# %% [markdown]
# > **Exercise 1.** Write `tenure_bucket(tenure_days)` that maps a Series of
# > tenure in days to an **ordered categorical** Series with the labels
# > `["0-30d", "30-90d", "90-180d", "180-365d", "1-2y", "2y+"]` for the edges
# > `[0, 30, 90, 180, 365, 730, ∞)`, intervals closed on the left (so 30 days
# > is `"30-90d"`). Missing tenure stays missing; the index is preserved.

# %%
def tenure_bucket(tenure_days):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(tenure_bucket)

# %% [markdown]
# > **Exercise 2.** Write `worst_slices(report, metric, min_rows, n)` that returns
# > the `n` rows of a slice report with the lowest `metric`, ignoring rows with
# > fewer than `min_rows` rows or a missing metric, sorted from worst. Do not
# > modify `report`.

# %%
def worst_slices(report, metric, min_rows, n):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(worst_slices)

# %% [markdown]
# ## 8.2 · Errors: what do the misses look like?
#
# At the chosen threshold, every validation row is a TP, FP, FN or TN.
# Compare their median features. The question for each group is: *could any
# model have got this right from these features?*

# %%
pred = scores >= threshold
v["outcome"] = np.select([pred & (y == 1), pred & (y == 0), ~pred & (y == 1)], ["TP", "FP", "FN"], "TN")
cols = ["sessions_4w", "sessions_trend", "seat_utilization_4w", "weeks_since_last_active",
        "tickets_90d", "cancellation_tickets_30d", "failed_payments_90d", "tenure_days", "mrr_usd"]
profile = v.groupby("outcome")[cols].median().T
profile.loc["model score"] = pd.Series(scores, index=v.index).groupby(v["outcome"]).median()
profile.loc["rows"] = v["outcome"].value_counts()
profile[["TP", "FN", "FP", "TN"]].round(2)

# %% [markdown]
# **Expected (our run):** true positives look like churn: falling or zero
# sessions, longer since last activity, more tickets, short tenure. False
# negatives (the ≈ 300 missed churners) look much more like true negatives
# than like true positives: normal usage, no cancellation tickets, scores
# around 0.06 (true negatives 0.03, true positives 0.25). False positives look
# like true positives (low and falling usage, longer since last activity, a
# failed payment): the model flags them for the right *visible* reasons, and
# they simply did not churn (yet).
#
# How many of the misses were "sudden churners", healthy by every signal we
# have?

# %%
healthy = ((v["sessions_trend"] >= 0.8) & (v["weeks_since_last_active"] < 1)
           & (v["cancellation_tickets_30d"] == 0) & (v["failed_payments_90d"] == 0))
fn = v["outcome"] == "FN"
print(f"missed churners (FN): {fn.sum()}")
print(f"  of which 'healthy' (usage stable or rising, active last week, no cancel tickets, no failed payments): "
      f"{(fn & healthy).sum()} ({(fn & healthy).sum() / fn.sum():.0%})")
print(f"churn rate among all healthy-looking rows: {v.loc[healthy, LABEL].mean():.3f} "
      f"vs overall {v[LABEL].mean():.3f}")

# %% [markdown]
# **Expected (our run):** 147 of the 311 missed churners (47%) look healthy,
# and the churn rate among all healthy-looking rows is still 0.032, against
# 0.052 overall. Those customers leave for reasons not in this data (a budget
# cut, an acquisition, a competitor's offer). **That error is irreducible with
# these features**; tuning the model will not reach it. Saying so is part of
# the analysis.
#
# **Feature ideas this suggests** (proposed, not implemented; each needs a
# point-in-time definition and a leakage check like notebook 03's):
#
# * contract renewal date and "days until renewal" (churn clusters at renewal);
# * change in seat count, and logins of the account's admin / champion user;
# * product-specific usage (which features, not how many), and integrations connected;
# * billing events: downgrade requests, plan changes, payment-method expiry;
# * text of support tickets (sentiment, "cancel", "export", "competitor").
#
# **What we would do next**, in order:
#
# 1. measure the save rate (the cost assumption that moves the threshold most, notebook 07);
# 2. add renewal-date features, the likeliest large gain from the error profile;
# 3. re-run the model comparison with the new features (trees may win once interactions appear);
# 4. set up monitoring for the score distribution and for flags per month (Day 4);
# 5. only then, more hyperparameter tuning.

# %% [markdown]
# ## 8.3 · A model bundle: the model plus everything needed to trust it
#
# `churnlab.artifacts` saves a **bundle**: a directory with `model.joblib` and
# `metadata.json`. The metadata answers the questions you will be asked about
# any deployed model: trained on what (fingerprint, rows, period), which
# features in which order, which library versions, what threshold, which
# validation metrics, and the SHA-256 of the model file.
#
# > **Intuition.** A bundle is a medicine bottle, not a loose pill: the
# > label says what is inside, the batch, the dose (threshold), the expiry
# > (library versions) and carries a seal (the hash). *Where it breaks down:*
# > a seal stored next to the bottle can be re-made by anyone who can open
# > it; real tamper-proofing needs signatures or access control.

# %%
from churnlab.artifacts import ArtifactError, ModelMetadata, dataframe_fingerprint, load_bundle, save_bundle

meta = ModelMetadata(
    model_name="logreg",
    version="demo",
    features=FEATURES,
    categorical=CATEGORICAL,
    threshold=round(threshold, 4),
    training_data={
        "fingerprint": dataframe_fingerprint(train[FEATURES + [LABEL]]),
        "rows": len(train),
        "first_snapshot": str(train["snapshot_date"].min().date()),
        "last_snapshot": str(train["snapshot_date"].max().date()),
    },
    metrics={"validation": report},
    params={"C": 1.0},
    notes="demo bundle from notebook 08",
)
bundle = save_bundle(model, meta, ART / "bundles" / "logreg_demo", overwrite=True)
print(sorted(p.name for p in bundle.iterdir()))
print(json.dumps({k: v for k, v in json.loads((bundle / "metadata.json").read_text()).items()
                  if k in ("model_name", "threshold", "training_data", "libraries", "model_sha256")}, indent=2))

# %% [markdown]
# **Expected:** two files, and metadata with a 16-character training-data
# fingerprint, 25,278 training rows, the library versions you are running,
# and a 64-character `model_sha256`.
#
# The fingerprint hashes the table's *values*. Hashes can silently depend on
# how a library stores data: pandas 3.0 keeps text in a new `str` dtype that
# hashes differently from pandas 2.2's `object`, so `dataframe_fingerprint`
# casts text columns to `object` first. With that, the same rows give the same
# fingerprint on both versions (checked in our verification runs). Any time you
# hash data, test the hash across the environments you support.
#
# Loading verifies the hash and warns if the sklearn version differs from the
# one the model was trained with (pickles are not guaranteed across sklearn
# versions, and failures can be silent):

# %%
loaded, loaded_meta = load_bundle(bundle)
same = np.allclose(predict_proba(loaded, valid.head(200)), scores[:200])
print("loaded", loaded_meta["model_name"], "| identical predictions on 200 rows:", same)

# %% [markdown]
# Now tamper with a **copy** of the bundle: change one byte of the model file,
# as a corrupted download or a malicious edit would.

# %%
import shutil
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    copy = Path(tmp) / "tampered"
    shutil.copytree(bundle, copy)
    raw = bytearray((copy / "model.joblib").read_bytes())
    raw[len(raw) // 2] ^= 0xFF  # flip one byte in the middle
    (copy / "model.joblib").write_bytes(bytes(raw))
    try:
        load_bundle(copy)
        print("loaded without complaint (this should not happen)")
    except ArtifactError as exc:
        print("refused:", exc)

# %% [markdown]
# **Expected:** `refused: model.joblib does not match the hash in
# metadata.json - corrupted or tampered`. The check runs **before**
# unpickling, which matters because loading a pickle executes code. A hash
# stored next to the file protects against corruption and accidents; against
# a deliberate attacker who can also edit `metadata.json`, you need signed
# artifacts or a registry with access control.

# %% [markdown]
# ## 8.4 · A registry: versions, a pointer, and an audit log
#
# `churnlab.registry.Registry` is a directory of immutable bundles (`v1/`,
# `v2/`, ...), a `CURRENT` file naming the live version, and an append-only
# `history.jsonl`. Deploying is writing a version into `CURRENT`; rolling back
# is writing the previous one (notebook 14 does that under pressure).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-bundle-registry.svg" alt="A registry with immutable version folders v1, v2 and v3, a CURRENT file pointing at the live version, an append-only history log, and rollback as rewriting CURRENT" width="100%">
#
# *What to notice:* nothing in the version folders ever changes; the only mutable thing is one tiny file, which is why rollback takes seconds.
#
# > **Intuition.** The registry is a library shelf of numbered editions and
# > `CURRENT` is the bookmark saying which one the service reads. Rolling back
# > moves the bookmark; it does not reprint a book. *Where it breaks down:* a
# > rollback restores the model, not the world: if the input data changed,
# > the old model may be wrong too.
#
# Register the linear model (v1) and the boosted trees (v2), then promote the
# one with the better validation PR-AUC. The cell is idempotent: re-running it
# reuses existing versions instead of piling up v3, v4, ...

# %%
from churnlab.registry import Registry

registry = Registry(ART / "registry")
if not registry.versions():
    bootstrap.train_and_register(registry, "logreg", split, notes="linear baseline with log/scale preprocessing")
    bootstrap.train_and_register(registry, "hgb", split, notes="gradient-boosted trees challenger")
pr = {ver: registry.metadata(ver)["metrics"]["validation"]["pr_auc"] for ver in registry.versions()}
best_version = max(pr, key=pr.get)
if registry.current() != best_version:
    registry.promote(best_version, reason=f"highest validation PR-AUC ({pr[best_version]:.3f})")
print("versions:", registry.versions(), "| CURRENT:", registry.current())
print({ver: (registry.metadata(ver)["model_name"], round(p, 4)) for ver, p in pr.items()})
print((registry.root / "history.jsonl").read_text())

# %% [markdown]
# **Expected:** `versions: ['v1', 'v2'] | CURRENT: v1`; v1 is logreg
# (≈ 0.163), v2 is hgb (≈ 0.135); the history shows two `register` events and
# one `promote` with its reason. Versions are never overwritten: a retrained
# model is v3, and v1 stays loadable for rollback.
#
# This is a deliberately small version of what **MLflow** (Model Registry),
# SageMaker or Vertex AI provide: the same ideas (immutable versions, a
# stage/alias pointer, lineage, an audit trail) plus a UI, access control and
# a tracking server. If you understand these 100 lines, those tools are
# configuration.

# %% [markdown]
# ## 8.5 · A model card
#
# A model card is the short document that travels with a model: what it is
# for, what it is **not** for, what data it saw, how good it is (with
# uncertainty), how the threshold was chosen, and where it fails. It is
# written from the metadata, so the numbers cannot drift from the artifact.

# %% [markdown]
# > **Exercise 3.** Write `write_model_card(meta, path)` that writes a markdown
# > model card to `path` and returns `path`. `meta` is a registry metadata dict
# > (as returned by `registry.metadata("v1")`). The card must start with a
# > `# Model card` title and contain these exact headings: `## Intended use`,
# > `## Training data`, `## Evaluation`, `## Decision threshold`,
# > `## Limitations`, `## Ethical considerations`. It must mention the model
# > name, version and training rows, and show every validation metric and the
# > threshold with three decimals (`f"{x:.3f}"`).

# %%
def write_model_card(meta, path):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(write_model_card)

# %% [markdown]
# The course's version, generated for the promoted model, with the slice and
# interval evidence from this notebook:

# %%
m = registry.metadata(registry.current())
val = m["metrics"]["validation"]
worst_rows = readable.sort_values("pr_auc").head(3)
card = f"""# Model card · TaskFlow churn risk · {m['model_name']} {m['version']}

*Generated from `registry/{m['version']}/metadata.json` by notebook 08. Synthetic data; teaching project.*

## Intended use

Rank active TaskFlow accounts each month by the risk that they cancel within 30 days, so the
customer-success team can choose whom to call (top 40 per month, or score ≥ threshold).
**Not intended for:** pricing, discounts or credit decisions; any automated action without a person;
scoring customers in their first month (no usage history yet); periods far from the training data
without re-validation.

## Training data

- {m['training_data']['rows']:,} (snapshot, customer) rows, monthly snapshots {m['training_data']['first_snapshot']} to {m['training_data']['last_snapshot']}
- fingerprint `{m['training_data']['fingerprint']}`; label `{m['label']}` = cancels within 30 days of the snapshot
- {len(m['features'])} features: {', '.join(m['features'])}
- point-in-time features (notebook 03); 30-day embargo before validation (notebook 04)

## Evaluation

Validation period 2024-04-01 to 2024-06-01 ({val['n']:,} rows, positive rate {val['positive_rate']:.3f}).
The test period (2024-07 onwards) is untouched until notebook 15.

| metric | value |
|---|---|
| PR-AUC | {val['pr_auc']:.3f} (95% customer-bootstrap CI {lo_all:.3f}-{hi_all:.3f}) |
| ROC-AUC | {val['roc_auc']:.3f} |
| Brier | {val['brier']:.3f} |
| lift @ top 10% | {val['lift@10%']:.2f} |

Compared with: base rate (PR-AUC {val['positive_rate']:.3f}) and gradient-boosted trees
(registry v2, PR-AUC {pr.get('v2', float('nan')):.3f}); see notebook 06 for the paired comparison.

Weakest slices with ≥ 300 rows by PR-AUC. The worst one's customer-bootstrap CI ({lo:.3f}-{hi:.3f})
{"overlaps the overall CI, so no slice is clearly worse at this sample size" if hi >= lo_all else "does not overlap the overall CI: investigate it"}:
{chr(10).join(f"- {r.feature} = {r.slice}: {r.pr_auc:.3f} ({int(r.rows)} rows, {int(r.positives)} positives)" for r in worst_rows.itertuples())}

## Decision threshold

Score ≥ **{m['threshold']:.3f}**, the value-maximising threshold on validation under **assumed** costs:
a call costs ${bootstrap.COST_FP:.0f}; reaching a would-be churner is worth ${bootstrap.VALUE_TP:.0f}
($600 margin × an assumed, unmeasured 25% save rate). Break-even for calibrated scores is 1/6.
Halving the save rate moves the threshold to ≈ 0.45; doubling it to ≈ 0.10 (notebook 07).
Scores are close to calibrated on validation (Brier {val['brier']:.3f}).

## Limitations

- Trained and validated on **synthetic** data; nothing here is a fact about real customers.
- Validation covers three months; seasonality beyond that is unknown.
- About half of missed churners look healthy on every feature; they leave for reasons not in the data.
- Threshold chosen on validation, so its validation value is optimistic.
- Slice estimates are noisy; small slices (e.g. unknown region) cannot be assessed.

## Ethical considerations

- The action is a supportive call. Being flagged must never lead to worse service, higher prices or
  withheld discounts, and the score must not be used for pricing decisions.
- Customers can be contacted repeatedly; cap contacts per account and honour opt-outs.
- Monitor flag rates by plan, region and company size so no group is systematically over- or under-contacted.
- Scores are probabilities about accounts, not judgements about people; do not share them with the customer's staff.
"""
(ART / "model_card.md").write_text(card, encoding="utf-8")
print(card)

# %% [markdown]
# **Expected:** a one-page card with real numbers from this run: PR-AUC
# ≈ 0.163 with its customer-bootstrap interval, threshold ≈ 0.19, the cost
# assumptions in words, the three weakest readable slices, and the "not for
# pricing" line. Read it as a reviewer would: every sentence should be either a
# number from the metadata or a decision someone can disagree with.

# %% [markdown]
# > **Exercise 4 (written).** The sales director reads the model card and asks
# > to use the score to decide which customers get a 20% discount at renewal
# > ("the risky ones, obviously"). Write your reply in five or six sentences:
# > what the card says, what could go wrong (think about what the score
# > rewards, and about the label once discounts change behaviour), and what you
# > would propose instead. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %%
print(f"notebook compute time: {time.time() - T0:.0f} s")

# %% [markdown]
# **Expected:** well under a minute.

# %% [markdown]
# ## Key equations
#
# **Lift at the top $q$ of scores:**
#
# $$
# \text{lift@}q = \frac{\text{churn rate among the top } q \text{ of scores}}{\pi}
# $$
#
# *In words:* how many times more often the flagged group churns than average.
# *Symbols:* $q$ = share flagged (10%), $\pi$ = base rate. *Example:* top 10%
# of 6,956 rows = 696 accounts, of which ≈ 116 churn: $116/696 = 0.167$, and
# $0.167 / 0.052 = 3.2$.
#
# **Lift-normalised slice PR-AUC.** Because a random ranking has PR-AUC
# $= \pi_s$ (the slice's positive rate), compare slices with
# $\text{PR-AUC}_s / \pi_s$ or with ROC-AUC. *Example (illustrative):* a
# slice with PR-AUC 0.113 at $\pi_s = 0.04$ is $2.8\times$ its floor, against
# $0.163 / 0.052 = 3.1\times$ overall: most of the apparent gap between 0.113
# and 0.163 is the lower base rate, not worse ranking.
#
# **How wide is a slice's interval?** Roughly $\propto 1/\sqrt{N_{+}}$ (number of
# positives). *Example:* going from 360 positives (overall CI width ≈ 0.06)
# to 40 positives multiplies the width by $\sqrt{360/40} = 3$, to ≈ 0.18.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ArtifactError: ... already exists; bundles are immutable` | saving to an existing bundle directory | use a new version (the registry does this), or `overwrite=True` only for throwaway demos |
# | `UserWarning: model trained with sklearn X, running Y` | the bundle was made in another environment (e.g. local vs Colab) | re-train in this environment, or pin the same sklearn; use `strict_versions=True` in a service |
# | Registry shows v3, v4, ... after re-running | you called `register` unconditionally | register only when the model changed; the cell in 8.4 checks first |
# | `slice_report` row has no `pr_auc` | the slice has a single class (no churners) | it is too small to evaluate; report it as such |
# | `bootstrap_ci` fails on a slice | almost every resample had no positives | do not bootstrap slices with a handful of positives |
# | `tenure_bucket` puts 30 days in "0-30d" | `pd.cut` closes intervals on the right by default | pass `right=False` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. The model's PR-AUC on one region is 0.08 vs 0.16 overall. What do you check before reporting a fairness problem?
# 2. Describe the model's false negatives. Which of them could a better model catch, and which could not, and how do you know?
# 3. What is in `metadata.json`, and which question does each field answer?
# 4. What does the hash in the bundle protect against, and what does it not protect against?
# 5. Walk through deploying v2 and rolling back to v1 with this registry. Why can rollback not half-succeed?
# 6. Name two things your model card says the model must *not* be used for, and why.
#
# **Next chapter:** [`09_fastapi_and_contracts.ipynb`](../ch10-fastapi-and-contracts/notebook.ipynb) — serve the promoted model behind a typed API contract.
