# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 11 · Consistent serving
#
# **Part I · Chapter 11 · about 45 minutes · CPU only**
#
# The most costly serving errors raise no exceptions. The API returns 200,
# the probabilities look plausible, and yet they differ from the ones that
# were validated, because the inputs or the preprocessing are not the same as
# in training. This lab first makes that failure visible, then removes its
# main cause, and finally builds a check that detects what remains.
#
# After this chapter you should be able to:
#
# 1. show and measure training/serving skew: maximum score difference and decisions flipped at the threshold;
# 2. explain why serving the saved `Pipeline` object removes a whole class of skew;
# 3. write a parity test: offline scores vs API scores on the same rows;
# 4. load a model safely: hash verification, version warnings, readiness without a model;
# 5. trace any prediction to a model version and a request id.
#
# **Lab output:** `artifacts/serving/parity_valid.csv`, a parity report (offline vs API) on 500 validation rows.

# %% [markdown]
# **Read first:** [Chapter 11 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: features, split and a registry (notebooks 03, 04, 08) --------
from churnlab import bootstrap
from grading import ch11 as grade

registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()
train, valid = split.train, split.valid  # the test period is not touched here
print(f"train {len(train):,} rows | valid {len(valid):,} rows")

# %% [markdown]
# **Expected:** `CURRENT=v1`, then `train 25,278 rows | valid 6,956 rows`.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Training/serving skew** | the served model computes features differently from the one that was evaluated | a forgotten `log1p` in the service |
# | **Re-implementation** | writing the same transform a second time, from a description | the backend's version of the notebook function |
# | **Fitted statistics** | numbers learned from training data that the transform depends on | medians, means, standard deviations |
# | **Column order** | the position of each feature in the array the model receives | swapping two columns silently swaps two coefficients |
# | **Decision flip** | a row whose call/no-call decision changes between two implementations | several hundred flips out of 6,956 |
# | **Parity test** | scores the same rows offline and through the running API and requires them to match | max difference ≈ 5e-7 |
# | **Tolerance** | the largest difference a test accepts as "equal" | 1e-6, because the API rounds to 6 decimals |
# | **Unseen category** | a categorical value not present in training | a new region; one-hot gives all zeros |
# | **Request id** | an identifier carried through logs so one call can be traced | the `x-request-id` header |
# | **Model version header** | the response names which model answered | `v1` |

# %% [markdown]
# ## 11.1 · Skew, made visible
#
# Imagine the model was trained *without* a pipeline: a data scientist wrote
# the preprocessing as a function in a notebook, trained a logistic regression
# on the resulting NumPy array, and handed over the coefficients. A backend
# engineer then **re-implemented** the preprocessing in the service, working
# from a description. The training side, written out by hand, follows (it does
# the same as `churnlab.modeling.make_preprocessor("linear")`, minus the
# missing-value indicators and rare-category grouping):

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from churnlab.evaluation import best_threshold_by_value
from churnlab.features import CATEGORICAL, LABEL, NUMERIC
from churnlab.modeling import LOG_FEATURES

print("log1p applied to:", LOG_FEATURES)


def fit_stats(df: pd.DataFrame) -> dict:
    """Everything the transform learns from TRAINING data."""
    X = df[NUMERIC].astype(float)
    median = X.median()
    X = X.fillna(median)
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return {"median": median.to_dict(), "mean": X.mean().to_dict(),
            "std": X.std(ddof=0).replace(0, 1.0).to_dict(),
            "categories": {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}}


def train_transform(df: pd.DataFrame, stats: dict) -> np.ndarray:
    """median-impute -> log1p (some columns) -> standardise -> one-hot."""
    X = df[NUMERIC].astype(float).fillna(stats["median"])
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    X = (X - pd.Series(stats["mean"])) / pd.Series(stats["std"])
    onehot = [(df[c].astype(str) == v).to_numpy(float) for c in CATEGORICAL for v in stats["categories"][c]]
    return np.column_stack([X[NUMERIC].to_numpy(), *onehot])


stats = fit_stats(train)
manual_model = LogisticRegression(max_iter=2000).fit(train_transform(train, stats), train[LABEL])
p_offline = manual_model.predict_proba(train_transform(valid, stats))[:, 1]
threshold, _ = best_threshold_by_value(valid[LABEL], p_offline, bootstrap.VALUE_TP, bootstrap.COST_FP)
print(f"validation PR-AUC {average_precision_score(valid[LABEL], p_offline):.3f} | threshold {threshold:.3f}"
      f" | flagged {(p_offline >= threshold).sum()} of {len(valid)}")

# %% [markdown]
# **Expected:** PR-AUC about 0.158, a threshold near 0.19, and about 170 of
# the 6,956 validation rows flagged.
#
# Next, the service side. It receives JSON bodies (dicts), so it is written
# row by row. It contains **three** bugs, each of a kind that occurs in
# practice.

# %%
from churnlab.api import example_payload


def serve_preprocess(customers: list[dict], stats: dict) -> np.ndarray:
    """The service's re-implementation. Looks right; is not."""
    rows = []
    for c in customers:
        nums = []
        for col in sorted(NUMERIC):  # (1) alphabetical order, not the training order
            v = c.get(col)
            if v is None:
                v = 0.0  # (2) "missing means zero" - training used the median
            if col in LOG_FEATURES and col != "tickets_90d":  # (3) one log1p forgotten
                v = np.log1p(v)
            nums.append((v - stats["mean"][col]) / stats["std"][col])
        onehot = [1.0 if str(c[col]) == v else 0.0 for col in CATEGORICAL for v in stats["categories"][col]]
        rows.append(nums + onehot)
    return np.array(rows)


payloads = [example_payload(row) for _, row in valid.iterrows()]
p_served = manual_model.predict_proba(serve_preprocess(payloads, stats))[:, 1]


def skew(p_a, p_b, t=threshold) -> dict:
    return {"max_abs_diff": round(float(np.max(np.abs(p_a - p_b))), 4),
            "decision_flips": int(np.sum((p_a >= t) != (p_b >= t))),
            "pr_auc": round(float(average_precision_score(valid[LABEL], p_b)), 3)}


print("offline vs served:", skew(p_offline, p_served))

# %% [markdown]
# **Expected:** a maximum difference around 0.7 and several hundred flipped
# decisions out of 6,956, with no error raised anywhere.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-serving-skew.svg" alt="Wrong: preprocessing written once in a notebook and again in the service, with one step forgotten, gives different scores from the same coefficients. Right: one fitted pipeline is saved and served" width="100%">
#
# *What to notice:* the coefficients are identical in both rows of the top half; the damage is done entirely before the model sees the numbers.
#
# > **Intuition.** Skew is two translators working from the same novel
# > without comparing notes: each chapter reads fine, and the two books tell
# > different stories. *Where it breaks down:* translators make random
# > mistakes; skew bugs are systematic (a whole column scaled wrongly), which
# > is why a single parity test on real rows catches them.
#
# Which bug did how much damage? Apply each one alone to the correct transform:

# %%
X = train_transform(valid, stats)
i = NUMERIC.index("tickets_90d")
X_nolog = X.copy()
X_nolog[:, i] = (valid["tickets_90d"].astype(float) - stats["mean"]["tickets_90d"]) / stats["std"]["tickets_90d"]
X_zero = train_transform(valid.fillna({c: 0.0 for c in NUMERIC}), stats)
X_order = X.copy()
X_order[:, :len(NUMERIC)] = X[:, [NUMERIC.index(c) for c in sorted(NUMERIC)]]

bug_table = pd.DataFrame({
    "(1) column order": skew(p_offline, manual_model.predict_proba(X_order)[:, 1]),
    "(2) impute 0, not median": skew(p_offline, manual_model.predict_proba(X_zero)[:, 1]),
    "(3) forgot log1p(tickets_90d)": skew(p_offline, manual_model.predict_proba(X_nolog)[:, 1]),
    "none (offline)": skew(p_offline, p_offline),
}).T
bug_table

# %% [markdown]
# **Expected:**
#
# * **Column order** is the most destructive (several hundred flips, PR-AUC
#   falls to about 0.07) and the *easiest to miss in code review*, because
#   every line looks correct.
# * **Imputing 0** flips only a few dozen decisions: just two columns have
#   missing values, and only some rows.
# * **The forgotten log1p** flips about a hundred decisions while PR-AUC
#   barely moves (0.158 → 0.157). An aggregate metric computed on served
#   scores would *not* reveal it. Individual customers get different
#   decisions all the same.
#
# The lesson is not "be more careful". Two implementations of the same
# transform will drift apart, so there should be only **one**.

# %% [markdown]
# ## 11.2 · The fix: one object, trained and served
#
# The registered models are scikit-learn `Pipeline`s. Preprocessing (impute,
# log, scale, one-hot) is *inside* the object that was fitted, saved and
# hashed. The service does not re-implement anything: it builds a DataFrame
# from the request and calls `predict_proba`.

# %%
from churnlab.artifacts import load_bundle
from churnlab.features import FEATURES

model, meta = load_bundle(registry.root / registry.current())
print("pipeline steps:", [(name, type(step).__name__) for name, step in model.steps])
for name, transformer, cols in model.named_steps["pre"].transformers_:
    if name != "remainder":
        print(f"  {name:<4} {transformer if isinstance(transformer, str) else type(transformer).__name__:<14} {cols}")
print("features stored in metadata == API order:", meta["features"] == FEATURES)

# %% [markdown]
# **Expected:** a `Pipeline` with two steps (`pre`: `ColumnTransformer`,
# `clf`: `LogisticRegression`); inside `pre`, the `log` and `num` branches
# (each a small `Pipeline` of imputer, optional log1p and scaler) and the `cat`
# one-hot encoder, each with its column list; then `True`.
#
# Two further defences are already in `churnlab.api`:
#
# * `ColumnTransformer` selects columns **by name**, so the order of keys in
#   the JSON cannot matter (bug 1 is impossible).
# * At start-up, `ModelHolder.load` refuses a model whose stored feature list
#   differs from the API's (`meta["features"] != FEATURES`).
#
# This is what the API does with a request, in four lines (from `_score` in
# `api.py`). Compare it with the offline path on the same rows:

# %%
rows = valid.sample(500, random_state=0).reset_index(drop=True)
bodies = [example_payload(r) for _, r in rows.iterrows()]

frame = pd.DataFrame(bodies)[FEATURES]
frame[NUMERIC] = frame[NUMERIC].astype(float)  # None -> NaN for the optional fields
p_api_path = model.predict_proba(frame)[:, 1]
p_offline_path = model.predict_proba(rows[FEATURES])[:, 1]
print("max |offline - api path|:", np.max(np.abs(p_offline_path - p_api_path)))

# %% [markdown]
# **Expected:** `0.0`: bit-for-bit identical. Same object, same inputs, same
# arithmetic.

# %% [markdown]
# ## 11.3 · The parity test: offline vs the running API
#
# The previous cell imitated the API. A **parity test** calls the real one:
# score the same rows offline and through `/predict/batch`, then compare
# scores *and* decisions. It catches anything between the request and the
# model: schema coercions, missing-value handling, a stale model, a wrong
# threshold.

# %%
from fastapi.testclient import TestClient

from churnlab.api import create_app
from churnlab.logs import ListHandler, get_logger
from churnlab.settings import Settings

log_handler = ListHandler()
quiet_logger = get_logger("churn-api-nb10", handler=log_handler)
api = TestClient(create_app(Settings(registry_dir=registry.root), logger=quiet_logger))

info = api.get("/model").json()
resp = api.post("/predict/batch", json={"customers": bodies}).json()
p_http = np.array([p["churn_probability"] for p in resp["predictions"]])
flag_http = np.array([p["flagged"] for p in resp["predictions"]])

diff = np.abs(p_offline_path - p_http)
print(f"model {info['model_version']} | threshold {info['threshold']}")
print(f"max abs diff        : {diff.max():.2e}")
print(f"max diff vs rounded : {np.max(np.abs(np.round(p_offline_path, 6) - p_http)):.2e}")
print(f"decision flips      : {int(np.sum((p_offline_path >= info['threshold']) != flag_http))}")
print(f"ids in order        : {[p['customer_id'] for p in resp['predictions']] == rows['customer_id'].tolist()}")

# %% [markdown]
# **Expected:** a maximum difference of about **5e-7** (not 1e-15); zero
# after rounding; zero decision flips; ids in order.
#
# > **Intuition.** A parity test is weighing the same parcel on the
# > warehouse scale and the shop scale. They need not agree to the microgram
# > (the API rounds), but a gap bigger than the rounding means one scale is
# > broken. *Where it breaks down:* it only tests the rows you send; pick rows
# > with missing values and rare categories, not just typical ones.
#
# Why not exactly zero? Read `_score` in `api.py`:
# `churn_probability=round(float(s), 6)`. The API **rounds to six decimals**
# for readability, so differences up to half of 1e-6 are expected; a test
# should allow a little more (1e-6) so that floating-point noise in the
# rounding cannot make it flaky.
# The decision (`flagged`) is computed *before* rounding, so decisions match
# exactly. A parity test is only useful once you know which differences are
# expected and why. Choose a tolerance from the contract, not from whatever
# number makes the test pass.
#
# Save the evidence:

# %%
(ART / "serving").mkdir(exist_ok=True)
parity = pd.DataFrame({"customer_id": rows["customer_id"], "snapshot_date": rows["snapshot_date"],
                       "p_offline": p_offline_path, "p_api": p_http,
                       "flag_offline": p_offline_path >= info["threshold"], "flag_api": flag_http})
parity.to_csv(ART / "serving" / "parity_valid.csv", index=False)
parity.head(3)

# %% [markdown]
# > **Exercise 1.** Write `parity_check(model, http_client, rows_df)` that
# >
# > 1. reads the threshold from `GET /model`;
# > 2. sends every row of `rows_df` (a slice of the feature table) to
# >    `POST /predict/batch` in **one** request (use `example_payload`);
# > 3. scores the same rows offline with `model.predict_proba`;
# > 4. returns `{"max_abs_diff": float, "n_decision_flips": int}`, where a flip
# >    is a row whose offline decision (`p >= threshold`) differs from the API's
# >    `flagged`.
# >
# > The grader runs it against the correct model (expects a difference under
# > 1e-6 and no flips) and against a subtly wrong one (expects your check to
# > notice).

# %%
def parity_check(model, http_client, rows_df: pd.DataFrame) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(parity_check)

# %% [markdown]
# ## 11.4 · Field order, missing optional fields, unseen categories
#
# Three things a caller can legitimately do that must not change the answer
# (or crash the service):

# %%
base = bodies[0]
variants = {
    "as sent": base,
    "keys reversed": dict(reversed(list(base.items()))),
    "optional fields null": {**base, "features_used_4w": None, "avg_resolution_hours_90d": None},
    "optional fields omitted": {k: v for k, v in base.items()
                                if k not in ("features_used_4w", "avg_resolution_hours_90d")},
}
for name, body in variants.items():
    r = api.post("/predict", json=body)
    print(f"{name:<24} {r.status_code}  p={r.json()['churn_probability']}")

nan_row = rows.iloc[[0]][FEATURES].copy()
nan_row[["features_used_4w", "avg_resolution_hours_90d"]] = np.nan
print(f"{'offline with NaN':<24}      p={model.predict_proba(nan_row)[0, 1]:.6f}")

# %% [markdown]
# **Expected:** the first two give the same probability; *null* and *omitted*
# give the same probability as each other and as the offline row with NaN.
# A missing optional value has exactly one representation inside the service
# (NaN), and the pipeline imputes it with the *training* median.
#
# Now a region the model has never seen. The contract accepts it (a plain
# string, chapter 10); the one-hot encoder ignores unknown categories:

# %%
known_regions = sorted(train["region"].astype(str).unique())
mars = {**base, "region": "MARS"}
r = api.post("/predict", json=mars)
print("known regions:", known_regions)
print("region=MARS ->", r.status_code, r.json()["churn_probability"])

# Monitoring's job: count it. The app keeps recent inputs in memory (notebook 14 builds on this).
recent = pd.DataFrame(list(api.app.state.metrics.recent_inputs))
unseen = ~recent["region"].isin(known_regions)
print(f"recent inputs: {len(recent)} | unseen region values: {int(unseen.sum())} "
      f"({recent.loc[unseen, 'region'].unique().tolist()})")

# %% [markdown]
# **Expected:** `known regions: ['APAC', 'EU', 'LATAM', 'NA', 'unknown']`; MARS
# returns 200 with a valid probability; and the monitoring count finds one
# unseen value. The trade-off from chapter 10 in action: the request
# succeeds, and the unusual input is **counted** rather than refused. A spike in
# that count after a deploy (or after the sales team opens a new market) is
# an alert, not a crash.

# %% [markdown]
# ## 11.5 · Loading a model safely
#
# A bundle is a directory with `model.joblib` and `metadata.json` (Chapter
# 9). `load_bundle` checks two things before returning the model:
#
# 1. the SHA-256 of `model.joblib` equals `metadata["model_sha256"]`, so a corrupted
#    or swapped file is refused;
# 2. the scikit-learn version recorded at training time has the same
#    major.minor as the running one. A mismatch is a **warning** by default and
#    an **error** with `strict_versions=True`, because pickles can load "fine"
#    across versions and still behave differently.
#
# Never experiment on the live registry. Copy a bundle to a scratch
# registry and break the copy.

# %%
import json
import shutil
import warnings

from churnlab.artifacts import ArtifactError
from churnlab.registry import Registry

scratch = ART / "nb10_scratch"
shutil.rmtree(scratch, ignore_errors=True)


def copy_registry(name: str) -> Registry:
    reg = Registry(scratch / name)
    shutil.copytree(registry.root / "v1", reg.root / "v1")
    reg.promote("v1", reason="copy for an experiment")
    return reg


# (a) tampering: one extra byte in the model file
tampered = copy_registry("tampered")
with open(tampered.root / "v1" / "model.joblib", "ab") as fh:
    fh.write(b"\x00")
try:
    load_bundle(tampered.root / "v1")
except ArtifactError as exc:
    print("tampered model ->", type(exc).__name__ + ":", exc)

# (b) a version mismatch: edit only the recorded sklearn version
old_version = copy_registry("old_sklearn")
meta_path = old_version.root / "v1" / "metadata.json"
m = json.loads(meta_path.read_text())
m["libraries"]["sklearn"] = "1.2.2"
meta_path.write_text(json.dumps(m, indent=2))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    load_bundle(old_version.root / "v1")
print("version mismatch ->", [str(w.message) for w in caught])
try:
    load_bundle(old_version.root / "v1", strict_versions=True)
except ArtifactError as exc:
    print("strict mode      ->", type(exc).__name__ + ":", exc)

# %% [markdown]
# **Expected:** the tampered file raises `ArtifactError ... does not match the
# hash`; the edited version gives a `UserWarning` *"model trained with sklearn
# 1.2.2, running 1.x"* and, in strict mode, an `ArtifactError` with the same
# text.
#
# Notice what was **not** detected: editing `metadata.json` did not trip the
# hash check, because the hash covers only `model.joblib`. That means someone
# could also edit the `threshold` in metadata and the service would serve it
# silently. A stronger bundle would also hash (or sign) the metadata. The
# general point is worth stating in an interview: *know what your integrity
# check does not cover.*
#
# What does the **service** do with a bad bundle, or no bundle at all? It must
# not crash: it starts, answers `/health`, and reports **not ready** until a
# good model is available.

# %%
empty_dir = scratch / "empty_registry"
apps = {
    "empty registry": Settings(registry_dir=empty_dir),
    "tampered bundle": Settings(registry_dir=tampered.root),
}
for name, s in apps.items():
    c = TestClient(create_app(s, logger=quiet_logger))
    ready = c.get("/ready")
    print(f"{name}: /health {c.get('/health').status_code} | /ready {ready.status_code} "
          f"| /predict {c.post('/predict', json=base).status_code}")
    print("    detail:", ready.json()["detail"][:90])

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    c = TestClient(create_app(Settings(registry_dir=old_version.root), logger=quiet_logger))
print(f"old sklearn version: /ready {c.get('/ready').status_code} | warnings at start-up: {len(caught)}")

# %% [markdown]
# **Expected:** both broken registries give `/health 200 | /ready 503 |
# /predict 503`, with the reason in `detail` (`no CURRENT model in ...` and
# `ArtifactError: model.joblib does not match ...`). The version-mismatch
# registry is **ready** (200) and logged one warning at start-up.
#
# Why 503 and not a crash? An orchestrator (Kubernetes, a load balancer)
# sends traffic only to instances whose readiness check passes. A process that
# crashes on start is restarted in a loop; a process that reports "not ready"
# stays visible, keeps its logs, and starts serving the moment a reload
# succeeds (chapter 15 uses `/admin/reload` for rollback).

# %% [markdown]
# ## 11.6 · Which model answered? Versions and request ids
#
# Every prediction must be traceable to the exact model that produced it,
# and every request to its log line. The service does both:

# %%
r = api.post("/predict", json=base, headers={"x-request-id": "demo-req-0001"})
print("body model_version :", r.json()["model_version"])
print("x-model-version    :", r.headers["x-model-version"])
print("x-request-id       :", r.headers["x-request-id"])

line = [e for e in log_handler.parsed() if e.get("request_id") == "demo-req-0001"]
print("log line           :", line[-1])

# %% [markdown]
# **Expected:** `v1` in the body and in the header; the request id you sent
# echoed back; and a JSON log line with the same `request_id`, `path`,
# `status`, `latency_ms` and `model_version`.
#
# The caller sets `x-request-id` (or the service generates one). When a
# customer-success manager asks *"why was this account flagged on 3 June?"*,
# the id links their screenshot to the log line, and the version to the exact
# bundle, its training data fingerprint and its threshold.

# %% [markdown]
# > **Exercise 2.** The buggy `serve_preprocess` from 11.1 is still in this
# > notebook. Write `fixed_serve_preprocess(customers, stats)` that takes a
# > list of request dicts (as produced by `example_payload`; optional fields
# > may be `None` or missing, keys may be in any order) and returns **exactly**
# > what `train_transform` returns for the same rows. The grader compares your
# > output with the reference transform on random validation rows, including
# > missing values and an unseen region.
# >
# > (There is a one-line solution. Why is it the *right* solution and not a
# > shortcut?)

# %%
def fixed_serve_preprocess(customers: list[dict], stats: dict) -> np.ndarray:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(fixed_serve_preprocess)

# %% [markdown]
# > **Exercise 3 (written).** The parity test in 11.3 replays *stored* feature
# > rows through the API. List three sources of training/serving skew that it
# > would **not** catch, and for each, how you would detect or prevent it.
# > (Think about where the feature values in a live request come from.)
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **How a skewed feature moves a score.** For logistic regression the log-odds
# are $\eta = \beta_0 + \sum_j \beta_j z_j$. If serving computes feature $j$ as
# $z_j + \delta_j$ instead of $z_j$:
#
# $$
# \Delta \eta = \beta_j\, \delta_j,
# \qquad
# p_{\text{served}} = \sigma\big(\eta + \beta_j \delta_j\big)
# $$
#
# *In words:* the error in the feature, times its coefficient, is added
# straight to the log-odds. *Symbols:* $\beta_j$ = coefficient,
# $\delta_j$ = feature error after scaling, $\sigma$ = sigmoid. *Example:*
# forgetting `log1p` on `sessions_4w` turns a standardised value of 0.5 into,
# say, 4.0 ($\delta = 3.5$); with $\beta = -0.6$ the log-odds fall by 2.1, and
# a 0.20 probability becomes $\sigma(\ln(0.25) - 2.1) = 0.03$: a call
# silently dropped.
#
# **Parity criterion.** For the same rows $i$:
#
# $$
# \max_i \left\lvert p^{\text{offline}}_i - p^{\text{API}}_i \right\rvert \le \varepsilon,
# \qquad \varepsilon = 10^{-6}
# $$
#
# *In words:* the largest disagreement must be below what rounding to six
# decimals can explain ($0.5 \times 10^{-6}$), with a little slack.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Parity test fails with differences around 5e-7 | the API rounds probabilities to 6 decimals | compare with `atol=1e-6`, or compare against `np.round(p, 6)` |
# | Parity fails with large differences on a few rows | NaN vs `None` vs `0` handling, or a different model version | check `/model` version; check rows with missing optional fields |
# | Parity fails on every row | wrong column order into a NumPy model, a stale model, or a different threshold | serve the `Pipeline`; compare `meta["features"]`; read the threshold from `/model` |
# | `UserWarning: model trained with sklearn X, running Y` | the bundle was trained in a different environment (e.g. local vs Colab) | retrain in the serving environment, or pin the same version; use `strict_versions=True` in production |
# | `ArtifactError: ... does not match the hash` | the model file was modified, partially copied or corrupted | re-copy or retrain; never "fix" the hash in metadata |
# | `/ready` returns 503 | empty or wrong `registry_dir`, no `CURRENT`, or a failing bundle | read `detail`; `bootstrap.ensure_registry()` |
# | `ValueError: Out of range float values are not JSON compliant` | NaN in a request body | build bodies with `example_payload` (NaN → `None`) |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. What is training/serving skew, and why does it not show up as errors?
# 2. Name three ways a re-implemented preprocessing step can differ from training, and which is hardest to spot in review.
# 3. How does serving a saved `Pipeline` prevent skew, and what kinds of skew can it *not* prevent?
# 4. Your parity test shows a maximum difference of 4e-7. Pass or fail? What do you need to know to decide?
# 5. What does `load_bundle` verify, what does it not verify, and what should the service do when loading fails?
# 6. A customer complains about a prediction made last week. What do you need from the response or the logs to investigate it?
#
# **Next chapter:** [Chapter 12 · API clients & tests](../ch12-api-clients-and-tests/notebook.ipynb) — calling the service over real HTTP with timeouts and retries, and writing a test suite for it.

