# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 55 · ML platform design
#
# **Part III · Chapter 55 · about 90 minutes · CPU is enough (no GPU work in this lab)**
#
# Chapter 18 designed *one model* (churn) and Chapter 37 designed *one product*
# (a feed, a search engine). Staff-level design interviews often go one level
# higher: "design the ML platform that fifty teams will use to build, deploy and
# monitor their models", where the difficult parts are the contracts between
# teams, cost and adoption as much as algorithms.
#
# By the end of this lab you should be able to:
#
# 1. structure a platform design answer around its users, its contracts and its service levels, not around boxes;
# 2. design a feature platform (offline/online stores, streaming, point-in-time correctness, backfills, ownership) and size it;
# 3. design a training platform (data versioning, scheduling and GPU quota, experiment tracking, reproducibility) and cost it, including idle capacity;
# 4. design the path from registry to production (CI/CD/CT, shadow, canary, rollback) and compute how long a canary must run to see a regression;
# 5. design a monitoring platform that copes with delayed labels;
# 6. argue the organisational trade-offs (build vs buy, golden paths, cost governance) and record a decision as an ADR with a risk register.
#
# **You will produce:** `artifacts/level3/nb17/platform_costs.csv`,
# `artifacts/level3/nb17/risk_register.csv` and
# `artifacts/level3/nb17/adr-001-online-feature-store.md`, plus your own ADR (Exercise 4).
#
# **What is measured here.** Nothing in this notebook is a measurement of a
# real platform. Every number is computed from an input printed under
# `ASSUMPTIONS`; change the inputs and the conclusions may change. That is
# the skill being practised.

# %% [markdown]
# Read first: [Chapter 55 text](README.md), which explains the components of an ML platform, their contracts and their costs in more detail than this lab.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# Part III is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing to rebuild; this notebook is design and arithmetic ----
import csv

from scaleml import design as ds
from grading import ch55 as grade

OUT = ART / "level3" / "nb17"
OUT.mkdir(parents=True, exist_ok=True)
print("calculators:", ", ".join(n for n, f in vars(ds).items()
                               if callable(f) and getattr(f, "__module__", "") == ds.__name__))

# %% [markdown]
# **Expected:** the artifact path and a list of calculator names from
# [`src/scaleml/design.py`](../../src/scaleml/design.py). Read that file: it is
# about 250 lines of arithmetic, and every function says what it assumes.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | ML platform | Shared infrastructure many teams use to build, deploy and monitor models | "fifty teams" in the prompt |
# | Feature store | A system that defines features once and serves them for training (offline) and serving (online) | offline columnar + online key-value |
# | Point-in-time correctness | Training rows only use feature values known *at* the label's time | as-of joins on `(entity, label_time)` |
# | Backfill | Computing a new feature's history over past data with the same code | a new feature over the training window |
# | MFU | Model FLOPs utilisation: achieved useful FLOP/s ÷ the hardware's peak | 0.30 assumed |
# | Gang scheduling | Starting all workers of a multi-GPU job together, or none | distributed training jobs |
# | Showback / chargeback | Showing each team its cost / actually billing it | per-team GPU spend |
# | Model registry | A database of model versions with signature, data snapshot, evaluation and approval | the step before deployment |
# | CI / CD / CT | Continuous integration / delivery / training: automated tests, rollout and retraining | section 55.3 |
# | Shadow / canary | New model scores live traffic without effect / serves a small share of real traffic | 1% canary |
# | Label maturity | Fraction of eventual labels already observed | 30% after 7 days (20-day mean delay) |
# | ADR | Architecture decision record: one decision, its options, consequences and revisit triggers | `adr-001-online-feature-store.md` |

# %% [markdown]
# ## 55.0 · From "design a model" to "design a platform"
#
# A platform question changes who the user is. In a product design the user
# is the person scrolling the feed; in a platform design the users are **other
# engineers and scientists**, and the platform succeeds only if they choose to
# use it. That changes the answer's skeleton:
#
# | # | Step | Questions to answer out loud | Minutes (of 45) |
# |---|---|---|---|
# | 1 | **Users and use cases** | Which teams, how many models, what kinds (ranking, forecasting, LLM apps)? Batch or online? What do they do today, and what hurts? | 4 |
# | 2 | **Requirements and SLOs** | Freshness, latency (p99), availability, scale (entities, QPS, training jobs/day), compliance (deletion, audit) | 4 |
# | 3 | **Components** | Feature platform, training platform, registry + deployment, serving, monitoring. Draw them once, then go deep on two | 10 |
# | 4 | **Contracts** | The interfaces between teams: feature definitions, model signatures, ownership, on-call | 6 |
# | 5 | **Trade-offs** | For each deep dive: two options, the criterion that decides, what would change your mind | 8 |
# | 6 | **Adoption and migration** | Golden path, what you build first, how the first three teams onboard, what you deliberately do *not* build | 5 |
# | 7 | **Cost and governance** | Storage, compute, GPU quota, who pays (showback/chargeback), idle capacity | 4 |
# | 8 | **Risks** | A short register: likelihood × impact, mitigation, owner | 4 |
#
# A one-screen picture of the components to walk through (draw something
# like it in the first ten minutes, then *stop drawing* and go deep):
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-ml-platform.svg" alt="Sources feed a feature platform, a training platform, a registry with deployment and serving; a metadata layer spans them and a monitoring platform underneath can trigger retraining" width="100%">
#
# *What to notice:* every arrow is a contract between teams (feature definitions, data snapshots, model signatures); draw this once in the interview, then go deep on two boxes.
#
# **What a staff answer adds over a senior one** (generic signals interviewers
# commonly look for, not any company's ladder): a senior answer designs each
# component correctly; a staff answer also says **which component to build
# first and which to buy or postpone**, defines the *contracts* that let fifty
# teams move independently, puts numbers on cost and on the platform's own
# success (time from idea to first online experiment, share of models on the
# golden path), and names the organisational failure modes (a platform no one
# adopts; a feature store nobody owns).
#
# Published reading for more depth (each describes a published design at
# the time it was written, not anyone's current practice): *"Hidden Technical
# Debt in Machine Learning Systems"* (Sculley et al., NeurIPS 2015), *"TFX: A
# TensorFlow-Based Production-Scale Machine Learning Platform"* (Baylor et al.,
# KDD 2017), and *"Monolith: Real Time Recommendation System With Collisionless
# Embedding Table"* (Liu et al., 2022).

# %% [markdown]
# ## 55.1 · The feature platform
#
# Chapters 21–22 built point-in-time joins and streaming features
# for one project. At platform scale the questions are about **guarantees and
# ownership**:
#
# | Concern | Design choice | Trade-off to say out loud |
# |---|---|---|
# | **One definition, two stores** | Features defined once in code (a transformation + entity key + timestamp); the platform materialises them to an *offline* columnar store (training) and an *online* key-value store (serving) | Without it, training/serving skew is the default; with it, every feature has to fit the platform's DSL, and exotic features need an escape hatch |
# | **Point-in-time correctness** | Offline store keeps history with event timestamps; training sets built by as-of joins on `(entity, label_time)` | History costs storage (versions × retention); snapshot-only stores are cheaper and leak the future |
# | **Streaming** | Windowed aggregates (counts, sums over 5 min / 1 h / 1 d) computed from event streams and written to the online store; the same logic backfilled in batch | Two code paths (stream + batch) drift; one engine for both is simpler but may limit latency or throughput |
# | **Freshness SLO** | Per feature group: e.g. "user_last_10_clicks ≤ 60 s", "user_30d_spend ≤ 24 h" | Fresher is more expensive; most features do not need it. Ask which features move the metric when fresh |
# | **Backfill** | New feature → compute its history over the training window from raw logs, with the *same* code; versioned, never overwrite in place | Backfills compete with production jobs for compute; schedule and cap them; a backfill that uses today's dimension tables leaks |
# | **Ownership** | Every feature group has an owning team, an on-call, a freshness SLO, a deprecation path; consumers are tracked (lineage) | Without owners, a 5,000-feature store becomes unmaintainable and nobody dares delete anything |
# | **Privacy and deletion** | Entity-keyed storage makes "delete user X" a keyed delete; offline history must honour it too | Retention is a cost lever *and* a compliance requirement |
#
# > **Intuition.** A feature store is one recipe cooked in two kitchens: a slow, large batch kitchen
# > that prepares every past meal for training, and a fast counter that serves today's meal in
# > milliseconds. Training/serving skew is the two kitchens following slightly different recipes.
# > *Where the analogy breaks down:* streaming features are computed *as events arrive*, so even with
# > one recipe the two kitchens can differ in timing (late events, windows), which is why
# > point-in-time joins and backfills matter.
#
# **Sizing.** The online store is usually the expensive one, because it lives in
# memory or on fast SSD and is replicated. A worked estimate:

# %%
fs = {
    "entities (users)": 200_000_000,
    "online features per entity": 400,
    "bytes per value (fp32/int32 average)": 4,
    "key bytes": 16,
    "store overhead factor": 1.5,
    "replicas (online store)": 3,
    "offline: rows per day (one snapshot per user)": 200_000_000,
    "offline: days retained": 365,
    "offline: compression ratio (columnar)": 4,
    "price per GB-month, in-memory KV (placeholder)": 4.00,
    "price per GB-month, object storage (placeholder)": 0.02,
}
ds.show_assumptions(fs)
online_gb = ds.online_store_gb(fs["entities (users)"], fs["online features per entity"],
                               fs["bytes per value (fp32/int32 average)"], key_bytes=fs["key bytes"],
                               overhead=fs["store overhead factor"], replication=fs["replicas (online store)"])
offline_tb = ds.offline_store_tb(fs["offline: rows per day (one snapshot per user)"], fs["online features per entity"],
                                 fs["bytes per value (fp32/int32 average)"], fs["offline: days retained"],
                                 compression_ratio=fs["offline: compression ratio (columnar)"])
cost_rows = [
    {"component": "online feature store", "size": f"{online_gb:,.0f} GB",
     "monthly_cost": ds.storage_monthly_cost(online_gb, fs["price per GB-month, in-memory KV (placeholder)"])},
    {"component": "offline feature history", "size": f"{offline_tb:,.1f} TB",
     "monthly_cost": ds.storage_monthly_cost(offline_tb * 1000, fs["price per GB-month, object storage (placeholder)"])},
]
ds.table(cost_rows)

# %% [markdown]
# **Expected:** about **1,454 GB** for the online store (200M × (16 + 1,600)
# bytes × 1.5 overhead × 3 replicas) and **29.2 TB** of compressed offline
# history; with the placeholder prices, roughly 5,800 vs 580 a month. The
# prices are *placeholders* (look up the provider's current list prices); the
# **ratio** is the lesson: the online store costs ~10× the offline one while
# holding ~1/20 of the bytes.
#
# The levers a staff answer names, each with its cost: serve only the features
# that a model in production actually reads (a usage-based garbage collector
# on the online store); quantise or bucket values (fp16, int8); keep a hot tier
# for active users and fall back to defaults or a slower tier for dormant ones;
# store one snapshot per *change* rather than per day offline.
#
# > **Exercise 1.** Write `feature_store_storage_gb(entities, features, bytes_per, versions, replication)`:
# > the raw storage in **decimal GB** (1 GB = 10⁹ bytes) of `entities` rows ×
# > `features` values × `bytes_per` bytes, kept in `versions` copies (e.g. 2
# > while a new materialisation is written next to the live one, blue/green)
# > and `replication` replicas. Return a float; raise `ValueError` if any input
# > is negative. Then say aloud the answer for 200M × 400 × 4 B × 2 × 3 without running it.

# %%
def feature_store_storage_gb(entities, features, bytes_per, versions, replication):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(feature_store_storage_gb)

# %% [markdown]
# ## 55.2 · The training platform
#
# | Concern | Design choice | Trade-off |
# |---|---|---|
# | **Data versioning** | Training sets are immutable, content-addressed snapshots (a manifest of files + hashes + the feature/label query + its as-of time); a run records the snapshot id | Storage grows; garbage-collect snapshots not referenced by a registered model |
# | **Scheduling** | A queue with **quotas per team** (guaranteed share + borrowable burst), priorities, **gang scheduling** for multi-GPU jobs (all workers start together or none do), preemption of low-priority jobs with checkpoint/resume (Chapter 48) | Strict quotas waste idle GPUs; fully shared pools let one team starve others. "Guaranteed + borrow + preempt" is the usual middle |
# | **Experiment tracking** | Every run logs code version, config, data snapshot, environment (image digest), metrics, artefacts; searchable | Logging is cheap; *discipline* is not: make it the default in the golden-path template, not a wiki page |
# | **Reproducibility** | Pinned images, seeds, deterministic data order, recorded hardware; "reproducible" defined as *within a stated tolerance*, not bit-identical (GPU kernels are not always deterministic) | Full determinism costs speed (`torch.use_deterministic_algorithms`); decide which jobs need it (audited models) |
# | **GPU quota and cost** | Showback (teams see their spend) then chargeback; idle detection; right-sizing (a T4-class card for small fine-tunes, larger cards for large models); spot/preemptible capacity for checkpointed jobs | Chargeback changes behaviour but adds friction; start with showback |
#
# **Costing a training workload.** Two ways to estimate GPU-hours: from a
# measured step time (best; Chapters 39–40 and 47–48 showed how to measure
# it), or from FLOPs when there is nothing to measure yet. For a dense
# transformer, training compute ≈ **6 × parameters × tokens**; divide by the
# GPU's peak throughput times the **model FLOPs utilisation (MFU)** one
# expects to reach.

# %%
tr = {
    "model parameters": 1_000_000_000,
    "training tokens": 2_000_000_000,
    "GPU peak fp16 FLOP/s (T4 spec sheet: 65 TFLOPS)": 65e12,
    "MFU (assumed; 0.3-0.5 is typical for well-tuned training)": 0.30,
    "price per GPU-hour (placeholder)": 0.50,
}
ds.show_assumptions(tr)
flops = ds.train_flops(tr["model parameters"], tr["training tokens"])
gpu_h = ds.gpu_hours_from_flops(flops, tr["GPU peak fp16 FLOP/s (T4 spec sheet: 65 TFLOPS)"],
                                tr["MFU (assumed; 0.3-0.5 is typical for well-tuned training)"])
print(f"\ncompute: {flops:.2e} FLOPs -> {gpu_h:,.0f} GPU-hours at the assumed MFU "
      f"-> {gpu_h * tr['price per GPU-hour (placeholder)']:,.0f} at the placeholder price")
print("note: a 1B-parameter model does not fit a T4's 16 GB for full fine-tuning with Adam "
      "(16 bytes/param, Chapter 40) - the arithmetic says how long, not whether it fits.")

# %% [markdown]
# **Expected:** `1.20e+19 FLOPs -> 171 GPU-hours`, about 85 at the placeholder
# price, and the reminder that memory, not time, rules out this job on one T4.
# Both questions (does it fit? how long?) belong in every training-platform answer.
#
# The bill for a *platform* is many such jobs, and the number that surprises
# people is **idle capacity**: reserved GPUs that sit waiting for data,
# for a notebook someone left open, or for a job stuck in a queue behind a
# gang-scheduled one.
#
# > **Exercise 2.** Write `gpu_cost(hours, gpus, price)`: the cost of running
# > `gpus` GPUs for `hours` hours at `price` per GPU-hour. Return a float; raise
# > `ValueError` if any input is negative.

# %%
def gpu_cost(hours, gpus, price):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(gpu_cost)

# %% [markdown]
# > **Exercise 3.** Write `utilization_waste(hours, gpus, price, utilization)`
# > returning a dict with keys `total_cost` (as in Exercise 2), `useful_cost`
# > (the share actually used, `total_cost × utilization`), `wasted_cost` (the
# > rest) and `cost_per_useful_gpu_hour` (`price / utilization`). `utilization`
# > is a fraction in (0, 1]; raise `ValueError` outside that range or for
# > negative inputs. Do not round.

# %%
def utilization_waste(hours, gpus, price, utilization):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(utilization_waste)

# %% [markdown]
# A team-level view, using the same arithmetic (the module's own helpers, so
# this cell runs whether or not the exercises have been completed):

# %%
teams = [  # (team, reserved GPUs, average utilisation) - ASSUMED for the illustration
    ("ads ranking", 64, 0.72), ("search", 32, 0.55), ("llm apps", 48, 0.35), ("research", 16, 0.20),
]
price, hours = 2.00, ds.HOURS_PER_MONTH  # placeholder price per GPU-hour for a larger data-centre GPU
ds.show_assumptions({"price per GPU-hour (placeholder)": price, "hours per month": hours,
                     "reservations and utilisation": "illustrative, not measured"})
rows = []
for team, gpus, util in teams:
    total = gpus * hours * price
    rows.append({"team": team, "gpus": gpus, "utilisation": util, "monthly_cost": total,
                 "idle_cost": total * (1 - util), "cost_per_useful_gpu_h": price / util})
ds.table(rows)
idle = sum(r["idle_cost"] for r in rows)
total = sum(r["monthly_cost"] for r in rows)
print(f"\nidle share of the bill: {idle / total:.0%}  ({idle:,.0f} of {total:,.0f} a month)")

# %% [markdown]
# **Expected:** a four-row table and an idle share of **48%** (111,427 of
# 233,600 a month at the placeholder price). The "research" row pays 10 per
# *useful* GPU-hour for a card that costs 2 per hour. Fixes a staff engineer would propose, in order
# of effort: showback dashboards; idle reaping of interactive sessions;
# a shared, preemptible burst pool so reserved-but-idle GPUs serve other teams'
# checkpointed jobs; then right-sizing (small fine-tunes on small cards). Each
# needs a *policy* agreed between teams, which is why this is a staff topic.
#
# ## 55.3 · Registry, CI/CD/CT and safe rollout
#
# A model registry is a database of **versions with contracts**: the artefact,
# its input/output **signature**, the training snapshot and code version, the
# evaluation report, the model card, the approval state and who approved it.
# Around it:
#
# * **CI** (every change): unit tests on feature code, a schema/signature test,
#   a tiny training run that must reach a sanity loss, a serving smoke test.
# * **CD** (every candidate model): offline evaluation against the current
#   champion on the *same* snapshot with intervals (Chapter 38),
#   slice checks, latency and memory tests in the serving image, then
#   staged rollout.
# * **CT** (continuous training): retraining on a schedule or trigger
#   (drift, new data volume), producing a *candidate* that goes through the
#   same CD gates. Automating retraining without automating evaluation lets
#   bad models reach production unattended.
#
# | Stage | What it proves | What it cannot prove |
# |---|---|---|
# | **Shadow** (new model scores live traffic, results discarded) | it runs at production load, latency and error rate; prediction distribution vs champion | any effect on users (no one sees its output) |
# | **Canary** (1–5% of traffic) | no *large* regression in errors, latency, guardrail metrics | small effects: too little traffic, too little time (computed below) |
# | **A/B test** (50/50 or a fixed slice, days–weeks) | the business effect, with an interval | long-term effects; network effects if units interact |
# | **Rollback** | one command, previous version still loaded or warm; automatic on guardrail breach | rollback of *data* (a bad feature backfill) is harder: version features too |
#
# How long must a canary run? It is a two-proportion test like any A/B test
# (Chapter 29). Suppose the guardrail is the request error rate:

# %%
can = {"baseline error rate": 0.005, "regression to detect (relative)": 0.20,
       "alpha (two-sided)": 0.05, "power": 0.80, "service peak QPS": 2_000, "canary share": 0.01}
ds.show_assumptions(can)
n = ds.two_proportion_n(can["baseline error rate"], can["regression to detect (relative)"],
                        can["alpha (two-sided)"], can["power"])
canary_qps = can["service peak QPS"] * can["canary share"]
print(f"\nrequests needed per arm: {n:,}")
print(f"canary at {canary_qps:.0f} QPS -> {n / canary_qps / 60:,.0f} minutes at peak traffic")
for rel in (0.5, 1.0):
    m = ds.two_proportion_n(can["baseline error rate"], rel)
    print(f"  to detect +{rel:.0%} instead: {m:,} requests -> {m / canary_qps / 60:,.0f} minutes")

# %% [markdown]
# **Expected:** about **85,900** requests per arm, so **~72 minutes** of a 1%
# canary at peak just to see a +20% error-rate regression; a doubling of errors
# is visible in ~4 minutes. Hence the common staging pattern: a short canary
# catches *crashes and large* regressions, and small effects are left to the
# A/B test. In an interview this should be stated explicitly: it shows an understanding of what each stage can and cannot detect.
#
# > **Intuition.** A 1% canary is a smoke alarm, not a thermometer: it reliably notices a fire (a
# > crash, a doubling of errors within minutes) but cannot tell whether the room got one degree
# > warmer (a +2% effect needs far more traffic and time). *Where the analogy breaks down:* the
# > required sample grows with $1/\Delta^2$, so halving the effect to be detected quadruples the
# > canary's duration; there is no fixed "canary length".
#
# ## 55.4 · The monitoring platform
#
# Chapter 15 built drift checks (PSI) and a rollback for one model.
# A platform offers them as a **service** every model gets by registering:
#
# | Layer | Signals | Page or ticket? |
# |---|---|---|
# | **Data quality** (inputs) | schema violations, null rate, range checks, feature freshness vs its SLO, upstream job late | page if a production model reads it and freshness SLO is broken; otherwise ticket the owner |
# | **Prediction health** | score distribution drift vs a reference window, fallback/default rate, latency, error rate | page on fallback rate and errors; ticket on drift alone |
# | **Delayed-label performance** | metric on labels as they mature, compared at the *same maturity* | ticket; page only for large, confirmed drops |
# | **Business guardrails** | the product metrics the model can hurt | owned by the product team's dashboards, linked from the model page |
# | **Cost** | GPU-hours, serving replicas, storage per model | weekly showback |
#
# **Delayed labels** are the main difficulty. A fraud chargeback or a subscription
# cancellation can arrive weeks after the prediction. If label delay is roughly
# exponential with a mean of 20 days, the fraction of eventual labels already
# visible is:

# %%
mean_delay = 20  # days, ASSUMED
print("days since prediction:", "  ".join(f"{k:>4}" for k in (1, 7, 14, 30, 60, 90)))
print("labels visible:       ", "  ".join(f"{ds.label_maturity(k, mean_delay):>4.0%}" for k in (1, 7, 14, 30, 60, 90)))

# %% [markdown]
# **Expected:** 5%, 30%, 50%, 78%, 95%, 99%. Two consequences a platform must
# build in: (1) compare model versions **at equal label maturity** (yesterday's
# predictions have fewer positives than last month's, which looks like a drop
# in recall but is not); (2) use **leading indicators** while labels mature:
# input drift, score drift, early proxy labels (a disputed transaction before
# the chargeback), and a small labelled sample from human review.
#
# > **Intuition.** Grading an exam while students are still handing in papers: yesterday's batch
# > always looks like it has fewer correct answers, simply because fewer papers are in. Compare two
# > models only on papers that have had the same time to arrive. *Where the analogy breaks down:*
# > label delay can depend on the case itself (big frauds are reported faster), so "same maturity"
# > is necessary but not always sufficient.
#
# ## 55.5 · Organisational trade-offs
#
# **Build vs buy.** Decide per component, not for the whole platform:
#
# | Criterion | Favours buy (managed service / open source as-is) | Favours build |
# |---|---|---|
# | Differentiation | commodity (experiment tracking, object storage, schedulers) | what makes the organisation's models better (feature logic at its scale, a custom serving path) |
# | Scale and latency | inside the product's documented limits | far beyond them, or a latency the product cannot meet |
# | Team capacity | the team cannot staff on-call for it | the team already runs something similar |
# | Lock-in and cost curve | cost grows gently with use | per-unit pricing becomes the largest line at the organisation's scale |
# | Compliance | vendor meets the data-residency and audit needs | it does not |
#
# **Adoption: golden paths.** A platform nobody adopts is a cost. The pattern
# that tends to work: one *paved road* (a project template with feature
# definitions, a training job, registry entry, deployment config and
# dashboards wired up) that takes a new model from repository to shadow
# traffic in days; escape hatches for teams with unusual needs; migration
# help for the first two or three teams, chosen because they are influential
# and have a real pain the platform removes. Measure the platform like a
# product: *time from new repo to first shadow deployment*, *share of
# production models on the paved road*, *pages per model per month*.
#
# **Cost governance.** Showback per team and per model; quotas with borrowable
# burst; idle reaping; a monthly review of the top ten cost lines; a rule that
# every production model has an owner who can be asked "is this still worth
# its cost?". Deleting unused features and models is a feature of the platform.
#
# **What not to build first.** A common staff-level answer: build the
# registry + deployment path and the monitoring hooks first (every model needs
# them, and they make rollback safe); adopt an existing experiment tracker and
# scheduler; build the feature platform incrementally, starting from the
# offline store and point-in-time training sets, adding the online store when
# the first latency-sensitive model needs it.
#
# Now write the platform's cost summary to disk, for inclusion in a design document:

# %%
platform_costs = cost_rows + [
    {"component": "training GPUs (4 teams, reserved)", "size": f"{sum(t[1] for t in teams)} GPUs",
     "monthly_cost": total},
    {"component": "  of which idle", "size": f"{idle / total:.0%}", "monthly_cost": idle},
]
with (OUT / "platform_costs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["component", "size", "monthly_cost"])
    w.writeheader()
    w.writerows([{**r, "monthly_cost": round(r["monthly_cost"], 2)} for r in platform_costs])
ds.table(platform_costs)
print("\nwrote", (OUT / "platform_costs.csv").relative_to(ART))

# %% [markdown]
# **Expected:** four rows (online store, offline history, training GPUs, idle)
# and `wrote level3/nb17/platform_costs.csv`. Under these assumptions GPUs
# dominate the bill by more than an order of magnitude over feature storage,
# which is where cost governance should start.
#
# ## 55.6 · A risk register
#
# Staff engineers are expected to say what could go wrong *with the
# programme*, not only with a model. A register scores each risk by
# likelihood × impact (1–5 each) and names a mitigation and an owner:

# %%
risks = [
    {"risk": "Training/serving skew from duplicated feature logic", "likelihood": "likely", "impact": "major",
     "mitigation": "single feature definition; skew monitor comparing logged online values to offline recompute",
     "owner": "feature platform"},
    {"risk": "Platform not adopted; teams keep bespoke stacks", "likelihood": "possible", "impact": "major",
     "mitigation": "paved road with 2 pilot teams; measure time-to-shadow; migration support", "owner": "platform PM/lead"},
    {"risk": "Online store outage takes down all rankers", "likelihood": "unlikely", "impact": "severe",
     "mitigation": "per-model default features + cached fallbacks; multi-zone replicas; load-shedding", "owner": "serving"},
    {"risk": "GPU spend grows faster than value", "likelihood": "likely", "impact": "moderate",
     "mitigation": "showback, idle reaping, preemptible burst pool, quarterly review", "owner": "infra + finance"},
    {"risk": "Retrained model ships a regression automatically", "likelihood": "possible", "impact": "major",
     "mitigation": "CT produces candidates only; CD gates with intervals and slices; auto-rollback on guardrails",
     "owner": "ML platform"},
    {"risk": "User deletion requests not honoured in offline history", "likelihood": "possible", "impact": "severe",
     "mitigation": "entity-keyed storage, deletion jobs with audit log, retention limits", "owner": "data governance"},
]
register = ds.risk_register(risks)
with (OUT / "risk_register.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(register[0]))
    w.writeheader()
    w.writerows(register)
ds.table([{k: r[k] for k in ("risk", "L", "I", "score", "level", "owner")} for r in register])

# %% [markdown]
# **Expected:** six risks sorted by score: training/serving skew (16) and
# unhonoured deletion requests (15) are HIGH; GPU spend, adoption and an
# automatically deployed regression score 12; the online-store outage 10 (rare
# but severe, which is why its mitigation is designed in from day one). The scores are judgments, not data; their value is forcing
# the conversation about mitigations and owners.
#
# ## 55.7 · An architecture decision record (ADR)
#
# Staff-level work is written down. An ADR records one decision: the context,
# the options, the decision, its consequences, and *when to revisit it*. Here
# is one, filled in for a single platform decision, generated by
# `ds.adr_markdown` and saved:

# %%
adr = ds.adr_markdown(
    1, "Online feature store: managed key-value service vs self-hosted cluster",
    status="Accepted (illustrative)", decided_on="2026-01-15",
    context=("Ten ranking models need ~400 features per user at p99 < 10 ms for ~2,000 peak lookups/s "
             "(ASSUMED). Sizing in 55.1 gives ~1.5 TB in memory with 3 replicas. The platform team "
             "has 4 engineers and no existing on-call for stateful stores."),
    options={
        "A · managed KV service": "no on-call for storage nodes, autoscaling; per-GB price higher; vendor limits on item size",
        "B · self-hosted open-source KV cluster": "lowest unit cost at scale, full control; needs storage expertise and 24/7 on-call",
        "C · no online store yet (precompute scores in batch)": "cheapest; fails for features that must be fresh within minutes",
    },
    decision=("Option A for the first 12 months, behind a thin feature-serving API owned by the platform, "
              "so that the storage engine can be replaced without changing any model's code. "
              "Option C stays the default for models that do not need fresh features."),
    consequences=[
        "Higher storage cost per GB than B (see platform_costs.csv); accepted to avoid a stateful on-call rotation.",
        "The feature-serving API becomes a contract: versioned, with latency SLO and per-model default values.",
        "Vendor limits (item size, throughput per partition) must be load-tested before the second model onboards.",
    ],
    revisit_when=[
        "online store cost exceeds 25% of the ML platform's monthly bill",
        "the platform team can staff a storage on-call (>= 6 engineers)",
        "p99 lookup latency SLO is missed in two consecutive months",
    ],
)
(OUT / "adr-001-online-feature-store.md").write_text(adr)
print(adr)

# %% [markdown]
# **Expected:** the ADR printed as Markdown (title, status, context, three
# options, decision, consequences, revisit triggers) and saved as
# `adr-001-online-feature-store.md`. Note what makes it useful: the decision
# is *reversible by design* (the API hides the engine), and the "revisit when"
# triggers are measurable.
#
# > **Exercise 4 (written ADR).** Write ADR-002 for one decision from sections 55.2–55.5,
# > for example *"GPU quotas: strict per-team reservations vs a shared
# > preemptible pool"*, *"Continuous training: scheduled vs drift-triggered"* or
# > *"Experiment tracking: adopt an existing tool vs build"*. Use the same
# > sections. Include at least one number from this notebook's calculators
# > (with its assumptions), at least two options with the criterion that
# > separates them, and two measurable *revisit when* triggers. You can
# > generate it with `ds.adr_markdown(...)` and save it to `OUT`.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · Online feature-store size.**
#
# $$
# \text{GB} = \frac{E \cdot (k + F \cdot b) \cdot o \cdot r}{10^9}
# $$
#
# *Symbols:* $E$ entities, $k$ key bytes, $F$ features, $b$ bytes per value, $o$ store overhead, $r$
# replicas. *Example (section 55.1):* $2 \times 10^8 \cdot (16 + 400 \cdot 4) \cdot 1.5 \cdot 3 / 10^9 = 1{,}454$ GB.
#
# **2 · Training compute and GPU-hours.**
#
# $$
# \text{FLOPs} \approx 6\, N\, D, \qquad \text{GPU-hours} = \frac{6 N D}{\text{peak FLOP/s} \cdot \mathrm{MFU} \cdot 3600}
# $$
#
# *Symbols:* $N$ parameters, $D$ training tokens (6 = 2 forward + 4 backward FLOPs per parameter per
# token). *Example:* $N = 10^9$, $D = 2 \times 10^9$: $1.2 \times 10^{19}$ FLOPs; at 65 TFLOP/s and MFU 0.3,
# $1.2 \times 10^{19} / (1.95 \times 10^{13} \cdot 3600) = 171$ GPU-hours.
#
# **3 · The price of idle capacity.** $\text{cost per useful GPU-hour} = \text{price} / u$. *Example:* a \$2 card used 20%
# of the time costs \$10 per useful hour.
#
# **4 · Canary sample size** (two proportions, two-sided $\alpha$, power $1 - \beta$).
#
# $$
# n = \frac{\left( z_{1-\alpha/2} \sqrt{2 \bar{p} (1 - \bar{p})} + z_{1-\beta} \sqrt{p_1 (1 - p_1) + p_2 (1 - p_2)} \right)^2}{(p_2 - p_1)^2}, \qquad \bar{p} = \frac{p_1 + p_2}{2}
# $$
#
# *Example:* error rate $p_1 = 0.005 \to p_2 = 0.006$ (+20%), $z = 1.96$ and $0.84$: $n \approx 85{,}900$ per arm,
# i.e. 72 minutes at 20 QPS (1% of 2,000). Doubling the effect to $p_2 = 0.010$ needs only ~4,700.
#
# **5 · Label maturity** (exponential delay with mean $\tau$).
#
# $$
# F(t) = 1 - e^{-t/\tau}
# $$
#
# *Example:* $\tau = 20$ days: $F(7) = 1 - e^{-0.35} = 30\%$ of eventual labels are visible after a week.
#
# **6 · Risk score.** $\text{risk} = \text{likelihood} \times \text{impact}$, each on 1–5 (HIGH ≥ 15, MEDIUM ≥ 8).
# *Example:* likely (4) × major (4) = 16, HIGH.
#
# The cell below recomputes 1, 2, 4 and 5 by hand and compares them with `scaleml.design`.

# %%
# Key equations, recomputed by hand and compared with the calculators (instant).
import math
from statistics import NormalDist

by_hand_gb = 200e6 * (16 + 400 * 4) * 1.5 * 3 / 1e9
print(f"online store: by hand {by_hand_gb:,.0f} GB | ds.online_store_gb {ds.online_store_gb(200e6, 400, 4):,.0f} GB")
gpu_hours = 6 * 1e9 * 2e9 / (65e12 * 0.30 * 3600)
print(f"GPU-hours: by hand {gpu_hours:,.0f} | ds {ds.gpu_hours_from_flops(ds.train_flops(1e9, 2e9), 65e12, 0.30):,.0f}")
p1, p2 = 0.005, 0.006
za, zb = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(0.80)
pbar = (p1 + p2) / 2
n_hand = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
print(f"canary n per arm: by hand {math.ceil(n_hand):,} | ds.two_proportion_n {ds.two_proportion_n(p1, 0.20):,}")
print(f"label maturity after 7 days (mean delay 20): by hand {1 - math.exp(-7 / 20):.3f} | ds {ds.label_maturity(7, 20):.3f}")

# %% [markdown]
# **Expected:** each pair agrees: `1,454` GB, `171` GPU-hours, the same canary size (≈ 85,900; the
# number printed in section 55.3), and `0.295`.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Exercise 1 off by ~7% | used 2³⁰ (GiB) instead of 10⁹ | the exercise asks for decimal GB; state the unit in an interview |
# | Exercise 1 off by 1,000× | returned MB or TB | divide bytes by 10⁹ exactly once |
# | Exercise 3 `wasted_cost` negative | utilisation given as a percentage (72) instead of a fraction (0.72) | validate the range (0, 1] and raise `ValueError` |
# | A cost table disagrees with a colleague's by 10× | different hidden assumptions (replicas, overhead, price) | print the assumptions next to every estimate (`ds.show_assumptions`) |
# | The interviewer keeps asking "who owns this?" | the design has boxes but no contracts | name an owning team, an SLO and an on-call for each platform component |
# | `ModuleNotFoundError: scaleml` | setup cell not run | run the first code cell |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under three minutes each:
#
# 1. Design a feature platform for 50 teams: what is the unit of definition, how do the offline and online stores stay consistent, and how do you guarantee point-in-time correctness for training sets?
# 2. Estimate the online store for 200M users × 400 features (4 bytes each) with 3 replicas. Name three ways to cut it in half and what each costs you.
# 3. Your GPU fleet averages 45% utilisation and the bill doubled this year. What do you measure first, and what policies do you propose? Who has to agree?
# 4. Walk through how a retrained model gets from the training job to 100% of traffic. Which regressions does a 1% canary catch, and which can it not catch? Show the arithmetic.
# 5. Labels arrive 3–6 weeks after predictions. How does your monitoring platform tell a real performance drop from label immaturity?
# 6. When would you buy a managed feature store instead of building one, and what would make you reverse that decision?
#
# **Next chapter:** [Chapter 56 · Large-scale design cases](../ch56-large-scale-design-cases/notebook.ipynb) — the platform's customers: four product designs (ads, video, an LLM assistant, fraud) with numbers, deep dives and failure modes.
