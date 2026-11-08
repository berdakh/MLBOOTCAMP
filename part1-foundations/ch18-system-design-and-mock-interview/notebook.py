# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 18 · System design & mock interview
#
# **Part I · Chapter 18 · about 70 minutes · CPU only**
#
# An ML system design round is not a test of whether you know the newest model.
# It checks whether you can turn a vague product goal into a system that makes
# the right decision, is measured honestly, and keeps working after launch, and
# whether you can say all that in a structured way in 35-45 minutes.
#
# After this chapter you should be able to:
#
# 1. run a design conversation through a fixed sequence of headings, starting from the decision and the metric, not the model;
# 2. do back-of-envelope capacity and latency arithmetic out loud;
# 3. present a worked design at scale (churn for 10M accounts) and a shorter one (moderation triage), with explicit trade-offs;
# 4. run a timed 45-minute mock interview on yourself, score it with a rubric, and track the scores across weeks;
# 5. tell two research stories in STAR form that answer behavioural questions.
#
# **Lab output:** `artifacts/mock_interview.json`, dated rubric scores you append to each time you do a mock.

# %% [markdown]
# **Read first:** [Chapter 18 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: only the feature table and split, to MEASURE scoring speed in 17.3 ---
from churnlab import bootstrap
from grading import ch18 as grade

split = bootstrap.ensure_split()
print(f"train rows: {len(split.train):,}")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **ML system design interview** | a 35-45 minute spoken design of an ML product, end to end | "churn prediction for 10M accounts" |
# | **Offline / online metric** | measured on historical data / measured on live users, usually by experiment | PR-AUC / retained revenue |
# | **Guardrail metric** | something that must not get worse while you optimise the main metric | latency, complaint rate |
# | **QPS** | queries (requests) per second the system must handle | ≈ 520 at peak for an in-app badge |
# | **Peak factor** | how much busier the busiest hour is than the average | ×3 |
# | **Replica** | one copy of the service behind a load balancer | 18 replicas at 50 QPS each, 60% utilised |
# | **Latency budget** | an end-to-end time target split among the steps of a request | 200 ms: feature lookup, model, network |
# | **Batch vs online scoring** | precompute scores on a schedule vs compute per request | daily list vs live badge |
# | **Feature store** | a shared system that serves the same features for training and serving | point-in-time joins offline, low-latency reads online |
# | **Feedback loop** | the model's actions change the data it is later trained on | saved customers stop looking like churners |
# | **STAR** | Situation, Task, Action, Result: a structure for behavioural answers | most of the time on Action |

# %% [markdown]
# ## 18.1 · The framework
#
# Use the same nine headings every time, in this order. Interviewers forgive a
# missing detail; they rarely forgive starting with "I'd use a transformer"
# before anyone knows what decision the model supports.
#
# | # | Heading | Questions to answer (out loud) | Typical time |
# |---|---|---|---|
# | 1 | **Clarify goal & constraints** | Who acts on the prediction, how often, with what capacity? What happens today without a model? Scale, latency, budget, regulation | 3-5 min |
# | 2 | **Success metrics** | *Offline* (PR-AUC, recall@capacity, NDCG@10), *online* (the business outcome, measured by an experiment), *guardrails* (latency, complaint rate, fairness across slices) | 3-4 min |
# | 3 | **Data & labels** | Where do labels come from, how late do they arrive, how noisy are they, are they affected by our own actions? | 4-5 min |
# | 4 | **Features** | Batch vs online features; point-in-time correctness; the same code for training and serving (feature store or shared library) | 4-5 min |
# | 5 | **Model** | Baseline first (a rule, logistic regression), then the step up and what would justify it. Training cadence | 3-5 min |
# | 6 | **Serving** | Batch vs online vs both; latency budget; capacity (QPS, replicas); fallbacks when a dependency is down | 4-5 min |
# | 7 | **Evaluation & experimentation** | Time-based offline evaluation; A/B test design (unit, metric, duration, power); interleaving for ranking | 3-5 min |
# | 8 | **Monitoring & iteration** | Data/score drift, delayed-label performance, alert thresholds, retraining triggers, rollback | 3-4 min |
# | 9 | **Risks & ethics** | Feedback loops, unequal error rates across groups, privacy, abuse, what the model must never do on its own | 2-3 min |
#
# Two habits that score points: **state assumptions with numbers** ("say 10M
# accounts, 2% churn a month") and **name the trade-off** every time you choose
# ("batch is cheaper and simpler; we lose freshness, which matters for X but not Y").
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-design-framework.svg" alt="Nine steps of an ML system design answer: clarify, metrics, data, features, model, serving, evaluation, monitoring, risks" width="100%">
#
# *What to notice:* the model is step 5 of 9; more than half the answer is about the problem, the data and what happens after deployment.
#
# > **Intuition.** The framework is an architect's checklist for a house:
# > who lives there and what they need comes before the choice of bricks.
# > *Where it breaks down:* a real interviewer will interrupt and steer; the
# > order is a default, not a script. Follow them, then come back to the
# > heading you skipped.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/serving.svg" alt="Serving: batch scoring and online prediction paths sharing one model bundle" width="100%">

# %% [markdown]
# ## 18.2 · Back-of-envelope helpers
#
# Capacity arithmetic should take you 30 seconds and be done out loud. These
# helpers exist so you can check your mental arithmetic, not replace it.
#
# > **Intuition.** A day has 86,400 seconds; call it $10^5$. Then "requests
# > per day ÷ $10^5$" is the average QPS in your head, and a peak factor of 2-4
# > covers the busy hour. *Where it breaks down:* traffic can be spikier than
# > any daily average suggests (a push notification sends everyone at once);
# > ask about bursts before sizing for averages.

# %%
import math

SECONDS_PER_DAY = 86_400


def capacity(dau: float, requests_per_user_per_day: float, peak_factor: float = 3.0,
             per_replica_qps: float | None = None, target_utilisation: float = 0.6) -> dict:
    """Average and peak QPS; replicas needed if one replica sustains `per_replica_qps`."""
    daily = dau * requests_per_user_per_day
    avg_qps = daily / SECONDS_PER_DAY
    peak = avg_qps * peak_factor
    out = {"requests_per_day": daily, "avg_qps": avg_qps, "peak_qps": peak}
    if per_replica_qps:
        # Run replicas at ~60% so a spike or a lost replica does not tip you over.
        out["replicas"] = max(1, math.ceil(peak / (per_replica_qps * target_utilisation)))
    return out


def table(rows: list[dict], floatfmt: str = "{:,.1f}") -> None:
    """Print rows as an aligned text table (handy for design notes)."""
    cols = list(rows[0])
    cells = [[floatfmt.format(r[c]) if isinstance(r[c], float) else
              (f"{r[c]:,}" if isinstance(r[c], int) else str(r[c])) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for row, raw in zip(cells, rows):
        print("  ".join(v.ljust(w) if isinstance(raw[c], str) else v.rjust(w)
                        for v, w, c in zip(row, widths, cols)))


table([
    {"scenario": "support console lookups", **capacity(2_000, 60, peak_factor=4, per_replica_qps=50)},
    {"scenario": "in-app risk badge", **capacity(3_000_000, 5, peak_factor=3, per_replica_qps=50)},
    {"scenario": "feed ranking (illustrative)", **capacity(50_000_000, 30, peak_factor=2.5, per_replica_qps=200)},
])

# %% [markdown]
# **Expected:** the support console needs about 1.4 requests/second on
# average and ~5.6 at peak (one replica; you would still run two for
# availability);
# an in-app badge for 3M daily users needs ~520 QPS at peak (18 replicas at 50
# QPS each); feed ranking is in the tens of thousands. The point of doing this
# in an interview: the answer decides the architecture. One QPS can be a
# database lookup of a precomputed score; 50,000 QPS needs caching, batching and
# a feature store designed for it.
#
# The per-replica QPS numbers above are **assumptions**; in a real design you
# measure them. Chapter 14 did, for this project's service.

# %% [markdown]
# > **Exercise 1.** Write `peak_qps(dau, requests_per_user_per_day, peak_factor)`
# > returning peak requests per second as a float. Then, without running
# > anything, say the answer for 1M DAU × 20 requests × peak factor 3 to the
# > nearest 10 (one significant figure is what interviewers expect).

# %%
def peak_qps(dau: float, requests_per_user_per_day: float, peak_factor: float) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(peak_qps)

# %% [markdown]
# A **latency budget** splits an end-to-end target into parts, so everyone knows
# how much the model may take. Work from the user backwards:
#
# > **Exercise 2.** Write `latency_budget(total_ms, components)` where
# > `components` maps a step name to its milliseconds. Return the milliseconds
# > left over. Raise `ValueError` (with a message saying by how much) if the
# > components exceed the budget. Do not modify the input.

# %%
def latency_budget(total_ms: float, components: dict[str, float]) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(latency_budget)

# %% [markdown]
# Here is the budget for the support-console API in design #1 below (p95
# numbers, **assumed** for the design; you would measure each):

# %%
console_budget = {"console -> API network": 30, "auth + request validation": 10,
                  "online feature lookup (key-value store)": 25, "model inference": 10,
                  "explanations (top reasons)": 15, "serialisation + logging": 5}
total = 200
spent = sum(console_budget.values())
table([{"component": k, "p95_ms": v} for k, v in console_budget.items()] +
      [{"component": "HEADROOM", "p95_ms": total - spent}])

# %% [markdown]
# **Expected:** 95 ms of components and 105 ms of headroom in a 200 ms budget.
# Headroom is not waste: p95s do not add up exactly, and the tail of the
# slowest dependency (here the feature lookup) dominates. If you call your
# `latency_budget` with a 90 ms total it should raise.

# %% [markdown]
# ## 18.3 · Worked design #1: churn prediction for 10M accounts
#
# *Prompt: "Design a churn prediction system for a SaaS company with 10 million
# accounts. Customer success wants a daily ranked list, and support agents want
# a churn-risk score in their console when a customer calls."*
#
# This is the TaskFlow project at scale. Walk it through the nine headings.
#
# **1 · Clarify.** Who acts? CS managers call the top of a daily list (say
# capacity 5,000 calls/day); support agents see a score during live calls. What
# exists today? (A rule like "failed payment + usage drop".) Scale: 10M accounts,
# ~2% monthly churn (assumed). Latency: the list is overnight; the console needs
# < 200 ms. Constraints: customer data is personal data (retention, access).
#
# **2 · Metrics.** Offline: PR-AUC on a time split, and **precision/recall at the
# call capacity** (5,000/day), because that is the decision. Online: retention
# rate of contacted vs a randomised holdout of equally-scored accounts (the
# *uplift*, not the score). Guardrails: complaint/unsubscribe rate of contacted
# customers, console latency p95, calls per CS manager.
#
# **3 · Data & labels.** Label = cancelled within 30 days of the snapshot,
# known 30 days late. **Interventions contaminate labels**: a saved customer
# looks like a false positive. So log every contact, keep a random holdout that is
# never contacted, and train on holdout + pre-launch data or model the treatment.
#
# **4 · Features.** The same point-in-time definitions as `features.py`, but
# computed by a scheduled job into a **feature store**: an offline table keyed
# by (snapshot, account) for training, and an online key-value copy of the
# latest values for the console. One definition, two materialisations: that is
# what prevents training/serving skew. Training sets are built with
# **point-in-time joins** ("as of" the snapshot), never with today's values.
#
# **5 · Model.** Baseline: the current rule, then logistic regression (on
# TaskFlow it beat boosted trees on validation, so do not assume trees win).
# Gradient boosting is the natural next step at 10M rows, where there is more
# data to support interactions. Retrain monthly on a rolling window; promote only
# if a paired comparison on the latest validation month says so.
#
# **6 · Serving.** *Both* paths, sharing one model bundle from the **registry**:
#
# * **Batch** (the main product): a nightly job scores all active accounts
#   and writes `(account_id, score, model_version, scored_at, top_reasons)` to a
#   table the CS tool reads. No online latency problem at all.
# * **Online** (the console): the API reads the *precomputed* score first. Only
#   if the agent asks for "refresh with today's activity" does it compute
#   features online and call the model. Capacity is tiny (18.2), so this is about
#   latency and availability, not throughput. Fallback: show yesterday's score
#   with its date.
#
# How long does the nightly batch take? Measure, don't guess: score with this
# course's actual pipeline and extrapolate.

# %%
import time

import numpy as np
import pandas as pd

from churnlab.features import FEATURES, LABEL
from churnlab.modeling import make_model

model = make_model("logreg").fit(split.train[FEATURES], split.train[LABEL])
rows = pd.concat([split.train[FEATURES]] * 4, ignore_index=True)  # ~100k rows, train period only

start = time.perf_counter()
model.predict_proba(rows)
elapsed = time.perf_counter() - start
rows_per_s = len(rows) / elapsed
hours_10m_one_core = 10_000_000 / rows_per_s / 3600
print(f"scored {len(rows):,} rows in {elapsed:.2f} s -> {rows_per_s:,.0f} rows/s on one process")
print(f"10M accounts: ~{hours_10m_one_core * 60:.1f} min of model time on one process "
      "(feature computation and I/O will dominate)")

# %% [markdown]
# **Expected:** on the order of 10⁵-10⁶ rows per second on one CPU process, so
# scoring 10M accounts takes **minutes** of model time (your number depends on
# the machine). The model is not the bottleneck; computing features for 10M
# accounts from billions of usage events is. That is the sentence an interviewer
# wants: *"inference is cheap; the expensive, risky part is the feature
# pipeline, so that is where I would spend the design effort"*.
#
# **7 · Evaluation & experimentation.** Offline: rolling-origin backtests on
# monthly snapshots, cluster bootstrap by account. Online: randomise **accounts**
# (not calls) into treatment (call if flagged) and holdout (never called);
# primary metric 60-day retention among flagged accounts; power the test using
# the baseline churn rate among flagged accounts, and run it for at least two
# label windows.
#
# **8 · Monitoring.** Daily: input drift (PSI per feature), score distribution,
# share of accounts flagged, batch job success/row counts. Monthly, once labels
# arrive: PR-AUC and calibration on the holdout. Online: p95 latency, error rate,
# fallback rate. Alerts route to a person; rollback = point the registry's
# CURRENT back (chapter 15).
#
# **9 · Risks.** Feedback loops (above); the model penalising a region or plan
# because of how the business treated it historically (slice metrics);
# privacy (no raw ticket text in features without review); over-contacting
# (cap calls per account per quarter).
#
# **Trade-offs to say explicitly:**
#
# | Choice | Option A | Option B | What decides |
# |---|---|---|---|
# | Scoring mode | nightly batch only | batch + online refresh | does a same-day signal change the agent's action? |
# | Feature store | build/buy one | shared feature library + two jobs | number of models and teams reusing features |
# | Model | logistic regression | gradient boosting | measured paired gain on validation vs interpretability for CS |
# | Threshold | fixed probability | top-k by capacity | capacity is fixed daily → top-k is simpler and robust to drift |
# | Retraining | scheduled monthly | triggered by drift | label delay makes triggers slow anyway; schedule + alerts |
#
# > **Research bridge.** "Inference is cheap, the pipeline is the system"
# > mirrors a BCI pipeline: the classifier is milliseconds; artefact rejection,
# > filtering and epoching are where real-time systems fail. You already know
# > how to say that; now say it about features.

# %% [markdown]
# ## 18.4 · Worked design #2 (shorter): a moderation triage queue
#
# *Prompt: "Design a system that decides which user comments go to human
# moderators. Reviewers can handle 20,000 comments a day; 5 million are
# posted."* (This is capstone 5.)
#
# * **Clarify.** The model **ranks** for human review; it removes nothing on
#   its own. Capacity: 20k / 5M = 0.4% of comments. Is there a fast lane for
#   severe categories (threats)?
# * **Metrics.** Offline: PR-AUC; **recall at the capacity cut-off** (fraction
#   of truly violating comments that land in the top 20k). Online: violating
#   content seen by users (prevalence), time-to-review. Guardrails: false-positive
#   rate on comments that merely *mention* identity groups, reviewer agreement.
# * **Data & labels.** Reviewer decisions are labels, but only for comments
#   the model sent to review (**selection bias**). Fix: send a small random sample
#   of unflagged comments to review too, for unbiased evaluation.
# * **Features & model.** Text model (start with TF-IDF + logistic regression,
#   then a fine-tuned small transformer); account features (age, prior
#   violations) online.
# * **Serving.** Near-real-time: score on post (5M/day ≈ 58 QPS average, ~170 at
#   a peak factor of 3), put into a priority queue ordered by score × severity;
#   latency budget is seconds, not milliseconds, because nobody waits on it.
# * **Evaluation.** Time-split offline; online A/B on queue ordering by
#   **interleaving** reviewer queues or by randomising days.
# * **Monitoring.** Score drift (new slang, coordinated attacks), queue length,
#   reviewer overturn rate per slice.
# * **Risks.** Disparate error rates on identity mentions; adversarial
#   obfuscation; reviewer wellbeing (exposure limits).

# %%
table([
    {"quantity": "comments per day", "value": 5_000_000},
    {"quantity": "review capacity per day", "value": 20_000},
    {"quantity": "share reviewed (%)", "value": 100 * 20_000 / 5_000_000},
    {"quantity": "avg scoring QPS", "value": capacity(5_000_000, 1)["avg_qps"]},
    {"quantity": "peak scoring QPS (x3)", "value": capacity(5_000_000, 1, peak_factor=3)["peak_qps"]},
], floatfmt="{:,.2f}")

# %% [markdown]
# **Expected:** 0.40% reviewed, ~57.87 QPS average, ~173.61 QPS at peak. Small
# enough for a handful of CPU replicas with a small text model, which is
# worth saying before anyone suggests a GPU cluster.

# %% [markdown]
# ## 18.5 · Practice prompts
#
# Do at least one fully (Exercise 3). For each, what a strong answer covers:
#
# | # | Prompt | A strong answer covers |
# |---|---|---|
# | 1 | **Fraud detection for card payments** (decision in < 100 ms) | online features (velocity counters), latency budget, extreme imbalance, delayed and noisy chargeback labels, cost-based threshold, rules + model, adversarial drift |
# | 2 | **Recommendations "for you" on a shopping site** | candidate generation + ranking, Recall@K/NDCG offline, CTR/conversion online with interleaving, cold start, popularity bias, feedback loops |
# | 3 | **Demand forecasting for 5,000 stores × 20,000 products** | hierarchical/global models, backtesting by time, intermittent demand, sales ≠ demand (stock-outs), batch cadence, business metric (waste, availability) |
# | 4 | **Predictive maintenance for a fleet of turbines** | entity-level splits, rare failures, RUL vs classification framing, sensor drift, cost of a missed failure vs an unnecessary inspection |
# | 5 | **Short-term electricity load forecasting for a grid operator** | seasonal baselines, weather forecasts as features (and their errors), peak-hour error, probabilistic forecasts, holidays |
# | 6 | **Search ranking for a help-centre** | query understanding, relevance labels (clicks are biased), NDCG, interleaving, latency, fallback to keyword search |
#
# Prompts 3-5 match capstones 3, 2 and 1: a design you have built is a design
# you can defend in depth.

# %% [markdown]
# > **Exercise 3 (written).** Pick one practice prompt and write a full design
# > under the nine headings of 18.1, including at least one capacity or latency
# > number and one explicit trade-off table. Give yourself 35 minutes, as in an
# > interview. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 18.6 · The mock interview (45 minutes)
#
# Do this with a friend reading the interviewer's part, or alone with a timer
# and a recording. Do not pause the clock.
#
# | Segment | Minutes | What happens |
# |---|---|---|
# | Intro | 0-5 | "Tell me about yourself" (≤ 90 s) + why this role |
# | Project deep-dive | 5-20 | Walk through TaskFlow (or your capstone); interviewer drills into 2-3 decisions |
# | ML system design | 20-35 | One prompt from 18.5 (drawn at random below) |
# | Coding / SQL | 35-45 | One drill from chapter 17 you have *not* just practised |
# | Your questions | last 2-5 | Two questions you ask them |

# %%
import random

QUESTION_BANK = {
    "intro": [
        "Walk me through your background in 90 seconds.",
        "Why move from research to an ML engineering / applied scientist role?",
        "What is a project you are proud of, and what was your part in it?",
    ],
    "project": [
        "What decision does your model support, and who acts on it?",
        "How did you split the data, and what would have gone wrong with a random split?",
        "Why PR-AUC? What was the positive rate?",
        "How did you choose the threshold? What happens if the costs change?",
        "Your test score is higher than validation. Why?",
        "Logistic regression beat gradient boosting. Convince me that is real.",
        "How would you know in production that the model has degraded, given labels arrive 30 days late?",
        "Walk me through a rollback.",
        "What would you do differently with real data?",
    ],
    "design": [
        "Fraud detection for card payments", "Recommendations 'for you' on a shopping site",
        "Demand forecasting for 5,000 stores x 20,000 products", "Predictive maintenance for a turbine fleet",
        "Short-term electricity load forecasting", "Search ranking for a help-centre",
    ],
    "coding": [
        "Sliding window maximum in O(n)", "Merge overlapping subscription periods",
        "Cohort retention in SQL", "Longest inactive streak in SQL (gaps and islands)",
        "Implement precision@k and explain tie handling", "Stratified split by hand",
    ],
}

rng = random.Random()  # deliberately unseeded: a different mock each time
draw = {segment: rng.sample(questions, k=min(3, len(questions))) if segment == "project"
        else rng.choice(questions) for segment, questions in QUESTION_BANK.items()}
for segment, q in draw.items():
    print(f"{segment:>8}: {q}")

# %% [markdown]
# **Expected:** one intro question, three project questions, one design prompt
# and one coding drill, different each time you run the cell (it is the one
# deliberately unseeded cell in the course).
#
# ### Scoring rubric
#
# Score each dimension 1-4 immediately after the mock, from the recording if
# you have one. Score what you **did**, not what you know.
#
# | Dimension | 1 | 2 | 3 | 4 |
# |---|---|---|---|---|
# | **Communication** | rambling, no structure; interviewer had to pull answers out | structured in places; long pauses or jargon | signposted structure, checked in with the interviewer, concise | crisp, adapts depth to the listener, summarises before moving on |
# | **Technical depth** | vague or incorrect on core ML ideas | correct but shallow; struggles with follow-ups | correct and precise; handles two levels of "why?" | goes deep where it matters and knows where the limits of their knowledge are |
# | **Trade-offs** | one option presented as obvious | mentions alternatives without criteria | compares options with explicit criteria and picks one | quantifies trade-offs and says what evidence would change the choice |
# | **Evaluation rigor** | single number, no baseline or split named | baseline or split, not both; no uncertainty | baseline, leakage-safe split, interval, metric matched to the decision | pre-registration, paired comparisons, online experiment design, label bias |
# | **Operations awareness** | no mention of serving or monitoring | mentions monitoring generically | latency/capacity numbers, drift with delayed labels, rollback | failure modes and fallbacks, alert design, cost, on-call reality |
# | **Ownership** | "we" everywhere, unclear personal role | own role stated but passive | clear decisions they made and why, including mistakes | owns outcomes end to end, including what they would change and follow-up actions |

# %%
import json
from datetime import date

RUBRIC = ["communication", "technical_depth", "trade_offs", "evaluation_rigor",
          "operations_awareness", "ownership"]

# --- Fill these in after your mock (1-4), then run the cell. Leave as None to skip saving. ---
my_scores = {dim: None for dim in RUBRIC}
weakest_moments = ["", ""]  # the two moments you would redo, in one line each
segment_minutes = {"intro": None, "project": None, "design": None, "coding": None}

log_path = ART / "mock_interview.json"
history = json.loads(log_path.read_text()) if log_path.exists() else []

if all(v is None for v in my_scores.values()):
    print("no scores entered yet: do the mock, fill in my_scores (1-4) and re-run this cell")
else:
    bad = {k: v for k, v in my_scores.items() if v is not None and v not in (1, 2, 3, 4)}
    assert not bad, f"scores must be integers 1-4, got {bad}"
    history.append({"date": date.today().isoformat(), "scores": my_scores,
                    "mean": round(float(np.mean([v for v in my_scores.values() if v is not None])), 2),
                    "weakest_moments": weakest_moments, "segment_minutes": segment_minutes,
                    "questions_drawn": draw})
    log_path.write_text(json.dumps(history, indent=2))
    print(f"saved mock #{len(history)} to {log_path}")
print(f"{len(history)} mock(s) on record in {log_path.name}")

# %% [markdown]
# **Expected:** on a fresh run, `no scores entered yet ...` and `0 mock(s) on
# record`. After a mock, each run appends one dated entry; nothing is
# overwritten.
#
# Track progress across weeks. Below is what the table looks like with your
# history (or with a clearly labelled **example** history if you have none yet,
# so you can see the format; the example is not saved):

# %%
show = history or [
    {"date": "EXAMPLE week 1", "scores": dict(zip(RUBRIC, [2, 3, 2, 3, 1, 2]))},
    {"date": "EXAMPLE week 3", "scores": dict(zip(RUBRIC, [3, 3, 3, 3, 2, 3]))},
]
progress = pd.DataFrame([{"date": h["date"], **h["scores"]} for h in show]).set_index("date")
progress["mean"] = progress.mean(axis=1).round(2)
progress

# %% [markdown]
# **Expected:** one row per mock (here two EXAMPLE rows if you have not saved
# any). Aim for 3 in every column before interviewing; use the lowest column to
# choose next week's practice (the readiness rubric maps each to resources).

# %% [markdown]
# ## 18.7 · Behavioural questions: STAR with research stories
#
# Behavioural rounds are scored as seriously as technical ones, and researchers
# often under-sell themselves by describing the field instead of their
# decisions. Use **STAR**, with most of the time on **A**:
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-star.svg" alt="STAR answer structure as a bar: Situation 10%, Task 10%, Action 60%, Result 20%" width="100%">
#
# *What to notice:* the context (S and T) together get a fifth of the time; a research story that spends two minutes on the field before reaching "I decided" has run out of time.
#
# | | What | Share of the answer |
# |---|---|---|
# | **S**ituation | the context in one or two sentences | 10% |
# | **T**ask | what *you* were responsible for | 10% |
# | **A**ction | what *you* did, the decisions and why, alternatives rejected | 60% |
# | **R**esult | the outcome with a number if one exists, and what you learned / do differently now | 20% |
#
# Three prompts that map well to research experience:
#
# 1. **"Tell me about a time you found a flaw in your own evaluation."** E.g.
#    discovering cross-subject leakage: trials from the same subject in train and
#    test, or normalisation statistics computed on all subjects. Action: how you
#    found it, how you measured its effect, what you changed. Result: the
#    corrected number (lower) and the check you now always run.
# 2. **"Tell me about a time a result did not replicate."** A published or
#    earlier result that failed on new data or a new seed; how you separated noise,
#    a bug and a real difference; what you reported.
# 3. **"Tell me about a time you had to simplify a model for a user."** A
#    clinician, collaborator or student who needed something they could trust and
#    run; what you gave up (accuracy, flexibility) and how you decided it was worth it.
#
# Rules: real stories only, your own role explicit ("I", not "we"), a number
# where one honestly exists, and never inflate an outcome. An interviewer will
# ask a follow-up that exposes it.

# %% [markdown]
# > **Exercise 4 (written).** Write two STAR stories from your own experience
# > for two of the three prompts above. Keep each to about 250 words (two
# > minutes spoken). Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **Capacity.**
#
# $$
# \text{QPS}_{\text{avg}} = \frac{\text{DAU} \times r}{86{,}400},
# \qquad
# \text{QPS}_{\text{peak}} = f \cdot \text{QPS}_{\text{avg}},
# \qquad
# \text{replicas} = \left\lceil \frac{\text{QPS}_{\text{peak}}}{u \cdot \text{QPS}_{\text{replica}}} \right\rceil
# $$
#
# *Symbols:* DAU = daily active users, $r$ = requests per user per day, $f$ =
# peak factor, $u$ = target utilisation (0.6), $\text{QPS}_{\text{replica}}$ =
# measured throughput of one replica. *Example:* 3M DAU × 5 requests:
# $1.5 \times 10^7 / 86{,}400 = 174$ QPS average, $\times 3 = 521$ at peak,
# $\lceil 521 / (0.6 \times 50) \rceil = 18$ replicas.
#
# **Little's law** for concurrency: $L = \lambda W$. *Example:* 521 QPS at
# 100 ms each keeps about 52 requests in flight, so the service must allow at
# least that many concurrent requests across replicas.
#
# **Latency budget:**
#
# $$
# \text{headroom} = T_{\text{total}} - \sum_{s} t_s
# $$
#
# *Example:* 200 ms total, 95 ms of components → 105 ms of headroom. Budget with
# p95s, and remember p95s do not add exactly: the slowest dependency's tail
# dominates.
#
# **Batch-scoring time:** $t = N / \text{rows per second}$. *Example:* 10M
# accounts at 50,000 rows/s is 200 s: a nightly batch job, not an online service.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `AssertionError: scores must be integers 1-4` | a score like `3.5` or `"3"` | use whole numbers 1-4; half points hide indecision |
# | `mock_interview.json` disappeared | Colab runtime recycled | set `USE_DRIVE = True` in the setup cell before your next mock |
# | Your batch-scoring estimate looks wildly slow | first call includes warm-up, or the machine is busy | run the timing cell twice; quote the order of magnitude, not the decimal |
# | You ran out of time in the design segment at heading 5 | spent too long clarifying or on data | timebox: say "I'll assume X and come back to it" and move on |
# | The question draw is the same as last time | random chance with small banks | add your own questions to `QUESTION_BANK`; the interview-questions guide has more |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Name the nine headings of the design framework in order, and say why metrics come before the model.
# 2. 4M daily users, 10 requests each, peak factor 3: what peak QPS, and roughly how many replicas at 100 QPS each with 60% target utilisation?
# 3. In the 10M-account churn design, why serve a precomputed score in the console instead of computing it online every time?
# 4. How do interventions (calls, moderation) corrupt your future labels, and what do you build to keep evaluation honest?
# 5. What is interleaving, and why is it more sensitive than an A/B test for comparing two rankers?
# 6. Tell your "flaw in my own evaluation" story in STAR form in 90 seconds.
#
# **Next chapter:** the capstones. Choose your first one from the
# [capstones page](https://github.com/berdakh/MLBOOTCAMP/blob/main/part4-capstones/README.md),
# and score yourself honestly with the
# [readiness rubric](https://github.com/berdakh/MLBOOTCAMP/blob/main/appendices/readiness-rubric.md)
# before you start; repeat the rubric and a mock every week.
