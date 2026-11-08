# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 56 · Large-scale design cases
#
# **Part III · Chapter 56 · about 90 minutes · CPU is enough (no GPU work in this lab)**
#
# Four prompts cover most of what large consumer companies ask in ML system
# design interviews at senior and staff level: ads ranking, video
# recommendations, an LLM-powered assistant and real-time fraud. Each has a
# signature difficulty (calibration and auctions; long-term objectives;
# evaluation of generated text; adversaries and label delay) that a generic
# "retrieve, rank, serve" answer misses.
#
# By the end of this lab you should be able to:
#
# 1. run each case through the same twelve-part structure, with numbers stated as assumptions;
# 2. explain *with a small simulation* why ad predictions must be calibrated, and why budgets must be paced;
# 3. show why a raw watch-time objective favours long videos, and what to do about it;
# 4. cost an LLM assistant per conversation, including the history that is re-sent every turn, against a human fallback;
# 5. split a 100 ms fraud-scoring budget across stages and explain why fan-out to many feature services hurts the tail;
# 6. size a serving fleet with headroom, and say what a staff-level answer adds to a senior one in each case.
#
# **You will produce:** `artifacts/level3/nb18/case_numbers.json` (every
# assumption and derived number of the four cases, reusable in your written
# design) and your own 45-minute design (Exercise 3).
#
# **What is measured here.** The simulations are small synthetic models
# built to show a *mechanism*; their numbers are not estimates of any real
# system. Traffic, prices and latencies are printed as `ASSUMPTIONS`.

# %% [markdown]
# Read first: [Chapter 56 text](README.md), which works through the four design cases and their signature difficulties in more detail than this lab.

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

# Level 3 is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing to rebuild; calculators come from scaleml.design -----
import json

import numpy as np

from scaleml import design as ds
from grading import ch56 as grade

OUT = ART / "level3" / "nb18"
OUT.mkdir(parents=True, exist_ok=True)
NUMBERS: dict = {}  # every case's assumptions and derived numbers, saved at the end

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | DAU | Daily active users | 300M for the ads case |
# | Peak QPS | Requests per second at the busiest time: average × peak factor | ~138,889 ad requests/s |
# | Headroom | Spare capacity kept for spikes, a lost replica or a deploy | 30% in the ads fleet |
# | Replica | One copy of a serving process; the fleet is many of them | 3,969 ranker replicas |
# | Latency budget | The end-to-end time limit split into per-stage allowances | 100 ms for fraud scoring |
# | Fan-out | One request calling many services in parallel and waiting for all | 6 feature services |
# | Tail latency | The slowest requests (p99), often caused by one slow dependency | 5.9% wait on a slow call |
# | Budget pacing | Spreading an advertiser's daily budget over the day | participation rate 0.40 |
# | Selective labels | Only the cases the current system let through get labels | declined payments never get chargebacks |
# | Deflection | Share of conversations resolved without a human | 30–70% in the assistant case |
# | Fail-open / fail-closed | When a dependency fails: allow by default / block by default | chosen per fraud risk tier |

# %% [markdown]
# ## 56.0 · One structure for every case
#
# This extends the ten-step framework of Chapter 37. Each case below is
# a table with these twelve rows; in an interview each row takes one to four
# minutes, with deeper discussion of the two the interviewer picks.
#
# | Part | The question it answers |
# |---|---|
# | Requirements | Who, what decision, scale, latency, constraints (policy, money, regulation) |
# | Metrics | **Online** (the business outcome, from experiments) · **offline** (what can be computed on logs) · **guardrails** (what must not get worse) |
# | Data & labels | Where labels come from, how late, how biased by the current system |
# | Candidate generation | How the space is narrowed (or "not applicable", said explicitly) |
# | Ranking / model | Model family and objective; what is predicted and how it is combined |
# | Serving & latency | QPS, stages, per-stage ms, fallbacks |
# | Training cadence | Batch, incremental, online; why |
# | Experimentation | Unit, metrics, duration, interference |
# | Monitoring | What pages, what tickets |
# | Failure modes | What breaks, and what the system does then |
# | Senior answer | Generic signals of a strong senior answer |
# | Staff adds | What a staff-level answer adds on top |
#
# The levelling rows describe *signals interviewers commonly look for*, not
# any company's ladder.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-design-cases.svg" alt="Four columns: ads (targeting, retrieval, pCTR times pCVR model, auction and pacing), video (sources, pre-ranker, multi-task ranker, blending), LLM assistant (input guardrail, retrieval, LLM with citations, check or hand off), fraud (velocity features, rules, model, decision), each with its signature difficulty" width="100%">
#
# *What to notice:* the four pipelines have the same shape (narrow, score, decide); what separates a strong answer is the yellow box, the case's signature difficulty, so that is where the deep dives go.
#
# ## 56.1 · Two calculators used in every case
#
# Chapter 18 asked for a latency budget that returns the
# leftover milliseconds. At this level the budget is a *design artefact*:
# it must validate its inputs, show each stage's share and point at the
# stage to attack first.
#
# > **Exercise 1.** Write `latency_budget_split(total_ms, stages)` where
# > `stages` maps stage name → milliseconds. Return a dict with
# > `remaining_ms` (total minus the sum), `shares` (stage → its ms as a
# > fraction of `total_ms`) and `largest_stage` (the name of the most
# > expensive stage, `None` if there are no stages). Raise `ValueError` if
# > `total_ms <= 0`, if any stage is negative, or if the stages exceed the
# > total (say by how many ms in the message). Do not modify `stages`.

# %%
def latency_budget_split(total_ms, stages):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(latency_budget_split)

# %% [markdown]
# > **Exercise 2.** Write `serving_fleet_size(peak_qps, per_replica_qps, headroom)`:
# > the number of replicas (an `int`) so that at peak each replica runs at
# > most `1 - headroom` of its measured capacity `per_replica_qps`
# > (`headroom=0.3` keeps 30% spare for spikes, a lost replica, a deploy).
# > Never return fewer than **2** (no single point of failure). Raise
# > `ValueError` if `per_replica_qps <= 0`, `peak_qps < 0` or `headroom` is
# > outside `[0, 1)`.

# %%
def serving_fleet_size(peak_qps, per_replica_qps, headroom):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(serving_fleet_size)

# %% [markdown]
# The cells below use the module's equivalents (`ds.latency_table`,
# `ds.replicas_needed`), so they run whether or not you have done the
# exercises. `ds.replicas_needed` also supports spreading over zones so that
# losing a whole zone leaves enough capacity.

# %% [markdown]
# ## 56.2 · Case A: ads ranking and the auction
#
# *Prompt: "Design the system that chooses which ads to show in a feed with
# 300M daily users."*
#
# | Part | Answer |
# |---|---|
# | Requirements | Fill ~3 ad slots per feed request; ad server p99 ≤ 100 ms inside the feed's budget; advertisers set bids and daily budgets; policy review before serving |
# | Metrics | **Online:** revenue per mille (RPM), advertiser value (conversions per spend, retention of advertisers), user engagement with the feed. **Offline:** normalised entropy / log loss, **calibration** (predicted/observed by bucket and by segment), AUC. **Guardrails:** ad load, hide/report rate, latency, feed engagement |
# | Data & labels | Impressions with clicks (seconds later) and conversions (hours to **days** later); labels exist only for ads the current system showed (selection bias); heavy negative down-sampling for training |
# | Candidate generation | Targeting filters (eligibility, budget left, policy) → retrieval of ~1,000 ads (two-tower over user and ad embeddings, plus rules) |
# | Ranking | Multi-task model (pCTR, pConversion) with sparse ID embeddings and cross features (Chapter 51); score = **bid × pCTR (× pCVR)** = expected value per impression (eCPM); auction (second-price-like) sets prices; then quality/ad-load rules |
# | Serving & latency | ~139k peak ad requests/s (below), ~200 ads scored per request; precomputed ad embeddings; feature cache; timeouts with "no ad" as the fallback (never a broken feed) |
# | Training cadence | Frequent incremental updates (hours) because ads and campaigns churn; daily full retrains; **recalibrate after down-sampling** |
# | Experimentation | User-level A/B for ranking changes, but advertisers' budgets are shared across arms (budget cannibalisation): use budget-split or advertiser-side experiments for auction changes |
# | Monitoring | Calibration by segment hourly, spend pacing vs plan, revenue per request, prediction/label delay, feature freshness |
# | Failure modes | Miscalibration shifts money between advertisers (below); a budget spent by noon; delayed conversions under-train new campaigns; click spam |
# | Senior answer | Correct multi-stage pipeline, calibration and log loss, delayed feedback, A/B with guardrails, numbers |
# | Staff adds | Marketplace view: advertiser ROI and retention as long-term metrics, auction/pricing and ranking teams' contracts, budget-split experimentation, how a calibration incident is detected and who owns the fix, privacy constraints on conversion data |
#
# **Numbers.**

# %%
ads = {"DAU": 300_000_000, "feed requests per user per day": 20, "ad slots per request": 3,
       "peak / average": 2.0, "ads retrieved per request": 1_000, "ads ranked per request": 200,
       "ranker QPS per replica (measured on your hardware; ASSUMED here)": 50,
       "headroom": 0.3}
ds.show_assumptions(ads)
ad_peak = ds.peak_qps(ads["DAU"], ads["feed requests per user per day"], ads["peak / average"])
scored = ad_peak * ads["ads ranked per request"]
fleet = ds.replicas_needed(ad_peak, ads["ranker QPS per replica (measured on your hardware; ASSUMED here)"],
                           target_utilisation=1 - ads["headroom"])
print(f"\npeak ad requests: {ad_peak:,.0f}/s -> ads scored: {scored / 1e6:,.1f}M/s -> ranker replicas: {fleet:,}")
NUMBERS["ads"] = {"assumptions": ads, "peak_ad_requests_per_s": ad_peak, "ads_scored_per_s": scored,
                  "ranker_replicas": fleet}

# %% [markdown]
# **Expected:** about **138,889** ad requests/s at peak, **27.8M** ad scores
# per second and **3,969** ranker replicas at 50 requests/s each with 30%
# headroom. The replica count is the interesting number: it says the ranker
# must be batched and probably distilled or cascaded (a light pre-ranker on
# 1,000, the heavy model on 200), which is where Chapters 49–50 apply.
#
# > **Intuition.** Fleet sizing is booking restaurant tables for the dinner rush, not the daily
# > average, and leaving a few tables empty for the party that walks in unannounced (headroom). The
# > "at least 2" rule is the second kitchen in case the first catches fire. *Where the analogy breaks
# > down:* per-replica capacity must be *measured* at the target latency; a replica that does 50
# > req/s at p99 = 300 ms may do only 30 at p99 = 100 ms.
#
# **Deep dive: why calibration matters.** In ranking-only systems a model
# that is wrong by a constant factor still orders items correctly. In an
# auction the prediction is *multiplied by a bid*, so a model that
# over-predicts one segment's CTR by 50% hands that segment auctions it
# should lose. A small simulation (synthetic CTRs and bids, generalised
# second-price pricing):

# %%
rng = np.random.default_rng(0)
N_AUCTIONS, K = 50_000, 10
bid = rng.lognormal(0.0, 0.5, (N_AUCTIONS, K))          # advertiser's value per click
true_ctr = rng.beta(2, 60, (N_AUCTIONS, K))              # mean ~3%
segment_b = rng.random((N_AUCTIONS, K)) < 0.5            # half the ads belong to "segment B"
noise = rng.lognormal(-0.02, 0.2, (N_AUCTIONS, K))       # an imperfect but unbiased model


def run_auctions(pred_ctr):
    ecpm = bid * pred_ctr
    order = np.argsort(-ecpm, axis=1)
    rows, win, second = np.arange(N_AUCTIONS), order[:, 0], order[:, 1]
    price_per_click = ecpm[rows, second] / pred_ctr[rows, win]  # GSP: just enough to beat the runner-up
    p = true_ctr[rows, win]
    return {"platform revenue": float((p * price_per_click).sum()),
            "advertiser value (welfare)": float((p * bid[rows, win]).sum()),
            "segment B win share": float(segment_b[rows, win].mean())}


results = {"oracle (true CTR)": run_auctions(true_ctr),
           "calibrated model": run_auctions(true_ctr * noise),
           "segment B over-predicted x1.5": run_auctions(true_ctr * noise * np.where(segment_b, 1.5, 1.0))}
ds.table([{"model": k, **v} for k, v in results.items()])
NUMBERS["ads"]["calibration_sim"] = results

# %% [markdown]
# **Expected (deterministic, seed 0):** the calibrated model wins segment B
# **50%** of auctions, as it should; over-predicting segment B by 1.5× raises
# its win share to **~69%**, and *both* platform revenue (3,172 → 3,057) and
# advertiser value (5,019 → 4,893) fall, because auctions go to ads that are
# not actually better. Ranking metrics like AUC barely notice a per-segment
# scale error; calibration plots by segment do. This is why ads teams track
# predicted/observed ratios per segment and recalibrate after negative
# down-sampling (if a fraction *w* of negatives is kept, a calibrated
# model's odds must be multiplied by *w*).
#
# **Deep dive: budget pacing.** An advertiser with a daily budget that would
# be spent in a few hours at full participation should not vanish from the
# afternoon. The simplest pacer throttles participation so that spend tracks
# the remaining traffic:

# %%
hours = np.arange(24)
traffic = 1 + 0.8 * np.sin((hours - 9) / 24 * 2 * np.pi)   # ASSUMED diurnal shape, peak mid-afternoon
traffic = traffic / traffic.sum()
full_day_spend, daily_budget = 1_000.0, 400.0              # unthrottled spend would be 2.5x the budget

spend_greedy, left = [], daily_budget
for h in hours:
    s = min(left, full_day_spend * traffic[h])
    spend_greedy.append(s)
    left -= s
spend_paced, left, rates = [], daily_budget, []
for h in hours:
    rate = min(1.0, left / (full_day_spend * traffic[h:].sum()))  # participate with this probability
    s = full_day_spend * traffic[h] * rate
    spend_paced.append(s)
    rates.append(rate)
    left -= s
exhausted = int(np.argmax(np.cumsum(spend_greedy) >= daily_budget - 1e-9))
print(f"greedy: budget exhausted during hour {exhausted}:00; hours with no ads: {24 - exhausted - 1}")
print(f"paced:  participation rate {rates[0]:.2f} all day; spend in last 6 hours {sum(spend_paced[-6:]):.0f} "
      f"(greedy: {sum(spend_greedy[-6:]):.0f})")
NUMBERS["ads"]["pacing"] = {"greedy_exhausted_hour": exhausted, "paced_rate": rates[0]}

# %% [markdown]
# **Expected:** greedy spending exhausts the budget during hour **13**, leaving
# 10 hours with no ads (the evening audience never sees the campaign); the
# pacer participates at rate **0.40** all day and spends ~109 in the last six
# hours instead of 0. Real pacers are feedback controllers (they correct for
# forecast errors every few minutes) and interact with bidding; in an
# interview, pacing should be named together with the reason it exists.
#
# **Deep dive: delayed feedback.** If conversions arrive with a delay of days,
# yesterday's impressions look like negatives that will later flip. Options
# to name: wait (a fixed attribution window; stale model), train on early
# labels and correct with a delay model (importance weights by the
# probability the label has arrived), or fake-negative re-labelling when the
# conversion comes in. Chapter 51 measured this on a synthetic Criteo-like
# stream; `ds.label_maturity` gives the fraction observed for an exponential delay.
#
# ## 56.3 · Case B: video recommendations
#
# *Prompt: "Design the 'up next' and home recommendations for a video platform
# with 200M daily users and 1B videos."*
#
# | Part | Answer |
# |---|---|
# | Requirements | Home feed + "up next"; p99 ≤ 200 ms; uploads become recommendable within an hour; creator ecosystem health matters |
# | Metrics | **Online:** satisfied watch time per user, return visits (days active per week), surveys. **Offline:** recall@K per source, NDCG on held-out sessions, calibration of watch-time predictions. **Guardrails:** creator diversity, new-creator exposure share, policy violations served, latency |
# | Data & labels | Impressions, clicks, watch time, completion, likes/dislikes, "not interested", surveys; watch time biased toward long videos and toward what the current system shows |
# | Candidate generation | Several sources merged to ~5,000: two-tower (user history → video), co-watch item-to-item, subscriptions, fresh uploads, trending, exploration slots |
# | Ranking | Pre-ranker → multi-task heavy ranker (p(click), E[watch time], p(like), p(dislike), survey score) → blended value function → re-ranking for diversity and policy |
# | Serving & latency | Embedding lookups, ANN per source in parallel; per-stage budget; precomputed candidates for logged-out/cold paths |
# | Training cadence | Streaming/incremental updates for freshness (minutes–hours) plus periodic full retrains; new video embeddings from content until interactions accumulate |
# | Experimentation | User-level A/B, long-running holdouts for long-term effects; creator-side metrics need their own analysis (two-sided marketplace) |
# | Monitoring | Source contribution and recall, fresh-upload exposure, watch-time per request, distribution of watch time by video length, rate of "not interested" |
# | Failure modes | Clickbait (click without watch), length bias, feedback loops (popular gets more popular), cold start for new creators, harmful content amplified |
# | Senior answer | Multi-source retrieval with numbers, multi-task ranker, watch-time objective with its bias, exploration, A/B |
# | Staff adds | Long-term value vs short-term watch time (holdouts, surveys), creator-side health as a first-class objective, who owns the value function's weights and how they are changed, cost per recommendation |
#
# A published design that can be cited: *"Deep Neural Networks for YouTube
# Recommendations"* (Covington et al., RecSys 2016) describes a two-stage
# candidate-generation + ranking system and trains the ranker with
# **weighted logistic regression**, weighting positive impressions by watch
# time so the learned odds approximate expected watch time. It describes a
# design published in 2016, not current practice anywhere.
#
# **Numbers.**

# %%
vid = {"DAU": 200_000_000, "requests per user per day": 25, "peak / average": 2.5,
       "videos": 1_000_000_000, "embedding dim": 64, "bytes per value (fp16)": 2,
       "candidates per request": 5_000, "ranked by heavy model": 500}
ds.show_assumptions(vid)
vid_peak = ds.peak_qps(vid["DAU"], vid["requests per user per day"], vid["peak / average"])
emb_gb = vid["videos"] * vid["embedding dim"] * vid["bytes per value (fp16)"] / ds.GB
print(f"\npeak requests: {vid_peak:,.0f}/s | video embeddings: {emb_gb:,.0f} GB fp16 | "
      f"heavy-ranker scores: {vid_peak * vid['ranked by heavy model'] / 1e6:,.1f}M/s")
NUMBERS["video"] = {"assumptions": vid, "peak_qps": vid_peak, "embedding_gb": emb_gb}

# %% [markdown]
# **Expected:** ~**144,676** requests/s at peak, **128 GB** of fp16 video
# embeddings (sharded, or compressed with product quantisation as in
# Chapter 37), and ~72M heavy-ranker scores per second. Most of the 1B
# videos have almost no views: retrieval indexes usually cover the eligible,
# recently active subset, and the tail is reached through content features
# and exploration.
#
# **Deep dive: the watch-time objective.** Ranking by expected watch time
# ranks by video *length* as much as by quality. A synthetic catalogue in
# which click probability does not depend on length, and completion falls
# with length:

# %%
rng = np.random.default_rng(1)
n_items = 20_000
length_min = rng.lognormal(np.log(6), 0.9, n_items).clip(0.5, 120)   # minutes
quality = rng.normal(0, 1, n_items)
p_click = 1 / (1 + np.exp(-(quality - 3)))                             # independent of length
completion = 1 / (1 + np.exp(-(quality + 1 - 0.6 * np.log(length_min))))
exp_watch = p_click * completion * length_min                            # minutes per impression


def top(score, k=100):
    idx = np.argsort(-score)[:k]
    return {"median length (min)": float(np.median(length_min[idx])), "mean quality": float(quality[idx].mean())}


objectives = {"p(click)": p_click, "E[watch minutes]": exp_watch,
              "E[completion]": p_click * completion,
              "E[watch] / length^0.5": exp_watch / np.sqrt(length_min)}
ds.table([{"objective": k, **top(v)} for k, v in objectives.items()])
print(f"catalogue median length: {np.median(length_min):.1f} min")
NUMBERS["video"]["objective_sim"] = {k: top(v) for k, v in objectives.items()}

# %% [markdown]
# **Expected (seed 1):** the top 100 by expected watch minutes has a median
# length of ~**27** minutes against 6 for the catalogue and ~6–7 for the click
# or completion objectives; its mean quality is the *lowest* of the four
# blends. No single objective is right: completion favours short clips,
# watch minutes favours long ones. Real systems blend several heads with
# weights chosen by experiment and check the length distribution of what
# they serve. **Exploration** (a few percent of slots for new or uncertain
# videos, Chapter 52's bandits) and **new-creator exposure** guardrails stop
# the loop from only feeding what already won.
#
# ## 56.4 · Case C: an LLM-powered customer-support assistant
#
# *Prompt: "Design an assistant that answers customer-support chats for a
# company with 1M support conversations a day, handing off to humans when needed."*
#
# | Part | Answer |
# |---|---|
# | Requirements | Answer from the company's help content and the customer's account data; time to first token ≤ 1.5 s; never take irreversible account actions without confirmation; hand off to a human with a summary |
# | Metrics | **Online:** resolution without human (verified by no re-contact within 7 days), customer satisfaction, handle time saved. **Offline:** answer correctness and groundedness on a labelled evaluation set, retrieval recall@k, refusal correctness, guardrail precision/recall (Chapter 54). **Guardrails:** wrong-action rate, policy violations, escalation latency, cost per conversation |
# | Data & labels | Historical conversations with resolutions (noisy), help-centre articles (versioned), human-graded evaluation set stratified by intent; LLM-as-judge only after agreement with humans is measured |
# | Candidate generation | **Retrieval** (RAG): hybrid lexical + dense over help articles and policies (Chapters 45–46), filtered by product and locale; account data via tools with access control |
# | Ranking / model | Re-ranker over retrieved passages; a hosted or self-served instruction model generates the answer with citations; intent and risk classifiers decide *answer / ask clarification / hand off* |
# | Serving & latency | Streaming generation; retrieval ≤ 150 ms; guardrail classifiers on input and output; per-turn timeouts with graceful hand-off |
# | Training cadence | Index updates when articles change (minutes); prompt and policy changes through the same evaluation gates as model changes; fine-tuning only when evaluation shows prompting + retrieval is not enough |
# | Experimentation | A/B by conversation or customer; human review of samples in both arms; ramp by intent (start with low-risk intents) |
# | Monitoring | Hand-off rate by intent, re-contact rate, groundedness on a daily human-graded sample, injection attempts flagged, cost and latency per conversation, retrieval index freshness |
# | Failure modes | Hallucinated policy (refund promised that does not exist), prompt injection through user text or retrieved content, stale articles, loops that never hand off, cost blow-up on long chats |
# | Senior answer | RAG with evaluation set, guardrails, hand-off, cost and latency numbers |
# | Staff adds | Evaluation as a product (who labels, agreement, refresh), phased rollout by intent risk, ownership of help content as a dependency, vendor vs self-hosted model trade-off with cost curves, an incident process for harmful answers |
#
# **Deep dive: cost per conversation.** Each turn re-sends the system prompt,
# the retrieved passages *and the conversation so far*, so prompt tokens grow
# with every turn:
#
# > **Intuition.** A model has no memory between calls, so each turn is like sending a letter that
# > must photocopy the whole correspondence so far. The sixth letter is the thickest, and the cost
# > is per page. *Where the analogy breaks down:* prompt caching (if offered) bills repeated pages at a
# > discount, and summarising old turns trades cost for possible loss of detail.

# %%
llm = {"turns per conversation": 6, "system prompt tokens": 800, "retrieved context tokens per turn": 1_500,
       "user tokens per turn": 60, "answer tokens per turn": 200,
       "price per 1M input tokens (placeholder)": 0.50, "price per 1M output tokens (placeholder)": 2.00,
       "conversations per day": 1_000_000, "human-handled conversation cost (placeholder)": 4.00}
ds.show_assumptions(llm)
rows, prompt_total = [], 0
for turn in range(1, llm["turns per conversation"] + 1):
    history = (turn - 1) * (llm["user tokens per turn"] + llm["answer tokens per turn"])
    prompt = llm["system prompt tokens"] + llm["retrieved context tokens per turn"] + history + llm["user tokens per turn"]
    prompt_total += prompt
    rows.append({"turn": turn, "prompt_tokens": prompt, "of which history": history})
ds.table(rows)
avg_prompt = prompt_total / llm["turns per conversation"]
per_conv = ds.llm_cost_per_conversation(llm["turns per conversation"], avg_prompt, llm["answer tokens per turn"],
                                        llm["price per 1M input tokens (placeholder)"],
                                        llm["price per 1M output tokens (placeholder)"])
print(f"\nLLM cost per conversation: {per_conv:.4f} | per day: {per_conv * llm['conversations per day']:,.0f}")
for deflect in (0.3, 0.5, 0.7):
    blended = per_conv + (1 - deflect) * llm["human-handled conversation cost (placeholder)"]
    print(f"  if {deflect:.0%} resolved without a human: blended cost per conversation {blended:.2f} "
          f"(all-human: {llm['human-handled conversation cost (placeholder)']:.2f})")
NUMBERS["llm_assistant"] = {"assumptions": llm, "prompt_tokens_by_turn": rows, "llm_cost_per_conversation": per_conv}

# %% [markdown]
# **Expected:** prompt tokens grow from 2,360 on turn 1 to 3,660 on turn 6
# (1,300 of them history); about **0.0114** per conversation at the placeholder
# prices (~11,400 a day for 1M conversations). The human fallback dominates
# the blended cost at every deflection rate, which is why the **quality of
# hand-off decisions** (resolve correctly, or hand off early with a good
# summary) matters far more than shaving tokens. Shaving still matters at
# scale: summarise history after a few turns, cap retrieved context, cache the
# static system prompt if the provider supports it (`cached_fraction` in
# `ds.llm_cost_per_conversation`).
#
# **Evaluation in one paragraph (a frequent interview topic).** A versioned
# evaluation set, stratified by intent and risk, with human labels for
# correctness, groundedness (every claim supported by a cited passage) and
# correct hand-off; retrieval measured separately (recall@k of the gold
# article) so failures can be attributed; an LLM judge used only where its
# agreement with humans is measured and reported (Chapter 54); adversarial
# sets for prompt injection; and every prompt, retrieval or model change goes
# through the same gate, because a prompt edit is a model change.
#
# ## 56.5 · Case D: real-time payment fraud
#
# *Prompt: "Design fraud detection for card payments: 5,000 transactions per
# second at peak, decision within 100 ms."*
#
# | Part | Answer |
# |---|---|
# | Requirements | Approve / decline / step-up (e.g. extra verification) per transaction within **≤ 100 ms p99**; explainable reasons for declines; regulators and card networks set constraints |
# | Metrics | **Online:** fraud loss (amount), false-decline rate (good customers declined), step-up rate, approval rate. **Offline:** recall at a fixed false-positive rate, PR-AUC, **amount-weighted** recall, calibration. **Guardrails:** latency, decline rate by merchant/segment, complaint rate |
# | Data & labels | Chargebacks and confirmed fraud arrive **weeks** later; declined transactions never get labels (selective labels: it is never observed whether a declined payment was fraud); analysts' labels are biased toward cases rules flagged |
# | Candidate generation | Not applicable: every transaction is scored. Say so explicitly |
# | Ranking / model | **Rules + model**: hard rules for known patterns and compliance (fast, explainable, editable by analysts the same day); gradient-boosted trees or a neural model on velocity features (counts and sums per card / device / merchant over 1 min–30 d), graph features (shared devices, accounts); cost-based thresholds (Chapter 8) |
# | Serving & latency | Synchronous call from the payment path; streaming feature updates (velocity counters); strict per-stage timeouts; **fail-open or fail-closed** decided per risk tier, never by accident |
# | Training cadence | Weekly retrains with mature labels; fast rule updates for new attack patterns; a small **random holdout of approvals** (or below-threshold sampling) to keep unbiased labels |
# | Experimentation | Shadow scoring first; A/B with loss measured after label maturity (weeks); champion/challenger thresholds on a small share |
# | Monitoring | Score distribution, decline rate by segment in near-real time, rule hit rates, feature-service timeouts, fraud loss at label maturity |
# | Failure modes | Adversarial drift (attackers probe and adapt within days), label delay hides it, a feature service timing out silently declines everyone or approves everyone, false declines that lose good customers |
# | Senior answer | Rules + model, velocity features, latency budget, label delay, cost-based threshold |
# | Staff adds | Selective-label bias and the exploration holdout that fixes it, the analyst–model feedback loop as a system, fail-open/closed policy by tier with sign-off, adversarial red-teaming cadence, regulatory explainability |
#
# **Deep dive: the 100 ms budget.**

# %%
fraud_stages = {"network + auth": 10, "parse + rules (in-process)": 5, "feature fetch (parallel, 6 services)": 30,
                "model inference (GBDT, 600 trees)": 8, "decision + reason codes": 4, "logging (async enqueue)": 3}
ds.table(ds.latency_table(100, fraud_stages))
slow = 0.01  # ASSUMED: each feature service exceeds its 30 ms timeout on 1% of calls, independently
print(f"\nP(at least one of 6 parallel feature calls is slow) = {ds.fanout_tail_probability(slow, 6):.1%}")
print(f"... with 20 feature services:                      {ds.fanout_tail_probability(slow, 20):.1%}")
NUMBERS["fraud"] = {"latency_stages_ms": fraud_stages, "fanout_slow_6": ds.fanout_tail_probability(slow, 6),
                    "fanout_slow_20": ds.fanout_tail_probability(slow, 20)}

# %% [markdown]
# **Expected:** 60 ms of stages and **40 ms (40%)** of headroom, the feature
# fetch the largest stage at 30%; with 6 parallel feature calls each slow 1%
# of the time, **5.9%** of transactions wait on a slow call, and with 20 calls
# **18.2%** do. That is why fraud systems keep the hottest features in one
# low-latency store, set per-call timeouts with **default values** for missing
# features (and a model trained to expect them), and track the
# missing-feature rate as a paged metric.
#
# > **Intuition.** Waiting for six friends before dinner starts: each is late only 1% of the time,
# > but the chance that *someone* is late is almost 6%, and with twenty friends 18%. The request is as
# > slow as its slowest dependency. *Where the analogy breaks down:* the formula assumes independent
# > delays; in practice slowness is correlated (one overloaded cache slows several calls at once),
# > which can make the tail better or much worse.
#
# Fleet size for the scoring service, using the module's helper (equivalent
# to Exercise 2, plus zone redundancy):

# %%
fraud_peak, per_replica = 5_000, 400  # TPS at peak; requests/s one replica sustains (ASSUMED, measure it)
for zones in (1, 3):
    n = ds.replicas_needed(fraud_peak, per_replica, target_utilisation=0.7, zones=zones)
    print(f"zones={zones}: {n} replicas")
NUMBERS["fraud"]["replicas"] = {z: ds.replicas_needed(fraud_peak, per_replica, 0.7, zones=z) for z in (1, 3)}

# %% [markdown]
# **Expected:** 18 replicas in one zone; **27** across 3 zones (9 per zone, so
# losing any one zone leaves 18). Redundancy is a deliberate cost:
# a fraud scorer on the payment path is usually worth it.
#
# **Adversarial drift.** Unlike ads or video, the data-generating process
# adapts against the system. Defences to name: velocity and graph features that are costly for
# attackers to fake; fast analyst-editable rules for new patterns; frequent
# retrains on the freshest *mature* labels plus early signals (customer
# disputes, analyst confirmations); monitoring of rule-hit and score
# distributions by merchant/channel to see probing; and red-team exercises.
#
# ## 56.6 · Comparing the four cases
#
# | | Ads | Video | LLM assistant | Fraud |
# |---|---|---|---|---|
# | Signature difficulty | calibration × bids, marketplace | long-term value vs watch time | evaluating generated text | adversaries + delayed, selective labels |
# | Latency (p99) | ~100 ms inside the feed | ~200 ms | TTFT ~1.5 s, streaming | ≤ 100 ms on the payment path |
# | Label delay | seconds (click) to days (conversion) | seconds to days (return visits) | days (re-contact) + human grading | weeks (chargebacks) |
# | Fallback | no ad | popular / precomputed list | hand off to a human | rules-only decision, per-tier fail-open/closed |
# | Most likely staff probe | budget-split experiments, advertiser ROI | holdouts, creator ecosystem | evaluation programme, rollout by intent risk | exploration holdout, fail-open policy |
#
# Save every number for the written design:

# %%
(OUT / "case_numbers.json").write_text(json.dumps(NUMBERS, indent=2, default=float))
print("wrote", (OUT / "case_numbers.json").relative_to(ART), "with cases:", list(NUMBERS))

# %% [markdown]
# **Expected:** `wrote level3/nb18/case_numbers.json with cases: ['ads', 'video', 'llm_assistant', 'fraud']`.
#
# ## 56.7 · The 45-minute format
#
# | Minutes | Do |
# |---|---|
# | 0–5 | Clarify: users, decision, scale, latency, constraints; write assumptions on the board |
# | 5–9 | Metrics: online, offline, guardrails; the one the launch decision rests on |
# | 9–14 | Data and labels: sources, delay, bias; the training set's unit and time split |
# | 14–24 | Architecture with numbers: stages, QPS, per-stage latency, fleet size, memory |
# | 24–34 | Two deep dives (the interviewer usually picks one; offer the other) |
# | 34–40 | Experimentation, rollout, monitoring, failure modes and fallbacks |
# | 40–45 | Trade-offs made, what to build first, what would change the design; questions |
#
# > **Exercise 3 (written: a full design).** Pick one of the four cases (or a
# > variation: ads for a search engine, short-video feed, an internal-docs
# > assistant, account-takeover detection). Set a 45-minute timer and write the
# > design in the twelve-part structure, with: at least one capacity number
# > from `ds` or your Exercise 2, a latency budget checked with your
# > Exercise 1, two deep dives (one must be the case's *signature difficulty*
# > from section 56.6), three failure modes with their fallbacks, and a final
# > paragraph "what a staff-level answer would add" in your own words. Then
# > present it aloud in 15 minutes.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · Peak QPS.**
#
# $$
# \text{QPS}_{\text{peak}} = \frac{\text{DAU} \times \text{requests per user per day}}{86{,}400} \times \frac{\text{peak}}{\text{average}}
# $$
#
# *Example (ads):* $3 \times 10^8 \times 20 / 86{,}400 \times 2 = 138{,}889$ requests/s.
#
# **2 · Fleet size with headroom.**
#
# $$
# \text{replicas} = \max\!\left(2,\ \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS}_{\text{replica}} \cdot (1 - h)} \right\rceil \right)
# $$
#
# *Symbols:* $h$ the headroom fraction. *Example:* $138{,}889 / (50 \times 0.7) = 3{,}968.3 \to 3{,}969$
# replicas. With $z$ zones that must survive losing one, size each zone for $1/(z-1)$ of the load (fraud: 9 per zone × 3 = 27).
#
# **3 · Latency budget and fan-out tail.**
#
# $$
# \text{headroom} = T_{\text{total}} - \sum_{\text{stages}} t_s, \qquad P(\text{some call slow}) = 1 - (1 - p)^{k}
# $$
#
# *Example (fraud):* $100 - 60 = 40$ ms of headroom; $k = 6$ calls each slow with $p = 0.01$:
# $1 - 0.99^6 = 5.9\%$; $k = 20$: 18.2%. (Percentiles do not add: the sum of stage p99s is only an upper bound on the end-to-end p99.)
#
# **4 · LLM cost per conversation.**
#
# $$
# P_t = S + R + U + (t - 1)(U + A), \qquad \text{cost} = \sum_{t=1}^{n} \big( P_t\, p_{\text{in}} + A\, p_{\text{out}} \big)
# $$
#
# *Symbols:* $S$ system-prompt, $R$ retrieved, $U$ user and $A$ answer tokens per turn; $p_{\text{in}}, p_{\text{out}}$
# prices per token. *Example:* $P_1 = 800 + 1{,}500 + 60 = 2{,}360$, $P_6 = 2{,}360 + 5 \times 260 = 3{,}660$; 18,060
# input and 1,200 output tokens: $18{,}060 \times 0.5/10^6 + 1{,}200 \times 2/10^6 = \$0.0114$.
#
# **5 · Blended cost with a human fallback.** $c = c_{\text{LLM}} + (1 - d)\, c_{\text{human}}$, with deflection $d$.
# *Example:* $0.0114 + 0.5 \times 4.00 = \$2.01$: the human term dominates, so hand-off quality matters more than tokens.
#
# **6 · eCPM** (ads, Chapter 52): $\mathrm{eCPM} = \mathrm{pCTR} \times \text{bid} \times 1000$, which is why a per-segment calibration error moves money.
#
# The cell below recomputes 1–4 by hand and compares them with the numbers this notebook printed.

# %%
# Key equations, recomputed by hand (instant).
import math

peak = 300e6 * 20 / 86_400 * 2.0
print(f"ads peak QPS: by hand {peak:,.0f} | notebook {NUMBERS['ads']['peak_ad_requests_per_s']:,.0f}")
print(f"ranker replicas: by hand {max(2, math.ceil(peak / (50 * (1 - 0.3)))):,} | notebook {NUMBERS['ads']['ranker_replicas']:,}")
for k in (6, 20):
    sims = (np.random.default_rng(k).random((200_000, k)) < 0.01).any(axis=1).mean()
    print(f"fan-out {k:2d}: 1 - 0.99^k = {1 - 0.99**k:.4f} | simulated {sims:.4f}")
P = [800 + 1_500 + 60 + (t - 1) * (60 + 200) for t in range(1, 7)]
cost = sum(p_ * 0.50 / 1e6 + 200 * 2.00 / 1e6 for p_ in P)
print(f"prompt tokens by turn {P} | cost by hand {cost:.5f} | notebook "
      f"{NUMBERS['llm_assistant']['llm_cost_per_conversation']:.5f}")

# %% [markdown]
# **Expected:** `138,889` and `3,969` both ways; `0.0585` and `0.1821` for the formula with simulated
# values within about ±0.001; prompt tokens `[2360, 2620, 2880, 3140, 3400, 3660]` and `0.01143` both ways.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Exercise 2 returns a float | `math.ceil` result divided again, or `/` at the end | compute the ratio, then `int(math.ceil(...))`, then `max(2, ...)` |
# | Exercise 2 one replica short | rounded instead of ceiling | capacity must be ≥ load: always round **up** |
# | Exercise 1 crashes on `{}` | `max()` of an empty sequence | return `largest_stage=None` when there are no stages |
# | The QPS is 86,400× too big or too small | per day vs per second | write the unit next to every number |
# | The design runs out of time at "architecture" | too long on clarification and metrics | timebox 0–14 minutes strictly; state assumptions and move on |
# | Interviewer: "but how do you know it is better?" | no evaluation story | for each case, name offline metric + online experiment + guardrail, and when labels mature |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under three minutes each:
#
# 1. Why must an ads CTR model be calibrated, when a feed ranker often does not need to be? What happens to calibration when you down-sample negatives, and how do you fix it?
# 2. Your video ranker optimises expected watch time and median served length has doubled in a month. Is that a problem? What do you measure, and what do you change?
# 3. Estimate the LLM cost of a 6-turn support conversation and say which two levers you would pull first to halve it. Why is hand-off quality usually the bigger cost lever?
# 4. A fraud model's online decline rate jumped from 1% to 4% at 2 a.m. Walk through your diagnosis, from feature-service timeouts to adversarial behaviour.
# 5. Declined transactions never get labels. How does that bias your next model, and what do you build to fix it?
# 6. Pick one case and say, concretely, what a staff-level answer adds over a strong senior answer.
#
# **Next chapter:** [Chapter 57 · ML depth & research discussion](../ch57-ml-depth-and-research-discussion/notebook.ipynb) — the depth round. Optimisation, generalisation, transformers and recsys questions with small experiments, and how to critique a paper.
