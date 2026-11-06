# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 19 · Recommendation & search system design
#
# **Day 10 · ~90 minutes · CPU only**
#
# "Design the home feed" and "design product search" are the two most common
# ML system design prompts at large consumer companies, and they reward the
# same thing: a structured walk from the product goal to a cascade you can
# put numbers on, with the failure modes named before the interviewer asks.
# This notebook gives you two complete worked answers, the arithmetic behind
# them, and six practice prompts with what a strong answer must cover.
#
# By the end you will be able to:
#
# 1. run the Level 1 design framework on recommendation and search problems, adding the cascade-specific steps;
# 2. do the capacity arithmetic out loud: QPS, candidates per stage, scoring load, embedding and index memory;
# 3. ground ANN choices in a measured recall–latency curve rather than folklore;
# 4. design a home feed and a product search system end to end, including position bias, freshness, experimentation and monitoring;
# 5. recognise the failure modes that sink otherwise good answers.
#
# **You will produce:** `artifacts/level2/design/ann_tradeoff.csv` and
# `artifacts/level2/design/capacity_home_feed.csv`, plus your written design
# (Exercise 3).

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level2/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: nothing to rebuild; this notebook is arithmetic and design ----
import math
import time

import numpy as np
import pandas as pd

from shoprank.grading import nb19 as grade

OUT = ART / "level2" / "design"
OUT.mkdir(parents=True, exist_ok=True)
SECONDS_PER_DAY = 86_400

# %% [markdown]
# ## 19.0 · The framework, extended for ranking systems
#
# Level 1 (handbook chapter 7 and Level 1 notebook 17) gave you an eight-step
# framework. Recommendation and search problems use the same steps with two
# additions that interviewers explicitly look for: **candidate generation**
# and **ranking** as separate stages, because no system can score the whole
# catalogue per request.
#
# | # | Step | Minutes (of 40) | The ranking-specific questions |
# |---|---|---|---|
# | 1 | **Clarify** | 3–4 | Surface (feed, search, email)? Users, items, requests/day? Latency? Item churn (how fast do new items arrive)? |
# | 2 | **Metrics** | 3 | North star (e.g. purchases, long-term engagement) vs proxies (CTR); offline (NDCG, recall@K); guardrails (latency, diversity, seller fairness, complaints) |
# | 3 | **Data & labels** | 4 | Implicit feedback (clicks, dwell, purchases, skips); **position bias**; exploration traffic; label delay (purchases); negative sampling |
# | 4 | **Features** | 3 | User, item, user×item, context; real-time vs batch; point-in-time training sets |
# | 5 | **Candidate generation** | 5 | Multiple sources (two-tower ANN, co-visitation, popularity, follow graph, fresh items); recall@K per source; cold start |
# | 6 | **Ranking** | 5 | Pre-ranker → heavy ranker (multi-task: click, purchase, dwell) → re-ranking (diversity, rules, ads blending); loss and calibration |
# | 7 | **Serving** | 5 | QPS, latency budget per stage, caching, fallbacks, feature-store reads, index updates |
# | 8 | **Experimentation** | 4 | A/B unit (user), primary and guardrail metrics, interleaving for rankers, novelty effects, network effects |
# | 9 | **Monitoring** | 3 | Fallback rates, feature freshness, score distributions, coverage, online–offline gaps |
# | 10 | **Risks** | 2 | Feedback loops, popularity bias, filter bubbles, abuse (fake engagement), privacy |
#
# Say the step names out loud as you go. A clear map lets the interviewer
# steer you to the part they want to probe, which is usually 5–7.

# %% [markdown]
# ## 19.1 · The back-of-envelope toolkit
#
# Two primitives come up in every ranking design. Write them, then use the
# helpers below (which build on the same arithmetic) for the full tables.
#
# > **Exercise 1.** Write `embedding_memory_gb(n, dim, bytes_per)`: the memory
# > in **decimal gigabytes** (1 GB = 10⁹ bytes) of a dense table of `n`
# > vectors with `dim` dimensions and `bytes_per` bytes per number (4 for
# > float32, 2 for float16, 1 for int8). Return a float.

# %%
def embedding_memory_gb(n, dim, bytes_per):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(embedding_memory_gb)

# %% [markdown]
# > **Exercise 2.** Write `qps(dau, req_per_user, peak)`: **peak** queries per
# > second for `dau` daily active users making `req_per_user` requests a day,
# > with `peak` the ratio of peak to average traffic. Return a float. Then,
# > *without running anything*, say the answer for 50M DAU × 20 requests ×
# > peak 2.5 to one significant figure.

# %%
def qps(dau, req_per_user, peak):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(qps)

# %% [markdown]
# The helpers below cover the rest: ANN index memory by index type, and
# servers needed at a target utilisation. Every *input* to them in this
# notebook is an **assumption stated for the design**; in a real design you
# would measure the per-core throughput (as in notebooks 17 and 18).

# %%
def gb(n_bytes: float) -> float:
    return n_bytes / 1e9


def ann_index_gb(n: int, dim: int, kind: str = "flat", bytes_per: int = 4, hnsw_m: int = 32,
                 pq_bytes: int = 64) -> float:
    """Approximate memory of an ANN index (vectors + structure; ids are 8 bytes)."""
    if kind == "flat":       # raw vectors
        per = dim * bytes_per
    elif kind == "hnsw":     # raw vectors + ~2*M neighbour links (4-byte ids) on the base layer
        per = dim * bytes_per + 2 * hnsw_m * 4
    elif kind == "ivf_pq":   # compressed codes + 8-byte ids (centroids are negligible)
        per = pq_bytes + 8
    else:
        raise ValueError(kind)
    return gb(n * per)


def peak_qps_of(dau: float, req_per_user: float, peak: float) -> float:
    return dau * req_per_user / SECONDS_PER_DAY * peak


def servers(load_per_s: float, per_core_per_s: float, cores_per_server: int = 32, utilisation: float = 0.6) -> int:
    """Servers to sustain `load_per_s` units of work, running each at `utilisation` of capacity."""
    return max(1, math.ceil(load_per_s / (per_core_per_s * cores_per_server * utilisation)))


print(f"100M items x 128 dims: flat fp32 {ann_index_gb(100_000_000, 128):.1f} GB | "
      f"HNSW fp32 {ann_index_gb(100_000_000, 128, 'hnsw'):.1f} GB | "
      f"IVF-PQ 64 B {ann_index_gb(100_000_000, 128, 'ivf_pq'):.1f} GB")

# %% [markdown]
# **Expected:** 51.2 GB flat, 76.8 GB with HNSW links, 7.2 GB with IVF-PQ
# codes. That one line changes the architecture: 51 GB does not fit one
# commodity serving host's RAM comfortably alongside everything else, 7 GB
# does — at the cost of recall, which is the next section.

# %% [markdown]
# ## 19.2 · Measure, don't guess: the ANN recall–latency curve
#
# "We'll use HNSW" is not an answer; "we'll pick the index and its search
# parameter from a recall@10 vs latency curve at our target p99" is. Here is
# that curve for 100,000 clustered 64-dim unit vectors, single-threaded,
# against exact search as ground truth. (The vectors are synthetic and
# harder than random ones of the same size: clusters overlap.)

# %%
import faiss

rng = np.random.default_rng(0)
n, dim = 100_000, 64
centres = rng.normal(size=(200, dim)).astype("float32")


def sample(m):
    z = (centres[rng.integers(0, 200, m)] + 1.5 * rng.normal(size=(m, dim))).astype("float32")
    return z / np.linalg.norm(z, axis=1, keepdims=True)


base, queries = sample(n), sample(300)
faiss.omp_set_num_threads(1)
exact = faiss.IndexFlatIP(dim)
exact.add(base)
t = time.perf_counter()
_, truth = exact.search(queries, 10)
flat_ms = (time.perf_counter() - t) * 1000 / len(queries)

rows = [{"index": "Flat (exact)", "param": "-", "build s": 0.0, "ms / query": flat_ms, "recall@10": 1.0,
         "memory MB": ann_index_gb(n, dim) * 1000}]
for name, factory, params in [("IVF256,Flat", lambda: faiss.index_factory(dim, "IVF256,Flat", faiss.METRIC_INNER_PRODUCT), [1, 4, 16, 64]),
                              ("HNSW M=16", lambda: faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT), [16, 64, 256])]:
    index = factory()
    t = time.perf_counter()
    faiss.omp_set_num_threads(2)
    if not index.is_trained:
        index.train(base[:20_000])
    index.add(base)
    build = time.perf_counter() - t
    faiss.omp_set_num_threads(1)
    for p in params:
        if name.startswith("IVF"):
            index.nprobe = p
        else:
            index.hnsw.efSearch = p
        t = time.perf_counter()
        _, found = index.search(queries, 10)
        ms = (time.perf_counter() - t) * 1000 / len(queries)
        recall = np.mean([len(set(found[i]) & set(truth[i])) / 10 for i in range(len(queries))])
        mem = ann_index_gb(n, dim, "hnsw", hnsw_m=16) if name.startswith("HNSW") else ann_index_gb(n, dim)
        rows.append({"index": name, "param": f"{'nprobe' if name.startswith('IVF') else 'efSearch'}={p}",
                     "build s": build, "ms / query": ms, "recall@10": recall, "memory MB": mem * 1000})
ann = pd.DataFrame(rows)
ann.to_csv(OUT / "ann_tradeoff.csv", index=False)
ann.round(3)

# %% [markdown]
# **Expected (our runs):** exact search about 1.1 ms per query. IVF256 from
# about 0.01 ms at recall 0.80 (nprobe 1) to about 0.4 ms at recall 0.99
# (nprobe 64); HNSW M=16 from about 0.04 ms at recall 0.72 (efSearch 16) to
# about 0.3 ms at recall 0.96 (efSearch 256), with a build about 10× slower
# than IVF's. On this data IVF wins at every recall level; on other data
# HNSW often does. That is the point: **measure on your vectors**.
#
# How to scale the numbers in your head: exact search cost grows linearly with
# `n` (100M items would be ~1 s per query single-threaded: impossible), while
# IVF cost grows with `n × nprobe / nlist` and HNSW roughly with `log n`. At
# 100M items you shard (each shard holds 5–20M vectors), search shards in
# parallel and merge, and the slowest shard sets your tail latency.

# %% [markdown]
# ## 19.3 · Worked design A: the home feed of a 50M-DAU marketplace
#
# *Prompt: "Design the home-feed recommendations for a marketplace app with
# 50 million daily active users."* Every number below is an **assumption for
# the design**, stated as you would state it aloud ("tell me if that's
# off").
#
# ### 1. Clarify
#
# * Surface: an infinite-scroll feed of item cards on app open; ~20 feed
#   requests (pages) per user per day; 20 items per page.
# * Catalogue: 100M active listings, ~2M new per day (marketplaces churn fast:
#   items sell out). 200M monthly active users.
# * Latency: p99 ≤ 200 ms at the client, server ≤ 150 ms.
# * Out of scope (say so): ads blending, notifications, search.
#
# ### 2. Metrics
#
# * **North star:** purchases (or gross merchandise value) per user per week;
#   long-term retention as a slower guardrail.
# * **Online proxies:** feed CTR, add-to-cart rate, purchases from feed,
#   session length. CTR alone rewards clickbait; tie it to purchases.
# * **Offline:** recall@K per candidate source; NDCG@20 and calibrated log
#   loss for the ranker, **IPS-corrected** for position bias (notebook 05) or on
#   exploration traffic.
# * **Guardrails:** p99 latency, error/fallback rate, catalogue coverage,
#   seller concentration (share of impressions to top 1 % of sellers),
#   complaint/hide rate.
#
# ### 3. Data and labels
#
# * Impression logs with position, request id and model version; clicks,
#   add-to-carts, purchases (delayed up to days), hides (negative).
# * Labels per task: click (fast, noisy, biased by position), purchase (slow,
#   sparse, most valuable), dwell time.
# * **Position bias:** the logging policy decides what gets labels. Keep a
#   small exploration bucket (e.g. 1 % of requests with randomised order in the
#   top slots) to estimate propensities and to evaluate unbiasedly.
# * Negatives: shown-but-not-clicked, not random catalogue items, for the
#   ranker; in-batch or sampled negatives for retrieval (with the
#   popularity correction of notebook 08).
#
# ### 4. Features
#
# * **User:** long-term category/brand/price-band affinities (daily batch),
#   recent sequence of the last 50 interactions (real time, streaming),
#   country, device.
# * **Item:** category, price, seller quality, age, CTR/CVR by surface
#   (hourly), image/text embeddings (computed at listing time).
# * **User × item:** retrieval scores, co-visitation scores, "seen N times
#   without click" (fatigue), price relative to the user's usual band.
# * **Context:** time of day, page index, session depth.
# * Point-in-time training sets from the feature log (log the features *as
#   served* at request time: the cleanest way to avoid skew, notebook 03).
#
# ### 5. Candidate generation (~2,000 per request)
#
# | Source | Candidates | Why |
# |---|---|---|
# | Two-tower ANN (user tower on recent sequence) | 800 | personalised, broad recall |
# | Item-to-item co-visitation from recent clicks | 500 | strong short-term intent |
# | Followed sellers / saved searches | 300 | explicit intent |
# | Fresh items in affine categories | 200 | cold start: new items get impressions |
# | Trending / popular by country | 200 | fallback, new users |
#
# Merge, dedupe, drop sold-out and blocked items (filtering *before* ranking
# saves ranker compute).
#
# ### 6. Ranking
#
# * **Pre-ranker:** 2,000 → 500, a cheap two-tower-style dot product or tiny
#   MLP (distilled from the heavy ranker, notebook 18).
# * **Heavy ranker:** 500 → ranked list, a multi-task DNN with shared bottom
#   and heads for P(click), P(add-to-cart), P(purchase | click); final score a
#   weighted combination tuned by online experiments. Position as a training
#   feature set to a constant at serving (notebook 17), or a separate
#   position tower.
# * **Re-ranking:** diversity (category cap / MMR), seller caps, freshness
#   boost, policy filters, dedupe across pages of the same session.
#
# ### 7. Serving: the numbers

# %%
DAU, REQ_PER_USER, PEAK = 50_000_000, 20, 2.5
peak = peak_qps_of(DAU, REQ_PER_USER, PEAK)
ranker_rows_per_s = peak * 500          # heavy ranker scores 500 candidates per request
preranker_rows_per_s = peak * 2_000
feature_reads_per_s = peak * (1 + 500)  # one user row + 500 item rows per request
RANKER_ROWS_PER_CORE = 20_000           # ASSUMPTION for a mid-size multi-task DNN on CPU; measure it
PRERANK_ROWS_PER_CORE = 1_000_000       # ASSUMPTION: a dot product / tiny MLP (notebook 18 measured ~4M/s for a 128-64 MLP)

capacity = pd.DataFrame([
    {"quantity": "requests per day", "value": DAU * REQ_PER_USER, "unit": "requests"},
    {"quantity": "average QPS", "value": DAU * REQ_PER_USER / SECONDS_PER_DAY, "unit": "req/s"},
    {"quantity": "peak QPS (x2.5)", "value": peak, "unit": "req/s"},
    {"quantity": "pre-ranker rows scored at peak", "value": preranker_rows_per_s, "unit": "rows/s"},
    {"quantity": "heavy-ranker rows scored at peak", "value": ranker_rows_per_s, "unit": "rows/s"},
    {"quantity": "heavy-ranker servers (32 cores, 60% util)", "value": servers(ranker_rows_per_s, RANKER_ROWS_PER_CORE), "unit": "servers"},
    {"quantity": "pre-ranker servers", "value": servers(preranker_rows_per_s, PRERANK_ROWS_PER_CORE), "unit": "servers"},
    {"quantity": "feature-store reads at peak", "value": feature_reads_per_s, "unit": "keys/s"},
    {"quantity": "user embeddings 200M x 128 fp16", "value": ann_index_gb(200_000_000, 128, bytes_per=2), "unit": "GB"},
    {"quantity": "item index 100M x 128 HNSW fp32", "value": ann_index_gb(100_000_000, 128, "hnsw"), "unit": "GB"},
    {"quantity": "item index 100M x 128 IVF-PQ 64B", "value": ann_index_gb(100_000_000, 128, "ivf_pq"), "unit": "GB"},
])
capacity.to_csv(OUT / "capacity_home_feed.csv", index=False)
capacity.assign(value=capacity["value"].map(lambda v: f"{v:,.1f}"))

# %% [markdown]
# **Expected:** 1 billion requests a day, ~11,600 average and ~28,900 peak
# QPS; 14.5M heavy-ranker rows per second at peak, which at the assumed
# 20,000 rows/s per core is 38 servers of 32 cores; 4 pre-ranker servers;
# ~14.5M feature-store key reads per second; 51 GB of fp16 user embeddings,
# 77 GB for an HNSW item index or 7 GB with IVF-PQ.
#
# What the numbers decide:
#
# * **Feature reads dominate.** 14.5M item-feature reads per second is why
#   item features live in a **local in-memory cache** on ranker hosts (hot
#   items repeat across users) and only the user row is a remote read.
# * **The heavy ranker is the cost centre:** halving its rows (a better
#   pre-ranker) or its per-row cost (distillation, notebook 18) is worth ~19
#   servers. That is the business case for notebook 18.
# * **The item index needs sharding or compression:** 77 GB of HNSW goes to
#   4–8 shards with replicas; IVF-PQ fits one host but loses recall, so
#   re-score its top few hundred with exact vectors.
# * **Freshness:** 2M new items/day must be searchable within minutes →
#   an index that supports incremental adds (HNSW/IVF both do) plus a nightly
#   rebuild; new items also need a cold-start source (the "fresh" row in 5).
#
# **Latency budget (server p99 150 ms):** request fan-in and user features
# 10 ms → retrieval sources in parallel 25 ms → merge/filter 5 ms → item
# features (local cache) 10 ms → pre-ranker 10 ms → heavy ranker 45 ms →
# re-rank and rules 10 ms → logging, serialisation 10 ms → slack 25 ms.
# Fallbacks: ranker timeout → pre-ranker order; retrieval source timeout →
# drop that source (with a counter); everything down → cached trending list
# per country.
#
# ### 8. Experimentation
#
# * A/B by **user** (feeds have carry-over within a user), 1–2 weeks to cover
#   weekly cycles and novelty; primary metric purchases per user; guardrails
#   from step 2. Power the test with CUPED on pre-period purchases
#   (notebook 12).
# * **Interleaving** (team-draft) to rank many ranker candidates quickly with
#   ~10–100× less traffic, then A/B the winner.
# * Watch for **marketplace interference**: if treatment users buy more of a
#   scarce item, control users see it sell out. Seller-side metrics or
#   switchback/cluster designs if the effect matters.
#
# ### 9. Monitoring
#
# Per-stage latency and fallback rates; candidate-source contribution (share
# of final items from each source); feature freshness (age of the newest
# value per feature family); score distribution and calibration (predicted vs
# observed CTR by decile, daily); coverage and seller concentration; the
# online–offline gap after each launch.
#
# ### 10. Risks
#
# Feedback loops (only shown items get labels → popularity snowballs; keep
# exploration), filter bubbles (diversity constraints, measured), seller
# fairness (new sellers need impressions), fake engagement (bot clicks
# poisoning co-visitation: filter by account quality), privacy (sequence
# features are sensitive; retention limits).

# %% [markdown]
# ## 19.4 · Worked design B: product search ranking
#
# *Prompt: "Design the ranking for product search in the same marketplace."*
# Again, every number is an assumption.
#
# **1. Clarify.** 50M DAU × 3 searches/day = 150M searches/day, ~1,700
# average and ~5,000 peak QPS (×3: search is peakier than feed). p99 ≤ 300 ms
# end to end including the text-index call. 100M listings, multilingual,
# structured attributes (brand, size, colour). Queries: very skewed — the
# head (top 10K queries) is ~30 % of traffic, the long tail is unique.
#
# **2. Metrics.** North star: search purchase conversion (purchases per
# search session) and revenue per search. Proxies: CTR@k, time to first click,
# **reformulation rate** and **zero-result rate** (both strong signals of
# failure), abandonment. Offline: NDCG@10 on **human-graded relevance**
# (exact / substitute / complement / irrelevant) for relevance, plus
# IPS-weighted click/purchase NDCG for engagement. Guardrails: latency,
# zero-result rate, seller concentration.
#
# **3. Data and labels.** Two label sources with different failure modes:
# human relevance ratings (unbiased w.r.t. position, expensive, a few
# hundred thousand query–item pairs; cover the tail by sampling) and
# behavioural labels (clicks, add-to-carts, purchases; plentiful, but
# **position-biased** and only for shown items). Estimate position
# propensities from randomisation (swap adjacent results in a small slice of
# traffic, or the exploration bucket) and train with IPS-weighted LambdaMART
# (notebooks 05–06), or model position explicitly and set it to a constant at
# serving.
#
# **4. Query understanding** (before retrieval): normalisation and spelling
# correction; tokenisation per language; **category classification** of the
# query (a classifier over the taxonomy, trained on purchase logs);
# **attribute extraction** ("red nike running shoes size 10" → colour, brand,
# size filters) with a sequence tagger; query expansion with synonyms;
# intent (navigational "nike store" vs exploratory "gift for runner").
#
# **5. Retrieval (~1,500 candidates).**
#
# * **Lexical:** an inverted index with BM25 over title, attributes and
#   description fields (Elasticsearch/Lucene-style), filtered by extracted
#   attributes; top 1,000. Precise for exact terms, model numbers and brands.
# * **Semantic:** a two-tower model (query encoder, item encoder over title +
#   attributes + image embedding) trained on query→purchase pairs; ANN top
#   500. Catches vocabulary mismatch ("sofa" vs "couch") and the long tail.
# * Merge; for head queries also a **cache** of results (refreshed hourly).
#
# **6. Ranking.** L1: LambdaMART over ~100 features (BM25 per field,
# semantic similarity, query-category match probability, attribute matches,
# price vs the query's typical price, item CTR/CVR **for this query** and
# overall, seller quality, item age, personalisation such as the user's
# brand affinity). Top 200 → L2: a cross-encoder (query and item text
# jointly) distilled to fit the budget, for the top 50. Then business rules:
# out-of-stock filtering, ads slots, diversity for ambiguous queries.
#
# **Freshness.** New listings have no engagement features → their CTR
# features are smoothed toward a category prior (notebook 17's
# `item_ctr` smoothing), and a small, capped **freshness boost** or exploration
# slot gives them impressions; index updates must be near-real-time (seconds
# to minutes) because sold items must disappear quickly.
#
# **7. Serving.** Latency budget (p99 300 ms): query understanding 20 ms,
# lexical + semantic retrieval in parallel 60 ms, feature fetch 30 ms, L1
# 40 ms, L2 on 50 items 60 ms, rules and ads 20 ms, network and slack 70 ms.
# Fallbacks: L2 timeout → L1 order; semantic retrieval down → lexical only;
# query understanding down → raw query to BM25.
#
# **8. Experimentation.** Interleaving is the workhorse for search ranking
# (sensitive, needs little traffic); A/B by user for launches; human-rated
# side-by-side evaluations for relevance changes that engagement metrics
# cannot see (e.g. fixing offensive results for a rare query).
#
# **9–10. Monitoring and risks.** Zero-result and reformulation rates per
# query segment (head/torso/tail, language); latency per stage; drift in query
# mix (seasonality: "costume" in October). Risks: spam listings stuffing
# keywords (field-level features and quality models), popularity feedback
# loops on head queries, sensitive queries needing policy filters, and
# offline relevance labels drifting from what users want.

# %% [markdown]
# ## 19.5 · Practice prompts
#
# Do one per day, timed (40 minutes, out loud, recorded). Use the 10 steps.
# The right column is what an interviewer's rubric typically looks for
# beyond the framework; check yourself against it afterwards, not before.
#
# | # | Prompt | A strong answer covers |
# |---|---|---|
# | 1 | **Short-video feed** ("For You", 500M DAU) | watch-time and completion as labels (and their bias toward long videos), sequence models over recent views, fast feedback (minutes) and online learning, freshness and creator cold start, exploration budget, integrity filters before ranking |
# | 2 | **Ads click-through prediction** for a feed | **calibration** matters (bids × pCTR in an auction), log loss not just AUC, delayed conversions, sample/negative down-sampling and recalibration, advertiser budget pacing, auction effects on metrics |
# | 3 | **"People you may know"** | graph candidates (friends-of-friends, shared contacts/groups), sparse positive labels (accepted invites), privacy constraints, spam/abuse, asymmetric acceptance, network effects in A/B tests (cluster randomisation) |
# | 4 | **Push-notification recommendations** | uplift not CTR (would they have opened anyway?), volume caps and fatigue, unsubscribe as a hard guardrail, send-time optimisation, long-term holdout groups |
# | 5 | **Query autocomplete** | prefix tries / FSTs for latency (< 50 ms), popularity + personalisation + freshness (trending), offensive-suggestion filtering, evaluation by keystrokes saved and downstream search success |
# | 6 | **"Similar items" on a product page** | item–item retrieval (co-visitation vs content embeddings vs substitutes vs complements), cold-start via content, evaluation with add-to-cart from the module, avoiding near-duplicates, precomputation (batch) vs online |
#
# ## 19.6 · Common failure modes in design rounds
#
# | Failure | What it sounds like | Fix |
# |---|---|---|
# | Model-first | "I'd use a transformer…" in minute 2 | clarify and set metrics first; name a baseline |
# | One-stage system | scoring the whole catalogue per request | cascade with numbers per stage |
# | CTR as the goal | optimising clicks only | north star + proxies + guardrails; multi-task heads |
# | Ignoring position bias | training on raw clicks, evaluating on clicks | exploration traffic, IPS, or position-as-feature |
# | No numbers | "it should be fast" | QPS, candidates, ms per stage, GB of embeddings |
# | No cold start | new users/items silently get nothing | popularity/fresh sources, content features, exploration |
# | Offline-only evaluation | "NDCG went up, ship it" | A/B or interleaving, guardrails, novelty, online–offline gap |
# | No failure handling | single happy path | timeouts and fallbacks per stage, counted and alerted |
# | Monologue | 15 minutes without a check-in | signpost each step; ask where to go deeper |
# | Buzzword stacking | five model names, no trade-offs | one choice per stage with the criterion that decided it |
#
# **Mid-level vs senior** (generic expectations, not any company's ladder): a
# mid-level answer gets the cascade and metrics right and can implement each
# box; a senior answer also *prioritises* (what to build first and why),
# quantifies trade-offs, anticipates organisational and operational issues
# (who owns the feature store, how a launch is rolled back, on-call load),
# and says what evidence would change the design.
#
# > **Exercise 3 (written design).** Pick prompt 1, 2 or 6 from 19.5. In a
# > Markdown cell, write a design in the 10-step structure, at most one page,
# > with at least: a capacity table (peak QPS, candidates per stage, scoring
# > rows/s, embedding memory, computed with your Exercise 1–2 functions or the
# > helpers above, each input labelled as an assumption), a per-stage latency
# > budget, two candidate sources with their failure modes, the training
# > labels and how you correct their bias, the online experiment, and the two
# > risks you consider most serious. Then time yourself presenting it in 10
# > minutes.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ImportError: faiss` | `faiss-cpu` not installed | re-run the setup cell |
# | ANN timings far from the Expected line | shared/slower CPU (Colab) | compare *ratios* between rows; the curve's shape is the lesson |
# | HNSW build takes minutes | more threads contending, or a larger `n` | keep `n = 100_000`; `faiss.omp_set_num_threads(2)` |
# | Your capacity numbers differ by 1,000× from the Expected line | GB vs MB, or per day vs per second | write units next to every number; 86,400 s per day |
# | Exercise 1 off by ~7 % | used 2³⁰ (GiB) instead of 10⁹ | the exercise asks for decimal GB; say which one you use in interviews |
# | The design takes 70 minutes | depth everywhere | 2–3 minutes per step, then go deep where the interviewer points |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under three minutes each:
#
# 1. Design the candidate generation for a home feed: which sources, how many candidates each, how you would measure each source's value, and how new items get in.
# 2. Estimate the peak QPS, ranker load and embedding memory for a feed with 100M DAU, 30 requests/user/day, 1,000 ranked candidates per request and 256-dim item embeddings for 500M items. Which number changes your architecture?
# 3. In product search, when do lexical and semantic retrieval each fail, and how do you combine them?
# 4. Your search ranker is trained on clicks. Explain position bias, how you would estimate it, and two ways to correct for it in training and evaluation.
# 5. How would you A/B test a new feed ranker in a marketplace where treatment and control compete for the same inventory?
# 6. Walk through what you monitor after launching a new ranker, and the three alerts you would page on.
#
# ➡️ **Next:** [`20_final_evaluation_and_mock_loop.ipynb`](20_final_evaluation_and_mock_loop.ipynb) — open the test week once, with a pre-registered plan, then run a full timed mock loop.
