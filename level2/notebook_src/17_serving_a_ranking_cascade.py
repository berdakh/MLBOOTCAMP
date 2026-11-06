# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 17 · Serving a ranking cascade
#
# **Day 9 · ~90 minutes · CPU only**
#
# Every recommender you will be asked about in an interview is a *cascade*:
# a cheap stage narrows thousands of items to hundreds, an expensive model
# orders those hundreds, and rules make the result shippable, all inside a
# latency budget of tens of milliseconds. Building one end to end, measuring
# each stage, and breaking it on purpose is the fastest way to be able to
# talk about one with numbers instead of adjectives.
#
# By the end you will be able to:
#
# 1. build retrieval (FAISS + co-visitation + popularity) → feature fetch → ranker → business rules from point-in-time data;
# 2. measure the cascade's quality against the hidden truth on the validation week, and separate retrieval loss from ranking loss;
# 3. put an LRU cache in front of a feature store and reason about its hit rate;
# 4. make the service degrade gracefully (ranker error or timeout → popularity order; missing features → defaults) and test it;
# 5. serve it with FastAPI, load-test it with concurrent clients, and fill in a per-stage latency budget from measurements.
#
# **You will produce:** `artifacts/level2/serving/latency_budget.csv` and
# `artifacts/level2/serving/load_test.csv`.

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
# --- Catch up: the ShopStream logs (regenerated deterministically) ----------
import time

import numpy as np
import pandas as pd

from shoprank import serving as sv
from shoprank import store
from shoprank.grading import nb17 as grade

OUT = ART / "level2" / "serving"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
world, users, items, imps = store.load("small")
print(f"{len(imps):,} impressions, {len(users):,} users, {len(items):,} items  ({time.time() - t0:.0f}s)")
print("train days", store.TRAIN_DAYS, "| valid days", store.VALID_DAYS, "| test days: not used here")

# %% [markdown]
# **Expected:** `2,446,900 impressions, 10,000 users, 4,000 items` (about
# 10–40 s the first time, while the data is generated; a second or two after).

# %% [markdown]
# ## 17.1 · The pieces, built point-in-time
#
# A cascade for one request:
#
# | Stage | In → out | What runs here | Typical budget share |
# |---|---|---|---|
# | Retrieval | catalogue (4,000; in real life 10⁶–10⁹) → ~300 | ANN over embeddings, co-visitation, popularity | 15–25 % |
# | Feature fetch | 300 ids → 300 × F matrix | online feature store lookups (network!) | 20–30 % |
# | Ranker | 300 rows → 300 scores | GBDT or a neural net | 30–40 % |
# | Rules | 300 → 10 | filter, dedupe, diversity, boosts | < 5 % |
#
# Everything the service knows must come from data **before** the day it
# serves. We do it twice, exactly as a production pipeline would:
#
# * **Ranker training:** features from days 0–34, click labels from days 35–41.
#   `position` is a training feature (it explains part of every click) and is
#   fixed to 0 when serving.
# * **Serving state:** features, embeddings and co-visitation refreshed with
#   all train days 0–41, used to serve the validation week 42–48. Counts are
#   normalised *per day*, so the refreshed state has the same feature scale as
#   the one the ranker learned from (a 42-day count is not a 35-day count).
#
# The code is in [`src/shoprank/serving.py`](../../src/shoprank/serving.py):
# `build_state` (SVD embeddings, co-visitation, smoothed CTRs) and
# `train_ranker` (LightGBM on 10 features). Read `build_state` once; it is
# ~80 lines of pandas/NumPy you have written before.

# %%
t0 = time.time()
state_lbl, ranker, train_frame = sv.train_ranker(imps, items, len(users),
                                                 feature_days=(0, 34), label_days=(35, 41))
state = sv.build_state(imps, items, len(users), feature_days=(0, 41))
print(f"ranker trained on {len(train_frame):,} impressions (CTR {train_frame.clicked.mean():.3f}) "
      f"in {time.time() - t0:.1f}s")
print("serving state from days", state.feature_days, "| embeddings", state.item_emb.shape,
      "| co-visitation entries", f"{state.user_covis.nnz:,}")
print(pd.Series(ranker.feature_importances_, sv.FEATURES).sort_values(ascending=False).to_string())

# %% [markdown]
# **Expected:** 300,000 training impressions with CTR 0.138, in 5–30 s
# depending on how busy the CPU is;
# embeddings `(4000, 32)`; about 1.2 M co-visitation entries. Feature
# importances (split counts) spread over all ten features, with
# `user_cat_share`, `item_ctr` and `mf_score` near the top.

# %% [markdown]
# ## 17.2 · One request, stage by stage
#
# `Cascade` wires the pieces together. Its online store simulates a 1 ms
# network round trip per call, a realistic cost for Redis or a feature-store
# API in the same data centre; without it every lookup would look free.
# Here is one validation request, walked through by hand with a stopwatch:

# %%
valid = imps[imps["day"].between(store.VALID_DAYS[0], store.VALID_DAYS[-1])]
cascade = sv.Cascade(state, ranker, store=sv.OnlineStore(state, latency_s=0.001),
                     n_ann=200, n_covis=100, n_pop=100, max_per_category=3)

u, d = int(valid["user_id"].iloc[0]), int(valid["day"].iloc[0])
t = [time.perf_counter()]
user, defaulted = cascade._user(u)
cands, sources = cascade.retrieve(u, d, user)
t.append(time.perf_counter())
X = cascade.features(u, user, cands, d)
t.append(time.perf_counter())
scores = sv.score_with(ranker, X)
order = cands[np.argsort(-scores, kind="stable")]
t.append(time.perf_counter())
top, dropped = sv.business_rules(order, state.item_cat, state.item_created, d, 3, 10)
t.append(time.perf_counter())

print(f"user {u}, day {d}: {len(cands)} candidates from {sources}")
for name, a, b in zip(["retrieve", "features", "rank", "rules"], t[:-1], t[1:]):
    print(f"  {name:<9}{(b - a) * 1000:7.2f} ms")
print("top 10:", top.tolist(), "| dropped by rules:", dropped)

# %% [markdown]
# **Expected:** about 350 candidates (`ann` 200, `covis` up to 100,
# `popular` 100, minus overlaps); each stage a few milliseconds or less
# (the first call is slower: FAISS and LightGBM warm up); a top 10 and a
# count of items the category cap pushed out.
#
# The ANN index is `faiss.IndexFlatIP`: exact inner-product search. With
# 4,000 items exact search costs well under a millisecond; at millions of
# items you would switch to IVF or HNSW (notebook 08) and trade a little
# recall for a lot of latency.

# %% [markdown]
# ## 17.3 · Is the cascade any good? Quality on the validation week
#
# We know the truth: `world.true_click_prob(user, item)` for every item in
# the catalogue. So for each request we can compute **catalogue NDCG@10**:
# the recommended list against the user's true best 10 launched items. 1.0
# is perfect, and the number is not polluted by position bias.
#
# To see *where* quality is lost, score several orderings on the same
# 1,000 validation requests:
#
# * **popularity**: the same top 10 for everyone;
# * **ANN only**: the 10 nearest items by embedding;
# * **candidates, popularity order**: what the service falls back to when the ranker is down;
# * **cascade**: the LightGBM ranker over the candidates;
# * **oracle over candidates**: the candidates sorted by the true probability — the ceiling retrieval allows;
# * **logged policy**: what users were actually shown (not capped).

# %%
req = valid.groupby("request_id", sort=True)[["user_id", "day"]].first().sample(1000, random_state=0)
P = sv.true_prob_matrix(world, req["user_id"].to_numpy(), req["day"].to_numpy())

quiet = sv.Cascade(state, ranker, store=sv.OnlineStore(state, latency_s=0.0))  # no simulated network
cand_lists, score_lists, ann_lists, recall = [], [], [], []
t0 = time.time()
for j, (u, d) in enumerate(zip(req["user_id"], req["day"])):
    user, _ = quiet._user(u)
    c, _ = quiet.retrieve(u, d, user)
    cand_lists.append(c)
    score_lists.append(sv.score_with(ranker, quiet.features(u, user, c, d)))
    if user["emb"] is None:  # user unseen in train days -> default features, no embedding
        ann_lists.append(state.popular[:10])
    else:
        ann_lists.append(quiet.index.search(user["emb"][None, :], 10)[1][0])
    true_top = np.argsort(-P[j])[:10]
    recall.append(np.isin(true_top, c).mean())
print(f"scored {len(req)} requests in {time.time() - t0:.1f}s; "
      f"retrieval recall of the true top-10: {np.mean(recall):.3f} with ~{np.mean([len(c) for c in cand_lists]):.0f} candidates")

shown = valid[valid["request_id"].isin(req.index)].sort_values(["request_id", "position"])
logged_lists = shown.groupby("request_id")["item_id"].apply(np.array).loc[req.index].tolist()


def capped(order, d, cap):
    return sv.business_rules(np.asarray(order), state.item_cat, state.item_created, d, cap, 10)[0]


def orderings(j, d):
    c, s = cand_lists[j], score_lists[j]
    pop = state.popular[: 300]
    return {
        "popularity": pop,
        "ANN only": np.concatenate([ann_lists[j], pop]),
        "candidates, popularity order": c[np.argsort(cascade.pop_rank[c], kind="stable")],
        "cascade (LightGBM)": c[np.argsort(-s, kind="stable")],
        "oracle over candidates": c[np.argsort(-P[j, c], kind="stable")],
    }


rows = {}
for cap in (3, 10):
    lists: dict[str, list] = {}
    for j, d in enumerate(req["day"]):
        for name, order in orderings(j, d).items():
            lists.setdefault(name, []).append(capped(order, d, cap))
    rows[f"cap {cap}/category" if cap < 10 else "no cap"] = {n: sv.catalogue_ndcg(P, L).mean() for n, L in lists.items()}
quality = pd.DataFrame(rows)
quality.loc["logged policy (shown lists)"] = [np.nan, sv.catalogue_ndcg(P, logged_lists).mean()]
quality.round(3)

# %% [markdown]
# **Expected (catalogue NDCG@10, 1,000 validation requests):** retrieval
# recall of the true top 10 about 0.85 with ~320 candidates. Without a cap:
# popularity 0.39, logged policy 0.41, ANN-only 0.79, the LightGBM cascade
# 0.875, the oracle over candidates 0.99. With the cap of 3 per category every
# personalised row drops (ANN-only 0.58, cascade 0.71, oracle 0.85):
# ShopStream users concentrate on 2–3 favourite categories, so a diversity
# rule costs relevance *as this metric measures it*.
#
# Three things to say about this table in an interview:
#
# 1. **Retrieval sets the ceiling.** The oracle row is the best any ranker
#    could do with these candidates; the gap between it and 1.0 is retrieval
#    loss, the gap between it and the cascade is ranking loss.
# 2. **The ranker earns its latency** only if it beats the cheaper rows. Here
#    it beats ANN-only and the popularity fallback by a clear margin.
# 3. **Business rules have a price.** The cap is a product decision (diversity,
#    seller fairness, long-term engagement) that this short-term metric cannot
#    value. Report the cost; don't hide it, and don't let the metric veto it.

# %% [markdown]
# ## 17.4 · Caching
#
# User features change slowly; fetching them costs a network round trip. An
# **LRU (least-recently-used) cache** in the service process keeps the most
# recent rows and evicts the oldest when full. Its value depends entirely on
# the **hit rate**, which depends on how often the same keys come back.
#
# > **Exercise 1.** Implement `LRUCache(capacity)` with `get(key, default=None)`,
# > `put(key, value)`, `len(cache)`, counters `hits` and `misses` (updated by
# > `get`), and a `hit_rate` property (`hits / (hits + misses)`, 0.0 before
# > any `get`). `put` on an existing key updates the value and makes it most
# > recent. Both operations must be **O(1)**: the grader times them at two
# > cache sizes.

# %%
class LRUCache:
    def __init__(self, capacity: int):
        # YOUR CODE HERE
        raise NotImplementedError

    def get(self, key, default=None):
        raise NotImplementedError

    def put(self, key, value) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def hit_rate(self) -> float:
        raise NotImplementedError


grade.ex1(LRUCache)

# %% [markdown]
# The service uses `sv.LRUCache` (same idea, plus a lock for threads). How big
# should the user cache be? Replay one validation day of requests, in time
# order, through caches of different sizes:

# %%
day_requests = (valid[valid["day"] == store.VALID_DAYS[0]]
                .groupby("request_id")[["user_id", "ts"]].first().sort_values("ts"))
hit_rates = {}
for capacity in (100, 500, 2_000, 5_000, 10_000):
    cache = sv.LRUCache(capacity)
    for user_id in day_requests["user_id"]:
        if cache.get(user_id) is None:
            cache.put(user_id, True)
    hit_rates[capacity] = round(cache.hit_rate, 3)
print(f"{len(day_requests):,} requests from {day_requests['user_id'].nunique():,} distinct users")
print("hit rate by capacity:", hit_rates)

# %% [markdown]
# **Expected:** 4,483 requests from 3,177 distinct users; the hit rate climbs
# from 0.015 at 100 entries to 0.29 once the cache holds every user seen that
# day. It cannot go higher: 1 − 3,177 / 4,483 = 0.29 of requests are repeats.
#
# Read it as a capacity-planning curve: memory per row × capacity versus round
# trips saved. And remember the price of caching: **staleness**. A cached row
# is as old as its fill time. Production caches add a TTL (time to live)
# matched to how fast the feature changes — seconds for "items viewed in this
# session", hours or days for "favourite categories".

# %% [markdown]
# ## 17.5 · Business rules
#
# Rules run last, on the ranked list: remove what must never be shown, then
# shape what remains. The order matters: dedupe and eligibility first
# (correctness), diversity second (product), truncation last.
#
# > **Exercise 2.** Write `apply_business_rules(ranked_items, item_meta, max_per_category, k)`.
# > `ranked_items` is a list of item ids, best first (it may contain
# > duplicates). `item_meta[item_id]` is a dict with `"category"` and
# > `"launched"` (bool). Return a list of at most `k` ids that keeps rank
# > order and:
# >
# > 1. drops repeated ids (keep the first occurrence) and ids that are missing from `item_meta` or not launched;
# > 2. allows at most `max_per_category` items per category;
# > 3. if the cap leaves fewer than `k` items, **backfills** with the highest-ranked items that were skipped *only* because of the cap, appended in rank order.

# %%
def apply_business_rules(ranked_items, item_meta, max_per_category, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(apply_business_rules)

# %% [markdown]
# What does the cap buy? Count distinct categories in each cascade top 10:

# %%
for cap in (2, 3, 5, 10):
    tops = [capped(cand_lists[j][np.argsort(-score_lists[j], kind="stable")], d, cap)
            for j, d in enumerate(req["day"])]
    n_cat = np.mean([len(set(state.item_cat[t])) for t in tops])
    print(f"cap {cap:>2}: catalogue NDCG@10 {sv.catalogue_ndcg(P, tops).mean():.3f} | "
          f"distinct categories in top 10: {n_cat:.1f}")

# %% [markdown]
# **Expected:** NDCG falls as the cap tightens (0.875 with a cap of 10, i.e.
# none, down to 0.65 at a cap of 2) while the distinct categories in a top 10
# rise from 1.3 to 5.6. Without a cap, most users see a single category. That
# is the trade-off a product team must choose on, ideally with an online
# experiment measuring long-term engagement (notebooks 11–12), not this
# offline metric alone.

# %% [markdown]
# ## 17.6 · Graceful degradation
#
# A ranking service should *always* return something sensible. The usual
# ladder:
#
# | Failure | Fallback | Cost |
# |---|---|---|
# | Ranker errors or exceeds its timeout | candidates in popularity order | relevance |
# | User row missing (new user, store down) | default features, popularity + co-visitation retrieval | personalisation |
# | Retrieval fails | cached popular list | everything but availability |
#
# The mechanism is a call with a timeout and a fallback.
#
# > **Exercise 3.** Write `with_fallback(primary_fn, fallback_fn, timeout_s)`
# > that returns a **new function**. Calling it with any `*args, **kwargs`
# > runs `primary_fn(*args, **kwargs)` and returns `(result, "ok")`; if
# > `primary_fn` raises, it returns `(fallback_fn(*args, **kwargs), "error")`;
# > if it takes longer than `timeout_s` seconds, it returns
# > `(fallback_fn(*args, **kwargs), "timeout")` **without waiting** for it to
# > finish. Hint: `concurrent.futures`.

# %%
def with_fallback(primary_fn, fallback_fn, timeout_s):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(with_fallback)

# %% [markdown]
# `Cascade` uses the module's version, `sv.call_with_timeout`, around the
# ranker (timeout 50 ms), and has fault switches for testing. Break each
# dependency on purpose and check what users would get:

# %%
def run_mode(fault: dict, n: int = 150) -> dict:
    c = sv.Cascade(state, ranker, store=sv.OnlineStore(state, latency_s=0.001), rank_timeout_s=0.05)
    c.faults.update(fault)
    outs = [c.recommend(u, d) for u, d in zip(req["user_id"][:n], req["day"][:n])]
    total = [o["timings_ms"]["total"] for o in outs]
    return {"ranker status": pd.Series([o["ranker"] for o in outs]).value_counts().to_dict(),
            "user features": pd.Series([o["user_features"] for o in outs]).value_counts().to_dict(),
            "p50 ms": round(float(np.median(total)), 1), "max ms": round(float(np.max(total)), 1),
            "catalogue NDCG@10": round(float(sv.catalogue_ndcg(P[:n], [o["items"] for o in outs]).mean()), 3)}


modes = {"healthy": {}, "ranker raises": {"ranker_error": True},
         "ranker takes 200 ms": {"ranker_sleep_s": 0.2}, "feature store down": {"store_down": True}}
degradation = pd.DataFrame({name: run_mode(f, n=40 if "200" in name else 150) for name, f in modes.items()}).T
degradation

# %% [markdown]
# **Expected:** healthy: all `ok` (3 of 150 users have no train-days row and
# get defaults), p50 about 5 ms, NDCG 0.72 (cap 3). Ranker raises: all
# `error`, *faster* (no model), NDCG 0.38 — the popularity fallback. Ranker
# takes 200 ms: all `timeout`, p50 about 54 ms (the 50 ms timeout, not the
# 200 ms), the same fallback quality (0.40). Store down: every user on
# `default` features, ranker still `ok`, NDCG 0.45: worse than healthy, better
# than popularity, because co-visitation from recent clicks still works.
#
# Two things the table hides. First, a timed-out ranker call **keeps running**
# in its thread: the request was protected, the CPU was not. Under real
# overload, timeouts can make things worse (retries, piled-up work), which is
# why services also *shed load* and shrink candidate sets. Second, every
# fallback should be **counted** (`/metrics` below) and alerted on: a service
# that silently serves popularity for a week has an outage nobody sees.

# %% [markdown]
# ## 17.7 · The service, under load
#
# `sv.create_app(cascade)` is an app factory in the style of Level 1's
# `churnlab.api` (read it next to [`src/churnlab/api.py`](../../src/churnlab/api.py)):
# `POST /recommend` returns the items plus a `timings_ms` debug field and a
# W3C `Server-Timing` header; `GET /metrics` gives per-stage percentiles,
# fallback counters and cache hit rates. `sv.BackgroundServer` runs uvicorn
# in a daemon thread on a free localhost port so the notebook can call it.

# %%
import httpx

cascade = sv.Cascade(state, ranker, store=sv.OnlineStore(state, latency_s=0.001), max_per_category=3)
app = sv.create_app(cascade)
server = sv.BackgroundServer(app).start()
print("serving at", server.url)

with httpx.Client(base_url=server.url, timeout=5, trust_env=False) as client:
    r = client.post("/recommend", json={"user_id": int(req["user_id"].iloc[0]), "day": int(req["day"].iloc[0])})
    print(r.status_code, r.json()["items"])
    print("server-timing:", r.headers["server-timing"])
    print("x-ranker-status:", r.headers["x-ranker-status"])
    print("bad request ->", client.post("/recommend", json={"user_id": -1, "day": 42}).status_code)

# %% [markdown]
# **Expected:** `200`, ten item ids, a `server-timing` header such as
# `retrieve;dur=1.6, features;dur=1.3, rank;dur=1.9, rules;dur=0.1, total;dur=4.9`,
# `x-ranker-status: ok`, and `422` for the invalid user id (Pydantic validation).
#
# Now a load test: replay validation requests from several client threads,
# each with its own `httpx.Client` (connection reuse), and record client-side
# latency. `trust_env=False` keeps httpx from routing localhost through any
# proxy configured in the environment.

# %%
from concurrent.futures import ThreadPoolExecutor

bodies = [{"user_id": int(u), "day": int(d)}
          for u, d in valid.groupby("request_id")[["user_id", "day"]].first().head(3_000).to_numpy()]


def load_test(url: str, bodies: list[dict], concurrency: int) -> dict:
    def worker(chunk):
        out = []
        with httpx.Client(base_url=url, timeout=5, trust_env=False) as client:
            for body in chunk:
                t = time.perf_counter()
                r = client.post("/recommend", json=body)
                ms = (time.perf_counter() - t) * 1000
                out.append((ms, r.status_code, r.json()["timings_ms"] if r.status_code == 200 else {}))
        return out

    chunks = [bodies[i::concurrency] for i in range(concurrency)]
    start = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        results = [x for part in pool.map(worker, chunks) for x in part]
    wall = time.perf_counter() - start
    ms = np.array([r[0] for r in results])
    stages = pd.DataFrame([r[2] for r in results])  # server-side timings, from each response
    return {"concurrency": concurrency, "requests": len(ms),
            "errors": sum(r[1] != 200 for r in results),
            "p50 ms": np.percentile(ms, 50), "p95 ms": np.percentile(ms, 95),
            "p99 ms": np.percentile(ms, 99), "throughput rps": len(ms) / wall,
            **{f"server {s} p99": stages[s].quantile(0.99) for s in stages.columns}}


runs = [load_test(server.url, bodies[i * 600:(i + 1) * 600], c) for i, c in enumerate((1, 4, 16))]
load = pd.DataFrame(runs).set_index("concurrency").round(1)
load.to_csv(OUT / "load_test.csv")
load.T

# %% [markdown]
# **Expected (our runs: 4 shared cores, 2 NumPy threads):** no errors. At
# concurrency 1, client p50 about 6.5 ms, p99 about 13 ms, ~140 requests/s.
# At 4, throughput rises to ~235/s (the simulated 1 ms store waits overlap)
# and p99 grows to 26–31 ms. At 16 the service is **saturated**: throughput
# *falls* (~210–215/s), p50 ≈ 60–66 ms, p99 ≈ 100–125 ms across our runs, and
# even the server-side stage p99s inflate 3–4× because the stages now fight
# for the CPU. Your numbers
# will differ (Colab CPUs are slower and noisier); the shape will not.
#
# Be honest about what this measures. Client and server share **one Python
# process and one GIL**, so the clients steal CPU from the server; a real load
# test runs the load generator on other machines (`locust`, `k6`, `wrk`).
# Treat these numbers as *relative* (stage vs stage, concurrency vs
# concurrency), not as capacity.

# %%
with httpx.Client(base_url=server.url, timeout=5, trust_env=False) as client:
    m = client.get("/metrics").json()
server.stop()
print("server stopped:", not server.thread.is_alive())
print("counters:", m["counters"])
print("user cache hit rate:", m["user_cache_hit_rate"], "| popular cache hit rate:", m["popular_cache_hit_rate"])
stage_lat = pd.DataFrame(m["latency_ms"]).T
stage_lat

# %% [markdown]
# **Expected:** `server stopped: True`; 1,801 requests (one, plus 3 × 600),
# all `ranker_ok` — in one of our runs a single `ranker_timeout` appeared
# during the concurrency-16 burst, the 50 ms timeout firing under CPU
# contention, which is exactly what it is for. User-cache hit rate about 0.13
# (these requests come mostly from different users; compare the 17.4 curve);
# popular-cache hit rate 0.999 (one entry per day). Server-side p50s of a few
# milliseconds per stage.
#
# Note the gap between client p99 and server `total` p99: queueing in the
# server's thread pool, HTTP parsing, JSON encoding and the GIL live *outside*
# the stages you instrumented. A budget that ignores them will be missed.

# %% [markdown]
# ## 17.8 · The latency budget
#
# A budget splits the SLO across stages *before* anyone builds anything, and
# leaves slack for what you did not instrument. For a **p99 ≤ 100 ms**
# end-to-end SLO on this service:

# %%
stages = ["retrieve", "features", "rank", "rules", "total"]
budget = pd.DataFrame({"budget p99 ms": [20, 20, 30, 5, 75, 25, 100]},
                      index=[*stages, "outside the stages (HTTP, queueing, GIL)", "end to end (client)"])
for c in load.index:
    measured = [load.loc[c, f"server {s} p99"] for s in stages]
    outside = load.loc[c, "p99 ms"] - load.loc[c, "server total p99"]
    budget[f"measured p99 @ concurrency {c}"] = [*measured, outside, load.loc[c, "p99 ms"]]
budget.round(1).to_csv(OUT / "latency_budget.csv")
ok = budget.iloc[:, 1:].le(budget["budget p99 ms"], axis=0)
print("within budget at each concurrency:", ok.all().to_dict())
budget.round(1)

# %% [markdown]
# **Expected:** within budget at concurrency 1 and 4 (client p99 about 13 and
# 26–31 ms). At concurrency 16 the budget is blown: the end-to-end p99 sits
# at or above the 100 ms SLO (98–125 ms across our runs), the ranker stage
# reaches or exceeds its 30 ms share, and the biggest single item is
# *outside* the stages (45–65 ms of queueing, HTTP and GIL). The fix for that
# row is not a faster model: it is more replicas (or processes) and load
# shedding, so each one runs below saturation. Your machine may differ; the
# table is the habit, not the numbers.
#
# Rules of thumb for budgets:
#
# * Percentiles **do not add**: the p99 of a sum is less than the sum of p99s,
#   so per-stage p99 budgets summing to the SLO are conservative (good).
# * Fan-out makes tails worse: a request that waits for 10 parallel shards is
#   as slow as the slowest; at p99 per shard, about 10 % of requests hit at
#   least one p99 shard. Hedged requests and timeouts per shard are the fix.
# * Spend the budget where it buys quality: the ranker, usually. Retrieval
#   and features should be boring and fast.
#
# > **Exercise 4 (written).** Your team's home-feed service has a **200 ms
# > p99** SLO measured at the client. The ranker is a deep model on a GPU
# > pool behind its own RPC; features come from a remote feature store; ANN
# > retrieval is a separate service with 3 shards queried in parallel. Write
# > a per-stage p99 budget (network, retrieval, feature fetch, ranker,
# > rules, slack), justify each number in one line, say which stage you would
# > cut first if a new model needs 40 ms more, and what fallback each stage
# > has when it blows its budget.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `RuntimeError: uvicorn did not start` | port taken, or a previous server still running in this kernel | `server.stop()` the old one; `sv.free_port()` picks a new port; restart the runtime if stuck |
# | `httpx.ConnectError` / proxy errors on `127.0.0.1` | an `HTTP(S)_PROXY` variable routes localhost through a proxy | pass `trust_env=False` to `httpx.Client` (as here) |
# | Every response says `"ranker": "timeout"` | `rank_timeout_s` too small for this CPU (Colab can be 2–3× slower) or a fault left on | raise the timeout to 0.1–0.2 s; check `cascade.faults` |
# | Load-test p99 jumps wildly between runs | shared CPU (Colab, other notebooks), GIL shared by clients and server | repeat runs; compare relative numbers; run the load generator elsewhere for real capacity |
# | Catalogue NDCG near 0.4 for the "cascade" | ranker silently falling back (counters show `ranker_error`) | check `cascade.counters` and `x-ranker-status`; a fallback should be visible, not silent |
# | `ImportError: faiss` | `faiss-cpu` not installed | re-run the setup cell (installs `level2/requirements.txt`) |
# | Exercise 1 says "not O(1)" | list-based recency (`list.remove`) is O(n) | `collections.OrderedDict` with `move_to_end` / `popitem(last=False)` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Walk through a recommendation cascade for a home feed. What does each stage take in and emit, and roughly how many milliseconds and candidates would you give it at 100M users?
# 2. Your offline NDCG went up but the cascade's quality online did not. Using this notebook's oracle-over-candidates row, explain how to tell whether retrieval or ranking is the bottleneck.
# 3. How would you make a ranking service degrade gracefully? What do you return when the ranker times out, when features are missing, and when retrieval fails, and how do you know it happened?
# 4. Where would you cache in this service, what is the hit rate likely to be, and what goes wrong with caching user features?
# 5. Your p99 latency doubled after a deploy but p50 did not move. List likely causes and how you would find the culprit stage.
# 6. Product asks for "at most 3 items per category". What does it cost, how would you measure whether it is worth it, and where in the cascade does it belong?
#
# ➡️ **Next:** [`18_model_optimization_for_inference.ipynb`](18_model_optimization_for_inference.ipynb) — make the ranker itself cheaper: ONNX, quantisation and distillation, with accuracy measured against latency.
