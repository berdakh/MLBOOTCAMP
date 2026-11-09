# Capstone 4 · Movie recommendations (MovieLens)

*Top-10 movie recommendations on MovieLens, evaluated in time.*

**Recommended order: 4th** · MVP ≈ 4–5 days at 3 h/day · CPU only

**Guided solution:** [walkthrough.ipynb](walkthrough.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap4-movie-recommendations/walkthrough.ipynb)
— a complete worked solution of the MVP below (framing, download and audit,
time-aware split with leakage checks, popularity baselines, item-kNN, EASE and
implicit ALS, bootstrap confidence intervals, cold-start slices, a tested
FastAPI service, and a results table generated from the run). Attempt each
step yourself first. The walkthrough was verified on the course build machine
only with its synthetic offline sample (that machine cannot reach
`files.grouplens.org`); the real-data run has not been executed there.

This is a public-data demonstration inspired by recommendation and
feed-ranking problems on large platforms. It uses the MovieLens research
datasets published by GroupLens. It is not work for, or endorsed by, Meta,
Amazon, Netflix or any other platform, and because of the dataset's licence it
must remain a **non-commercial** demonstration.


---

## Business question

*For each returning user, which 10 movies they have not rated yet should we
show next, so that they are likely to engage with at least one?*

**Who acts on it:** the owner of a "recommended for you" surface (a product
manager and ranking team). They decide which model ranks the slate, what to
show users with almost no history, and whether a new model is worth an online
A/B test. Offline metrics here are a **screening** tool: they decide what is
worth testing online, not what wins online.

## Why this project

* **Ranking metrics** (Recall@K, NDCG@K) and how they differ from rating
  prediction (RMSE), a standard interview question.
* **A time-aware split** that does not let the model see the future, and a
  demonstration that random splits inflate scores.
* **A popularity baseline that is hard to beat**, and the discipline of
  reporting it.
* **Cold start** for users and items, and fallbacks in serving.
* A **two-stage serving design** (candidate generation + ranking,
  precomputed vs online) you can discuss in system-design interviews.

## Data

| Fact | Status |
|---|---|
| Index of all releases: <https://files.grouplens.org/datasets/movielens/> | **to verify** (path from research notes; open it) |
| Per-dataset README pages, e.g. <https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html> | **verified from an official page snippet** |
| File names `ml-latest-small.zip`, `ml-25m.zip`, `ml-32m.zip` | **to verify** |
| **ml-latest-small:** 100,836 ratings, 3,683 tag applications, 9,742 movies, 610 users, ratings from 1996-03-29 to 2018-09-24 | **verified from an official page snippet** |
| ml-latest-small's README says it **may change over time and is not appropriate for shared research results** | **verified from an official page snippet** |
| **ml-32m:** 32,000,204 ratings, 87,585 movies, 200,948 users, released 05/2024 | **verified from an official page snippet** |
| ml-25m: about 25 million ratings and about 162,000 users | **verified from an official page snippet** (rounded figures) |
| Ratings on a 0.5–5 star scale in half-star steps; timestamps in Unix seconds (UTC) | **to verify** |
| Every included user has rated at least 20 movies | **to verify** (stated in READMEs per release; check the one you use) |
| Files `ratings.csv`, `movies.csv` (with pipe-separated genres), `tags.csv`, `links.csv` | **to verify** |

**Which release?** Develop on **ml-latest-small** (fast iteration on a laptop
or free Colab). Because its README says it is not appropriate for shared
research results, **report your final results on a fixed, versioned release**
— e.g. ml-32m restricted to a documented random sample of users (state the
seed and sample size), or a smaller stable release if you verify one exists
on the index page. Record the exact file and checksum.

**Licence / usage terms** — **read the README of the release you download.**
The conditions summarised here come from a third-party copy of the README
(the official snippet was truncated), so they are **to verify**:

* the data may be used for research purposes, under conditions;
* no endorsement by the University of Minnesota or GroupLens may be stated or
  implied;
* you must acknowledge the use of the dataset in publications (cite below);
* you may redistribute or derive from it only under the same licence terms;
* **no commercial or revenue-bearing use without separate permission.**

Practical consequences: keep this a non-commercial portfolio project; do not
commit the raw files to your repository (download them in `make data`); do
not deploy the API as part of a product.

**Citation** (**verified from an official page snippet**):

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
> History and Context. *ACM Transactions on Interactive Intelligent Systems
> (TiiS)* 5, 4, Article 19. DOI: 10.1145/2827872

**Known gotchas** (**to verify** on your copy):

* **Ratings are bursty:** many users rate dozens of movies in their first
  session (onboarding), so timestamps reflect *rating time*, not viewing
  time, and a user's "last" ratings may be from the same minute.
* **Old movies rated recently**: release year and rating time differ widely.
* **Ties in timestamps** within a user: break them deterministically (e.g.
  by movieId) and document it.
* `movies.csv` titles include the year in parentheses; genres are
  pipe-separated with a "no genres listed" value.
* The larger releases do not fit comfortably in free-Colab memory as a dense
  matrix; use sparse matrices (`scipy.sparse`) and a user sample.

## Problem framing

* **Unit of prediction:** a (user, request time) pair → a ranked list of 10
  movie IDs.
* **Feedback definition:** two defensible choices. (a) **Any rating is an
  interaction** (implicit feedback: "the user chose to engage with this
  movie"). (b) **Relevant = rating ≥ 4.0** ("the user engaged and liked it").
  Report both; they answer different questions. The guided solution uses (b)
  as the primary label, because recommending movies users rate 1–2 stars is
  not success, and reports (a) as a secondary analysis.
* **Candidates:** all movies that existed in the data before the request
  time and that the user has not already rated.
* **Label:** the held-out interaction(s) of that user (see protocol).
* **Horizon:** the user's next interaction(s) after the request time.

## Baselines

1. **Global popularity:** most-interacted movies in the training data, minus
   those the user already rated.
2. **Recent popularity:** most-interacted in the last N days before the
   request time (N chosen on validation). Often much stronger than (1).
3. **Item-item co-occurrence / item-kNN** on the user's recent history (cosine
   on the binary interaction matrix).

## Improvements

* **EASE** (Steck, 2019: a closed-form linear item-item model;
  `P = (XᵀX + λI)⁻¹`, `B_ij = −P_ij / P_jj` off the diagonal and
  `B_ii = 0`; check against the paper) — a strong, CPU-friendly
  model when the item count is moderate. Filter to items with ≥ k training
  interactions so the item × item inverse fits in memory (estimate the matrix
  size before you run it).
* **Implicit matrix factorisation** (ALS, e.g. the `implicit` library, or a
  NumPy implementation), or **BPR**.
* **Hybrid for cold start:** genre/tag-based content similarity for items
  with too few interactions; popularity fallback for users with < k
  training interactions.
* **Extension:** a two-stage design — candidate generation (EASE/ALS top-200)
  then a learned ranker (gradient boosting on user, item and interaction
  features); a sequence model (GRU4Rec/SASRec-style) — **GPU recommended** for
  the larger releases.

## Evaluation protocol

**Time-aware leave-last-out (primary).**

* Sort each user's interactions by timestamp (ties broken deterministically,
  e.g. by `movieId`).
* **Test:** each user's **last** interaction. **Validation:** the
  second-to-last. **Training:** everything earlier.
* **Make it time-aware, not just per-user:** a plain leave-last-out lets the
  model train on *other* users' interactions that happened after this user's
  test time (global future leakage). Mitigate and measure:
  1. restrict each user's candidate set to items whose first interaction in
     the data precedes the test timestamp;
  2. compute popularity baselines using only interactions before the test
     timestamp (or before a global cut-off);
  3. **robustness check:** a **global temporal split** — train on everything
     before time *T*, test on users' first interaction after *T* — and report
     both protocols. If they disagree, say which you trust and why.
* **Model selection** on validation (second-to-last); test touched **once**.
* **Full ranking:** rank against **all** candidate items, not 100 sampled
  negatives. Sampled metrics can change model rankings (Krichene & Rendle,
  "On Sampled Metrics for Item Recommendation", KDD 2020).
* **Uncertainty:** bootstrap over users for 95% CIs; paired comparison with the
  popularity baseline on the same users.

**Metrics.**

* **Recall@10** (hit rate with one held-out item): did the held-out movie
  appear in the top 10?
* **NDCG@10**: like Recall@10 but rewards ranking it higher.
* **Coverage** (share of catalogue ever recommended) and **average
  popularity** of recommended items — a model can win on Recall by
  recommending blockbusters to everyone.
* **Slices:**
  * **cold-start users:** users with < 5 (or < 10) training interactions —
    on releases where every user has ≥ 20 ratings, simulate this by
    truncating some users' histories to their first k interactions;
  * **cold / long-tail items:** held-out items with < k training interactions
    (CF models cannot recommend never-seen items; report how often that
    happens);
  * by user activity decile.

## Leakage traps

* **Random split of ratings**: the model trains on interactions from after the
  test interaction. Show the inflated score once and explain it.
* **Global popularity computed over all data** including the test period.
* **Recommending items that did not exist yet** at the test time.
* **Not filtering already-rated items**, or filtering them only in evaluation
  but not in serving (training/serving skew).
* **Using the rating value of the test interaction** as a feature (e.g.
  "user's mean rating" computed including it).
* **Tuning λ, factors or N on the test interaction.**
* **Tag features computed with tags applied after the test time.**

## MVP (≈ 4–5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repo from template; read the release README and licence; `make data` with checksum; load into DuckDB/pandas; EDA: interactions per user/item, timestamp bursts, ties; split builder (leave-last-out + global temporal) with **leakage tests** |
| 2 | Metrics (Recall@10, NDCG@10, coverage, popularity) unit-tested on hand-built cases; popularity, recent-popularity and item-kNN baselines; full-ranking evaluator with already-rated filtering |
| 3 | EASE (or ALS) on ml-latest-small; tune on validation; cold-start fallback; random-vs-time split comparison |
| 4 | Final runs on the fixed reporting release (user sample) once; bootstrap CIs; slices (cold users/items, activity deciles) |
| 5 (buffer) | API with precomputed top-N and fallback; README, model card, data card; licence statement |

**Extensions** (optional):

* Content-based model on genres/tags for cold items (CPU).
* Two-stage retrieval + GBM ranker (CPU on a user sample).
* Diversity / novelty-aware re-ranking and its trade-off with NDCG.
* Sequence models on the large releases — **GPU recommended**.

## API design

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model and precomputed tables loaded; returns `model_version`, `trained_through` |
| GET | `/recommendations/{user_id}?k=10` | top-k for a known user |
| POST | `/recommendations` | top-k with request-time context (recent items, exclusions) |
| GET | `/similar/{movie_id}?k=10` | item-to-item neighbours |

`POST /recommendations` request:

```json
{"user_id": 4211, "k": 10, "recent_movie_ids": [318, 296], "exclude_movie_ids": [2571]}
```

Response:

```json
{
  "model_version": "v4",
  "user_id": 4211,
  "strategy": "ease",
  "items": [{"movie_id": 1196, "score": 0.81, "rank": 1}]
}
```

(IDs and scores are illustrative placeholders, not model output.)

**Validation rules:** `k` in 1–100; `user_id` a positive integer;
`recent_movie_ids` ≤ 200 known movie IDs (unknown IDs are ignored and
reported, not an error); no duplicates; `exclude_movie_ids` are always
honoured, as are already-rated items. Unknown or cold users get
`"strategy": "popularity_fallback"` — the response must say which path served
it.

**Precomputed vs online:** precompute nightly (batch) the top-200 candidates
per known user, the item-item similarity top-k, and recent popularity.
Online: merge request-time `recent_movie_ids` (e.g. item-kNN on them),
filter exclusions and already-rated items, re-rank, truncate to `k`. Explain
the trade-off: freshness vs latency vs cost.

## Limitations

* **Ratings ≠ consumption.** A rating is an explicit act by a subset of
  viewers, often long after watching; non-raters and abandoned views are
  invisible. Offline Recall predicts "what will they rate next", not "what
  will they enjoy or watch".
* **Survivorship / selection:** releases include only users meeting a minimum
  number of ratings (**to verify** per release); casual users — the cold-start
  problem you care about most — are under-represented.
* **Onboarding bursts** make "next interaction" partly an artefact of the
  MovieLens sign-up flow.
* **Offline ≠ online:** popularity bias, position bias and feedback loops are
  not captured; a real launch needs an A/B test.
* **Licence:** research use only, **no commercial use without permission**
  — the API is a demonstration and must not be deployed as a product.
* ml-latest-small changes over time; results on it are not reproducible
  against other people's numbers.

## Interview questions

1. **Why Recall@10/NDCG@10 instead of RMSE?** Strong answer: the product
   shows a ranked slate; RMSE rewards predicting ratings of items users chose
   to rate; ranking metrics evaluate the decision actually made.
2. **Describe your split. How could it still leak?** Covers: leave-last-out
   per user, global-time leakage from other users, item-existence filter,
   global temporal split as a robustness check.
3. **How much did you beat popularity by, and is that meaningful?** Covers:
   paired bootstrap over users; recent-popularity as the strong baseline;
   coverage and average popularity of recommendations.
4. **Why full ranking instead of sampled negatives?** Covers: sampled
   metrics are biased and can reorder models (Krichene & Rendle 2020).
5. **How do you handle a brand-new user? A brand-new movie?** Covers:
   popularity/onboarding fallback; content features for items; exploration;
   measured slices.
6. **Explain EASE (or ALS) in two minutes. When would it fail?** Covers:
   item-item linear model with closed form; item-count memory; no side
   features; no sequence.
7. **Design the serving system for 10 million users.** Covers: batch
   candidates, online re-ranking, feature freshness, caching, latency budget,
   fallbacks, logging for training.
8. **Offline NDCG improved but the A/B test was flat. Why?** Covers: proxy
   label (rating ≠ watch), popularity bias, position bias, novelty, metric
   mismatch with the business goal.
9. **What feedback loops does a recommender create?** Covers: rich-get-richer,
   training on own exposures, exploration, counterfactual evaluation (IPS).
10. **What does the MovieLens licence let you do with this project?**
    Covers: research/non-commercial use, citation, no endorsement, no
    redistribution outside the licence.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `movierec`)
- [ ] `DATA_CARD.md`: release used, checksum, README licence conditions quoted, citation, user sample seed
- [ ] Split builder with tests: no training interaction after the user's test time for that user; candidate items existed before test time
- [ ] Metrics unit-tested on hand-computed examples
- [ ] Popularity, recent-popularity, item-kNN baselines and at least one model on identical users
- [ ] Results table: Recall@10, NDCG@10, coverage, mean popularity, with CIs — **TBD — fill from your run**
  (step 10 of the walkthrough generates this table from the computed numbers)
- [ ] Random vs time-aware split comparison — **TBD — fill from your run**
- [ ] Cold-start user/item slices
- [ ] API with precomputed candidates, fallback strategy reported in responses, validation tests
- [ ] Limitations: ratings ≠ consumption, survivorship, non-commercial licence
- [ ] CI green on a tiny synthetic interactions fixture (no download)

## Results

Filled by step 10 of the [walkthrough](walkthrough.ipynb), which generates the
markdown from the real-data run and saves it to
`artifacts/cap4/results_table.md`. Never copy numbers printed in the synthetic
sample mode.

| Model | Recall@10 (95% CI) | NDCG@10 (95% CI) | NDCG@10 − recent popularity | Coverage | Mean popularity |
|---|---|---|---|---|---|
| Global popularity | TBD — fill from your run | TBD — fill from your run | TBD — fill from your run | TBD | TBD |
| Recent popularity | TBD — fill from your run | TBD — fill from your run | — | TBD | TBD |
| Item-kNN | TBD — fill from your run | TBD — fill from your run | TBD — fill from your run | TBD | TBD |
| EASE / ALS | TBD — fill from your run | TBD — fill from your run | TBD — fill from your run | TBD | TBD |

Split: time-aware leave-last-out (test = each user's last positive); release,
checksum and user sample: TBD — fill from your run.

## Related chapters

[Chapter 5](../../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md) (leakage-safe splits),
[Chapter 23](../../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md) (implicit feedback),
[Chapter 25](../../part2-ml-at-scale/ch25-candidate-generation/README.md) (candidate generation),
[Chapter 24](../../part2-ml-at-scale/ch24-learning-to-rank/README.md) (learning to rank, for the re-ranker extension) and
[Chapter 37](../../part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb) (recommender system design).

## Stretch: connect to the course's churnlab patterns

* **Registry:** version the model *and* its precomputed tables together, so a
  promotion swaps both atomically and a rollback restores both —
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py).
* **Monitoring:** log every served slate (user, items, strategy, model
  version); monitor the share of fallback responses, catalogue coverage and
  the popularity distribution of served items (PSI against the validation
  slates, as in
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py)).
  Explain why online engagement labels are delayed and biased by exposure.
