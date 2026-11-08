# Capstones

Five independent portfolio projects on real public data. Each one takes the
habits from the five-day bootcamp (checksummed data, leakage-safe splits, an
honest baseline, a tested API, a model card) and applies them to a problem
with a different shape: run-to-failure sensor data, an hourly time series,
messy transactions, a ranking problem and a text classifier with a human in
the loop.


---

## Read this first: the honesty rule

These are **public-data demonstrations inspired by** energy, industrial,
retail and social-platform problems. They are **not** work done for, with, or
endorsed by ERCOT, NASA, any energy company, Amazon, Meta, Netflix, Google,
Jigsaw or any other organisation, and your README must not suggest otherwise.

* **Never invent results.** Every number in a README, model card, slide or
  interview answer comes from a run of your code, on a split you name. If you
  have not run it, the cell says *TBD — fill from your run*.
* **Never invent business impact.** "Reduced forecast error by X% versus a
  seasonal-naive baseline on 2023 rolling-origin folds" is a measurement.
  "Would save $2M a year" is a fabrication, unless you build and state a
  cost model and label it as an assumption.
* **Results tables start empty.** The template ships with `TBD — fill from
  your run` in every results cell. Do not copy numbers from papers or other
  repositories into them; cite a paper's number only in a separate, clearly
  labelled "published reference points" note, and only if you have read it.
* **Data terms are the publisher's, not ours.** The briefs were written on a
  machine that could not reach the dataset hosts. Facts are marked
  **verified from an official page snippet** (seen in a search-engine extract
  of the official page) or **to verify**. Either way, **open the official page
  yourself and read the licence or terms before you download**, and quote the
  key condition in your `DATA_CARD.md`.

---

## The five capstones

The numbering is **by domain** (it matches the table in the repository
README), not the order to do them in. The recommended order is below.

| # | Capstone | Core skill it proves | Recommended order | MVP budget at 3 h/day |
|---|---|---|---|---|
| 1 | [ERCOT hourly load forecasting](cap2-electricity-load-forecasting/README.md) | Rolling-origin backtests, seasonal baselines, peak-hour error | 2nd | 4–5 days (12–15 h) |
| 2 | [C-MAPSS predictive maintenance](cap1-predictive-maintenance/README.md) | Engine-level splits, RUL metrics, a compact PyTorch temporal model | **1st** | 4 days (12 h) |
| 3 | [Online Retail demand forecasting](cap3-retail-demand-forecasting/README.md) | SQL cleaning of messy transactions, intermittent demand, sales ≠ demand | 3rd | 5 days (15 h) |
| 4 | [MovieLens recommendations](cap4-movie-recommendations/README.md) | Time-aware ranking evaluation, Recall@10 / NDCG@10, cold start | 4th | 4–5 days (12–15 h) |
| 5 | [Civil Comments moderation triage](cap5-content-moderation/README.md) | PR-AUC, thresholds under review capacity, subgroup error rates | 5th | 5 days (15 h) |

A copyable repository layout for every capstone (package skeleton, FastAPI
app factory, tests, CI, README / model card / data card templates) is in
[`capstones/template/`](https://github.com/berdakh/MLBOOTCAMP/tree/main/part4-capstones/template).

---

## Recommended order, and why

**1. C-MAPSS first.** It is the closest to a signal-processing / EEG
background: multichannel sensor time series, windowing, per-channel
normalisation, and — most importantly — the evaluation looks like
cross-subject evaluation. Engines are subjects: if windows from the same
engine land in both train and validation, the score is inflated in exactly the
way a within-subject BCI score is. You already have the instinct; the project
turns it into an engineering artifact (package, API, model card). The data is
small, clean-ish and CPU-friendly, so it is a quick early win that sets up
your template repository for the rest.

**2. ERCOT second.** Still a numeric time series, but now the hard part is
the *protocol*: rolling-origin backtests, forecasts issued at a fixed time
with only the data available then, daylight-saving days, and an error metric
that cares about peaks. It is also the most relevant capstone for Houston
energy employers. Doing it after C-MAPSS means your template and test habits
are already in place and the new ideas are purely about time.

**3. Online Retail third.** The first project where the data fights back:
cancellations, non-product stock codes, missing customer IDs, wholesale
orders, and the fact that recorded sales are not demand. It is SQL-heavy
(DuckDB is a natural fit, as in the bootcamp's notebook 03) and introduces
intermittent demand. Doing it after ERCOT lets you reuse backtesting while
learning data cleaning under ambiguity.

**4. MovieLens fourth.** A change of problem type: ranking instead of
regression. New metrics (Recall@10, NDCG@10), a strong popularity baseline
that many published models struggle to beat, and cold-start users and items.
The time-aware split is a direct continuation of ERCOT and Retail.

**5. Civil Comments last.** The most different and the most open-ended: text,
a pretrained model (GPU optional), heavy label noise, threshold selection
under a human-review capacity constraint, and subgroup error analysis with
identity labels. It builds on the bootcamp's thresholds and calibration day
and is the strongest material for "responsible ML" and product-sense
questions, which are easier to answer well after four projects.

**When to deviate.** If you are targeting a specific role, move the matching
capstone earlier: recommender / ads / feed roles → MovieLens second; trust &
safety / NLP roles → Civil Comments second; supply-chain / retail science →
Online Retail second. Keep C-MAPSS first unless you already have a polished
portfolio repository: it is the fastest way to build the template you will
reuse.

---

## Time budget, honestly

A month at 3 h/day is about 90 hours. The bootcamp takes the first five days
(~15 h). That leaves ~75 h, and the five MVPs above add up to roughly 66–72 h
**if nothing goes wrong**, before any extensions and before interview
practice. Things will go wrong (a download, a time-zone bug, a long training
run). So:

* **Plan for three polished capstones, and treat four and five as a bonus.**
  Three projects you can defend line by line are worth more than five
  half-finished ones. A sensible core is C-MAPSS + ERCOT + the one closest to
  the roles you are applying for.
* **Time-box each MVP.** When the budget runs out, stop adding features,
  fill the README with what you have, and write the limitations section.
  An honest "not done: weather features" is fine; an unfinished repository
  with no results table is not.
* **Extensions are optional.** Every brief separates the MVP from extensions,
  and marks GPU-only items. None of the MVPs needs a GPU.
* Keep doing ~30 minutes of SQL / Python drills and one mock question per day
  alongside the capstones; see the [study plan](../appendices/study-plan.md).

| Capstone | MVP | Typical extensions | Where time usually goes |
|---|---|---|---|
| C-MAPSS | 4 days | FD002/FD004 operating-condition normalisation, uncertainty | Windowing and RUL-cap choice |
| ERCOT | 4–5 days | Weather features, probabilistic forecasts | Parsing yearly files, DST, hour-ending |
| Online Retail | 5 days | Hierarchical reconciliation, price effects | Cleaning rules and deciding the label |
| MovieLens | 4–5 days | Implicit MF / EASE tuning, content features for cold items | Building the leakage-free evaluation |
| Civil Comments | 5 days | Transformer fine-tuning (GPU), calibration per subgroup | Data access for identity labels; subsampling |

---

## Shared definition of done (every capstone)

A capstone is **done** when all of these are true. Copy this list into your
repository as an issue or a checklist.

1. **Repository** created from the template; package renamed; `make setup`,
   `make data`, `make train`, `make evaluate`, `make test` and `make serve`
   all work from a clean clone (state the Python version).
2. **Data** is downloaded by code from the official source, checksummed, and
   never committed. `DATA_CARD.md` names the source, licence/terms (with a
   short quote), citation, size you measured, and every cleaning decision.
3. **Question** stated in business terms, with who would act on the answer
   and what decision changes.
4. **Evaluation protocol** written *before* modelling: split, metrics, which
   data selects models and which is touched once. A test asserts the split
   does not leak (time, entity or user).
5. **At least one baseline** from the brief, and **at least one improvement**,
   both evaluated on the same split, with a measure of uncertainty
   (bootstrap over entities/days/users, or across backtest folds).
6. **Results table filled from your run**, with the split named. No invented
   or copied numbers.
7. **Error analysis**: the slices from the brief (peak hours, late
   predictions, intermittent items, cold-start users, identity subgroups),
   with at least one hypothesis per bad slice.
8. **API** with `/health`, `/ready` and a prediction endpoint; Pydantic
   validation; the model version in every response; tests for each
   validation rule and for the not-ready state.
9. **CI** green: `ruff` + `pytest` on toy data, no network.
10. **`MODEL_CARD.md`** and a **Limitations** section that a sceptical
    reviewer would agree with.
11. **You can answer the brief's interview questions** aloud in 2 minutes
    each, using your own numbers.

---

## Shared capstone rubric (100 points)

Score your own repository, then ask someone else to. The
[readiness rubric](../appendices/readiness-rubric.md) uses the total.

| Area | Points | Full marks | Half marks | Zero |
|---|---:|---|---|---|
| **Problem framing** | 10 | Business question, decision owner, unit of prediction, horizon and label are precise; the metric follows from the decision | Question is clear but metric or horizon is generic | "Predict X with deep learning" |
| **Data provenance & validation** | 10 | Official source, licence quoted, checksums, contract checks, every cleaning decision justified and counted | Source and licence stated; cleaning ad hoc | Data from an unnamed mirror; licence ignored |
| **Evaluation protocol & leakage** | 20 | Split matches deployment (time / entity / user); leakage traps from the brief tested in code; test touched once; uncertainty reported | Correct split but no leakage tests or no uncertainty | Random split on temporal or grouped data; tuned on test |
| **Baselines & improvement** | 15 | Strong, domain-appropriate baseline; improvement justified and measured on the same split; negative results reported | Weak baseline only (e.g. global mean) | No baseline |
| **Error analysis & limitations** | 10 | Slices from the brief analysed with counts and CIs; limitations specific to the data | Generic limitations; one slice | None |
| **Engineering quality** | 15 | Package, config, Makefile, tests that would catch real bugs, CI green, reproducible from a clean clone | Runs, but only from a notebook or with manual steps | Cannot be run by someone else |
| **Serving** | 10 | App factory, health vs readiness, validation rules tested, model version in responses, clear precomputed-vs-online split | Endpoint works; little validation | No API |
| **Communication** | 10 | README a hiring manager can read in 3 minutes; model card; honest framing; you can defend every number | Complete but long or vague | Hype, unclear claims |
| **Total** | **100** | | | |

**Automatic caps** (apply regardless of the total):

* Any invented or copied result presented as yours → **score 0** until removed.
* Test-set leakage found (e.g. scaler fitted on all data, same engine/user in
  train and test, threshold tuned on test) → **max 50**.
* Data used against its terms (e.g. MovieLens in a revenue-bearing product,
  raw files redistributed where not allowed) → **max 50** until fixed.

A score of **75+** is a solid portfolio piece. **85+** with a clean interview
walkthrough is the bar to aim for on at least two capstones.

---

## Stretch for all capstones: reuse the bootcamp patterns

The bootcamp's `churnlab` package already contains small, readable versions of
the operational pieces. Copy the ideas, not necessarily the code:

* **Registry with a CURRENT pointer, promote and rollback** —
  [`src/churnlab/registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py).
* **Model bundles with metadata and integrity checks** —
  [`src/churnlab/artifacts.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/artifacts.py).
* **Drift (PSI), alert rules and latency summaries** —
  [`src/churnlab/monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py).
* **App factory with `/metrics` and `/admin/reload`** —
  [`src/churnlab/api.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/api.py).

Each brief ends with a specific suggestion for that dataset. For interview
practice beyond the per-capstone questions, see the
[interview question bank](../appendices/interview-questions.md).
