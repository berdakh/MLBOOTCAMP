# Part IV · Capstone projects

*Five projects on public data, each with a complete guided solution · four to six days each at three hours a day*

The capstones apply the practice of Parts I–III to problems of different
shapes: run-to-failure sensor data, an hourly time series, messy retail
transactions, a ranking problem, and a text classifier whose outputs are
reviewed by people. Each project folder contains:

| File | Purpose |
|---|---|
| `README.md` | The project brief: question, data and its terms, framing, baselines, evaluation protocol, milestones, interview questions |
| `walkthrough.ipynb` | A guided solution. Each step states the goal, asks you to decide or implement something, then presents and explains a solution, with alternatives and common mistakes |

Attempt each step of a walkthrough before reading its solution. The aim is
not to reproduce our code but to be able to defend every decision in your own
version.

## The projects

<!-- toc:start -->
| | Capstone | Guided solution |
|---|---|---|
| 1 | [Capstone 2 · Predictive maintenance on NASA C-MAPSS](cap1-predictive-maintenance/README.md) | [walkthrough](cap1-predictive-maintenance/walkthrough.ipynb) |
| 2 | [Capstone 1 · Day-ahead hourly load forecasting for the Texas grid (ERCOT)](cap2-electricity-load-forecasting/README.md) | in preparation |
| 3 | [Capstone 3 · Weekly product demand forecasting on UCI Online Retail II](cap3-retail-demand-forecasting/README.md) | [walkthrough](cap3-retail-demand-forecasting/walkthrough.ipynb) |
| 4 | [Capstone 4 · Movie recommendations (MovieLens)](cap4-movie-recommendations/README.md) | [walkthrough](cap4-movie-recommendations/walkthrough.ipynb) |
| 5 | [Capstone 5 · Toxic-comment triage for human review on Civil Comments](cap5-content-moderation/README.md) | in preparation |
<!-- toc:end -->

| Project | Problem type | Main skill | Typical duration |
|---|---|---|---|
| 1 · Predictive maintenance (NASA C-MAPSS) | Regression on multichannel sensor sequences | Engine-level splits; remaining-useful-life metrics; a compact temporal model | 4 days |
| 2 · Electricity load forecasting (ERCOT) | Hourly time-series forecasting | Rolling-origin backtests; seasonal baselines; peak-hour error | 4–5 days |
| 3 · Retail demand forecasting (UCI Online Retail) | Forecasting from transactions | Cleaning ambiguous data; intermittent demand; the difference between sales and demand | 5 days |
| 4 · Movie recommendations (MovieLens) | Ranking | Time-aware evaluation; Recall@10 and NDCG@10; cold start | 4–5 days |
| 5 · Content moderation triage (Civil Comments) | Text classification with human review | PR-AUC; thresholds under reviewer capacity; subgroup error rates | 5 days |

A repository template for your own version of any project (package
skeleton, FastAPI app factory, tests, CI, README, model card and data card
templates) is in [template/](template/README.md).

## Order

The projects are numbered in the recommended order.

1. **Predictive maintenance** comes first because it is closest to a
   signal-processing background: multichannel sensor time series, windowing,
   per-channel normalisation, and an evaluation that resembles cross-subject
   evaluation. Engines play the role of subjects: if windows from the same
   engine appear in both training and validation data, the score is inflated
   in the same way a within-subject brain-computer-interface score is. The
   data is small and trains on a CPU, so the project also serves to set up
   the template repository used for the others.
2. **Electricity load forecasting** remains a numerical time series, but the
   difficulty moves to the evaluation protocol: rolling-origin backtests,
   forecasts issued at a fixed time with only the data available then,
   daylight-saving days, and a metric that emphasises peak hours.
3. **Retail demand forecasting** is the first project in which the data is
   genuinely ambiguous: cancellations, non-product stock codes, missing
   customer identifiers, wholesale orders, and the fact that recorded sales
   are not the same as demand. It relies on SQL (DuckDB, as in Chapter 4) and
   introduces intermittent demand.
4. **Movie recommendations** changes the problem type from regression to
   ranking, with new metrics, a popularity baseline that is hard to beat, and
   users and items that have no history.
5. **Content moderation triage** is the most open-ended: text, a pretrained
   model, noisy labels, thresholds chosen under the capacity of a human review
   team, and error rates compared across identity subgroups.

A reader preparing for a particular kind of role may move the matching
project earlier: recommendation or advertising roles, Project 4; trust and
safety or language roles, Project 5; supply-chain roles, Project 3.

## Time

At three hours a day, a month provides about 90 hours. Part I uses about 15,
which leaves about 75; the five minimum versions together need roughly 66–72
hours if nothing goes wrong. In practice it is better to plan three finished
projects and to treat the fourth and fifth as optional. Three projects whose
every decision you can explain are worth more than five unfinished ones.
Time-box each project: when the budget is spent, stop adding features, record
the results you have, and write the limitations section.

| Project | Minimum version | Typical extensions | Where time is usually spent |
|---|---|---|---|
| 1 · C-MAPSS | 4 days | FD002/FD004 with operating-condition normalisation; uncertainty estimates | Windowing; the choice of RUL cap |
| 2 · ERCOT | 4–5 days | Weather features; probabilistic forecasts | Parsing the yearly files; daylight saving; hour-ending timestamps |
| 3 · Online Retail | 5 days | Hierarchical reconciliation; price effects | Cleaning rules; deciding the label |
| 4 · MovieLens | 4–5 days | Tuned matrix factorisation or EASE; content features for new items | Building an evaluation without leakage |
| 5 · Civil Comments | 5 days | Transformer fine-tuning (GPU); calibration per subgroup | Access to identity labels; subsampling |

## Rules for reporting results

These are public-data exercises inspired by problems in energy, industry,
retail and online platforms. They are not work done for, with, or endorsed
by ERCOT, NASA, Jigsaw, Google or any other organisation, and your write-up
must not suggest otherwise.

* **Report only measured results.** Every number in a README, model card or
  interview answer must come from a run of your code on a named split. A
  result you have not produced is written as *TBD — fill from your run*.
* **Do not invent impact.** "Mean absolute error 12% lower than the
  seasonal-naive baseline on 2023 rolling-origin folds" is a measurement.
  "Would save two million dollars a year" is not, unless you state a cost
  model and label it as an assumption.
* **Keep published results separate.** If you cite a number from a paper, put
  it in a clearly labelled note of published reference points, and only if
  you have read the paper.
* **Read the data terms yourself.** The briefs were written on a machine that
  could not reach every data host; facts about each dataset are marked
  *verified from an official page snippet* or *to verify*. Read the licence
  on the official page before downloading, and quote its key condition in
  your data card.

## Definition of done

A project is finished when all of the following hold.

1. A repository created from the template, with `make setup`, `make data`,
   `make train`, `make evaluate`, `make test` and `make serve` working from a
   clean clone.
2. Data downloaded by code from the official source, checked against a size
   or checksum, and not committed. The data card names the source, licence
   (with a short quotation), citation, the size you measured, and every
   cleaning decision.
3. The question stated precisely: who acts on the answer, and which decision
   it changes.
4. An evaluation protocol written before modelling: the split, the metrics,
   which data selects models and which is used once at the end. A test
   asserts that the split does not leak across time, entities or users.
5. At least one baseline and one improvement, evaluated on the same split,
   with a measure of uncertainty (a bootstrap over entities, days or users,
   or the spread across backtest folds).
6. A results table filled from your own run, with the split named.
7. Error analysis on the slices named in the brief, with at least one
   hypothesis for each poor slice.
8. A prediction API with `/health`, `/ready` and a prediction endpoint,
   validated inputs, the model version in every response, and tests for each
   validation rule and for the not-ready state.
9. Continuous integration passing: lint and tests on small toy data, without
   network access.
10. A model card and a limitations section that a sceptical reviewer would
    accept.
11. The ability to answer the brief's interview questions aloud, about two
    minutes each, using your own numbers.

## Assessment rubric (100 points)

Score your own repository, then ask someone else to score it. The
[readiness rubric](../appendices/readiness-rubric.md) uses the total.

| Area | Points | Full marks | Half marks | No marks |
|---|---:|---|---|---|
| Problem framing | 10 | Question, decision owner, unit of prediction, horizon and label are precise; the metric follows from the decision | The question is clear but the metric or horizon is generic | "Predict X with deep learning" |
| Data provenance and validation | 10 | Official source, licence quoted, checksums, contract checks, every cleaning decision justified and counted | Source and licence stated; cleaning ad hoc | Data from an unnamed copy; licence ignored |
| Evaluation protocol and leakage | 20 | The split matches deployment; the brief's leakage risks are tested in code; the test data is used once; uncertainty reported | Correct split, but no leakage tests or no uncertainty | Random split on temporal or grouped data; tuning on the test set |
| Baselines and improvement | 15 | A strong, domain-appropriate baseline; the improvement is justified and measured on the same split; negative results reported | A weak baseline only, such as the global mean | No baseline |
| Error analysis and limitations | 10 | The brief's slices analysed with counts and confidence intervals; limitations specific to the data | Generic limitations; one slice | None |
| Engineering quality | 15 | Package, configuration, Makefile, tests that would catch real defects, CI passing, reproducible from a clean clone | Runs, but only from a notebook or with manual steps | Cannot be run by another person |
| Serving | 10 | App factory, health and readiness separated, validation rules tested, model version in responses, a clear division between precomputed and online computation | The endpoint works; little validation | No API |
| Communication | 10 | A README a reviewer can read in three minutes; a model card; accurate framing; every number can be explained | Complete but long or vague | Overstated or unclear claims |
| **Total** | **100** | | | |

**Caps that apply regardless of the total.** An invented or copied result
presented as your own: score 0 until removed. Leakage into the test set (a
scaler fitted on all data, the same engine or user in training and test, a
threshold tuned on the test set): at most 50. Data used against its terms
(for example MovieLens in a commercial product, or raw files redistributed
where this is not allowed): at most 50 until corrected.

A score of 75 or more is a solid portfolio project; 85 or more, with a clear
explanation in an interview, is a good target for at least two projects.

## Reusing Part I components

The `churnlab` package from Part I contains small, readable versions of the
operational components each project needs. Reuse the ideas rather than
necessarily the code:

* a registry with a CURRENT pointer, promotion and rollback:
  [`src/churnlab/registry.py`](../src/churnlab/registry.py);
* model bundles with metadata and integrity checks:
  [`src/churnlab/artifacts.py`](../src/churnlab/artifacts.py);
* drift measurement (PSI), alert rules and latency summaries:
  [`src/churnlab/monitoring.py`](../src/churnlab/monitoring.py);
* an app factory with `/metrics` and `/admin/reload`:
  [`src/churnlab/api.py`](../src/churnlab/api.py).

For interview practice beyond each project's own questions, see the
[interview question bank](../appendices/interview-questions.md).
