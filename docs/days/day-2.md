---
title: "Day 2 · Evaluation & improvement"
parent: Daily plan
nav_order: 2
---

# Day 2 — Evaluation & improvement
{: .no_toc }

**Goal:** by the end of today you have chosen a model for a reason you can
defend, attached uncertainty to every number you report, turned scores into a
decision with a threshold derived from stated costs, and saved the result as a
versioned artifact that someone else could load and audit.

1. TOC
{:toc}

---

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–0:50 | `Pipeline`, `ColumnTransformer`, rolling-origin CV, preprocessing leaks | `05_pipelines_and_cross_validation` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb) |
| 0:50–1:35 | Comparing models with confidence intervals; search on a budget | `06_model_selection` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb) |
| 1:35–1:40 | *Break* | — |
| 1:40–2:20 | PR curves, cost-based thresholds, calibration, capacity | `07_thresholds_and_calibration` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb) |
| 2:20–3:05 | Slice analysis, model bundles, a registry, a model card | `08_error_analysis_and_artifacts` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/08_error_analysis_and_artifacts.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/08_error_analysis_and_artifacts.ipynb) |

## The one idea

> A model is not "better" because its validation score is higher. It is better
> if the difference survives resampling the *right unit* (customers, not rows),
> and if it makes better *decisions* at the threshold you will actually use.

## What to pay attention to

**In notebook 05**, where the scaler and the imputer are fitted. A
`StandardScaler` fitted on all the data before cross-validation leaks the
validation fold's mean and variance into training. Putting preprocessing
inside the `Pipeline` fixes that, and it is also what makes serving
consistent on Day 3: the object you save is the object you serve.

**In notebook 06**, the confidence intervals, and *how* they are computed.
The same customer appears at many monthly snapshots, so its rows are
correlated. A row-level bootstrap treats them as independent and produces
intervals that are too narrow. The cluster bootstrap resamples whole
customers. Compare the widths. On this dataset they come out almost
the same, because within the validation window each customer has only a few
rows and at most one positive. The notebook shows when the gap appears and why
you should still use the cluster version by default. Also notice whether
the boosted trees actually beat logistic regression here, and think about why
the answer might be no on a small, mostly linear-in-logs problem.

**In notebook 07**, the cost assumptions. The course assumes a retention call
costs **$30** and that reaching a would-be churner is worth **$150** in
expectation (a $600 margin times an assumed 25% success rate). Those numbers
are **teaching assumptions, not facts about any business.** With calibrated
probabilities they imply calling whenever p > 30 / (150 + 30) = **1/6**. Watch
what happens to the chosen threshold when the probabilities are *not*
calibrated, and when the team can only make a fixed number of calls a week.

**In notebook 08**, the slice table and the small-slice flag. A metric on 40
rows is mostly noise; the code flags it rather than hiding it. Then look
inside a model bundle: `metadata.json` records the training-data fingerprint,
the feature list in order, library versions, the threshold and validation
metrics. Every one of those fields answers a question you will be asked.

## End-of-day evidence

| Evidence | Where it comes from |
|---|---|
| A cross-validation result from a rolling-origin scheme, with preprocessing inside the pipeline | Notebook 05 |
| A comparison table: two or more models, validation PR-AUC with a **customer-level** bootstrap interval, and a paired difference | Notebook 06 |
| A threshold, the cost assumptions it came from (stated in a sentence), and the precision, recall and alert volume it produces on validation | Notebook 07 |
| A registry in your artifact directory with at least two versions and a `CURRENT` pointer, and a model card draft | Notebook 08 |
| Spoken answers to every checkpoint question in 05–08 | Checkpoints |

{: .important }
> All of today's numbers are **validation** numbers. The test period is touched
> once, on Day 5. If you looked at test scores to choose anything today, your
> Day 5 result is no longer an unbiased estimate. Say so if it happened.

## If you only have 90 minutes

| Time | Do |
|---|---|
| 0:00–0:45 | Notebook 06: the bootstrap sections and the paired comparison |
| 0:45–1:30 | Notebook 07: the cost-based threshold and the calibration table |

Read the `Pipeline` section of notebook 05 and the `metadata.json` of one
bundle from notebook 08. The catch-up cell on Day 3 builds the registry for you.

## Optional reading

- [Handbook 4 · Evaluation and decisions](../handbook/04-evaluation-and-decisions.html)
- [Interview questions: evaluation](../guides/interview-questions.html#evaluation-and-metrics)
