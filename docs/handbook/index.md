---
title: Handbook
nav_order: 3
has_children: true
---

# Handbook

The ideas behind each day, without code. Each chapter is readable in 20–30
minutes. Read the chapter for a day the evening before, or after, depending on
whether you prefer the map before or after the territory.

| # | Chapter | Read with | Covers |
|---|---|---|---|
| 1 | [The ML engineering job](01-the-ml-engineering-job.html) | Before Day 1 | MLE vs Applied Scientist vs Algorithm Engineer; which research habits transfer and which hurt |
| 2 | [Reproducibility](02-reproducibility.html) | Day 1 | What "re-runnable" means, environments, seeds, config, run records, data versions |
| 3 | [Data, SQL and leakage](03-data-sql-and-leakage.html) | Day 1 | Point-in-time correctness, label leakage, group and temporal leakage, the cross-subject analogy |
| 4 | [Evaluation and decisions](04-evaluation-and-decisions.html) | Day 2 | Metrics for imbalance, confidence intervals, cluster bootstrap, thresholds from costs, calibration, slices |
| 5 | [Serving](05-serving.html) | Day 3 | Contracts, training/serving skew, the app factory, idempotence, batching, latency |
| 6 | [Operations and monitoring](06-operations-and-monitoring.html) | Day 4 | Health vs readiness, logs, SLOs, drift vs performance, delayed labels, rollback, canary and shadow |
| 7 | [ML system design](07-ml-system-design.html) | Day 5 | A framework, and one worked sketch |

## Conventions

- **Numbers.** When a chapter quotes a number from the TaskFlow project, it
  says so, and it is a fact about synthetic teaching data. Cost figures
  ($150 per reached churner, $30 per call) are stated *assumptions*.
- **Tools.** Chapters name tools as examples (DuckDB, scikit-learn, FastAPI,
  MLflow, Prometheus). The ideas carry across tools; the tool choices are what
  this course happens to use.
- **Research analogies.** Where an engineering idea has a direct counterpart
  in experimental research (especially EEG/BCI and cross-subject evaluation),
  the chapter says so. The analogies are there to help you explain the ideas
  in an interview using experience you already have.
