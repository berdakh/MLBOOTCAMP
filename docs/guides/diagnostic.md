# Interpreting the diagnostic

[Notebook 00](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb)
([Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb))
measures five areas: **Python, pandas, SQL, testing and evaluation**. This page
explains how to read your result and where to spend extra time.


---

## What the diagnostic measures and what it does not

It measures whether you can produce small, correct pieces of code in each area
**unaided and under time pressure**, which is what a technical screen measures.
It does not measure research ability, mathematical depth, or how good your
models are. A low score in an area you have never been trained in (SQL and
testing, for most researchers) is the expected starting point, not a verdict.

## Turning your result into a level

For each area, work out the share of items you passed, and note how many you
needed to look something up for. Then:

| Level | Share passed, unaided | Time | What it means |
|---|---|---|---|
| **Strong** | 80% or more | Within the suggested time | Do the area's days normally; spend spare time elsewhere |
| **Partial** | 50–79% | Or over time, or with lookups | Do the area's days carefully, including every exercise; add the area to your daily drills |
| **Gap** | Below 50% | — | Prioritise the area's days; read its handbook chapter first; daily drills from week 2 |

Being *over time* on an item you passed counts as half a pass for this
purpose. Interviews are timed.

## Routing table

| Area | Days and notebooks that train it | Handbook | Extra practice |
|---|---|---|---|
| **Python** (functions, data structures, errors, modules) | Day 1 (01), Day 3 (09–11), Day 5 (16) | [2 · Reproducibility](../handbook/02-reproducibility.md) | Rebuild one `churnlab` module from a blank file each week |
| **pandas** (selection, groupby, merge, time) | Day 1 (02, 04), Day 2 (08) | [3 · Data, SQL and leakage](../handbook/03-data-sql-and-leakage.md) | Translate each SQL drill into pandas as well |
| **SQL** (joins, aggregation, windows, dates) | Day 1 (03), Day 5 (16) | [3 · Data, SQL and leakage](../handbook/03-data-sql-and-leakage.md) | Daily 20–30 minute SQL drills in weeks 2–4 |
| **Testing** (pytest, fixtures, what to assert) | Day 3 (09, 11), Day 4 (12) | [5 · Serving](../handbook/05-serving.md) | Write tests first for each capstone module |
| **Evaluation** (metrics for imbalance, uncertainty, thresholds) | Day 1 (04), Day 2 (05–08), Day 5 (15) | [4 · Evaluation and decisions](../handbook/04-evaluation-and-decisions.md) | Explain one capstone result per week aloud with its interval |

Two areas the diagnostic does not cover directly, and which most researchers
also need to practise: **APIs and deployment** (Days 3–4) and **communication**
(Day 5, and every checkpoint). Assume you need them unless you have shipped a
service before.

## Typical profiles

| Profile | Common pattern | Suggested emphasis |
|---|---|---|
| Strong modeller, new to engineering | Evaluation strong; SQL and testing gaps | Day 1 notebook 03 slowly; Day 3 in full; SQL drills daily from week 2 |
| Strong programmer, light ML | Python and testing strong; evaluation partial | Day 2 in full; Handbook 4 before Day 2 |
| Data analyst background | SQL and pandas strong; testing and Python modules partial | Day 3 and Day 4; build capstones as packages with tests |
| Even across the board | Partial everywhere | Follow the plan as written; the biggest gains come from Days 1 and 3 |

## Re-measure

Retake the diagnostic, or compare with notebook 16's drills, at the end of
Day 5 and again at the end of each week. Keep the numbers in one place. The
[readiness rubric](readiness-rubric.md) is the broader measure; the
diagnostic is the quick one.
