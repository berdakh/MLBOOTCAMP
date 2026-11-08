# Interpreting the diagnostic

The skills diagnostic of
[Chapter 1](../part1-foundations/ch01-diagnostic/README.md)
([lab](../part1-foundations/ch01-diagnostic/notebook.ipynb),
[open in Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb))
measures five areas: **Python, pandas, SQL, testing and evaluation**. This
appendix explains how to read the result and how to use it to decide where to
spend additional time.

---

## What the diagnostic measures, and what it does not

The diagnostic measures whether you can produce small, correct pieces of code
in each area **without assistance and within a time limit**, which is also
what a technical screening interview measures. It does not measure research
ability, mathematical depth, or the quality of the models you build. A low
score in an area in which you have had no training (for most researchers, SQL
and software testing) is the expected starting point, not a judgement of
ability.

## Converting the result into a rating

For each area, compute the share of items you passed, and note how many of
them required you to look something up. Then assign a rating:

| Rating | Share passed without assistance | Time | Interpretation |
|---|---|---|---|
| **Strong** | 80% or more | Within the suggested time | Work through the area's chapters normally; spend any spare time on weaker areas |
| **Partial** | 50–79% | Or over time, or with look-ups | Work through the area's chapters carefully, including every exercise, and add the area to your regular drills |
| **Gap** | Below 50% | — | Give the area's chapters priority; read each chapter's text in full before its lab; drill the area regularly after Part I |

An item that was passed but took longer than the suggested time counts as
half a pass for this purpose, because interviews are timed.

## Where each area is taught

| Area | Chapters that teach it | Further practice |
|---|---|---|
| **Python** (functions, data structures, errors, modules) | [Chapter 2](../part1-foundations/ch02-reproducible-project/README.md); [Chapters 10–12](../part1-foundations/ch10-fastapi-and-contracts/README.md); [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md) | Rebuild one `churnlab` module from an empty file at regular intervals; [Chapters 33–34](../part2-ml-at-scale/ch33-coding-patterns-1/README.md) (coding patterns) |
| **pandas** (selection, grouping, merging, time) | [Chapter 3](../part1-foundations/ch03-data-acquisition-and-validation/README.md), [Chapter 5](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md), [Chapter 9](../part1-foundations/ch09-error-analysis-and-artifacts/README.md) | Translate each SQL drill of Chapter 17 into pandas as well |
| **SQL** (joins, aggregation, window functions, dates) | [Chapter 4](../part1-foundations/ch04-sql-for-ml/README.md), [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md) | Regular 20–30 minute SQL drills after Part I; [Chapter 20](../part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md) (aggregation and joins at scale) |
| **Testing** (pytest, fixtures, what to assert) | [Chapter 10](../part1-foundations/ch10-fastapi-and-contracts/README.md), [Chapter 12](../part1-foundations/ch12-api-clients-and-tests/README.md), [Chapter 13](../part1-foundations/ch13-packaging-and-configuration/README.md) | Write the tests first for each capstone module |
| **Evaluation** (metrics under class imbalance, uncertainty, thresholds) | [Chapter 5](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md); [Chapters 6–9](../part1-foundations/ch06-pipelines-and-cross-validation/README.md); [Chapter 16](../part1-foundations/ch16-final-evaluation-and-docs/README.md) | Explain one capstone result aloud, with its confidence interval, after each capstone; [Chapters 29–30](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md) (experimentation) |

Two areas are not covered directly by the diagnostic, and most researchers
also need to practise them: **APIs and deployment**
([Chapters 10–15](../part1-foundations/ch10-fastapi-and-contracts/README.md))
and **communication**
([Chapters 16–18](../part1-foundations/ch16-final-evaluation-and-docs/README.md),
and the checkpoint questions at the end of every lab). Unless you have
previously built and operated a service that other people used, assume that
both need attention.

## Typical profiles

| Profile | Common pattern | Suggested emphasis |
|---|---|---|
| Strong modeller, new to engineering | Evaluation strong; SQL and testing weak | Chapter 4 slowly, with every exercise; Chapters 10–12 in full; regular SQL drills after Part I |
| Strong programmer, little machine learning | Python and testing strong; evaluation partial | Chapters 6–9 in full, reading each chapter's text carefully before its lab |
| Background in data analysis | SQL and pandas strong; testing and Python modules partial | Chapters 10–15; build each capstone as a package with tests |
| Even across all areas | Partial everywhere | Follow the [study plan](study-plan.md) as written; Chapters 2–5 and 10–12 usually give the largest improvement |

## Measuring again

Repeat the diagnostic, or compare your times on the drills of
[Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md), at
the end of Part I and again after each capstone. Record the numbers in one
place, so that change over time is visible. The
[readiness rubric](readiness-rubric.md) is the broader measure; the
diagnostic is the quick one.
