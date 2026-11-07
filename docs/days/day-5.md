# Day 5 — Interview demonstration

**Goal:** by the end of today you have one honest, final number for the
project, the documents that let a stranger understand and reuse it, and a
first measurement of how you perform when someone is asking the questions.


---

## Schedule

| Time | Activity | Notebook |
|---|---|---|
| 0:00–0:50 | Touch the test set once; report with uncertainty; README and model card | `15_final_evaluation_and_docs` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/15_final_evaluation_and_docs.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/15_final_evaluation_and_docs.ipynb) |
| 0:50–1:50 | Timed, self-checking Python and SQL drills | `16_python_and_sql_drills` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb) |
| 1:50–1:55 | *Break* | — |
| 1:55–3:05 | A system design framework, a worked design, a scored mock interview | `17_system_design_and_mock_interview` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb) |

## The one idea

> Interviewers are not checking whether your model is good. They are checking
> whether you know how good it is, how you would find out if you were wrong,
> and whether you can say both clearly.

## What to pay attention to

**In notebook 15**, the discipline of touching the test period once. The
model, the threshold and every preprocessing choice were fixed on validation
data on Day 2. Today you score the test period, compute the metrics with a
customer-level bootstrap interval, and report them alongside the validation
numbers, whatever they are. If test is worse than validation, that is a
finding to explain (time drift? selection on validation?), not a reason to
re-tune. Then write the README and model card for someone who has never seen
the project: intended use, data period, metrics with intervals and split named,
known limitations, and the fact that the data is synthetic.

**In notebook 16**, your times. The drills check correctness automatically;
the clock is your job. Record how long each one took and which ones needed a
lookup. Compare with your Day 0 diagnostic. Expect improvement in the areas
the week covered and much less elsewhere: that is what the remaining three
weeks are for.

**In notebook 17**, structure before detail. The framework asks you to pin down
the decision being made, how predictions are consumed, and how success is
measured *before* any model is named. In the mock, score yourself with the
rubric honestly. Recording yourself is uncomfortable and far more useful than
rereading notes.

## End-of-day evidence

| Evidence | Where it comes from |
|---|---|
| One test-period result: metric, customer-level 95% interval, split named, and the validation number next to it | Notebook 15 |
| A project README and model card that state limitations and that the data is synthetic | Notebook 15 |
| Your drill times, next to your Day 0 diagnostic results | Notebook 16 |
| A scored mock interview: your rubric scores and the two weakest moments, written down | Notebook 17 |
| A five-minute spoken walkthrough of the whole TaskFlow project, recorded | Your phone or laptop |

The last item matters most. If you can explain the project end to end in five
minutes, including what you would do differently with real data, you have
something to bring to an interview.

## If you only have 90 minutes

| Time | Do |
|---|---|
| 0:00–0:40 | Notebook 15: the one-time test evaluation and the model card |
| 0:40–1:30 | Notebook 17: read the worked design, then do the mock interview |

Do notebook 16's drills on the first day of week 2; they are the template for
your daily practice.

## What comes next

Five days is a start. Use the [readiness rubric](../guides/readiness-rubric.md)
to score yourself today, then follow the [four-week study plan](../guides/study-plan.md):
capstones, daily drills and weekly mock interviews. The capstone briefs and
recommended order are in [Capstones](../capstones/index.md).

## Optional reading

- [Handbook 7 · ML system design](../handbook/07-ml-system-design.md)
- [Handbook 1 · The ML engineering job](../handbook/01-the-ml-engineering-job.md)
- [Interview question bank](../guides/interview-questions.md)
