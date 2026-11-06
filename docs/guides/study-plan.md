---
title: Four-week study plan
parent: Guides
nav_order: 5
---

# Four-week study plan
{: .no_toc }

About a month at **3 hours a day**. Week 1 is the bootcamp. Weeks 2–4 turn what
the bootcamp introduced into skills you can use under pressure: capstones on
real public data, daily SQL and Python drills, and weekly mock interviews.

1. TOC
{:toc}

---

## What five days does, and what it does not

Be clear about this, with yourself and with interviewers.

| Five days gives you | Needs continued practice (weeks 2–4 and beyond) |
|---|---|
| One working, tested, documented end-to-end project you can walk through | Building the same kind of project again, on messier data, without step-by-step notebooks |
| The vocabulary: leakage, embargo, skew, readiness, drift, rollback, calibration | Using that vocabulary naturally, under follow-up questions |
| Point-in-time SQL you have read and modified | Writing window functions and date joins from a blank editor, quickly and correctly |
| A FastAPI service and tests you have run | Writing a service and its tests from a blank file in under an hour |
| One worked system design and one scored mock | Ten or more designs, aloud, timed, with feedback |
| A first measurement of your gaps | Closing them |

Fluency comes from repetition with feedback. No course compresses that.

## Daily shape in weeks 2–4

| Block | Minutes | What |
|---|---|---|
| Drills | 30 | SQL or Python, timed, from a blank editor (alternate days). Notebook 16 is the template. |
| Capstone | 120 | The current capstone, following its brief |
| Explain | 30 | Explain today's capstone work aloud or in writing, as if to an interviewer; answer two questions from the [question bank](interview-questions.html) |

On mock-interview days (one per week), replace the capstone block with the
mock and its review.

## Week 1 — the bootcamp

| Day | Session (~3 h) |
|---|---|
| 1 | Day 0: [before you start](before-you-start.html), the diagnostic, then start [Day 1](../days/day-1.html) |
| 2 | Finish [Day 1](../days/day-1.html): reproducible project, validation, SQL, splits |
| 3 | [Day 2](../days/day-2.html): pipelines, model selection, thresholds, artifacts |
| 4 | [Day 3](../days/day-3.html): FastAPI, consistent serving, clients and tests |
| 5 | [Day 4](../days/day-4.html): packaging, health and latency, monitoring and rollback |
| 6 | [Day 5](../days/day-5.html): final evaluation, drills, system design and mock |
| 7 | Rest, or catch up. Score yourself on the [readiness rubric](readiness-rubric.html). Choose your first capstone |

Week 1 deliverable: the TaskFlow project in your own git repository, a
recorded five-minute walkthrough, and your first rubric scores.

## Week 2 — first capstone, SQL fluency

| Days | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 8–11 | Capstone A: MVP. Data download with checksums, validation, a leakage-safe split, a baseline, evaluation with intervals | SQL: joins and aggregation; then window functions (`ROW_NUMBER`, `LAG`, running sums) |
| 12 | Capstone A: package, tests, README, model card | Python: a small module with tests from a blank file |
| 13 | **Mock interview 1**: a project deep-dive on TaskFlow (30 min) + system design (40 min), recorded | — |
| 14 | Review the recording; rubric; rest | — |

## Week 3 — second capstone, serving and testing fluency

| Days | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 15–18 | Capstone B: MVP, plus a FastAPI service or batch scoring job with tests | SQL: date logic and point-in-time joins; pandas equivalents |
| 19 | Capstone B: monitoring sketch (PSI on a later period), README, model card | Python: write a retrying client and its tests from a blank file |
| 20 | **Mock interview 2**: coding (45 min: pandas or SQL) + behavioural (20 min, STAR) | — |
| 21 | Review; rubric; rest | — |

## Week 4 — third capstone, interview readiness

| Days | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 22–25 | Capstone C: MVP, focusing on the skill the rubric says is weakest | Mixed, timed: one SQL, one Python, one evaluation question per day |
| 26 | Portfolio polish: each capstone has a README with results, intervals, limitations, and how to run it | Re-take the diagnostic |
| 27 | **Mock interview 3**: full loop, ideally with another person: coding, ML depth, system design, behavioural | — |
| 28 | Final rubric; decide what to keep practising | — |

## Which capstones, in which order

Five capstone briefs are in [Capstones](../capstones/), with a recommended
order and reasoning. At the pace above, **three capstone MVPs** fit in weeks
2–4 alongside drills and mocks: an MVP takes roughly 4–5 days at 3 hours a
day, and extensions take longer. The capstone index recommends C-MAPSS
first, then ERCOT, then the one closest to the roles you are applying for as
capstones A, B and C. Do the remaining two afterwards. To choose the third:

| If you are targeting… | Prioritise |
|---|---|
| Energy, utilities, forecasting teams | [ERCOT load forecasting](../capstones/01-ercot-load-forecasting.html) |
| Industrial, IoT, sensor and signal-processing roles (closest to EEG work) | [C-MAPSS predictive maintenance](../capstones/02-cmapss-predictive-maintenance.html) |
| Retail, marketplace, supply chain | [Online Retail forecasting](../capstones/03-online-retail-forecasting.html) |
| Recommendations, ranking, personalisation | [MovieLens recommendations](../capstones/04-movielens-recommendations.html) |
| Trust and safety, content, NLP-adjacent | [Civil Comments moderation](../capstones/05-civil-comments-moderation.html) |

## Mock interviews

- **Record every mock.** Watching it back is uncomfortable and is where most of
  the learning happens.
- **Use a real person when you can**: a colleague, a friend in industry, a
  peer-practice community. Self-scored mocks are better than none, but you
  will be generous to yourself.
- **Score with the [readiness rubric](readiness-rubric.html)**, and write down
  the two weakest moments and what a better answer would have said.
- **Vary the format**: project deep-dive, live coding, ML system design,
  ML breadth questions, behavioural.

## If you have less time

| Available | Keep | Drop or postpone |
|---|---|---|
| 2 h/day | Bootcamp over 8–9 days; drills daily; one mock per week | Third capstone |
| 1 h/day | Bootcamp using each day's "If you only have 90 minutes" path, over two weeks; drills | Capstones until the bootcamp is done |
| Weekends only | One bootcamp day per weekend day; drills on weekday evenings for 20 minutes | Expect the plan to take about three months |

## After the month

Keep the daily drills going at 20–30 minutes; fluency fades quickly without
them. Keep adding to the question bank answers you have practised aloud. And
apply: real interviews are themselves practice, and their feedback is the most
accurate measure of readiness you will get.
