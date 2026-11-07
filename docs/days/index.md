# The five-day plan

Five days of about **3 hours** each, plus a one-hour diagnostic before you start.
Self-paced: a "day" is a session, not a calendar day, and nothing breaks if a
day takes two sittings.


---

## How to work through a day

1. **Read the day page first** (5 minutes). It says what the point of the day is
   and what proves you are done.
2. **Run every cell.** In Colab: open the notebook link, then *Runtime → Run all*,
   then go back to the top and read as you re-run cell by cell.
3. **Do the exercises before reading the solutions.** Each graded exercise
   prints ✅, ❌ (with a hint) or ⏳ TODO. A ✅ means the mechanics are right; it
   does not mean you can explain them.
4. **Answer the checkpoint questions aloud**, without notes, at the end of each
   notebook. They are interview questions on purpose. If you cannot answer one
   in two or three sentences, that is the most useful thing you learned today.
5. **Collect the end-of-day evidence** listed on each day page. If you cannot
   produce it, the day is not finished.

Every notebook starts with a setup cell and a catch-up cell. The catch-up cell
rebuilds anything earlier notebooks would have produced, so **any notebook runs
in a fresh Colab runtime** even if you skipped the previous one or your session
was recycled. See the [Colab guide](../guides/colab.md).

## The week at a glance

| Day | Theme | Notebooks | Notebook time |
|---|---|---|---|
| 0 | The diagnostic | 00 | 60 min |
| [1](day-1.md) | Reproducible ML projects | 01–04 | ~185 min |
| [2](day-2.md) | Evaluation & improvement | 05–08 | ~180 min |
| [3](day-3.md) | Serving predictions | 09–11 | ~160 min + review |
| [4](day-4.md) | Deployment & operations | 12–14 | ~160 min + review |
| [5](day-5.md) | Interview demonstration | 15–17 | ~180 min |

Times are for working through the notebook *and* its exercises at a steady
pace. Expect the first day to run long while you get used to the setup.

---

<a id="day-0"></a>

## Day 0 — the diagnostic

**Goal:** an honest measurement of where you start, so you spend the week on
your weak areas rather than your comfortable ones.

| Time | Activity | Notebook |
|---|---|---|
| 0:00–0:10 | [Before you start](../guides/before-you-start.md) checklist | — |
| 0:10–1:10 | The diagnostic: Python, pandas, SQL, testing, evaluation | `00_diagnostic` · [GitHub](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb) · [Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb) |
| 1:10–1:20 | Write down your results and read the routing table | [Diagnostic guide](../guides/diagnostic.md) |

### The one idea

> Measure before you practise. The skills you avoid are the ones you most need
> to work on, and you will not notice you are avoiding them.

### What to pay attention to

- **Time yourself honestly.** The diagnostic is timed because interviews are.
  Finishing an item in 25 minutes that "should" take 8 is real information.
- **Do not look things up.** Looking up `pd.merge_asof` or `ROW_NUMBER() OVER`
  is normal on the job, but the point here is to find out what you can do
  unaided. Note every lookup you were tempted to make.
- **The SQL and testing sections** are where researchers usually score lowest.
  That is expected and is exactly what the week is for.

### End-of-day evidence

- Your result for each area, written down somewhere you will see again on Day 5.
- A short list: the two areas you scored lowest in, and which days cover them
  (the [diagnostic guide](../guides/diagnostic.md) has the routing table).

### If you only have 90 minutes

You have enough. Do the checklist, the diagnostic, and write your results down.

---

## If you fall behind

The notebooks are ordered by dependency, but the catch-up cells mean you can
skip and come back. If time is short, this is the order of importance:

| Priority | Notebooks | Why |
|---|---|---|
| Essential | 03, 04, 06, 07, 09, 14, 15 | Point-in-time SQL, leakage, honest comparison, thresholds, an API, monitoring, the final report: the core of most interviews |
| Important | 02, 05, 08, 10, 11, 13, 17 | Validation, pipelines, artifacts, skew, tests, operations, system design |
| Valuable | 01, 12, 16 | Reproducibility setup, packaging, drills: you will practise these all month anyway |

Each day page has an **If you only have 90 minutes** section with the minimum
version of that day.

## After the five days

The bootcamp leaves you with one project and a map of your gaps. The next
three weeks are capstones, daily SQL and Python drills, and mock interviews:
see the [four-week study plan](../guides/study-plan.md) and the
[capstones](../capstones/index.md).
