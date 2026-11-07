# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 16 · Python & SQL drills
#
# **Day 5 · ~60 minutes · CPU only**
#
# Most ML engineering loops include a live coding round and, increasingly, a SQL
# round, and both are timed. You already know how to solve these problems; the
# skill being tested is solving them **quickly, correctly and out loud**, which
# only comes from practice against a clock.
#
# By the end you will be able to:
#
# 1. solve five classic Python interview problems with the right complexity, and say what that complexity is;
# 2. write window-function, cohort and gaps-and-islands SQL against real-shaped tables;
# 3. use a timer and a self-check to practise under realistic conditions;
# 4. know which drill types to repeat over the next three weeks, and how.
#
# **You will produce:** `artifacts/drills/<date>.json`, your times and results for today, so you can see progress.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: the SQL drills need the CLEAN TaskFlow tables (notebook 02) ---
from churnlab import bootstrap
from churnlab.grading import nb16 as grade

tables = bootstrap.load_clean()
con = grade.connection()  # DuckDB with customers, usage_weekly, invoices, tickets registered
for name, df in tables.items():
    print(f"{name:<14}{len(df):>8,} rows   {', '.join(df.columns)}")

# %% [markdown]
# **Expected:** four tables: `customers` 5,000 rows, `usage_weekly` ~210,000,
# `invoices` ~55,600, `tickets` ~22,400. These are the *cleaned* tables:
# duplicates removed, the `-1` sessions sentinel turned into missing, region
# `"NA"` kept as North America.

# %% [markdown]
# ## 📖 Words to know
#
# Vocabulary only: the drills themselves are yours to solve.
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Time complexity (Big-O)** | how the work grows with input size $n$ | a loop inside a loop over $n$ is $O(n^2)$ |
# | **Amortised cost** | average cost per step over a whole run, even if some steps are expensive | each item pushed and popped at most once |
# | **Deque** | a list you can add to or remove from at both ends in $O(1)$ | `collections.deque` |
# | **Stable tie-break** | a rule that decides equal keys the same way every time | "the later record wins" |
# | **Context manager** | an object used with `with` that runs setup and cleanup code | the drill timer |
# | **Precision@k** | share of the k highest-scored rows that are positive | "of 40 calls, how many reach churners?" |
# | **Stratified split** | a split that keeps each class's share in both parts | 5% churners in train and test |
# | **Running total** | a cumulative sum ordered by time | `SUM(...) OVER (ORDER BY month)` |
# | **Cohort** | the customers who share a start period | everyone who signed up in March 2023 |
# | **Retention at month +k** | share of a cohort still active k months after its start | `ret_m1`, `ret_m2`, `ret_m3` |

# %% [markdown]
# ## 16.1 · How to use this notebook
#
# Each drill has a **target time**. Treat it like the real thing:
#
# 1. Read the prompt once. Say (or type in a comment) your approach and its
#    complexity **before** you code. Interviewers grade that sentence.
# 2. Start the timer cell, write the function, run the check.
# 3. If you go over the target by more than 50%, stop, read the hint in the
#    grader message, and **put the drill on your repeat list** (section 16.4).
#
# > 💡 **Intuition.** Saying the complexity before coding is reading the map
# > before driving: an interviewer cares more that you know the route is
# > $O(n \log n)$ than that you reach the end. *Where it breaks down:* Big-O
# > hides constants; for $n$ = 100, a "slow" $O(n^2)$ loop can beat a clever
# > structure, so state the input size too.
#
# Being slower than target today is normal and is the point: you are measuring
# where you start. The hints are deliberately one line; the full worked
# solutions are in `solutions/16_python_and_sql_drills.md`.
#
# The timer is a small context manager. Writing one is itself a classic
# Python question (`__enter__`/`__exit__`).

# %%
import json
import time
from datetime import date

RESULTS: dict[str, dict] = {}


class Timer:
    """`with Timer("ex1", target_min=6): ...` prints elapsed vs target and records it."""

    def __init__(self, name: str, target_min: float):
        self.name, self.target_min = name, target_min

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_min = (time.perf_counter() - self.start) / 60
        verdict = "within target" if self.elapsed_min <= self.target_min else "OVER target - add to repeat list"
        print(f"⏱  {self.name}: {self.elapsed_min:.1f} min (target {self.target_min} min) - {verdict}")
        RESULTS.setdefault(self.name, {}).update(minutes=round(self.elapsed_min, 2), target=self.target_min)
        return False  # never swallow an exception


def start(name: str, target_min: float) -> None:
    """Start a drill clock across cells (in a notebook you write the answer in a separate cell)."""
    RESULTS[name] = {"started": time.perf_counter(), "target": target_min}
    print(f"⏱  {name} started - target {target_min} min. Go.")


def stop(name: str, check_result=None) -> None:
    rec = RESULTS.get(name, {})
    if "started" in rec:
        rec["minutes"] = round((time.perf_counter() - rec.pop("started")) / 60, 2)
    if check_result is not None:
        rec["status"] = check_result.status
    RESULTS[name] = rec
    mins = rec.get("minutes")
    shown = f"{mins:.1f} min" if mins is not None else "not timed"
    print(f"⏱  {name}: {shown} (target {rec.get('target')} min), check: {rec.get('status', '-')}")


with Timer("demo", target_min=0.1):
    sum(i * i for i in range(100_000))

# %% [markdown]
# **Expected:** `⏱  demo: 0.0 min (target 0.1 min) - within target`.
#
# In the drills below, `start(...)` goes in the prompt cell and `stop(...)`
# wraps the grader, because you will spend minutes between the two cells. When
# you re-run the notebook later just to see results, the times will be tiny;
# that is fine, the check status is what is recorded.

# %% [markdown]
# ## 16.2 · Python drills
#
# These five come up again and again in ML engineer screens because each hides
# one idea: a dictionary pass, sorting first, a monotonic deque, a metric from
# first principles, and stratification. Data-flavoured wording, classic
# algorithms underneath.

# %% [markdown]
# > **Exercise 1.** *(target 5 min)* `dedupe_keep_latest(records, key, ts)`: `records`
# > is a list of dicts (e.g. CRM updates). Return one record per `key` value: the
# > one with the largest `ts`; on a tie in `ts`, the one that appears **later** in
# > the input. Output order = order of first appearance of each key. Do not
# > modify the input. Say the complexity before coding.

# %%
start("ex1", 5)


def dedupe_keep_latest(records: list[dict], key: str, ts: str) -> list[dict]:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex1", grade.ex1(dedupe_keep_latest))

# %% [markdown]
# > **Exercise 2.** *(target 6 min)* `merge_intervals(intervals)`: a customer's
# > subscription periods as `(start, end)` pairs, possibly overlapping and
# > unsorted (re-subscriptions, plan changes). Return the merged, sorted list of
# > `(start, end)` tuples. Periods that **touch** (`end == next start`) merge.
# > Values may be ints or dates.

# %%
start("ex2", 6)


def merge_intervals(intervals):
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex2", grade.ex2(merge_intervals))

# %% [markdown]
# > **Exercise 3.** *(target 8 min)* `sliding_window_max(values, k)`: the
# > maximum of every window of length `k` (e.g. peak weekly sessions over a
# > rolling k-week window). Return a list of length `len(values) - k + 1`. It must
# > be **O(n)**: the grader includes n = 200,000, k = 5,000 with a generous time
# > limit; `max(values[i:i+k])` in a loop is O(nk) and will not pass.

# %%
start("ex3", 8)

from collections import deque  # noqa: E402  (you will want it)


def sliding_window_max(values, k: int) -> list:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex3", grade.ex3(sliding_window_max))

# %% [markdown]
# > **Exercise 4.** *(target 4 min)* `precision_at_k(y_true, scores, k)`:
# > precision among the `k` highest-scored rows ("if the retention team can call
# > k customers, what fraction are real churners?"). Plain Python: no NumPy
# > sorting helpers, `sorted` is allowed. State what you do on `k <= 0`.

# %%
start("ex4", 4)


def precision_at_k(y_true, scores, k: int) -> float:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex4", grade.ex4(precision_at_k))

# %% [markdown]
# > **Exercise 5.** *(target 7 min)* `stratified_split(labels, test_frac, seed)`:
# > return `(train_indices, test_indices)` as lists, so that every class keeps its
# > ratio in the test set to within ±1 row, the two lists partition
# > `range(len(labels))`, and the same seed gives the same split. Standard library
# > only (`random.Random(seed)`).

# %%
start("ex5", 7)


def stratified_split(labels, test_frac: float, seed: int):
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex5", grade.ex5(stratified_split))

# %% [markdown]
# > 🧠 **Research bridge.** Exercise 5 is the same idea as keeping class balance
# > per subject in a BCI dataset. The interview follow-up is usually *"and if
# > rows are grouped?"*: then you stratify **groups**, not rows (scikit-learn's
# > `StratifiedGroupKFold`), exactly as you would never split one subject's
# > trials across train and test.

# %% [markdown]
# ## 16.3 · SQL drills on TaskFlow
#
# Each SQL exercise is a function that **returns the query string**. The grader
# runs it in DuckDB on the clean tables and compares the result with a reference
# query (row order is ignored unless the prompt says otherwise; numbers are
# compared with a small tolerance). Column names matter: use the ones in the
# prompt.
#
# Try your query interactively first. This helper shows the first rows:

# %%
def run(query: str, n: int = 5):
    return con.sql(query).df().head(n)


run("SELECT status, COUNT(*) AS n, SUM(amount_usd) AS usd FROM invoices GROUP BY status")

# %% [markdown]
# **Expected:** two statuses, `paid` (~53,200 invoices) and `failed` (~2,300).
#
# Schemas you need: `customers(customer_id, signup_date, plan, ...)`,
# `usage_weekly(customer_id, week_start, sessions, active_users, features_used)`
# (one row per active week; `week_start` is a Monday),
# `invoices(invoice_id, customer_id, invoice_date, amount_usd, status)`,
# `tickets(ticket_id, customer_id, created_at, category, resolution_hours)`.

# %% [markdown]
# > **Exercise 6.** *(target 5 min)* Monthly revenue from **paid** invoices, with
# > a running total. Columns `month` (first day of the month, from
# > `date_trunc('month', invoice_date)`), `revenue`, `running_revenue`, **ordered
# > by month**.

# %%
start("ex6", 5)


def monthly_revenue_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex6", grade.ex6(monthly_revenue_sql))

# %% [markdown]
# > **Exercise 7.** *(target 10 min)* Signup-month cohort retention. Cohort =
# > `date_trunc('month', signup_date)`, for customers who signed up from
# > 2023-01-01 up to (not including) 2024-10-01. A customer is retained in month
# > +k if they have **any** `usage_weekly` row whose `date_trunc('month', week_start)`
# > equals cohort month + k months. Columns `cohort_month`, `cohort_size`,
# > `ret_m1`, `ret_m2`, `ret_m3` (fractions between 0 and 1).

# %%
start("ex7", 10)


def cohort_retention_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex7", grade.ex7(cohort_retention_sql))

# %% [markdown]
# > **Exercise 8.** *(target 5 min)* Each customer's **third** invoice (by
# > `invoice_date`, ties broken by `invoice_id`), any status. Customers with fewer
# > than three invoices do not appear. Columns `customer_id`, `invoice_id`,
# > `invoice_date`, `amount_usd`.

# %%
start("ex8", 5)


def third_invoice_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex8", grade.ex8(third_invoice_sql))

# %% [markdown]
# > **Exercise 9.** *(target 12 min)* Gaps and islands. For each customer with
# > any usage, the longest run of consecutive **inactive** weeks (weeks with no
# > `usage_weekly` row) between their first and last active week. Customers who
# > never skipped a week get 0. Columns `customer_id`, `longest_inactive_streak`
# > (an integer number of weeks). Treat duplicate rows for the same week as one
# > week.
# >
# > Two valid approaches: `LEAD()` to the next active week and measure the gap,
# > or the classic "difference of two `ROW_NUMBER()`s" over a generated weekly
# > grid. Be ready to explain both.

# %%
start("ex9", 12)


def longest_inactive_streak_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex9", grade.ex9(longest_inactive_streak_sql))

# %% [markdown]
# > **Exercise 10.** *(target 4 min)* Median ticket resolution hours per
# > category. Columns `category`, `median_hours`. Use `MEDIAN` or
# > `quantile_cont(x, 0.5)` (not `AVG`: resolution times are skewed, and the
# > interviewer will ask why you chose the median).

# %%
start("ex10", 4)


def median_resolution_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


# %%
stop("ex10", grade.ex10(median_resolution_sql))

# %% [markdown]
# For reference, the answer to Exercise 10 is cheap to reveal and makes a
# useful sanity check of the data (why is one category so much faster?):

# %%
run(grade.REFERENCE_SQL[10])

# %% [markdown]
# **Expected:** five categories; `cancellation` tickets have a much lower median
# (about 5 hours) than the others (about 15). That is how the synthetic data was
# generated (cancellation requests are routed to a dedicated team); on real data
# it would be a question to ask the support team, not a conclusion.

# %% [markdown]
# ## 16.4 · Scoreboard and what to practise
#
# Your results today, saved with the date so that next week's run can be
# compared:

# %%
import pandas as pd  # noqa: E402

board = pd.DataFrame([
    {"drill": name, "minutes": rec.get("minutes"), "target": rec.get("target"),
     "status": rec.get("status", "-")}
    for name, rec in RESULTS.items() if name.startswith("ex")
])
board["within_target"] = (board["status"] == "pass") & (board["minutes"] <= board["target"])
print(board.to_string(index=False))

drills_dir = ART / "drills"
drills_dir.mkdir(parents=True, exist_ok=True)
out = drills_dir / f"{date.today().isoformat()}.json"
out.write_text(json.dumps(board.to_dict(orient="records"), indent=2, default=str))
print("\npassed:", int((board["status"] == "pass").sum()), "of", len(board), "->", out)

# %% [markdown]
# **Expected:** ten rows. In a fresh "Run all" every status is `todo`, every
# time is near zero and `within_target` is `False` (a drill only counts when it
# passes). After you work through the drills, statuses become `pass`
# and the minutes are real.
#
# ### A 30-minute daily drill plan for weeks 2-4
#
# Fluency is the gap five days cannot close (see the README's "Five days is a
# start" table). Thirty minutes a day, every day, closes it faster than a weekend
# cram:
#
# | Minutes | What | How |
# |---|---|---|
# | 0-5 | Warm-up | Re-solve one drill from your repeat list **from a blank cell**, timed |
# | 5-20 | One new problem | Alternate days: Python (arrays/hashing, intervals, heaps, two pointers, sliding window) and SQL (window functions, cohort/retention, funnels, gaps-and-islands, top-N per group) |
# | 20-27 | Explain it | Say the approach and complexity out loud, or record 2 minutes on your phone. Note one thing you would say better |
# | 27-30 | Log it | Add a row to `artifacts/drills/`: problem, minutes, passed, one-line lesson |
#
# Week by week:
#
# * **Week 2:** re-run this notebook (fresh cells) until all ten pass within
#   target. Add 5 SQL problems on joins + `GROUP BY` edge cases (NULLs, `COUNT(col)` vs `COUNT(*)`, `LEFT JOIN` filters in `WHERE` vs `ON`).
# * **Week 3:** ML-flavoured coding: implement ROC-AUC from ranks, a train/valid
#   split by time, a k-means step, a batched inference loop. Explain each
#   complexity. Two SQL problems per day on your capstone's own data.
# * **Week 4:** mixed, interview-length sessions: one 25-minute problem + 5
#   minutes of follow-ups you ask yourself ("what if it doesn't fit in memory?",
#   "what if the data arrives as a stream?").
#
# Track the trend, not single days: the goal is that a drill that took 15
# minutes in week 1 takes 5 in week 4.

# %% [markdown]
# ## 🧮 Key equations
#
# **Precision at k:**
#
# $$
# \text{precision@}k = \frac{1}{k} \sum_{i \in \text{top-}k} y_i
# $$
#
# *Example:* scores ranked, labels of the top 4 are 1, 0, 1, 0: precision@4 = 0.5.
#
# **Cohort retention at month $+k$:**
#
# $$
# \text{ret}_k(c) = \frac{\lvert\{\text{customers of cohort } c \text{ active in month } c + k\}\rvert}{\lvert \text{cohort } c \rvert}
# $$
#
# *Example:* 120 sign-ups in March, 96 with any usage in April: $\text{ret}_1 = 0.80$.
#
# **Why $O(nk)$ fails the sliding-window drill:** $n = 200{,}000$,
# $k = 5{,}000$ gives $10^9$ comparisons (minutes in Python), against
# $\approx 2n = 4 \times 10^5$ operations for an $O(n)$ method.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | SQL check: `missing column(s) [...]` | column not aliased as the prompt asks | add `AS month`, `AS ret_m1`, ... |
# | Ex 6: "rows must be ordered by ['month']" | no `ORDER BY` in the outer query | a running total needs `ORDER BY month` in the window **and** the result |
# | Ex 7 retention > 1 or cohort_size too big | joining usage multiplied rows before counting | `COUNT(DISTINCT customer_id)` or aggregate per customer first |
# | Ex 9 off by one week | counting the gap as `days / 7` instead of `days / 7 - 1` | consecutive active weeks are 7 days apart and have **0** inactive weeks between them |
# | Ex 9 `Binder Error` on `generate_series` | DuckDB needs timestamps or dates, not mixed types | use `INTERVAL 7 DAY` as the step with `week_start` bounds |
# | Ex 3 times out | O(nk): slicing or `max()` inside the loop | deque of indices; each index is pushed and popped at most once |
# | Ex 5 fails "±1 row" | splitting all rows at once then hoping | split **each class** separately and concatenate |
# | Timer shows 0.0 min after work | you re-ran the `stop` cell without the `start` cell | the timer measures from the last `start(...)`; re-run it before you begin |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. What is the time and space complexity of your sliding-window maximum, and why is each index processed at most twice?
# 2. `ROW_NUMBER`, `RANK` and `DENSE_RANK`: which would you use for "each customer's third invoice", and what happens on ties with each?
# 3. Your cohort-retention query returns 1.4 for one cohort. What is the most likely bug, and how would you find it in one query?
# 4. Explain the "difference of two row numbers" trick for gaps and islands to someone who has never seen it.
# 5. Why report the median resolution time rather than the mean, and what would you add to make the comparison between categories honest?
#
# ➡️ **Next:** [`17_system_design_and_mock_interview.ipynb`](17_system_design_and_mock_interview.ipynb) — the other half of the loop: designing an ML system out loud, and a scored mock interview.
