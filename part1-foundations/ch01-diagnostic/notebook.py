# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 1 · Skills diagnostic
#
# **Day 0 · ~60 minutes · CPU only**
#
# The bootcamp is five days long and you will not have time to do everything
# with the same care. This diagnostic measures the *engineering* skills the
# course teaches - not your modelling, which is assumed - so you know before
# Day 1 where to slow down and where you can move quickly.
#
# By the end you will be able to:
#
# 1. see, from your own work, which of Python, pandas, SQL, testing, evaluation judgement and APIs are solid and which are rusty;
# 2. rate yourself honestly on the skills that are hard to auto-grade (deployment, monitoring, explaining under questioning);
# 3. name the notebooks that deserve extra time for you.
#
# **You will produce:** `artifacts/diagnostic.json`, your profile, to compare against when you retake this at the end of week 4.
#
# ### How to use it
#
# * **Timebox each exercise to 8-10 minutes.** Set a timer. If it rings, stop,
#   run the check, and move on: an unfinished exercise is information, not failure.
# * **No solutions, no AI assistant, no searching for the answer.** Looking up
#   the name of a pandas or Pydantic function in the official docs is fine -
#   that is what you would do at work. Pasting the task into a chatbot measures
#   the chatbot.
# * **Score honestly.** Nobody sees this but you. A flattering profile sends
#   you to the wrong notebooks.
# * Run the setup cell first, then work top to bottom. Every check prints
#   [verified] / / and never stops the notebook.

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
# --- Catch up: nothing from other notebooks is needed here -----------------
from grading import ch01 as grade

results = []  # every check below appends its result here; the profile reads it

# %% [markdown]
# ## Key terms
#
# Plain definitions of the words the exercises use. They tell you what a word
# means, not the answer to any question.
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Sentinel value** | a special number that stands for "unknown" instead of a real measurement | `sessions = -1` in the usage log |
# | **`LEFT JOIN`** | keep every row of the left table, with NULLs where the right table has no match | customers with no invoices still appear |
# | **Window function** | a SQL value computed from related rows without collapsing them | "the latest invoice per customer" |
# | **Positive rate** | share of rows whose label is 1 | 5% of customers churn |
# | **PR-AUC** | how well a score ranks positives above negatives, focused on the top of the list | the headline metric from Day 1 on |
# | **Calibrated probability** | a score that matches reality: of accounts scored 0.2, about 20% churn | needed before a cost-based threshold |
# | **Bootstrap** | resample the evaluation set with replacement to see how much a metric moves | "is 0.745 really above 0.741?" |
# | **Leakage** | information in training or evaluation that would not exist at prediction time | a column exported after the snapshot date |
# | **Schema validation** | checking a request against declared field types and limits before using it | Pydantic rejects an unknown field |
# | **Test that fails on a bug** | an assertion that a correct function passes and a wrong one does not | the grader runs your test against hidden buggy versions |

# %% [markdown]
# ## 0.1 · Python
#
# > **Exercise 1.** Write `top_k_frequent(items, k)` that returns a list of the
# > `k` most frequent items, most frequent first. Ties are broken by the item
# > itself in ascending order (alphabetical for strings). If there are fewer than
# > `k` distinct items, return all of them; `k = 0` or an empty input returns `[]`.
# >
# > `top_k_frequent(["b", "a", "c", "a", "b"], 2)` → `["a", "b"]`

# %%
def top_k_frequent(items, k):
    # YOUR CODE HERE
    raise NotImplementedError


results.append(grade.ex1(top_k_frequent))

# %% [markdown]
# ## 0.2 · pandas
#
# A weekly usage log, as it comes out of a real logging pipeline: a duplicated
# row, a `-1` that means "unknown", a missing value, and a week in which nobody
# was active. Look at it before you write anything.

# %%
import pandas as pd

usage_example = pd.DataFrame({
    "customer_id": ["C1", "C1", "C2", "C3", "C1", "C2", "C3", "C2"],
    "week_start": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01",
                                  "2024-01-08", "2024-01-08", "2024-01-08", "2024-01-15"]),
    "sessions": [4, 4, 0, -1, 2, 7, None, 0],
})
usage_example

# %% [markdown]
# > **Exercise 2.** Write `weekly_active(usage)` that returns a `pd.Series`
# > indexed by `week_start` (sorted, one entry for **every** week in the input)
# > whose values are the number of **distinct** customers with `sessions > 0`
# > that week. Duplicates must not be double-counted; `-1` and missing values
# > are not activity; a week with nobody active shows `0`. Do not modify the input.
# >
# > On `usage_example` the answer is `1, 2, 0` for the three weeks.

# %%
def weekly_active(usage: pd.DataFrame) -> pd.Series:
    # YOUR CODE HERE
    raise NotImplementedError


results.append(grade.ex2(weekly_active))

# %% [markdown]
# ## 0.3 · SQL
#
# DuckDB runs SQL directly on pandas DataFrames: register a DataFrame under a
# table name, run a query, get a DataFrame back. No server, no setup. This is
# how the whole course uses SQL.

# %%
import duckdb

customers_example = pd.DataFrame({"customer_id": ["A", "B", "C"], "plan": ["basic", "pro", "team"]})
invoices_example = pd.DataFrame({
    "customer_id": ["A", "A", "B"],
    "invoice_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15"]),
    "amount_usd": [29.0, 31.0, 79.0],
    "status": ["paid", "failed", "paid"],
})

con = duckdb.connect()
con.register("customers", customers_example)
con.register("invoices", invoices_example)
con.execute("SELECT customer_id, COUNT(*) AS n_invoices FROM invoices GROUP BY customer_id ORDER BY customer_id").df()

# %% [markdown]
# **Expected:** two rows, A with 2 invoices and B with 1. Customer C has no
# invoices, so a plain `GROUP BY` over `invoices` cannot see it.
#
# > **Exercise 3.** Write `customer_summary_sql()` that returns a SQL string
# > over `customers(customer_id, plan)` and
# > `invoices(customer_id, invoice_date, amount_usd, status)` producing one row
# > **per customer** with columns `customer_id`, `plan`, `latest_amount_usd`
# > (the amount of the customer's most recent invoice by `invoice_date`) and
# > `failed_invoices` (how many of their invoices have `status = 'failed'`).
# > Customers with no invoices must appear, with `latest_amount_usd` NULL and
# > `failed_invoices` 0. Try it on the example with
# > `con.execute(customer_summary_sql()).df()`.

# %%
def customer_summary_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


results.append(grade.ex3(customer_summary_sql))

# %% [markdown]
# ## 0.4 · Testing
#
# Here is a small, **correct** function. Read its specification carefully.

# %%
def rolling_mean_last_n(values: list, n: int) -> list:
    """Return a list the same length as ``values`` whose element i is the mean of
    the last n values ending at position i. At the start, where fewer than n
    values exist, average the ones there are. The input list is not modified."""
    out = []
    for i in range(len(values)):
        window = values[max(0, i - n + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


rolling_mean_last_n([1, 2, 3, 4, 5], 3)

# %% [markdown]
# **Expected:** `[1.0, 1.5, 2.0, 3.0, 4.0]`.
#
# > **Exercise 4.** Write `test_rolling_mean(fn)`: a test that takes an
# > implementation `fn` (with the same signature) and checks it using plain
# > `assert` statements. The grader runs your test against the correct function
# > above, where it must **pass**, and against three subtly buggy
# > implementations it keeps hidden, where it must **fail**. You are graded on
# > whether your test catches bugs - which is exactly what a test is for.
# > (Compare floats with `math.isclose` rather than `==`.)

# %%
import math


def test_rolling_mean(fn):
    # YOUR CODE HERE - call fn on inputs you choose and assert on the results
    raise NotImplementedError


results.append(grade.ex4(test_rolling_mean))

# %% [markdown]
# ## 0.5 · Evaluation judgement
#
# Six short scenarios. Pick one letter each. The grader tells you how many are
# right and which question numbers to revisit, but not the right answers -
# notebooks 04-07 do that properly.
#
# **Q1 · A feature.** You predict monthly churn from snapshots taken on the
# first of each month in 2023. A colleague adds `days_since_last_login`,
# computed from the `users` table as it was exported this morning. Validation
# PR-AUC jumps from 0.16 to 0.71.
# (a) Great feature; ship it. (b) It leaks: it is computed from data after each
# snapshot date. (c) It is only a problem for tree models. (d) Standardising it
# would fix the jump.
#
# **Q2 · A split.** The model will score existing customers every month for
# "will cancel next month". Which validation design matches that use?
# (a) Random 80/20 split of all (customer, month) rows. (b) Stratified 5-fold
# CV over rows. (c) Train on earlier months, validate on later months, with a
# gap at least as long as the label window. (d) Group K-fold by customer only.
#
# **Q3 · A metric.** 5% of customers churn. The retention team can call the
# 200 highest-scored accounts each month. Which metric best reflects the task?
# (a) Precision among the top 200 (and PR-AUC as the threshold-free summary).
# (b) Accuracy. (c) ROC-AUC alone. (d) Recall at threshold 0.5.
#
# **Q4 · A threshold.** Scores are calibrated probabilities. A retention call
# costs \$30; a correctly targeted churner is worth \$150 in expectation; missed
# churners and ignored loyal customers cost nothing extra. Whom do you call?
# (a) Everyone with p > 0.5. (b) The threshold that maximises F1.
# (c) The threshold where precision equals recall. (d) Everyone with p > 1/6.
#
# **Q5 · A comparison.** On one validation set (7,000 rows, ~360 positives),
# model A has ROC-AUC 0.741 and model B 0.745, each trained once with one seed.
# (a) Ship B, it is better. (b) The gap is probably within noise: estimate it
# with a paired bootstrap and across seeds/folds before claiming anything.
# (c) Always ship the simpler model. (d) Evaluate both on the test set
# repeatedly until the difference is significant.
#
# **Q6 · A production symptom.** On the morning after deployment, before
# customers have had any time to change behaviour, the distribution of
# production scores looks very different from the validation scores. Most likely:
# (a) training/serving skew - features are computed or encoded differently
# online than in training. (b) concept drift. (c) label delay.
# (d) the model overfitted.
#
# > **Exercise 5.** Fill in `answers()` with one letter per question.

# %%
def answers() -> dict:
    # YOUR CODE HERE: return {1: "?", 2: "?", 3: "?", 4: "?", 5: "?", 6: "?"}
    raise NotImplementedError


results.append(grade.ex5(answers))

# %% [markdown]
# ## 0.6 · APIs
#
# A prediction service is only as reliable as the validation at its front
# door. Pydantic models declare what a valid request looks like, and reject
# everything else with a precise error. Here is the pattern on a different
# object:

# %%
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0)
    label: str | None = None


print(Point(x=1.5))
try:
    Point(x=-1, colour="red")
except ValidationError as exc:
    print(exc.error_count(), "errors:", [e["loc"] for e in exc.errors()])

# %% [markdown]
# **Expected:** `x=1.5 label=None`, then `2 errors` for `x` (below 0) and
# `colour` (an extra field).
#
# > **Exercise 6.** Write `make_model()` that defines and **returns the class**
# > `Order` with fields `order_id` (non-empty string), `quantity` (integer from 1
# > to 1000 inclusive), `unit_price` (float strictly greater than 0) and
# > `coupon` (optional string, default `None`). Unknown fields must be rejected.

# %%
def make_model():
    # YOUR CODE HERE: define class Order(BaseModel): ... and return Order
    raise NotImplementedError


results.append(grade.ex6(make_model))

# %% [markdown]
# ## 0.7 · Self-rating: what is hard to auto-grade
#
# Deployment, operations and communication are mostly learned by doing them
# for real, so a notebook cannot test them in ten minutes. Rate each statement:
#
# **1** = never done it · **2** = done it with substantial help · **3** = done it on my own · **4** = could teach it
#
# | id | area | statement |
# |---|---|---|
# | `deploy_package` | Deployment | I have packaged a Python project (`pyproject.toml`) and installed it in a fresh environment |
# | `deploy_docker` | Deployment | I have written a Dockerfile for a Python service and explained each line |
# | `deploy_ci` | Deployment | I have set up CI that runs tests on every push |
# | `monitor_health` | Monitoring | I have added health checks and structured logs to a service |
# | `monitor_drift` | Monitoring | I have monitored a deployed model for drift or degraded performance |
# | `monitor_rollback` | Monitoring | I have planned or performed a rollback of a bad release |
# | `comm_explain` | Communication | I can explain a modelling decision to a non-specialist in two minutes |
# | `comm_defend` | Communication | I can defend an evaluation choice when an interviewer pushes back |
# | `comm_design` | Communication | I have done a timed ML system design interview (real or mock) |
#
# Replace each `None` with your rating. "Done it" means you did it, not that
# you watched a tutorial.

# %%
self_ratings = {
    "deploy_package": None,
    "deploy_docker": None,
    "deploy_ci": None,
    "monitor_health": None,
    "monitor_drift": None,
    "monitor_rollback": None,
    "comm_explain": None,
    "comm_defend": None,
    "comm_design": None,
}

# %% [markdown]
# > **Exercise 7 (written).** [open-ended] In three sentences: describe the most rigorous
# > evaluation you have done in your research, and name its main threat to
# > validity. Write it as you would say it to an interviewer. Time yourself: two
# > minutes. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 0.8 · Your profile
#
# The profile reads the check results above and your self-ratings, prints
# area → status → which notebooks to prioritise, and saves
# `artifacts/diagnostic.json`. Statuses: **strong** (passed / rated ≥ 3),
# **review** (close), **priority** (failed or rated < 2), **not attempted**.

# %%
profile = grade.summary(results, self_ratings)

# %% [markdown]
# **Expected (unattempted):** every graded area shows `not attempted`, every
# self-rated area `not rated`, and the list of notebooks to prioritise is the
# whole course. Once you have done the exercises it becomes your plan.
#
# How to use it:
#
# * **priority** areas: do those notebooks first each day, do every exercise
#   without the solutions, and attempt the checkpoint questions aloud.
# * **strong** areas: go faster, but still do the checkpoint questions. Being
#   able to *do* something and being able to *explain* it under questioning are
#   different skills.
# * **Retake this diagnostic at the end of week 4** of the four-week study plan
#   (`docs/guides/study-plan.md`) with the same rules and compare the two
#   `diagnostic.json` files. The exercises will feel easier; the useful
#   question is whether you now finish each within the timebox.
#
# > **Research bridge.** This is a pre-test / post-test design with you as
# > the only subject. The obvious threat to validity is practice: you will have
# > seen these exact items. Mitigate it the way you would in a study - judge the
# > retake on speed and on the *written* answer, which you cannot memorise.

# %% [markdown]
# ### Where the profile sends you
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/roadmap.svg" alt="Five-day roadmap: reproducible projects, evaluation, serving, operations, demonstration, then capstones" width="100%">
#
# *What to notice:* each day's notebooks build on the previous day's artifacts, but every notebook can rebuild what it needs, so a "priority" area can be started in any order.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `NameError: name 'results' is not defined` | the catch-up cell was skipped | run the setup and catch-up cells, then the exercise cells |
# | The profile shows an old result after you fixed an exercise | the check cell was not re-run | re-run the check cell; the newest result for each exercise wins |
# | Exercise 3: `Catalog Error: Table with name ... does not exist` | you queried a table other than `customers` / `invoices` | the grader registers exactly those two names |
# | Exercise 4 says your test fails the *correct* function | an expected value in your test is wrong, or exact float `==` | recompute by hand; compare with `math.isclose` |
# | Exercise 6: `PydanticUserError` about a non-annotated attribute | a field without a type annotation | every field needs `name: type = ...` |
# | `diagnostic.json` disappeared | the Colab runtime was recycled | set `USE_DRIVE = True` or download it; the retake needs it |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why does a test that only checks one "happy path" example miss most bugs? Which cases would you always add?
# 2. When does `LEFT JOIN` differ from `JOIN`, and how can a join silently *multiply* rows?
# 3. Why is accuracy a poor headline metric at a 5% positive rate, and what would you report instead?
# 4. What does validating requests at an API boundary protect you from that validating training data does not?
# 5. Which of your areas came out weakest, and what will you do differently in those notebooks?
#
# **Next chapter:** [`01_reproducible_project.ipynb`](../ch02-reproducible-project/notebook.ipynb) — start the project: pin code, data, environment and randomness so every number you produce can be reproduced.
