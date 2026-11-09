# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 4 · SQL for ML
#
# **Part I · Chapter 4 · about 55 minutes · CPU only**
#
# In most companies the training table does not exist until someone writes the
# SQL that builds it, and most silent leakage is introduced right there: one
# join on the wrong date, one week counted before it was over. This notebook
# builds the project's features and labels in SQL, one step at a time, and then
# checks your version against the package's.
#
# By the end you will be able to:
#
# 1. query pandas DataFrames with SQL in DuckDB;
# 2. aggregate with `GROUP BY` and `FILTER`, and explain inner vs left joins;
# 3. recognise and prevent the join **fan-out** bug;
# 4. use window functions: `LAG`, `ROW_NUMBER`/`QUALIFY`, and rolling `RANGE` frames;
# 5. build point-in-time features and a label for a snapshot date T, and explain every inequality.
#
# **You will produce:** no file; the feature table itself is cached by `bootstrap.ensure_features()` for later notebooks.

# %% [markdown]
# Read first: [Chapter 4 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: the clean tables from Chapter 3 ---------------------------
import pandas as pd

from churnlab import bootstrap
from grading import ch04 as grade

tables = bootstrap.load_clean()  # raw extract, verified and cleaned (Chapter 3)
{name: df.shape for name, df in tables.items()}

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **DuckDB** | an analytical SQL database that runs inside Python, no server | `con.execute(sql).df()` |
# | **Inner vs left join** | inner keeps matching pairs only; left keeps every left row, NULL where no match | customers with no tickets survive only a `LEFT JOIN` |
# | **One-to-many** | one row on one side matches many on the other | one customer, 41 usage weeks |
# | **Fan-out** | a join that repeats rows, inflating sums and counts silently | 212 sessions reported as 1,484 |
# | **Window function** | a value computed from related rows (the window) without collapsing them | `LAG(sessions)` over a customer's weeks |
# | **`QUALIFY`** | filters on a window function, like `HAVING` filters on an aggregate | keep `ROW_NUMBER() = 1` per customer |
# | **Frame (`ROWS` / `RANGE`)** | which rows a window includes: a count of rows, or a span of values such as dates | `RANGE BETWEEN INTERVAL 21 DAY PRECEDING` |
# | **Snapshot date T** | the day a prediction is made | 2024-03-01 |
# | **Point-in-time feature** | a feature built only from rows that existed before T | usage from weeks that ended by T |
# | **Half-open interval [a, b)** | includes a, excludes b | the label window [T, T + 30 days) |
# | **`COALESCE`** | the first non-NULL argument | "no tickets" becomes 0 after a left join |

# %% [markdown]
# ## 4.1 · DuckDB: SQL on DataFrames
#
# DuckDB is an in-process analytical database: no server, nothing to install
# beyond `pip`. `register` exposes a DataFrame as a table (without copying it),
# `execute(...).df()` returns a DataFrame. The SQL you write here is the same
# you would write against Postgres, BigQuery or Snowflake, give or take a
# function name.

# %%
import duckdb

con = duckdb.connect()
for name, df in tables.items():
    con.register(name, df)


def sql(query: str) -> pd.DataFrame:
    return con.execute(query).df()


sql("SELECT plan, COUNT(*) AS customers, AVG(seats) AS avg_seats FROM customers GROUP BY plan ORDER BY plan")

# %% [markdown]
# **Expected:** three plans (`basic`, `pro`, `team`, already lower-cased by
# cleaning) with about 2,465 / 1,757 / 778 customers.

# %% [markdown]
# ## 4.2 · Aggregates, and `FILTER`
#
# `COUNT(*) FILTER (WHERE ...)` counts only the rows matching a condition,
# inside an ordinary `GROUP BY`. It replaces the `SUM(CASE WHEN ... THEN 1
# ELSE 0 END)` idiom you will still see in older SQL.

# %%
sql("""
SELECT date_trunc('month', invoice_date)            AS month,
       COUNT(*)                                      AS invoices,
       COUNT(*) FILTER (WHERE status = 'failed')     AS failed,
       ROUND(SUM(amount_usd))                        AS revenue_usd
FROM invoices
WHERE invoice_date < DATE '2023-05-01'
GROUP BY month
ORDER BY month
""")

# %% [markdown]
# **Expected:** four months; around 1,550-1,900 invoices a month, of which
# 60-90 failed.

# %% [markdown]
# ## 4.3 · Joins, and the fan-out bug
#
# An **inner** join keeps only matching pairs; a **left** join keeps every row
# of the left table, with NULLs where there is no match.

# %%
sql("""
SELECT
  (SELECT COUNT(*) FROM customers c JOIN      tickets t USING (customer_id)) AS inner_rows,
  (SELECT COUNT(*) FROM customers c LEFT JOIN tickets t USING (customer_id)) AS left_rows,
  (SELECT COUNT(*) FROM customers c LEFT JOIN tickets t USING (customer_id)
    WHERE t.ticket_id IS NULL)                                              AS customers_without_tickets
""")

# %% [markdown]
# **Expected:** 22,354 inner rows (one per ticket), 22,888 left rows: the
# 534 customers who never filed a ticket survive only in the left join. For
# features, that matters: "zero tickets" is information, and an inner join
# deletes those customers.
#
# Now the most common silent SQL bug in feature pipelines. You want, per
# customer, total sessions **and** number of tickets, so you join both tables
# to the customer and aggregate:

# %%
sql("""
SELECT c.customer_id,
       SUM(u.sessions)     AS total_sessions,
       COUNT(t.ticket_id)  AS tickets
FROM customers c
JOIN      usage_weekly u USING (customer_id)
LEFT JOIN tickets      t USING (customer_id)
WHERE c.customer_id = 'C00002'
GROUP BY c.customer_id
""")

# %% [markdown]
# Looks plausible. Now compute each number on its own:

# %%
sql("""
SELECT
  (SELECT SUM(sessions) FROM usage_weekly WHERE customer_id = 'C00002') AS true_sessions,
  (SELECT COUNT(*)      FROM usage_weekly WHERE customer_id = 'C00002') AS usage_rows,
  (SELECT COUNT(*)      FROM tickets      WHERE customer_id = 'C00002') AS true_tickets
""")

# %% [markdown]
# **Expected:** the joined query said 1,484 sessions and 287 tickets; the truth
# is 212 sessions over 41 weeks, and 7 tickets. 1,484 = 212 × 7 and 287 = 41 × 7.
#
# Joining two **one-to-many** relationships on the same key produces every
# (week, ticket) pair: each usage row is repeated once per ticket and each
# ticket once per week. No error, no warning, plausible-looking numbers. Over
# the whole table:
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-join-fanout.svg" alt="Joining usage and tickets to a customer before aggregating creates 287 row pairs and inflates sessions to 1,484; aggregating each table to one row first gives 212 sessions and 7 tickets" width="100%">
#
# *What to notice:* the bug is in the order of operations, not in any single clause; the same tables and the same SUM give the right answer once each table is collapsed to one row per key first.
#
# > **Intuition.** Fan-out is a multiplication table: 41 weeks along one
# > side, 7 tickets along the other, and the join fills in all 287 cells.
# > Any sum over that table counts each week 7 times. *Where it breaks down:*
# > with a one-to-one or many-to-one join there is no table to fill, which is
# > why the bug only appears when **two** one-to-many tables meet.

# %%
sql("""
SELECT
  (SELECT SUM(sessions) FROM usage_weekly)                                      AS true_total,
  (SELECT SUM(u.sessions) FROM usage_weekly u JOIN tickets t USING (customer_id)) AS fanned_out_total
""")

# %% [markdown]
# **Expected:** about 1.70 million true sessions vs about 11.0 million after
# the fan-out, a 6.5× inflation that differs per customer (it scales with
# their ticket count), so it does not even cancel out in a model.
#
# The fix: **aggregate each table to one row per key first, then join**:

# %%
sql("""
WITH s AS (SELECT customer_id, SUM(sessions) AS total_sessions FROM usage_weekly GROUP BY customer_id),
     k AS (SELECT customer_id, COUNT(*)      AS tickets        FROM tickets      GROUP BY customer_id)
SELECT c.customer_id, s.total_sessions, COALESCE(k.tickets, 0) AS tickets
FROM customers c
LEFT JOIN s USING (customer_id)
LEFT JOIN k USING (customer_id)
WHERE c.customer_id = 'C00002'
""")

# %% [markdown]
# **Expected:** 212 sessions and 7 tickets. `churnlab.features` builds every
# feature group this way: one aggregate table per source, joined at the end.

# %% [markdown]
# ## 4.4 · Window functions
#
# A window function computes a value for each row from a set of related rows
# (its *window*), without collapsing them like `GROUP BY` does.
# `PARTITION BY` says which rows are related; `ORDER BY` orders them.
#
# ### `LAG`: compare a row with the previous one

# %%
sql("""
SELECT customer_id, week_start, sessions,
       LAG(sessions) OVER (PARTITION BY customer_id ORDER BY week_start) AS prev_sessions
FROM usage_weekly
WHERE customer_id = 'C00002'
ORDER BY week_start
LIMIT 5
""")

# %% [markdown]
# **Expected:** `prev_sessions` is NULL in the first row (there is no previous
# week) and then repeats the previous row's `sessions` (1, 3, 2, 5).

# %% [markdown]
# > **Exercise 1.** Write `weekly_change_sql()` returning SQL over
# > `usage_weekly` with one row per input row and the columns `customer_id`,
# > `week_start`, `sessions`, `prev_sessions` (the customer's previous logged
# > week) and `sessions_change` (= `sessions - prev_sessions`, NULL when either
# > is NULL).

# %%
def weekly_change_sql() -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(weekly_change_sql)

# %% [markdown]
# Note "previous **logged** week": inactive weeks are not in the table, so the
# previous row can be two or more weeks back. Whether that is what you want is
# a modelling decision - the same detail matters for rolling windows below.
#
# ### `ROW_NUMBER` and `QUALIFY`: the latest row per group
#
# "Latest invoice per customer" is a top-1-per-group problem. Number the rows
# inside each customer by date, newest first, and keep number 1. `QUALIFY`
# filters on a window function the way `HAVING` filters on an aggregate.

# %%
sql("""
SELECT customer_id, invoice_date, amount_usd, status
FROM invoices
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) = 1
ORDER BY customer_id
LIMIT 3
""")

# %% [markdown]
# **Expected:** one row per customer, e.g. `C00000`'s last invoice is from
# February 2023 (it churned then). On databases without `QUALIFY`, wrap the
# query: `SELECT * FROM (... ROW_NUMBER() ... AS rn) WHERE rn = 1`.
#
# For a feature, "latest" must mean "latest **before the snapshot date**".

# %% [markdown]
# > **Exercise 2.** Write `latest_invoice_sql(as_of)` returning SQL that gives,
# > for every customer with at least one invoice **strictly before** `as_of`
# > (a date string such as `"2024-03-01"`), that latest invoice's
# > `customer_id`, `invoice_date` and `amount_usd`. Build the date into the SQL
# > with an f-string, e.g. `TIMESTAMP '{as_of}'`. The grader also checks your
# > amounts against `mrr_usd` from the package's feature table.

# %%
def latest_invoice_sql(as_of: str) -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(latest_invoice_sql)

# %% [markdown]
# ### Rolling sums: `ROWS` vs `RANGE`
#
# "Sessions in the last 4 weeks" as a window. `ROWS BETWEEN 3 PRECEDING`
# means "this row and the 3 rows before it"; `RANGE BETWEEN INTERVAL 21 DAY
# PRECEDING` means "rows whose `week_start` is within 21 days before this
# one". They agree only when no week is missing. Customer `C00008` has gaps:

# %%
sql("""
SELECT week_start, sessions,
       SUM(sessions) OVER (PARTITION BY customer_id ORDER BY week_start
                           ROWS  BETWEEN 3 PRECEDING AND CURRENT ROW)              AS last_4_rows,
       SUM(sessions) OVER (PARTITION BY customer_id ORDER BY week_start
                           RANGE BETWEEN INTERVAL 21 DAY PRECEDING AND CURRENT ROW) AS last_28_days
FROM usage_weekly
WHERE customer_id = 'C00008' AND week_start BETWEEN DATE '2024-03-11' AND DATE '2024-05-06'
ORDER BY week_start
""")

# %% [markdown]
# **Expected:** the two columns differ in the rows just after a missing week
# (e.g. after 2024-04-01, which is not in the table): `ROWS` reaches back more
# than 28 days to find four rows; `RANGE` stays within 28 days. For "sessions in
# the last 4 weeks", `RANGE` is the correct definition.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-rows-vs-range.svg" alt="With one week missing, ROWS 3 PRECEDING reaches back five weeks to collect four rows, while RANGE 21 DAY PRECEDING stays within four weeks and collects three" width="100%">
#
# *What to notice:* the two frames differ only when a week is missing, which is exactly the case casual testing skips.
#
# > **Intuition.** `ROWS 3 PRECEDING` is "my last four diary entries";
# > `RANGE 21 DAY PRECEDING` is "everything I wrote in the last four weeks".
# > For someone who skips a week, the last four entries go back further.
# > *Where it breaks down:* `RANGE` needs an ordering column you can subtract
# > (dates, numbers); for "the previous event, however long ago", `ROWS` or
# > `LAG` is the right tool.

# %% [markdown]
# ## 4.5 · Point-in-time features for one snapshot, step by step
#
# The prediction task: at a monthly snapshot date **T**, for every account still
# active at T, predict whether it cancels in **[T, T + 30 days)**. Every
# feature may use only information available **before T**.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/point-in-time.svg" alt="Point-in-time features: features use rows before snapshot T, the label uses the 30 days after T" width="100%">
#
# We build it for one snapshot, T = 2024-03-01, then compare with the package.

# %%
T = "2024-03-01"

# %% [markdown]
# ### Step 1 · Who is in the table at T?
#
# Accounts that signed up before T and had not churned before T. Accounts that
# churned earlier are not scored (there is nobody left to call); accounts that
# signed up after T did not exist yet.

# %%
con.execute(f"""
CREATE OR REPLACE TEMP TABLE population AS
SELECT customer_id, plan, signup_date,
       date_diff('day', signup_date, TIMESTAMP '{T}') AS tenure_days
FROM customers
WHERE signup_date < TIMESTAMP '{T}'
  AND (churn_date IS NULL OR churn_date >= TIMESTAMP '{T}')
""")
sql("SELECT COUNT(*) AS accounts_active_at_T FROM population")

# %% [markdown]
# **Expected:** 2,246 accounts.
#
# > **Intuition.** A snapshot is a photo taken at midnight on day T. A
# > feature may only use what is in the photo; the label is what you learn
# > by visiting again 30 days later. *Where it breaks down:* a photo is
# > instant, but a usage week is a 7-day film: it is only "in the photo" once
# > it has finished, which is step 2.
#
# ### Step 2 · Which weeks are "before T"?
#
# A usage row covers the week **starting** on `week_start`. Its numbers are
# known only once the week is over, at `week_start + 7 days`. So the rule is
# `week_start + 7 days <= T`, not `week_start < T`:

# %%
sql(f"""
SELECT DISTINCT week_start,
       week_start + INTERVAL 7 DAY                   AS week_end,
       week_start < TIMESTAMP '{T}'                  AS naive_rule,
       week_start + INTERVAL 7 DAY <= TIMESTAMP '{T}' AS week_complete
FROM usage_weekly
WHERE week_start BETWEEN TIMESTAMP '{T}' - INTERVAL 14 DAY AND TIMESTAMP '{T}' + INTERVAL 7 DAY
ORDER BY week_start
""")

# %% [markdown]
# **Expected:** the week starting 2024-02-26 ends on 2024-03-04. The naive rule
# includes it, the correct rule does not. T is a Friday, so the naive rule
# would leak **3 days of the future** (Mar 1-3) into every usage feature, for
# every account. (When T falls on a Monday the two rules agree, which is why
# this bug survives casual testing.)
#
# ### Step 3 · Features from completed weeks and earlier events
#
# Each feature group is aggregated to one row per customer **first** (no
# fan-out), then left-joined to the population, with `COALESCE` turning "no
# rows" into 0:

# %%
step_by_step = sql(f"""
WITH usage_4w AS (
    SELECT customer_id, SUM(sessions) AS sessions_4w
    FROM usage_weekly
    WHERE week_start + INTERVAL 7 DAY <= TIMESTAMP '{T}'                       -- complete before T
      AND week_start + INTERVAL 7 DAY >  TIMESTAMP '{T}' - INTERVAL 28 DAY     -- ended in the last 4 weeks
    GROUP BY customer_id
),
failed_90d AS (
    SELECT customer_id, COUNT(*) AS failed_payments_90d
    FROM invoices
    WHERE status = 'failed'
      AND invoice_date <  TIMESTAMP '{T}'
      AND invoice_date >= TIMESTAMP '{T}' - INTERVAL 90 DAY
    GROUP BY customer_id
)
SELECT p.customer_id, p.plan, p.tenure_days,
       COALESCE(u.sessions_4w, 0)         AS sessions_4w,
       COALESCE(f.failed_payments_90d, 0) AS failed_payments_90d
FROM population p
LEFT JOIN usage_4w   u USING (customer_id)
LEFT JOIN failed_90d f USING (customer_id)
ORDER BY p.customer_id
""")
step_by_step.head()

# %% [markdown]
# ### Step 4 · The label: what happens in [T, T + 30 days)
#
# The label is the only thing allowed to look past T, and only 30 days past it.
# This is the **one** legitimate use of `churn_date`.

# %%
labels = sql(f"""
SELECT p.customer_id,
       CASE WHEN c.churn_date >= TIMESTAMP '{T}'
             AND c.churn_date <  TIMESTAMP '{T}' + INTERVAL 30 DAY
            THEN 1 ELSE 0 END AS churned_30d
FROM population p JOIN customers c USING (customer_id)
""")
step_by_step = step_by_step.merge(labels, on="customer_id")
print(f"rows: {len(step_by_step)}, positives: {step_by_step['churned_30d'].sum()}, "
      f"rate: {step_by_step['churned_30d'].mean():.4f}")

# %% [markdown]
# **Expected:** 2,246 rows, 129 positives, rate ≈ 0.057.
#
# ### Step 5 · Check against the package
#
# `churnlab.features.build_features` builds all features for any list of
# snapshots. If your step-by-step logic is the same, the numbers must be
# **identical**, not close:

# %%
from churnlab import features as F

ref = F.build_features(tables, snapshots=[pd.Timestamp(T)])
compare = ref.merge(step_by_step, on="customer_id", suffixes=("_pkg", "_mine"))
print("rows (package, mine, matched):", len(ref), len(step_by_step), len(compare))
for col in ["tenure_days", "sessions_4w", "failed_payments_90d", "churned_30d"]:
    same = (compare[f"{col}_pkg"].astype(float) == compare[f"{col}_mine"].astype(float))
    print(f"{col:<22} identical: {same.all()}")
    assert same.all(), col

# %% [markdown]
# **Expected:** 2,246 rows on both sides, and `identical: True` for all four
# columns.
#
# Testing a feature pipeline against an independent, slower, obviously-correct
# implementation for one case is one of the most useful tests you can write.
# The repository's `tests/` do the same.
#
# > **Research bridge.** "Week complete" is the same idea as a causal filter
# > in real-time EEG decoding: an epoch's features may only use samples up to
# > the decision time. A zero-phase (forward-backward) filter applied offline
# > quietly uses future samples - exactly like counting an unfinished week.

# %% [markdown]
# Now the package's real SQL. It is the same logic, written for many snapshots
# at once (the `snapshots` table holds one row per T):

# %%
print(F.SNAPSHOT_SQL)
print(F.TICKETS_SQL)

# %% [markdown]
# Read `TICKETS_SQL` closely: the join condition restricts tickets to
# `[T - 90d, T)`; `FILTER` then narrows to 30 days for `tickets_30d`. Your
# turn - one feature, done properly.
#
# > **Exercise 3.** Write `tickets_30d_sql(snapshot)` returning SQL that gives,
# > for **every account active at `snapshot`** (the population from step 1),
# > `customer_id` and `tickets_30d`: the number of tickets created in
# > `[snapshot - 30 days, snapshot)`. Accounts with no such tickets must appear
# > with 0. The grader compares with `build_features` for two snapshots.

# %%
def tickets_30d_sql(snapshot: str) -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(tickets_30d_sql)

# %% [markdown]
# ## 4.6 · Anti-pattern: joining an export-time column
#
# Chapter 3 showed that `customers.status` describes each account **as of the
# export**. It is one join away, and very tempting:

# %%
con.register("labels", labels)
sql("""
SELECT c.status, COUNT(*) AS accounts, SUM(l.churned_30d) AS churners_next_30d
FROM population p
JOIN customers c USING (customer_id)
JOIN labels    l USING (customer_id)
GROUP BY c.status
ORDER BY c.status
""")

# %% [markdown]
# **Expected:** of the 2,246 accounts active at T, 990 (44%) are `cancelled`
# by export time, and **all** 129 churners are among them. The query is
# syntactically perfect and the resulting "feature" is the future. No SQL
# engine will warn you; only the rule "every feature uses only rows before T"
# does. Chapter 5 measures how much damage it does.
#
# Finally, cache the full feature table (all 21 monthly snapshots) for the
# notebooks that follow:

# %%
features = bootstrap.ensure_features()
print(features.shape, features["snapshot_date"].nunique(), "snapshots,",
      f"positive rate {features['churned_30d'].mean():.4f}")

# %% [markdown]
# **Expected:** 44,361 rows, 21 snapshots, positive rate ≈ 0.055.

# %% [markdown]
# > **Exercise 4 (written).** Back to section 4.3. In your own words: what causes the fan-out bug,
# > how would you **detect** it in a query you are reviewing, and name **two**
# > different ways to prevent it. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **Fan-out row count.** Joining two one-to-many tables $U$ and $K$ to the same key:
#
# $$
# \text{rows after join} = \sum_{c} n_U(c)\, n_K(c)
# \quad\Rightarrow\quad
# \text{SUM}_{\text{joined}}(x) = \sum_{c} n_K(c) \sum_{u \in U_c} x_u
# $$
#
# *In words:* each key contributes the product of its row counts, so every
# value from $U$ is counted once per matching row of $K$. *Symbols:* $c$ = a
# customer, $n_U(c)$, $n_K(c)$ = its rows in each table, $x_u$ = a value in
# $U$. *Example:* $n_U = 41$, $n_K = 7$: $287$ rows, and 212 true sessions
# become $7 \times 212 = 1{,}484$.
#
# **Completed-week rule.** A usage week starting on $w$ is usable at snapshot $T$ only if
#
# $$
# w + 7\ \text{days} \le T
# $$
#
# *In words:* the week must have ended by T. *Example:* T = Fri 2024-03-01,
# $w$ = Mon 2024-02-26: $w + 7 = $ 2024-03-04 $> T$, so the week is excluded
# (the naive $w < T$ would include 3 future days).
#
# **Label.** For account $i$ active at $T$:
#
# $$
# y_i(T) = \mathbb{1}\left[\, T \le \text{churn\_date}_i < T + 30\ \text{days} \,\right]
# $$
#
# *In words:* 1 if the account cancels in the 30 days starting at T, else 0.
# *Example:* T = 2024-03-01, churn on 2024-03-20 → $y = 1$; churn on
# 2024-04-02 → $y = 0$ (it counts for the April snapshot instead).
#
# **Positive rate.** $\hat{\pi} = \frac{1}{N}\sum_i y_i$. *Example:* 129
# churners among 2,246 accounts: $\hat{\pi} = 129 / 2246 = 0.057$.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `Catalog Error: Table with name ... does not exist` | the DataFrame was not registered (or the kernel restarted) | re-run the `con.register` cell |
# | `Binder Error: Cannot compare values of type TIMESTAMP and VARCHAR` | a date written as a bare string `'2024-03-01'` compared with a timestamp | write `TIMESTAMP '2024-03-01'` (or `DATE '...'`) |
# | `Parser Error: syntax error at or near "weeks"` | an alias that is a SQL keyword (`weeks`, `month`, ...) used without `AS` | write `AS n_weeks`, or quote it |
# | A sum is several times too large | fan-out from joining two one-to-many tables | aggregate each table per key first, then join |
# | Customers with no events are missing | `JOIN` (inner) instead of `LEFT JOIN`, or a `WHERE` on the right table | `LEFT JOIN` from the population, put conditions in `ON`, `COALESCE` |
# | Your feature matches the package except for a few rows | `<=` vs `<` at a window edge | write each window as a half-open interval `[start, end)` and check both ends |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Explain the fan-out bug with a concrete example, and two ways to prevent it.
# 2. When would you use `LEFT JOIN` instead of `JOIN` when building features, and what goes wrong with a `WHERE` on the right-hand table?
# 3. What is the difference between `ROWS` and `RANGE` window frames, and when does it matter?
# 4. Why is a usage week counted only when `week_start + 7 days <= T`? What leaks otherwise, and why does it survive casual testing?
# 5. Write (aloud) the label definition for snapshot T, and say which column is allowed to look into the future and how far.
# 6. How did you test that your SQL is right, and why compare to an independent implementation rather than eyeballing rows?
#
# **Next chapter:** [Chapter 5 · Leakage-safe splits & a baseline](../ch05-leakage-safe-splits-and-baseline/notebook.ipynb) — split the feature table without leaking the future, measure what leakage does, and set honest baselines.
