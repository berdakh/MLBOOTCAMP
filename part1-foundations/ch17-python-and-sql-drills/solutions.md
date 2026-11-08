# Solutions · Chapter 17 · Python & SQL drills

Try each drill against the clock before reading its answer. Markers: [verified]
verified by running · [expected] expected, depends on your run · [open-ended] personal /
open-ended.

All coded answers are in [`solutions.py`](solutions.py) and graded in CI.
In an interview, say the approach and complexity first, then code.

---

## Exercise 1 · `dedupe_keep_latest`

[verified] `ex1`. One pass, a dict keyed by `record[key]`, replace when
`record[ts] >= latest[k][ts]`.

* **O(n) time, O(u) space** for u distinct keys.
* `>=` (not `>`) implements "on a tie keep the later one".
* Re-assigning an existing dict key does **not** move it, so `list(latest.values())`
  is already in order of first appearance. Say this out loud; it is a Python
  3.7+ guarantee worth stating explicitly.

**Common mistakes.** Sorting by `ts` first (O(n log n), and it destroys
first-appearance order); using `pandas.drop_duplicates(keep="last")` (keeps the
last *seen*, not the latest `ts`, unless you sort, and the interviewer asked for
plain Python).

**Follow-up you should expect.** "The records don't fit in memory." Hash
partition by key to disk, dedupe each partition; or in SQL,
`QUALIFY ROW_NUMBER() OVER (PARTITION BY key ORDER BY ts DESC) = 1`.

---

## Exercise 2 · `merge_intervals`

[verified] `ex2`. Sort by start, then sweep: if `start <= last_end` extend
`last_end = max(last_end, end)`, else open a new interval.

* **O(n log n)** for the sort, O(n) sweep.
* `<=` makes touching periods merge; `max` handles nested periods.
* Works for any comparable type (ints, dates, timestamps), because it only
  uses `<=` and `max`.

**Common mistake.** `merged[-1][1] = end` instead of `max(...)`: a short
period inside a long one then *shrinks* the long one.

---

## Exercise 3 · `sliding_window_max`

[verified] `ex3`. A deque of **indices** whose values are strictly decreasing from
left to right. For each new index: pop from the right every index whose
value is `<=` the new one (they can never be a maximum again), append the new
index, drop the left index if it has left the window, and the left element is
the window maximum.

* **O(n) time:** every index is appended once and popped at most once.
  **O(k) space.**
* Storing indices, not values, is what lets you detect when the front has
  left the window.

**Common mistakes.** `max(values[i:i+k])` is O(nk), 10⁹ operations at the
grader's size. A heap is O(n log n) and needs lazy deletion; acceptable as a
first answer if you then improve it.

---

## Exercise 4 · `precision_at_k`

[verified] `ex4`. `top = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]`,
then `sum(y_true[i] for i in top) / k`. Raise `ValueError` for `k <= 0`
(dividing by zero silently would be worse).

* O(n log n). For k much smaller than n, `heapq.nlargest(k, ...)` is
  O(n log k). Mention it.
* **Ties.** `sorted` is stable, so tied scores keep input order. With many ties
  (e.g. a rounded score) precision@k depends on input order. Say so; the honest
  fixes are to break ties randomly with a seed or to report the expected value
  over tied rows.

---

## Exercise 5 · `stratified_split`

[verified] `ex5`. Group indices by label, shuffle each group with one
`random.Random(seed)`, put `round(len(group) * test_frac)` of each into test.

* Iterate classes in a **fixed order** (`sorted`), otherwise the random stream is
  consumed in a different order and the same seed can give a different split.
* Per-class rounding is what guarantees ±1 row per class.

**Follow-ups.** Grouped data (customers, subjects): stratify groups,
`StratifiedGroupKFold`. Time-ordered data: do not shuffle at all. That is the
mistake chapter 5 measured.

---

## Exercise 6 · Monthly revenue with a running total

[verified] `ex6`.

```sql
SELECT date_trunc('month', invoice_date) AS month,
       SUM(amount_usd) AS revenue,
       SUM(SUM(amount_usd)) OVER (ORDER BY date_trunc('month', invoice_date)) AS running_revenue
FROM invoices
WHERE status = 'paid'
GROUP BY 1
ORDER BY 1
```

The nested `SUM(SUM(...)) OVER` is a window over the *grouped* rows: the inner
`SUM` is the aggregate, the outer is the window. A CTE with the monthly totals
and then `SUM(revenue) OVER (ORDER BY month)` is equally good and easier to
read aloud.

**Interview detail.** The default window frame with `ORDER BY` is `RANGE
BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. With duplicate order keys `RANGE`
includes all peers; here months are unique, so it does not matter. Say you know
the difference from `ROWS`.

---

## Exercise 7 · Cohort retention

[verified] `ex7`. Two shapes are accepted. Reference (grader) uses
`COUNT(DISTINCT CASE WHEN active_month = cohort_month + INTERVAL k MONTH THEN customer_id END) / COUNT(DISTINCT customer_id)`;
the solution file aggregates to one row per customer with 0/1 flags first,
then `AVG(flag)`. The second is easier to debug: you can inspect the per-customer
flags.

[expected] On this data retention is high (around 0.95-0.98 at month +1, ~0.85-0.92 at
+3). Synthetic data; real SaaS cohorts usually decay faster.

**Why the date bounds.** Usage starts in January 2023, so earlier cohorts
would look like they had no usage in their first months: a **censoring**
artefact, not churn. Cohorts after September 2024 do not have three full months
observed. Interviewers love asking how you handle the right-censored edge.

---

## Exercise 8 · Each customer's 3rd invoice

[verified] `ex8`. `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date, invoice_id)`
in a CTE, then `WHERE rn = 3`. (DuckDB also allows `QUALIFY rn = 3` directly.)

* `ROW_NUMBER` gives exactly one row per position, even on ties, which is what
  "the third invoice" means. `RANK` would skip numbers after ties (1, 1, 3) and
  `DENSE_RANK` would not (1, 1, 2): both can return zero or several rows.
* The tie-breaker `invoice_id` makes the result deterministic. Without it, two
  invoices on the same day can swap between runs.

---

## Exercise 9 · Longest inactive streak (gaps and islands)

[verified] `ex9`. The grader's reference uses `LEAD`:

```sql
WITH weeks AS (SELECT DISTINCT customer_id, week_start FROM usage_weekly),
gaps AS (
  SELECT customer_id,
         date_diff('day', week_start,
                   LEAD(week_start) OVER (PARTITION BY customer_id ORDER BY week_start)) / 7 - 1 AS gap
  FROM weeks)
SELECT customer_id, CAST(COALESCE(MAX(gap), 0) AS INTEGER) AS longest_inactive_streak
FROM gaps GROUP BY customer_id
```

The solution file uses the classic islands trick on a generated weekly grid:
`ROW_NUMBER() OVER (PARTITION BY customer ORDER BY week) - ROW_NUMBER() OVER
(PARTITION BY customer, inactive ORDER BY week)` is constant within each run
of equal `inactive` values, so it labels the runs; count rows per run and take
the longest inactive one. Both give identical results (checked in CI).

[expected] Most customers never skip a week (median 0); the maximum is about 7 weeks.

The `LEAD` version is shorter and O(rows); the grid version generalises to
"longest *active* streak" or "streaks of weeks with < 3 sessions", where gaps
are not simply missing rows. Know both.

---

## Exercise 10 · Median resolution hours by category

[verified] `ex10`. `MEDIAN(resolution_hours)` or `quantile_cont(resolution_hours, 0.5)`,
`GROUP BY category`.

[expected] Cancellation tickets ≈ 4.8 h median vs ≈ 15 h for the others.

`quantile_cont` interpolates between the two middle values, `quantile_disc`
returns an actual observed value. Resolution times are right-skewed (a few
tickets stay open for days), so the mean is pulled up by the tail and the
median describes a typical ticket.

---

## Checkpoint answers

**1. Sliding-window complexity.** O(n) time, O(k) space. Each index enters the
deque once and leaves it at most once (from the right when dominated, or from
the left when it leaves the window), so the total work over all iterations is
at most 2n deque operations, even though one iteration may pop many.

**2. ROW_NUMBER / RANK / DENSE_RANK.** `ROW_NUMBER` with a deterministic
tie-breaker: it numbers rows 1, 2, 3... regardless of ties, so "third" is always
exactly one row. `RANK` gives tied rows the same number and skips (1, 1, 3);
`DENSE_RANK` gives the same number without skipping (1, 1, 2). With ties, both
can return zero or two "third" invoices.

**3. Retention of 1.4.** The numerator counts rows, not customers: the join to
weekly usage multiplied each customer by the number of active weeks (or the
denominator counts distinct customers while the numerator does not). Find it
by running the per-customer CTE and checking
`SELECT customer_id, COUNT(*) ... GROUP BY 1 HAVING COUNT(*) > 1`, or by
comparing `COUNT(*)` with `COUNT(DISTINCT customer_id)` in the cohort.

**4. Islands trick.** Number all of a customer's weeks in order (1, 2, 3, ...).
Separately number the weeks within each state (active / inactive). Within one
uninterrupted run of the same state, both numbers increase by one per week, so
their difference stays constant; when the run is broken, the difference jumps.
The difference is therefore a run label: group by (customer, state, label)
and count.

**5. Median vs mean.** Resolution time is skewed by a long tail, and the mean
is dominated by a few very long tickets; the median describes the typical
ticket. To compare categories honestly add the count per category, a spread
(p90 or IQR), and a check of whether the mix differs (e.g. cancellation tickets
may be routed to a different team, or opened at different hours), before
drawing any conclusion about the team's performance.
