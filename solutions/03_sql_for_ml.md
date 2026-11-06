# Solutions · 03 · SQL for ML

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 03-1 · `weekly_change_sql`

✅ Reference code: [`code/nb03.py`](code/nb03.py) (`ex1`), graded in CI.

```sql
SELECT customer_id, week_start, sessions,
       LAG(sessions) OVER w            AS prev_sessions,
       sessions - LAG(sessions) OVER w AS sessions_change
FROM usage_weekly
WINDOW w AS (PARTITION BY customer_id ORDER BY week_start)
ORDER BY customer_id, week_start
```

**Why it works.** `PARTITION BY customer_id` restarts the window per customer,
so the first week of each customer has no previous row (NULL). Arithmetic with
NULL is NULL, which is the honest answer when the previous count is unknown
(a cleaned `-1`).

**Common mistakes.** Forgetting `PARTITION BY` (customer B's first week is
compared with customer A's last). Ordering the window by `sessions` or not at
all.

**Push further.** `LAG` gives the previous *logged* week. For "change vs the
previous calendar week, missing weeks counting as 0", first build a dense
customer × week grid (`generate_series` + `LEFT JOIN`), then `LAG` over it.
Which definition is right depends on what the feature should mean: say which
one you chose.

---

## 03-2 · `latest_invoice_sql(as_of)`

✅ Reference code: [`code/nb03.py`](code/nb03.py) (`ex2`).

```python
def latest_invoice_sql(as_of: str) -> str:
    return f"""
    SELECT customer_id, invoice_date, amount_usd, status
    FROM invoices
    WHERE invoice_date < TIMESTAMP '{as_of}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) = 1
    """
```

**Why the `WHERE` comes first.** `WHERE` filters before the window function
runs, so the ranking only sees invoices before `as_of`. Filtering *after*
ranking (rank all invoices, keep rn = 1, then require date < as_of) drops
every customer whose overall-latest invoice is after `as_of`, which is most of
them.

**`<` not `<=`.** An invoice dated exactly at the snapshot is "at T", not
"before T". The package's `mrr_usd` uses the same convention, which is why the
grader can compare with it exactly.

**Production note.** An f-string is fine for a trusted date in a notebook.
For values that come from users, use parameters (`con.execute(sql, [as_of])`
with `?` placeholders) to avoid SQL injection.

---

## 03-3 · `tickets_30d_sql(snapshot)`

✅ Reference code: [`code/nb03.py`](code/nb03.py) (`ex3`).

```sql
WITH active AS (
    SELECT customer_id FROM customers
    WHERE signup_date < TIMESTAMP '{snapshot}'
      AND (churn_date IS NULL OR churn_date >= TIMESTAMP '{snapshot}')
)
SELECT a.customer_id, COUNT(t.ticket_id) AS tickets_30d
FROM active AS a
LEFT JOIN tickets AS t
  ON  t.customer_id = a.customer_id
  AND t.created_at >= TIMESTAMP '{snapshot}' - INTERVAL 30 DAY
  AND t.created_at <  TIMESTAMP '{snapshot}'
GROUP BY a.customer_id
```

**The three details the grader checks.**

1. **Population.** Start from the accounts active at T; an inner join from
   `tickets` drops every customer with zero tickets.
2. **Conditions in `ON`, not `WHERE`.** A `WHERE t.created_at >= ...` after a
   `LEFT JOIN` removes the NULL rows again and silently turns it into an inner
   join.
3. **`COUNT(t.ticket_id)`, not `COUNT(*)`.** With a left join, a customer with
   no tickets still has one row (all NULLs); `COUNT(*)` counts it as 1.

The window is half-open, `[T - 30d, T)`. Using `<=` on the right or `31 DAY`
on the left makes about 30 customers disagree with the package at 2024-02-01.

---

## 03-4 · The fan-out bug (written)

✍️ A strong answer:

* **Cause.** Joining a table to two (or more) one-to-many tables on the same
  key returns every combination of their rows. Each row of one is repeated
  once per matching row of the other, so `SUM`/`COUNT` are multiplied (here
  212 sessions × 7 tickets = 1,484; 41 weeks × 7 = 287 tickets).
* **Detect.** Compare row counts before and after each join (a join to a
  "lookup" should not change the row count); check a total against the source
  table (`SUM(sessions)` 1.70 M vs 11.0 M); look for `COUNT(DISTINCT ...)`
  added "to make the numbers right", which usually hides a fan-out.
* **Prevent (any two).** (1) Aggregate each source to one row per key in its
  own CTE, then join the aggregates. (2) Use correlated/scalar subqueries per
  feature. (3) Assert uniqueness of the join key on the "one" side (a data
  contract or a test). (4) `COUNT(DISTINCT id)` works for counts but not for
  sums, so it is a patch, not a fix.

---

## Checkpoint · 03

**1. Fan-out.** See 03-4: per-customer sessions joined with tickets multiplies
sessions by the ticket count. Prevent by aggregating per key before joining,
and by asserting key uniqueness.

**2. `LEFT JOIN` for features.** Whenever "no events" is a valid value (zero
tickets, no failed payments): the population must survive. A `WHERE` on a
right-table column discards the NULL-extended rows, turning it back into an
inner join; put such conditions in `ON`.

**3. `ROWS` vs `RANGE`.** `ROWS` counts physical rows; `RANGE` uses the
`ORDER BY` value (here dates). They differ when rows are missing (inactive
weeks are not logged): "last 4 rows" can span 6 weeks. For time windows, use
`RANGE` with an interval (or filter on dates explicitly).

**4. Week complete.** A week's usage is known only at `week_start + 7 days`.
With `week_start < T`, the week containing T leaks up to 6 days of the future
into every usage feature. It survives casual testing because when T is a
Monday (a week boundary) both rules give the same answer.

**5. The label.** `churned_30d = 1` if `churn_date` is in `[T, T + 30 days)`,
for accounts that signed up before T and had not churned before T.
`churn_date` is the only column allowed past T, and only 30 days past it, and
only for the label.

**6. Testing the SQL.** An independent, step-by-step implementation for one
snapshot compared with `build_features` for *equality* on every row. Eyeballing
a few rows misses boundary errors that affect 1-2% of rows; an exact
comparison does not.
