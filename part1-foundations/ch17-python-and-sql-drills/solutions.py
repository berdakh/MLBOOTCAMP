"""Reference solutions for notebook 16. Graded by tests/test_solutions.py."""

from __future__ import annotations

import random
from collections import deque


def ex1(records: list[dict], key: str, ts: str) -> list[dict]:
    latest: dict = {}
    for record in records:
        k = record[key]
        if k not in latest or record[ts] >= latest[k][ts]:
            latest[k] = record  # re-assigning an existing key keeps its original position
    return list(latest.values())


def ex2(intervals):
    merged: list[tuple] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def ex3(values, k: int) -> list:
    window: deque[int] = deque()  # indices; their values are decreasing
    out = []
    for i, v in enumerate(values):
        while window and values[window[-1]] <= v:
            window.pop()
        window.append(i)
        if window[0] <= i - k:
            window.popleft()
        if i >= k - 1:
            out.append(values[window[0]])
    return out


def ex4(y_true, scores, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    scores = list(scores)
    y_true = list(y_true)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return sum(y_true[i] for i in top) / k


def ex5(labels, test_frac: float, seed: int):
    rng = random.Random(seed)
    by_class: dict = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, []).append(i)
    train, test = [], []
    for label in sorted(by_class, key=str):
        idx = by_class[label][:]
        rng.shuffle(idx)
        n_test = round(len(idx) * test_frac)
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return sorted(train), sorted(test)


def ex6() -> str:
    return """
    SELECT date_trunc('month', invoice_date) AS month,
           SUM(amount_usd) AS revenue,
           SUM(SUM(amount_usd)) OVER (ORDER BY date_trunc('month', invoice_date)) AS running_revenue
    FROM invoices
    WHERE status = 'paid'
    GROUP BY 1
    ORDER BY 1
    """


def ex7() -> str:
    return """
    WITH cohorts AS (
        SELECT customer_id, date_trunc('month', signup_date) AS cohort_month
        FROM customers
        WHERE signup_date >= DATE '2023-01-01' AND signup_date < DATE '2024-10-01'
    ),
    active AS (
        SELECT DISTINCT customer_id, date_trunc('month', week_start) AS active_month
        FROM usage_weekly
    ),
    flags AS (
        SELECT c.customer_id, c.cohort_month,
               MAX(CASE WHEN a.active_month = c.cohort_month + INTERVAL 1 MONTH THEN 1 ELSE 0 END) AS m1,
               MAX(CASE WHEN a.active_month = c.cohort_month + INTERVAL 2 MONTH THEN 1 ELSE 0 END) AS m2,
               MAX(CASE WHEN a.active_month = c.cohort_month + INTERVAL 3 MONTH THEN 1 ELSE 0 END) AS m3
        FROM cohorts AS c
        LEFT JOIN active AS a USING (customer_id)
        GROUP BY 1, 2
    )
    SELECT cohort_month, COUNT(*) AS cohort_size,
           AVG(m1) AS ret_m1, AVG(m2) AS ret_m2, AVG(m3) AS ret_m3
    FROM flags
    GROUP BY cohort_month
    ORDER BY cohort_month
    """


def ex8() -> str:
    return """
    WITH numbered AS (
        SELECT customer_id, invoice_id, invoice_date, amount_usd,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date, invoice_id) AS rn
        FROM invoices
    )
    SELECT customer_id, invoice_id, invoice_date, amount_usd
    FROM numbered
    WHERE rn = 3
    """


def ex9() -> str:
    # Classic gaps-and-islands: build the full weekly grid between each
    # customer's first and last active week, flag inactive weeks, and label
    # runs with the difference of two ROW_NUMBERs.
    return """
    WITH weeks AS (
        SELECT DISTINCT customer_id, week_start FROM usage_weekly
    ),
    bounds AS (
        SELECT customer_id, MIN(week_start) AS first_week, MAX(week_start) AS last_week
        FROM weeks GROUP BY customer_id
    ),
    grid AS (
        SELECT b.customer_id, UNNEST(generate_series(b.first_week, b.last_week, INTERVAL 7 DAY)) AS week
        FROM bounds AS b
    ),
    flagged AS (
        SELECT g.customer_id, g.week, w.week_start IS NULL AS inactive
        FROM grid AS g
        LEFT JOIN weeks AS w ON w.customer_id = g.customer_id AND w.week_start = g.week
    ),
    islands AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY week)
             - ROW_NUMBER() OVER (PARTITION BY customer_id, inactive ORDER BY week) AS island
        FROM flagged
    ),
    runs AS (
        SELECT customer_id, inactive, island, COUNT(*) AS run_length
        FROM islands GROUP BY customer_id, inactive, island
    )
    SELECT customer_id,
           COALESCE(MAX(CASE WHEN inactive THEN run_length END), 0) AS longest_inactive_streak
    FROM runs
    GROUP BY customer_id
    """


def ex10() -> str:
    return """
    SELECT category, quantile_cont(resolution_hours, 0.5) AS median_hours
    FROM tickets
    GROUP BY category
    ORDER BY median_hours DESC
    """
