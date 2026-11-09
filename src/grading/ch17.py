"""Self-checks for Chapter 17 · Python & SQL drills.

Python drills are checked on small hand-made cases plus randomised cases
against a brute-force reference. SQL drills run your query and a reference
query in DuckDB on the CLEAN TaskFlow tables and compare the results.
"""

from __future__ import annotations

import random
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from churnlab.checks import run_check

# ---------------------------------------------------------------------------
# Python drills
# ---------------------------------------------------------------------------


def ex1(dedupe_keep_latest):
    def test():
        records = [
            {"id": "a", "ts": 1, "v": "old"},
            {"id": "b", "ts": 5, "v": "b-only"},
            {"id": "a", "ts": 3, "v": "new"},
            {"id": "c", "ts": 2, "v": "c1"},
            {"id": "a", "ts": 2, "v": "middle"},
            {"id": "c", "ts": 2, "v": "c2"},
        ]
        out = dedupe_keep_latest(records, "id", "ts")
        assert isinstance(out, list), f"expected a list of records, got {type(out).__name__}"
        assert [r["id"] for r in out] == ["a", "b", "c"], (
            f"expected one record per key in order of first appearance ['a', 'b', 'c'], got {[r['id'] for r in out]}")
        assert out[0]["v"] == "new", "for key 'a' keep the record with the LATEST ts (ts=3), not the last one seen"
        assert out[2]["v"] == "c2", "on a tie in ts, keep the record that appears later in the input"
        assert dedupe_keep_latest([], "id", "ts") == [], "an empty input should return an empty list"
        rng = random.Random(0)
        big = [{"k": rng.randrange(50), "t": rng.random(), "i": i} for i in range(2000)]
        got = {r["k"]: r["i"] for r in dedupe_keep_latest(big, "k", "t")}
        want = {}
        for r in big:
            if r["k"] not in want or r["t"] >= want[r["k"]]["t"]:
                want[r["k"]] = r
        assert got == {k: r["i"] for k, r in want.items()}, "wrong record kept on a randomised input"
        assert len(big) == 2000, "do not modify the input list"

    return run_check("Exercise 1 · dedupe_keep_latest", test,
                     hint="one pass with a dict: latest[key] = record when record[ts] >= latest[key][ts]; "
                          "dicts keep insertion order")


def _merge_brute(intervals):
    points = set()
    for s, e in intervals:
        points.update(range(2 * s, 2 * e + 1))  # doubled so touching ends share a point
    out, run = [], None
    for p in sorted(points):
        if run and p == run[1] + 1:
            run[1] = p
        else:
            if run:
                out.append(run)
            run = [p, p]
    if run:
        out.append(run)
    return [(a // 2, b // 2) for a, b in out]


def ex2(merge_intervals):
    def test():
        out = merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)])
        assert [tuple(x) for x in out] == [(1, 6), (8, 10), (15, 18)], f"got {out}"
        out = merge_intervals([(5, 7), (1, 2), (2, 4)])
        assert [tuple(x) for x in out] == [(1, 4), (5, 7)], (
            f"unsorted input, and touching periods (1,2)+(2,4) must merge; got {out}")
        out = merge_intervals([(1, 10), (2, 3), (4, 5)])
        assert [tuple(x) for x in out] == [(1, 10)], f"nested periods must be absorbed; got {out}"
        assert list(merge_intervals([])) == [], "empty input -> empty list"
        dates = merge_intervals([(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-01")),
                                 (pd.Timestamp("2024-02-15"), pd.Timestamp("2024-04-01"))])
        assert [tuple(x) for x in dates] == [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01"))], (
            "should work for any comparable values, e.g. dates")
        rng = random.Random(1)
        for _ in range(200):
            ivs = []
            for _ in range(rng.randrange(1, 8)):
                s = rng.randrange(0, 30)
                ivs.append((s, s + rng.randrange(0, 6)))
            got = [tuple(x) for x in merge_intervals(list(ivs))]
            assert got == _merge_brute(ivs), f"input {ivs}: expected {_merge_brute(ivs)}, got {got}"

    return run_check("Exercise 2 · merge_intervals", test,
                     hint="sort by start; extend the last merged period while start <= its end, else start a new one")


def ex3(sliding_window_max):
    def test():
        assert list(sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)) == [3, 3, 5, 5, 6, 7]
        assert list(sliding_window_max([4, 2], 1)) == [4, 2], "k=1 returns the input"
        assert list(sliding_window_max([4, 2, 9], 3)) == [9], "k=n returns a single maximum"
        assert list(sliding_window_max([2, 2, 2, 1], 2)) == [2, 2, 2], "duplicates must be handled"
        rng = random.Random(2)
        for _ in range(100):
            vals = [rng.randrange(-20, 20) for _ in range(rng.randrange(1, 40))]
            k = rng.randrange(1, len(vals) + 1)
            want = [max(vals[i:i + k]) for i in range(len(vals) - k + 1)]
            assert list(sliding_window_max(vals, k)) == want, f"wrong on values={vals}, k={k}"
        big = np.random.default_rng(3).integers(0, 1_000_000, 200_000).tolist()
        k = 5_000
        start = time.perf_counter()
        out = sliding_window_max(big, k)
        elapsed = time.perf_counter() - start
        assert len(out) == len(big) - k + 1, "wrong output length on the large input"
        assert out[0] == max(big[:k]) and out[-1] == max(big[-k:]), "wrong values on the large input"
        assert elapsed < 2.0, (
            f"took {elapsed:.2f}s on n=200,000, k=5,000; an O(n) deque solution takes well under 0.5s")

    return run_check("Exercise 3 · sliding_window_max", test,
                     hint="keep a deque of INDICES with decreasing values; pop from the right while smaller, "
                          "pop from the left when it falls out of the window")


def ex4(precision_at_k):
    from churnlab.evaluation import precision_at_k as reference

    def test():
        assert abs(precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 2) - 0.5) < 1e-9
        assert abs(precision_at_k([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], 2) - 1.0) < 1e-9
        assert abs(precision_at_k([1, 0, 0], [0.5, 0.9, 0.1], 1) - 0.0) < 1e-9, "the top-1 row is the 0.9 score"
        rng = np.random.default_rng(4)
        for _ in range(50):
            n = int(rng.integers(5, 200))
            y = rng.integers(0, 2, n)
            s = rng.permutation(n) / n  # no ties
            k = int(rng.integers(1, n + 1))
            got = precision_at_k(list(y), list(s), k)
            want = reference(y, s, k)
            assert abs(got - want) < 1e-9, f"n={n}, k={k}: expected {want:.4f}, got {got:.4f}"

    return run_check("Exercise 4 · precision_at_k", test,
                     hint="order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]; "
                          "return sum(y_true[i] for i in order) / k")


def ex5(stratified_split):
    def test():
        labels = [0] * 90 + [1] * 10
        train, test = stratified_split(labels, 0.2, 0)
        train, test = list(train), list(test)
        assert sorted(train + test) == list(range(100)), "train and test must partition all indices exactly once"
        assert not set(train) & set(test), "an index appears in both train and test"
        n1 = sum(labels[i] for i in test)
        assert abs(n1 - 2) <= 1 and abs((len(test) - n1) - 18) <= 1, (
            f"test should hold ~2 positives and ~18 negatives (20%), got {n1} and {len(test) - n1}")
        again = stratified_split(labels, 0.2, 0)
        assert list(again[1]) == test, "same seed must give the same split"
        other = stratified_split(labels, 0.2, 1)
        assert list(other[1]) != test, "a different seed should give a different split"
        rng = random.Random(5)
        for _ in range(30):
            classes = rng.randrange(2, 5)
            lab = [rng.randrange(classes) for _ in range(rng.randrange(20, 400))]
            frac = rng.choice([0.1, 0.2, 0.25, 0.3, 0.5])
            tr, te = (list(x) for x in stratified_split(lab, frac, rng.randrange(1000)))
            assert sorted(tr + te) == list(range(len(lab))), "indices must partition the input"
            for c in set(lab):
                n_c = lab.count(c)
                got = sum(1 for i in te if lab[i] == c)
                assert abs(got - n_c * frac) <= 1, (
                    f"class {c}: {n_c} rows, test_frac={frac} -> expected ~{n_c * frac:.1f} in test, got {got}")

    return run_check("Exercise 5 · stratified_split", test,
                     hint="group indices by label, shuffle each group with random.Random(seed), "
                          "put round(len(group) * test_frac) of each into test")


# ---------------------------------------------------------------------------
# SQL drills
# ---------------------------------------------------------------------------

REFERENCE_SQL = {
    6: """
SELECT date_trunc('month', invoice_date) AS month,
       SUM(amount_usd) AS revenue,
       SUM(SUM(amount_usd)) OVER (ORDER BY date_trunc('month', invoice_date)) AS running_revenue
FROM invoices
WHERE status = 'paid'
GROUP BY 1
ORDER BY 1
""",
    7: """
WITH cohorts AS (
    SELECT customer_id, date_trunc('month', signup_date) AS cohort_month
    FROM customers
    WHERE signup_date >= DATE '2023-01-01' AND signup_date < DATE '2024-10-01'
),
active AS (
    SELECT DISTINCT customer_id, date_trunc('month', week_start) AS active_month
    FROM usage_weekly
)
SELECT c.cohort_month,
       COUNT(DISTINCT c.customer_id) AS cohort_size,
       COUNT(DISTINCT CASE WHEN a.active_month = c.cohort_month + INTERVAL 1 MONTH THEN c.customer_id END)
           / COUNT(DISTINCT c.customer_id) AS ret_m1,
       COUNT(DISTINCT CASE WHEN a.active_month = c.cohort_month + INTERVAL 2 MONTH THEN c.customer_id END)
           / COUNT(DISTINCT c.customer_id) AS ret_m2,
       COUNT(DISTINCT CASE WHEN a.active_month = c.cohort_month + INTERVAL 3 MONTH THEN c.customer_id END)
           / COUNT(DISTINCT c.customer_id) AS ret_m3
FROM cohorts AS c
LEFT JOIN active AS a USING (customer_id)
GROUP BY 1
ORDER BY 1
""",
    8: """
SELECT customer_id, invoice_id, invoice_date, amount_usd
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date, invoice_id) AS rn
    FROM invoices
)
WHERE rn = 3
""",
    9: """
WITH weeks AS (
    SELECT DISTINCT customer_id, week_start FROM usage_weekly
),
gaps AS (
    SELECT customer_id,
           date_diff('day', week_start,
                     LEAD(week_start) OVER (PARTITION BY customer_id ORDER BY week_start)) / 7 - 1 AS gap
    FROM weeks
)
SELECT customer_id, CAST(COALESCE(MAX(gap), 0) AS INTEGER) AS longest_inactive_streak
FROM gaps
GROUP BY customer_id
""",
    10: """
SELECT category, MEDIAN(resolution_hours) AS median_hours
FROM tickets
GROUP BY category
ORDER BY median_hours DESC
""",
}

#: (expected columns, columns that define row order or None, sort keys for order-insensitive compare)
SQL_SPECS = {
    6: (["month", "revenue", "running_revenue"], ["month"], ["month"]),
    7: (["cohort_month", "cohort_size", "ret_m1", "ret_m2", "ret_m3"], None, ["cohort_month"]),
    8: (["customer_id", "invoice_id", "invoice_date", "amount_usd"], None, ["customer_id"]),
    9: (["customer_id", "longest_inactive_streak"], None, ["customer_id"]),
    10: (["category", "median_hours"], None, ["category"]),
}


@lru_cache(maxsize=1)
def _tables():
    from churnlab.bootstrap import load_clean

    return load_clean()


def connection():
    """A DuckDB connection with the clean TaskFlow tables registered."""
    import duckdb

    con = duckdb.connect()
    for name, df in _tables().items():
        con.register(name, df)
    return con


def run_sql(query: str) -> pd.DataFrame:
    # Keep a reference to the connection until the result is fetched: older
    # DuckDB versions close a connection as soon as it is garbage-collected.
    con = connection()
    result = con.sql(query).df()
    con.close()
    return result


def reference_result(number: int) -> pd.DataFrame:
    return run_sql(REFERENCE_SQL[number])


def _compare(got: pd.DataFrame, want: pd.DataFrame, number: int) -> None:
    columns, ordered_by, keys = SQL_SPECS[number]
    missing = [c for c in columns if c not in got.columns]
    assert not missing, f"missing column(s) {missing}; your columns: {list(got.columns)}"
    got = got[columns]
    assert len(got) == len(want), f"expected {len(want)} rows, got {len(got)}"
    if ordered_by:
        sorted_got = got.sort_values(ordered_by).reset_index(drop=True)
        assert got.reset_index(drop=True)[ordered_by].equals(sorted_got[ordered_by]), (
            f"rows must be ordered by {ordered_by} (a running total only makes sense in order)")
    g = got.sort_values(keys).reset_index(drop=True)
    w = want[columns].sort_values(keys).reset_index(drop=True)
    for col in columns:
        a, b = g[col], w[col]
        if pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(b):
            assert pd.api.types.is_numeric_dtype(a), f"column {col!r} should be numeric"
            diff = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
            bad = np.flatnonzero(~(diff <= 1e-6 * np.maximum(1.0, np.abs(b.to_numpy(dtype=float)))))
            assert len(bad) == 0, (
                f"column {col!r} differs in {len(bad)} row(s); first: {g.loc[bad[0], keys].to_dict()} "
                f"expected {b.iloc[bad[0]]!r}, got {a.iloc[bad[0]]!r}")
        else:
            a = pd.to_datetime(a) if pd.api.types.is_datetime64_any_dtype(b) else a.astype(str)
            b = b if pd.api.types.is_datetime64_any_dtype(b) else b.astype(str)
            neq = np.flatnonzero(a.to_numpy() != b.to_numpy())
            assert len(neq) == 0, (
                f"column {col!r} differs in {len(neq)} row(s); first: expected {b.iloc[neq[0]]!r}, got {a.iloc[neq[0]]!r}")


def _sql_check(number: int, title: str, fn, hint: str):
    def test():
        query = fn()
        assert isinstance(query, str) and query.strip(), "your function must return the SQL query as a string"
        got = run_sql(query)
        _compare(got, reference_result(number), number)

    return run_check(f"Exercise {number} · {title}", test, hint=hint)


def ex6(monthly_revenue_sql):
    return _sql_check(6, "monthly revenue with running total", monthly_revenue_sql,
                      "GROUP BY date_trunc('month', invoice_date) for paid invoices; "
                      "running total = SUM(SUM(amount_usd)) OVER (ORDER BY month)")


def ex7(cohort_retention_sql):
    return _sql_check(7, "signup-month cohort retention", cohort_retention_sql,
                      "cohort = date_trunc('month', signup_date); active months = DISTINCT date_trunc('month', week_start); "
                      "COUNT(DISTINCT CASE WHEN active_month = cohort_month + INTERVAL k MONTH ...) / cohort_size")


def ex8(third_invoice_sql):
    return _sql_check(8, "each customer's 3rd invoice", third_invoice_sql,
                      "ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date, invoice_id) in a "
                      "subquery, then WHERE rn = 3")


def ex9(longest_inactive_streak_sql):
    return _sql_check(9, "longest inactive streak (gaps and islands)", longest_inactive_streak_sql,
                      "DISTINCT weeks per customer; LEAD(week_start) gives the next active week; "
                      "inactive weeks in between = days / 7 - 1; take the MAX, 0 if none")


def ex10(median_resolution_sql):
    return _sql_check(10, "median resolution hours by category", median_resolution_sql,
                      "MEDIAN(resolution_hours) or quantile_cont(resolution_hours, 0.5), GROUP BY category")
