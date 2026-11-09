"""Self-checks for Chapter 4 · SQL for ML.

Every exercise returns a SQL string; the checks run it in DuckDB against small
fixtures with hand-computed answers and, where it matters, against the real
clean tables and ``churnlab.features.build_features``.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from churnlab.checks import run_check

_CLEAN: dict | None = None


def _clean_tables() -> dict:
    global _CLEAN
    if _CLEAN is None:
        from churnlab.bootstrap import load_clean

        _CLEAN = load_clean()
    return _CLEAN


def _run(sql, tables: dict) -> pd.DataFrame:
    assert isinstance(sql, str), f"the function must return the SQL text as a string, got {type(sql).__name__}"
    con = duckdb.connect()
    try:
        for name, df in tables.items():
            con.register(name, df)
        return con.execute(sql).df()
    finally:
        con.close()


def ex1(weekly_change_sql):
    def test():
        usage = pd.DataFrame({
            "customer_id": ["C1", "C2", "C1", "C2", "C1"],
            "week_start": pd.to_datetime(["2024-01-08", "2024-01-01", "2024-01-01", "2024-01-08", "2024-01-22"]),
            "sessions": [5.0, float("nan"), 3.0, 4.0, 2.0],
        })
        got = _run(weekly_change_sql(), {"usage_weekly": usage})
        need = {"customer_id", "week_start", "sessions", "prev_sessions", "sessions_change"}
        assert need <= set(got.columns), f"missing columns {sorted(need - set(got.columns))}"
        assert len(got) == 5, f"expected one row per input row (5), got {len(got)}"
        got = got.sort_values(["customer_id", "week_start"]).reset_index(drop=True)
        prev = got["prev_sessions"].tolist()
        change = got["sessions_change"].tolist()
        # C1: 3, 5, 2 (the 2024-01-15 week was not logged) ; C2: NULL, 4
        assert pd.isna(prev[0]) and pd.isna(prev[3]), "the first week of each customer has no previous week (NULL)"
        assert prev[1] == 3 and prev[2] == 5, (f"C1 prev_sessions should be [NULL, 3, 5], got {prev[:3]} - "
                                               "PARTITION BY customer_id ORDER BY week_start")
        assert change[1] == 2 and change[2] == -3, f"C1 sessions_change should be [NULL, 2, -3], got {change[:3]}"
        assert pd.isna(change[4]), "C2's second week: previous sessions is NULL, so the change must be NULL too"

    return run_check("Exercise 1 · weekly_change_sql", test,
                     hint="LAG(sessions) OVER (PARTITION BY customer_id ORDER BY week_start)")


def ex2(latest_invoice_sql):
    def test():
        invoices = pd.DataFrame({
            "invoice_id": ["I1", "I2", "I3", "I4", "I5", "I6"],
            "customer_id": ["A", "A", "A", "B", "C", "C"],
            "invoice_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01",
                                            "2024-03-05", "2024-02-15", "2024-01-15"]),
            "amount_usd": [10.0, 20.0, 30.0, 99.0, 15.0, 12.0],
            "status": ["paid", "paid", "failed", "paid", "paid", "paid"],
        })
        got = _run(latest_invoice_sql("2024-03-01"), {"invoices": invoices})
        need = {"customer_id", "invoice_date", "amount_usd"}
        assert need <= set(got.columns), f"missing columns {sorted(need - set(got.columns))}"
        got = got.sort_values("customer_id").reset_index(drop=True)
        assert list(got["customer_id"]) == ["A", "C"], (
            f"as_of 2024-03-01: expected customers A and C (B's only invoice is later), got {list(got['customer_id'])}")
        assert got["amount_usd"].tolist() == [20.0, 15.0], (
            f"expected amounts [20.0, 15.0], got {got['amount_usd'].tolist()} - "
            "'strictly before' excludes an invoice dated exactly as_of")
        later = _run(latest_invoice_sql("2024-03-02"), {"invoices": invoices}).sort_values("customer_id")
        assert later["amount_usd"].tolist() == [30.0, 15.0], "the as_of argument must be used in the query"

        # the real data: must agree with mrr_usd from build_features
        from churnlab.features import build_features

        tables = _clean_tables()
        real = _run(latest_invoice_sql("2024-03-01"), {"invoices": tables["invoices"]})
        assert real["customer_id"].is_unique, "more than one row per customer on the real invoices"
        ref = build_features(tables, snapshots=[pd.Timestamp("2024-03-01")])
        merged = ref.merge(real, on="customer_id", how="inner")
        ref_with_invoice = (ref["mrr_usd"] > 0).sum()
        assert len(merged) >= ref_with_invoice, "some active customers with earlier invoices are missing"
        bad = merged[merged["mrr_usd"] != merged["amount_usd"]]
        assert bad.empty, f"{len(bad)} customers disagree with features.mrr_usd, e.g. {bad['customer_id'].iloc[0]}"

    return run_check("Exercise 2 · latest_invoice_sql", test,
                     hint="WHERE invoice_date < TIMESTAMP '{as_of}' ... QUALIFY ROW_NUMBER() OVER "
                          "(PARTITION BY customer_id ORDER BY invoice_date DESC) = 1")


def ex3(tickets_30d_sql):
    def test():
        from churnlab.features import build_features

        tables = _clean_tables()
        for snapshot in ("2024-02-01", "2023-09-01"):
            got = _run(tickets_30d_sql(snapshot), tables)
            assert {"customer_id", "tickets_30d"} <= set(got.columns), "need columns customer_id and tickets_30d"
            ref = build_features(tables, snapshots=[pd.Timestamp(snapshot)])[["customer_id", "tickets_30d"]]
            assert got["customer_id"].is_unique, f"{snapshot}: more than one row per customer"
            extra = set(got["customer_id"]) - set(ref["customer_id"])
            missing = set(ref["customer_id"]) - set(got["customer_id"])
            assert not missing, (f"{snapshot}: {len(missing)} active customers missing (customers with zero "
                                 "tickets must appear with 0 - LEFT JOIN from the active population)")
            assert not extra, (f"{snapshot}: {len(extra)} customers who were not active at the snapshot "
                               "(signed up on/after it, or already churned)")
            merged = ref.merge(got, on="customer_id", suffixes=("_ref", ""))
            assert merged["tickets_30d"].notna().all(), f"{snapshot}: NULL counts - COALESCE them to 0"
            diff = merged[merged["tickets_30d_ref"] != merged["tickets_30d"].astype(float)]
            assert diff.empty, (f"{snapshot}: {len(diff)} customers have a different count, e.g. "
                                f"{diff['customer_id'].iloc[0]}: yours {diff['tickets_30d'].iloc[0]}, "
                                f"reference {diff['tickets_30d_ref'].iloc[0]} - window is [T - 30 days, T)")

    return run_check("Exercise 3 · tickets_30d_sql", test,
                     hint="active = signup_date < T AND (churn_date IS NULL OR churn_date >= T); "
                          "LEFT JOIN tickets ON created_at >= T - INTERVAL 30 DAY AND created_at < T; COUNT(t.ticket_id)")
