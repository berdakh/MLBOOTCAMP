"""Reference solutions for notebook 02. Graded by tests/test_solutions.py."""

from __future__ import annotations

import pandas as pd

TIMESTAMP_COLUMNS = {
    "customers": "signup_date",
    "usage_weekly": "week_start",
    "invoices": "invoice_date",
    "tickets": "created_at",
}


def ex1(df: pd.DataFrame, key) -> pd.DataFrame:
    key = [key] if isinstance(key, str) else list(key)
    return df[df.duplicated(subset=key, keep=False)].sort_values(key)


def ex2(tables: dict[str, pd.DataFrame]) -> dict[str, set]:
    known = set(tables["customers"]["customer_id"])
    return {name: set(tables[name]["customer_id"]) - known
            for name in ("usage_weekly", "invoices", "tickets") if name in tables}


def ex3(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for name, col in TIMESTAMP_COLUMNS.items():
        if name in tables:
            ts = tables[name][col]
            rows[name] = {"min": ts.min(), "max": ts.max(), "rows": len(ts)}
    return pd.DataFrame.from_dict(rows, orient="index")
