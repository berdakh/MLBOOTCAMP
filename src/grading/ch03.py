"""Self-checks for Chapter 3 · Data acquisition & validation."""

from __future__ import annotations

import pandas as pd

from churnlab.checks import run_check

#: The timestamp column that says how recent each table is.
TIMESTAMP_COLUMNS = {
    "customers": "signup_date",
    "usage_weekly": "week_start",
    "invoices": "invoice_date",
    "tickets": "created_at",
}


def ex1(find_duplicate_keys):
    def test():
        df = pd.DataFrame({
            "customer_id": ["A", "A", "B", "C", "C", "C", "D"],
            "week_start": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-08",
                                          "2024-01-08", "2024-01-15", "2024-01-01"]),
            "sessions": [3, 3, 1, 2, 5, 4, 0],
        }, index=[10, 11, 12, 13, 14, 15, 16])
        before = df.copy()
        got = find_duplicate_keys(df, ["customer_id", "week_start"])
        pd.testing.assert_frame_equal(df, before, obj="the input DataFrame (do not modify it)")
        assert isinstance(got, pd.DataFrame), f"expected a DataFrame, got {type(got).__name__}"
        assert len(got) == 4, (f"expected 4 rows (BOTH copies of A/2024-01-01 and of C/2024-01-08), got {len(got)}; "
                               "keep=False keeps every member of a duplicated group")
        assert sorted(got.index) == [10, 11, 13, 14], f"wrong rows returned: index {sorted(got.index)}"
        assert set(got.columns) == set(df.columns), "return whole rows (all columns), not just the key"
        one = find_duplicate_keys(df, "customer_id")
        assert sorted(one.index) == [10, 11, 13, 14, 15], f"a single string key should work too; got {sorted(one.index)}"
        none = find_duplicate_keys(df.drop_duplicates(["customer_id"]), "customer_id")
        assert len(none) == 0, "a frame with no duplicated keys should give an empty DataFrame"

    return run_check("Exercise 1 · find_duplicate_keys", test,
                     hint="df[df.duplicated(subset=key, keep=False)] - and sort by the key so copies sit together")


def ex2(orphan_ids):
    def test():
        tables = {
            "customers": pd.DataFrame({"customer_id": ["C1", "C2", "C3"]}),
            "usage_weekly": pd.DataFrame({"customer_id": ["C1", "C4", "C4", "C2"]}),
            "invoices": pd.DataFrame({"customer_id": ["C5", "C1", "C4"]}),
            "tickets": pd.DataFrame({"customer_id": ["C3", "C3"]}),
        }
        got = orphan_ids(tables)
        assert isinstance(got, dict), f"expected a dict {{table: set_of_ids}}, got {type(got).__name__}"
        for name in ("usage_weekly", "invoices", "tickets"):
            assert name in got, f"missing key {name!r} (include tables with no orphans, as an empty set)"
        assert "customers" not in got, "customers is the reference table; do not include it"
        assert set(got["usage_weekly"]) == {"C4"}, f"usage_weekly orphans should be {{'C4'}}, got {got['usage_weekly']}"
        assert set(got["invoices"]) == {"C4", "C5"}, f"invoices orphans should be {{'C4', 'C5'}}, got {got['invoices']}"
        assert set(got["tickets"]) == set(), f"tickets has no orphans, got {got['tickets']}"

    return run_check("Exercise 2 · orphan_ids", test,
                     hint="known = set(tables['customers'].customer_id); "
                          "{name: set(df.customer_id) - known for name, df in ...}")


def ex3(freshness):
    def test():
        tables = {
            "customers": pd.DataFrame({"signup_date": pd.to_datetime(["2023-05-01", "2022-01-03"])}),
            "usage_weekly": pd.DataFrame({"week_start": pd.to_datetime(["2024-01-01", "2024-02-05"])}),
            "invoices": pd.DataFrame({"invoice_date": pd.to_datetime(["2024-03-01", "2023-12-01"])}),
            "tickets": pd.DataFrame({"created_at": pd.to_datetime(["2024-02-10 13:00", None, "2023-07-01 08:30"])}),
        }
        got = freshness(tables)
        assert isinstance(got, pd.DataFrame), f"expected a DataFrame, got {type(got).__name__}"
        assert {"min", "max"} <= set(got.columns), f"need columns 'min' and 'max', got {list(got.columns)}"
        assert set(got.index) == set(tables), f"index should be the table names, got {list(got.index)}"
        expect = {
            "customers": ("2022-01-03", "2023-05-01"),
            "usage_weekly": ("2024-01-01", "2024-02-05"),
            "invoices": ("2023-12-01", "2024-03-01"),
            "tickets": ("2023-07-01 08:30", "2024-02-10 13:00"),
        }
        for name, (lo, hi) in expect.items():
            got_lo, got_hi = pd.Timestamp(got.loc[name, "min"]), pd.Timestamp(got.loc[name, "max"])
            assert got_lo == pd.Timestamp(lo), f"{name}: min should be {lo}, got {got_lo}"
            assert got_hi == pd.Timestamp(hi), f"{name}: max should be {hi}, got {got_hi} (missing timestamps must be ignored)"

    return run_check("Exercise 3 · freshness", test,
                     hint="loop over TIMESTAMP_COLUMNS: {'min': df[col].min(), 'max': df[col].max()}; "
                          "pd.DataFrame.from_dict(rows, orient='index')")
