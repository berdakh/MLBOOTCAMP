"""Reference solutions for notebook 00. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd


def ex1(items, k):
    counts = Counter(items)
    return sorted(counts, key=lambda item: (-counts[item], item))[:k]


def ex2(usage: pd.DataFrame) -> pd.Series:
    weeks = pd.Index(sorted(usage["week_start"].unique()), name="week_start")
    active = usage[usage["sessions"] > 0]  # NaN > 0 and -1 > 0 are both False
    counts = active.groupby("week_start")["customer_id"].nunique()
    return counts.reindex(weeks, fill_value=0).rename("active_customers")


def ex3() -> str:
    return """
    WITH ranked AS (
        SELECT customer_id, amount_usd,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) AS rn
        FROM invoices
    ),
    latest AS (
        SELECT customer_id, amount_usd AS latest_amount_usd FROM ranked WHERE rn = 1
    ),
    failed AS (
        SELECT customer_id, COUNT(*) FILTER (WHERE status = 'failed') AS failed_invoices
        FROM invoices GROUP BY customer_id
    )
    SELECT c.customer_id, c.plan, l.latest_amount_usd,
           COALESCE(f.failed_invoices, 0) AS failed_invoices
    FROM customers AS c
    LEFT JOIN latest AS l USING (customer_id)
    LEFT JOIN failed AS f USING (customer_id)
    ORDER BY c.customer_id
    """


def ex4(fn):
    """A test function: passes for a correct rolling_mean_last_n, fails otherwise."""
    # 1. a hand-computed example, long enough that the window is full
    values = [1, 2, 3, 4, 5]
    original = list(values)
    got = fn(values, 3)
    expected = [1.0, 1.5, 2.0, 3.0, 4.0]
    assert len(got) == len(expected), f"length {len(got)} != {len(expected)}"
    assert all(math.isclose(a, b) for a, b in zip(got, expected)), f"{got} != {expected}"
    # 2. the input must not be modified
    assert values == original, "the input list was modified"
    # 3. input shorter than the window: running mean of what there is
    got = fn([2, 4], 5)
    assert len(got) == 2 and all(math.isclose(a, b) for a, b in zip(got, [2.0, 3.0])), got
    # 4. edge cases: n = 1 is the identity; empty input gives empty output
    assert [float(x) for x in fn([7, 8, 9], 1)] == [7.0, 8.0, 9.0]
    assert list(fn([], 3)) == []


def ex5() -> dict:
    return {1: "b", 2: "c", 3: "a", 4: "d", 5: "b", 6: "a"}


def ex6():
    from pydantic import BaseModel, ConfigDict, Field

    class Order(BaseModel):
        model_config = ConfigDict(extra="forbid")

        order_id: str = Field(min_length=1)
        quantity: int = Field(ge=1, le=1000)
        unit_price: float = Field(gt=0)
        coupon: str | None = None

    return Order
