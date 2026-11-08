"""Reference solutions for notebook 13. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math

import pandas as pd


def ex1(latencies, qs=(50, 95, 99)) -> dict:
    xs = sorted(float(x) for x in latencies)
    if not xs:
        raise ValueError("no latencies")
    n = len(xs)
    return {q: xs[max(1, math.ceil(q * n / 100)) - 1] for q in qs}


def ex2(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in records if r.get("msg") == "request"])
    rows = []
    for path, group in df.groupby("path", sort=True):
        rows.append({
            "path": path,
            "count": len(group),
            "error_rate": float((group["status"] >= 500).mean()),
            "p95_latency_ms": ex1(group["latency_ms"], qs=(95,))[95],
        })
    return pd.DataFrame(rows).set_index("path")
