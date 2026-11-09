"""Self-checks for Chapter 14 · Health, logs & latency."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from churnlab.checks import run_check


def _nearest_rank(values, q: float) -> float:
    """Nearest-rank percentile: x_k with k = ceil(q * n / 100), 1-based, on sorted values."""
    xs = sorted(float(v) for v in values)
    k = max(1, math.ceil(q * len(xs) / 100))
    return xs[k - 1]


def ex1(percentiles):
    def test():
        got = percentiles([15, 20, 35, 40, 50], qs=(50, 95))
        assert isinstance(got, dict), f"return a dict, got {type(got).__name__}"
        assert set(got) == {50, 95}, f"keys should be the requested qs, got {sorted(got)}"
        assert float(got[50]) == 35 and float(got[95]) == 50, f"textbook example wrong: {got}"
        got = percentiles([3.0, 1.0, 2.0])
        assert set(got) == {50, 95, 99}, "default qs should be (50, 95, 99)"
        assert float(got[50]) == 2.0, "unsorted input: sort before ranking"
        assert float(percentiles([7.5], qs=(50, 99))[99]) == 7.5, "a single value is every percentile"
        ten = list(range(1, 11))
        got = percentiles(ten, qs=(10, 50, 90, 100))
        assert [float(got[q]) for q in (10, 50, 90, 100)] == [1, 5, 9, 10], (
            f"for 1..10 expected p10=1, p50=5, p90=9, p100=10 (no interpolation), got {got}")
        rng = np.random.default_rng(13)
        for n in (20, 100, 200, 1000, 2000):
            data = rng.lognormal(1.5, 0.6, n)
            got = percentiles(data, qs=(50, 95, 99))
            for q in (50, 95, 99):
                expected = _nearest_rank(data, q)
                assert abs(float(got[q]) - expected) < 1e-12, (
                    f"n={n}, p{q}: expected {expected:.6f}, got {float(got[q]):.6f}")
                assert float(got[q]) in set(data.tolist()), "nearest-rank must return an observed value"
        try:
            percentiles([])
        except ValueError:
            pass
        else:
            raise AssertionError("an empty input should raise ValueError")

    return run_check("Exercise 1 · percentiles (nearest rank)", test,
                     hint="xs = sorted(latencies); k = math.ceil(q * len(xs) / 100); value = xs[k - 1]")


def _records():
    rng = np.random.default_rng(7)
    records = [{"msg": "model_loaded", "model_version": "v1", "level": "INFO"}]
    plan = [("/predict", 200, 37), ("/predict", 500, 3), ("/predict", 422, 4),
            ("/predict/batch", 200, 5), ("/predict/batch", 503, 1), ("/health", 200, 2)]
    for path, status, n in plan:
        for _ in range(n):
            records.append({"msg": "request", "request_id": f"r{len(records)}", "method": "GET",
                            "path": path, "status": status,
                            "latency_ms": round(float(rng.lognormal(1.5, 0.7)), 2), "model_version": "v1"})
    records.append({"msg": "model_reload", "previous": "v1", "current": "v1"})
    order = rng.permutation(len(records))
    return [records[i] for i in order]


def _reference(records) -> pd.DataFrame:
    rows = {}
    for r in records:
        if r.get("msg") != "request":
            continue
        rows.setdefault(r["path"], []).append(r)
    out = []
    for path in sorted(rows):
        group = rows[path]
        out.append({"path": path, "count": len(group),
                    "error_rate": sum(r["status"] >= 500 for r in group) / len(group),
                    "p95_latency_ms": _nearest_rank([r["latency_ms"] for r in group], 95)})
    return pd.DataFrame(out).set_index("path")


def ex2(summarize_logs):
    def test():
        records = _records()
        got = summarize_logs([dict(r) for r in records])
        expected = _reference(records)
        assert isinstance(got, pd.DataFrame), f"return a DataFrame, got {type(got).__name__}"
        assert got.index.name == "path", f"the index should be named 'path', got {got.index.name!r}"
        assert list(got.index) == list(expected.index), (
            f"one row per path, sorted: expected {list(expected.index)}, got {list(got.index)}")
        for col in ("count", "error_rate", "p95_latency_ms"):
            assert col in got.columns, f"missing column {col!r}"
        assert [int(x) for x in got["count"]] == list(expected["count"]), (
            f"count wrong (only msg == 'request' records count): {got['count'].to_dict()}")
        for path in expected.index:
            e, g = expected.loc[path], got.loc[path]
            assert abs(float(g["error_rate"]) - e["error_rate"]) < 1e-9, (
                f"{path}: error_rate expected {e['error_rate']:.4f}, got {float(g['error_rate']):.4f} "
                "(errors are status >= 500; a 422 is not)")
            assert abs(float(g["p95_latency_ms"]) - e["p95_latency_ms"]) < 1e-9, (
                f"{path}: p95 expected {e['p95_latency_ms']} (nearest rank), got {float(g['p95_latency_ms'])}")

    return run_check("Exercise 2 · summarize_logs", test,
                     hint="df = pd.DataFrame(records); keep df['msg'] == 'request'; then groupby('path') "
                          "with count, (status >= 500).mean() and your nearest-rank p95")
