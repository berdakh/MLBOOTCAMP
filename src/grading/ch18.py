"""Self-checks for Chapter 18 · System design & mock interview."""

from __future__ import annotations

from churnlab.checks import close, run_check


def ex1(peak_qps):
    def test():
        got = peak_qps(1_000_000, 20, 3)
        want = 1_000_000 * 20 / 86_400 * 3
        assert close(got, want, 1e-6), f"1M DAU x 20 requests/day x peak factor 3: expected {want:.1f} QPS, got {got}"
        assert close(peak_qps(86_400, 1, 1), 1.0), "86,400 requests spread evenly over a day is 1 request/second"
        assert close(peak_qps(10_000_000, 0.5, 4), 10_000_000 * 0.5 / 86_400 * 4), "fractional requests per user"
        assert close(peak_qps(0, 10, 3), 0.0), "no users, no traffic"
        assert isinstance(peak_qps(1000, 10, 2), float), "return a float (QPS is a rate, not a count)"

    return run_check("Exercise 1 · peak_qps", test,
                     hint="average QPS = dau * requests_per_user_per_day / 86_400 seconds; multiply by peak_factor")


def ex2(latency_budget):
    def test():
        components = {"network": 20, "feature_lookup": 15, "model": 10, "serialisation": 5}
        got = latency_budget(100, components)
        assert close(got, 50), f"100 ms total - 50 ms of components should leave 50 ms, got {got}"
        assert close(latency_budget(50, components), 0), "exactly on budget leaves 0 ms (and is allowed)"
        assert close(latency_budget(80, {}), 80), "no components: the whole budget remains"
        assert close(latency_budget(30.5, {"a": 0.25, "b": 0.25}), 30.0), "milliseconds may be fractional"
        try:
            latency_budget(40, components)
        except ValueError as exc:
            msg = str(exc)
            assert msg, "give the ValueError a message saying by how much the budget is exceeded"
        else:
            raise AssertionError("components totalling 50 ms in a 40 ms budget must raise ValueError")
        before = dict(components)
        latency_budget(100, components)
        assert components == before, "do not modify the components dict"

    return run_check("Exercise 2 · latency_budget", test,
                     hint="spent = sum(components.values()); if spent > total_ms: raise ValueError(...); "
                          "return total_ms - spent")
