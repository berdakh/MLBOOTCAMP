"""Self-checks for Level 3 notebook 18 · Large-scale design cases."""

from __future__ import annotations

from churnlab.checks import close, run_check


def _raises_value_error(fn, *args) -> str | None:
    try:
        fn(*args)
    except ValueError as exc:
        return str(exc) or " "
    return None


def ex1(latency_budget_split):
    def test():
        stages = {"network": 10, "features": 30, "model": 8, "decision": 4}
        before = dict(stages)
        out = latency_budget_split(100, stages)
        assert stages == before, "do not modify the stages dict"
        assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
        need = {"remaining_ms", "shares", "largest_stage"}
        assert need <= set(out), f"missing keys: {sorted(need - set(out))}"
        assert close(out["remaining_ms"], 48), f"100 - 52 ms of stages leaves 48 ms, got {out['remaining_ms']}"
        assert set(out["shares"]) == set(stages), "shares needs one entry per stage (and no HEADROOM entry)"
        assert close(out["shares"]["features"], 0.30), (
            f"shares are fractions of total_ms: features = 30/100 = 0.30, got {out['shares']['features']}")
        assert close(out["shares"]["decision"], 0.04), "decision = 4/100 = 0.04"
        assert out["largest_stage"] == "features", f"largest stage is 'features', got {out['largest_stage']!r}"
        exact = latency_budget_split(52, stages)
        assert close(exact["remaining_ms"], 0), "exactly on budget is allowed and leaves 0 ms"
        empty = latency_budget_split(80, {})
        assert close(empty["remaining_ms"], 80) and empty["shares"] == {} and empty["largest_stage"] is None, (
            "no stages: remaining = total, shares = {}, largest_stage = None")
        frac = latency_budget_split(1.5, {"a": 0.25, "b": 0.5})
        assert close(frac["remaining_ms"], 0.75) and frac["largest_stage"] == "b", "milliseconds may be fractional"
        msg = _raises_value_error(latency_budget_split, 40, stages)
        assert msg is not None, "52 ms of stages in a 40 ms budget must raise ValueError"
        assert "12" in msg, f"say by how much the budget is exceeded (12 ms) in the message; got {msg!r}"
        assert _raises_value_error(latency_budget_split, 100, {"a": -5, "b": 10}) is not None, \
            "a negative stage is a bug in the inputs: raise ValueError"
        assert _raises_value_error(latency_budget_split, 0, {}) is not None, "total_ms <= 0 must raise ValueError"

    return run_check("Exercise 1 · latency_budget_split", test,
                     hint="validate total_ms > 0 and every stage >= 0; spent = sum(...); if spent > total_ms raise "
                          "ValueError(f'over by {spent - total_ms} ms'); shares = {k: v / total_ms}; "
                          "largest = max(stages, key=stages.get) if stages else None")


def ex2(serving_fleet_size):
    def test():
        got = serving_fleet_size(138_889, 50, 0.3)
        assert isinstance(got, int) and not isinstance(got, bool), f"return an int (replicas), got {type(got).__name__}"
        assert got == 3969, f"138,889 QPS at 50 per replica with 30% headroom: expected 3,969, got {got}"
        assert serving_fleet_size(1000, 100, 0.0) == 10, "no headroom: exactly load / capacity"
        assert serving_fleet_size(1001, 100, 0.0) == 11, "round UP: 10.01 replicas means 11"
        assert serving_fleet_size(700, 100, 0.3) == 10, "700 / (100 * 0.7) = 10 exactly"
        assert serving_fleet_size(5, 100, 0.3) == 2, "never fewer than 2 replicas"
        assert serving_fleet_size(0, 100, 0.3) == 2, "even with no traffic, keep 2 for availability"
        for bad in [(100, 0, 0.3), (100, -5, 0.3), (-1, 100, 0.3), (100, 100, 1.0), (100, 100, -0.1)]:
            assert _raises_value_error(serving_fleet_size, *bad) is not None, f"{bad} must raise ValueError"

    return run_check("Exercise 2 · serving_fleet_size", test,
                     hint="validate; n = math.ceil(peak_qps / (per_replica_qps * (1 - headroom))); return max(2, int(n))")
