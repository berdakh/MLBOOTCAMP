"""Self-checks for Level 3 notebook 17 · ML platform design."""

from __future__ import annotations

from churnlab.checks import close, run_check


def _raises_value_error(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def ex1(feature_store_storage_gb):
    def test():
        got = feature_store_storage_gb(200_000_000, 400, 4, 2, 3)
        want = 200_000_000 * 400 * 4 * 2 * 3 / 1e9
        if close(got, want * 1e9 / 2**30, 1e-6):
            raise AssertionError(f"got {got:.1f}: that is GiB (2**30 bytes); the exercise asks for decimal GB (1e9)")
        if close(got, want * 1000, 1e-6) or close(got, want / 1000, 1e-6):
            raise AssertionError(f"got {got}, expected {want:.1f} GB: off by 1,000 - check the unit conversion")
        assert close(got, want, 1e-9), f"200M x 400 x 4 B x 2 versions x 3 replicas: expected {want:,.1f} GB, got {got}"
        assert isinstance(got, float), "return a float"
        assert close(feature_store_storage_gb(1_000_000_000, 1, 1, 1, 1), 1.0), "1e9 one-byte values is exactly 1 GB"
        assert close(feature_store_storage_gb(10, 3, 2.5, 1, 1), 75e-9), "bytes_per may be fractional (average size)"
        assert close(feature_store_storage_gb(0, 400, 4, 2, 3), 0.0), "no entities, no storage"
        one = feature_store_storage_gb(1e6, 100, 4, 1, 1)
        assert close(feature_store_storage_gb(1e6, 100, 4, 3, 1), 3 * one), "versions multiply the storage"
        assert close(feature_store_storage_gb(1e6, 100, 4, 1, 3), 3 * one), "replication multiplies the storage"
        for bad in [(-1, 400, 4, 1, 1), (10, -1, 4, 1, 1), (10, 4, -4, 1, 1), (10, 4, 4, -1, 1), (10, 4, 4, 1, -3)]:
            assert _raises_value_error(feature_store_storage_gb, *bad), f"negative input {bad} must raise ValueError"

    return run_check("Exercise 1 · feature_store_storage_gb", test,
                     hint="validate (any x < 0 -> ValueError); "
                          "return entities * features * bytes_per * versions * replication / 1e9")


def ex2(gpu_cost):
    def test():
        assert close(gpu_cost(730, 64, 2.0), 93_440.0), f"64 GPUs x 730 h x 2.0: expected 93,440, got {gpu_cost(730, 64, 2.0)}"
        assert close(gpu_cost(3, 8, 0.5), 12.0), "3 hours x 8 GPUs x 0.5 per GPU-hour = 12"
        assert close(gpu_cost(0.25, 1, 4.0), 1.0), "hours may be fractional"
        assert close(gpu_cost(10, 0, 2.0), 0.0), "zero GPUs cost nothing"
        assert isinstance(gpu_cost(1, 1, 1), float), "return a float"
        for bad in [(-1, 1, 1.0), (1, -1, 1.0), (1, 1, -1.0)]:
            assert _raises_value_error(gpu_cost, *bad), f"negative input {bad} must raise ValueError"

    return run_check("Exercise 2 · gpu_cost", test,
                     hint="if min(hours, gpus, price) < 0: raise ValueError(...); return float(hours * gpus * price)")


def ex3(utilization_waste):
    def test():
        out = utilization_waste(730, 64, 2.0, 0.72)
        assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
        need = {"total_cost", "useful_cost", "wasted_cost", "cost_per_useful_gpu_hour"}
        assert need <= set(out), f"missing keys: {sorted(need - set(out))}"
        total = 730 * 64 * 2.0
        assert close(out["total_cost"], total), f"total_cost: expected {total:,.0f}, got {out['total_cost']}"
        assert close(out["useful_cost"], total * 0.72), "useful_cost = total_cost * utilization"
        assert close(out["wasted_cost"], total * 0.28), "wasted_cost = total_cost * (1 - utilization)"
        assert close(out["cost_per_useful_gpu_hour"], 2.0 / 0.72), "cost_per_useful_gpu_hour = price / utilization"
        full = utilization_waste(10, 2, 3.0, 1.0)
        assert close(full["wasted_cost"], 0.0) and close(full["cost_per_useful_gpu_hour"], 3.0), \
            "at 100% utilization nothing is wasted"
        assert close(out["useful_cost"] + out["wasted_cost"], out["total_cost"]), "useful + wasted must equal total"
        for bad in [(10, 2, 3.0, 0.0), (10, 2, 3.0, 72), (10, 2, 3.0, -0.1), (-10, 2, 3.0, 0.5)]:
            assert _raises_value_error(utilization_waste, *bad), (
                f"{bad} must raise ValueError (utilization is a fraction in (0, 1]; inputs non-negative)")

    return run_check("Exercise 3 · utilization_waste", test,
                     hint="check 0 < utilization <= 1; total = hours*gpus*price; "
                          "return {'total_cost': total, 'useful_cost': total*u, 'wasted_cost': total*(1-u), "
                          "'cost_per_useful_gpu_hour': price/u}")
