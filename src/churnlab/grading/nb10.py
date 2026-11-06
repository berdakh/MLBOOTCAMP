"""Self-checks for notebook 10 · Consistent serving."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..checks import run_check
from ..features import CATEGORICAL, FEATURES, NUMERIC
from ..modeling import LOG_FEATURES

# -- the reference training transform (identical to the one in notebook 10) ----


def reference_fit_stats(df: pd.DataFrame) -> dict:
    X = df[NUMERIC].astype(float)
    median = X.median()
    X = X.fillna(median)
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return {"median": median.to_dict(), "mean": X.mean().to_dict(),
            "std": X.std(ddof=0).replace(0, 1.0).to_dict(),
            "categories": {c: sorted(df[c].astype(str).unique()) for c in CATEGORICAL}}


def reference_transform(df: pd.DataFrame, stats: dict) -> np.ndarray:
    X = df[NUMERIC].astype(float).fillna(stats["median"])
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    X = (X - pd.Series(stats["mean"])) / pd.Series(stats["std"])
    onehot = [(df[c].astype(str) == v).to_numpy(float) for c in CATEGORICAL for v in stats["categories"][c]]
    return np.column_stack([X[NUMERIC].to_numpy(), *onehot])


class _Shifted:
    """A model that is 'almost' the served one: every score moved up by ``shift``."""

    def __init__(self, model, shift: float = 0.03):
        self.model, self.shift = model, shift

    def predict_proba(self, X):
        p = np.clip(self.model.predict_proba(X)[:, 1] + self.shift, 0, 1)
        return np.column_stack([1 - p, p])


def _served_setup():
    from fastapi.testclient import TestClient

    from ..api import create_app
    from ..artifacts import load_bundle
    from ..bootstrap import ensure_registry, ensure_split
    from ..logs import ListHandler, get_logger
    from ..settings import Settings

    registry = ensure_registry(verbose=False)
    model, _ = load_bundle(registry.root / registry.current())
    app = create_app(Settings(registry_dir=registry.root),
                     logger=get_logger("churn-api-grader", handler=ListHandler()))
    rows = ensure_split().valid.sample(300, random_state=7).reset_index(drop=True)
    return model, TestClient(app), rows


def ex1(parity_check):
    def test():
        model, client, rows = _served_setup()
        out = parity_check(model, client, rows)
        assert isinstance(out, dict), f"expected a dict, got {type(out).__name__}"
        for key in ("max_abs_diff", "n_decision_flips"):
            assert key in out, f"missing key {key!r}"
        assert float(out["max_abs_diff"]) <= 1e-6, (
            f"the served model IS the offline model, yet max_abs_diff = {out['max_abs_diff']}"
            " (the API rounds to 6 decimals, so expect < 5e-7)")
        assert int(out["n_decision_flips"]) == 0, f"expected 0 flips for the same model, got {out['n_decision_flips']}"
        assert float(out["max_abs_diff"]) == float(out["max_abs_diff"]), "max_abs_diff is NaN"

        # A subtly wrong offline model: every score shifted up just enough that
        # the highest-scoring unflagged row crosses the threshold.
        threshold = float(client.get("/model").json()["threshold"])
        p = model.predict_proba(rows[FEATURES])[:, 1]
        shift = max(0.03, threshold - float(p[p < threshold].max()) + 1e-3)
        bad = parity_check(_Shifted(model, shift), client, rows)
        assert float(bad["max_abs_diff"]) > 0.02, (
            f"an offline model shifted by +{shift:.3f} should give max_abs_diff about {shift:.3f}, "
            f"got {bad['max_abs_diff']}")
        assert int(bad["n_decision_flips"]) > 0, (
            f"shifting every score by +{shift:.3f} moves at least one row across the threshold; your check "
            "reported 0 flips (compare offline p >= threshold from /model with the API's 'flagged')")
        assert isinstance(out["n_decision_flips"], (int, np.integer)), "n_decision_flips should be an int"

    return run_check("Exercise 1 · parity_check", test,
                     hint="threshold = http_client.get('/model').json()['threshold']; "
                          "preds = http_client.post('/predict/batch', json={'customers': [...]}).json()['predictions']")


def _random_payloads(seed: int = 0, n: int = 120) -> tuple[list[dict], pd.DataFrame]:
    from ..api import example_payload
    from ..bootstrap import ensure_split

    rng = np.random.default_rng(seed)
    rows = ensure_split().valid.sample(n, random_state=seed).reset_index(drop=True)
    # make missing values and an unseen category common enough to matter
    rows.loc[rng.random(n) < 0.3, "features_used_4w"] = np.nan
    rows.loc[rng.random(n) < 0.3, "avg_resolution_hours_90d"] = np.nan
    rows.loc[rng.random(n) < 0.1, "region"] = "MARS"
    payloads = []
    for _, row in rows.iterrows():
        body = example_payload(row)
        if rng.random() < 0.3:  # omit the optional fields instead of sending null
            body = {k: v for k, v in body.items() if v is not None}
        items = list(body.items())
        rng.shuffle(items)  # any key order
        payloads.append(dict(items))
    return payloads, rows


def ex2(fixed_serve_preprocess):
    def test():
        from ..bootstrap import ensure_split

        stats = reference_fit_stats(ensure_split().train)
        for seed in (0, 1):
            payloads, rows = _random_payloads(seed)
            expected = reference_transform(rows, stats)
            got = np.asarray(fixed_serve_preprocess(payloads, stats), dtype=float)
            assert got.shape == expected.shape, f"shape {got.shape}, expected {expected.shape}"
            n_num = len(NUMERIC)
            bad_cols = np.flatnonzero(~np.isclose(got, expected, atol=1e-9).all(axis=0))
            if len(bad_cols):
                c = int(bad_cols[0])
                name = NUMERIC[c] if c < n_num else f"one-hot column {c - n_num}"
                raise AssertionError(f"{len(bad_cols)} column(s) differ from the training transform; "
                                     f"first: column {c} ({name})")

    return run_check("Exercise 2 · fixed_serve_preprocess", test,
                     hint="build a DataFrame from the dicts (missing keys become NaN) and reuse train_transform")
