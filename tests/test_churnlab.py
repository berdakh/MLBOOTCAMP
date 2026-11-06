"""Unit tests for the churnlab package. No network; runs in about a minute."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnlab import data as data_mod
from churnlab.features import FEATURES, LABEL, NUMERIC, build_features
from churnlab.validate import Column, clean, validate, validate_all, validate_clean

ROOT_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


# --- data --------------------------------------------------------------------

def test_committed_raw_data_matches_checksums():
    assert all(data_mod.verify_raw(ROOT_RAW).values())


def test_load_raw_keeps_north_america():
    customers = data_mod.load_raw(ROOT_RAW)["customers"]
    assert (customers["region"] == "NA").sum() > 1000, "'NA' must not be parsed as missing"
    assert customers["region"].isna().mean() < 0.05


def test_fetch_verifies_checksum(tmp_path):
    src = ROOT_RAW / "customers.csv.gz"
    good = data_mod.fetch(src.as_uri(), tmp_path / "c.csv.gz", sha256=data_mod.RAW_SHA256["customers"])
    assert good.exists()
    with pytest.raises(ValueError, match="checksum"):
        data_mod.fetch(src.as_uri(), tmp_path / "d.csv.gz", sha256="0" * 64)
    assert not (tmp_path / "d.csv.gz").exists() and not (tmp_path / "d.csv.gz.part").exists()


def test_generator_is_small_and_has_expected_tables():
    tables = data_mod.generate(data_mod.GeneratorConfig(n_customers=200, n_weeks=20, seed=1))
    assert set(tables) == set(data_mod.TABLES)
    assert tables["customers"]["customer_id"].is_unique


# --- validation ----------------------------------------------------------------

def test_raw_data_has_the_planted_defects():
    reports = validate_all(data_mod.load_raw(ROOT_RAW))
    rules = {(i.column, i.rule.split()[0]) for r in reports.values() for i in r.issues}
    assert ("plan", "value") in rules
    assert ("sessions", "below") in rules
    assert ("customer_id+week_start", "duplicate") in rules


def test_clean_data_passes_its_contract():
    reports = validate_clean(clean(data_mod.load_raw(ROOT_RAW)))
    assert all(r.ok for r in reports.values()), [str(r) for r in reports.values() if not r.ok]


def test_validate_reports_missing_column_and_types():
    df = pd.DataFrame({"a": ["x", "y"]})
    report = validate(df, [Column("a", "int"), Column("b", "str")], "t")
    assert not report.ok and len(report.issues) == 2
    with pytest.raises(ValueError):
        report.raise_if_failed()


# --- features ------------------------------------------------------------------

def test_feature_table_shape_and_label(features):
    assert set(FEATURES + [LABEL, "snapshot_date", "customer_id"]) <= set(features.columns)
    assert features.duplicated(["snapshot_date", "customer_id"]).sum() == 0
    assert 0.03 < features[LABEL].mean() < 0.09
    assert features[NUMERIC].drop(columns=["features_used_4w", "avg_resolution_hours_90d"]).notna().all().all()


def test_features_are_point_in_time():
    """Adding future rows must not change features for an earlier snapshot."""
    tables = clean(data_mod.load_raw(ROOT_RAW))
    t = pd.Timestamp("2024-01-01")
    before = build_features(tables, snapshots=[t]).set_index("customer_id")
    future = {k: v.copy() for k, v in tables.items()}
    extra = future["tickets"].head(500).copy()
    extra["created_at"] = t + pd.Timedelta(hours=1)
    extra["ticket_id"] = "X" + extra["ticket_id"]
    future["tickets"] = pd.concat([future["tickets"], extra])
    week = future["usage_weekly"].head(500).copy()
    week["week_start"] = t - pd.Timedelta(days=3)  # a week that ends after T
    week["sessions"] = 999
    future["usage_weekly"] = pd.concat([future["usage_weekly"], week])
    after = build_features(future, snapshots=[t]).set_index("customer_id")
    cols = [c for c in NUMERIC if c not in ("features_used_4w",)]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_leaky_columns_are_not_features():
    from churnlab.features import LEAKY_COLUMNS

    assert not set(LEAKY_COLUMNS) & set(FEATURES)


# --- splits --------------------------------------------------------------------

def test_time_split_has_embargo(split):
    from churnlab.splits import assert_no_leakage

    assert_no_leakage(split)
    assert split.train["snapshot_date"].max() < split.valid["snapshot_date"].min()


def test_time_split_without_gap_does_not_duplicate_boundary(features):
    from churnlab.splits import time_split

    s = time_split(features, gap_days=0)
    assert not set(s.train["snapshot_date"]) & set(s.valid["snapshot_date"])
    assert len(s.train) + len(s.valid) + len(s.test) == len(features)


def test_group_split_separates_customers(features):
    from churnlab.splits import assert_no_leakage, group_split

    s = group_split(features, seed=3)
    with pytest.raises(AssertionError):
        assert_no_leakage(s)  # time overlaps, by design
    a, b = set(s.train.customer_id), set(s.test.customer_id)
    assert not a & b


def test_rolling_origin_folds_respect_time(split):
    from churnlab.splits import rolling_origin_folds

    folds = list(rolling_origin_folds(split.train, n_folds=3))
    assert len(folds) == 3
    for tr, va in folds:
        t = split.train["snapshot_date"]
        assert t.iloc[tr].max() + pd.Timedelta(days=30) <= t.iloc[va].min()


# --- evaluation ----------------------------------------------------------------

def test_metrics_basic():
    from churnlab.evaluation import classification_report, lift_at_fraction, precision_at_k

    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    assert precision_at_k(y, s, 3) == 1.0
    assert lift_at_fraction(y, s, 0.5) == pytest.approx(2.0)
    assert classification_report(y, s)["roc_auc"] == 1.0


def test_bootstrap_ci_contains_point_and_cluster_is_wider():
    from churnlab.evaluation import bootstrap_ci

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 10)
    signal = np.repeat(rng.normal(size=200), 10)
    y = (signal + rng.normal(scale=0.5, size=2000) > 1).astype(int)
    s = signal + rng.normal(scale=1.0, size=2000)
    p, lo, hi = bootstrap_ci(y, s, n_boot=200)
    pc, loc, hic = bootstrap_ci(y, s, n_boot=200, groups=groups)
    assert lo <= p <= hi and loc <= pc <= hic
    assert (hic - loc) > (hi - lo)


def test_threshold_helpers():
    from churnlab.evaluation import best_threshold_by_value, confusion_at, threshold_for_precision

    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    assert confusion_at(y, s, 0.65) == {"tp": 2, "fp": 0, "fn": 0, "tn": 3}
    t, v = best_threshold_by_value(y, s, value_tp=10, cost_fp=5)
    assert 0.6 < t <= 0.7 and v == 20
    assert threshold_for_precision(y, s, 1.0) == pytest.approx(0.7)


# --- artifacts and registry ----------------------------------------------------

def test_registry_promote_rollback(registry):
    assert registry.versions() == ["v1", "v2"]
    current = registry.current()
    meta = registry.metadata(current)
    assert meta["features"] == FEATURES
    assert meta["metrics"]["validation"]["pr_auc"] >= max(
        registry.metadata(v)["metrics"]["validation"]["pr_auc"] for v in registry.versions())


def test_rollback_restores_previous(tmp_path, split):
    from churnlab.bootstrap import train_and_register
    from churnlab.registry import Registry

    reg = Registry(tmp_path / "reg")
    small = type(split)(split.train.sample(3000, random_state=0), split.valid, split.test)
    v1 = train_and_register(reg, "logreg", small)
    v2 = train_and_register(reg, "dummy", small)
    reg.promote(v1)
    reg.promote(v2)
    assert reg.rollback("bad") == v1
    assert reg.current() == v1
    assert [e["event"] for e in reg.history()] == ["register", "register", "promote", "promote", "rollback"]


def test_tampered_bundle_is_rejected(registry, tmp_path):
    import shutil

    from churnlab.artifacts import ArtifactError, load_bundle

    copy = tmp_path / "copy"
    shutil.copytree(registry.root / registry.current(), copy)
    with open(copy / "model.joblib", "ab") as fh:
        fh.write(b"\0")
    with pytest.raises(ArtifactError, match="hash"):
        load_bundle(copy)


def test_bundles_are_immutable(registry):
    from churnlab.artifacts import ArtifactError, ModelMetadata, save_bundle

    with pytest.raises(ArtifactError):
        save_bundle(object(), ModelMetadata("x", "v1", [], [], 0.5), registry.root / "v1")


# --- API -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(registry):
    from fastapi.testclient import TestClient

    from churnlab.api import create_app
    from churnlab.logs import ListHandler, get_logger
    from churnlab.settings import Settings

    handler = ListHandler()
    app = create_app(Settings(registry_dir=registry.root, max_batch_size=50),
                     logger=get_logger("test-api", handler=handler))
    app.state.test_handler = handler
    return TestClient(app)


def payloads(split, n=5):
    from churnlab.api import example_payload

    return [example_payload(r) for _, r in split.valid.head(n).iterrows()]


def test_health_ready_model(client, registry):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").json()["model_version"] == registry.current()
    assert client.get("/model").json()["features"] == FEATURES


def test_predict_matches_offline_model(client, registry, split):
    model, meta = registry.load_current()
    rows = split.valid.head(40)
    offline = model.predict_proba(rows[FEATURES])[:, 1]
    from churnlab.api import example_payload

    body = {"customers": [example_payload(r) for _, r in rows.iterrows()]}
    r = client.post("/predict/batch", json=body)
    assert r.status_code == 200
    online = np.array([p["churn_probability"] for p in r.json()["predictions"]])
    np.testing.assert_allclose(online, offline, atol=1e-6)
    assert r.headers["x-model-version"] == meta["version"]


def test_contract_errors(client, split):
    body = payloads(split, 1)[0]
    assert client.post("/predict", json={**body, "sessions_4w": -1}).status_code == 422
    assert client.post("/predict", json={**body, "typo_field": 1}).status_code == 422
    big = {"customers": payloads(split, 1) * 51}
    assert client.post("/predict/batch", json=big).status_code == 413


def test_unseen_category_still_predicts(client, split):
    body = {**payloads(split, 1)[0], "region": "MARS"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1


def test_request_id_is_echoed_and_logged(client, split):
    r = client.post("/predict", json=payloads(split, 1)[0], headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    logged = [rec for rec in client.app.state.test_handler.parsed() if rec.get("request_id") == "abc123"]
    assert logged and logged[0]["status"] == 200


def test_not_ready_without_model(tmp_path):
    from fastapi.testclient import TestClient

    from churnlab.api import create_app
    from churnlab.logs import get_logger
    from churnlab.settings import Settings

    app = create_app(Settings(registry_dir=tmp_path / "empty"),
                     logger=get_logger("empty", handler=logging.NullHandler()))
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/ready").status_code == 503
    metrics = c.get("/metrics")  # no predictions yet: null percentiles, not a 500
    assert metrics.status_code == 200 and metrics.json()["latency_ms"]["p95"] is None


# --- client, settings, monitoring ----------------------------------------------

def test_client_does_not_retry_validation_errors(client, split):
    from churnlab.client import ChurnAPIError, ChurnClient

    calls = {"n": 0}
    original = client.request

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    client.request = counting
    try:
        with pytest.raises(ChurnAPIError) as exc:
            ChurnClient(http=client, backoff=0).predict({**payloads(split, 1)[0], "seats": 0})
    finally:
        client.request = original
    assert exc.value.status == 422 and calls["n"] == 1


def test_settings_from_env():
    from churnlab.settings import Settings

    s = Settings.from_env(environ={"CHURN_MAX_BATCH_SIZE": "10", "CHURN_LOG_LEVEL": "debug"})
    assert s.max_batch_size == 10 and s.log_level == "DEBUG"
    with pytest.raises(ValueError):
        Settings.from_env(environ={"CHURN_THRESHOLD_OVERRIDE": "2"})


def test_psi_behaviour():
    from churnlab.monitoring import categorical_psi, psi

    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(1, 1, size=5000)) > 0.25
    assert categorical_psi(["a"] * 50 + ["b"] * 50, ["a"] * 50 + ["b"] * 50) == pytest.approx(0)


def test_alert_rules():
    from churnlab.monitoring import evaluate_rules

    table = evaluate_rules({"error_rate": 0.05, "latency_p95_ms": 10, "max_psi": float("nan")})
    fired = set(table.loc[table.fires, "rule"])
    assert fired == {"error rate high"}


def test_run_check_never_raises(capsys):
    from churnlab.checks import run_check

    def boom():
        raise KeyError("x")

    assert run_check("t", boom).status == "error"
    assert run_check("t", lambda: None).passed
    out = capsys.readouterr().out
    assert "ERROR" in out and "PASS" in out


def test_colab_zip_roundtrip(tmp_path):
    from churnlab.colab import export_artifacts, restore_artifacts

    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "a.json").write_text(json.dumps({"x": 1}))
    archive = export_artifacts(src, tmp_path / "a.zip", download=False)
    out = restore_artifacts(archive, tmp_path / "out")
    assert json.loads((out / "sub" / "a.json").read_text()) == {"x": 1}
    with pytest.raises(FileExistsError):
        restore_artifacts(archive, out)
