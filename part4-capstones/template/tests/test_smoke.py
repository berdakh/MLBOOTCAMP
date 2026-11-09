"""Smoke tests: the pipeline runs end to end on toy data, with no network.

Keep these fast (seconds). Add dataset-specific tests next to them:
a leakage test for your split, a test that features only use the past,
and an API test for each validation rule.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from capstone.api import create_app
from capstone.config import load_config
from capstone.data import group_split, make_toy_data, time_split, validate
from capstone.evaluate import bootstrap_ci, evaluate, mae, wape
from capstone.features import FEATURES, build_features
from capstone.models import MeanBaseline, Metadata, build_model, load_bundle, save_bundle


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def toy():
    return validate(make_toy_data())


def test_config_loads(config):
    assert {"project", "data", "split", "model", "paths"} <= set(config)


def test_splits_do_not_leak(toy):
    train, test = time_split(toy, "2019-12-01")
    assert train["timestamp"].max() < test["timestamp"].min()
    gtrain, gtest = group_split(toy, test_fraction=0.25, seed=1)
    assert not set(gtrain["entity_id"]) & set(gtest["entity_id"])


def test_validate_rejects_duplicates(toy):
    with pytest.raises(ValueError, match="duplicate"):
        validate(toy.iloc[[0, 0]])


def test_model_beats_baseline_and_roundtrips(toy, config, tmp_path):
    train, test = time_split(toy, "2019-12-01")
    Xtr, Xte = build_features(train), build_features(test)
    base = MeanBaseline().fit(Xtr, train["target"])
    model = build_model(config).fit(Xtr, train["target"])
    assert mae(test["target"], model.predict(Xte)) < mae(test["target"], base.predict(Xte))
    save_bundle(model, Metadata(name="toy", version="v1", features=FEATURES), tmp_path / "v1")
    loaded, meta = load_bundle(tmp_path / "v1")
    assert meta["version"] == "v1"
    assert (loaded.predict(Xte) == model.predict(Xte)).all()


def test_metrics():
    assert wape([0, 2], [1, 1]) == 1.0
    lo, hi = bootstrap_ci([1, 2, 3, 4], [1, 2, 2, 5], n=50)
    assert lo <= hi
    assert set(evaluate([1, 2], [1, 3])) >= {"mae", "rmse", "wape", "n"}


def test_api_health_ready_predict(toy, config, tmp_path):
    model = build_model(config).fit(build_features(toy), toy["target"])
    save_bundle(model, Metadata(name="toy", version="v1", features=FEATURES), tmp_path / "v1")
    client = TestClient(create_app(tmp_path / "v1"))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_version"] == "v1"
    ok = client.post("/predict", json={"instances": [{"id": "a", "x1": 0.1, "x2": -0.2}]})
    assert ok.status_code == 200
    assert ok.json()["predictions"][0]["id"] == "a"
    # validation: unknown fields and empty batches are rejected
    assert client.post("/predict", json={"instances": [{"id": "a", "x1": 0, "x2": 0, "z": 1}]}).status_code == 422
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_api_not_ready_without_model(tmp_path):
    client = TestClient(create_app(tmp_path / "missing"))
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", json={"instances": [{"id": "a", "x1": 0, "x2": 0}]}).status_code == 503
