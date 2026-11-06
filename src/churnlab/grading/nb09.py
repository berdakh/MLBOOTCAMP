"""Self-checks for notebook 09 · FastAPI & contracts."""

from __future__ import annotations

from ..checks import run_check

#: A request body that satisfies churnlab.schemas.CustomerFeatures.
VALID_CUSTOMER = {
    "customer_id": "C00042",
    "plan": "pro",
    "region": "EU",
    "acquisition_channel": "organic",
    "company_size": "11-50",
    "seats": 12.0,
    "tenure_days": 400.0,
    "mrr_usd": 480.0,
    "sessions_4w": 35.0,
    "sessions_prev_4w": 40.0,
    "sessions_trend": 0.88,
    "seat_utilization_4w": 0.6,
    "features_used_4w": 5.5,
    "weeks_since_last_active": 0.0,
    "tickets_30d": 1.0,
    "tickets_90d": 2.0,
    "billing_tickets_90d": 0.0,
    "cancellation_tickets_30d": 0.0,
    "avg_resolution_hours_90d": 6.5,
    "failed_payments_90d": 0.0,
}


def ex1(make_app):
    def test():
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = make_app()
        assert isinstance(app, FastAPI), f"make_app() should return a FastAPI app, got {type(app).__name__}"
        assert make_app() is not app, "make_app() must build a NEW app on every call (a factory)"
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200, f"GET /health returned {r.status_code}"
        assert r.json() == {"status": "ok"}, f"GET /health returned {r.json()}"
        for x in (0, 3, 21):
            r = client.post("/double", json={"x": x})
            assert r.status_code == 200, f"POST /double {{'x': {x}}} returned {r.status_code}: {r.text}"
            assert r.json().get("result") == 2 * x, f"POST /double {{'x': {x}}} returned {r.json()}"
        bad = {
            "a negative x": {"x": -1},
            "a missing x": {},
            "a non-integer string": {"x": "three"},
            "an extra field": {"x": 2, "y": 1},
            "a misspelled field": {"X": 2},
        }
        for what, body in bad.items():
            r = client.post("/double", json=body)
            assert r.status_code == 422, f"{what} ({body}) should give 422, got {r.status_code}"

    return run_check("Exercise 1 · make_app", test,
                     hint="class DoubleIn(BaseModel): model_config = ConfigDict(extra='forbid'); "
                          "x: int = Field(ge=0) - and create FastAPI() INSIDE make_app")


def ex2(make_schema_v2):
    def test():
        from pydantic import BaseModel, ValidationError

        from ..schemas import CustomerFeatures

        model = make_schema_v2()
        assert isinstance(model, type) and issubclass(model, BaseModel), \
            "make_schema_v2() should return a Pydantic model CLASS (not an instance)"
        missing = set(CustomerFeatures.model_fields) - set(model.model_fields)
        assert not missing, f"v2 lost existing fields: {sorted(missing)}"
        assert "nps_score" in model.model_fields, "v2 has no field called nps_score"

        old = model(**VALID_CUSTOMER)
        assert getattr(old, "nps_score", "absent") is None, \
            "a v1 payload (no nps_score) must still validate, with nps_score defaulting to None"
        for score in (0, 7, 10):
            m = model(**VALID_CUSTOMER, nps_score=score)
            assert m.nps_score == score, f"nps_score={score} was stored as {m.nps_score!r}"
        assert model(**VALID_CUSTOMER, nps_score=None).nps_score is None, "an explicit null must be allowed"

        def rejects(body, what):
            try:
                model(**body)
            except ValidationError:
                return
            raise AssertionError(f"v2 accepted {what}; it must be rejected")

        rejects({**VALID_CUSTOMER, "nps_score": 11}, "nps_score=11 (above 10)")
        rejects({**VALID_CUSTOMER, "nps_score": -1}, "nps_score=-1 (below 0)")
        rejects({**VALID_CUSTOMER, "nps_score": 7.5}, "nps_score=7.5 (not an integer)")
        rejects({**VALID_CUSTOMER, "nps": 7}, "an unknown field 'nps' (extras must stay forbidden)")
        rejects({**VALID_CUSTOMER, "sessions_4w": -1}, "sessions_4w=-1 (existing bounds must still apply)")
        rejects({k: v for k, v in VALID_CUSTOMER.items() if k != "plan"}, "a payload without 'plan'")

    return run_check("Exercise 2 · make_schema_v2", test,
                     hint="class CustomerFeaturesV2(CustomerFeatures): "
                          "nps_score: int | None = Field(default=None, ge=0, le=10)")
