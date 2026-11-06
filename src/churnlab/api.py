"""The prediction service.

``create_app`` is an *app factory*: it builds a fresh FastAPI app from a
``Settings`` object. Tests and notebooks create apps with a temporary registry,
production creates one from environment variables (see ``serve.py`` written on
Day 4) - same code path, no global state hidden in module imports.

Endpoints
---------
GET  /health          liveness: the process is up (never touches the model)
GET  /ready           readiness: a model is loaded and can predict
GET  /model           metadata of the loaded model
POST /predict         one customer
POST /predict/batch   up to ``max_batch_size`` customers
GET  /metrics         request counts, error counts, latency percentiles
POST /admin/reload    re-read the registry pointer (used for rollback);
                      in production this would sit behind authentication.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .artifacts import load_bundle
from .features import CATEGORICAL, NUMERIC
from .logs import get_logger, log_event
from .registry import Registry
from .schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfo,
    Prediction,
)
from .settings import Settings

FEATURES = CATEGORICAL + NUMERIC


class ModelHolder:
    """Holds the live model; swapping it is one assignment under a lock."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.meta: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        try:
            registry = Registry(self.settings.registry_dir)
            version = self.settings.model_version or registry.current()
            if version is None:
                raise RuntimeError(f"no CURRENT model in {self.settings.registry_dir}")
            model, meta = load_bundle(registry.root / version)
            if meta["features"] != FEATURES:
                raise RuntimeError("model feature list does not match the API contract")
            with self._lock:
                self.model, self.meta, self.error = model, meta, None
        except Exception as exc:  # keep serving the old model; report not-ready if none
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def threshold(self) -> float:
        return self.settings.threshold_override or float(self.meta["threshold"])

    def snapshot(self):
        with self._lock:
            return self.model, self.meta


class Metrics:
    """Thread-safe in-process counters. A real service exports these to Prometheus."""

    def __init__(self, window: int = 5000):
        self._lock = threading.Lock()
        self.requests: dict[str, int] = {}
        self.errors: dict[str, int] = {}
        self.latency_ms: deque[float] = deque(maxlen=window)
        self.recent_inputs: deque[dict] = deque(maxlen=window)
        self.recent_scores: deque[float] = deque(maxlen=window)

    def record(self, path: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[path] = self.requests.get(path, 0) + 1
            if status >= 400:
                key = f"{path} {status}"
                self.errors[key] = self.errors.get(key, 0) + 1
            if path.startswith("/predict"):
                self.latency_ms.append(ms)

    def observe(self, rows: list[dict], scores) -> None:
        with self._lock:
            self.recent_inputs.extend(rows)
            self.recent_scores.extend(float(s) for s in scores)

    def summary(self) -> dict:
        with self._lock:
            lat = np.array(self.latency_ms, dtype=float)

            def pct(q: float) -> float | None:
                # None (JSON null) before the first prediction: NaN is not valid JSON,
                # and a scraper that starts early must not see the service as broken.
                return float(np.percentile(lat, q)) if lat.size else None

            return {
                "requests": dict(self.requests),
                "errors": dict(self.errors),
                "latency_ms": {"count": int(lat.size), "p50": pct(50), "p95": pct(95), "p99": pct(99)},
                "predictions_observed": len(self.recent_scores),
                "mean_score": float(np.mean(self.recent_scores)) if self.recent_scores else None,
            }


def create_app(settings: Settings | None = None, logger: logging.Logger | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logger = logger or get_logger(settings.service_name, settings.log_level)
    holder = ModelHolder(settings)
    holder.load()
    metrics = Metrics()

    app = FastAPI(title="TaskFlow churn API", version="1.0.0")
    app.state.holder = holder
    app.state.metrics = metrics
    app.state.settings = settings
    app.state.logger = logger

    if holder.ready:
        log_event(logger, "model_loaded", model_version=holder.meta["version"])
    else:
        log_event(logger, "model_load_failed", logging.ERROR, error=holder.error)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:  # unhandled -> 500, but logged with the request id
            status = 500
            logger.exception("unhandled_error", extra={"fields": {"request_id": request_id}})
            response = JSONResponse({"detail": "internal error", "request_id": request_id}, 500)
        ms = (time.perf_counter() - start) * 1000
        metrics.record(request.url.path, status, ms)
        response.headers["x-request-id"] = request_id
        response.headers["x-model-version"] = (holder.meta or {}).get("version", "none")
        log_event(logger, "request", request_id=request_id, method=request.method,
                  path=request.url.path, status=status, latency_ms=round(ms, 2),
                  model_version=(holder.meta or {}).get("version"))
        return response

    def _require_model():
        if not holder.ready:
            raise HTTPException(status_code=503, detail=f"model not loaded: {holder.error}")
        return holder.snapshot()

    def _score(customers: list[CustomerFeatures]):
        model, meta = _require_model()
        rows = [c.model_dump() for c in customers]
        frame = pd.DataFrame(rows)[FEATURES]
        # Pydantic gives None for optional fields; make them NaN for sklearn.
        frame[NUMERIC] = frame[NUMERIC].astype(float)
        scores = model.predict_proba(frame)[:, 1]
        metrics.observe([{k: r[k] for k in FEATURES} for r in rows], scores)
        threshold = holder.threshold
        preds = [
            Prediction(customer_id=c.customer_id, churn_probability=round(float(s), 6),
                       flagged=bool(s >= threshold), threshold=threshold,
                       model_version=meta["version"])
            for c, s in zip(customers, scores)
        ]
        return preds, meta

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready():
        if not holder.ready:
            return JSONResponse(HealthResponse(status="not_ready", detail=holder.error).model_dump(), 503)
        return HealthResponse(status="ready", model_version=holder.meta["version"])

    @app.get("/model", response_model=ModelInfo)
    def model_info():
        _, meta = _require_model()
        return ModelInfo(model_version=meta["version"], model_name=meta["model_name"],
                         threshold=holder.threshold, features=meta["features"],
                         trained_at=meta["trained_at"], metrics=meta["metrics"],
                         libraries=meta["libraries"])

    @app.post("/predict", response_model=Prediction)
    def predict(customer: CustomerFeatures):
        preds, _ = _score([customer])
        return preds[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        if len(batch.customers) > settings.max_batch_size:
            raise HTTPException(status_code=413,
                                detail=f"batch of {len(batch.customers)} exceeds max {settings.max_batch_size}")
        preds, meta = _score(batch.customers)
        return BatchResponse(predictions=preds, model_version=meta["version"],
                             request_id=request.state.request_id)

    @app.get("/metrics")
    def get_metrics():
        return metrics.summary()

    @app.post("/admin/reload", response_model=HealthResponse)
    def reload():
        before = (holder.meta or {}).get("version")
        holder.load()
        after = (holder.meta or {}).get("version")
        log_event(logger, "model_reload", previous=before, current=after, error=holder.error)
        if holder.error:
            raise HTTPException(status_code=500, detail=f"reload failed, still serving {before}: {holder.error}")
        return HealthResponse(status="reloaded", model_version=after)

    return app


def example_payload(row: pd.Series | dict) -> dict:
    """Turn a feature-table row into a JSON-ready request body."""
    row = dict(row)
    body = {"customer_id": str(row["customer_id"])}
    for col in CATEGORICAL:
        body[col] = str(row[col])
    for col in NUMERIC:
        value = row[col]
        body[col] = None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)
    return body
