"""Prediction service: an app factory with /health, /ready and /predict.

Run:  uvicorn capstone.api:create_app --factory --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .features import FEATURES
from .models import load_bundle, predict_frame

MAX_BATCH = 1000


class Instance(BaseModel):
    """TODO: replace with your request schema. Validate ranges here, not in the model."""

    model_config = ConfigDict(extra="forbid")   # unknown fields are an error, not ignored
    id: str = Field(min_length=1, max_length=64)
    x1: float = Field(ge=-1e6, le=1e6)
    x2: float = Field(ge=-1e6, le=1e6)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Instance] = Field(min_length=1, max_length=MAX_BATCH)


class PredictionOut(BaseModel):
    id: str
    prediction: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[PredictionOut]


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    """Build the app. `model_dir` defaults to $MODEL_DIR or artifacts/current."""
    model_dir = Path(model_dir or os.environ.get("MODEL_DIR", "artifacts/current"))
    app = FastAPI(title="capstone API", version="0.1.0")   # TODO: rename
    state: dict = {"model": None, "meta": None, "error": None}
    try:
        state["model"], state["meta"] = load_bundle(model_dir)
        if state["meta"]["features"] != FEATURES:
            raise RuntimeError("model features do not match the API contract")
    except Exception as exc:  # serve /health anyway; /ready reports the reason
        state["model"], state["error"] = None, f"{type(exc).__name__}: {exc}"
    app.state.model_state = state

    @app.get("/health")
    def health():
        """Liveness: the process answers. Never touches the model."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        """Readiness: a model is loaded and can predict."""
        if state["model"] is None:
            return JSONResponse({"status": "not_ready", "detail": state["error"]}, status_code=503)
        return {"status": "ready", "model_version": state["meta"]["version"]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        frame = pd.DataFrame([i.model_dump() for i in req.instances])
        # TODO: build the same features as training - import, don't copy, the code.
        preds = predict_frame(state["model"], frame[FEATURES])
        return PredictResponse(
            model_version=state["meta"]["version"],
            predictions=[PredictionOut(id=i.id, prediction=round(float(p), 6))
                         for i, p in zip(req.instances, preds, strict=True)],
        )

    return app
