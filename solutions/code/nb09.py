"""Reference solutions for notebook 09. Graded by tests/test_solutions.py."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from churnlab.schemas import CustomerFeatures


class DoubleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)


class DoubleResponse(BaseModel):
    result: int


def ex1() -> FastAPI:
    app = FastAPI(title="double")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/double", response_model=DoubleResponse)
    def double(body: DoubleRequest):
        return DoubleResponse(result=2 * body.x)

    return app


class CustomerFeaturesV2(CustomerFeatures):
    """v1 contract plus an optional Net Promoter Score."""

    nps_score: int | None = Field(default=None, ge=0, le=10)


def ex2():
    return CustomerFeaturesV2
