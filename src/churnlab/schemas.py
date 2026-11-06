"""The API contract, written as Pydantic models.

The contract is the promise the service makes to its callers. Writing it as
types gives you three things for free: validation of every request (with a
readable 422 error instead of a stack trace), an OpenAPI document at
``/docs``, and a single place to review when the model's inputs change.

Design choices worth defending in an interview:

* ``extra="forbid"`` - a misspelled field ("sesions_4w") is an error, not a
  silently-imputed missing value.
* Bounds on numbers - a negative session count is rejected at the door.
* Categorical fields are plain strings, *not* enums: an unseen region should
  reach the model (which handles unknown categories) rather than fail the
  request; it is counted by monitoring instead. That is a deliberate trade-off.
* ``features_used_4w`` and ``avg_resolution_hours_90d`` are optional because
  they are legitimately missing (no usage, no tickets) - the pipeline imputes.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(min_length=1, max_length=64)
    plan: str = Field(examples=["pro"])
    region: str = Field(examples=["EU"])
    acquisition_channel: str = Field(examples=["organic"])
    company_size: str = Field(examples=["11-50"])
    seats: float = Field(ge=1, le=10_000)
    tenure_days: float = Field(ge=0, le=36_500)
    mrr_usd: float = Field(ge=0)
    sessions_4w: float = Field(ge=0)
    sessions_prev_4w: float = Field(ge=0)
    sessions_trend: float = Field(ge=0)
    seat_utilization_4w: float = Field(ge=0, le=1.5)
    features_used_4w: float | None = Field(default=None, ge=0, le=12)
    weeks_since_last_active: float = Field(ge=0)
    tickets_30d: float = Field(ge=0)
    tickets_90d: float = Field(ge=0)
    billing_tickets_90d: float = Field(ge=0)
    cancellation_tickets_30d: float = Field(ge=0)
    avg_resolution_hours_90d: float | None = Field(default=None, ge=0)
    failed_payments_90d: float = Field(ge=0)


class Prediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    flagged: bool
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(min_length=1)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    model_name: str
    threshold: float
    features: list[str]
    trained_at: str
    metrics: dict
    libraries: dict
