# Solutions · Chapter 10 · FastAPI & contracts

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `make_app`

[verified] Reference code: [`code/nb09.py`](solutions.py) (`ex1`), graded in CI.

```python
class DoubleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: int = Field(ge=0)


class DoubleResponse(BaseModel):
    result: int


def make_app() -> FastAPI:
    app = FastAPI(title="double")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/double", response_model=DoubleResponse)
    def double(body: DoubleRequest):
        return DoubleResponse(result=2 * body.x)

    return app
```

**Why it works.** The request schema carries every rule: `int` rejects
`"three"`, `ge=0` rejects `-1`, a field without a default is required, and
`extra="forbid"` rejects `{"x": 2, "y": 1}` and the misspelled `{"X": 2}`.
FastAPI turns each `ValidationError` into a 422 with `loc`/`type`/`msg`.
Because `FastAPI()` is created *inside* the function, every call returns an
independent app, which is what the grader's `make_app() is not app` check
looks for.

**Common mistakes.**

* `def double(x: int)`: without a Pydantic model FastAPI treats `x` as a
  **query** parameter (`/double?x=3`), so a JSON body gives a 422 for
  "missing query parameter".
* Forgetting `extra="forbid"`: `{"x": 2, "y": 1}` returns 200.
* Validating by hand inside the endpoint (`if x < 0: raise HTTPException(400)`):
  that works, but it gives a different status code, documents nothing in
  OpenAPI, and has to be repeated in every endpoint.
* Creating the app at module level and returning the same object: two tests
  would then share state.

---

## Exercise 2 · `make_schema_v2`

[verified] Reference code: [`code/nb09.py`](solutions.py) (`ex2`).

```python
from churnlab.schemas import CustomerFeatures


class CustomerFeaturesV2(CustomerFeatures):
    nps_score: int | None = Field(default=None, ge=0, le=10)


def make_schema_v2():
    return CustomerFeaturesV2
```

**Why it works.** A Pydantic subclass inherits all fields, their constraints
and `model_config` (so `extra="forbid"` still applies). The new field has a
default, so a v1 payload still validates: this is a backwards-compatible
change. `int` rejects `7.5`, and `ge`/`le` reject `-1` and `11`.

**Common mistakes.**

* `nps_score: int = Field(ge=0, le=10)` with no default makes the field
  **required**, which breaks every existing client.
* Copy-pasting the v1 schema into a new class instead of subclassing: it works
  today, and the two drift apart the next time a v1 bound changes.
* Returning an instance (`CustomerFeaturesV2(...)`) instead of the class.

**Push further.** The model does not use NPS yet. The contract change can ship
first (callers start sending it, monitoring records its distribution), and a
model trained with NPS later. Contract and model are versioned separately.

---

## Exercise 3 · Which contract changes break clients? (written)

[open-ended] Open-ended. Assume clients are written against today's schema.

| # | Change | Breaking? | Why |
|---|---|---|---|
| 1 | Add a **required** request field `contract_end_date` | **Yes** | every existing request lacks it → 422 for every caller. Make it optional with a sensible default, or ship it under `/v2` |
| 2 | Add `explanation` to the `/predict` **response** | **No** (usually) | well-behaved clients ignore unknown response keys. It breaks only clients that validate responses strictly (e.g. a generated client with "forbid extra"), so say so in the changelog |
| 3 | Rename `churn_probability` → `score` | **Yes** | every client reading `churn_probability` gets a `KeyError` or `None`. Add `score` alongside the old name, deprecate it, remove it in the next major version |
| 4 | Tighten `seats` from `le=10000` to `le=5000` | **Yes** | requests that were valid (6,000 seats) now get 422. Tightening validation is breaking; relaxing it is not |
| 5 | `region` from free string to an enum | **Yes** | any value outside the enum (a new market, `"unknown"`, a lowercase spelling) that used to get a prediction now gets a 422. It also changes the *policy*: unseen regions are refused instead of counted by monitoring |

A strong answer also says **what to do** for the breaking ones: version the
API (`/v1` → `/v2`, or a version header), run both versions for a deprecation
period, and measure who still calls the old one before removing it.

---

## Checkpoint answers

**1. Misspelled field.** `"sesions_4w": 3` returns **422** with two errors:
`sesions_4w` → `extra_forbidden`, `sessions_4w` → `missing` ([verified] observed in
section 9.7). Without `extra="forbid"`, the typo is silently dropped; for a
required field you still get a 422 (missing), but for an *optional* field
(e.g. `features_used_4w` misspelled) the request succeeds and the pipeline
imputes the training median. That is a wrong prediction with no error anywhere.

**2. Reading the error.** "In the request body, `seat_utilization_4w` is
above its maximum." The caller should send a value ≤ 1.5 (utilisation is a
fraction; they probably sent a percentage, e.g. 60 instead of 0.6).

**3. Strings, not enums.** So that a new or unexpected category still gets a
prediction (the one-hot encoder uses `handle_unknown="ignore"`) and is
*counted by monitoring* instead of rejected. Choose an enum when an unknown
value makes the prediction meaningless or dangerous, when the set is truly
closed (e.g. `currency` for a payments model), or when the caller is internal
and you want typos to fail loudly.

**4. App factory.** A function `create_app(settings)` that builds a new app
and everything it needs (model holder, metrics, logger) from configuration.
Loading at import time means importing the module (in a test, in a docs
build) touches disk at a hard-coded path, tests cannot use different
configurations in one process, and a missing model crashes the import instead
of returning 503 from `/ready`.

**5. Health vs ready; 413/422/503.** `/health`: the process is alive (never
touches the model) and is used for restarts. `/ready`: a model is loaded and
predictions will work; it is used to route traffic. **413**: batch too large;
split it. **422**: the request is invalid; fix it and do not retry. **503**:
the service is temporarily unable (no model, overloaded); retry later or
elsewhere.

**6. Compatible vs breaking.** Compatible: adding an optional request field,
adding a response field, relaxing a bound, adding an endpoint. Breaking:
adding a required field, renaming/removing a field, tightening a bound,
changing a type or a unit/meaning.
