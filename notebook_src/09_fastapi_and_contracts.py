# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/09_fastapi_and_contracts.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 09 · FastAPI & contracts
#
# **Day 3 · ~55 minutes · CPU only**
#
# A model in a notebook helps one person; a model behind an API helps every
# system that can send an HTTP request. The hard part is not FastAPI's syntax.
# It is the **contract**: what callers may send, what they get back, and what
# happens when they send something wrong.
#
# By the end you will be able to:
#
# 1. build a small FastAPI app with `GET /health` and `POST /predict`, and test it in-process without starting a server;
# 2. read a 422 validation error (`loc`, `msg`, `type`) and explain why it beats a 500;
# 3. use Pydantic to forbid unknown fields, bound numbers, allow optional fields and filter responses;
# 4. explain the app-factory pattern and why there is no model loaded at import time;
# 5. call every endpoint of the course's real service (`churnlab.api`) and read its OpenAPI document;
# 6. say which contract changes break existing clients and which do not.
#
# **You will produce:** `artifacts/api/openapi.json`, the generated contract of the churn service.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: a registry with trained models (notebook 08) -----------------
# Builds the feature table, the time split and two registered models if they
# are not already in ART (about 20-30 seconds the first time).
from churnlab import bootstrap
from churnlab.grading import nb09 as grade

registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()

# %% [markdown]
# **Expected:** `versions ['v1', 'v2'], CURRENT=v1` (v1 is the logistic
# regression, which won on validation PR-AUC in notebook 06).

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Endpoint** | an HTTP method plus a path the service answers | `POST /predict` |
# | **FastAPI** | a Python web framework that builds request handling and docs from type hints | `@app.post("/predict")` |
# | **Pydantic model** | a class that declares field types and limits and validates data against them | `CustomerFeatures` |
# | **Contract** | request and response schemas plus the meaning of every status code | 422 = your request is wrong |
# | **422 Unprocessable Content** | the body parsed but broke the schema | `tenure_days = -5` |
# | **413 Content Too Large** | the request is bigger than the service accepts | a batch of 2,500 rows |
# | **503 Service Unavailable** | the service cannot answer right now | no model loaded yet |
# | **`extra="forbid"`** | reject fields the schema does not declare | a typo such as `sesions_4w` |
# | **`response_model`** | the schema the response is filtered and validated against | internal fields never leak out |
# | **App factory** | a function that builds a fresh, configured app | `create_app(settings)` |
# | **`TestClient`** | calls the app in-process with real HTTP semantics, no server | `client.post("/predict", json=body)` |
# | **OpenAPI** | the machine-readable description of the contract | `/openapi.json` |

# %% [markdown]
# ## 9.1 · What a prediction service is
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/serving.svg" alt="Serving: a request is validated against the schema, scored by the pipeline loaded from the registry, and returned with the model version" width="100%">
#
# Read the diagram left to right. A request arrives as JSON. Before any model
# code runs, it is **validated** against a schema; invalid requests are
# answered with a 422 and never reach the model. Valid requests are turned
# into a one-row DataFrame and scored by the *same* `Pipeline` object that was
# trained and saved in the registry. The response carries the score, the
# decision, the threshold and the **model version**.
#
# Three vocabulary items you will use all day:
#
# | Term | Meaning here |
# |---|---|
# | **Endpoint** | a method + path, e.g. `POST /predict` |
# | **Contract** | the request and response schemas plus the meaning of each status code |
# | **In-process testing** | calling the app as a Python object (`TestClient`): real HTTP semantics, no socket, no server |
#
# Each request takes one path through the service, and every rejection
# happens before any model code runs:
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l1-request-lifecycle.svg" alt="A prediction request is validated (422 on failure), checked for batch size (413), checked for a loaded model (503), then scored and answered with 200 and the model version" width="100%">
#
# *What to notice:* the three exits are ordered from "your fault" to "our fault, try later"; only the last one is worth retrying (notebook 11).
#
# > 💡 **Intuition.** The schema is a bouncer with a guest list at the door:
# > a wrong name never reaches the bar (the model), and the bouncer tells you
# > exactly what was wrong with your ticket. *Where it breaks down:* the
# > bouncer checks the form, not the truth: a well-formed request with
# > plausible but wrong values (last month's data) walks straight in; that is
# > monitoring's job (notebook 14).

# %% [markdown]
# ## 9.2 · A minimal FastAPI app, from scratch
#
# Before reading the real service, build a toy one. The "model" is a hand-made
# rule: fewer sessions and more failed payments mean higher risk. The point is
# the shape of the app, not the model.

# %%
import math

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel


class ToyCustomer(BaseModel):
    customer_id: str
    sessions_4w: float
    failed_payments_90d: float


def toy_score(c: ToyCustomer) -> float:
    z = -1.0 - 0.05 * c.sessions_4w + 0.8 * c.failed_payments_90d
    return 1 / (1 + math.exp(-z))


toy = FastAPI(title="toy churn API")


@toy.get("/health")
def toy_health():
    return {"status": "ok"}


@toy.post("/predict")
def toy_predict(customer: ToyCustomer):
    p = toy_score(customer)
    return {"customer_id": customer.customer_id, "churn_probability": round(p, 4), "flagged": p >= 0.3}


client = TestClient(toy)
print(client.get("/health").json())
r = client.post("/predict", json={"customer_id": "c1", "sessions_4w": 2, "failed_payments_90d": 2})
print(r.status_code, r.json())

# %% [markdown]
# **Expected:** `{'status': 'ok'}`, then `200` and a probability of about
# 0.62 with `flagged: True`.
#
# Three things happened that you did not write:
#
# * FastAPI read the type annotation `customer: ToyCustomer` and parsed the
#   JSON body into that Pydantic model, converting `2` to `2.0`.
# * The returned dict was serialised to JSON.
# * `TestClient` sent a real HTTP request (method, headers, body) through the
#   app's ASGI interface. No port was opened. This is how API tests run in CI.

# %% [markdown]
# ## 9.3 · Anatomy of a 422, and why it beats a 500
#
# Send three wrong requests and read what comes back.

# %%
import json

bad_requests = {
    "string for a number": {"customer_id": "c1", "sessions_4w": "lots", "failed_payments_90d": 0},
    "missing field": {"customer_id": "c1", "sessions_4w": 3},
    "null for a required number": {"customer_id": "c1", "sessions_4w": None, "failed_payments_90d": 0},
}
for name, body in bad_requests.items():
    r = client.post("/predict", json=body)
    print(f"{name:<28} -> {r.status_code}")
    for err in r.json()["detail"]:
        print("    loc:", err["loc"], "| type:", err["type"], "| msg:", err["msg"])

# %% [markdown]
# **Expected:** three 422 responses. Each error names
#
# * `loc` - **where**: `["body", "sessions_4w"]` (the part of the request, then the field path);
# * `type` - **which rule** failed, machine-readable (`float_parsing`, `missing`, `float_type`);
# * `msg` - a human-readable sentence.
#
# Now the same mistake against an endpoint that skips validation and reads a
# raw `dict`:

# %%
unvalidated = FastAPI()


@unvalidated.post("/predict")
def predict_raw(payload: dict):
    z = -1.0 - 0.05 * float(payload["sessions_4w"]) + 0.8 * float(payload["failed_payments_90d"])
    return {"churn_probability": 1 / (1 + math.exp(-z))}


raw_client = TestClient(unvalidated, raise_server_exceptions=False)
r = raw_client.post("/predict", json={"customer_id": "c1", "sessions_4w": 3})
print(r.status_code, r.text)

# %% [markdown]
# **Expected:** `500 Internal Server Error`, with no detail.
#
# | | 422 from the schema | 500 from a crash |
# |---|---|---|
# | Who is at fault? | the caller, and the response says so | unclear - looks like *our* bug |
# | Can the caller fix it? | yes: `loc` says which field | no information |
# | Should a client retry? | **never** - it will fail identically | maybe, which multiplies load |
# | Does it page the on-call engineer? | no (it is a 4xx) | often yes (5xx rate alerts) |
#
# > 💡 **Intuition.** A 422 is a returned letter stamped "street name
# > missing"; a 500 is a letter that vanished in the sorting office. The first
# > tells the sender what to fix; the second makes everyone investigate the
# > post office. *Where it breaks down:* a 4xx can still be the service's
# > fault, if the contract changed without telling the callers (section 9.9).
#
# A 500 for bad input is a bug in the service even when the input is wrong.

# %% [markdown]
# ## 9.4 · Making the schema strict: forbid, bounds, optional, examples
#
# The toy schema has a quiet problem. Send a **misspelled** field:

# %%
r = client.post("/predict", json={"customer_id": "c1", "sesions_4w": 2,
                                  "sessions_4w": 40, "failed_payments_90d": 0})
print(r.status_code, r.json())

# %% [markdown]
# **Expected:** `200`. Pydantic's default is to *ignore* unknown fields. Here
# the typo was harmless because `sessions_4w` was also sent; if the caller
# had sent only the misspelled one *and* the field had a default (as optional
# fields do), the model would silently have scored an imputed value. No error,
# a wrong prediction, and nobody notices.
#
# A stricter schema, with each Pydantic feature you will see in the real one:

# %%
from pydantic import ConfigDict, Field, ValidationError


class StrictCustomer(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown fields are errors

    customer_id: str = Field(min_length=1, max_length=64)
    sessions_4w: float = Field(ge=0, examples=[12.0])  # bounds: >= 0
    failed_payments_90d: float = Field(ge=0, le=20)
    features_used_4w: float | None = Field(default=None, ge=0, le=12)  # optional: may be null/absent


checks = {
    "valid, optional omitted": {"customer_id": "c1", "sessions_4w": 3, "failed_payments_90d": 0},
    "misspelled field": {"customer_id": "c1", "sesions_4w": 3, "sessions_4w": 3, "failed_payments_90d": 0},
    "negative sessions": {"customer_id": "c1", "sessions_4w": -1, "failed_payments_90d": 0},
    "optional out of range": {"customer_id": "c1", "sessions_4w": 3, "failed_payments_90d": 0,
                              "features_used_4w": 30},
    "empty customer_id": {"customer_id": "", "sessions_4w": 3, "failed_payments_90d": 0},
}
for name, body in checks.items():
    try:
        m = StrictCustomer(**body)
        print(f"{name:<26} OK   features_used_4w={m.features_used_4w}")
    except ValidationError as exc:
        e = exc.errors()[0]
        print(f"{name:<26} 422  loc={e['loc']} type={e['type']}")

# %% [markdown]
# **Expected:** the first is accepted with `features_used_4w=None`; the other
# four are rejected with types `extra_forbidden`, `greater_than_equal`,
# `less_than_equal` and `string_too_short`. (Constructing the model directly
# raises `ValidationError`; inside FastAPI the same errors become the 422 body.)
#
# `Field(examples=[...])` changes no behaviour. It goes into the generated
# documentation, so a caller reading `/docs` sees a realistic value.
#
# **Optional is a contract decision, not a convenience.** In the real schema,
# `features_used_4w` and `avg_resolution_hours_90d` are optional because they
# are *legitimately* missing (no usage weeks, no tickets in 90 days) and the
# pipeline was trained with those NaNs and imputes them. Making
# `sessions_4w` optional would invite callers to omit a value they actually
# have.

# %% [markdown]
# ## 9.5 · `response_model`: the response is part of the contract too
#
# Internal values have a way of leaking into responses: a debug field, a raw
# feature vector, a customer's e-mail. Declaring `response_model` makes
# FastAPI **filter** the returned object to exactly the declared fields, and
# validate it.

# %%
class ToyPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    flagged: bool


leaky = FastAPI()


@leaky.post("/predict_unfiltered")
def predict_unfiltered(customer: StrictCustomer):
    p = toy_score(ToyCustomer(**customer.model_dump(exclude={"features_used_4w"})))
    return {"customer_id": customer.customer_id, "churn_probability": p, "flagged": p >= 0.3,
            "debug_logit": math.log(p / (1 - p)), "owner_email": "someone@example.com"}


@leaky.post("/predict", response_model=ToyPrediction)
def predict_filtered(customer: StrictCustomer):
    return predict_unfiltered(customer)


leaky_client = TestClient(leaky)
body = {"customer_id": "c1", "sessions_4w": 3, "failed_payments_90d": 1}
print("without response_model:", sorted(leaky_client.post("/predict_unfiltered", json=body).json()))
print("with response_model:   ", sorted(leaky_client.post("/predict", json=body).json()))

# %% [markdown]
# **Expected:** the unfiltered endpoint returns five keys including
# `owner_email`; the filtered one returns only `churn_probability`,
# `customer_id`, `flagged`.

# %% [markdown]
# ## 9.6 · The app factory: no globals at import time
#
# The toy apps above were built at module level: `toy = FastAPI()` runs when
# the file is imported. Real services usually load a model, read settings and
# open connections. If that happens at import time:
#
# * importing the module in a test loads a model from a hard-coded path;
# * two tests cannot use two different configurations in one process;
# * a missing model file crashes the import instead of producing a "not ready" answer.
#
# The fix is an **app factory**: a function that takes configuration and
# returns a fresh app. Everything the app needs is created *inside* it.
#
# > 💡 **Intuition.** A module-level app is a restaurant that cooks its one
# > dish the moment you walk past the building; a factory is a kitchen that
# > cooks when you order, to your order. *Where it breaks down:* in
# > production there is usually exactly one app per process, so the factory's
# > benefit shows up mostly in tests and in start-up error handling.

# %%
def make_toy_app(threshold: float = 0.3) -> FastAPI:
    app = FastAPI()
    app.state.threshold = threshold  # configuration lives on the app, not in a global

    @app.post("/predict", response_model=ToyPrediction)
    def predict(customer: ToyCustomer):
        p = toy_score(customer)
        return ToyPrediction(customer_id=customer.customer_id, churn_probability=p,
                             flagged=p >= app.state.threshold)

    return app


body = {"customer_id": "c1", "sessions_4w": 10, "failed_payments_90d": 1}
for t in (0.3, 0.5):
    print(f"threshold={t}:", TestClient(make_toy_app(t)).post("/predict", json=body).json())

# %% [markdown]
# **Expected:** the same probability (about 0.33) flagged under threshold 0.3
# and not flagged under 0.5. Two independent apps in one process, each with
# its own configuration. That is exactly what tests need: one app per test,
# with a temporary registry.
#
# > 🧠 **Research bridge.** You already do this with models: `make_model(cfg)`
# > returns a fresh network instead of a global `model` that every experiment
# > mutates. An app factory is the same idea applied to the service.

# %% [markdown]
# > **Exercise 1.** Write `make_app()` that returns a **new** FastAPI app with
# > two endpoints:
# >
# > * `GET /health` → `{"status": "ok"}`;
# > * `POST /double` accepting a JSON body `{"x": <int ≥ 0>}` and returning
# >   `{"result": 2 * x}`.
# >
# > A negative `x`, a missing `x`, a non-integer string, or any extra field
# > must give a 422. The grader calls your factory and tests it with
# > `TestClient`.

# %%
def make_app() -> FastAPI:
    # YOUR CODE HERE: define the request schema (extra="forbid", x: int >= 0),
    # create the app, register the two endpoints, return the app.
    raise NotImplementedError


grade.ex1(make_app)

# %% [markdown]
# ## 9.7 · The real service: `churnlab.api.create_app`
#
# Now read the real thing. Open `src/churnlab/schemas.py` (the contract) and
# `src/churnlab/api.py` (the app factory) side by side; both are short.
# Everything in sections 9.2-9.6 appears there:
#
# | Idea | Where in the real code |
# |---|---|
# | strict schema | `CustomerFeatures`: `extra="forbid"`, bounds, two optional fields |
# | response filtering | `response_model=Prediction` / `BatchResponse` / `ModelInfo` |
# | app factory | `create_app(settings)` builds a `ModelHolder`, `Metrics`, the endpoints |
# | no import-time model | the model is loaded inside `create_app`, from `settings.registry_dir` |
#
# The service logs one JSON line per request. Here the logs go to an
# in-memory handler so they do not flood the notebook; notebook 13 reads them.

# %%
from churnlab.api import create_app, example_payload
from churnlab.logs import ListHandler, get_logger
from churnlab.settings import Settings

log_handler = ListHandler()
settings = Settings(registry_dir=registry.root)
app = create_app(settings, logger=get_logger("churn-api-nb09", handler=log_handler))
api = TestClient(app)

for path in ("/health", "/ready"):
    r = api.get(path)
    print(path, r.status_code, r.json())

info = api.get("/model").json()
print({k: info[k] for k in ("model_version", "model_name", "threshold", "trained_at")})
print("validation PR-AUC:", round(info["metrics"]["validation"]["pr_auc"], 3))

# %% [markdown]
# **Expected:** `/health` 200 `{'status': 'ok', ...}`; `/ready` 200 with
# `model_version: 'v1'`; `/model` reports `logreg`, a threshold near 0.19 and
# a validation PR-AUC near 0.163.
#
# `/health` answers "is the process alive?" and never touches the model.
# `/ready` answers "can it serve predictions?". Load balancers and Kubernetes
# use them differently (notebook 13).
#
# Predictions, using rows from the **validation** period. The test period stays
# untouched until notebook 15. `example_payload` turns a feature-table row into
# a JSON-ready body (NaN becomes `null`).

# %%
valid = split.valid
one = example_payload(valid.iloc[0])
print(json.dumps(one, indent=1)[:420], "...")

r = api.post("/predict", json=one)
print(r.status_code, r.json())
print("headers:", {k: r.headers[k] for k in ("x-model-version", "x-request-id")})

# %% [markdown]
# **Expected:** a body with 20 fields; `200` and a prediction with
# `churn_probability` between 0 and 1, `flagged`, `threshold` and
# `model_version: 'v1'`. The same version is in the `x-model-version` header,
# and each response carries an `x-request-id` (notebook 10 uses both).

# %%
batch = [example_payload(row) for _, row in valid.head(25).iterrows()]
r = api.post("/predict/batch", json={"customers": batch})
out = r.json()
print(r.status_code, "| predictions:", len(out["predictions"]), "| version:", out["model_version"],
      "| request_id:", out["request_id"])
print("flagged in this batch:", sum(p["flagged"] for p in out["predictions"]))

too_big = Settings(registry_dir=registry.root, max_batch_size=10)
small_api = TestClient(create_app(too_big, logger=get_logger("churn-api-nb09", handler=log_handler)))
r = small_api.post("/predict/batch", json={"customers": batch})
print("batch of 25 with max_batch_size=10 ->", r.status_code, r.json())

# %% [markdown]
# **Expected:** `200 | predictions: 25 | version: v1`, one of them flagged; then a
# **413** (`Content Too Large`) with a message naming the limit. A batch limit
# protects memory and latency; the client in notebook 11 splits big jobs into
# chunks below it.
#
# The real schema rejects the same kinds of mistakes you saw in 9.4:

# %%
mistakes = {
    "misspelled field": {**{k: v for k, v in one.items() if k != "sessions_4w"}, "sesions_4w": 3.0},
    "negative sessions": {**one, "sessions_4w": -5},
    "seat utilisation 300%": {**one, "seat_utilization_4w": 3.0},
    "missing plan": {k: v for k, v in one.items() if k != "plan"},
}
for name, body in mistakes.items():
    r = api.post("/predict", json=body)
    print(f"{name:<22} {r.status_code}", [(e["loc"][-1], e["type"]) for e in r.json()["detail"]])

print(api.get("/metrics").json())

# %% [markdown]
# **Expected:** four 422s. The misspelled one reports *two* errors:
# `sesions_4w` is `extra_forbidden` and `sessions_4w` is `missing`. Then
# `/metrics` shows request counts per path, the 413 and 422 errors, and
# latency percentiles. (`/metrics` is per app: the 413 went to `small_api`, so
# it is not counted here.)

# %% [markdown]
# ## 9.8 · The OpenAPI document: the contract, machine-readable
#
# FastAPI generates an OpenAPI (Swagger) document from the schemas. It is what
# `/docs` renders, and what other teams generate clients from. It is also a
# file you can commit and **diff** in code review: a changed contract shows up
# as a changed document.

# %%
spec = api.get("/openapi.json").json()
for path, methods in spec["paths"].items():
    print(f"{path:<16}", ", ".join(m.upper() for m in methods))

schema = spec["components"]["schemas"]["CustomerFeatures"]
optional = sorted(set(schema["properties"]) - set(schema["required"]))
print("\nrequired fields:", len(schema["required"]), "| optional:", optional)
print("additionalProperties:", schema.get("additionalProperties"))
print("sessions_4w:", schema["properties"]["sessions_4w"])

(ART / "api").mkdir(exist_ok=True)
_ = (ART / "api" / "openapi.json").write_text(json.dumps(spec, indent=2))

# %% [markdown]
# **Expected:** seven paths (`/health`, `/ready`, `/model`, `/predict`,
# `/predict/batch`, `/metrics`, `/admin/reload`); 18 required fields and the two
# optional ones (`avg_resolution_hours_90d`, `features_used_4w`);
# `additionalProperties: False` (that is `extra="forbid"` in OpenAPI); and
# `sessions_4w` with `minimum: 0`.

# %% [markdown]
# ## 9.9 · Contract design decisions
#
# Every line of a schema is a decision someone can ask you to defend. The ones
# in this service:
#
# | Decision | Chosen | Why | Cost / alternative |
# |---|---|---|---|
# | Categoricals (`region`, `plan`, ...) | plain `str`, **not** an enum | An unseen region (a new market) still gets a prediction: the one-hot encoder ignores unknown categories. Monitoring counts unseen values instead of the API refusing customers | Typos (`"Eu"`) are not caught at the door; you need monitoring. An enum is the right choice when an unknown value would make the prediction meaningless |
# | Unknown fields | `extra="forbid"` → 422 | A typo cannot silently become a missing value | Removing a field later is a breaking change for callers who still send it |
# | Numeric bounds | `ge=0`, `le=…` from domain knowledge | Impossible values never reach the model | Too-tight bounds reject legitimate rare customers; review bounds when the business changes |
# | Batch size | max 1000 → **413** | Bounded memory and latency per request | Callers must chunk (the client does it) |
# | Versioning | not yet; add a `/v1` path prefix (or an `Accept-Version` header) **before** the first breaking change | Old clients keep calling `/v1/predict` while new ones move to `/v2` | Two versions to run for a while |
# | Model version | in every response body and the `x-model-version` header | Any prediction can be traced to a model | none worth mentioning |
#
# **Backwards compatibility** is the rule that decides most of these:
#
# * **Compatible** (old clients keep working): adding an *optional* request
#   field; adding a response field (well-behaved clients ignore unknown
#   response keys); relaxing a bound; adding an endpoint.
# * **Breaking**: adding a *required* request field; removing or renaming a
#   field (with `extra="forbid"` even removing a request field breaks callers
#   who still send it); tightening a bound; changing a type or a meaning
#   (probability → percentage); removing a response field clients read.

# %% [markdown]
# > **Exercise 2.** Marketing wants to send a Net Promoter Score with each
# > customer. Write `make_schema_v2()` that returns a Pydantic model which
# > **extends** `churnlab.schemas.CustomerFeatures` with an optional
# > `nps_score`: an integer from 0 to 10, defaulting to `None`. Everything
# > else must behave as before: all existing fields and bounds, and extra
# > fields still forbidden. (Hint: subclassing a Pydantic model keeps its
# > fields and its `model_config`.)

# %%
from churnlab.schemas import CustomerFeatures


def make_schema_v2():
    # YOUR CODE HERE: define a class that extends CustomerFeatures and return it
    raise NotImplementedError


grade.ex2(make_schema_v2)

# %% [markdown]
# Is your v2 a compatible change? Old clients do not send `nps_score`, so it
# defaults to `None`: compatible. But the *model* was not trained with NPS, so
# the field is only useful once a model that uses it is trained, registered
# and promoted. Contract changes and model changes are separate releases.

# %% [markdown]
# > **Exercise 3 (written).** Five contract changes are proposed for the
# > churn API. For each, say whether it is **breaking** for existing clients
# > and why. Assume clients are written against today's schema.
# >
# > 1. Add a required request field `contract_end_date`.
# > 2. Add a field `explanation` (top three feature contributions) to the `/predict` response.
# > 3. Rename the response field `churn_probability` to `score`.
# > 4. Tighten `seats` from `le=10000` to `le=5000`.
# > 5. Change `region` from a free string to an enum of the six known regions.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Every request returns 422 | the body is not JSON, or you passed `data=` instead of `json=` | use `client.post(path, json=body)`; read `detail[].loc` |
# | 422 with `loc: ['body']` and type `missing` | no body at all, or the wrong `Content-Type` | send `json=...`; check the endpoint expects a body |
# | 422 `extra_forbidden` on a field you are sure exists | a typo, or a field that belongs to a newer schema version | compare against `/openapi.json` |
# | Your endpoint returns 500 in `TestClient` and the cell raises | `TestClient` re-raises server exceptions by default | read the traceback; use `TestClient(app, raise_server_exceptions=False)` to see the 500 instead |
# | `/ready` returns 503 | no model in `registry_dir` (wrong path, empty registry) | `bootstrap.ensure_registry()`; check `Settings(registry_dir=...)` |
# | Changes to a schema class do not take effect | the old class object is still referenced by an app built earlier | rebuild the app (that is what the factory is for) |
# | `ValueError: Out of range float values are not JSON compliant: nan` | you put a NaN in a request body | use `example_payload`, which turns NaN into `None` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. A caller sends `"sesions_4w": 3`. What does this service return, and what would happen without `extra="forbid"`?
# 2. Read this error aloud and say what the caller should change: `{"loc": ["body", "seat_utilization_4w"], "type": "less_than_equal"}`.
# 3. Why are `region` and `plan` plain strings instead of enums? When would you choose an enum?
# 4. What is an app factory, and what goes wrong if the model is loaded at import time?
# 5. What is the difference between `/health` and `/ready`, and between a 413, a 422 and a 503?
# 6. Name two backwards-compatible and two breaking changes to an API contract.
#
# ➡️ **Next:** [`10_consistent_serving.ipynb`](10_consistent_serving.ipynb) — make sure the API returns exactly the scores the notebook computed, and load models safely.
