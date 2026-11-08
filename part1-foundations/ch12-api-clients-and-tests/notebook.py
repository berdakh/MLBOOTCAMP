# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 12 · API clients & tests
#
# **Day 3 · ~60 minutes · CPU only**
#
# A service is only as reliable as the code that calls it and the tests that
# guard it. Networks are slow, servers restart, and someone will eventually
# send a negative session count. This notebook runs the churn API as a real
# HTTP server on your machine, calls it the way a production client should,
# and then writes a test suite that would catch real bugs, and checks that
# claim by planting bugs on purpose.
#
# By the end you will be able to:
#
# 1. run a FastAPI app with uvicorn in a background thread on `127.0.0.1`, and stop it cleanly;
# 2. explain why every call needs a timeout, and show one firing;
# 3. retry only what is worth retrying (429, 502, 503, 504, connection errors) with exponential backoff, and never a 422;
# 4. send large jobs in chunks below the server's batch limit;
# 5. write a pytest suite in four categories (contract, behaviour, parity, operations) and run it;
# 6. use mutation testing to find out which bugs your tests would actually catch.
#
# **You will produce:** `generated/tests_api/test_api.py`, a pytest suite for the service, and a mutation-testing table.

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
# --- Catch up: features, split and a registry (notebooks 03, 04, 08) --------
from churnlab import bootstrap
from grading import ch12 as grade

registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()
valid = split.valid  # validation rows only; the test period waits for notebook 15

# %% [markdown]
# **Expected:** `versions ['v1', 'v2'], CURRENT=v1`.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Client** | the code that calls a service and handles its answers and failures | `ChurnClient` |
# | **Timeout** | the longest a client waits before giving up on a call | 5 s by default |
# | **Retry** | sending the same request again after a failure that might not repeat | a 503 while the model loads |
# | **Exponential backoff** | waiting longer before each retry: $b, 2b, 4b, \dots$ | 0.2 s, 0.4 s, 0.8 s |
# | **Jitter** | a random fraction added to the wait so clients do not retry in lockstep | `delay × uniform(0.5, 1.5)` |
# | **Idempotent** | doing it twice has the same effect as doing it once | a prediction; not a payment |
# | **Chunking** | splitting a big job into requests below the service's limit | 2,500 rows as 1,000 + 1,000 + 500 |
# | **Fixture** | reusable setup that pytest passes into tests | a temporary registry |
# | **Contract test** | checks the API accepts and rejects exactly what it promised | unknown field → 422 |
# | **Mutation testing** | deliberately breaking the code to check that some test fails | a mutant that ignores the threshold |
# | **Killed / surviving mutant** | a test caught the change / no test noticed it | a survivor points at a missing test |

# %% [markdown]
# ## 11.1 · A real server, on your own machine only
#
# `TestClient` (notebooks 09-10) skips the network. A client library has to
# be tried against a real socket: real timeouts, real connection errors. So
# start uvicorn, the ASGI server FastAPI apps run on in production, in a
# **background thread** of this notebook.
#
# Two safety rules, both in the code below:
#
# * bind to **`127.0.0.1`** (this machine only), never `0.0.0.0`, and do not
#   create a public URL or tunnel. In Colab, 127.0.0.1 is the Colab VM itself,
#   reachable only from code in this notebook;
# * pick a **free port** by asking the operating system for one (bind to port
#   0), so a server left over from an earlier run cannot collide with this one.

# %%
import socket
import threading
import time

import uvicorn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # port 0 = "any free port"; the OS chooses
        return s.getsockname()[1]


class BackgroundServer:
    """Run an ASGI app with uvicorn in a daemon thread on 127.0.0.1."""

    def __init__(self, app, port: int | None = None):
        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout: float = 10.0) -> "BackgroundServer":
        self.thread.start()
        deadline = time.time() + timeout
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError(f"server on port {self.port} did not start")
            time.sleep(0.05)
        return self

    def stop(self) -> bool:
        self.server.should_exit = True  # uvicorn finishes in-flight requests, then returns
        self.thread.join(timeout=10)
        return not self.thread.is_alive()


servers: list[BackgroundServer] = []  # everything started here is stopped in 11.8

# %%
from churnlab.api import create_app, example_payload
from churnlab.client import ChurnAPIError, ChurnClient
from churnlab.logs import ListHandler, get_logger
from churnlab.settings import Settings

log_handler = ListHandler()
churn_server = BackgroundServer(create_app(Settings(registry_dir=registry.root),
                                           logger=get_logger("churn-api-nb11", handler=log_handler))).start()
servers.append(churn_server)
print("serving on", churn_server.url)

churn = ChurnClient(churn_server.url, timeout=5.0, retries=3, backoff=0.05)
print(churn.health())
print(churn.ready())
body = example_payload(valid.iloc[0])
print(churn.predict(body))

# %% [markdown]
# **Expected:** `serving on http://127.0.0.1:<some port>`, then
# `{'status': 'ok', ...}`, `{'status': 'ready', 'model_version': 'v1', ...}` and
# a prediction from `v1`. These went through a real TCP socket: the same code
# would call a server in another container or another data centre.
#
# Read `src/churnlab/client.py` now; it is about 70 lines. The rest of this
# section takes its design decisions one at a time.

# %% [markdown]
# ## 11.2 · Timeouts: the default is "wait forever"
#
# To see a timeout you need something slow. A toy app with three endpoints
# that misbehave on purpose:
#
# * `GET /slow?seconds=s` sleeps before answering;
# * `GET /flaky` returns **503** for the first *N* calls after `POST /flaky/reset?fail_first=N`, then 200;
# * `GET /items/{n}` returns 404 for negative `n`.

# %%
from fastapi import FastAPI, HTTPException


def make_toy_app() -> FastAPI:
    app = FastAPI()
    app.state.calls, app.state.fail_first = 0, 0

    @app.get("/slow")
    def slow(seconds: float = 1.0):
        time.sleep(seconds)
        return {"slept": seconds}

    @app.post("/flaky/reset")
    def reset(fail_first: int = 2):
        app.state.calls, app.state.fail_first = 0, fail_first
        return {"fail_first": fail_first}

    @app.get("/flaky")
    def flaky():
        app.state.calls += 1
        if app.state.calls <= app.state.fail_first:
            raise HTTPException(status_code=503, detail=f"warming up (call {app.state.calls})")
        return {"ok": True, "calls": app.state.calls}

    @app.get("/items/{n}")
    def item(n: int):
        if n < 0:
            raise HTTPException(status_code=404, detail="no such item")
        return {"n": n}

    return app


toy_app = make_toy_app()
toy_server = BackgroundServer(toy_app).start()
servers.append(toy_server)

# %%
import httpx

http = httpx.Client(base_url=toy_server.url, timeout=0.5)

start = time.perf_counter()
print(http.get("/slow", params={"seconds": 0.1}).json(), f"{time.perf_counter() - start:.2f}s")

start = time.perf_counter()
try:
    http.get("/slow", params={"seconds": 2})
except httpx.TimeoutException as exc:
    print(f"{type(exc).__name__} after {time.perf_counter() - start:.2f}s")

dead = httpx.Client(base_url=f"http://127.0.0.1:{free_port()}", timeout=0.5)  # nothing listens there
try:
    dead.get("/health")
except httpx.ConnectError as exc:
    print("ConnectError:", str(exc)[:60])

# %% [markdown]
# **Expected:** the fast call returns in about 0.1 s; the slow one raises
# `ReadTimeout` after about 0.5 s; the closed port raises `ConnectError`
# almost immediately (*connection refused*).
#
# Without a timeout, a caller waiting on a hung server holds a thread, a
# connection and often a user's request. When many callers do it at once the
# hang spreads to everything upstream. A timeout turns "hangs forever" into
# a fast, handleable error. Pick it from the latency you measure (notebook
# 13: a few times the p99), not from a guess. `httpx` defaults to 5 s;
# `requests` defaults to **no timeout at all**.

# %% [markdown]
# ## 11.3 · Retries: only for failures that can succeed next time
#
# | Response | Meaning | Retry? |
# |---|---|---|
# | connection error / timeout | the request may never have arrived | **yes**, if the call is idempotent |
# | 429 Too Many Requests | you are being rate-limited | **yes**, after backing off (honour `Retry-After` if sent) |
# | 502 / 503 / 504 | a gateway or the service is temporarily unavailable | **yes** |
# | 500 Internal Server Error | a bug: usually fails the same way again | no (or at most once) |
# | 400 / 404 / 413 / 422 | *your* request is wrong | **never**: it will fail identically |
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-retry-policy.svg" alt="A client call is classified: success returns; 429, 502, 503, 504, timeouts and connection errors back off exponentially and retry up to three times; 400, 404, 413 and 422 are raised at once" width="100%">
#
# *What to notice:* the decision depends only on *what kind* of failure happened, never on how much you want the answer; the dashed loop is bounded.
#
# **Exponential backoff** (`backoff × 2^attempt`) spaces retries out, so a
# struggling server is not hammered by every client retrying at once.
# Production clients add **jitter** (a random fraction of the delay) so that
# clients that failed together do not retry together.
#
# > 💡 **Intuition.** Retrying is redialling a busy phone line: worth it
# > when the line is busy (503, 429), pointless when the number does not exist
# > (404) or you are speaking the wrong language (422). Waiting a little longer
# > between each redial stops a whole office from jamming the switchboard.
# > *Where it breaks down:* a phone call has no side effects until someone
# > answers; an HTTP request may have been processed even though you timed out.
#
# **Idempotency**: a prediction has no side effects, so sending it twice is
# harmless. Retrying a `POST /orders` after a timeout could charge a customer
# twice; such endpoints need an idempotency key.
#
# The course client implements exactly this table. Watch it against the
# flaky endpoint:

# %%
class ToyClient(ChurnClient):
    """ChurnClient's retry loop, pointed at the toy endpoints."""

    def get(self, path: str) -> dict:
        return self._call("GET", path)


toy = ToyClient(toy_server.url, timeout=2.0, retries=3, backoff=0.05)

http.post("/flaky/reset", params={"fail_first": 2})
start = time.perf_counter()
print("fail_first=2, retries=3 ->", toy.get("/flaky"), f"in {time.perf_counter() - start:.2f}s")

http.post("/flaky/reset", params={"fail_first": 5})
try:
    toy.get("/flaky")
except ChurnAPIError as exc:
    print("fail_first=5, retries=3 -> gave up with HTTP", exc.status,
          "| calls the server received:", toy_app.state.calls)

# %% [markdown]
# **Expected:** the first call succeeds on the server's **third** call
# (`'calls': 3`) after about 0.15 s of backoff (0.05 + 0.10); the second gives
# up with HTTP 503 after 1 + 3 = 4 attempts.
#
# And a request that is wrong:

# %%
before = churn.http.get("/metrics").json()["errors"].get("/predict 422", 0)
start = time.perf_counter()
try:
    churn.predict({**body, "sessions_4w": -3})
except ChurnAPIError as exc:
    print(f"HTTP {exc.status} after {time.perf_counter() - start:.3f}s:", exc.detail[0]["loc"], exc.detail[0]["type"])
after = churn.http.get("/metrics").json()["errors"].get("/predict 422", 0)
print("422s the server received for that call:", after - before)

# %% [markdown]
# **Expected:** `HTTP 422` in a few milliseconds, naming
# `['body', 'sessions_4w']` / `greater_than_equal`, and exactly **1** request
# at the server. A client that retried 422s would have sent 4 identical bad
# requests and still failed.
#
# One more detail of `ChurnClient._call`: it generates one `x-request-id` per
# *logical* call and sends the same id on every retry, so all attempts can be
# found together in the server's logs.

# %% [markdown]
# > **Exercise 1.** Write `retrying_get(http, path, retries, backoff)` that
# > calls `http.get(path)` and returns the parsed JSON of the first successful
# > (2xx) response. Rules:
# >
# > * on **429, 502, 503, 504** or a connection error / timeout
# >   (`httpx.TransportError`), wait `backoff * 2**attempt` seconds and try
# >   again, at most `retries` more times; then raise;
# > * on **any other 4xx or 5xx**, raise immediately without retrying
# >   (`response.raise_for_status()` raises `httpx.HTTPStatusError`).
# >
# > The grader uses a fake `http` object that counts calls, with a tiny
# > backoff.

# %%
def retrying_get(http, path: str, retries: int = 3, backoff: float = 0.1) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(retrying_get)

# %% [markdown]
# ## 11.4 · Big jobs: chunk below the batch limit
#
# The service refuses batches above `max_batch_size` (1000) with a 413.
# Scoring the whole validation period at once fails; the client's
# `predict_batch` splits it into chunks:

# %%
customers = [example_payload(r) for _, r in valid.head(2500).iterrows()]

try:
    churn.http.post("/predict/batch", json={"customers": customers}).raise_for_status()
except httpx.HTTPStatusError as exc:
    print("one request of 2,500 ->", exc.response.status_code, exc.response.json()["detail"])

n_before = churn.http.get("/metrics").json()["requests"].get("/predict/batch", 0)
start = time.perf_counter()
preds = churn.predict_batch(customers, chunk_size=1000)
n_after = churn.http.get("/metrics").json()["requests"].get("/predict/batch", 0)
print(f"chunked: {len(preds)} predictions in {n_after - n_before} requests, {time.perf_counter() - start:.2f}s")
print("order preserved:", [p["customer_id"] for p in preds] == [c["customer_id"] for c in customers])

# %% [markdown]
# **Expected:** `413 batch of 2500 exceeds max 1000`; then 2,500 predictions
# in **3** requests (1000 + 1000 + 500), order preserved. (The rejected request
# counts as a 4th call to `/predict/batch` in `/metrics`; it is not included
# in the difference because it happened before `n_before` was read.)

# %% [markdown]
# ## 11.5 · A test suite for the service
#
# Notebooks 09-10 *checked* things by eye. A **test suite** checks them every
# time anyone changes anything, in CI, in seconds. Four categories, each
# answering a different question:
#
# | Category | Question | Examples |
# |---|---|---|
# | **Contract** | does the API accept exactly what it promised? | 422 for out-of-range / unknown / missing fields; 413 for oversize batches |
# | **Behaviour** | are the answers sensible? | probabilities in [0, 1]; `flagged` consistent with the threshold, *including at the boundary*; order preserved |
# | **Parity** | is it the model we validated? | API scores == offline scores on the same rows |
# | **Operations** | does it behave well when things go wrong? | `/ready` 503 without a model while `/health` stays 200; version header and request id present |
#
# Fixtures make the tests fast and independent: the registry is a **temporary
# directory** with a logistic regression trained on a quarter of the
# training rows (a test needs *a* model, not the best one), and every test
# gets a fresh app from the factory. Tests never touch your artifacts
# registry.
#
# One extra line makes the suite reusable in 11.7: the app factory is read
# from the environment variable `CHURN_APP_FACTORY` (default
# `churnlab.api:create_app`).

# %%
TEST_DIR = ROOT / "generated" / "tests_api"
TEST_DIR.mkdir(parents=True, exist_ok=True)

TEST_SOURCE = '''\
"""API tests for the churn service. Generated by notebook 11; run with
    python -m pytest -q generated/tests_api
"""

import importlib
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churnlab import bootstrap
from churnlab.api import example_payload
from churnlab.artifacts import load_bundle
from churnlab.features import FEATURES
from churnlab.logs import ListHandler, get_logger
from churnlab.registry import Registry
from churnlab.settings import Settings
from churnlab.splits import Split


def build_app(**settings):
    module, name = os.environ.get("CHURN_APP_FACTORY", "churnlab.api:create_app").split(":")
    factory = getattr(importlib.import_module(module), name)
    return factory(Settings(**settings), logger=get_logger("churn-api-tests", handler=ListHandler()))


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def split():
    full = bootstrap.ensure_split()
    # small and fast: every 4th training row, 2,000 validation rows, no test rows
    return Split(train=full.train.iloc[::4].reset_index(drop=True),
                 valid=full.valid.iloc[:2000].reset_index(drop=True),
                 test=full.test.iloc[:0])


@pytest.fixture(scope="session")
def registry(tmp_path_factory, split):
    reg = Registry(tmp_path_factory.mktemp("registry"))
    reg.promote(bootstrap.train_and_register(reg, "logreg", split), reason="test fixture")
    return reg


@pytest.fixture
def client(registry):
    return TestClient(build_app(registry_dir=registry.root))


@pytest.fixture(scope="session")
def rows(split):
    return split.valid.head(200)


@pytest.fixture(scope="session")
def payloads(rows):
    return [example_payload(r) for _, r in rows.iterrows()]


# ---------------------------------------------------------------- contract
@pytest.mark.parametrize("field,value", [("sessions_4w", -1), ("seat_utilization_4w", 3.0),
                                         ("seats", 0), ("features_used_4w", 13)])
def test_rejects_out_of_range(client, payloads, field, value):
    r = client.post("/predict", json={**payloads[0], field: value})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][-1] == field


def test_rejects_unknown_field(client, payloads):
    r = client.post("/predict", json={**payloads[0], "sesions_4w": 3.0})
    assert r.status_code == 422
    assert any(e["type"] == "extra_forbidden" for e in r.json()["detail"])


def test_rejects_missing_required_field(client, payloads):
    body = {k: v for k, v in payloads[0].items() if k != "plan"}
    assert client.post("/predict", json=body).status_code == 422


def test_oversize_batch_is_413(registry, payloads):
    small = TestClient(build_app(registry_dir=registry.root, max_batch_size=5))
    assert small.post("/predict/batch", json={"customers": payloads[:6]}).status_code == 413
    assert small.post("/predict/batch", json={"customers": payloads[:5]}).status_code == 200


# ---------------------------------------------------------------- behaviour
def test_probabilities_in_unit_interval_and_order_kept(client, payloads):
    preds = client.post("/predict/batch", json={"customers": payloads}).json()["predictions"]
    assert [p["customer_id"] for p in preds] == [b["customer_id"] for b in payloads]
    assert all(0.0 <= p["churn_probability"] <= 1.0 for p in preds)


def test_flagged_consistent_with_threshold(client, payloads):
    threshold = client.get("/model").json()["threshold"]
    preds = client.post("/predict/batch", json={"customers": payloads}).json()["predictions"]
    for p in preds:
        if abs(p["churn_probability"] - threshold) > 1e-6:  # responses are rounded to 6 dp
            assert p["flagged"] == (p["churn_probability"] >= threshold)


def test_flagged_at_exact_threshold(registry, rows, payloads):
    model, _ = load_bundle(registry.root / registry.current())
    p0 = float(model.predict_proba(rows[FEATURES].head(1))[0, 1])
    at_boundary = TestClient(build_app(registry_dir=registry.root, threshold_override=p0))
    assert at_boundary.post("/predict", json=payloads[0]).json()["flagged"] is True  # p >= t


# ---------------------------------------------------------------- parity
def test_api_matches_offline(client, registry, rows, payloads):
    model, _ = load_bundle(registry.root / registry.current())
    offline = model.predict_proba(rows[FEATURES])[:, 1]
    preds = client.post("/predict/batch", json={"customers": payloads}).json()["predictions"]
    served = np.array([p["churn_probability"] for p in preds])
    np.testing.assert_allclose(served, offline, rtol=0, atol=1e-6)  # API rounds to 6 dp


# ---------------------------------------------------------------- operations
def test_not_ready_without_a_model(tmp_path, payloads):
    empty = TestClient(build_app(registry_dir=tmp_path / "empty"))
    assert empty.get("/health").status_code == 200
    assert empty.get("/ready").status_code == 503
    assert empty.post("/predict", json=payloads[0]).status_code == 503


def test_version_header_and_request_id(client, registry, payloads):
    r = client.post("/predict", json=payloads[0], headers={"x-request-id": "test-123"})
    assert r.headers["x-model-version"] == registry.current() == r.json()["model_version"]
    assert r.headers["x-request-id"] == "test-123"
'''

(TEST_DIR / "test_api.py").write_text(TEST_SOURCE)
print("wrote", (TEST_DIR / "test_api.py").relative_to(ROOT), f"({TEST_SOURCE.count('def test_')} test functions)")

# %% [markdown]
# Run it the way CI would: a separate Python process, from the repository
# root. `BOOTCAMP_ARTIFACTS` points the suite's `ensure_split()` at this
# notebook's feature table so it does not rebuild it.

# %%
def run_pytest(extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "BOOTCAMP_ARTIFACTS": str(ART), **(extra_env or {})}
    # -o addopts= ignores the repo's pytest defaults in pyproject.toml; --color=no keeps the text readable
    cmd = [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider", "--color=no",
           "generated/tests_api"]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


start = time.perf_counter()
result = run_pytest()
print(result.stdout[-1500:] or result.stderr[-1500:])
print(f"exit code {result.returncode} in {time.perf_counter() - start:.1f}s")

# %% [markdown]
# **Expected:** a line of dots, `13 passed`, exit code 0, in a few seconds
# (most of it training the fixture model). There are 10 test functions; the
# four parametrised cases of `test_rejects_out_of_range` count as four tests.

# %% [markdown]
# > **Exercise 2.** Write a test function `test_rejects_negative_sessions(client)`
# > in pytest style: it receives a `TestClient` and **asserts** that `POST
# > /predict` with `sessions_4w = -1` (all other fields valid) is rejected
# > with a 422 that names `sessions_4w`. Use `valid_payload` (defined below) as
# > the valid starting point.
# >
# > The grader runs your test against the real app (it must pass) and against
# > two broken apps that are missing the `≥ 0` bound on `sessions_4w` (it must
# > fail on both). One of them answers 200; the other crashes with a 500.

# %%
valid_payload = example_payload(valid.iloc[0])


def test_rejects_negative_sessions(client):
    # YOUR CODE HERE: build the request, post it, assert on the response
    raise NotImplementedError


grade.ex2(test_rejects_negative_sessions)

# %% [markdown]
# ## 11.6 · Do the tests catch bugs? Mutation testing
#
# A test suite that passes tells you nothing until you know it can **fail**.
#
# > 💡 **Intuition.** Mutation testing is a fire drill for your tests: start a
# > small, controlled fire and check that an alarm goes off. *Where it breaks
# > down:* a drill only covers the fires you thought to start; a suite that
# > kills every hand-made mutant can still miss a bug nobody imagined.
# Mutation testing checks that: make a small, realistic change to the code
# (a *mutant*), run the suite, and see whether some test fails (the mutant is
# *killed*) or everything still passes (the mutant *survived*: no test checks
# that behaviour). Tools such as `mutmut` do this automatically for every line.
# Here you do it by hand for five mutants of `api.py`.
#
# Each mutant is a copy of `src/churnlab/api.py` with one edit, written to
# `generated/tests_api/mutants/`. The original file is never touched.

# %%
import re

API_SOURCE = (ROOT / "src" / "churnlab" / "api.py").read_text()
MUTANT_DIR = TEST_DIR / "mutants"
MUTANT_DIR.mkdir(exist_ok=True)

MUTANTS = {
    "flag uses > instead of >=": ("flagged=bool(s >= threshold)", "flagged=bool(s > threshold)"),
    "two features swapped": (
        "frame[NUMERIC] = frame[NUMERIC].astype(float)",
        "frame[NUMERIC] = frame[NUMERIC].astype(float)\n"
        "        frame[['sessions_4w', 'sessions_prev_4w']] = frame[['sessions_prev_4w', 'sessions_4w']].to_numpy()"),
    "sessions_4w bound dropped": (
        "FEATURES = CATEGORICAL + NUMERIC",
        "FEATURES = CATEGORICAL + NUMERIC\n\n\n"
        "class CustomerFeatures(CustomerFeatures):  # same schema, but sessions_4w loses ge=0\n"
        "    sessions_4w: float"),
    "version header missing": ('response.headers["x-model-version"]', 'response.headers["x-model-vers"]'),
    "4xx not counted as errors": ("if status >= 400:", "if status >= 500:"),
}


def write_mutant(i: int, old: str, new: str) -> str:
    assert API_SOURCE.count(old) == 1, f"mutation target not found exactly once: {old!r}"
    source = re.sub(r"^from \.", "from churnlab.", API_SOURCE.replace(old, new), flags=re.M)
    module = f"api_mutant_{i}"
    (MUTANT_DIR / f"{module}.py").write_text(source)
    return module


def failed_tests(output: str) -> list[str]:
    return sorted({m.split("::")[1].split(" ")[0] for m in re.findall(r"FAILED (\S+)", output)})


rows_out = []
for i, (name, (old, new)) in enumerate(MUTANTS.items(), 1):
    module = write_mutant(i, old, new)
    res = run_pytest({"CHURN_APP_FACTORY": f"{module}:create_app",
                      "PYTHONPATH": os.pathsep.join([str(MUTANT_DIR), str(ROOT / "src")])})
    killed_by = failed_tests(res.stdout)
    rows_out.append({"mutant": name, "result": "killed" if res.returncode else "SURVIVED",
                     "caught by": ", ".join(killed_by) or "-"})

import pandas as pd

mutation_table = pd.DataFrame(rows_out)
print(mutation_table.to_string(index=False))

# %% [markdown]
# **Expected:** four mutants killed, one survived:
#
# * `>` instead of `>=` is caught **only** by `test_flagged_at_exact_threshold`.
#   Without that boundary test, nothing would notice: real scores almost never
#   land exactly on the threshold, so the "consistent with threshold" test
#   passes for both operators. Boundary tests exist for this kind of bug.
# * Two swapped features are caught by the **parity** test only. Every
#   probability is still in [0, 1] and every contract test passes.
# * The dropped bound is caught by the contract test for `sessions_4w`.
# * The missing header is caught by the operations test.
# * **Survived:** 4xx responses no longer counted in `/metrics`. No test
#   checks the metrics endpoint, so a dashboard of client errors would
#   silently go flat. The fix is a test, e.g. *"after one 422, `/metrics`
#   shows `/predict 422: 1`"*; writing it is a good extension.
#
# Mutation testing turns "we have tests" into "we know which bugs our tests
# catch".
#
# > 🧠 **Research bridge.** This is a positive control. In an EEG pipeline you
# > inject a known synthetic source and check that the localisation finds it;
# > here you inject a known bug and check that the suite finds it. A test
# > suite that has never failed is like a method that was never shown a
# > positive.

# %% [markdown]
# > **Exercise 3 (written).** You now have unit tests (`tests/`), API tests
# > that use `TestClient` (this notebook), and a server you can start. For a
# > real deployment, decide what runs **in CI on every push** and what runs
# > **in a staging environment** before production. Name at least three
# > checks for each, and say why each belongs where you put it. (Think about
# > speed, what needs real infrastructure, real data volumes and real network
# > paths.)
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 11.7 · Stop the servers
#
# Background threads outlive the cells that started them. Stop them
# explicitly: `should_exit = True` asks uvicorn to finish in-flight requests
# and return; `join` waits for the thread.

# %%
for s in servers:
    print(s.url, "stopped:", s.stop())
http.close()
dead.close()

try:
    httpx.get(churn_server.url + "/health", timeout=0.5)
except httpx.ConnectError:
    print("port closed: nothing is listening any more")

# %% [markdown]
# **Expected:** both servers report `stopped: True`, and the last request
# fails with a connection error.

# %% [markdown]
# ## 🧮 Key equations
#
# **Exponential backoff.** The wait before retry $n$ ($n = 0, 1, \dots$):
#
# $$
# d_n = b \cdot 2^{n} \quad (\times\ \text{a random factor, with jitter})
# $$
#
# *Symbols:* $b$ = base delay. *Example:* $b = 0.2$ s, 3 retries: waits of
# 0.2, 0.4 and 0.8 s, 1.4 s in total.
#
# **Worst-case time of one logical call** with timeout $\tau$ and $R$ retries:
#
# $$
# T_{\max} = (R + 1)\,\tau + \sum_{n=0}^{R-1} b\, 2^{n} = (R + 1)\,\tau + b\,(2^{R} - 1)
# $$
#
# *In words:* every attempt may hang until the timeout, plus all the waits.
# *Example:* $\tau = 5$ s, $R = 3$, $b = 0.2$: $4 \times 5 + 0.2 \times 7 = 21.4$ s.
# A caller with a 2-second budget cannot afford this client configuration.
#
# **Why retries help with transient failures.** If each attempt fails
# independently with probability $q$, all $R + 1$ attempts fail with
#
# $$
# P(\text{give up}) = q^{\,R + 1}
# $$
#
# *Example:* $q = 0.05$, $R = 3$: $0.05^4 = 6 \times 10^{-6}$. *Where it breaks
# down:* failures during an outage are not independent (if the server is down,
# every attempt fails), which is why retries must be bounded and backed off.
# For a 422, $q = 1$: retrying changes nothing.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `RuntimeError: server on port ... did not start` | an exception inside uvicorn (e.g. an import error in the app) | create the app in a cell first (`create_app(...)`) to see the error; then start the server |
# | `[Errno 98] address already in use` | you hard-coded a port that a previous server still holds | use `free_port()`; stop old servers (`for s in servers: s.stop()`) or restart the runtime |
# | `ConnectError` right after starting | the request was sent before the server finished starting | `BackgroundServer.start()` waits for `server.started`; use it |
# | A call hangs | no timeout on the client | always pass `timeout=`; `httpx` defaults to 5 s, `requests` to none |
# | `pytest` subprocess: `ModuleNotFoundError: churnlab` | run from a directory other than the repo root | `cwd=ROOT`, as in `run_pytest`; locally run `python -m pytest` from the repo root |
# | Mutant table: every mutant "killed" by every test | the suite failed for another reason (e.g. import error in the mutant) | print `res.stdout` for one mutant and read the first error |
# | The notebook kernel will not shut down | a server thread is still running | run section 11.7; threads are daemons, so restarting the runtime also ends them |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Which failures does a well-behaved client retry, which does it never retry, and why?
# 2. What is exponential backoff with jitter, and what problem does each part solve?
# 3. Why must every network call have a timeout? How would you choose its value?
# 4. Name the four categories of API tests in this notebook and one test from each.
# 5. What is mutation testing? Give an example of a bug that a "probabilities are in [0, 1]" test would miss and the test that catches it.
# 6. Why is retrying a prediction safe, but retrying a payment request not, and what do you do about it?
#
# ➡️ **Next:** [`12_packaging_and_configuration.ipynb`](../ch13-packaging-and-configuration/notebook.ipynb) — package the service, configure it from the environment, and write the Dockerfile and CI workflow that run these tests.
