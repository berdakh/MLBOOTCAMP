# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 14 · Health, logs & latency
#
# **Part I · Chapter 14 · about 55 minutes · CPU only**
#
# A running service has to answer three questions for the people and machines
# around it: *is it alive*, *what did it just do*, and *how fast is it*. If the
# first is answered wrongly, the platform may restart a healthy process in a
# loop; if the second is answered poorly, an incident takes hours to diagnose;
# if the third is answered with a mean, the reported latency is one that users
# do not actually experience.
#
# After this chapter you should be able to:
#
# 1. explain liveness versus readiness and show both on the real app;
# 2. capture the service's structured JSON logs, turn them into a table, and follow one request by its id;
# 3. say what must never be logged, and check for it;
# 4. measure latency as a distribution (p50/p95/p99), including the cold start and the first request;
# 5. measure the per-row cost of batching and the effect of concurrent clients, and explain what the GIL, the thread pool and `--workers` change.
#
# **Lab output:** `artifacts/latency/` with the measured latency tables (CSV) and the machine
# they were measured on (JSON).

# %% [markdown]
# **Read first:** [Chapter 14 text](README.md) explains the concepts, equations and caveats behind this lab.

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
# --- Catch up: a registry with a promoted model, and realistic request bodies -
from churnlab import bootstrap
from churnlab.api import example_payload
from grading import ch14 as grade

registry = bootstrap.ensure_registry()
split = bootstrap.ensure_split()
payloads = [example_payload(row) for _, row in split.valid.head(1000).iterrows()]
print(len(payloads), "request bodies from validation rows; first:", payloads[0]["customer_id"])

# %% [markdown]
# **Expected:** `versions ['v1', 'v2'], CURRENT=v1` and 1000 request bodies.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Liveness probe** | "is the process alive?"; failing it gets the process restarted | `GET /health` |
# | **Readiness probe** | "should this process get traffic?"; failing it only stops routing | `GET /ready` returns 503 without a model |
# | **Structured log** | one machine-readable record (JSON) per event | `{"request_id": ..., "status": 200, "ms": 4.1}` |
# | **Request id** | an identifier that ties all log lines of one request together | the `x-request-id` header |
# | **Latency** | time from sending a request to receiving the answer | milliseconds per `/predict` |
# | **Percentile (p50, p95, p99)** | the latency below which that share of requests fall | p95 = 95% of requests are faster |
# | **Tail latency** | the slow end of the distribution (p95 and above) | what users complain about |
# | **Throughput** | requests completed per second | `throughput_rps` in the load test |
# | **Batching** | sending many rows in one request so fixed costs are paid once | 1,000 rows per call |
# | **GIL** | CPython runs Python bytecode on one thread at a time per process | why threads do not add CPU throughput |
# | **Worker** | a separate server process with its own interpreter and model copy | `uvicorn --workers 2` |
# | **SLO** | a target for a service metric over a window | 99.9% of requests under 200 ms over 30 days |

# %% [markdown]
# ## 14.1 · Liveness versus readiness
#
# An orchestrator (Kubernetes, a cloud run-time, a load balancer) asks two
# different questions, on two endpoints:
#
# | Probe | Question | Our endpoint checks | If it fails, the platform... |
# |---|---|---|---|
# | **Liveness** | is the process alive? | nothing but "I can answer HTTP" | **restarts** the process |
# | **Readiness** | should I receive traffic? | a model is loaded and can predict | **stops routing** traffic to it, and leaves it running |
#
# The difference is shown below on an app whose registry is empty, as on a new
# machine where the model has not yet been copied:

# %%
import tempfile

from fastapi.testclient import TestClient

from churnlab.api import create_app
from churnlab.logs import ListHandler, get_logger
from churnlab.settings import Settings

quiet = get_logger("nb13-quiet", "ERROR", handler=ListHandler())
empty_dir = Path(tempfile.mkdtemp(prefix="empty-registry-"))
not_ready = TestClient(create_app(Settings(registry_dir=empty_dir), logger=quiet))
for path in ("/health", "/ready"):
    r = not_ready.get(path)
    print(f"empty registry  GET {path:<8} -> {r.status_code} {r.json()}")
r = not_ready.post("/predict", json=payloads[0])
print(f"empty registry  POST /predict -> {r.status_code} {r.json()['detail'][:60]}")

ready = TestClient(create_app(Settings(registry_dir=registry.root), logger=quiet))
for path in ("/health", "/ready"):
    r = ready.get(path)
    print(f"with a model    GET {path:<8} -> {r.status_code} {r.json()}")

# %% [markdown]
# **Expected:** with the empty registry `/health` is **200** and `/ready` is
# **503** with the reason (`no CURRENT model in ...`), and `/predict` is 503 too.
# With a model, both are 200 and `/ready` names `v1`.
#
# Why two probes? Consider a model that takes 40 seconds to load. If the
# *liveness* probe checked the model, the platform would kill the process at
# second 10 for being "dead", start a new one, kill that too, and so on: a
# restart loop caused by the health check itself. With the split, the process
# stays alive, is simply not sent traffic until it is ready, and if a reload
# fails later the old model keeps serving (`ModelHolder.load` only swaps on
# success). Rule of thumb: **liveness must never depend on anything outside
# the process** (model files, databases, other services).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-liveness-readiness.svg" alt="The liveness probe asks whether the process answers HTTP and a failure restarts it; the readiness probe asks whether a model is loaded and a failure only stops traffic" width="100%">
#
# *What to notice:* the two failures lead to opposite actions, so they must ask different questions; a restart cannot fix a missing model file.
#
# > **Intuition.** Liveness is "is the shop's owner breathing?"; readiness
# > is "is the shop open?". You call an ambulance for the first, and simply
# > send customers elsewhere for the second. *Where it breaks down:* a process
# > can be alive and answering `/health` while stuck on every real request
# > (a deadlock in the model call); the readiness check should exercise the
# > model, not just report that a file was loaded.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/serving.svg" alt="Serving: client, FastAPI app with health, ready and predict, model loaded from the registry" width="100%">

# %% [markdown]
# ## 14.2 · Structured logs
#
# The app writes one JSON object per request. In production those lines go to
# stdout and a log platform indexes them; here a `ListHandler` keeps them in a
# list so we can read them. Send a realistic mix of traffic, including some bad
# requests, and one request whose id we choose:

# %%
handler = ListHandler()
logger = get_logger("nb13-logs", "INFO", handler=handler)
client = TestClient(create_app(Settings(registry_dir=registry.root), logger=logger))

for body in payloads[:40]:
    client.post("/predict", json=body)
for body in payloads[40:43]:
    client.post("/predict", json={**body, "sessions_4w": -5})  # violates the contract -> 422
client.post("/predict/batch", json={"customers": payloads[100:150]})
client.get("/health")
client.get("/ready")
traced = client.post("/predict", json=payloads[200], headers={"x-request-id": "trace-me-123"})

print(len(handler.records), "log lines; the last two raw lines:")
for line in handler.records[-2:]:
    print(line)

# %% [markdown]
# **Expected:** 48 log lines: one `model_loaded` event and one `request` event per call.
# Each line is valid JSON: no regular expressions are needed to read it.

# %%
import pandas as pd

logs = pd.DataFrame(handler.parsed())
requests_log = logs[logs["msg"] == "request"].astype({"status": int})  # NaN-free once filtered
print(requests_log.groupby(["path", "status"]).size().rename("count").to_string())
requests_log[["request_id", "method", "path", "status", "latency_ms", "model_version"]].tail(4)

# %% [markdown]
# **Expected:** `/predict` 200 × 41, `/predict` 422 × 3, and one each of
# `/predict/batch`, `/health`, `/ready`.
#
# **Follow one request end to end.** The caller sent `x-request-id:
# trace-me-123`; the service returned it in the response header and wrote it in
# its log line. When a customer-success colleague says "the request I made at
# 10:42 was slow", this id is what connects their complaint to the server's
# record. (When the caller sends no id, the middleware makes one up and still
# returns it.)

# %%
print("response header :", traced.headers["x-request-id"], "| model", traced.headers["x-model-version"])
print("response body   :", traced.json())
print("server log line :", [r for r in handler.parsed() if r.get("request_id") == "trace-me-123"])

# %% [markdown]
# **What never to log.** Logs are copied widely (log platforms, support
# tickets, laptops) and kept long. Log **identifiers and aggregates**: request
# id, path, status, latency, model version. Do **not** log raw feature values,
# names, emails, IP addresses beyond what security requires, or anything a
# data-protection officer would call personal data. If you need the inputs
# (for drift monitoring or debugging), keep them in a store with access control
# and retention, not in the log stream. A check you can put in a test:

# %%
from churnlab.features import FEATURES

sensitive = set(FEATURES) | {"customer_id", "email", "name"}
logged_keys = set().union(*(record.keys() for record in handler.parsed()))
print("keys that appear in the logs:", sorted(logged_keys))
print("sensitive keys found        :", sorted(logged_keys & sensitive) or "none")

# %% [markdown]
# **Expected:** keys such as `latency_ms`, `model_version`, `path`,
# `request_id`, `status`; sensitive keys found: `none`. (The app does keep
# recent *inputs* in memory, `Metrics.recent_inputs`, for drift monitoring in
# chapter 15. They never reach the log.)

# %% [markdown]
# ## 14.3 · Latency is a distribution
#
# Users experience **every** request, not the average one. A service whose mean
# is 5 ms can still make one call in twenty take 50 ms, and a page that makes
# twenty calls hits that slow one almost every time. So latency is reported as
# percentiles: **p50** (the typical request), **p95** and **p99** (the tail
# that users complain about), and the number of requests they are based on.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-latency-percentiles.svg" alt="A right-skewed latency distribution with p50, mean, p95 and p99 marked; the tail lies far to the right of the typical request" width="100%">
#
# *What to notice:* the mean sits to the right of the median and far to the left of p95; neither average describes the requests users remember.
#
# > **Intuition.** Latency percentiles are queue times at a supermarket:
# > the median shopper waits two minutes, but the one in twenty stuck behind a
# > price check waits ten, and that is the one who writes the review. *Where
# > it breaks down:* a p99 from 200 requests rests on the 2 slowest; report the
# > count, and do not quote p99 from small samples.
#
# Before measuring, write down **the machine**. Every number below belongs to
# it; a Colab CPU is not a production server.

# %%
import json
import platform

LAT = ART / "latency"
LAT.mkdir(parents=True, exist_ok=True)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


machine = {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
           "platform": platform.platform(terse=True), "colab": "google.colab" in sys.modules}
(LAT / "machine.json").write_text(json.dumps(machine, indent=2))
machine

# %% [markdown]
# **Cold start.** Before its first request, a new process has to start Python,
# import pandas, scikit-learn and FastAPI, and load the model. Measure it the
# way a platform experiences it: start a real server process (uvicorn's CLI,
# as in a container; `--factory` tells uvicorn to call `create_app()`, which
# reads the `CHURN_*` variables) and time how long until `/ready` answers 200.
# Then time its first requests.

# %%
import socket
import time

import httpx
import numpy as np


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(workers: int = 1):
    """Start `uvicorn churnlab.api:create_app --factory` in a child process; wait for /ready."""
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "CHURN_REGISTRY_DIR": str(registry.root),
           "CHURN_LOG_LEVEL": "WARNING"}
    cmd = [sys.executable, "-m", "uvicorn", "churnlab.api:create_app", "--factory", "--host", "127.0.0.1",
           "--port", str(port), "--workers", str(workers), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url, t0 = f"http://127.0.0.1:{port}", time.perf_counter()
    while True:
        if proc.poll() is not None or time.perf_counter() - t0 > 90:
            proc.kill()
            raise RuntimeError("server failed to start")
        try:
            if httpx.get(url + "/ready", timeout=1).status_code == 200:
                return proc, url, time.perf_counter() - t0
        except httpx.TransportError:
            time.sleep(0.2)


def timed_calls(send, bodies) -> np.ndarray:
    out = []
    for body in bodies:
        t0 = time.perf_counter()
        r = send(body)
        out.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200, r.text
    return np.array(out)


proc, url, startup_s = start_server()
try:
    with httpx.Client(base_url=url, timeout=10) as http:
        cold = timed_calls(lambda b: http.post("/predict", json=b), payloads[:55])
finally:
    proc.terminate()
    proc.wait(timeout=15)

t0 = time.perf_counter()
create_app(Settings(registry_dir=registry.root), logger=quiet)
load_ms = (time.perf_counter() - t0) * 1000
print(f"process start until /ready      : {startup_s:.1f} s")
print(f"  of which loading the model    : {load_ms:.0f} ms (create_app, measured in-process)")
print("first three requests (ms)       :", [round(float(x), 1) for x in cold[:3]])
print(f"median of requests 6-55 (ms)    : {np.median(cold[5:]):.1f}")

# %% [markdown]
# **Expected (machine-specific):** one to a few seconds until ready, almost all
# of it imports (loading this small model takes only milliseconds), and the
# **first request** clearly slower than the typical one afterwards (2-5× in
# our runs)
# (one-time costs: Pydantic and FastAPI caches, the first pass through pandas
# and scikit-learn code paths). That is why a deployment sends a few
# **warm-up** requests before it reports ready, why autoscaling cannot react
# faster than the cold start, and why you **discard warm-up** before measuring.
#
# Now 300 single predictions, two ways: in-process through `TestClient` (no
# network, no server) and over real HTTP to a uvicorn server running in a
# background thread of this notebook.

# %%
import threading

import uvicorn


class ThreadServer:
    """uvicorn in a background thread, for the duration of a `with` block."""

    def __init__(self, app):
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 20
        while not self.server.started:
            if time.time() > deadline:
                raise TimeoutError("server did not start")
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=10)


c = TestClient(create_app(Settings(registry_dir=registry.root), logger=quiet))
c.post("/predict", json=payloads[0])  # warm-up

N = 300
bodies = [payloads[i % len(payloads)] for i in range(N)]
in_process = timed_calls(lambda b: c.post("/predict", json=b), bodies)

http_app = create_app(Settings(registry_dir=registry.root), logger=quiet)
with ThreadServer(http_app) as server, httpx.Client(base_url=server.url, timeout=5) as http:
    timed_calls(lambda b: http.post("/predict", json=b), bodies[:20])  # warm-up, discarded
    over_http = timed_calls(lambda b: http.post("/predict", json=b), bodies)
    server_metrics = http.get("/metrics").json()


def summary(ms: np.ndarray) -> dict:
    return {"n": len(ms), "mean": ms.mean(), "p50": np.percentile(ms, 50), "p95": np.percentile(ms, 95),
            "p99": np.percentile(ms, 99), "max": ms.max()}


single = pd.DataFrame({"in-process (TestClient)": summary(in_process),
                       "HTTP (uvicorn, keep-alive)": summary(over_http)}).T.round(2)
single.to_csv(LAT / "single_request_ms.csv")
single

# %% [markdown]
# **Expected (machine-specific):** p50 of a few to roughly twenty milliseconds
# for both (on localhost the network adds well under a millisecond, so the two
# paths are close and noise decides which is higher); p99 and max well above
# p50, and the mean pulled away from the median by the slow tail. The tail comes from things you do
# not control in a notebook: garbage collection, other processes, the OS
# scheduler. Re-run the cell and the p99 moves more than the p50: tails need
# many requests to estimate.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.5))
bins = np.linspace(0, np.percentile(over_http, 99.5), 50)
ax.hist(in_process, bins=bins, alpha=0.6, label="in-process")
ax.hist(over_http, bins=bins, alpha=0.6, label="HTTP")
for q, style in ((50, "-"), (95, "--"), (99, ":")):
    ax.axvline(np.percentile(over_http, q), color="k", linestyle=style, linewidth=1, label=f"HTTP p{q}")
ax.set_xlabel("latency per /predict request (ms)")
ax.set_ylabel("requests")
ax.set_title(f"{N} single predictions after warm-up")
ax.legend()
plt.show()

# %% [markdown]
# The service measures itself too: the middleware records every `/predict*`
# latency and `GET /metrics` reports percentiles. Compare its view with the
# client's:

# %%
print("server-side /metrics:", json.dumps(server_metrics["latency_ms"]))
print(f"client-side HTTP    : p50={np.percentile(over_http, 50):.2f}  p95={np.percentile(over_http, 95):.2f}")

# %% [markdown]
# **Expected:** the server's percentiles are **lower** than the client's: the
# server starts its clock after the request has arrived and stops it before
# the response is sent, so it does not see connection handling, serialization
# on the wire or client overhead. Users experience the client-side number. An
# SLO should say which one it means.
#
# *(The service's `/metrics` uses `np.percentile`'s default linear
# interpolation. The exercise below asks for the nearest-rank definition,
# which always returns a latency that actually happened.)*
#
# **A rough edge worth probing:** what does `/metrics` return *before any
# prediction*? Percentiles of an empty window are undefined: `np.percentile`
# of nothing is `NaN`, and JSON cannot encode `NaN`. An earlier version of
# `churnlab` crashed with a 500 here. Look at what this version does:

# %%
r = TestClient(create_app(Settings(registry_dir=registry.root), logger=quiet)).get("/metrics")
print("GET /metrics on a fresh app ->", r.status_code, r.text[:80])

# %% [markdown]
# **Expected:** `200` with `"p50":null` (and p95, p99 null). The 500 was
# found while this course was being written: a monitoring scraper that starts
# before the first prediction would have recorded the service as broken. The
# fix (`Metrics.summary` in `api.py`) returns `None` for an empty window. The
# matching regression test belongs in the suite you wrote in chapter 12.

# %% [markdown]
# > **Exercise 1.** Write `percentiles(latencies, qs=(50, 95, 99))` using the
# > **nearest-rank** definition, and return a dict `{q: value}`.
# >
# > *Definition.* Sort the n values: x₁ ≤ x₂ ≤ … ≤ xₙ. For 0 < q ≤ 100, the
# > q-th percentile is x_k with **k = ⌈q·n / 100⌉** (1-based). It is always one of
# > the observed values; no interpolation. Example: for `[15, 20, 35, 40, 50]`,
# > p50 = x₃ = 35 and p95 = x₅ = 50. Raise `ValueError` for an empty input.
# > Compute `q * n / 100` in that order (`q / 100 * n` can land a hair above an
# > integer in floating point, and the ceiling then picks the wrong rank).

# %%
def percentiles(latencies, qs=(50, 95, 99)) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(percentiles)

# %% [markdown]
# > **Exercise 2.** Back to the logs of section 14.2. Write `summarize_logs(records)` that turns parsed log
# > records (a list of dicts like `handler.parsed()`) into a DataFrame with **one
# > row per `path`**, the index named `path` and sorted, and three columns:
# >
# > * `count` — number of `request` records for that path;
# > * `error_rate` — fraction of them with `status >= 500` (server errors; a 4xx
# >   is the caller's mistake and does not spend the service's error budget);
# > * `p95_latency_ms` — the **nearest-rank** 95th percentile of `latency_ms`
# >   (as defined in Exercise 1).
# >
# > Ignore records whose `msg` is not `"request"` (such as `model_loaded`).
# > Reuse your `percentiles` from Exercise 1.

# %%
def summarize_logs(records: list[dict]) -> pd.DataFrame:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(summarize_logs)

# %% [markdown]
# ## 14.4 · Batching: the per-row cost
#
# A nightly job scoring 5,000 customers can send 5,000 requests or five
# batches of 1,000. Each request pays fixed costs (HTTP, JSON parsing,
# validation, building a DataFrame, one `predict_proba` call); a batch pays
# them once:
#
# $$
# t_{\text{request}}(n) \approx t_{\text{fixed}} + n \cdot t_{\text{row}}
# \quad\Rightarrow\quad
# \text{per-row cost} = \frac{t_{\text{fixed}}}{n} + t_{\text{row}}
# $$
#
# > **Intuition.** Batching is a bus instead of 1,000 taxis: the fare to
# > start the journey is paid once. *Where it breaks down:* everyone on the
# > bus arrives when the last passenger is processed, so batching raises the
# > latency of each individual answer; it is for throughput, not for a user
# > waiting on a page.

# %%
batch_rows = []
with ThreadServer(create_app(Settings(registry_dir=registry.root), logger=quiet)) as server, \
        httpx.Client(base_url=server.url, timeout=30) as http:
    for size, reps in ((1, 30), (10, 30), (100, 15), (1000, 5)):
        body = {"customers": payloads[:size]}
        http.post("/predict/batch", json=body)  # warm-up
        times = timed_calls(lambda b: http.post("/predict/batch", json=b), [body] * reps)
        batch_rows.append({"batch_size": size, "requests": reps, "p50_ms": np.median(times),
                           "p95_ms": np.percentile(times, 95), "ms_per_row": np.median(times) / size})
batch = pd.DataFrame(batch_rows).round(3)
batch.to_csv(LAT / "batch_ms.csv", index=False)
batch

# %% [markdown]
# **Expected (machine-specific):** the request time grows much more slowly
# than the batch size, so the per-row cost falls by one to two orders of
# magnitude from a batch of 1 to a batch of 1000. Batching buys throughput
# with latency: nobody waits for a 1000-row batch in an interactive screen,
# but an offline job should always batch (and `ChurnClient.predict_batch`
# splits large inputs into chunks under the server's `max_batch_size`).

# %% [markdown]
# ## 14.5 · Concurrent clients, the GIL, and `--workers`
#
# How does the service behave when several clients call it at once? The
# server is again a **separate process** started with uvicorn's CLI (one
# worker). Client threads in this notebook send 240 requests in total at
# concurrency 1, 4 and 16; each client keeps its own connection open.

# %%
from concurrent.futures import ThreadPoolExecutor


def load_test(url: str, clients: int, total: int = 240) -> dict:
    per_client = total // clients

    def client_thread(k):
        with httpx.Client(base_url=url, timeout=10) as http:
            sent = [payloads[(k * per_client + i) % len(payloads)] for i in range(per_client)]
            return timed_calls(lambda b: http.post("/predict", json=b), sent)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=clients) as pool:
        lat = np.concatenate(list(pool.map(client_thread, range(clients))))
    wall = time.perf_counter() - t0
    return {"clients": clients, "requests": len(lat), "throughput_rps": len(lat) / wall,
            "p50_ms": np.percentile(lat, 50), "p95_ms": np.percentile(lat, 95)}


proc, url, _ = start_server(workers=1)
try:
    load_test(url, 4, total=40)  # warm-up
    concurrency = pd.DataFrame([load_test(url, clients) for clients in (1, 4, 16)]).round(2)
finally:
    proc.terminate()
    proc.wait(timeout=15)
concurrency.to_csv(LAT / "concurrency.csv", index=False)
concurrency

# %% [markdown]
# **Expected (machine-specific; the table is the evidence, not this text):**
# throughput does **not** grow from 1 to 16 clients (in our runs it fell by a
# quarter or so, because the client threads compete with the server for the
# same CPUs), while p50 and p95 grow roughly in proportion to the number of
# clients. Requests are not processed faster when more arrive at once; they
# **queue**.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(concurrency["clients"], concurrency["throughput_rps"], marker="o")
axes[0].set_ylabel("throughput (requests / s)")
axes[0].set_ylim(bottom=0)
axes[1].plot(concurrency["clients"], concurrency["p50_ms"], marker="o", label="p50")
axes[1].plot(concurrency["clients"], concurrency["p95_ms"], marker="o", label="p95")
axes[1].set_ylabel("latency (ms)")
axes[1].legend()
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16], labels=["1", "4", "16"])
    ax.set_xlabel("concurrent clients")
fig.suptitle(f"/predict, one uvicorn worker, {machine['logical_cpus']} logical CPUs ({machine['cpu'][:40]})")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Why it behaves like this.**
#
# * The `/predict` handler is a plain `def`, so FastAPI runs it in a **thread
#   pool** (40 threads by default) to keep the event loop free. Threads let
#   requests *wait* concurrently, but the work here is CPU-bound Python
#   (validation, a one-row DataFrame, `predict_proba`), and in CPython only one
#   thread at a time runs Python bytecode: the **GIL**. So one process has a
#   throughput ceiling, and extra concurrency turns into queueing, i.e. latency.
# * `--workers N` starts **N processes**, each with its own interpreter, GIL and
#   copy of the model. CPU-bound throughput scales with N up to the number of
#   free cores; memory scales with N too. And everything held in memory is
#   **per worker**: `/metrics` describes one worker, `/admin/reload` reloads
#   one worker (chapter 15 relies on a single worker).
# * This notebook does not benchmark `--workers 2`, for two reasons. On a
#   2-core Colab runtime the client threads compete with the server for the
#   same cores, so the comparison says little about the server. And on the
#   machine this course was verified on, uvicorn's multi-process mode added
#   about 40 ms to *every* request on a kept-alive connection (even `/health`;
#   seen with uvicorn 0.34 and 0.54; the size of the delay points at the TCP
#   Nagle / delayed-ACK interaction), so it was *slower* for a single client. A
#   configuration that "obviously" helps can hurt: measure it on the target
#   machine before and after.
# * In production you usually scale out instead: one or two workers per
#   container, more containers behind a load balancer, autoscaled on CPU or
#   on p95.
# * For *throughput*, batching (14.4) beats concurrency by a wide margin.
#
# > **Research bridge.** The same trade-off appears in PyTorch:
# > `DataLoader(num_workers=...)` uses processes, not threads, for the same
# > GIL reason, and inference throughput comes from batching. A web service is
# > the same problem with HTTP in front.

# %% [markdown]
# > **Exercise 3 (written).** Propose a **service level objective** for this
# > churn API and an **alert** on it. Your SLO names the metric (success rate
# > and/or latency, and whether it is measured at the server or the client),
# > the target, the window, and what counts as a "bad" request. Your alert says
# > when someone is paged versus when a ticket is opened. Use the latency you
# > measured above (and its machine) to justify the target, and say who calls
# > this service and how fast they really need an answer. Write your answer in
# > the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **Percentile.** For $N$ sorted latencies $x_{(1)} \le \dots \le x_{(N)}$, the
# $q$-th percentile is (nearest-rank form)
#
# $$
# p_q = x_{(\lceil q N / 100 \rceil)}
# $$
#
# *In words:* the value with $q\%$ of observations at or below it. NumPy's
# default interpolates between neighbours, so the two differ slightly on small
# samples. *Example:* 20 latencies, $p_{95} = x_{(19)}$: the second slowest.
#
# **Tail amplification.** A page that makes $k$ independent calls is slower
# than the per-call $p_q$ whenever any call is:
#
# $$
# P(\text{at least one slow call}) = 1 - (q/100)^{k}
# $$
#
# *Example:* $q = 95$, $k = 20$: $1 - 0.95^{20} = 0.64$.
#
# **Little's law** (any stable system):
#
# $$
# L = \lambda\, W
# $$
#
# *In words:* average requests in flight = arrival rate × average time in the
# system. *Symbols:* $L$ = concurrency, $\lambda$ = throughput (req/s), $W$ =
# latency (s). *Example:* one worker that completes $\lambda = 200$ req/s with
# 16 clients always waiting ($L = 16$) must have $W = 16/200 = 80$ ms, even if
# one request takes only 5 ms of CPU: the rest is queueing, which is what 14.5
# measured.
#
# **Single-process ceiling:** $\lambda_{\max} \approx 1 / t_{\text{CPU per request}}$.
# *Example:* 5 ms of CPU-bound Python per request → about 200 req/s per
# process, whatever the number of threads (the GIL).

# %%
# Check the percentile definitions, tail amplification and Little's law with numbers.
lat_toy = np.array([3.1, 3.4, 3.6, 3.8, 4.0, 4.1, 4.3, 4.4, 4.6, 4.8,
                    5.0, 5.3, 5.6, 6.0, 6.5, 7.1, 8.0, 9.5, 14.0, 31.0])  # ms, 20 requests
nearest_rank = np.sort(lat_toy)[int(np.ceil(0.95 * lat_toy.size)) - 1]
print(f"p50 {np.percentile(lat_toy, 50):.2f} ms | mean {lat_toy.mean():.2f} ms | "
      f"p95 nearest-rank {nearest_rank:.1f} ms, numpy {np.percentile(lat_toy, 95):.2f} ms")
print(f"P(20-call page hits a slower-than-p95 call) = {1 - 0.95 ** 20:.3f}")
print(f"Little's law: 16 in flight at 200 req/s -> W = {16 / 200 * 1000:.0f} ms")

# %% [markdown]
# **Expected:** p50 4.90 ms, mean 6.90 ms (pulled up by one 31 ms request),
# p95 14.0 ms by nearest rank and 14.85 ms by NumPy's interpolation; 0.642;
# 80 ms.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `TimeoutError: server did not start` (thread server) | port taken, or a previous server in this kernel still running | re-run the cell (a new free port is chosen); restart the runtime if it persists |
# | `RuntimeError: server failed to start` (subprocess) | `churnlab` not importable in the child, or registry path wrong | the child needs `PYTHONPATH=<repo>/src` and an absolute `CHURN_REGISTRY_DIR`; run it by hand in a terminal to see its error |
# | Latency numbers very different from a classmate's | different machine, other load, first-request effects | expected: report the machine (`machine.json`) next to every number, discard warm-up |
# | p99 jumps between runs | a few hundred requests estimate a tail poorly | send more requests, or report p95 and say so |
# | `GET /metrics` shows `null` percentiles | no prediction has been recorded yet | expected; send a prediction first |
# | With `--workers 2`, every request is ~40 ms slower | multi-process mode on some Linux set-ups (see 14.5) | measure before adopting; compare with one worker per container and more containers |
# | The notebook takes much longer than 3 minutes | the CPU is shared or slow | reduce `N` and `total` in 14.3 and 14.5 |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. What is the difference between a liveness and a readiness probe, and what goes wrong if liveness checks the model?
# 2. A colleague reports "average latency is 6 ms, we're fine". What do you ask for instead, and why?
# 3. What should a request log line contain, and what must it never contain?
# 4. Why does a server-side latency metric read lower than what the client measures?
# 5. Your service is CPU-bound and p95 doubles at peak traffic. Name three things you could change, and the cost of each.
# 6. What does `--workers 4` change, and what does it break for in-memory metrics and model reloads?
#
# **Next chapter:** [Chapter 15 · Monitoring & rollback](../ch15-monitoring-and-rollback/notebook.ipynb) — with the service shown to be healthy, we check whether the *model* still is, and practise undoing a bad deployment.

