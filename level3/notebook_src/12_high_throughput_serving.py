# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 12 · High-throughput serving
#
# **Day 6 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Serving cost is dominated by how much useful work each accelerator does per second,
# and the single biggest lever is **batching**: one model call on 32 requests costs far
# less than 32 calls on one. But batching makes requests wait, so every serving system
# trades latency against throughput. This notebook builds a dynamic batcher, puts it
# behind a FastAPI endpoint, load-tests it, and then moves to text generation, where KV
# caches, continuous batching and paged attention decide how many users one GPU can serve.
#
# By the end you will be able to:
#
# 1. measure why batching raises throughput (memory-bound small batches) and reason about GPU utilisation;
# 2. implement a dynamic batcher with `asyncio` (max batch size, max wait) and serve it with FastAPI;
# 3. load-test at several concurrency levels and read throughput vs p50/p95 latency curves, including the effect of `max_wait_ms`;
# 4. measure generation speed with and without a KV cache (GPT-2 small);
# 5. explain static vs continuous batching (with a simulation) and paged attention;
# 6. compute the cost per 1,000 requests from throughput, price and utilisation.
#
# **You will produce:** `artifacts/level3/nb12/load_test.json` (your latency/throughput
# measurements) and `artifacts/level3/nb12/generation.json`. Code:
# [`src/scaleml/serving.py`](../../src/scaleml/serving.py).
#
# **Model:** [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) (124M
# parameters, MIT licence), pinned to a revision, for the generation part. The batching part
# serves a small synthetic MLP so that model quality is not the point.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*. The server and the load generator share
# one machine (and one Python process), so absolute numbers are pessimistic; the *shapes*
# of the curves are what transfer.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level3/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# Level 3 is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing from earlier notebooks is needed -------------------------
import asyncio
import json
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
import torch

from scaleml import profiling as prof
from scaleml import serving as S
from scaleml.device import seed_everything, sync
from scaleml.grading import nb12 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb12"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)

# The model we serve: an MLP whose weights (135 MB in fp32) are large compared with one input.
IN_DIM, HIDDEN, N_CLASSES = 64, 4096, 10
torch.manual_seed(0)
mlp = torch.nn.Sequential(torch.nn.Linear(IN_DIM, HIDDEN), torch.nn.ReLU(), torch.nn.Linear(HIDDEN, HIDDEN),
                          torch.nn.ReLU(), torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
                          torch.nn.Linear(HIDDEN, N_CLASSES)).to(DEVICE).eval()
n_params = sum(p.numel() for p in mlp.parameters())
print(f"served model: {n_params / 1e6:.1f}M parameters ({n_params * 4 / 1e6:.0f} MB fp32) on {DEVICE}")

# %% [markdown]
# **Expected:** `served model: 33.9M parameters (135 MB fp32)`.
#
# ## 12.1 · Why batching helps: latency vs batch size
#
# One request through a Linear layer is a matrix-**vector** product: every weight is read
# from memory and used for just one multiply-add, about **1 FLOP per 2 bytes** (fp16) or 4
# bytes (fp32). The processor waits on memory, not on arithmetic. A batch of B requests
# reuses each weight B times for the same memory traffic, so until the batch is large
# enough to become compute-bound, **a bigger batch costs almost no extra time**.

# %%
rows = []
with torch.inference_mode():
    for bs in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.randn(bs, IN_DIM, device=DEVICE)
        r = prof.benchmark(lambda x=x: mlp(x), DEVICE, warmup=3, iters=15, min_seconds=0.2)
        t = r["min_s"]  # best of the repeats: least disturbed by other processes on a shared machine
        rows.append({"batch": bs, "ms": t * 1e3, "median_ms": r["median_s"] * 1e3, "examples_per_s": bs / t})
        print(f"batch {bs:4d}: {t * 1e3:7.2f} ms per call (median {r['median_s'] * 1e3:7.2f}) | "
              f"{t * 1e3 / bs:6.3f} ms per example | {bs / t:9,.0f} examples/s")
BATCH_CURVE = rows

# %% [markdown]
# **Expected (CPU, 2 threads, measured):** batch 1 takes a few milliseconds, and the time per
# call grows much more slowly than the batch: throughput rises by roughly an order of
# magnitude between batch 1 and batch 64–128 (our run: see the printout; the exact numbers
# depend on the CPU and its load). **On a T4 you should see** the same shape, more extreme:
# a batch-1 call is dominated by kernel launches and weight reads (~320 GB/s of memory
# bandwidth against 65 TFLOP/s of fp16 compute, a ratio of ~200 FLOPs per byte), so
# throughput keeps rising until batches of hundreds (not measured in our verification runs).
#
# **GPU utilisation reasoning.** `nvidia-smi` "GPU-Util" only says *a kernel was running*,
# not how much of the GPU it used: a batch-1 server can show 100% "utilisation" while using
# 1% of the arithmetic. Better measures: achieved FLOP/s vs peak (MFU, notebook 10), memory
# bandwidth used vs peak, and simply *throughput at your latency target*.
#
# ## 12.2 · A dynamic batcher
#
# Requests arrive one at a time from many clients. A **dynamic batcher** sits between the
# request handlers and the model:
#
# 1. each handler puts `(input, future)` on a queue and `await`s its future;
# 2. a single background task takes the first waiting request, then keeps collecting until it
#    has `max_batch_size` requests **or** `max_wait_ms` has passed since that first request;
# 3. it runs the model **once** on the batch and sets each future's result: every caller gets
#    *its own* answer.
#
# `max_batch_size` bounds the cost of one call (and memory); `max_wait_ms` bounds the extra
# latency a request can suffer while waiting for company. Under heavy load batches fill
# instantly and the wait never triggers; under light load the wait is pure added latency.
# Triton Inference Server, TorchServe and Ray Serve all implement this.

# %% [markdown]
# > **Exercise 1.** Complete `DynamicBatcher`. `submit(item)` must enqueue the item with a fresh
# > future (`asyncio.get_running_loop().create_future()`) and return the awaited result. `run()` loops
# > forever: await the first item, then collect more with `asyncio.wait_for(queue.get(), remaining)` until
# > the batch has `max_batch_size` items or `max_wait_ms` has elapsed since the first item; call
# > `self.process_batch(items)` once (it returns one result per item, in order) and set each future's
# > result. Create the `asyncio.Queue` lazily inside the running loop (the helper `_q` does it). The grader
# > runs it in its own event loop and checks batch sizes, the wait and that results reach the right caller.

# %%
class DynamicBatcher:
    def __init__(self, process_batch, max_batch_size=16, max_wait_ms=5.0):
        self.process_batch = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = None

    def _q(self):
        if self.queue is None:
            self.queue = asyncio.Queue()
        return self.queue

    async def submit(self, item):
        # YOUR CODE HERE
        raise NotImplementedError

    async def run(self):
        # YOUR CODE HERE
        raise NotImplementedError


grade.ex1(DynamicBatcher)

# %% [markdown]
# The course's version, `scaleml.serving.DynamicBatcher`, is the same idea plus two
# production details: a synchronous `process_batch` runs in a worker thread
# (`asyncio.to_thread`) so the event loop keeps accepting requests while the model computes,
# and an exception in one batch is delivered to *that batch's* callers instead of killing the
# loop. The server below uses it.
#
# ## 12.3 · A FastAPI endpoint with dynamic batching
#
# The batcher's background task is started in the app's *lifespan* (it must live in the
# server's event loop). Each `/predict` handler just awaits `batcher.submit(...)`. We run
# uvicorn on `127.0.0.1` in a background thread, so this notebook can be the client too.

# %%
from fastapi import FastAPI
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: list[float]


def predict_batch(items):
    x = torch.tensor(items, dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        return mlp(x).argmax(-1).tolist()


batcher = S.DynamicBatcher(predict_batch, max_batch_size=32, max_wait_ms=2.0)


@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(batcher.run())  # the batching loop lives in uvicorn's event loop
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(req: PredictRequest):
    if len(req.features) != IN_DIM:
        return {"error": f"expected {IN_DIM} features"}
    return {"label": await batcher.submit(req.features)}


N_THREADS = torch.get_num_threads()
torch.set_num_threads(1)  # leave a core for the server's event loop (and the load generator)
server = S.UvicornThread(app, port=S.free_port()).start()
print("serving at", server.url)
rng = np.random.default_rng(0)
payloads = [{"features": rng.standard_normal(IN_DIM).round(4).tolist()} for _ in range(400)]
want = predict_batch([p["features"] for p in payloads[:5]])
got = [httpx.post(server.url + "/predict", json=p).json()["label"] for p in payloads[:5]]
print("server answers:", got, "| direct model call:", want)

# %% [markdown]
# **Expected:** a localhost URL with a free port, and the server's 5 answers equal to the
# direct model call.
#
# ## 12.4 · Load test: throughput vs latency
#
# A closed-loop load test: `concurrency` clients, each sending its next request as soon as
# the previous answer arrives (`scaleml.serving.load_test`, using `httpx.AsyncClient`), run
# in a **separate process** (`load_test_process`) so that the client does not steal the
# server's CPU time and GIL. We
# compare **no batching** (`max_batch_size=1`) with **dynamic batching** (`max_batch_size=32,
# max_wait_ms=2`) at 1, 4, 16 and 32 concurrent clients, and record throughput, p50 and p95
# latency, and the mean batch size the model actually saw. The model runs on **one** thread
# here, so that the server's event loop and the load generator get the remaining cores (on a
# 2-vCPU Colab machine everything else would starve).

# %%
N_REQ = 300 if DEVICE.type == "cpu" else 1000


def run_config(max_batch, max_wait_ms, concurrency, n=N_REQ):
    batcher.max_batch_size, batcher.max_wait_ms = max_batch, max_wait_ms
    batcher.batch_sizes.clear()
    S.load_test(server.url + "/predict", payloads[:concurrency], concurrency=concurrency)  # warm-up
    batcher.batch_sizes.clear()
    res = S.load_test_process(server.url + "/predict", (payloads * 10)[:n], concurrency=concurrency)
    row = {"max_batch": max_batch, "max_wait_ms": max_wait_ms, **res.summary(),
           "mean_batch": float(np.mean(batcher.batch_sizes)) if batcher.batch_sizes else float("nan")}
    print(f"max_batch {max_batch:3d}, wait {max_wait_ms:4.1f} ms | conc {concurrency:3d} | "
          f"{row['throughput_rps']:7.1f} req/s | p50 {row['p50_ms']:7.1f} ms | p95 {row['p95_ms']:7.1f} ms | "
          f"mean batch {row['mean_batch']:5.1f} | errors {row['errors']}")
    return row


LOAD = []
t0 = time.time()
for max_batch, wait in [(1, 0.0), (32, 2.0)]:
    for conc in (1, 4, 16, 32):
        LOAD.append(run_config(max_batch, wait, conc))
print(f"({time.time() - t0:.0f} s)")

# %% [markdown]
# **Expected (CPU, measured; server and clients share the machine):**
#
# * at concurrency 1 the two configurations are similar (a batch can only ever hold one
#   request); batching adds at most the 2 ms wait;
# * without batching, throughput **plateaus** once the model is saturated, and adding clients
#   only adds queueing: p50 and p95 grow roughly in proportion to the concurrency;
# * with dynamic batching the mean batch size grows with concurrency (our run: ~1 → ~4 → ~15 → ~30),
#   throughput keeps rising, and p95 at 64 clients is several times lower than without batching.
#
# Read it as a **throughput–latency curve**: each configuration traces latency rising with
# load, and batching moves the curve right (more requests per second at the same latency).
# Capacity planning picks the highest load whose p95 (or p99) still meets the SLO.
#
# **The `max_wait_ms` knob.** At low load, waiting cannot fill a batch; it only adds latency.
# At 4 concurrent clients:

# %%
WAIT = [run_config(32, w, 4) for w in (0.0, 2.0, 10.0, 25.0)]

# %% [markdown]
# **Expected (CPU, measured):** as the wait grows from 0 to 25 ms the batches get a bit
# fuller (at most 4 requests can ever be waiting) but p50 grows by roughly the wait, and
# throughput *falls*, because each client waits longer per request. So: set `max_wait_ms` to
# a small fraction of the latency SLO (a few ms), let `max_batch_size` be what the
# hardware can process within the SLO, and let load decide the batch size. Some systems
# adapt the wait to the observed arrival rate.

# %%
server.stop()
torch.set_num_threads(N_THREADS)
print("server stopped:", not server.thread.is_alive())
(OUT / "load_test.json").write_text(json.dumps({"device": str(DEVICE), "batch_curve": BATCH_CURVE,
                                                "load": LOAD, "max_wait_sweep": WAIT}, indent=2))
print("wrote", OUT / "load_test.json")

# %% [markdown]
# **Expected:** `server stopped: True`. Always stop servers you start in a notebook; a
# leftover one keeps its port and a thread busy.
#
# ## 12.5 · Generation: why the KV cache matters
#
# A decoder generates one token per forward pass. Without a cache, step t re-runs the model
# over all t previous tokens; with a **KV cache** (notebook 03) it runs only the newest token and
# reuses the stored keys and values: per-step work drops from O(t) to O(1) for the projections
# and MLP. Generation then splits into **prefill** (the prompt, processed in one parallel pass,
# compute-bound) and **decode** (one token at a time, memory-bound: every step reads all the
# weights and the whole cache for a single token per sequence).

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

GPT2_ID, GPT2_REV = "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
print(f"model card: https://huggingface.co/{GPT2_ID}")
gtok = AutoTokenizer.from_pretrained(GPT2_ID, revision=GPT2_REV)
gtok.pad_token = gtok.eos_token
gtok.padding_side = "left"  # decoder-only models generate after the prompt: pad on the left
gpt2 = AutoModelForCausalLM.from_pretrained(GPT2_ID, revision=GPT2_REV, use_safetensors=True).to(DEVICE).eval()
if DEVICE.type == "cuda":
    gpt2 = gpt2.half()

NEW_TOKENS = 32 if DEVICE.type == "cpu" else 128
prompts = ["The key to serving large language models efficiently is",
           "In distributed training, the all-reduce operation",
           "A good machine learning engineer always",
           "The history of the transformer architecture began",
           "To reduce inference latency, we can",
           "Batching requests on a GPU improves",
           "The cost of running a model in production depends on",
           "Quantisation reduces the memory footprint of a model by"]


def gen_speed(batch_prompts, use_cache, new_tokens=NEW_TOKENS):
    enc = gtok(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        gpt2.generate(**enc, max_new_tokens=2, do_sample=False, use_cache=use_cache, pad_token_id=gtok.eos_token_id)
        sync(DEVICE)
        t0 = time.perf_counter()
        out = gpt2.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                            use_cache=use_cache, pad_token_id=gtok.eos_token_id)
        sync(DEVICE)
        dt = time.perf_counter() - t0
    n_new = (out.shape[1] - enc["input_ids"].shape[1]) * len(batch_prompts)
    return {"batch": len(batch_prompts), "use_cache": use_cache, "seconds": dt, "tokens_per_s": n_new / dt,
            "text": gtok.decode(out[0, enc["input_ids"].shape[1]:])}


GEN = []
for batch_prompts, use_cache in [(prompts[:1], False), (prompts[:1], True), (prompts, True)]:
    r = gen_speed(batch_prompts, use_cache)
    GEN.append({k: v for k, v in r.items() if k != "text"})
    print(f"batch {r['batch']}, use_cache={str(use_cache):5} -> {r['tokens_per_s']:7.1f} tokens/s "
          f"({r['seconds']:.2f} s for {NEW_TOKENS} new tokens per sequence)")
print("sample (greedy):", repr(r["text"][:120]))
_ = (OUT / "generation.json").write_text(json.dumps({"device": str(DEVICE), "new_tokens": NEW_TOKENS,
                                                    "runs": GEN}, indent=2))

# %% [markdown]
# **Expected (CPU, 2 threads, measured):** with the cache, batch 1 generates a few tens of
# tokens per second; without it, noticeably fewer (our run: see the printout; the gap grows with
# the number of generated tokens, since the uncached step cost grows with length); batch 8
# with the cache produces several times more tokens per second than batch 1, because decode is
# memory-bound and the weights are read once for all 8 sequences. Greedy GPT-2 text is
# repetitive; that is the model, not a bug. **On a T4 you should see** hundreds of tokens/s
# at batch 1 in fp16 and thousands at batch 8+ (not measured in our verification runs).
#
# ## 12.6 · Static vs continuous batching
#
# With **static batching**, a batch of sequences starts together and the batch is busy
# until its *longest* sequence finishes; slots whose sequence already ended do nothing
# (padding), and new requests wait for the whole batch. **Continuous** (iteration-level)
# batching, introduced by Orca (Yu et al. 2022) and used by vLLM, TGI and TensorRT-LLM,
# re-forms the batch at **every decode step**: a finished sequence leaves, and a waiting
# request takes its slot immediately. Output lengths vary a lot in practice (a "yes" vs a
# 500-token essay), which is exactly when static batching wastes most.
#
# A simulation on 512 requests with skewed output lengths (lognormal, median ~60 tokens,
# capped at 1,024), batch size 16, counting decode steps:

# %%
rng = np.random.default_rng(0)
lengths = np.clip(rng.lognormal(mean=np.log(60), sigma=1.0, size=512).astype(int), 1, 1024).tolist()
print(f"output lengths: median {np.median(lengths):.0f}, mean {np.mean(lengths):.0f}, max {max(lengths)}")
SIM = {}
for name, fn in [("static", S.simulate_static_batching), ("continuous", S.simulate_continuous_batching)]:
    r = fn(lengths, batch_size=16)
    SIM[name] = r
    print(f"{name:10} batching: {r['steps']:6,} decode steps | slot utilisation {r['utilization']:.0%}")
print(f"continuous batching finishes the same work in {SIM['static']['steps'] / SIM['continuous']['steps']:.1f}x "
      "fewer steps")
sorted_static = S.simulate_static_batching(sorted(lengths), 16)
print(f"(static batching with requests sorted by length - impossible online, lengths are unknown: "
      f"{sorted_static['utilization']:.0%})")

# %% [markdown]
# **Expected (simulation, seeded):** median ~60, mean ~100, max a few hundred tokens; static
# batching keeps only ~35–45% of the slots busy, continuous batching ~95%+, so it needs about
# 2–3× fewer decode steps for the same tokens. Since a decode step costs about the same
# whatever the batch holds (memory-bound), **fewer steps means proportionally more
# throughput**. Sorting by length would help static batching, but output lengths are unknown
# until generation ends.
#
# ## 12.7 · Paged attention: the memory side
#
# Continuous batching needs memory for many sequences' KV caches at once. KV cache size per
# token = 2 (K and V) × layers × KV heads × head dim × bytes:

# %%
configs = {"GPT-2 small (12 L, 12 heads x 64)": (12, 12, 64),
           "Llama-2-7B-shaped (32 L, 32 heads x 128)": (32, 32, 128)}
for name, (layers, heads, hd) in configs.items():
    per_tok = S.kv_cache_bytes(layers, heads, hd, 1, bytes_per=2)
    print(f"{name:42} fp16: {per_tok / 1024:6.0f} KB/token | 2,048-token sequence: "
          f"{S.kv_cache_bytes(layers, heads, hd, 2048) / 1e9:5.2f} GB")

# reserve max length per request (contiguous) vs allocate 16-token blocks on demand (paged)
prompt_len, max_len, block = 200, 1024, 16
used = [prompt_len + n for n in lengths[:64]]
contiguous = 64 * (prompt_len + max_len)
paged = sum(-(-u // block) * block for u in used)
print(f"64 sequences: tokens actually stored {sum(used):,} | reserved contiguously (prompt + max_new={max_len}) "
      f"{contiguous:,} -> {sum(used) / contiguous:.0%} used | paged, {block}-token blocks: {paged:,} -> "
      f"{sum(used) / paged:.0%} used")

# %% [markdown]
# **Expected (arithmetic):** GPT-2 small needs 36 KB per token in fp16; a Llama-2-7B-shaped
# model (32 layers, 32 heads of 128, no grouped-query attention) needs 512 KB per token, i.e.
# ~1 GB for a single 2,048-token sequence: the KV cache, not the weights, limits how many
# sequences fit. Reserving the maximum length per request leaves most of that memory unused
# (our simulation: ~25% used); allocating fixed-size blocks on demand uses ~95%+.
#
# **Paged attention** (vLLM, Kwon et al. 2023) stores each sequence's KV cache in fixed-size
# blocks (e.g. 16 tokens) anywhere in GPU memory, with a per-sequence *block table* (like
# virtual-memory pages), and an attention kernel that reads through the table. Results:
# almost no fragmentation, so far more concurrent sequences per GPU (larger batches → higher
# throughput), and cheap sharing of common prefixes (same system prompt; parallel samples)
# by pointing several block tables at the same blocks. The vLLM paper reports that the
# earlier systems it measured used only ~20–40% of their KV-cache memory for actual tokens.
# Not installed here (it needs a CUDA GPU); explained, not measured. Grouped-query attention
# (fewer KV heads) and KV-cache quantisation shrink the per-token cost itself.
#
# ## 12.8 · Cost per 1,000 requests
#
# A server that handles `r` requests/s when busy, rented at `p` dollars/hour, but busy only
# a fraction `u` of the time (traffic has peaks; you provision for the peak and pay for the
# troughs) costs `1000 × p / (r × 3600 × u)` dollars per 1,000 requests.

# %% [markdown]
# > **Exercise 2.** Write `cost_per_1k(requests_per_second, price_per_hour, utilization)` returning
# > dollars per 1,000 requests.

# %%
def cost_per_1k(requests_per_second, price_per_hour, utilization):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(cost_per_1k)

# %% [markdown]
# Apply it to *our measured* throughputs. **Assumed prices** (illustrative; check your cloud's
# current list price): a 4-vCPU CPU machine at $0.20/hour. Average utilisation 40% (peak
# provisioning). The GPU row is a **what-if** with an assumed price ($0.50/hour for a
# T4-class instance) and an assumed throughput of 10× our best CPU number; we did not
# measure it.

# %%
best_nobatch = max(r["throughput_rps"] for r in LOAD if r["max_batch"] == 1)
best_batch = max(r["throughput_rps"] for r in LOAD if r["max_batch"] > 1)
cases = [("CPU, no batching (measured rps)", best_nobatch, 0.20),
         ("CPU, dynamic batching (measured rps)", best_batch, 0.20),
         ("T4 what-if: 10x best CPU rps (NOT measured)", 10 * best_batch, 0.50)]
for name, rps, price in cases:
    print(f"{name:45} {rps:9.0f} req/s @ ${price:.2f}/h, 40% utilisation -> "
          f"${S.cost_per_1k(rps, price, 0.40):.5f} per 1k requests")

# %% [markdown]
# **Expected:** dynamic batching cuts the cost per request by the same factor as it raised
# throughput (our run: see the printout), with the same machine and price. The levers on cost
# are throughput (batching, quantisation, distillation, better kernels), price (spot/reserved
# instances, cheaper hardware), and utilisation (autoscaling, mixing batch and online
# traffic). For generation, quote cost per 1,000 *tokens*, and remember output tokens cost more
# than prompt tokens (decode is memory-bound, prefill is not).

# %% [markdown]
# > **Exercise 3.** You serve a 7B chat model on GPUs with static batching (batch 8) and the p95 latency
# > is fine, but the cost per conversation is too high and GPU memory is 40% empty at peak. (a) Explain,
# > with the simulations of 12.6 and 12.7, why throughput is low. (b) Propose changes in order (serving
# > engine, batching policy, KV-cache memory, model) and the metric you would watch for each. (c) Product
# > now wants a 2× longer context. What happens to the number of concurrent sequences per GPU, and what can
# > you do about it? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | `RuntimeError: asyncio.run() cannot be called from a running event loop` | calling `asyncio.run` in Jupyter/Colab | use `scaleml.serving.run_async` (a fresh thread) or top-level `await` |
# | `uvicorn did not start ... (port in use?)` | a previous server still running | `server.stop()`, or restart the runtime; `S.free_port()` picks an unused port |
# | Requests hang forever with your batcher | `run()` not started, crashed, or futures never resolved | create the task in the lifespan; wrap `process_batch` in `try/except` and set exceptions on the futures |
# | Batch sizes stay at 1 under load | `process_batch` blocks the event loop (sync model call inside `async`) so requests cannot queue | run it with `asyncio.to_thread` (the course's version does) or in a separate worker process |
# | Load-test numbers vary a lot between runs | client and server share 2–4 CPU cores with the notebook | compare configurations in the same run; take medians; a real test uses separate machines |
# | `generate` warns about `pad_token_id` or right-padding | GPT-2 has no pad token; decoder-only models need left padding | set `pad_token = eos_token` and `padding_side = "left"` (done above) |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Why does batching increase throughput on GPUs (and CPUs)? Explain in terms of memory bandwidth and arithmetic intensity, and say when it stops helping.
# 2. Design a dynamic batcher. Which two parameters does it have, how do you set them from an SLO, and what happens at low and at high load?
# 3. You load-test a model server. Which numbers do you report, why percentiles rather than means, and how do you find the capacity of one replica?
# 4. Explain prefill vs decode in LLM inference. Which is compute-bound, which is memory-bound, and how does the KV cache change the cost of decode?
# 5. What is continuous batching, and why does it beat static batching for generation? What does paged attention add?
# 6. Estimate the cost per 1,000 requests for a service, and name three ways to halve it.
#
# ➡️ **Next:** [`13_ctr_prediction_at_scale.ipynb`](13_ctr_prediction_at_scale.ipynb) — from serving to the models behind ads and feeds: sparse features, embedding tables and calibrated click-through-rate prediction at scale.
