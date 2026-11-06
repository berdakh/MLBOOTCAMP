"""Serving models at high throughput: dynamic batching, load tests, generation scheduling, cost.

Introduced in Level 3 notebook 12. Pieces:

* ``DynamicBatcher`` - an ``asyncio`` queue that collects requests until it
  has ``max_batch_size`` of them or the oldest has waited ``max_wait_ms``,
  then runs the model **once** on the whole batch and hands each caller its
  own result. This is what Triton Inference Server, TorchServe and Ray Serve
  call dynamic (or adaptive) batching.
* ``UvicornThread`` - run a FastAPI app on localhost in a background thread
  (so a notebook can start, load-test and stop it).
* ``load_test`` - fire N requests at a fixed concurrency and record latencies.
* ``simulate_static_batching`` / ``simulate_continuous_batching`` - token-level
  schedules for text generation, to compare GPU slot utilisation.
* ``kv_cache_bytes``, ``cost_per_1k`` - the arithmetic behind capacity plans.
* Model compression used by notebook 11 (quantisation & distillation):
  ``quantize_per_channel_int8`` / ``dequantize``, ``fake_quantize_linears``,
  ``kd_loss`` and ``file_size_mb``.

Everything runs on a CPU. GPU numbers in the notebook are reasoning, not
measurements, and say so.
"""

from __future__ import annotations

import asyncio
import inspect
import statistics
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- dynamic batching


class DynamicBatcher:
    """Collect concurrent requests into batches: up to ``max_batch_size``, waiting at most ``max_wait_ms``.

    ``process_batch(items: list) -> list`` is called with one batch and must
    return one result per item, in order. It may be a normal function (it is
    then run in a worker thread, so the event loop keeps accepting requests
    while the model runs) or an ``async`` function.

    Usage (inside a running event loop)::

        batcher = DynamicBatcher(predict_batch, max_batch_size=16, max_wait_ms=5)
        task = asyncio.create_task(batcher.run())
        result = await batcher.submit(x)      # from many concurrent handlers
        ...
        task.cancel()

    The wait starts when the *first* item of a batch arrives, so a lone request
    is delayed by at most ``max_wait_ms`` (plus the model's own time).
    """

    def __init__(self, process_batch: Callable[[list], Sequence], max_batch_size: int = 16,
                 max_wait_ms: float = 5.0):
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.process_batch = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue | None = None
        self.batch_sizes: list[int] = []

    def _q(self) -> asyncio.Queue:
        if self.queue is None:  # created lazily, inside the event loop that uses it
            self.queue = asyncio.Queue()
        return self.queue

    async def submit(self, item):
        """Enqueue one item and wait for its own result."""
        fut = asyncio.get_running_loop().create_future()
        await self._q().put((item, fut))
        return await fut

    async def _collect(self) -> list:
        q = self._q()
        batch = [await q.get()]  # block until there is at least one request
        deadline = time.perf_counter() + self.max_wait_ms / 1000
        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(q.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        while len(batch) < self.max_batch_size and not q.empty():  # take what is already waiting
            batch.append(q.get_nowait())
        return batch

    async def _process(self, items: list):
        if inspect.iscoroutinefunction(self.process_batch):
            return await self.process_batch(items)
        return await asyncio.to_thread(self.process_batch, items)

    async def run(self) -> None:
        """Serve forever (until cancelled): collect a batch, run it, deliver results."""
        while True:
            batch = await self._collect()
            items = [item for item, _ in batch]
            self.batch_sizes.append(len(items))
            try:
                results = await self._process(items)
                if len(results) != len(items):
                    raise RuntimeError(f"process_batch returned {len(results)} results for {len(items)} items")
            except Exception as exc:  # one bad batch must not kill the server: fail its callers
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            for (_, fut), res in zip(batch, results):
                if not fut.done():  # the caller may have given up (cancelled)
                    fut.set_result(res)


def run_async(coro_fn: Callable[[], object], timeout: float | None = None):
    """Run ``asyncio.run(coro_fn())`` in a fresh thread and return its result.

    Jupyter (and Colab) already run an event loop in the main thread, where
    ``asyncio.run`` is not allowed; a separate thread has none. Works the
    same in a plain script.
    """
    box: dict = {}

    def target():
        try:
            box["result"] = asyncio.run(coro_fn())
        except BaseException as exc:  # re-raised in the caller's thread
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError("coroutine did not finish in time")
    if "error" in box:
        raise box["error"]
    return box.get("result")


# --------------------------------------------------------------------------- a localhost server


class UvicornThread:
    """Run ``app`` with uvicorn on 127.0.0.1 in a daemon thread; ``stop()`` shuts it down cleanly."""

    def __init__(self, app, port: int = 8012, log_level: str = "warning"):
        import uvicorn

        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level=log_level,
                                                    access_log=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 20.0) -> UvicornThread:
        self.thread.start()
        start = time.time()
        while not self.server.started:
            if time.time() - start > timeout or not self.thread.is_alive():
                raise RuntimeError(f"uvicorn did not start on port {self.port} (port in use?)")
            time.sleep(0.05)
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self.server.should_exit = True
        self.thread.join(timeout)


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- load testing


@dataclass
class LoadResult:
    concurrency: int
    n_requests: int
    seconds: float
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def throughput(self) -> float:
        return (self.n_requests - self.errors) / self.seconds

    def percentile(self, q: float) -> float:
        return percentile(self.latencies_ms, q)

    def summary(self) -> dict:
        return {"concurrency": self.concurrency, "requests": self.n_requests, "errors": self.errors,
                "throughput_rps": round(self.throughput, 1), "p50_ms": round(self.percentile(50), 1),
                "p95_ms": round(self.percentile(95), 1), "p99_ms": round(self.percentile(99), 1)}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), without numpy."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


async def _load(url: str, payloads: Sequence[dict], concurrency: int, timeout: float) -> LoadResult:
    import httpx

    latencies: list[float] = []
    errors = 0
    next_i = 0

    async def worker(client):
        nonlocal next_i, errors
        while next_i < len(payloads):
            i = next_i
            next_i += 1
            t0 = time.perf_counter()
            try:
                r = await client.post(url, json=payloads[i])
                r.raise_for_status()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        start = time.perf_counter()
        await asyncio.gather(*(worker(client) for _ in range(concurrency)))
        seconds = time.perf_counter() - start
    return LoadResult(concurrency, len(payloads), seconds, latencies, errors)


def load_test(url: str, payloads: Sequence[dict], concurrency: int, timeout: float = 30.0) -> LoadResult:
    """POST every payload to ``url`` with ``concurrency`` requests in flight (closed loop)."""
    return run_async(lambda: _load(url, payloads, concurrency, timeout))


# --------------------------------------------------------------------------- generation scheduling


def simulate_static_batching(output_lengths: Sequence[int], batch_size: int) -> dict:
    """Requests are grouped ``batch_size`` at a time, in arrival order; a batch runs until its longest finishes.

    Each decode step processes one token for every *slot* in the batch; a
    slot whose request already finished is wasted (padding). Returns steps,
    useful token-slots and utilisation = useful / (steps x batch_size).
    """
    steps = 0
    for i in range(0, len(output_lengths), batch_size):
        steps += max(output_lengths[i:i + batch_size])
    useful = sum(output_lengths)
    return {"steps": steps, "useful": useful, "utilization": useful / (steps * batch_size)}


def simulate_continuous_batching(output_lengths: Sequence[int], batch_size: int) -> dict:
    """Iteration-level scheduling (Orca, vLLM): a finished request's slot is refilled at the next step.

    Prefill is ignored (it is counted in neither schedule). Returns the same
    keys as ``simulate_static_batching``.
    """
    waiting = list(output_lengths)
    running: list[int] = []
    steps = 0
    while waiting or running:
        while len(running) < batch_size and waiting:
            running.append(waiting.pop(0))
        steps += 1
        running = [r - 1 for r in running if r > 1]
    useful = sum(output_lengths)
    return {"steps": steps, "useful": useful, "utilization": useful / (steps * batch_size)}


def kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, tokens: int, bytes_per: int = 2) -> int:
    """KV cache size: 2 (K and V) x layers x kv_heads x head_dim x tokens x bytes."""
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per


# --------------------------------------------------------------------------- cost


def cost_per_1k(requests_per_second: float, price_per_hour: float, utilization: float) -> float:
    """Dollars per 1,000 requests for one server.

    The server can do ``requests_per_second`` when busy, but is busy only a
    fraction ``utilization`` of the hours you pay for (traffic is not flat;
    you provision for the peak). Requests served per paid hour =
    ``requests_per_second x 3600 x utilization``.
    """
    if requests_per_second <= 0 or not 0 < utilization <= 1:
        raise ValueError("need requests_per_second > 0 and 0 < utilization <= 1")
    return 1000 * price_per_hour / (requests_per_second * 3600 * utilization)


def latency_stats(latencies_ms: Sequence[float]) -> dict:
    return {"mean_ms": statistics.fmean(latencies_ms), "p50_ms": percentile(latencies_ms, 50),
            "p95_ms": percentile(latencies_ms, 95)}


# --------------------------------------------------------------------------- compression (notebook 11)


def quantize_per_channel_int8(w, bits: int = 8):
    """Symmetric per-output-channel quantisation of a 2-D weight: returns (q int8, scale per row).

    ``scale = max|w_row| / qmax`` with ``qmax = 2**(bits-1) - 1`` (127 for int8,
    7 for int4); ``q = round(w / scale)`` clamped to ``[-qmax, qmax]``. All-zero
    rows get scale 1 so that nothing divides by zero. Stored in an int8 tensor
    whatever ``bits`` is (real int4 kernels pack two values per byte).
    """
    import torch

    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale[:, None]).clamp(-qmax, qmax).to(torch.int8)
    return q, scale


def dequantize(q, scale):
    """Inverse of ``quantize_per_channel_int8``: ``q * scale`` per row, in fp32."""
    return q.float() * scale.float()[:, None]


def fake_quantize_linears(model, bits: int = 8):
    """Replace every ``nn.Linear`` weight by its quantise->dequantise round trip, in place.

    Simulates *weight-only* quantisation (what LLM int8/int4 loaders do) while
    still computing in floating point, so its accuracy cost can be measured on
    any device. Returns the number of layers changed.
    """
    import torch

    n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                q, s = quantize_per_channel_int8(m.weight.float(), bits)
                m.weight.copy_(dequantize(q, s).to(m.weight.dtype))
                n += 1
    return n


def kd_loss(student_logits, teacher_logits, labels, T: float = 2.0, alpha: float = 0.5):
    """Hinton et al. (2015) distillation loss.

    ``alpha * T^2 * KL(softmax(t/T) || softmax(s/T)) + (1 - alpha) * CE(s, labels)``.
    The ``T^2`` keeps the soft term's gradients on the same scale as the hard
    term's when you change the temperature.
    """
    import torch.nn.functional as F

    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1), F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean") * (T * T)
    return alpha * soft + (1 - alpha) * F.cross_entropy(student_logits, labels)


def file_size_mb(obj, path) -> float:
    """``torch.save`` ``obj`` to ``path`` and return the file size in MB (10^6 bytes)."""
    import os

    import torch

    torch.save(obj, path)
    return os.path.getsize(path) / 1e6
