"""Measuring speed and memory honestly: benchmarks, roofline numbers, memory arithmetic.

Introduced in Level 3 notebooks 01 (GPU fundamentals & profiling) and 02
(training efficiency & memory). Everything here works on a CPU *and* on a
CUDA GPU; GPU-only measurements (``torch.cuda.max_memory_allocated``) are
used only when a GPU is present.

Two rules every function follows:

* **Synchronise before reading a clock.** CUDA kernels are queued and run
  asynchronously; ``scaleml.device.sync`` waits for them.
* **Warm up, repeat, report the median.** The first call pays for kernel
  selection, memory-allocator growth and caches; one timing is noise.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .device import sync

# --------------------------------------------------------------------------- timing


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int = 3, iters: int = 10,
              min_seconds: float = 0.0) -> dict:
    """Median wall time of ``fn()`` on ``device`` (synchronised), after ``warmup`` calls.

    ``min_seconds`` keeps repeating (beyond ``iters``) until that much time was
    measured, so very fast functions still get a stable estimate.
    Returns ``{"median_s", "min_s", "iters"}``.
    """
    for _ in range(warmup):
        fn()
    times: list[float] = []
    total = 0.0
    while len(times) < iters or total < min_seconds:
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        dt = time.perf_counter() - start
        times.append(dt)
        total += dt
    return {"median_s": statistics.median(times), "min_s": min(times), "iters": len(times)}


def matmul_flops(n: int) -> int:
    """FLOPs of an (n x n) @ (n x n) matmul: n^2 outputs, each n multiplies + n adds."""
    return 2 * n**3


def bench_matmul(sizes, device: torch.device, dtype: torch.dtype = torch.float32,
                 warmup: int = 2, iters: int = 5, min_seconds: float = 0.2) -> list[dict]:
    """TFLOP/s and arithmetic intensity of square matmuls of each size."""
    rows = []
    elem = torch.tensor([], dtype=dtype).element_size()
    for n in sizes:
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        r = benchmark(lambda a=a, b=b: a @ b, device, warmup, iters, min_seconds)
        flops, nbytes = matmul_flops(n), 3 * n * n * elem  # read A, B; write C (ideal caching)
        rows.append({"n": n, "dtype": str(dtype).replace("torch.", ""), "ms": r["median_s"] * 1e3,
                     "tflops": flops / r["median_s"] / 1e12, "flops_per_byte": flops / nbytes})
        del a, b
    return rows


def bench_elementwise(numel: int, device: torch.device, dtype: torch.dtype = torch.float32,
                      warmup: int = 2, iters: int = 5, min_seconds: float = 0.2) -> dict:
    """``x + y`` on ``numel`` elements: 1 FLOP per element, 3 elements of memory traffic."""
    x = torch.randn(numel, device=device, dtype=dtype)
    y = torch.randn(numel, device=device, dtype=dtype)
    r = benchmark(lambda: x + y, device, warmup, iters, min_seconds)
    nbytes = 3 * numel * x.element_size()
    return {"numel": numel, "ms": r["median_s"] * 1e3, "gb_per_s": nbytes / r["median_s"] / 1e9,
            "tflops": numel / r["median_s"] / 1e12, "flops_per_byte": numel / nbytes}


# --------------------------------------------------------------------------- input pipeline


class SlowDataset(torch.utils.data.Dataset):
    """A dataset whose ``__getitem__`` sleeps, standing in for JPEG decoding or tokenising.

    Defined in a module (not the notebook) so DataLoader workers can import it
    under every multiprocessing start method (``spawn`` on macOS/Windows).
    """

    def __init__(self, n: int = 256, dim: int = 32, delay_s: float = 0.002):
        self.n, self.dim, self.delay_s = n, dim, delay_s

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        time.sleep(self.delay_s)  # the "expensive" per-sample work, done on the CPU
        g = torch.Generator().manual_seed(i)
        return torch.randn(self.dim, generator=g), torch.tensor(i % 2, dtype=torch.float32)


def time_loader(loader, step_seconds: float = 0.0, max_batches: int | None = None) -> float:
    """Seconds to iterate ``loader``; ``step_seconds`` simulates the model's compute per batch."""
    start = time.perf_counter()
    for b, _ in enumerate(loader):
        if step_seconds:
            time.sleep(step_seconds)
        if max_batches and b + 1 >= max_batches:
            break
    return time.perf_counter() - start


# --------------------------------------------------------------------------- memory arithmetic

#: bytes per parameter for optimizer state (fp32), by optimizer name
OPTIMIZER_STATE_BYTES = {"sgd": 0, "sgd_momentum": 4, "adam": 8, "adamw": 8}


def memory_breakdown(n_params: int, optimizer: str = "adam", mixed_precision: bool = False,
                     activation_bytes: int = 0) -> dict:
    """Training memory (bytes) by category, PyTorch-AMP accounting.

    * weights fp32 (4 B/param) and their gradients fp32 (4 B/param) - with
      ``torch.autocast`` the *parameters stay fp32*; only the computation is fp16;
    * optimizer state fp32: SGD 0, momentum 4, Adam/AdamW 8 (two moments);
    * mixed precision adds the fp16 copy of the weights that autocast caches
      during the forward pass (2 B/param) and roughly halves activations
      (pass the activation estimate you computed for the chosen dtype).

    Excludes the CUDA context (~0.3-0.5 GB), allocator fragmentation and
    temporary workspaces, which is why measured peaks are higher.
    """
    opt = OPTIMIZER_STATE_BYTES[optimizer.lower()]
    out = {"weights": 4 * n_params, "gradients": 4 * n_params, "optimizer": opt * n_params,
           "fp16_weight_copy": 2 * n_params if mixed_precision else 0, "activations": int(activation_bytes)}
    out["total"] = sum(out.values())
    return out


def transformer_activation_bytes(batch: int, seq: int, hidden: int, layers: int, heads: int,
                                 bytes_per_value: int = 4) -> int:
    """Activation memory saved for backward by the decoder *layers* of ``scaleml.tinygpt``.

    Per layer and token, about 16 hidden-sized values (LayerNorm inputs and
    outputs, Q, K, V, the attention output, the residual stream, the 4x-wide
    MLP's GELU input and output) plus one ``heads x seq`` row of attention
    probabilities - the part that grows *quadratically* with sequence length.
    The constant depends on the implementation: Megatron-LM's estimate with
    dropout masks is ``34 h + 5 a s`` bytes per token per layer in fp16, and
    fused/flash attention never stores the ``seq x seq`` probabilities at all.
    Embeddings and the output logits (``batch x seq x vocab``) come on top.
    Treat it as an estimate and *measure* (``SavedActivations``).
    """
    per_layer = batch * seq * (16 * hidden + heads * seq)
    return int(per_layer * layers * bytes_per_value)


def tensor_bytes(tensors) -> int:
    return int(sum(t.numel() * t.element_size() for t in tensors if t is not None))


def measured_state_bytes(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Sum the actual tensors: parameters, their ``.grad`` and the optimizer state."""
    params = list(model.parameters())
    opt_bytes = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            opt_bytes += tensor_bytes(v for v in state.values() if torch.is_tensor(v) and v.dim() > 0)
    return {"weights": tensor_bytes(params), "gradients": tensor_bytes(p.grad for p in params),
            "optimizer": opt_bytes}


@dataclass
class SavedTensorStats:
    bytes: int = 0
    tensors: int = 0


class SavedActivations:
    """Count the bytes autograd *saves for backward* - activation memory, measurable on a CPU.

    Usage::

        with SavedActivations(model) as acts:
            loss = model(x).sum()
        print(acts.stats.bytes)

    Storage is de-duplicated (views of one tensor count once) and storage
    belonging to the model's parameters is excluded: weights are not
    activations. This is what activation checkpointing reduces.
    """

    def __init__(self, model: torch.nn.Module | None = None):
        self._param_ptrs = ({p.untyped_storage().data_ptr() for p in model.parameters()}
                            if model is not None else set())
        self._seen: dict[int, int] = {}
        self.stats = SavedTensorStats()
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)

    def _pack(self, t: torch.Tensor):
        # Return a detached alias: same storage and version counter, but no grad_fn. Returning
        # ``t`` itself would make a saved *output* reference its own graph node - a reference
        # cycle that keeps the whole graph (and its activations) alive after backward.
        t = t.detach()
        st = t.untyped_storage()
        ptr = st.data_ptr()
        if ptr not in self._param_ptrs and ptr not in self._seen:
            self._seen[ptr] = st.nbytes()
            self.stats.bytes += st.nbytes()
            self.stats.tensors += 1
        return t

    def __enter__(self):
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


def cuda_peak_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**20


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"
