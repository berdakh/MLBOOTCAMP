"""Devices, precision and size budgets: one place that knows what hardware you have.

Level 3 is written for a free Colab **T4** GPU (16 GB, compute capability
7.5: fast fp16 Tensor Cores, **no bf16**). Every notebook also runs on a CPU,
with smaller sizes, so that it can be read, debugged and verified anywhere -
the numbers are then much worse, and the notebook says so.

``budget()`` is the switch: it returns the dataset and model sizes to use for
the current device. Notebooks never hard-code "use 50,000 examples"; they ask
the budget, and print what they got.
"""

from __future__ import annotations

import contextlib
import os
import platform
import time
from dataclasses import asdict, dataclass

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """CUDA if available (or as requested), else Apple MPS, else CPU.

    ``BOOTCAMP_DEVICE=cpu`` forces CPU - useful to reproduce a CPU-only run.
    """
    want = prefer or os.environ.get("BOOTCAMP_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device: torch.device | None = None) -> dict:
    device = device or get_device()
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads()}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        info.update({"gpu": p.name, "gpu_memory_gb": round(p.total_memory / 1e9, 1),
                     "capability": f"{p.major}.{p.minor}", "bf16_supported": torch.cuda.is_bf16_supported()})
    return info


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """The mixed-precision dtype worth using here, or None for full fp32.

    T4 (capability 7.5) -> fp16 with a GradScaler; Ampere+ -> bf16 (no scaler
    needed); CPU -> None (CPU autocast exists but rarely speeds up training).
    """
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device, enabled: bool = True):
    dtype = amp_dtype(device)
    if not enabled or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@dataclass(frozen=True)
class Budget:
    """Sizes appropriate for the current hardware. Read them; don't hard-code."""

    name: str
    train_examples: int
    eval_examples: int
    max_seq_len: int
    batch_size: int
    epochs: int
    hidden: int  # width for from-scratch models
    layers: int

    def as_dict(self) -> dict:
        return asdict(self)


BUDGETS = {
    "gpu": Budget("gpu", train_examples=20_000, eval_examples=4_000, max_seq_len=128,
                  batch_size=64, epochs=2, hidden=256, layers=4),
    "cpu": Budget("cpu", train_examples=1_500, eval_examples=500, max_seq_len=64,
                  batch_size=32, epochs=1, hidden=64, layers=2),
    # Used by automated verification and tests: proves the code runs, nothing more.
    "smoke": Budget("smoke", train_examples=200, eval_examples=100, max_seq_len=32,
                    batch_size=16, epochs=1, hidden=32, layers=1),
}


def budget(device: torch.device | None = None) -> Budget:
    """Pick a budget: ``BOOTCAMP_BUDGET`` env var wins, else GPU vs CPU."""
    forced = os.environ.get("BOOTCAMP_BUDGET")
    if forced:
        return BUDGETS[forced]
    device = device or get_device()
    return BUDGETS["gpu" if device.type == "cuda" else "cpu"]


def seed_everything(seed: int = 0, deterministic: bool = False) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def sync(device: torch.device) -> None:
    """Wait for queued GPU work - required before reading a timer. CUDA is asynchronous."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def timer(device: torch.device, results: dict | None = None, key: str = "seconds"):
    sync(device)
    start = time.perf_counter()
    yield
    sync(device)
    elapsed = time.perf_counter() - start
    if results is not None:
        results[key] = elapsed


def memory_report(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"note": "CUDA memory stats are only available on a GPU"}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(device) / 2**20, 1),
        "reserved_mb": round(torch.cuda.memory_reserved(device) / 2**20, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
    }


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
