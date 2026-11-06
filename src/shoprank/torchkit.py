"""Small PyTorch helpers for Level 2 (introduced in notebook 09).

Nothing here is clever. Each helper is a few lines you could write yourself,
and notebook 09 writes most of them by hand first. They live in one place so
notebook 10 (and any later notebook) does not copy-paste a training loop with a
slightly different bug in each copy.

* :func:`seed_everything`, :func:`seed_worker`, :func:`make_generator`,
  :func:`set_deterministic` - reproducibility, including DataLoader workers.
* :class:`TensorBatchDataset` - a map-style dataset that returns whole
  batches (fast for tabular data); :class:`ParquetDayStream` - an iterable
  dataset that shards its files across workers.
* :func:`train_epoch`, :func:`evaluate` - the loop, with optional gradient
  clipping, a per-step scheduler and gradient-norm logging.
* :func:`warmup_cosine` - linear warmup then cosine decay, per optimizer step.
* :func:`save_checkpoint`, :func:`load_checkpoint` - model, optimizer,
  scheduler AND every RNG state, so a resumed run equals an uninterrupted one.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, IterableDataset, get_worker_info

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and every CUDA device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for DataLoader: derive NumPy/random seeds from torch's.

    PyTorch seeds each worker's *torch* RNG from the loader's generator, but
    not NumPy's or Python's. Without this, NumPy augmentations in workers can
    repeat across workers (and, with older versions, across epochs).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A dedicated generator for shuffling, so data order does not depend on
    how many *other* random numbers the program happened to draw."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def set_deterministic(enabled: bool = True) -> None:
    """Ask PyTorch for deterministic kernels (matters mostly on GPU).

    On CUDA you must also set ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` *before* the
    first CUDA call; ``warn_only=True`` warns instead of raising for ops that
    have no deterministic implementation.
    """
    torch.use_deterministic_algorithms(enabled, warn_only=True)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def state_fingerprint(model: nn.Module) -> str:
    """Short hash of every parameter and buffer - equal fingerprints mean
    bitwise-identical weights."""
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class TensorBatchDataset(Dataset):
    """Map-style dataset whose ``__getitem__`` takes a *list of indices*.

    Use it with ``DataLoader(ds, sampler=BatchSampler(...), batch_size=None)``:
    one Python call per batch instead of one per row, which is 10-100x faster
    for small tabular rows where per-row overhead dominates.
    """

    def __init__(self, *tensors: torch.Tensor):
        n = len(tensors[0])
        assert all(len(t) == n for t in tensors), "all tensors need the same first dimension"
        self.tensors = tensors

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


class ParquetDayStream(IterableDataset):
    """Iterable dataset streaming rows from a list of Parquet files.

    Each worker reads a disjoint subset of the files (``files[worker_id::n]``).
    Set ``shard=False`` to see the classic bug: every worker streams every
    file, so each row arrives ``num_workers`` times.
    """

    def __init__(self, files, columns, batch_rows: int = 4096, shard: bool = True):
        self.files, self.columns = [str(f) for f in files], list(columns)
        self.batch_rows, self.shard = batch_rows, shard

    def __iter__(self):
        import pyarrow.parquet as pq

        info = get_worker_info()
        files = self.files
        if self.shard and info is not None:
            files = files[info.id::info.num_workers]
        for path in files:
            table = pq.read_table(path, columns=self.columns)
            for batch in table.to_batches(max_chunksize=self.batch_rows):
                yield {c: torch.from_numpy(batch.column(c).to_numpy(zero_copy_only=False))
                       for c in self.columns}


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def grad_norm(model: nn.Module) -> float:
    """Global L2 norm of all gradients (what ``clip_grad_norm_`` measures)."""
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    return float(torch.norm(torch.stack(norms), 2)) if norms else 0.0


def train_epoch(model, loader, optimizer, loss_fn, scheduler=None, clip: float | None = None,
                device="cpu", log: list | None = None) -> float:
    """One pass over ``loader``; returns the mean loss per example.

    ``loader`` yields ``(*inputs, target)`` tuples. ``scheduler`` (if given)
    is stepped once per optimizer step. ``log`` (if given) receives one dict
    per step with the loss, the pre-clipping gradient norm and the lr.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        *inputs, y = (t.to(device) for t in batch)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(*inputs), y)
        loss.backward()
        if clip is not None:
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        else:
            norm = grad_norm(model) if log is not None else float("nan")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if log is not None:
            log.append({"loss": loss.item(), "grad_norm": norm, "lr": optimizer.param_groups[0]["lr"]})
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cpu") -> float:
    """Mean loss per example in eval mode (dropout off), without autograd."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        *inputs, y = (t.to(device) for t in batch)
        total += loss_fn(model(*inputs), y).item() * len(y)
        count += len(y)
    model.train(was_training)
    return total / max(count, 1)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    """LambdaLR: linear warmup from ~0 to the base lr, then cosine to ``min_ratio`` x base."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(path, model, optimizer=None, scheduler=None, generator: torch.Generator | None = None,
                    **extra) -> Path:
    """Save everything needed to continue *exactly*: weights, optimizer moments,
    scheduler position, and the Python/NumPy/torch/shuffle-generator RNG states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "generator": generator.get_state() if generator is not None else None,
        },
        "extra": extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a half-written checkpoint
    return path


def load_checkpoint(path, model, optimizer=None, scheduler=None, generator: torch.Generator | None = None) -> dict:
    """Restore what :func:`save_checkpoint` saved; returns the ``extra`` dict."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if generator is not None and rng["generator"] is not None:
        generator.set_state(rng["generator"])
    return state["extra"]
