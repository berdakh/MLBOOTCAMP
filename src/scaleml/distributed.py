"""Distributed training by arithmetic and by hand: collectives, samplers, sharding, budgets.

Introduced in Level 3 notebooks 09 (distributed data parallel) and 10
(scaling to large models). Two kinds of helpers live here:

* **Simulations you can check against PyTorch** - averaging gradients the way
  an all-reduce does, the index sharding of ``DistributedSampler``, a Linear
  layer split across "devices" (tensor parallelism). They run in one process
  on a CPU and are compared, in the notebooks, with the real thing.
* **Calculators** - communication bytes of a ring all-reduce, memory per GPU
  under DDP and ZeRO/FSDP, training FLOPs and GPU-days, pipeline bubbles,
  checkpoint intervals. Every formula is stated in its docstring, together with
  the assumptions it makes, because in an interview (and in a capacity plan)
  the assumptions matter more than the digits.

Nothing here needs a GPU. Multi-GPU numbers in the notebooks are produced by
these calculators, not measured.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# --------------------------------------------------------------------------- collectives


def ring_allreduce_bytes(n_params: int, bytes_per: int, world_size: int) -> float:
    """Bytes each rank SENDS (and receives) in a ring all-reduce of ``n_params`` values.

    A ring all-reduce is a reduce-scatter (N-1 steps) followed by an all-gather
    (N-1 steps); in every step each rank sends one chunk of size ``S / N``,
    where ``S = n_params * bytes_per``. Total per rank: ``2 (N-1)/N * S``,
    which approaches ``2 S`` as N grows - the per-GPU traffic does *not* grow
    with the number of GPUs, which is why the ring is bandwidth-optimal.
    One rank (``world_size == 1``) communicates nothing.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return 2 * (world_size - 1) / world_size * n_params * bytes_per


def allreduce_seconds(n_params: int, bytes_per: int, world_size: int, bandwidth_gb_s: float,
                      latency_s: float = 0.0) -> float:
    """Time of one ring all-reduce: per-rank bytes / link bandwidth (+ 2(N-1) latency hops).

    ``bandwidth_gb_s`` is the *per-direction* bandwidth of the slowest link in
    the ring, in GB/s (1e9 bytes/s). A lower bound: real NCCL reaches 70-90 %
    of it for large messages.
    """
    hops = 2 * (world_size - 1)
    return ring_allreduce_bytes(n_params, bytes_per, world_size) / (bandwidth_gb_s * 1e9) + hops * latency_s


def ddp_step_seconds(compute_s: float, comm_s: float, overlap: float = 0.0) -> float:
    """Wall time of a DDP step when a fraction ``overlap`` of communication hides behind backward.

    DDP buckets gradients and starts all-reducing a bucket as soon as backward
    has produced it, so part of the communication runs while backward is still
    computing. ``overlap=0``: fully serial; ``overlap=1``: perfectly hidden
    (communication still cannot finish before compute does, hence the max).
    """
    return max(compute_s, compute_s + (1 - overlap) * comm_s) if overlap < 1 else max(compute_s, comm_s)


def scaling_efficiency(compute_s: float, comm_s: float, overlap: float = 0.0) -> float:
    """Throughput of N data-parallel workers / (N x one worker) = compute / step time."""
    return compute_s / ddp_step_seconds(compute_s, comm_s, overlap)


def average_gradients(grads_per_rank: Sequence[Sequence]) -> list:
    """What ``all_reduce(SUM)`` followed by ``/ world_size`` leaves on every rank.

    ``grads_per_rank[r][i]`` is rank r's gradient for parameter i. Returns a
    new list of tensors (one per parameter); inputs are not modified.
    """
    import torch

    world = len(grads_per_rank)
    if world == 0:
        raise ValueError("need at least one rank")
    n = len(grads_per_rank[0])
    if any(len(g) != n for g in grads_per_rank):
        raise ValueError("every rank must hold the same number of gradients")
    return [torch.stack([g[i] for g in grads_per_rank]).sum(0) / world for i in range(n)]


def shard_indices(n: int, world_size: int, rank: int, epoch: int = 0, seed: int = 0,
                  shuffle: bool = True) -> list[int]:
    """The indices ``torch.utils.data.DistributedSampler`` gives ``rank`` (``drop_last=False``).

    1. shuffle with a generator seeded by ``seed + epoch`` (that is why you must
       call ``sampler.set_epoch(epoch)``: otherwise every epoch has the same order);
    2. pad by repeating indices from the start until the length is divisible by
       ``world_size`` (so some examples are seen twice per epoch);
    3. rank r takes every ``world_size``-th index starting at r.
    """
    import torch

    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        indices = torch.randperm(n, generator=g).tolist()
    else:
        indices = list(range(n))
    total = math.ceil(n / world_size) * world_size
    pad = total - n
    if pad:
        indices += (indices * math.ceil(pad / n))[:pad]
    return indices[rank:total:world_size]


# --------------------------------------------------------------------------- launching


def torchrun(script: Path | str, nproc: int, args: Sequence[str] = (), max_restarts: int = 0,
             timeout: float = 300, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``torchrun --standalone --nproc_per_node=nproc script args`` and capture its output.

    Uses ``python -m torch.distributed.run`` (what the ``torchrun`` command is),
    so it works even when the ``torchrun`` script is not on ``PATH``. Each worker
    gets ``OMP_NUM_THREADS=1`` unless ``env`` says otherwise, so N processes do
    not fight over the cores.
    """
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}",
           f"--max-restarts={max_restarts}", str(script), *map(str, args)]
    full_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(env or {})}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=full_env)


# --------------------------------------------------------------------------- memory: DDP vs ZeRO


def zero_memory_per_gpu(n_params: int, world_size: int, stage: int, k_optimizer: int = 12) -> float:
    """Model-state bytes per GPU for mixed-precision Adam under DDP (stage 0) and ZeRO 1/2/3.

    The ZeRO paper's accounting (Rajbhandari et al., 2020), with ``Psi =
    n_params`` and ``N = world_size``:

    * fp16/bf16 parameters ``2 Psi``, fp16/bf16 gradients ``2 Psi``;
    * optimizer state ``K Psi`` with ``K = 12``: an fp32 master copy of the
      weights (4) and Adam's two fp32 moments (4 + 4).

    =========  =================================  ==============
    stage      what is sharded across N GPUs      bytes / GPU
    =========  =================================  ==============
    0 (DDP)    nothing                            (2+2+K) Psi
    1          optimizer state                    (2+2) Psi + K Psi / N
    2          + gradients                        2 Psi + (2+K) Psi / N
    3 (FSDP)   + parameters                       (2+2+K) Psi / N
    =========  =================================  ==============

    Activations, temporary buffers (stage 3 gathers one layer's full weights at
    a time), the CUDA context and fragmentation come on top.
    """
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2 or 3")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    p, g, o = 2.0 * n_params, 2.0 * n_params, float(k_optimizer) * n_params
    if stage >= 1:
        o /= world_size
    if stage >= 2:
        g /= world_size
    if stage >= 3:
        p /= world_size
    return p + g + o


def zero_table(n_params: int, world_sizes: Sequence[int] = (1, 8, 64)) -> list[dict]:
    """GB per GPU for stages 0-3 at several world sizes (rows: dicts, ready for a DataFrame)."""
    rows = []
    for n in world_sizes:
        row = {"gpus": n}
        for s in range(4):
            row[f"stage{s}_GB"] = zero_memory_per_gpu(n_params, n, s) / 1e9
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- compute budgets


def train_flops(n_params: float, tokens: float) -> float:
    """Training FLOPs of a dense transformer ~ 6 N D (Kaplan et al., 2020).

    2 N FLOPs per token for the forward pass (a multiply and an add per
    weight), 4 N for the backward pass (gradients w.r.t. activations and
    weights). Ignores attention's ``seq^2`` term (small when ``d_model`` >>
    ``seq / 6``) and activation recomputation (+2 N per token if you checkpoint
    everything).
    """
    return 6.0 * n_params * tokens


def train_flops_and_days(n_params: float, tokens: float, gpu_tflops: float, mfu: float,
                         n_gpus: int) -> tuple[float, float]:
    """(total training FLOPs, wall-clock days on ``n_gpus``) at model-FLOPs utilisation ``mfu``.

    ``days = 6 N D / (n_gpus * gpu_tflops * 1e12 * mfu) / 86400``. GPU-days are
    ``days * n_gpus``. ``gpu_tflops`` is the datasheet peak for the precision
    you train in (dense, no sparsity); ``mfu`` (0-1) is the fraction of it
    your job achieves - 30-50 % is typical for well-tuned large-model training.
    """
    if not 0 < mfu <= 1:
        raise ValueError("mfu must be in (0, 1]")
    flops = train_flops(n_params, tokens)
    seconds = flops / (n_gpus * gpu_tflops * 1e12 * mfu)
    return flops, seconds / 86_400


def chinchilla_optimal(compute_flops: float, tokens_per_param: float = 20.0) -> tuple[float, float]:
    """(params, tokens) that spend ``compute_flops`` with D = 20 N (Hoffmann et al., 2022 rule of thumb).

    From ``C = 6 N D`` and ``D = r N``: ``N = sqrt(C / (6 r))``, ``D = r N``.
    """
    n = math.sqrt(compute_flops / (6.0 * tokens_per_param))
    return n, tokens_per_param * n


def pipeline_bubble_fraction(stages: int, microbatches: int) -> float:
    """Idle fraction of a GPipe / 1F1B pipeline schedule: (p - 1) / (m + p - 1)."""
    if stages < 1 or microbatches < 1:
        raise ValueError("stages and microbatches must be >= 1")
    return (stages - 1) / (microbatches + stages - 1)


# --------------------------------------------------------------------------- fault tolerance


def young_interval(checkpoint_cost_s: float, mtbf_s: float) -> float:
    """Young's (1974) optimal time between checkpoints: sqrt(2 C M)."""
    return math.sqrt(2.0 * checkpoint_cost_s * mtbf_s)


def daly_interval(checkpoint_cost_s: float, mtbf_s: float) -> float:
    """Daly's (2006) higher-order refinement, valid for C < 2M: sqrt(2 C M) - C (simplified form)."""
    if checkpoint_cost_s >= 2 * mtbf_s:
        return mtbf_s
    return math.sqrt(2.0 * checkpoint_cost_s * mtbf_s) - checkpoint_cost_s


def expected_waste_fraction(interval_s: float, checkpoint_cost_s: float, mtbf_s: float,
                            restart_s: float = 0.0) -> float:
    """First-order fraction of wall time lost: writing checkpoints + redoing lost work.

    ``C / tau`` (time spent writing) ``+ (tau / 2 + R) / M`` (on each failure,
    on average half an interval of work is lost, plus restart time ``R``).
    Minimised at ``tau = sqrt(2 C M)`` - Young's formula.
    """
    return checkpoint_cost_s / interval_s + (interval_s / 2 + restart_s) / mtbf_s


def cluster_mtbf(per_node_mtbf_s: float, n_nodes: int) -> float:
    """Independent failures: a job on n nodes fails n times as often as one node."""
    return per_node_mtbf_s / n_nodes


# --------------------------------------------------------------------------- tensor parallelism


def column_parallel_linear(x, weight, bias=None, n_shards: int = 2):
    """y = x W^T + b with W's *output* rows split across ``n_shards`` "devices".

    Each shard computes a slice of the output features from the full input;
    concatenating the slices (an all-gather) gives the full output. This is
    the first Linear of a Megatron-style MLP.
    """
    import torch

    w_shards = weight.chunk(n_shards, dim=0)
    b_shards = bias.chunk(n_shards) if bias is not None else [None] * n_shards
    outs = [torch.nn.functional.linear(x, w, b) for w, b in zip(w_shards, b_shards)]
    return torch.cat(outs, dim=-1), outs


def row_parallel_linear(x, weight, bias=None, n_shards: int = 2):
    """y = x W^T + b with W's *input* columns (and x's features) split across shards.

    Each shard multiplies its slice of the input features by its slice of W
    and produces a *partial sum* of the full output; summing them (an
    all-reduce) gives the result. The bias is added once, after the sum.
    """
    import torch

    x_shards = x.chunk(n_shards, dim=-1)
    w_shards = weight.chunk(n_shards, dim=1)
    partial = [torch.nn.functional.linear(xs, ws) for xs, ws in zip(x_shards, w_shards)]
    y = torch.stack(partial).sum(0)
    return (y + bias if bias is not None else y), partial
