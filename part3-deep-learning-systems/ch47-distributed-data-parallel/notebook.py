# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 47 · Distributed data parallel
#
# **Day 5 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget; the distributed parts use CPU processes either way)**
#
# Every large model is trained on many GPUs, and the most common way to use
# them is *data parallelism*: each GPU holds a full copy of the model, sees a
# different slice of each batch, and the gradients are averaged before every
# optimizer step. "What does DDP actually do?", "why is my 8-GPU job only 5×
# faster?" and "what happens when a node dies?" are standard ML-engineering
# interview questions, and this notebook answers them with real
# `torch.distributed` code.
#
# Free Colab has one GPU, so we run **several processes on one machine** with
# the `gloo` backend (CPU tensors). The code is the code you would run on a
# cluster: on GPUs you change the backend to `nccl`, put each process on its
# own device, and launch on more machines. Speed-ups across GPUs are
# explained with arithmetic, not measured.
#
# By the end you will be able to:
#
# 1. launch a multi-process job with `torchrun` and explain rank, local rank, world size and the process group;
# 2. use the collectives (`all_reduce`, `broadcast`, `all_gather`, `barrier`) by hand;
# 3. show numerically that averaging gradients across ranks (by hand or with `DistributedDataParallel`) equals single-process training on the combined batch;
# 4. shard data with `DistributedSampler`, and explain the `set_epoch` bug and the padding it adds;
# 5. log and checkpoint from rank 0 only, and recover from a dead rank by resuming from a checkpoint (what `torchrun --max-restarts` automates);
# 6. estimate communication time (ring all-reduce ≈ 2(N−1)/N × gradient bytes) and say when DDP stops scaling.
#
# **You will produce:** generated scripts under `generated/nb09/` (a collectives demo and a
# data-parallel training script you can reuse), their outputs (per-rank parameters, sampler
# indices, checkpoints), and `artifacts/level3/nb09/summary.json`. You will use
# [`src/scaleml/distributed.py`](../../src/scaleml/distributed.py).
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*. Multi-GPU numbers are
# arithmetic with stated assumptions, never measurements.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

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
import importlib.util
import json
import shutil
import time

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from scaleml import distributed as D
from grading import ch47 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb09"
OUT.mkdir(parents=True, exist_ok=True)
GEN = ROOT / "generated" / "nb09"  # scripts and their outputs (gitignored)
GEN.mkdir(parents=True, exist_ok=True)
print("torch.distributed available:", dist.is_available(), "| gloo:", dist.is_gloo_available(),
      "| nccl:", dist.is_nccl_available())
print("CPU cores:", os.cpu_count())


def show(proc, keep=("rank", "epoch", "resumed", "attempt", "bandwidth", "size", "sum", "max",
                     "broadcast", "gather", "barrier", "env")):
    """Print the interesting lines a torchrun job wrote (drop gloo/socket chatter)."""
    noise = ("[Gloo]", "socket.cpp", "OMP_NUM_THREADS", "*****")
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.strip() and not any(n in line for n in noise) and any(k in line for k in keep):
            print(line)
    print(f"-> exit code {proc.returncode}")

# %% [markdown]
# **Expected:** `torch.distributed available: True | gloo: True`. `nccl` reports whether
# PyTorch was *built* with NCCL: our CUDA-enabled build printed `True` even though this
# machine has no GPU, and a CPU-only build prints `False`. Collectives on NCCL still need GPUs.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Rank / world size | a process's unique number / how many processes the job has | ranks 0–3, world size 4 |
# | Local rank | a process's index on its own machine; picks which GPU it uses | `torch.cuda.set_device(local_rank)` |
# | Process group / backend | the set of processes that talk to each other / the library that moves the bytes | gloo on CPU, NCCL on GPUs |
# | Collective | an operation every rank must call, in the same order | `all_reduce`, `broadcast`, `all_gather`, `barrier` |
# | All-reduce | every rank contributes a tensor and every rank receives the sum (or max, …) | sum = 10 for 1 + 2 + 3 + 4 |
# | Reduce-scatter / all-gather | the two halves of a ring all-reduce: sum one chunk each / share every chunk with all | 9.2, 9.6 |
# | DDP | PyTorch's data parallelism: replicas on every rank, gradients averaged during backward | `DistributedDataParallel` in 9.3 |
# | Gradient bucket | a group of gradients all-reduced together as soon as they are ready | ~25 MB, `bucket_cap_mb` |
# | DistributedSampler | gives each rank a disjoint, equal-sized slice of the (shuffled) dataset | `set_epoch(epoch)` reshuffles |
# | torchrun | the launcher that starts one process per GPU and sets RANK, WORLD_SIZE, … | `torchrun --standalone --nproc_per_node=4` |
# | Hang | ranks blocked forever in a collective one rank never reached | an `all_reduce` inside `if rank == 0:` |
# | Straggler | one slow rank that every other rank has to wait for at each collective | a ring is as fast as its slowest link |

# %% [markdown]
# ## 9.1 · Processes, ranks and the process group
#
# Data parallelism uses **one process per GPU**. Each process:
#
# * has a **rank** (0 … world_size − 1, unique across all machines) and a **local rank**
#   (its index on its own machine, which picks the GPU: `torch.cuda.set_device(local_rank)`);
# * joins a **process group** with `dist.init_process_group(backend)`, a rendezvous through
#   a key-value store at `MASTER_ADDR:MASTER_PORT`, after which the processes can run
#   *collectives*: operations every rank must call, in the same order;
# * runs the *same script*; behaviour differs only through `rank` (SPMD: single program,
#   multiple data).
#
# Backends: **NCCL** for GPU tensors (uses NVLink/InfiniBand), **gloo** for CPU tensors (TCP
# or shared memory), MPI rarely. `torchrun` (`python -m torch.distributed.run`) starts the
# processes and sets the environment variables `RANK`, `LOCAL_RANK`, `WORLD_SIZE`,
# `MASTER_ADDR`, `MASTER_PORT`. `--standalone` means "a single machine, pick a free port".
#
# Collectives cannot be called from the notebook's own process alone, so we write a
# script to `generated/nb09/` and launch it with 4 processes.

# %%
COLLECTIVES = '''\
"""Collectives demo. Launch: torchrun --standalone --nproc_per_node=4 collectives.py"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

dist.init_process_group("gloo")  # reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from torchrun
rank, world = dist.get_rank(), dist.get_world_size()
if rank == 0:
    print("env:", {k: os.environ[k] for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR")}, flush=True)
print(f"hello from rank {rank} of {world} (pid {os.getpid()})", flush=True)

t = torch.tensor([rank + 1.0])
dist.all_reduce(t, op=dist.ReduceOp.SUM)  # in place, on every rank
s = t.item()
t = torch.tensor([float(rank)])
dist.all_reduce(t, op=dist.ReduceOp.MAX)
print(f"rank {rank}: all_reduce sum of (rank+1) = {s:.0f}, max of rank = {t.item():.0f}", flush=True)

x = torch.arange(3.0) * 10 if rank == 0 else torch.zeros(3)
dist.broadcast(x, src=0)  # rank 0's tensor overwrites everyone's
gathered = [torch.zeros(1) for _ in range(world)]
dist.all_gather(gathered, torch.tensor([rank * rank * 1.0]))
print(f"rank {rank}: broadcast -> {x.tolist()}, all_gather of rank^2 -> {[g.item() for g in gathered]}", flush=True)

time.sleep(0.3 * rank)  # ranks arrive at different times ...
t0 = time.time()
dist.barrier()  # ... and nobody leaves until all have arrived
print(f"rank {rank}: waited {time.time() - t0:.2f} s at the barrier", flush=True)

# all_reduce bandwidth on this machine (gloo over loopback): message size vs time
results = []
for numel in (2**10, 2**14, 2**18, 2**21, 2**22):
    buf = torch.ones(numel)
    for _ in range(2):
        dist.all_reduce(buf)
    dist.barrier()
    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        dist.all_reduce(buf)
    sec = (time.perf_counter() - t0) / reps
    nbytes = numel * 4
    results.append({"bytes": nbytes, "ms": sec * 1e3, "algbw_gb_s": nbytes / sec / 1e9,
                    "busbw_gb_s": nbytes / sec / 1e9 * 2 * (world - 1) / world})
if rank == 0:
    for r in results:
        print(f"bandwidth size {r['bytes'] / 2**20:7.3f} MB: {r['ms']:7.2f} ms | algbw {r['algbw_gb_s']:.3f} GB/s"
              f" | busbw {r['busbw_gb_s']:.3f} GB/s", flush=True)
    with open(sys.argv[1], "w") as f:
        json.dump({"world": world, "allreduce": results}, f)
dist.destroy_process_group()
'''
(GEN / "collectives.py").write_text(COLLECTIVES)

t0 = time.time()
proc = D.torchrun(GEN / "collectives.py", nproc=4, args=[GEN / "collectives_4.json"], timeout=180)
show(proc)
print(f"wall time {time.time() - t0:.1f} s (process start-up + rendezvous dominate)")

# %% [markdown]
# **Expected:** the environment variables from rank 0, four `hello` lines in a
# *random order* (the processes run concurrently), and on every rank: sum = 10
# (1+2+3+4), max = 3, broadcast `[0.0, 10.0, 20.0]`, all_gather `[0, 1, 4, 9]`.
# At the barrier rank 0 waits longest (~0.9 s) and rank 3 almost nothing (all of them
# leave together). Then rank 0's bandwidth table, and the exit code 0. The job took
# about 5–10 s on our CPU, mostly start-up.
#
# Two things to notice:
#
# * **Every rank must call every collective, in the same order.** A rank that skips an
#   `all_reduce` (say, inside `if rank == 0:`) leaves the others waiting forever: a *hang*,
#   the most common distributed bug. This is why "rank 0 only" code must never contain a
#   collective.
# * **Bandwidth depends on message size.** Small messages are dominated by latency (fixed
#   cost per call), large ones approach the link bandwidth. *algbw* = bytes / time;
#   *busbw* = algbw × 2(N−1)/N, the bandwidth each link actually carried in a ring
#   (NCCL's `nccl-tests` report the same two numbers).

# %%
coll = json.loads((GEN / "collectives_4.json").read_text())
proc2 = D.torchrun(GEN / "collectives.py", nproc=2, args=[GEN / "collectives_2.json"], timeout=180)
coll2 = json.loads((GEN / "collectives_2.json").read_text()) if proc2.returncode == 0 else None
print(f"{'message':>10} | {'2 ranks ms':>10} | {'4 ranks ms':>10}")
for i, r4 in enumerate(coll["allreduce"]):
    r2 = coll2["allreduce"][i] if coll2 else {"ms": float("nan")}
    print(f"{r4['bytes'] / 2**20:8.3f}MB | {r2['ms']:10.2f} | {r4['ms']:10.2f}")

# %% [markdown]
# **Expected (CPU, gloo, one machine, measured):** with 2 ranks a 4 KB all-reduce takes
# ~0.3 ms and 16 MB ~12 ms; with 4 ranks, ~3–5 ms and ~21 ms (busbw ~1.2 GB/s in our run;
# it varies with machine load). Small messages are pure latency: a fixed cost per call
# that grows with the number of ranks (2(N−1) ring steps) and, here, with four processes
# competing for four shared cores. Large messages approach a bandwidth limit, and 4 ranks
# are slower than 2 because each rank sends 2(N−1)/N = 1.5× instead of 1× the data. **On real GPUs** NCCL over NVLink moves hundreds of GB/s and over PCIe or
# Ethernet 10–50 GB/s (datasheet figures, used as assumptions in 9.6; not measured).

# %% [markdown]
# ## 9.2 · Data parallelism = averaging gradients
#
# With a mean loss over a batch of B examples split into N equal shards of B/N,
#
# $$\nabla L_{\text{batch}} = \frac{1}{B}\sum_{j=1}^{B} \nabla \ell_j = \frac{1}{N}\sum_{r=1}^{N} \underbrace{\frac{N}{B}\sum_{j \in \text{shard } r} \nabla \ell_j}_{\text{rank } r\text{'s mean-loss gradient}}$$
#
# so if every rank computes the gradient of its own mean loss and the gradients are
# **summed with all-reduce and divided by N**, every rank holds exactly the full-batch
# gradient, takes the same optimizer step, and the replicas stay identical. That *is*
# data parallelism. Two conditions make it exact: the shards have equal size (DDP's
# `DistributedSampler` pads to guarantee this) and the replicas start identical (DDP
# **broadcasts rank 0's weights** when it wraps the model). BatchNorm breaks it (each
# rank's statistics come from its own shard; `SyncBatchNorm` fixes that at a
# communication cost).
#
# First, the arithmetic without processes.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-ddp-step.svg" alt="Three ranks each run forward and backward on their own shard and get different local gradients; an all-reduce averages them; every rank takes the same optimizer step" width="100%">
#
# *What to notice:* the only box that spans all ranks is the all-reduce; before it the ranks differ (different data, different g), after it they are identical again.
#
# > 💡 **Intuition.** Four teaching assistants each grade a quarter of the exam pile, then agree
# > on the class average, and all four write that same average in their notebooks. If the piles
# > are the same size, the result is exactly the average over the whole pile. *Where it breaks
# > down:* if one assistant's pile is smaller (unequal shards) or each uses a different curve
# > (BatchNorm statistics), the "average of averages" is no longer the true average.

# %% [markdown]
# > **Exercise 1.** Write `ring_allreduce_bytes(n_params, bytes_per, world_size)`: the number of
# > bytes each rank **sends** in a ring all-reduce of `n_params` values of `bytes_per` bytes each.
# > A ring all-reduce is a reduce-scatter followed by an all-gather; each has N−1 steps, and in
# > each step every rank sends one chunk of 1/N of the data. One rank sends nothing.

# %%
def ring_allreduce_bytes(n_params, bytes_per, world_size):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(ring_allreduce_bytes)

# %% [markdown]
# > **Exercise 2.** Write `average_gradients(grads_per_rank)`: `grads_per_rank[r][i]` is rank r's
# > gradient tensor for parameter i. Return a **new** list with one tensor per parameter: what
# > every rank holds after `all_reduce(SUM)` and division by the world size. Do not modify the
# > inputs. (A pure-tensor simulation; the next cell checks it against the full-batch gradient.)

# %%
def average_gradients(grads_per_rank):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(average_gradients)

# %%
torch.manual_seed(0)
sim_model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4))
xb, yb = torch.randn(32, 16), torch.randint(0, 4, (32,))


def grads_of(x, y):
    sim_model.zero_grad()
    torch.nn.functional.cross_entropy(sim_model(x), y).backward()
    return [p.grad.clone() for p in sim_model.parameters()]


full = grads_of(xb, yb)
avg_fn = average_gradients
try:
    avg_fn([[torch.zeros(1)]])
except NotImplementedError:
    avg_fn = D.average_gradients  # Exercise 2 not done yet: use the reference implementation
for world in (2, 4):
    shards = [grads_of(x, y) for x, y in zip(xb.chunk(world), yb.chunk(world))]
    err = max((a - b).abs().max().item() for a, b in zip(avg_fn(shards), full))
    print(f"{world} simulated ranks: max |averaged - full-batch gradient| = {err:.1e}")
uneven = [grads_of(xb[:8], yb[:8]), grads_of(xb[8:], yb[8:])]  # shards of 8 and 24
err = max((a - b).abs().max().item() for a, b in zip(avg_fn(uneven), full))
print(f"uneven shards (8 + 24): max difference = {err:.1e}  <- plain averaging is wrong for unequal shards")

# %% [markdown]
# **Expected:** differences around 1e-8 (floating-point summation order) for 2 and 4 equal
# shards, and a visibly larger difference (our run: 6.3e-02) for the uneven 8 + 24 split.
#
# ## 9.3 · A real data-parallel training script
#
# The script below trains a small MLP classifier on synthetic data with **2 processes**,
# in one of three modes:
#
# * `none`: no communication: each rank trains its own replica on its own shard. The
#   replicas *diverge*: this is what you get if you forget to sync.
# * `manual`: broadcast rank 0's initial weights, then after every `backward()`,
#   `all_reduce` each gradient and divide by the world size: DDP by hand.
# * `ddp`: wrap the model in `torch.nn.parallel.DistributedDataParallel`, which does the
#   same with gradient **bucketing** (≈25 MB buckets all-reduced as soon as they are ready,
#   overlapping communication with the rest of backward).
#
# It also shows the production habits: `DistributedSampler` with `set_epoch`, **rank-0
# logging** (`log()` prints only on rank 0), a **rank-0 checkpoint** at the end of every
# epoch followed by a `barrier()`, the *global* loss via `all_reduce`, and **resuming**
# from the checkpoint. On a GPU cluster you would change `"gloo"` to `"nccl"`, call
# `torch.cuda.set_device(local_rank)`, move the model and batches to that GPU, and pass
# `device_ids=[local_rank]` to DDP. Nothing else.
#
# Each rank deliberately initialises its replica with a **different seed**
# (`100 + rank`), so the broadcast is visibly necessary.

# %%
TRAIN_DDP = '''\
"""Data-parallel training on N processes (gloo, CPU).

Launch: torchrun --standalone --nproc_per_node=2 train_ddp.py --mode ddp --out OUTDIR
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

N_EXAMPLES, DIM, CLASSES, LOCAL_BATCH, LR, SAMPLER_SEED = 256, 16, 4, 16, 0.2, 0


def make_data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N_EXAMPLES, DIM, generator=g)
    y = (x @ torch.randn(DIM, CLASSES, generator=g)).argmax(1)  # a learnable rule
    return TensorDataset(x, y)


def make_model(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(DIM, 32), torch.nn.Tanh(), torch.nn.Linear(32, CLASSES))


def log(*args):  # rank-0 logging: one copy of every message, not world_size copies
    if dist.get_rank() == 0:
        print(*args, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["none", "manual", "ddp"], default="ddp")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument("--no-set-epoch", action="store_true")
    p.add_argument("--fail-at-epoch", type=int, default=-1)  # simulate a crash on rank 1
    args = p.parse_args()

    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    attempt = int(os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"))  # set by torchrun
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()  # nobody writes to out/ before it exists

    model = make_model(seed=100 + rank)  # deliberately different on every rank
    if args.mode == "manual":
        for p_ in model.parameters():
            dist.broadcast(p_.data, src=0)  # everyone starts from rank 0's weights
    elif args.mode == "ddp":
        model = DDP(model)  # broadcasts rank 0's weights (and buffers) here
    core = model.module if args.mode == "ddp" else model
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    ds = make_data()
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=SAMPLER_SEED)
    loader = DataLoader(ds, batch_size=LOCAL_BATCH, sampler=sampler)

    start_epoch, ckpt = 0, out / "checkpoint.pt"
    if ckpt.exists():  # every rank loads the same file: replicas stay identical
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        core.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        log(f"attempt {attempt}: resumed from {ckpt.name} -> starting at epoch {start_epoch}")
    else:
        log(f"attempt {attempt}: no checkpoint, starting at epoch 0 (mode={args.mode}, world={world})")

    seen = {}
    for epoch in range(start_epoch, args.epochs):
        if not args.no_set_epoch:
            sampler.set_epoch(epoch)  # reshuffle differently every epoch
        if epoch == args.fail_at_epoch and rank == 1 and attempt == 0:
            raise RuntimeError(f"simulated hardware failure on rank {rank} in epoch {epoch}")
        seen[epoch] = list(iter(sampler))
        total, count = torch.zeros(1), torch.zeros(1)
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()  # in ddp mode, gradients are all-reduced during this call
            if args.mode == "manual":
                for p_ in model.parameters():
                    dist.all_reduce(p_.grad, op=dist.ReduceOp.SUM)
                    p_.grad /= world
            opt.step()
            total += loss.detach() * len(y)
            count += len(y)
        if args.mode != "none":
            dist.all_reduce(total)  # the GLOBAL mean loss, not rank 0's shard
            dist.all_reduce(count)
        log(f"epoch {epoch}: mean train loss {(total / count).item():.4f}")
        if rank == 0:  # one writer; write to a temp file then rename (atomic)
            tmp = out / "checkpoint.tmp"
            torch.save({"model": core.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch}, tmp)
            tmp.replace(ckpt)
        dist.barrier()  # nobody runs ahead (or exits) before the checkpoint exists

    flat = torch.cat([p_.detach().flatten() for p_ in core.parameters()])
    torch.save(flat, out / f"params_rank{rank}.pt")
    (out / f"indices_rank{rank}.json").write_text(json.dumps(seen))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''
SCRIPT = GEN / "train_ddp.py"
SCRIPT.write_text(TRAIN_DDP)

spec = importlib.util.spec_from_file_location("train_ddp", SCRIPT)
train_ddp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_ddp)  # import it too: the reference run below reuses make_data/make_model
print("wrote", SCRIPT.relative_to(ROOT))

# %%
runs = {}
for mode in ("none", "manual", "ddp"):
    out_dir = GEN / f"run_{mode}"
    shutil.rmtree(out_dir, ignore_errors=True)  # a leftover checkpoint would be resumed
    t0 = time.time()
    proc = D.torchrun(SCRIPT, nproc=2, args=["--mode", mode, "--out", out_dir], timeout=180)
    print(f"--- mode={mode} ({time.time() - t0:.1f} s)")
    show(proc)
    runs[mode] = [torch.load(out_dir / f"params_rank{r}.pt") for r in range(2)] if proc.returncode == 0 else None

# %% [markdown]
# **Expected:** for every mode, rank 0's log only (one line per epoch, not two), the loss
# falling over 3 epochs (our run: 1.32 → 1.04 → 0.84) and exit code 0. In `none` mode the
# printed loss is rank 0's own shard (there is no global loss without communication). Each
# launch takes ~5–10 s on CPU, mostly process start-up.
#
# Now the claim of 9.2, end to end: **2-rank training = 1-process training on the
# concatenated batch.** The reference run below uses one process, rank 0's initial
# weights (seed 100), and at every step the union of the two ranks' mini-batches, read
# from `DistributedSampler` objects constructed for rank 0 and rank 1 (no process group
# is needed to construct one).

# %%
def single_process_reference(epochs=3, world=2):
    ds = train_ddp.make_data()
    model = train_ddp.make_model(seed=100)  # rank 0's initial weights
    opt = torch.optim.SGD(model.parameters(), lr=train_ddp.LR)
    samplers = [DistributedSampler(ds, num_replicas=world, rank=r, shuffle=True, seed=train_ddp.SAMPLER_SEED)
                for r in range(world)]
    bs = train_ddp.LOCAL_BATCH
    for epoch in range(epochs):
        order = []
        for s in samplers:
            s.set_epoch(epoch)
            order.append(list(s))
        for k in range(0, len(order[0]), bs):
            idx = sum((o[k:k + bs] for o in order), [])  # the global batch = all ranks' local batches
            x, y = ds.tensors[0][idx], ds.tensors[1][idx]
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return torch.cat([p.detach().flatten() for p in model.parameters()])


ref = single_process_reference()
summary = {}
for mode, params in runs.items():
    if params is None:
        print(f"{mode}: run failed, see its output above")
        continue
    replica_gap = (params[0] - params[1]).abs().max().item()
    vs_ref = (params[0] - ref).abs().max().item()
    summary[mode] = {"max_diff_rank0_vs_rank1": replica_gap, "max_diff_vs_single_process": vs_ref}
    print(f"{mode:6} | rank0 vs rank1: {replica_gap:.1e} | rank0 vs single-process: {vs_ref:.1e}")

# %% [markdown]
# **Expected (measured):**
#
# | mode | rank 0 vs rank 1 | vs single process |
# |---|---|---|
# | `none` | ~0.6 (replicas diverged) | ~0.07 |
# | `manual` | 0.0 (identical) | ~6e-08 |
# | `ddp` | 0.0 (identical) | ~6e-08 |
#
# The two synchronised modes agree with single-process training on the combined batch
# up to floating-point rounding (the sum happens in a different order), after 48 SGD
# steps. That is the whole contract of data parallelism: **same maths, more hardware.**
# The replicas are *bit-identical* with each other because every rank applies the same
# averaged gradient to the same weights.
#
# What DDP adds over the manual loop is engineering: gradient **buckets** all-reduced
# *during* backward (overlap), one big all-reduce per bucket instead of one per tensor,
# `no_sync()` for gradient accumulation (skip the all-reduce on all but the last
# micro-batch), and checks that every parameter received a gradient
# (`find_unused_parameters`).
#
# ## 9.4 · `DistributedSampler`: sharding, `set_epoch` and padding
#
# `DistributedSampler` gives each rank a disjoint slice of a (shuffled) index list. The
# shuffle is seeded with `seed + epoch`, and the epoch is whatever you last passed to
# `sampler.set_epoch(epoch)`. **Forget to call it, and every epoch uses the same order**:
# not a crash, just silently worse training (the same mini-batches in the same order every
# epoch). It is one of the classic DDP bugs, because nothing fails.

# %%
ds_demo = list(range(12))
for call_set_epoch in (True, False):
    s = DistributedSampler(ds_demo, num_replicas=2, rank=0, shuffle=True, seed=0)
    orders = []
    for epoch in range(3):
        if call_set_epoch:
            s.set_epoch(epoch)
        orders.append(list(s))
    label = "with set_epoch   " if call_set_epoch else "WITHOUT set_epoch"
    print(f"{label} rank 0, epochs 0-2: {orders}")

for r in range(4):  # 10 examples on 4 ranks: ceil(10/4) = 3 per rank -> 2 examples repeated
    print(f"n=10, 4 ranks, no shuffle: rank {r} gets {list(DistributedSampler(range(10), 4, r, shuffle=False))}")

# %% [markdown]
# **Expected:** with `set_epoch` the three epochs give three different orders; without it,
# the same 6 indices in the same order every epoch. With 10 examples on 4 ranks, ranks get
# `[0, 4, 8]`, `[1, 5, 9]`, `[2, 6, 0]`, `[3, 7, 1]`: indices 0 and 1 are used twice, so
# every rank has the same number of batches (otherwise the ranks with fewer batches would
# stop calling all-reduce and the others would hang). For training, a few duplicates are
# harmless. For **evaluation** they bias the metric: gather predictions with their indices
# and de-duplicate, or evaluate on rank 0 only for small sets
# (`drop_last=True` instead drops the tail).

# %% [markdown]
# > **Exercise 3.** Write `shard_indices(n, world_size, rank, epoch, seed, shuffle)` returning the list
# > of indices `DistributedSampler(range(n), world_size, rank, shuffle=shuffle, seed=seed)` yields after
# > `set_epoch(epoch)` (with the default `drop_last=False`). Use `torch.randperm` with a
# > `torch.Generator` seeded with `seed + epoch`. Then pad by repeating indices from the start of the list
# > (possibly several times, if n < world_size) until the length is a multiple of world_size, and give rank r
# > every world_size-th index starting at r. The grader compares with PyTorch's sampler.

# %%
def shard_indices(n, world_size, rank, epoch, seed, shuffle):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(shard_indices)

# %% [markdown]
# The training script recorded each rank's indices. Run it once more without
# `set_epoch` and look at what each rank saw.

# %%
out_dir = GEN / "run_no_set_epoch"
shutil.rmtree(out_dir, ignore_errors=True)
proc = D.torchrun(SCRIPT, nproc=2, args=["--mode", "ddp", "--out", out_dir, "--no-set-epoch"], timeout=180)
for name, d in (("with set_epoch   ", GEN / "run_ddp"), ("WITHOUT set_epoch", out_dir)):
    if not (d / "indices_rank0.json").exists():
        print(name, "run failed:", proc.stderr[-300:])
        continue
    idx = [json.loads((d / f"indices_rank{r}.json").read_text()) for r in range(2)]
    same_order = idx[0]["0"] == idx[0]["1"] == idx[0]["2"]
    overlap = set(idx[0]["0"]) & set(idx[1]["0"])
    print(f"{name}: rank 0 epoch-0 starts {idx[0]['0'][:6]}, epoch-1 starts {idx[0]['1'][:6]} | "
          f"same order every epoch: {same_order} | ranks overlap in epoch 0: {len(overlap)} indices")

# %% [markdown]
# **Expected:** with `set_epoch`, the epoch-0 and epoch-1 orders differ; without it,
# `same order every epoch: True`. In both cases the two ranks share **0** indices within an
# epoch (256 examples split 128 / 128).
#
# ## 9.5 · When a rank dies: failures and elastic restarts
#
# A job on 1,000 GPUs fails often (a GPU falls off the bus, a node reboots, NCCL times out),
# so failure handling is part of the design, not an afterthought:
#
# * **A rank crashes** (exception, OOM, killed): its process exits with an error. The other
#   ranks are blocked in the next collective. `torchrun`'s agent notices the dead worker,
#   **terminates the remaining workers**, and exits with an error report naming the root
#   cause (rank, exit code, traceback).
# * **A rank hangs** (deadlock, a skipped collective, a dead network link): nobody exits;
#   the others wait in the collective until the process-group **timeout** (`timeout=` in
#   `init_process_group`; NCCL's watchdog then aborts). Set it deliberately and log
#   heartbeats: a silent hang burns the whole cluster's time.
# * **Recovery** is a restart from the last checkpoint. `torchrun --max-restarts=k` restarts
#   *all* workers of the job (up to k times) in place; with `--nnodes=MIN:MAX` and a rendezvous
#   backend (*elastic* training) the job can also continue with fewer or more nodes. In
#   both cases **your script** must resume: load the latest checkpoint, the optimizer and
#   scheduler state, the epoch / step, and the sampler position.
#
# Our script raises an error on rank 1 at the start of epoch 1 (on the first attempt only;
# `TORCHELASTIC_RESTART_COUNT` counts torchrun's restarts). First, the crash:

# %%
fail_dir = GEN / "run_failure"
shutil.rmtree(fail_dir, ignore_errors=True)
t0 = time.time()
proc = D.torchrun(SCRIPT, nproc=2, args=["--mode", "ddp", "--out", fail_dir, "--fail-at-epoch", "1"],
                  max_restarts=0, timeout=180)
show(proc, keep=("attempt", "epoch", "RuntimeError", "exitcode", "rank      ", "ChildFailedError"))
print(f"wall time {time.time() - t0:.1f} s; checkpoint left behind: {(fail_dir / 'checkpoint.pt').exists()}")
fail_returncode = proc.returncode

# %% [markdown]
# **Expected:** `attempt 0: no checkpoint`, `epoch 0: …` (one epoch finished and was
# checkpointed), then rank 1's `RuntimeError: simulated hardware failure`, rank 0's own
# error (`Connection closed by peer`: it was blocked in DDP's all-reduce when its partner
# vanished), and `torchrun`'s `ChildFailedError` report with the failed rank and exit
# code. The job's **exit code is 1**, and the epoch-0 checkpoint is on disk. Nothing hung:
# the dead peer closed its sockets, and the agent terminated the survivors.
#
# Now recover. A scheduler (Slurm requeue, a Kubernetes job controller, or `torchrun`
# itself with `--max-restarts=k`) launches the *same command* again; the script finds the
# checkpoint and resumes. We play the scheduler and relaunch (without the simulated fault):

# %%
t0 = time.time()
proc = D.torchrun(SCRIPT, nproc=2, args=["--mode", "ddp", "--out", fail_dir], timeout=180)
show(proc, keep=("attempt", "epoch"))
print(f"wall time {time.time() - t0:.1f} s")
restart_params = [torch.load(fail_dir / f"params_rank{r}.pt") for r in range(2)] if proc.returncode == 0 else None
if restart_params is not None and runs.get("ddp") is not None:
    print(f"resumed run vs uninterrupted ddp run: max |param diff| = "
          f"{(restart_params[0] - runs['ddp'][0]).abs().max().item():.1e}")

# %% [markdown]
# **Expected:** `resumed from checkpoint.pt -> starting at epoch 1`, epochs 1 and 2 with the
# same losses as the uninterrupted `ddp` run (1.0427 and 0.8400 in ours), exit code 0, and a
# parameter difference of **0.0**: resuming at an epoch boundary with model + optimizer state +
# `set_epoch` reproduces the uninterrupted trajectory exactly. Mid-epoch resumption also needs
# the step inside the epoch and the RNG states, which is why large jobs checkpoint by *step*.
# How *often* to checkpoint is a trade-off you compute in notebook 10.
#
# **`torchrun --max-restarts=k`** does the relaunch for you, in place: it tears down all
# workers and starts them again with `TORCHELASTIC_RESTART_COUNT` incremented and a fresh
# rendezvous. *Elastic* launches (`--nnodes=MIN:MAX --rdzv-backend=c10d
# --rdzv-endpoint=HOST:PORT`) may also continue with a different number of nodes, which is
# why a resumable script must not assume the world size it was started with (re-shard the
# data; the global batch changes unless you adapt accumulation). In our CPU verification
# runs, in-place restarts with the gloo backend were flaky (the restarted workers
# sometimes failed to reconnect: `Connection refused` during `init_process_group`), so this
# notebook relaunches explicitly; try `max_restarts=1` with `--fail-at-epoch 1` yourself.
#
# `torch.multiprocessing.spawn(fn, nprocs=N)` is the other way to start workers from
# Python. It needs `fn` importable by the child processes (defined in a module, not in a
# notebook cell, under the `spawn` start method) and gives you no restarts, which is why
# this notebook (and most production code) uses `torchrun`.

# %% [markdown]
# ## 9.6 · Communication cost: when does DDP stop scaling?
#
# Per step, each GPU computes forward + backward on its shard (time `T_compute`, constant
# if the per-GPU batch is fixed) and then all-reduces the gradients:
# `T_comm ≈ 2(N−1)/N × (params × bytes per gradient) / link bandwidth`. DDP overlaps part
# of `T_comm` with backward. Scaling efficiency = `T_compute / T_step`.
#
# The table uses **assumed** numbers (no multi-GPU measurement here): fp16 gradients
# (2 bytes), 8 GPUs, link bandwidths from datasheets (A100 NVLink ≈ 300 GB/s per
# direction; PCIe 4.0 x16 ≈ 25 GB/s achievable; 100 Gb/s Ethernet = 12.5 GB/s), and a
# compute time per step that we simply assume (250 ms), as if the per-GPU batch were tuned
# to take that long.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-ring-allreduce.svg" alt="Four ranks in a ring each sending to their neighbour; a reduce-scatter of N minus 1 steps then an all-gather of N minus 1 steps, so each rank sends 2(N-1)/N times the message size" width="100%">
#
# *What to notice:* no rank ever talks to more than one neighbour, so the bytes each rank sends approach 2M and stop growing; what still grows with N is the number of steps (latency).
#
# > 💡 **Intuition.** Passing a collection bucket around a table: each person adds their share and
# > passes it on, then the totals go round once more so everyone knows them. Nobody ever talks to
# > the whole table, so the busiest person's work stays the same as the table grows. *Where it
# > breaks down:* the slowest person at the table sets the pace for everyone (stragglers), and
# > every extra person adds a hand-off (latency).

# %%
models = {"ResNet-50 (25.6M)": 25.6e6, "BERT-base (110M)": 110e6, "GPT-2 XL (1.5B)": 1.5e9, "7B": 7e9}
links = {"NVLink ~300 GB/s": 300, "PCIe ~25 GB/s": 25, "Ethernet 100Gb ~12.5 GB/s": 12.5}
T_COMPUTE, WORLD, OVERLAP = 0.250, 8, 0.5
print(f"assumptions: {WORLD} GPUs, fp16 gradients, T_compute={T_COMPUTE * 1e3:.0f} ms/step, "
      f"{OVERLAP:.0%} of communication hidden behind backward")
print(f"{'model':20} {'grad MB':>8} {'sent/GPU MB':>11} | " + " | ".join(f"{k:>25}" for k in links))
for name, n in models.items():
    sent = D.ring_allreduce_bytes(n, 2, WORLD)
    cells = []
    for bw in links.values():
        comm = D.allreduce_seconds(n, 2, WORLD, bw)
        eff = D.scaling_efficiency(T_COMPUTE, comm, OVERLAP)
        cells.append(f"{comm * 1e3:8.1f} ms -> {eff:4.0%} eff")
    print(f"{name:20} {n * 2 / 1e6:8.0f} {sent / 1e6:11.0f} | " + " | ".join(f"{c:>25}" for c in cells))

# %% [markdown]
# **Expected (arithmetic, not a measurement):** ResNet-50 and BERT-base scale almost
# perfectly on any of these links (all-reducing 51 MB or 220 MB of gradients is short
# compared with 250 ms of compute). A 1.5B model is fine on NVLink (97%) but drops to ~70%
# over PCIe and ~54% over 100 Gb Ethernet; a 7B model's 14 GB of fp16 gradients take ~1 s
# per step even on PCIe (34%): data parallelism *alone* stops scaling, and (notebook 10)
# its memory does not fit on one GPU anyway.
#
# What moves the needle, in order: **bigger per-GPU batches** (more compute per byte
# communicated; gradient accumulation with `no_sync()` all-reduces once per k micro-batches);
# **overlap** (DDP buckets; tune `bucket_cap_mb`); **compress or shrink the gradients**
# (fp16/bf16 gradients, `ddp_comm_hooks`, PowerSGD); **faster links** (keep data-parallel
# groups inside an NVLink node and cross nodes with InfiniBand); and, for big models,
# **different parallelism** (ZeRO/FSDP, tensor and pipeline parallelism: notebook 10).
# Also note that per-GPU traffic barely depends on N (2(N−1)/N → 2), but the *latency*
# term grows with N, and a ring is only as fast as its slowest link: one slow node slows
# every step (the *straggler* problem).

# %%
summary.update({
    "allreduce_4_ranks": coll["allreduce"], "failure_exit_code": fail_returncode,
    "restart_matches_uninterrupted": (None if restart_params is None or runs.get("ddp") is None
                                      else (restart_params[0] - runs["ddp"][0]).abs().max().item()),
})
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("wrote", OUT / "summary.json")
print("generated files:", sorted(p.name for p in GEN.iterdir()))

# %% [markdown]
# > **Exercise 4.** Your team moves a recommendation model's training from 1 GPU to 8 GPUs on one
# > node with DDP, keeping the *per-GPU* batch size the same. Throughput goes up only 5.5×, and the
# > validation metric after the same number of *epochs* is slightly worse. (a) List the likely causes
# > of the 5.5× and how you would measure each (think: communication, input pipeline, stragglers,
# > small batches). (b) Why might the metric be worse, and what would you change (learning rate,
# > warm-up, number of steps)? (c) One engineer proposes "only rank 0 evaluates and saves checkpoints,
# > and evaluation runs inside `if rank == 0:` with a `dist.all_reduce` of the metric". What goes
# > wrong? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 4):* ...

# %% [markdown]
# ## 🧮 Key equations
#
# **1. Data parallelism averages gradients.**
#
# $$
# g = \frac{1}{N} \sum_{r=1}^{N} g_r, \qquad g_r = \nabla_\theta \frac{1}{b} \sum_{j \in \text{shard } r} \ell_j
# $$
#
# In words: with equal shards of size $b$, the mean of the ranks' mean-loss gradients is the
# gradient of the mean loss over the global batch $N b$. Example: shards with gradients 0.2 and
# 0.4 give 0.3 on both ranks, exactly what one process on the concatenated batch gets.
#
# **2. Ring all-reduce traffic.**
#
# $$
# \text{bytes sent per rank} = 2\,\frac{N - 1}{N}\, M
# $$
#
# In words: $N - 1$ reduce-scatter steps plus $N - 1$ all-gather steps, each sending $M / N$.
# Example: 7B fp16 gradients ($M = 14$ GB) on 8 GPUs: $2 \cdot \tfrac{7}{8} \cdot 14 = 24.5$ GB per
# rank per step; on 1,000 GPUs it is still under 28 GB.
#
# **3. All-reduce time and bus bandwidth.**
#
# $$
# T_{\text{comm}} \approx 2\,\frac{N-1}{N}\,\frac{M}{\text{BW}} + 2(N-1)\,\alpha, \qquad \text{busbw} = \text{algbw} \cdot 2\,\frac{N-1}{N}
# $$
#
# In words: a bandwidth term that barely depends on $N$, plus a latency term ($\alpha$ per
# hop) that grows with it. Example: 1.5B fp16 gradients (3 GB), 8 GPUs, PCIe at 25 GB/s:
# $2 \cdot \tfrac78 \cdot 3 / 25 \approx 0.21$ s.
#
# **4. Scaling efficiency.**
#
# $$
# E = \frac{T_{\text{compute}}}{T_{\text{step}}}, \qquad T_{\text{step}} \approx T_{\text{compute}} + (1 - o)\, T_{\text{comm}}
# $$
#
# In words: the share of the step spent on useful compute; $o$ is the fraction of communication
# hidden behind backward. Example (9.6): $T_{\text{compute}} = 250$ ms, $T_{\text{comm}} = 210$ ms,
# $o = 0.5$: $E = 250 / 355 = 0.70$.
#
# **5. Global batch size.**
#
# $$
# B_{\text{global}} = N \cdot b_{\text{per GPU}} \cdot k_{\text{accum}}
# $$
#
# Example: 8 GPUs × 64 per GPU × 8 accumulation steps = 4,096; DDP's `no_sync()` skips the
# all-reduce on the first 7 micro-batches.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | Job hangs forever, no error | a collective called on some ranks only (inside `if rank == 0:`), or ranks with different numbers of batches | every rank calls every collective in the same order; `DistributedSampler` (pads) or `drop_last=True`; set `timeout=` in `init_process_group` |
# | `RuntimeError: ... Address already in use` / rendezvous timeout | a previous job still holds the port | `--standalone` picks a free port; kill leftover `python` processes; restart the runtime |
# | `errno: 97 - Address family not supported` warnings | no IPv6 on the machine; c10d tries both | harmless; the job falls back to IPv4 |
# | Replicas drift apart over time | no initial broadcast, a manual loop missing the gradient all-reduce, or different random dropout/augmentation that also changes *control flow* | wrap with DDP (broadcasts in the constructor); compare a parameter checksum across ranks with `all_gather` |
# | Same loss curve shape every epoch, worse accuracy than single GPU | forgot `sampler.set_epoch(epoch)` | call it at the start of every epoch |
# | Metrics slightly off in distributed evaluation | `DistributedSampler` padding duplicated examples | de-duplicate by index after `all_gather`, or evaluate on one rank |
# | `Expected to have finished reduction in the prior iteration` | parameters that get no gradient in some steps | `DDP(..., find_unused_parameters=True)`, or remove the unused parameters |
# | Checkpoint file corrupt after a crash | written by several ranks, or not atomically | write on rank 0 only, to a temp file, then `rename`; `barrier()` after |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Walk through what happens in one DDP training step, from the forward pass to the optimizer step. Where does communication happen, and how does DDP overlap it with computation?
# 2. Why is data-parallel training with N ranks mathematically equivalent to single-GPU training with an N-times larger batch? Name two things that break the equivalence.
# 3. Derive the per-GPU communication volume of a ring all-reduce. Why is it nearly independent of the number of GPUs, and why does DDP still stop scaling?
# 4. What does `DistributedSampler` do, what goes wrong if you forget `set_epoch`, and why does it pad the dataset?
# 5. One rank of a 512-GPU job crashes; another job hangs with no error. How does each look from the outside, how do you detect it, and how do you recover?
# 6. You must train with a global batch of 4,096 on 8 GPUs that each fit 64 examples. How do you get there, and what changes in the training recipe?
#
# ➡️ **Next:** [`10_scaling_to_large_models.ipynb`](../ch48-scaling-to-large-models/notebook.ipynb) — when the model itself no longer fits on one GPU: ZeRO/FSDP, tensor and pipeline parallelism, compute budgets and fault tolerance.
