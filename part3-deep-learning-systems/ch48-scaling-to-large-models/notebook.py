# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 48 · Scaling to large models
#
# **Part III · Chapter 48 · about 90 minutes · T4 GPU (runs on CPU; everything here is arithmetic, simulation and small CPU demos)**
#
# Data parallelism (Chapter 47) assumes a full copy of the model, its gradients and its
# optimizer state fits on every GPU. For a 7B-parameter model trained with Adam that is
# ~112 GB, which no single GPU holds. This lab covers what large-model teams do
# instead (shard the state, split the layers, pipeline the stages), how they decide how
# much compute a run needs and how large the model should be, and how they cope with
# hardware failures during runs that last weeks. These numbers underlie any answer to a
# question such as "how would you train a 70B model?".
#
# After this lab you should be able to:
#
# 1. compute model-state memory per GPU under DDP and ZeRO stages 1/2/3 (FSDP), and run a real 2-process FSDP step on CPU;
# 2. explain tensor parallelism by splitting a Linear layer across two simulated devices, and verify the result;
# 3. compute the pipeline-parallel bubble (p−1)/(m+p−1) and check it with a simulated schedule;
# 4. budget a training run: FLOPs ≈ 6·N·D, GPU-days at a stated MFU, and a compute-optimal model size from published scaling laws;
# 5. choose a checkpoint interval with Young's formula and check it by simulation;
# 6. explain fp16 overflow/underflow, loss scaling, and why bf16 does not need it.
#
# **You will produce:** `artifacts/level3/nb10/budget.json` (your calculators' outputs for
# the worked examples) and a generated FSDP script in `generated/nb10/`. You will use
# [`src/scaleml/distributed.py`](../../src/scaleml/distributed.py) (the calculators) and
# [`src/scaleml/tinygpt.py`](../../src/scaleml/tinygpt.py).
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*. Every multi-GPU number in this notebook
# is a calculation from stated assumptions, not a measurement.

# %% [markdown]
# Read first: [Chapter 48 text](README.md) explains the ideas behind this lab (Scaling to large models) in more depth; [solutions](solutions.md) has worked answers.

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
import math
import time

import numpy as np
import torch

from scaleml import distributed as D
from scaleml import tinygpt as tg
from grading import ch48 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb10"
OUT.mkdir(parents=True, exist_ok=True)
GEN = ROOT / "generated" / "nb10"
GEN.mkdir(parents=True, exist_ok=True)
RESULTS = {}
GB = 1e9

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Ψ (psi) | the number of model parameters, in memory formulas | 7B: Ψ = 7 × 10⁹ |
# | Model state | weights + gradients + optimizer state (not activations) | 16Ψ bytes for mixed-precision Adam |
# | ZeRO stage 1/2/3 | shard optimizer state / + gradients / + parameters across data-parallel GPUs | 7B on 8 GPUs: 38.5 / 26.2 / 14.0 GB |
# | FSDP | PyTorch's fully sharded data parallel: ZeRO-3, gathering each layer just in time | `fully_shard` in the 48.1 demo |
# | Tensor parallelism | splitting the matrices of one layer across GPUs | column- then row-parallel MLP in 48.2 |
# | Pipeline parallelism | putting consecutive groups of layers on different GPUs | 4 stages in 48.3 |
# | Micro-batch / bubble | a slice of the batch fed through the pipeline / the idle time while it fills and drains | 3/7 idle at p = m = 4 |
# | 3D parallelism | tensor × pipeline × data parallelism combined | TP inside a node, PP across nodes, DP across replicas |
# | MFU | model FLOPs utilisation: useful FLOP/s divided by the hardware peak | ~40% for well-tuned large runs |
# | Compute-optimal (Chinchilla) | the model size and token count that minimise loss for a fixed FLOP budget | ≈ 20 tokens per parameter |
# | MTBF | mean time between failures | 128 nodes at 50 days each → 9.4 h for the job |
# | Loss scaling | multiplying the loss so small fp16 gradients do not round to zero | `GradScaler`, S = 2¹⁶ to start |
# | bf16 | a 16-bit format with fp32's exponent range and less precision | A100/H100 training; no scaler needed |

# %% [markdown]
# ## 48.1 · Memory per GPU: DDP vs ZeRO-1/2/3 (FSDP)
#
# The standard accounting (ZeRO paper, Rajbhandari et al. 2020) for **mixed-precision
# Adam**, with Ψ parameters:
#
# | state | bytes / param |
# |---|---|
# | fp16/bf16 parameters (used in forward/backward) | 2 |
# | fp16/bf16 gradients | 2 |
# | fp32 master weights + Adam's two fp32 moments ("K", optimizer state) | 4 + 4 + 4 = 12 |
# | **total** | **16** |
#
# (Chapter 40 counted the PyTorch-autocast way, fp32 weights and gradients plus a cached
# fp16 copy: 18 bytes/param. Same order of magnitude; we use the paper's 16 here because the
# ZeRO stages are defined on it.) **Activations come on top** and depend on batch, sequence
# length and checkpointing (Chapter 40).
#
# DDP replicates all 16Ψ bytes on every GPU. ZeRO ("Zero Redundancy Optimizer") partitions
# the redundant copies across the N data-parallel GPUs:
#
# | stage | sharded | bytes / GPU | communication vs DDP |
# |---|---|---|---|
# | 0 = DDP | nothing | (2 + 2 + 12) Ψ | all-reduce of gradients (2Ψ·2 bytes moved) |
# | ZeRO-1 | optimizer state | (2 + 2) Ψ + 12 Ψ / N | same |
# | ZeRO-2 | + gradients | 2 Ψ + 14 Ψ / N | same (reduce-scatter + all-gather) |
# | ZeRO-3 = FSDP | + parameters | 16 Ψ / N | ~1.5× (parameters all-gathered in forward *and* backward) |
#
# PyTorch's `FullyShardedDataParallel` (FSDP, and FSDP2's `fully_shard`) implements stage 3
# (and, with `SHARD_GRAD_OP`, stage 2); DeepSpeed implements all three.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-zero-stages.svg" alt="Stacked bars of model-state bytes per parameter on each of 4 GPUs: DDP 16, ZeRO-1 7, ZeRO-2 5.5, ZeRO-3 4" width="100%">
#
# *What to notice:* each stage shards one more colour; the green optimizer block is the biggest, which is why stage 1 alone already saves most of the memory.
#
# > **Intuition.** With DDP every member of a study group carries the full encyclopedia.
# > ZeRO-1 splits the bulky index volumes (optimizer state), ZeRO-2 also the notes (gradients),
# > ZeRO-3 the encyclopedia itself: each person carries a few volumes and borrows the page they
# > need right before reading it. *Where it breaks down:* borrowing takes network time, twice per
# > step (forward and backward), and a slow network makes it the bottleneck.

# %% [markdown]
# > **Exercise 1.** Write `zero_memory_per_gpu(n_params, world_size, stage)` returning the model-state
# > bytes per GPU (a number) for stage 0 (DDP), 1, 2 or 3 using the table above (2 + 2 + 12 bytes/param).
# > Raise `ValueError` for any other stage.

# %%
def zero_memory_per_gpu(n_params, world_size, stage):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(zero_memory_per_gpu)

# %%
print("Model-state GB per GPU (mixed-precision Adam, no activations). A T4 has 16 GB; an A100 40 or 80 GB.")
for name, n in [("1.3B", 1.3e9), ("7B", 7e9), ("70B", 70e9)]:
    print(f"--- {name} parameters ({16 * n / GB:,.0f} GB in total)")
    for row in D.zero_table(n, world_sizes=(1, 8, 64, 512)):
        print(f"  {row['gpus']:4d} GPUs | DDP {row['stage0_GB']:8.1f} | ZeRO-1 {row['stage1_GB']:7.1f} | "
              f"ZeRO-2 {row['stage2_GB']:7.1f} | ZeRO-3/FSDP {row['stage3_GB']:7.2f}")
RESULTS["zero_7b_64gpus_GB"] = {f"stage{s}": D.zero_memory_per_gpu(7e9, 64, s) / GB for s in range(4)}

# %% [markdown]
# **Expected (arithmetic):** a 7B model needs 112 GB with DDP at any GPU count; on 8 GPUs
# ZeRO-1 brings it to 38.5 GB, ZeRO-2 to 26.2 GB and ZeRO-3 to 14.0 GB; on 64 GPUs ZeRO-3
# needs 1.75 GB per GPU. ZeRO-1 and 2 plateau at 4Ψ and 2Ψ (28 and 14 GB for 7B) however
# many GPUs you add, because the replicated fp16 parameters (and gradients) remain.
# Only stage 3 keeps shrinking, at the price of gathering every layer's parameters just in
# time, in forward and again in backward. A 70B model needs ZeRO-3 (or tensor/pipeline
# parallelism) even on 64 GPUs to leave room for activations.
#
# ### FSDP for real: 2 processes on CPU
#
# The script below shards a small `TinyGPT` with FSDP2's `fully_shard` (one FSDP unit per
# transformer block, plus the root) across 2 gloo processes, takes 3 AdamW steps, and
# reports how many parameter and optimizer-state elements each rank holds locally. It also
# prints the losses, which we then compare with a single process training on the
# concatenated batch, the same check as DDP in Chapter 47. FSDP is guarded by a flag
# because `fully_shard` on CPU/gloo needs a recent PyTorch (2.6+); if it fails, the cell
# explains and the notebook continues.

# %%
RUN_FSDP_DEMO = True

FSDP_SCRIPT = '''\
"""FSDP2 on CPU. Launch: torchrun --standalone --nproc_per_node=2 fsdp_demo.py OUT.json"""
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # the course package
from scaleml import tinygpt as tg  # noqa: E402

try:
    from torch.distributed.fsdp import fully_shard  # PyTorch >= 2.6
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard  # 2.4-2.5

CFG = tg.GPTConfig(vocab_size=256, block_size=32, n_embd=64, n_head=4, n_layer=2)


def batches(steps=3, batch=8):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, CFG.vocab_size, (batch, CFG.block_size + 1), generator=g) for _ in range(steps)]


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(0)  # same init on every rank (FSDP shards it; it does not broadcast)
    model = tg.TinyGPT(CFG)
    full = sum(p.numel() for p in model.parameters())
    for block in model.blocks:
        fully_shard(block)  # each block: its own all-gather / reduce-scatter unit
    fully_shard(model)  # the root: embeddings (tied with the output head), final LayerNorm
    local = sum(p.to_local().numel() for p in model.parameters())  # parameters are DTensors now
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for b in batches():
        x = b.chunk(world)[rank]  # this rank's half of the global batch
        _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()  # gradients are reduce-scattered: each rank keeps its shard
        opt.step()
        opt.zero_grad()
        loss = loss.detach()
        dist.all_reduce(loss)
        losses.append(loss.item() / world)
    state = sum(v.to_local().numel() for s in opt.state.values() for v in s.values()
                if torch.is_tensor(v) and v.dim() > 0)
    report = {"rank": rank, "full_params": full, "local_params": local, "local_optimizer_state": state,
              "losses": losses}
    print(json.dumps(report), flush=True)
    if rank == 0:
        Path(sys.argv[1]).write_text(json.dumps(report))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
'''
fsdp_report = None
if RUN_FSDP_DEMO:
    (GEN / "fsdp_demo.py").write_text(FSDP_SCRIPT)
    t0 = time.time()
    try:
        proc = D.torchrun(GEN / "fsdp_demo.py", nproc=2, args=[GEN / "fsdp_report.json"], timeout=240)
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                print(line)
        if proc.returncode == 0:
            fsdp_report = json.loads((GEN / "fsdp_report.json").read_text())
        else:
            print("FSDP demo failed (exit code", proc.returncode, "). Last lines of its error output:")
            print("\n".join(proc.stderr.strip().splitlines()[-5:]))
    except Exception as exc:
        print(f"FSDP demo could not run here ({type(exc).__name__}: {exc}).")
    print(f"({time.time() - t0:.1f} s)")
else:
    print("Skipped (RUN_FSDP_DEMO = False).")

# %%
if fsdp_report:
    spec = importlib.util.spec_from_file_location("fsdp_demo", GEN / "fsdp_demo.py")
    fsdp_demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fsdp_demo)
    torch.manual_seed(0)
    ref_model = tg.TinyGPT(fsdp_demo.CFG)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
    ref_losses = []
    for b in fsdp_demo.batches():  # one process, the whole (concatenated) batch
        _, loss = ref_model(b[:, :-1], b[:, 1:])
        loss.backward()
        ref_opt.step()
        ref_opt.zero_grad()
        ref_losses.append(loss.item())
    r = fsdp_report
    print(f"parameters: {r['full_params']:,} in the model, {r['local_params']:,} on rank 0 "
          f"({r['local_params'] / r['full_params']:.0%})")
    print(f"AdamW state on rank 0: {r['local_optimizer_state']:,} elements "
          f"(= {r['local_optimizer_state'] / r['local_params']:.0f} x its parameter shard, not of the full model)")
    print("loss per step, FSDP (2 ranks):   ", [round(v, 5) for v in r["losses"]])
    print("loss per step, single process:   ", [round(v, 5) for v in ref_losses])
    print(f"max difference: {max(abs(a - b) for a, b in zip(r['losses'], ref_losses)):.1e}")
    RESULTS["fsdp_demo"] = {**r, "single_process_losses": ref_losses}

# %% [markdown]
# **Expected (CPU, measured):** the model has 118,528 parameters and rank 0 holds 59,264
# (50%); its AdamW state is 2 × its shard (two moments), not 2 × the model. The FSDP losses
# match single-process training on the full batch to ~1e-5 over 3 steps (our run: 1.1e-05 on
# losses of ~5.55, a relative 2e-6; Adam divides by √v, which magnifies rounding differences
# in tiny gradients): sharding changes *where* the state lives, not the maths. The script takes ~5–10 s, mostly process
# start-up. If it fails on an older PyTorch, the message above says why; everything else in
# the notebook is independent of it.
#
# ## 48.2 · Tensor parallelism: split a layer across devices
#
# ZeRO shards *storage* but still computes each layer on one GPU with the full weights
# (gathered just in time). **Tensor parallelism** (Megatron-LM, Shoeybi et al. 2019) splits
# the *computation* of each layer. For `y = x Wᵀ` with W of shape (out, in):
#
# * **column parallel**: split W's output rows; each device computes a slice of `y`'s
#   features from the full `x`; concatenate (all-gather) to get `y`.
# * **row parallel**: split W's input columns (and `x`'s features); each device computes a
#   *partial sum* of the whole `y`; add them up (all-reduce).
#
# A transformer MLP uses column-parallel for the first Linear and row-parallel for the
# second: the GELU in between is elementwise, so each device applies it to its own slice and
# the only communication is **one all-reduce of the output** per MLP (and one per attention
# block, whose heads are split across devices). We check this on CPU with two simulated
# "devices" (two halves of the weight matrices).

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-tensor-parallel-mlp.svg" alt="The input is replicated on two GPUs; each multiplies by half the columns of the first weight, applies GELU, multiplies by the matching half of the second weight's rows, and one all-reduce sums the two partial outputs" width="100%">
#
# *What to notice:* the GELU sits between the halves without any communication, because it acts on each feature separately; the only exchange is the final sum.

# %%
torch.manual_seed(0)
h, B_T = 64, 10  # hidden size, batch x sequence positions
x = torch.randn(B_T, h)
fc, proj = torch.nn.Linear(h, 4 * h), torch.nn.Linear(4 * h, h)

y_full = fc(x)
y_col, col_parts = D.column_parallel_linear(x, fc.weight, fc.bias, n_shards=2)
print(f"column parallel: shard outputs {[tuple(p.shape) for p in col_parts]} -> concatenated "
      f"{tuple(y_col.shape)}, max |diff| vs unsplit = {(y_col - y_full).abs().max().item():.1e}")

z = torch.nn.functional.gelu(y_full)
out_full = proj(z)
out_row, row_parts = D.row_parallel_linear(z, proj.weight, proj.bias, n_shards=2)
print(f"row parallel: partial sums {[tuple(p.shape) for p in row_parts]} -> summed, "
      f"max |diff| = {(out_row - out_full).abs().max().item():.1e}")

# The Megatron MLP: device d holds fc rows d and proj columns d; no communication until the end.
outs = []
for d in range(2):
    w1, b1 = fc.weight.chunk(2, 0)[d], fc.bias.chunk(2)[d]
    w2 = proj.weight.chunk(2, 1)[d]
    outs.append(torch.nn.functional.gelu(x @ w1.T + b1) @ w2.T)  # all local
mlp_tp = outs[0] + outs[1] + proj.bias  # <- the single all-reduce
print(f"Megatron MLP on 2 devices vs unsplit MLP: max |diff| = {(mlp_tp - out_full).abs().max().item():.1e}")
print(f"per device: {fc.weight.numel() // 2 + proj.weight.numel() // 2:,} MLP weights instead of "
      f"{fc.weight.numel() + proj.weight.numel():,}; all-reduce of {B_T}x{h} activations per MLP")

# %% [markdown]
# **Expected:** column-parallel shards of shape (10, 128) concatenate to (10, 256) and
# match the unsplit layer exactly (0.0: each output is computed by the same dot product);
# the row-parallel partial sums of shape (10, 64) add up to the unsplit output up to
# rounding (2.7e-07: the sum is split in two); the 2-device Megatron MLP matches the
# unsplit MLP (2.7e-07) with half the weights per device.
#
# The catch is communication: an all-reduce of `batch × seq × hidden` activations **twice
# per layer in forward and twice in backward**, on the critical path (it cannot overlap
# like DDP's gradient all-reduce). That needs NVLink-class bandwidth, so tensor parallelism
# is kept **inside a node** (typically 2–8 GPUs). Sequence parallelism and context
# parallelism split the sequence dimension in a similar spirit.
#
# ## 48.3 · Pipeline parallelism and the bubble
#
# **Pipeline parallelism** puts consecutive groups of layers ("stages") on different
# devices. A batch is split into *m* micro-batches; stage *s* works on micro-batch *i* while
# stage *s+1* works on micro-batch *i−1*. At the start and end of each batch some stages
# have nothing to do: the **bubble**. For p stages and m micro-batches (GPipe, and 1F1B
# which has the same bubble but less activation memory):
#
# $$\text{bubble fraction} = \frac{p-1}{m+p-1}$$
#
# Communication is cheap (only the activations at stage boundaries, point-to-point), so
# pipelines work *across* nodes. The cost is the bubble and the need for many
# micro-batches. Let us simulate a GPipe schedule (all forwards, then all backwards;
# backward takes 2 time units, forward 1) and count idle slots.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-pipeline-bubble.svg" alt="A GPipe schedule for 4 stages and 4 micro-batches: forward passes form a staircase, backward passes a reversed staircase, and the empty triangles are idle time, 3/7 of the step" width="100%">
#
# *What to notice:* the idle triangles have the same size whatever m is; more micro-batches make the busy part longer, so the bubble's *share* shrinks.
#
# > **Intuition.** A car-wash with four stations: the first car needs four steps to reach the
# > end, and the last stations stand idle while the line fills (and the first ones while it
# > drains). Washing 16 cars per batch instead of 4 makes the fill-and-drain a small part of the
# > day. *Where it breaks down:* every car in flight must have its activations stored until its
# > backward pass, so more micro-batches also cost memory (1F1B schedules limit this).

# %%
def gpipe_schedule(p, m, fwd=1, bwd=2):
    """Return a (p x T) grid of 'F' (forward), 'B' (backward) or '.' (idle) per time unit."""
    t_f = [[(s + i) * fwd for i in range(m)] for s in range(p)]  # forward of mb i on stage s starts
    end_f = (p - 1 + m) * fwd
    t_b = [[end_f + (p - 1 - s + i) * bwd for i in range(m)] for s in range(p)]  # backwards in reverse stage order
    T = end_f + (p - 1 + m) * bwd
    grid = [["."] * T for _ in range(p)]
    for s in range(p):
        for i in range(m):
            for k in range(fwd):
                grid[s][t_f[s][i] + k] = "F"
            for k in range(bwd):
                grid[s][t_b[s][i] + k] = "B"
    return grid


for p, m in [(4, 4), (4, 16)]:
    grid = gpipe_schedule(p, m)
    idle = sum(row.count(".") for row in grid) / (len(grid) * len(grid[0]))
    print(f"p={p} stages, m={m} micro-batches: simulated idle fraction {idle:.3f}, "
          f"formula (p-1)/(m+p-1) = {D.pipeline_bubble_fraction(p, m):.3f}")
    if m == 4:
        for s, row in enumerate(grid):
            print(f"  stage {s}: " + "".join(row))

# %% [markdown]
# **Expected:** for p = 4, m = 4 both give 0.429 (3/7: almost half the time idle), and the
# picture shows the staircase of forwards (`F`), the reversed staircase of backwards (`B`)
# and the idle triangles (`.`); for m = 16 the bubble falls to 0.158. Rule of thumb:
# m ≥ 4p keeps it under ~20%. Interleaved schedules (each device holds several
# non-consecutive stages) shrink it further at the cost of more communication.
#
# **Putting it together (3D parallelism):** large runs combine tensor parallelism inside a
# node (e.g. 8-way), pipeline parallelism across a few nodes, and data parallelism (often
# with ZeRO-1 or FSDP) across the replicas. Total GPUs = TP × PP × DP.
#
# ## 48.4 · Compute budgets: FLOPs ≈ 6·N·D
#
# Training a dense transformer with N parameters on D tokens costs about **6·N·D** FLOPs
# (Kaplan et al. 2020): 2N per token for the forward pass (a multiply and an add per
# weight) and 4N for the backward pass. Attention over the context adds a term
# proportional to layers × context × width, small for large models. We can *measure* the
# constant on the CPU with PyTorch's FLOP counter.

# %%
from torch.utils.flop_counter import FlopCounterMode

cfg = tg.GPTConfig(vocab_size=1000, block_size=64, n_embd=128, n_head=4, n_layer=4)
m_ = tg.TinyGPT(cfg)
N = sum(p.numel() for p in m_.parameters())
idx = torch.randint(0, cfg.vocab_size, (8, cfg.block_size))
with FlopCounterMode(display=False) as counter:
    _, loss = m_(idx, idx)
    loss.backward()
flops = counter.get_total_flops()
tokens = idx.numel()
print(f"TinyGPT: N = {N:,} parameters, {tokens} tokens -> counted {flops:.3e} FLOPs (forward + backward)")
print(f"FLOPs per token per parameter: {flops / tokens / N:.2f}   (6 is the rule of thumb)")
attn = 12 * cfg.n_layer * cfg.block_size * cfg.n_embd  # Kaplan et al.'s attention term, per token
print(f"6N + attention term 12 * layers * ctx * d = {(6 * N + attn) / N:.2f} per parameter")
RESULTS["flops_per_token_per_param_measured"] = flops / tokens / N

# %% [markdown]
# **Expected (CPU, measured):** 929,536 parameters, 3.01e9 FLOPs for 512 tokens, i.e.
# **6.33** FLOPs per token per parameter; with Kaplan's attention term the formula predicts
# 6.42 (the counter sees the masked-out attention entries too, and the embedding lookup costs
# no FLOPs). For billion-parameter models the attention term is a few percent, so 6·N·D is
# the figure commonly quoted.
#
# Turning FLOPs into time needs the hardware's peak throughput and the fraction of it
# you actually achieve, the **model FLOPs utilisation (MFU)**: useful model FLOPs per
# second / peak FLOPs per second. Well-tuned large-model training reaches roughly 30–50%
# MFU; small models, small batches and slow interconnects get much less.

# %% [markdown]
# > **Exercise 2.** Write `train_flops_and_days(n_params, tokens, gpu_tflops, mfu, n_gpus)` returning a
# > tuple `(flops, days)`: the training FLOPs (6·N·D) and the **wall-clock** days on `n_gpus` GPUs, each
# > with a peak of `gpu_tflops` TFLOP/s (10¹² FLOP/s) running at utilisation `mfu` (a fraction).

# %%
def train_flops_and_days(n_params, tokens, gpu_tflops, mfu, n_gpus):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(train_flops_and_days)

# %% [markdown]
# Worked examples. **Assumptions, stated:** peak dense throughput from NVIDIA datasheets
# (A100: 312 TFLOP/s bf16/fp16; T4: 65 TFLOP/s fp16 with Tensor Cores), an MFU we choose
# (40% on A100 clusters, 25% for a single T4 since small batches and memory limits rarely
# let it run near peak), and model/token counts from the cited papers.

# %%
examples = [  # name, N, D, GPU, peak TFLOP/s, MFU, GPUs, source of N and D
    ("GPT-3 175B", 175e9, 300e9, "A100", 312, 0.40, 1024, "Brown et al. 2020"),
    ("Chinchilla 70B", 70e9, 1.4e12, "A100", 312, 0.40, 1024, "Hoffmann et al. 2022"),
    ("7B on 1T tokens", 7e9, 1e12, "A100", 312, 0.40, 256, "illustrative"),
    ("125M, 20 tok/param", 125e6, 2.5e9, "T4", 65, 0.25, 1, "illustrative"),
]
RESULTS["budgets"] = []
for name, n, d, gpu, tf, mfu, g, src in examples:
    fl, days = D.train_flops_and_days(n, d, tf, mfu, g)
    RESULTS["budgets"].append({"name": name, "flops": fl, "days": days, "gpu_days": days * g, "gpus": g,
                               "gpu": gpu, "mfu": mfu})
    print(f"{name:20} {fl:9.2e} FLOPs | {g:5d} x {gpu} @ {mfu:.0%} MFU -> {days:8.1f} days "
          f"({days * g:10,.0f} GPU-days)  [{src}]")

# %% [markdown]
# **Expected (arithmetic):** GPT-3-sized training is 3.15e23 FLOPs (the paper reports
# 3.14e23), ~29 days on 1,024 A100s at 40% MFU; Chinchilla 70B is 5.9e23 FLOPs, ~53 days on
# the same cluster; 7B on 1T tokens is 4.2e22 FLOPs, ~15 days on 256 A100s (~3,900
# GPU-days); a 125M model on 2.5B tokens is 1.9e18 FLOPs, ~1.3 days on one T4 at 25% MFU.
# Multiply GPU-days by an hourly price to get a cost. The "GPT-3 on A100" numbers are a
# what-if: GPT-3 was trained on V100s.
#
# ## 48.5 · Scaling laws: how big a model for a given budget?
#
# Published empirical results (cite them as such; do not extrapolate them blindly):
#
# * **Kaplan et al. 2020** ("Scaling Laws for Neural Language Models", OpenAI): test loss
#   falls as a power law in parameters N, data D and compute C over many orders of
#   magnitude, with small exponents (≈0.076 for N, ≈0.095 for D). Their compute-optimal
#   recipe grew the model much faster than the data.
# * **Hoffmann et al. 2022** ("Training Compute-Optimal Large Language Models", DeepMind,
#   "Chinchilla"): with learning-rate schedules matched to each run length, the
#   compute-optimal N and D grow in **equal proportion** (each ∝ C^0.5), about **20 tokens per
#   parameter**. Chinchilla (70B params, 1.4T tokens) outperformed Gopher (280B, 300B
#   tokens) trained with about the same compute.
# * Since then, models meant to be **served** are often trained far past 20 tokens/param
#   (e.g. LLaMA-1 7B on 1T tokens, Touvron et al. 2023): extra training compute buys a
#   smaller model that is cheaper at inference time, which the compute-optimal recipe ignores.
#
# Combining C = 6·N·D with D = 20·N gives N = √(C / 120):

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-scaling-frontier.svg" alt="Three U-shaped loss curves against model size for budgets C, 10C and 100C, with the minima joined by a dashed compute-optimal frontier; beside them C = 6ND, D = 20N and N = sqrt(C/120)" width="100%">
#
# *What to notice:* for a fixed budget, a model that is too big sees too few tokens and one that is too small cannot use them; the optimum moves right (bigger) as compute grows, but only as √C.
#
# > **Intuition: where 6 comes from.** Every training token visits every weight three times: in
# > the forward pass (activation × weight), in backward to get the activation gradient, and in
# > backward to get the weight gradient. Each visit is a multiply and an add: $3 \times 2 = 6$
# > FLOPs per parameter per token.

# %%
print(f"{'compute (FLOPs)':>16} | {'params':>10} | {'tokens':>10}")
for c in (1e19, 1e21, 1e23, 5.9e23, 1e25):
    n_opt, d_opt = D.chinchilla_optimal(c)
    print(f"{c:16.1e} | {n_opt / 1e9:9.2f}B | {d_opt / 1e9:9.0f}B")

# %% [markdown]
# **Expected (arithmetic):** 1e21 FLOPs → ~2.9B params on ~58B tokens; 5.9e23 (Chinchilla's
# budget) → ~70B params on ~1.4T tokens, recovering the published configuration; 1e25 →
# ~289B on ~5.8T tokens. In practice: given a budget, size the model and data; given a
# model, say whether it is under- or over-trained relative to 20 tokens/param, and why that
# may be deliberate.
#
# ## 48.6 · Fault tolerance: how often to checkpoint?
#
# A job on n nodes fails n times as often as one node (independent failures):
# MTBF_job = MTBF_node / n. Meta's Llama 3 paper (Dubey et al. 2024) reports 419
# unexpected interruptions in a 54-day period of pre-training on up to 16,384 H100 GPUs:
# roughly one every 3 hours. Every failure loses the work since the last checkpoint plus
# the restart time; every checkpoint costs time to write (unless written asynchronously).
#
# With checkpoint cost C, interval τ and MTBF M, the expected fraction of time wasted is
# about **C/τ + (τ/2 + R)/M** (writing + on average half an interval redone after each
# failure + restart R), minimised at **Young's (1974) interval τ* = √(2·C·M)**. Daly (2006)
# refined it (≈ √(2CM) − C) for when C is not small compared with M.

# %% [markdown]
# > **Exercise 3.** Write `optimal_checkpoint_interval(checkpoint_cost_s, mtbf_s)` returning Young's
# > optimal interval in seconds.

# %%
def optimal_checkpoint_interval(checkpoint_cost_s, mtbf_s):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(optimal_checkpoint_interval)

# %% [markdown]
# Now check the formula against a simulation. **Assumptions:** each 8-GPU node fails on
# average once every 50 days (an illustrative figure; measure your own fleet), the job uses
# 128 nodes, writing a checkpoint takes 5 minutes, restarting takes 10 minutes, and the job
# needs 30 days of uninterrupted compute. Failures are exponential (memoryless).

# %%
NODE_MTBF_S, N_NODES, C_S, R_S, WORK_S = 50 * 86400, 128, 300.0, 600.0, 30 * 86400
M_S = D.cluster_mtbf(NODE_MTBF_S, N_NODES)
tau_young = D.young_interval(C_S, M_S)
print(f"job MTBF {M_S / 3600:.1f} h | Young's interval {tau_young / 60:.0f} min | "
      f"Daly's {D.daly_interval(C_S, M_S) / 60:.0f} min")


def simulate_wall_time(tau, rng, n_runs=200):
    """Monte Carlo: wall-clock time to finish WORK_S of compute with checkpoints every tau seconds."""
    totals = []
    for _ in range(n_runs):
        done, wall = 0.0, 0.0
        next_fail = rng.exponential(M_S)
        while done < WORK_S:
            seg = min(tau, WORK_S - done) + C_S  # compute one interval, then write a checkpoint
            if wall + seg <= next_fail:
                wall += seg
                done += seg - C_S
            else:  # failure: lose the partial interval, pay the restart
                wall = next_fail + R_S
                next_fail = wall + rng.exponential(M_S)
        totals.append(wall)
    return float(np.mean(totals))


rng = np.random.default_rng(0)
taus_min = [10, 20, 40, 60, 80, 120, 240, 480]
rows = []
for t_min in taus_min:
    wall = simulate_wall_time(t_min * 60, rng)
    model = D.expected_waste_fraction(t_min * 60, C_S, M_S, R_S)
    rows.append((t_min, wall))
    print(f"checkpoint every {t_min:4d} min: simulated {wall / 86400:6.2f} days "
          f"({wall / WORK_S - 1:+6.1%} vs no failures) | first-order model {model:+6.1%}")
best = min(rows, key=lambda r: r[1])[0]
print(f"best simulated interval: {best} min (Young: {tau_young / 60:.0f} min)")
RESULTS["checkpointing"] = {"job_mtbf_h": M_S / 3600, "young_min": tau_young / 60, "best_simulated_min": best}

# %% [markdown]
# **Expected (simulation, seeded):** job MTBF 9.4 h, Young's interval 75 min (Daly 70 min).
# The simulated overhead is large for very frequent checkpoints (10 min: +55%, mostly
# writing) and for rare ones (480 min: +63%, mostly redone work), and smallest at
# 60–80 minutes (+17%; our seeded run picked 80 min, next to Young's 75). The curve is flat near the optimum,
# so being within 2× of τ* costs little. The first-order model tracks the simulation
# but deviates at long intervals, where several failures can hit one interval.
#
# Levers beyond the formula: **asynchronous checkpointing** (copy the state to CPU memory
# in seconds, write to storage in the background: C shrinks, so τ* shrinks and waste
# drops), **sharded checkpoints** (each rank writes its own shard: `torch.distributed.checkpoint`),
# in-memory redundancy (checkpoint to a peer's RAM), hot spare nodes to cut R, and fast
# failure *detection* (a hang that waits 30 min for a timeout adds 30 min to R).
#
# ## 48.7 · Mixed precision numerics: fp16 vs bf16, and loss scaling
#
# Half precision halves memory and speeds up matmuls on Tensor Cores, but its *range*
# and *precision* differ:

# %%
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):15} max {fi.max:9.3e} | smallest normal {fi.tiny:9.3e} | "
          f"eps (relative precision) {fi.eps:.2e} | bits {fi.bits}")
print("fp16(70000)        ->", torch.tensor(70000.0).half().item(), "  (overflow)")
print("fp16(1e-8)         ->", torch.tensor(1e-8).half().item(), "   (underflow)")
print("bf16(1e-8)         ->", torch.tensor(1e-8).bfloat16().item())
print("fp16(1.0 + 1e-4)   ->", (torch.tensor(1.0).half() + torch.tensor(1e-4).half()).item(),
      "   (the update is lost: below eps/2)")
print("bf16(256 + 1)      ->", (torch.tensor(256.0).bfloat16() + 1).item(), "  (bf16 has only 8 bits of mantissa)")

# %% [markdown]
# **Expected:** fp16 max 65,504, smallest normal 6.1e-05 (subnormals go down to ~6e-8),
# eps 9.8e-04; bf16 has fp32's range (max 3.4e38) but eps 7.8e-03. 70,000 overflows to
# `inf` in fp16; 1e-8 underflows to 0 in fp16 but not in bf16; 1.0 + 1e-4 stays 1.0 in fp16
# (so weight updates are kept in an **fp32 master copy**); 256 + 1 = 256 in bf16.
#
# **Loss scaling.** Gradients are often tiny. In fp16 many of them underflow to zero
# and the corresponding weights stop learning. Multiplying the loss by a scale S (e.g.
# 2¹⁶) multiplies every gradient by S (chain rule), lifting them into fp16's range;
# the optimizer divides by S in fp32 before the update. Too large an S overflows to `inf`,
# so `torch.amp.GradScaler` adapts it: skip the step and halve S when an `inf`/`nan`
# appears, double S after 2,000 clean steps. We simulate the fp16 backward on CPU by casting
# the real fp32 gradients of a small TinyGPT (after scaling) to fp16.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-mixed-precision.svg" alt="A log-scale number line of gradient magnitudes with fp16's underflow, subnormal, normal and overflow zones; raw gradients centred near 1e-6 straddle the underflow zones, and multiplying by 2 to the 16 moves them into the normal range" width="100%">
#
# *What to notice:* multiplying by S is a pure shift on a log scale: the shape of the gradient distribution is unchanged, it just moves out of the red and yellow zones.

# %%
torch.manual_seed(0)
small = tg.TinyGPT(tg.GPTConfig(vocab_size=256, block_size=32, n_embd=64, n_head=4, n_layer=2))
xb = torch.randint(0, 256, (4, 33))
_, loss = small(xb[:, :-1], xb[:, 1:])
loss = loss * 1e-3  # e.g. a loss averaged over many tokens / accumulation steps: small gradients
loss.backward()
g32 = torch.cat([p.grad.flatten() for p in small.parameters()])
nonzero = g32 != 0
print(f"{g32.numel():,} gradient entries; |g| median {g32[nonzero].abs().median().item():.1e}, "
      f"min nonzero {g32[nonzero].abs().min().item():.1e}")
for scale in (1.0, 2.0**8, 2.0**16, 2.0**24, 2.0**32):
    g16 = (g32 * scale).half()  # what an fp16 backward pass would hold
    lost = ((g16 == 0) & nonzero).float().mean().item()
    inf = torch.isinf(g16).any().item()
    back = g16.float() / scale  # unscale in fp32
    ok = nonzero & (g16 != 0) & ~torch.isinf(g16)
    rel = ((back[ok] - g32[ok]).abs() / g32[ok].abs()).median().item() if ok.any() else float("nan")
    print(f"scale 2^{int(math.log2(scale)):2d}: underflowed to 0: {lost:6.1%} | inf present: {str(inf):5} | "
          f"median relative error of the rest {rel:.1e}")

# %% [markdown]
# **Expected (measured on CPU, simulated fp16):** the median gradient is ~1e-6. Without
# scaling, 10.2% of the gradient entries become 0 in fp16 and the rest lose precision as
# subnormals (median relative error 8e-3); with 2⁸ only 0.1% underflow and the error drops
# to fp16's ~2e-4; with 2¹⁶–2²⁴ essentially nothing underflows; at 2³² some gradients
# overflow to `inf`. That is why the scaler
# starts high (2¹⁶ by default) and backs off on `inf`.
#
# **bf16** has fp32's exponent range, so gradients do not underflow and no loss scaling is
# needed; its coarser precision is usually harmless for training (accumulations happen in
# fp32). That is why A100/H100 training uses bf16. **A T4 has no bf16 Tensor Cores**, so on
# Colab you use fp16 with `GradScaler`, which `scaleml.train` does automatically.

# %%
RESULTS["assumptions"] = {
    "zero": "mixed-precision Adam, 2+2+12 bytes/param, no activations",
    "peak_tflops": {"A100_bf16_dense": 312, "T4_fp16_tensor": 65},
    "mfu": {"A100_cluster": 0.40, "T4": 0.25},
    "node_mtbf_days": 50, "nodes": N_NODES, "checkpoint_s": C_S, "restart_s": R_S,
}
(OUT / "budget.json").write_text(json.dumps(RESULTS, indent=2, default=float))
print("wrote", OUT / "budget.json")

# %% [markdown]
# > **Exercise 4.** You are asked to plan pre-training of a 13B-parameter decoder on 260B tokens, on a
# > cluster of A100-80GB nodes (8 GPUs each, NVLink inside a node, InfiniBand between nodes). (a) Is 260B
# > tokens compute-optimal by the Chinchilla rule of thumb? (b) How much memory does the model state
# > need under DDP, and which sharding/parallelism would you use and why? (c) With 512 GPUs at 40% MFU,
# > how many days, and how many GPU-days? (d) Node MTBF is 30 days (assumed) and a checkpoint takes
# > 4 minutes: what checkpoint interval do you pick, and what else reduces the waste? Use the calculators
# > in this notebook and state every assumption. Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 4):* ...

# %% [markdown]
# ## Key equations
#
# **1. Model-state memory per GPU (mixed-precision Adam, $N$ data-parallel GPUs).**
#
# $$
# M_{\text{DDP}} = 16\Psi, \quad M_{\text{Z1}} = 4\Psi + \frac{12\Psi}{N}, \quad M_{\text{Z2}} = 2\Psi + \frac{14\Psi}{N}, \quad M_{\text{Z3}} = \frac{16\Psi}{N}
# $$
#
# In words: each stage shards one more of optimizer state (12 bytes), gradients (2) and
# parameters (2). Example: 7B on 8 GPUs: 112, 38.5, 26.25 and 14.0 GB.
#
# **2. Training compute and time.**
#
# $$
# C \approx 6\,N D, \qquad t \approx \frac{C}{n_{\text{GPU}} \cdot P_{\text{peak}} \cdot \text{MFU}}
# $$
#
# In words: 6 FLOPs per parameter per token; time is that divided by the FLOP/s you actually get.
# Symbols: $N$ parameters, $D$ training tokens. Example: 7B on 1T tokens: $4.2 \times 10^{22}$
# FLOPs; on 256 A100s (312 TFLOP/s) at 40% MFU, $4.2\times10^{22} / (256 \cdot 312\times10^{12} \cdot 0.4) = 1.3 \times 10^6$ s ≈ 15 days.
#
# **3. Chinchilla compute-optimal sizing.**
#
# $$
# D^{\star} \approx 20\, N^{\star} \;\;\Rightarrow\;\; C = 6 N^{\star} (20 N^{\star}) \;\;\Rightarrow\;\; N^{\star} = \sqrt{C / 120}
# $$
#
# Example: $C = 5.9 \times 10^{23}$: $N^{\star} \approx 70$B parameters, $D^{\star} \approx 1.4$T tokens.
#
# **4. Pipeline bubble.**
#
# $$
# \text{bubble} = \frac{p - 1}{m + p - 1}
# $$
#
# In words: $p - 1$ slots of fill and drain out of $m + p - 1$. Example: $p = 4$: 0.43 at $m = 4$,
# 0.16 at $m = 16$.
#
# **5. Tensor parallelism of $y = xW^\top$.**
#
# $$
# \text{column: } y = \big[\, x W_1^\top,\; x W_2^\top \,\big] \;(\text{all-gather}), \qquad
# \text{row: } y = x_1 W_1^\top + x_2 W_2^\top \;(\text{all-reduce})
# $$
#
# Example: an MLP $4C$ wide on 2 GPUs: each holds a $C \times 2C$ half of both matrices and
# exchanges one $B \times T \times C$ sum.
#
# **6. Optimal checkpoint interval (Young 1974; Daly 2006).**
#
# $$
# \text{waste}(\tau) \approx \frac{C}{\tau} + \frac{\tau/2 + R}{M}, \qquad \tau^{\star} = \sqrt{2\,C\,M} \;\;(\text{Daly: } \approx \sqrt{2CM} - C)
# $$
#
# In words: checkpoint cost per interval plus, per failure, half an interval of lost work and a
# restart. Symbols: $C$ checkpoint write time, $M$ job MTBF ($= M_{\text{node}} / n_{\text{nodes}}$),
# $R$ restart time. Example: $C = 5$ min, $M = 9.4$ h: $\tau^{\star} = \sqrt{2 \cdot 300 \cdot 33{,}840}$ s ≈ 75 min.
#
# **7. Loss scaling.**
#
# $$
# \nabla_\theta (S \cdot \mathcal L) = S \cdot \nabla_\theta \mathcal L, \qquad \text{fp16 normal range} \approx [6.1 \times 10^{-5},\; 65{,}504]
# $$
#
# In words: scale the loss up so gradients are representable in fp16, then divide by $S$ in fp32
# before the update. Example: a gradient of $10^{-8}$ becomes 0 in fp16, but $10^{-8} \cdot 2^{16} \approx 6.6 \times 10^{-4}$
# is a normal fp16 number.
#
# The next cell checks 1, 3, 4 and 6 against the course's `scaleml.distributed` helpers.

# %%
psi, n_gpu = 7e9, 8
formula = {0: 16 * psi, 1: 4 * psi + 12 * psi / n_gpu, 2: 2 * psi + 14 * psi / n_gpu, 3: 16 * psi / n_gpu}
for stage, value in formula.items():
    print(f"ZeRO-{stage}: formula {value / 1e9:6.2f} GB | D.zero_memory_per_gpu "
          f"{D.zero_memory_per_gpu(psi, n_gpu, stage) / 1e9:6.2f} GB")

budget_flops = 5.9e23
n_opt, d_opt = D.chinchilla_optimal(budget_flops)
print(f"Chinchilla: sqrt(C/120) = {math.sqrt(budget_flops / 120) / 1e9:.1f}B | helper {n_opt / 1e9:.1f}B params, "
      f"{d_opt / 1e12:.2f}T tokens, 6ND = {6 * n_opt * d_opt:.2e}")
print(f"bubble p=4, m=16: formula {(4 - 1) / (16 + 4 - 1):.3f} | helper {D.pipeline_bubble_fraction(4, 16):.3f}")
mtbf = D.cluster_mtbf(50 * 86400, 128)
print(f"Young: sqrt(2 * 300 s * {mtbf / 3600:.1f} h) = {math.sqrt(2 * 300 * mtbf) / 60:.0f} min | "
      f"helper {D.young_interval(300, mtbf) / 60:.0f} min")

# %% [markdown]
# **Expected:** formula and helper agree for every line: 112.00 / 38.50 / 26.25 / 14.00 GB;
# 70.1B parameters on 1.40T tokens (6ND = 5.9e23); bubble 0.158; Young's interval 75 min for a
# 9.4-hour job MTBF.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | FSDP demo: `ImportError: cannot import name 'fully_shard'` | PyTorch older than 2.4 | upgrade PyTorch, or set `RUN_FSDP_DEMO = False`; the rest of the notebook does not need it |
# | FSDP demo hangs or times out | leftover processes holding resources; a very busy CPU | restart the runtime; it finishes in ~10 s on a free Colab CPU |
# | FSDP losses differ from single-process | different initialisation per rank (seed each rank the same before wrapping), or a per-rank batch that is not the matching half | `torch.manual_seed(0)` before building the model on every rank |
# | Your memory estimate is far below the real OOM point | activations, temporary all-gather buffers, the CUDA context and fragmentation are not in the 16 bytes/param | measure (Chapter 40); budget 20–40% headroom |
# | fp16 training: loss becomes `nan` | overflow in forward (large activations) or no loss scaling | `GradScaler`; check for `inf` in activations; keep softmax/LayerNorm in fp32 (autocast does); prefer bf16 where supported |
# | `GradScaler` scale keeps shrinking | gradients overflow every step (a real divergence, not scaling) | lower the learning rate, clip gradients, inspect the data |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. How much GPU memory does training a 7B model with mixed-precision Adam need, and how do ZeRO stages 1, 2 and 3 change the per-GPU number on 64 GPUs? What do you pay for stage 3?
# 2. Compare data, tensor and pipeline parallelism: what is split, what is communicated, and where in a cluster (inside a node or across nodes) does each belong?
# 3. Derive the pipeline bubble fraction. How do you make it small, and what does that cost?
# 4. Estimate the training compute and wall-clock time of a 7B model on 1T tokens on 256 A100s. Which assumption in your estimate is most uncertain?
# 5. What did the Chinchilla paper change about how we size models, and why do many production models deliberately train past the compute-optimal point?
# 6. Why does fp16 training need loss scaling and bf16 training not? What does the dynamic loss scaler do when it sees `inf`?
#
# **Next chapter:** [Chapter 49 · Quantisation & distillation](../ch49-quantization-and-distillation/notebook.ipynb) — from training big models to serving small ones: int8 quantisation and knowledge distillation, with accuracy, latency and size measured.
