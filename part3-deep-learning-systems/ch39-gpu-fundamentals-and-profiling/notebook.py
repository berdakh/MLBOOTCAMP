# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 39 · GPU fundamentals & profiling
#
# **Part III · Chapter 39 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Most questions about training speed, whether asked in an interview or raised
# when a training job runs slowly, reduce to three questions: is the hardware
# busy, is it busy with useful arithmetic, and if not, what is it waiting for?
# This lab builds a working model of how a GPU executes code and the measuring
# tools needed to answer those questions with numbers rather than guesses.
#
# After this lab you should be able to:
#
# 1. explain how a GPU executes work (SMs, warps, kernels, asynchronous launches) and why that changes how you time code;
# 2. time GPU code correctly (warm-up, `synchronize`, median of repeats);
# 3. compute achieved TFLOP/s and arithmetic intensity, and say from a roofline whether an operation is memory-bound or compute-bound;
# 4. know when fp16 autocast helps on a T4 and why;
# 5. read a `torch.profiler` table and name the bottleneck;
# 6. diagnose and fix an input-pipeline bottleneck with `num_workers`, prefetching and pinned memory.
#
# **You will produce:** `artifacts/level3/nb01/benchmarks.json` (your matmul and
# elementwise measurements, with the device they came from) and the helper module
# [`src/scaleml/profiling.py`](../../src/scaleml/profiling.py), reused in Chapter 40.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Every number
# quoted in an **Expected:** line below was printed by a CPU run unless it says
# *"on a T4 you should see … (not measured in our verification runs)"*. When you
# run on a T4, your cells print the GPU numbers.

# %% [markdown]
# Read first: [Chapter 39 text](README.md) explains the ideas behind this lab (GPU fundamentals & profiling) in more depth; [solutions](solutions.md) has worked answers.

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

# Part III is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing to rebuild; this is the first Part III notebook --------
import json
import time

import torch
from torch import nn

from scaleml import profiling as prof
from scaleml.device import autocast, seed_everything, sync, timer
from grading import ch39 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))  # like Colab's 2 vCPUs; also fair on shared machines
seed_everything(0)
OUT = ART / "level3" / "nb01"
OUT.mkdir(parents=True, exist_ok=True)
IS_CUDA = DEVICE.type == "cuda"
print(f"torch {torch.__version__} | device {DEVICE} | CPU threads {torch.get_num_threads()}")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Kernel | one function the GPU runs on thousands of threads; every PyTorch op launches at least one | `a @ b` launches one matrix-multiply (GEMM) kernel |
# | Asynchronous launch | the CPU puts a kernel in a queue and carries on without waiting for the result | the "without sync" time in 39.2 is only the launch |
# | Synchronize | make the CPU wait until every queued GPU kernel has finished | `torch.cuda.synchronize()`, inside `timer` and `prof.benchmark` |
# | FLOP/s | floating-point operations (multiplies, adds) per second: the speed of arithmetic | a T4 peaks near 8 TFLOP/s in fp32, 65 in fp16 |
# | Memory bandwidth | how many bytes per second can move between GPU memory and its cores | ~320 GB/s on a T4 |
# | Arithmetic intensity | FLOPs done per byte moved to or from memory | `x + y`: 1/12; a 1024 × 1024 matmul: ~170 |
# | Memory-bound / compute-bound | limited by moving bytes / limited by doing arithmetic | elementwise add / large matmul |
# | Roofline | the plot of attainable FLOP/s against arithmetic intensity | ridge point ≈ 25 FLOP/byte for a T4 in fp32 |
# | Tensor Core | a unit that multiplies small fp16 matrices many times faster than ordinary cores | why fp16 autocast speeds up a T4 (39.5) |
# | Profiler | a tool that records which operations took the time | the `torch.profiler` table in 39.6 |
# | Pinned memory | page-locked host RAM the GPU can copy from directly, in the background | `pin_memory=True` + `non_blocking=True` |
# | Worker / prefetch | a background process that prepares the next batches while the model trains | `num_workers=2` in 39.7 |

# %% [markdown]
# ## 39.1 · How a GPU executes your code
#
# A CPU has a few (2 on Colab, 8–64 on a workstation) powerful cores, each
# optimised to finish *one* thread fast: big caches, branch prediction,
# out-of-order execution. A GPU has many simple cores, organised to keep
# *thousands* of threads in flight and to hide memory latency by switching
# between them.
#
# | Concept | What it is | On a T4 (spec sheet) |
# |---|---|---|
# | **SM** (streaming multiprocessor) | an independent processor with its own registers, shared memory/L1 cache and schedulers | 40 SMs |
# | **CUDA core** | an fp32 lane inside an SM | 64 per SM → 2,560 |
# | **Tensor Core** | a unit that does small matrix multiply-accumulates (fp16 in, fp16/fp32 accumulate) | 320; ~65 TFLOP/s fp16 peak vs ~8 TFLOP/s fp32 |
# | **warp** | 32 threads that execute the *same instruction* together (SIMT); divergent branches serialise | — |
# | **kernel** | one function launched on a grid of thread blocks; each block runs on one SM | every PyTorch op launches ≥ 1 |
# | **HBM/GDDR memory** | the GPU's own DRAM | 16 GB GDDR6, ~320 GB/s |
# | **PCIe link** | the pipe between host (CPU) RAM and GPU memory | Gen3 x16, ~16 GB/s theoretical |
#
# Three consequences follow, and they shape the rest of Chapters 39–40:
#
# * **Asynchrony.** `torch.matmul` on a CUDA tensor does not compute anything
#   on the CPU: it *enqueues* a kernel on a CUDA stream and returns
#   immediately. The Python thread races ahead and only waits when it needs a
#   result (`.item()`, `.cpu()`, `print(tensor)`, or an explicit
#   `torch.cuda.synchronize()`). Timing without synchronising measures the
#   *launch*, not the work.
# * **Launch overhead.** Each kernel launch costs a few microseconds of CPU
#   time. A model made of thousands of tiny ops is limited by Python and
#   launches, not by the GPU ("CPU-bound" or "overhead-bound").
# * **Memory bandwidth vs arithmetic.** A T4 can do roughly 25 fp32 FLOPs (or
#   about 200 fp16 Tensor-Core FLOPs) in the time it reads one byte from its
#   memory. Operations that do little arithmetic per byte (elementwise ops,
#   LayerNorm, softmax) cannot use that compute: they are **memory-bound**.
#
# ## 39.2 · Host ↔ device transfers and asynchronous execution
#
# Data must be copied from host RAM to GPU memory over PCIe before a kernel can
# use it. A normal ("pageable") host tensor is first copied by the driver into
# a page-locked staging buffer, then DMA'd to the GPU. A **pinned**
# (page-locked) tensor skips the staging copy and allows
# `.to("cuda", non_blocking=True)` to overlap the copy with compute.

# %%
if IS_CUDA:
    x_host = torch.randn(64 * 2**20 // 4)  # 64 MB of fp32
    x_pinned = x_host.pin_memory()
    for name, src in [("pageable", x_host), ("pinned", x_pinned)]:
        r = prof.benchmark(lambda src=src: src.to(DEVICE, non_blocking=True), DEVICE, warmup=2, iters=10)
        print(f"host->device {name:8}: {64 / 1024 / r['median_s']:.1f} GB/s")
else:
    print("CPU runtime: there is no host->device copy to measure (the 'device' is host RAM).")
    print("On a T4 you should see roughly 3-6 GB/s pageable and 10-13 GB/s pinned "
          "(not measured in our verification runs, CPU only).")

# %% [markdown]
# **Expected:** on CPU, the skip message. On a T4, pinned copies should be
# clearly faster than pageable ones (not measured in our verification runs).
# Either way, 64 MB at ~12 GB/s is ~5 ms: copying a batch is cheap; copying
# the *dataset* every step, or shuttling activations back and forth with
# `.cpu()` inside the loop, is not.
#
# Now the timing trap. The cell below times a large matmul twice: once by
# reading the clock right after the call, once with `synchronize` on both sides
# (which is what `scaleml.device.timer` does).

# %%
n = 2048 if IS_CUDA else 1024
a = torch.randn(n, n, device=DEVICE)
b = torch.randn(n, n, device=DEVICE)
for _ in range(3):  # warm-up: the first CUDA call pays for context creation and kernel selection
    c = a @ b
sync(DEVICE)

start = time.perf_counter()
c = a @ b  # on CUDA this only *launches* the kernel
naive = time.perf_counter() - start

with timer(DEVICE, results := {}):  # synchronize before and after
    c = a @ b
print(f"{n}x{n} matmul on {DEVICE}: without sync {naive * 1e3:.3f} ms | with sync {results['seconds'] * 1e3:.3f} ms")

# %% [markdown]
# **Expected:** on CPU both numbers are about the same (our runs: 13–21 ms
# for a 1024×1024 matmul with 2 threads), because CPU ops are synchronous. On a
# T4 you should see the "without sync" time in the tens of microseconds (only the launch)
# and the "with sync" time in milliseconds (not measured in our verification runs).
#
# A correct timing function therefore needs: **warm-up** calls (first-call
# costs), **`synchronize` before** starting the clock (so earlier queued work is
# not billed to you) **and after** (so your own work is), **several repeats**,
# and the **median** (robust to a background hiccup).

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-async-launch.svg" alt="CPU launches kernels A, B, C instantly while the GPU runs them later; a timer read without synchronize measures only the launches" width="100%">
#
# *What to notice:* the red bracket ends long before the GPU has done any of the work; only a `synchronize` before the second clock read makes the green bracket cover the kernels.
#
# > **Intuition.** Launching a kernel is like a waiter handing an order to the kitchen: the
# > waiter (the CPU) is back at your table in seconds, the food (the result) arrives later.
# > Timing the waiter tells you nothing about the cooking; `synchronize` is waiting at the pass
# > until every order is out. *Where it breaks down:* a CUDA stream cooks strictly in order, one
# > kernel after another, which a real kitchen does not.

# %% [markdown]
# > **Exercise 1.** Write `time_fn(fn, device, warmup, iters)`: call `fn()` `warmup` times
# > without timing, then time `iters` calls one by one (synchronising `device` before and after
# > each call; `scaleml.device.sync` does the right thing on any device), and return the
# > **median** in seconds as a float. The grader uses a function whose warm-up calls and one
# > timed call are slow.

# %%
def time_fn(fn, device, warmup, iters):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(time_fn)

# %% [markdown]
# `prof.benchmark` in [`src/scaleml/profiling.py`](../../src/scaleml/profiling.py)
# is the same function, plus a `min_seconds` option that keeps repeating very
# fast operations until enough time has been measured.
#
# ## 39.3 · Throughput: TFLOP/s of a matmul
#
# An `n × n` by `n × n` matrix multiply computes `n²` outputs, each a dot
# product of length `n` (`n` multiplies and `n` adds): **2n³ FLOPs**. Divide by
# the measured time to get achieved FLOP/s, and compare with the hardware peak.

# %% [markdown]
# > **Exercise 2.** Write `tflops(n, seconds)`: achieved TFLOP/s (10¹² FLOP/s) of an `n × n`
# > matmul that took `seconds`. Return a float.

# %%
def tflops(n, seconds):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(tflops)

# %%
SIZES = [256, 512, 1024, 2048, 4096, 8192] if IS_CUDA else [128, 256, 512, 1024, 2048]
mm_rows = prof.bench_matmul(SIZES, DEVICE, torch.float32)
print(f"fp32 matmul on {DEVICE}")
print(f"{'n':>6} {'ms':>9} {'TFLOP/s':>9} {'FLOP/byte':>10}")
for r in mm_rows:
    print(f"{r['n']:>6} {r['ms']:>9.3f} {r['tflops']:>9.3f} {r['flops_per_byte']:>10.1f}")

# %% [markdown]
# **Expected (CPU, 2 threads, measured):** roughly 0.1–0.3 TFLOP/s (our runs:
# 0.13–0.26 at n ≥ 1024). On a CPU, small matrices fit in cache and can look as
# fast as big ones, and single sizes (512 in one of our runs) can dip on a shared
# machine: that is why we repeat and take the median, and why you should re-run a
# surprising number before explaining it.
#
# **On a T4 you should see** fp32 throughput rise to a plateau around 4–7
# TFLOP/s from n ≈ 2048, below the ~8 TFLOP/s spec peak (sustained clocks are
# lower than boost clocks under the 70 W power cap). Small sizes (256, 512) may be
# 10–100× lower: the GPU is mostly idle and you are measuring launch overhead.
# (Not measured in our verification runs, CPU only.)
#
# ## 39.4 · Arithmetic intensity and the roofline
#
# **Arithmetic intensity** = FLOPs performed ÷ bytes moved to and from memory.
# The **roofline model** says achievable FLOP/s =
# `min(peak FLOP/s, intensity × memory bandwidth)`. The *ridge point* (peak ÷
# bandwidth) separates memory-bound operations (left) from compute-bound ones
# (right). For a T4: fp32 ridge ≈ 8.1 TFLOP/s ÷ 320 GB/s ≈ 25 FLOP/byte; fp16
# Tensor-Core ridge ≈ 65 ÷ 0.32 ≈ 200 FLOP/byte.
#
# * `x + y` on fp32 vectors: 1 FLOP per element, 12 bytes moved (read 2 × 4,
#   write 4) → intensity 1/12. Memory-bound by a wide margin on any hardware.
# * `n × n` matmul in fp32: 2n³ FLOPs, at least 3n² × 4 bytes → n/6. Compute-bound
#   once n/6 exceeds the ridge point (n ≳ 150 on a T4 in fp32).

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-roofline.svg" alt="Log-log roofline for a T4: a sloped memory roof up to the ridge at 25 FLOP per byte, then a flat compute roof at 8.1 TFLOP/s; x + y and LayerNorm sit on the slope, a large matmul on the flat part" width="100%">
#
# *What to notice:* an operation's position on the x-axis is fixed by its maths, before you run anything; everything left of the dashed ridge line cannot go faster than bandwidth allows, however many FLOPs the chip has.
#
# > **Intuition.** A factory with a fast assembly line (compute) fed by a narrow conveyor belt
# > (memory). If each part needs one screw (`x + y`), the line waits for the belt; if each part
# > needs hundreds of operations (a big matmul), the belt keeps up and the line is the limit.
# > *Where it breaks down:* caches let a kernel reuse bytes for free, so the intensity a kernel
# > achieves depends on how it is written, not only on the maths (that is what fusion exploits).

# %% [markdown]
# > **Exercise 3.** Write `arithmetic_intensity(flops, bytes)`: FLOPs per byte, as a float.
# > Raise `ValueError` if `bytes <= 0`.

# %%
def arithmetic_intensity(flops, bytes):  # noqa: A002 - the name the exercise uses
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(arithmetic_intensity)

# %%
ew = prof.bench_elementwise(2**24, DEVICE)  # 16M fp32 values = 64 MB per tensor
big = mm_rows[-1]
print(f"x + y   : {ew['flops_per_byte']:.3f} FLOP/byte -> {ew['tflops'] * 1e3:8.2f} GFLOP/s "
      f"({ew['gb_per_s']:.1f} GB/s of memory traffic)")
print(f"matmul  : {big['flops_per_byte']:.1f} FLOP/byte -> {big['tflops'] * 1e3:8.2f} GFLOP/s (n={big['n']})")
print(f"matmul does {big['tflops'] / ew['tflops']:.0f}x more useful FLOP/s than the elementwise add on {DEVICE}")
# The bandwidth ceiling the add hit, used as this device's memory roof:
ridge_here = big["tflops"] * 1e12 / (ew["gb_per_s"] * 1e9)
print(f"rough ridge point on this device: {ridge_here:.0f} FLOP/byte "
      "(achieved matmul FLOP/s / achieved add bandwidth)")

(OUT / "benchmarks.json").write_text(json.dumps(
    {"device": describe(DEVICE), "matmul_fp32": mm_rows, "elementwise_add": ew}, indent=2))
print("wrote", OUT / "benchmarks.json")

# %% [markdown]
# **Expected (CPU, measured):** the add moves only ~5–10 GB/s of memory traffic,
# i.e. under 1 GFLOP/s (our runs: 5–9 GB/s, 0.4–0.8 GFLOP/s), while the matmul reaches
# ~130–270 GFLOP/s: about 300× more useful arithmetic per second (our runs: 315–346×),
# and a rough ridge point of 25–30 FLOP/byte. The add is memory-bound, the matmul
# compute-bound, on the same chip.
#
# **On a T4 you should see** the add at roughly 200–280 GB/s (close to the
# 320 GB/s spec) yet only ~0.02 TFLOP/s, and the matmul at several TFLOP/s
# (not measured in our verification runs).
#
# This is the reason **kernel fusion** matters: `gelu(x @ W + b)` as three separate
# kernels writes and re-reads the intermediate tensor twice; a fused kernel (what
# `torch.compile` produces, Chapter 40) reads it once. It is also why
# FlashAttention is fast: attention's softmax is memory-bound, and FlashAttention
# never writes the `seq × seq` matrix to GPU memory.
#
# ## 39.5 · fp16 autocast on a T4
#
# T4 Tensor Cores only run fp16 (and int8): fp32 matmuls use the much slower
# CUDA cores. `torch.autocast(device_type="cuda", dtype=torch.float16)` runs
# matmuls and convolutions in fp16 and keeps precision-sensitive ops (softmax,
# reductions, losses) in fp32. In training, a **GradScaler** multiplies the loss
# so that small fp16 gradients do not underflow to zero, then unscales before
# the optimizer step. The T4 has **no bf16** support (Ampere and later do; bf16 has fp32's
# range, so it needs no scaler). `scaleml.device.autocast` and `amp_dtype` pick
# the right thing for your hardware.

# %%
if IS_CUDA:
    for n in (1024, 2048, 4096):
        x = torch.randn(n, n, device=DEVICE)
        y = torch.randn(n, n, device=DEVICE)
        r32 = prof.benchmark(lambda x=x, y=y: x @ y, DEVICE, 2, 10)

        def amp_mm(x=x, y=y):
            with autocast(DEVICE):
                return x @ y

        r16 = prof.benchmark(amp_mm, DEVICE, 2, 10)
        print(f"n={n}: fp32 {prof.matmul_flops(n) / r32['median_s'] / 1e12:5.1f} TFLOP/s | "
              f"autocast fp16 {prof.matmul_flops(n) / r16['median_s'] / 1e12:5.1f} TFLOP/s")
else:
    print("Skipped on CPU: CPU autocast uses bf16 and rarely speeds anything up; the point of this "
          "cell is Tensor Cores.")
    print("On a T4 you should see autocast fp16 matmuls several times faster than fp32 at n >= 2048 "
          "(not measured in our verification runs, CPU only).")

# %% [markdown]
# **Expected:** on CPU, the skip message. On a T4 you should see fp16 at
# roughly 20–45 TFLOP/s for large n vs 4–7 for fp32 (not measured in our
# verification runs). The autocast time includes casting the fp32 inputs to
# fp16 on every call; in a real model the weights are cast once per forward.
#
# ## 39.6 · Profiling a training step with `torch.profiler`
#
# Timers tell you *how long*; a profiler tells you *where*. `torch.profiler`
# records every op (with `record_shapes=True`, its input shapes), its CPU time
# and, on a GPU, its kernel time. **Self** time excludes children; **total**
# time includes them. Sorting by self time finds the ops that actually burn time.

# %%
from torch.profiler import ProfilerActivity, profile, record_function

model = nn.Sequential(nn.Linear(512, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
                      nn.Linear(2048, 10)).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
xb = torch.randn(256, 512, device=DEVICE)
yb = torch.randint(0, 10, (256,), device=DEVICE)


def train_step():
    with record_function("forward"):
        loss = nn.functional.cross_entropy(model(xb), yb)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss


for _ in range(3):  # warm up outside the profiler
    train_step()
activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if IS_CUDA else [])
with profile(activities=activities, record_shapes=True) as p:
    for _ in range(5):
        train_step()
    sync(DEVICE)
sort_key = "self_cuda_time_total" if IS_CUDA else "self_cpu_time_total"
print(p.key_averages().table(sort_by=sort_key, row_limit=12, max_name_column_width=40))
phases = {e.key: e.cpu_time_total / 1e3 for e in p.key_averages() if e.key in ("forward", "backward", "optimizer")}
print("CPU time per phase over 5 steps (ms):", {k: round(v) for k, v in phases.items()})

# %% [markdown]
# **Expected (CPU, measured):** `aten::mm` and `aten::addmm` (the Linear layers,
# forward and backward) are the top two rows, 30–45% of self CPU time together
# in our runs. The next block is less expected: `aten::sqrt`, `div`, `mul_`,
# `addcdiv_`, `lerp_`, `addcmul_`, `add_` are the **AdamW update**, and together
# they take about as much time as the matmuls (`Optimizer.step#AdamW.step` total: ~40%). The optimizer step touches
# every one of the 5.3M parameters several times with a few FLOPs each:
# memory-bound elementwise work, exactly the `x + y` case from 1.4. On a GPU,
# `torch.optim.AdamW(..., fused=True)` (or the default multi-tensor `foreach`
# path) fuses these into a few kernels.
# The last line sums the `record_function` labels per phase. Our runs printed roughly
# forward 180, backward 180, optimizer 230 ms over 5 steps: the optimizer was the
# largest phase. The textbook rule "backward ≈ 2× forward" (two matmuls per Linear
# instead of one) did not hold here; small-model CPU timings are full of such
# surprises, which is the point of profiling instead of reasoning alone.
#
# **On a T4** the table gains CUDA columns. With this small model you should
# see GPU self time dominated by GEMM kernels, but also CPU time per step
# comparable to GPU time: launch overhead matters at this size (not measured
# in our verification runs).
#
# Two further profiler features are useful in practice:
# `p.key_averages(group_by_input_shape=True)` splits rows by shape (find the one
# odd-shaped matmul), and `p.export_chrome_trace("trace.json")` gives a timeline
# you open in `chrome://tracing` or Perfetto, where **gaps** between GPU kernels
# are the clearest evidence of a CPU or data-loading bottleneck.

# %% [markdown]
# > **Exercise 4.** Read the table below (constructed for this exercise, from a
# > typical shape of a real run, *not* measured here). It profiles 20 training steps of an image
# > classifier on one GPU. Name the bottleneck, the evidence for it, and two changes you
# > would try first, in order. Write your answer in the cell below.

# %%
PROFILE_TABLE = """
-----------------------------------------------  ------------  ------------  ------------  ------------
Name                                              Self CPU %    Self CPU      Self CUDA %   Self CUDA
-----------------------------------------------  ------------  ------------  ------------  ------------
enumerate(DataLoader)#_SingleProcessDataLoaderI      71.4%        6.912s          0.0%        0.000us
aten::copy_                                            6.2%      600.1ms          4.1%      48.20ms
cudaStreamSynchronize                                  9.8%      948.7ms          0.0%        0.000us
aten::convolution_backward                             1.1%      106.5ms         38.0%     446.9ms
aten::cudnn_convolution                                0.8%       77.4ms         27.5%     323.4ms
aten::addmm                                            0.2%       19.4ms          3.3%      38.8ms
Optimizer.step#AdamW.step                              2.0%      193.6ms          6.9%      81.1ms
aten::item                                             0.1%        9.7ms          0.0%        0.000us
-----------------------------------------------  ------------  ------------  ------------  ------------
Self CPU time total: 9.681s
Self CUDA time total: 1.176s
"""
print(PROFILE_TABLE)

# %% [markdown]
# *Your answer (Exercise 4):* ...
#
# ## 39.7 · The input pipeline: is the GPU waiting for data?
#
# In the table above, the GPU worked for 1.2 s out of ~9.7 s. The rest of the time it
# waited for the `DataLoader`. This is the most common real-world bottleneck,
# and it is a CPU problem. By default (`num_workers=0`) the main process loads
# and collates every batch *between* training steps, so loading and compute
# never overlap. With `num_workers=k`, `k` worker processes prepare batches in
# parallel and keep up to `prefetch_factor` (default 2) batches each queued
# ahead of the training loop.
#
# Below, `SlowDataset.__getitem__` sleeps 3 ms per example (standing in for JPEG
# decoding or tokenisation) and we simulate a 50 ms model step per batch.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-dataloader-overlap.svg" alt="Two timelines: with num_workers=0 the loop alternates loading and training and finishes 4 steps; with two prefetching workers it finishes 6 steps in the same time" width="100%">
#
# *What to notice:* nothing about the model changed; the second timeline wins only because loading happens *while* the GPU trains, so the train loop never waits.
#
# > **Intuition.** With `num_workers=0` the chef chops the vegetables, then cooks, then chops
# > again. Workers are kitchen assistants chopping the next order while the chef cooks. Adding
# > assistants helps until chopping keeps pace with cooking, and not after. *Where it breaks
# > down:* each assistant needs its own copy of the dataset object and time to start up.

# %%
from torch.utils.data import DataLoader

ds = prof.SlowDataset(n=384, delay_s=0.003)
STEP_S = 0.05  # pretend forward+backward takes 50 ms per batch
print(f"{len(ds)} examples, batch 32 -> 12 batches; loading ~{32 * 0.003 * 1e3:.0f} ms per batch, "
      f"compute {STEP_S * 1e3:.0f} ms per batch")
print(f"lower bound if loading fully overlaps compute: {12 * STEP_S:.2f} s")
loader_times = {}
for workers in (0, 1, 2):
    loader = DataLoader(ds, batch_size=32, num_workers=workers,
                        prefetch_factor=2 if workers else None, pin_memory=IS_CUDA)
    loader_times[workers] = prof.time_loader(loader, step_seconds=STEP_S)
    print(f"num_workers={workers}: {loader_times[workers]:.2f} s for one epoch")

# %% [markdown]
# **Expected (CPU, measured):** about 1.8–1.9 s with `num_workers=0` (12 × (96 ms
# load + 50 ms step)); about 1.2–1.4 s with one worker (loading overlaps compute, but one
# worker needs 96 ms per batch, more than the 50 ms step); about 0.7–0.9 s with two
# workers, close to the 0.6 s lower bound plus worker start-up (our runs: 1.85–1.86, 1.37–1.40, 0.80–0.83 s). Nothing about the *model* changed.
#
# What to know about the knobs:
#
# * **Colab has 2 vCPUs.** More than 2 workers rarely helps there; on a
#   training server, a common start is the number of physical cores per GPU.
#   Workers cost memory (each holds a copy of the dataset object) and start-up
#   time; `persistent_workers=True` keeps them alive across epochs.
# * **Prefetching** (`prefetch_factor`) only hides latency if the workers *on
#   average* keep up. If loading one batch takes longer than a training step even
#   with all workers, the fix is cheaper loading: pre-tokenise once, cache decoded data,
#   move augmentation to the GPU, store data in a format that reads in large blocks.
# * **`pin_memory=True`** makes the loader return batches in page-locked memory, so
#   `batch.to("cuda", non_blocking=True)` is a true asynchronous DMA that
#   overlaps with the previous step's kernels. Both halves are needed: pinned
#   source *and* `non_blocking=True`. It does nothing on a CPU-only runtime.
# * **Diagnose first.** Time an epoch that only iterates the loader (no model) and
#   one that reuses a single cached batch (no loader). If the first is close to the
#   full epoch time, the input pipeline is the bottleneck.
#
# On a T4 the same experiment matters more: a GPU step for a small model can
# be a few milliseconds, so a loader that needs 100 ms per batch leaves the GPU idle
# over 95% of the time (an arithmetic consequence of the numbers above, not a measurement).

# %% [markdown]
# ## Key equations
#
# A summary of the formulas used in this lab. Each entry gives the formula, its meaning in
# words, and a short worked example.
#
# **1. FLOPs of a matrix multiply.**
#
# $$
# \text{FLOPs}\big(A_{m \times k}\,B_{k \times n}\big) = 2\,m\,n\,k
# $$
#
# In words: each of the $m n$ outputs is a dot product of length $k$, i.e. $k$ multiplies and
# $k$ adds. Symbols: $m, k$ = rows and columns of $A$; $n$ = columns of $B$.
# Example: a 1024 × 1024 square matmul is $2 \cdot 1024^3 \approx 2.15 \times 10^9$ FLOPs; at
# 13 ms that is $2.15\times10^9 / 0.013 \approx 0.17$ TFLOP/s, the CPU number in 1.3.
#
# **2. Arithmetic intensity.**
#
# $$
# I = \frac{\text{FLOPs performed}}{\text{bytes moved to and from memory}}
# $$
#
# In words: how much arithmetic you get for every byte you read or write.
# Example: fp32 `x + y` does 1 FLOP per element and moves 4 + 4 + 4 = 12 bytes, so $I = 1/12$;
# an $n \times n$ fp32 matmul does $2n^3$ FLOPs on at least $3 \cdot 4 n^2$ bytes, so $I = n/6$
# (≈ 171 for $n = 1024$).
#
# **3. The roofline.**
#
# $$
# P_{\text{attainable}} = \min\big(P_{\text{peak}},\; I \cdot B_{\text{mem}}\big), \qquad I^{\star} = \frac{P_{\text{peak}}}{B_{\text{mem}}}
# $$
#
# In words: you cannot beat the chip's peak, and you cannot compute faster than memory can feed
# you. Symbols: $P_{\text{peak}}$ = peak FLOP/s, $B_{\text{mem}}$ = memory bandwidth (bytes/s),
# $I^{\star}$ = ridge point. Example (T4, fp32): $I^{\star} = 8.1\times10^{12} / 320\times10^{9} \approx 25$
# FLOP/byte; `x + y` can reach at most $320\times10^9 / 12 \approx 27$ GFLOP/s, 0.3% of peak.
#
# **4. Epoch time of an input pipeline.**
#
# $$
# T_{\text{epoch}} \approx n_{\text{batches}} \big(t_{\text{load}} + t_{\text{step}}\big) \;\;(\text{no workers}), \qquad
# T_{\text{epoch}} \approx n_{\text{batches}} \max\!\Big(\frac{t_{\text{load}}}{k},\; t_{\text{step}}\Big) \;\;(k \text{ workers})
# $$
#
# In words: without workers, loading and compute add up; with $k$ workers they overlap and the
# slower of the two sets the pace. Example (39.7): $12 \times (96 + 50)$ ms ≈ 1.75 s with no
# workers; with 2 workers $12 \times \max(48, 50)$ ms = 0.6 s, plus worker start-up (we measured
# 0.8 s).
#
# The next cell checks 1 and 3 numerically: PyTorch's FLOP counter against $2mnk$, and the
# roofline for the three operations in the figure above.

# %%
from torch.utils.flop_counter import FlopCounterMode

m_, k_, n_ = 64, 48, 32
with FlopCounterMode(display=False) as counter:
    torch.randn(m_, k_) @ torch.randn(k_, n_)
print(f"FLOP counter: {counter.get_total_flops():,} | formula 2mnk: {2 * m_ * n_ * k_:,}")

PEAK, BW = 8.1e12, 320e9  # T4 fp32 spec sheet: FLOP/s and bytes/s
print(f"ridge point: {PEAK / BW:.1f} FLOP/byte")
for name, intensity in [("x + y (fp32)", 1 / 12), ("LayerNorm (~1)", 1.0), ("matmul n=1024", 1024 / 6)]:
    attainable = min(PEAK, intensity * BW)
    bound = "compute" if attainable == PEAK else "memory"
    print(f"{name:15} I = {intensity:7.2f} FLOP/byte -> at most {attainable / 1e12:6.3f} TFLOP/s ({bound}-bound)")

# %% [markdown]
# **Expected:** the FLOP counter and the formula agree exactly (196,608); ridge point 25.3
# FLOP/byte; `x + y` at most 0.027 TFLOP/s and LayerNorm 0.32 (memory-bound), the matmul capped
# at the 8.1 TFLOP/s peak (compute-bound).

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | `describe` shows `cpu` in Colab | runtime type is CPU | Runtime → Change runtime type → T4 GPU, then Run all |
# | GPU timings are microseconds and do not grow with size | missing `torch.cuda.synchronize()` | time with `scaleml.device.timer` / `prof.benchmark` |
# | First measurement much slower than the rest | CUDA context creation, cuDNN/cuBLAS autotuning, allocator growth | warm up; report the median of repeats |
# | `RuntimeError: CUDA out of memory` in 39.3 | n=8192 needs 3 × 256 MB plus workspace; another notebook holds GPU memory | restart the runtime or drop 8192 from `SIZES` |
# | DataLoader with workers hangs or errors in a notebook | dataset defined in the notebook under the `spawn` start method (macOS/Windows), or a worker crashed | define the dataset in a module (like `SlowDataset`); set `num_workers=0` to see the real traceback |
# | Profiler table empty for CUDA columns | profiled on CPU or forgot `ProfilerActivity.CUDA` | check `DEVICE`; add the activity |
# | `gpuGetDeviceCount failed` printed by the profiler on a CPU runtime | the CUDA build of PyTorch probing for a GPU | harmless; ignore |
# | Timings vary a lot between runs | shared machine (Colab neighbours, other processes), CPU frequency scaling | repeat, use the median, compare within one session |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. A colleague times `y = model(x)` on a GPU with `time.time()` before and after and reports 0.1 ms for a large model. What is wrong, and how do you time it correctly?
# 2. Explain arithmetic intensity and the roofline model. Is LayerNorm memory-bound or compute-bound on a GPU? A 4096×4096 matmul? Why?
# 3. Why does fp16 autocast speed up training on a T4, why does fp16 training need a loss scaler, and why does bf16 not?
# 4. GPU utilisation in `nvidia-smi` hovers at 30% during training. List the likely causes in the order you would check them, and the tool you would use for each.
# 5. What do `num_workers`, `prefetch_factor`, `pin_memory` and `non_blocking=True` each do? When does adding workers stop helping?
# 6. A training step has thousands of tiny kernels and the GPU is idle between them. What is this bottleneck called and what are three ways to reduce it?
#
# **Next chapter:** [Chapter 40 · Training efficiency & memory](../ch40-training-efficiency-and-memory/notebook.ipynb) — now that you can measure time, measure memory: what fills a GPU during training and how to fit more into it.
