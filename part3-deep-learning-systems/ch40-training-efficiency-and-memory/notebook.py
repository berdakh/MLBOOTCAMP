# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 40 · Training efficiency & memory
#
# **Day 1 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# "CUDA out of memory" is the error you will see most often when training
# deep models, and "how much memory does training a 7B model need?" is a
# standard interview question. Both are answered by the same arithmetic: what
# lives in memory during a training step, how big each piece is, and which
# technique shrinks which piece.
#
# By the end you will be able to:
#
# 1. compute the memory of weights, gradients and optimizer state for any model and optimizer, with and without mixed precision;
# 2. estimate activation memory and explain why it scales with batch × sequence × hidden × layers (and seq² in attention);
# 3. check those estimates against measurements: summed tensor sizes and saved activations on CPU, `max_memory_allocated` on GPU;
# 4. implement gradient accumulation that gives exactly the same gradients as a large batch;
# 5. trade compute for memory with activation checkpointing, and try `torch.compile` safely;
# 6. debug an out-of-memory error methodically.
#
# **You will produce:** `artifacts/level3/nb02/memory_report.json` (estimated vs
# measured memory for a small transformer) and you will use
# [`src/scaleml/profiling.py`](../../src/scaleml/profiling.py) (memory helpers) and
# [`src/scaleml/tinygpt.py`](../../src/scaleml/tinygpt.py) (the model you build in notebook 03).
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*.

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
# --- Catch up: nothing from notebook 01 is needed; build the model here ------
import json
import time

import torch
from torch.utils.checkpoint import checkpoint

from scaleml import profiling as prof
from scaleml import tinygpt as tg
from scaleml.device import autocast, memory_report, seed_everything, sync
from grading import ch40 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
IS_CUDA = DEVICE.type == "cuda"
OUT = ART / "level3" / "nb02"
OUT.mkdir(parents=True, exist_ok=True)

# A GPT-style decoder sized from the budget (CPU: 128 wide, 4 layers; GPU: 512 wide, 8 layers).
VOCAB, HEADS = 1000, 4
CFG = tg.GPTConfig(vocab_size=VOCAB, block_size=BUDGET.max_seq_len, n_embd=2 * BUDGET.hidden,
                   n_head=HEADS, n_layer=2 * BUDGET.layers)
B, T = BUDGET.batch_size, BUDGET.max_seq_len


def new_model(seed=0):
    seed_everything(seed)
    return tg.TinyGPT(CFG).to(DEVICE)


def random_batch(batch=B, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (batch, T), generator=g)
    y = torch.randint(0, VOCAB, (batch, T), generator=g)
    return x.to(DEVICE), y.to(DEVICE)


model = new_model()
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"config: {CFG}")
print(f"{N_PARAMS:,} parameters | batch {B} x seq {T}")

# %% [markdown]
# **Expected (CPU budget):** `n_embd=128, n_layer=4`, 929,536 parameters,
# batch 32 × 64 tokens. On a T4 the GPU budget gives 512 wide, 8 layers (about 26M
# parameters), batch 64 × 128.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Parameter (Ψ) | one trainable number in the model; Ψ is the usual symbol for how many there are | 929,536 on the CPU budget |
# | Gradient | the slope of the loss for one parameter, computed by backward; same size as the weights | `p.grad`, 4 bytes per parameter in fp32 |
# | Optimizer state | extra numbers the optimizer keeps per parameter between steps | AdamW's two moments m and v: 8 bytes per parameter |
# | Activations | intermediate tensors from the forward pass that backward needs, kept until it uses them | 82 MB for one batch here, 5.8× the model state |
# | Mixed precision (autocast) | running matmuls in fp16/bf16 while keeping sensitive values in fp32 | `torch.autocast`; halves activations, not model state |
# | Micro-batch | one of the small pieces a large batch is split into | 4 micro-batches of 8 in 2.3 |
# | Gradient accumulation | adding the gradients of several micro-batches before one optimizer step | `TrainConfig(grad_accum_steps=k)` |
# | Activation checkpointing | keeping only some activations and recomputing the rest during backward | 82 MB → 14 MB in 2.4, for ~+30% time |
# | Peak vs leak | a single step that needs too much memory vs memory that grows every step | accumulating `loss` instead of `loss.item()` (2.6) |
# | Caching allocator | PyTorch keeps freed GPU memory reserved for reuse instead of returning it | `memory_reserved` > `memory_allocated` |
# | Kernel fusion | merging several elementwise ops into one kernel so the data is read once | what `torch.compile` does (2.5) |
# | OOM | out of memory: an allocation larger than what is free on the device | `torch.OutOfMemoryError` |

# %% [markdown]
# ## 2.1 · What fills memory during training
#
# During a training step, four things live in device memory:
#
# 1. **Weights** — fp32: 4 bytes per parameter.
# 2. **Gradients** — one per weight, same dtype: 4 bytes.
# 3. **Optimizer state** — SGD: none; SGD with momentum: one buffer (4 bytes);
#    Adam/AdamW: first and second moments (8 bytes).
# 4. **Activations** — every intermediate tensor the backward pass needs, kept from
#    forward until backward uses it. They scale with **batch × sequence × hidden ×
#    layers**, plus **batch × heads × seq²** for attention probabilities.
#
# Plus framework overhead you cannot remove: the CUDA context (~0.3–0.5 GB),
# cuBLAS/cuDNN workspaces, and allocator fragmentation.
#
# **Mixed precision does not shrink 1–3.** With `torch.autocast` (what this course
# and most PyTorch code use), parameters, gradients and optimizer state stay
# fp32; matmuls run in fp16, autocast caches an fp16 copy of each weight for the
# forward pass (+2 bytes/param), and *activations* roughly halve. Frameworks that
# keep fp16 weights and gradients plus an fp32 "master" copy (the ZeRO paper's
# accounting) get 2 + 2 + 4 + 8 = 16 bytes/param for Adam: the same total.
# The table used in this course (bytes per parameter, excluding activations):
#
# | optimizer | fp32 | mixed precision (autocast) |
# |---|---|---|
# | `sgd` | 8 | 10 |
# | `sgd_momentum` | 12 | 14 |
# | `adam` / `adamw` | 16 | 18 |
#
# So fine-tuning a 7B-parameter model with AdamW needs ~112 GB *before*
# activations. That arithmetic is why parameter-efficient fine-tuning (notebook 06)
# and sharded optimizers (ZeRO/FSDP, notebook 10) exist.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-training-memory.svg" alt="Stacked bars of bytes per parameter: fp32 AdamW 4 + 4 + 8 = 16; autocast adds a 2-byte fp16 copy for 18; ZeRO accounting 2 + 2 + 12 = 16; activations come on top" width="100%">
#
# *What to notice:* the Adam moments alone are as big as weights and gradients together, and switching to mixed precision moves bytes between the boxes without shrinking the total.
#
# > **Intuition.** Every parameter carries a backpack: its value (4 bytes), its gradient
# > (4 bytes) and Adam's two running averages (8 bytes). That is 16 bytes before any data goes
# > through the model. Activations are the scratch paper of the forward pass, kept until backward
# > reads them. *Where it breaks down:* the backpack is fixed per parameter, but the scratch paper
# > grows with batch size and sequence length, which is why activations usually dominate.

# %% [markdown]
# > **Exercise 1.** Write `training_memory_bytes(n_params, optimizer, mixed_precision)` that
# > returns the bytes for weights + gradients + optimizer state (+ the fp16 weight copy when
# > `mixed_precision`) using the table above, as an `int`. `optimizer` is one of `sgd`,
# > `sgd_momentum`, `adam`, `adamw` (any capitalisation); raise `ValueError` for anything else.

# %%
def training_memory_bytes(n_params, optimizer, mixed_precision):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(training_memory_bytes)

# %% [markdown]
# Now check the arithmetic against reality. `prof.memory_breakdown` is the
# same calculation (by category); `prof.measured_state_bytes` sums the sizes of the
# actual tensors after one AdamW step: every parameter, every `.grad`, and
# every tensor in `optimizer.state`.

# %%
model = new_model()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x, y = random_batch()
_, loss = model(x, y)
loss.backward()
opt.step()

est = prof.memory_breakdown(N_PARAMS, "adamw", mixed_precision=False)
got = prof.measured_state_bytes(model, opt)
print(f"{'':12} {'estimated':>12} {'measured':>12}")
for k in ("weights", "gradients", "optimizer"):
    print(f"{k:12} {prof.human_bytes(est[k]):>12} {prof.human_bytes(got[k]):>12}")
print(f"bytes/param: {sum(got.values()) / N_PARAMS:.2f}")

# %% [markdown]
# **Expected:** estimated and measured agree exactly (3.5 MB / 3.5 MB / 7.1 MB for the
# 929,536 parameters of the CPU budget), 16.00 bytes per parameter. Note the model has **weight tying**: the
# output layer shares the embedding matrix, and `model.parameters()` yields it
# once, so it is counted (and optimised) once.
#
# ## 2.2 · Activations: estimate, then measure
#
# Activations are harder to count by hand, but they can be measured on any
# device: autograd calls a *pack hook* for every tensor it saves for backward.
# `prof.SavedActivations` installs one (`torch.autograd.graph.saved_tensors_hooks`)
# and sums the unique storage it sees, excluding the parameters themselves.
#
# Our estimate for one decoder layer of `tinygpt` (see
# `prof.transformer_activation_bytes`): per token about **16 × hidden** values
# (LayerNorm inputs/outputs, Q, K, V, attention output, the 4×-wide MLP's GELU input
# and output) plus **heads × seq** attention probabilities. Megatron-LM's version of this
# formula, with dropout masks in fp16, is `34·h + 5·a·s` bytes per token per layer.

# %%
def saved_bytes(model, x, y, forward=None):
    with prof.SavedActivations(model) as acts:
        _, loss = (forward or model)(x, y)
    return acts.stats.bytes, loss


per_layer_est = prof.transformer_activation_bytes(B, T, CFG.n_embd, 1, HEADS)
rows = []
for n_layer in (1, 2, CFG.n_layer):
    seed_everything(0)
    m = tg.TinyGPT(tg.GPTConfig(VOCAB, T, CFG.n_embd, HEADS, n_layer)).to(DEVICE)
    measured, _ = saved_bytes(m, x, y)
    rows.append((n_layer, measured))
    print(f"{n_layer} layer(s): measured {prof.human_bytes(measured):>9} | "
          f"layer estimate {prof.human_bytes(per_layer_est * n_layer):>9}")
per_layer_measured = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
print(f"per layer: measured {prof.human_bytes(per_layer_measured)} vs estimate {prof.human_bytes(per_layer_est)}")
print(f"the rest (embeddings, final LayerNorm, logits {B}x{T}x{VOCAB}): "
      f"{prof.human_bytes(rows[0][1] - per_layer_measured)}")
ACT_BYTES = rows[-1][1]

# %% [markdown]
# **Expected (CPU budget, measured):** each extra layer adds 18.0 MB, and the
# estimate says 18.0 MB per layer. The fixed part (9.9 MB) is mostly the
# `batch × seq × vocab` logits' log-softmax that cross-entropy saves: with a
# 50k-token vocabulary (GPT-2) and long sequences that single tensor can be
# several GB, which is why LLM trainers compute the loss in chunks.
#
# Now the full picture for one step, and (on a GPU) a check against the CUDA
# caching allocator's own peak counter.

# %%
report = prof.memory_breakdown(N_PARAMS, "adamw", mixed_precision=False, activation_bytes=ACT_BYTES)
for k, v in report.items():
    print(f"{k:18} {prof.human_bytes(v):>10}")
print(f"activations are {ACT_BYTES / (report['total'] - ACT_BYTES):.1f}x the weights+grads+optimizer state")

measured_peak = None
if IS_CUDA:
    del model, opt
    torch.cuda.empty_cache()
    prof.reset_cuda_peak(DEVICE)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):  # the 2nd step has optimizer state allocated
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)  # keep grads allocated so the peak includes them
    sync(DEVICE)
    measured_peak = prof.cuda_peak_mb(DEVICE)
    print(f"CUDA peak allocated {measured_peak:.0f} MB vs estimate {report['total'] / 2**20:.0f} MB")
    print(memory_report(DEVICE))
else:
    print("CPU runtime: torch.cuda.max_memory_allocated is GPU-only; the tensor sums above are the CPU check.")

(OUT / "memory_report.json").write_text(json.dumps(
    {"device": str(DEVICE), "config": CFG.__dict__, "batch": B, "seq": T, "estimate_bytes": report,
     "measured_saved_activation_bytes": ACT_BYTES, "cuda_peak_mb": measured_peak}, indent=2))
print("wrote", OUT / "memory_report.json")

# %% [markdown]
# **Expected (CPU budget, measured):** activations 82 MB vs 14 MB of
# weights + gradients + AdamW state (5.8×). Even a tiny model with a
# modest batch is dominated by activations, so batch size and sequence length are the
# first memory levers, not the optimizer.
#
# **On a T4 you should see** the CUDA peak a bit above the estimate: temporaries
# (attention scores before softmax, the backward pass's own intermediates)
# and allocator rounding add 10–50% (not measured in our verification runs).
# `memory_reserved` is higher still: the caching allocator keeps freed blocks for reuse.
# That is normal and not a leak.
#
# ## 2.3 · Gradient accumulation: a big batch in small pieces
#
# If batch 256 does not fit but batch 32 does, run 8 micro-batches, calling
# `backward()` on each **without zeroing** (gradients add up in `.grad`), then
# step once. To get *the same* gradient as the big batch, each micro-batch's
# mean loss must be weighted by its share of the examples: `loss × n_i / N`
# (= `loss / k` when all `k` micro-batches have equal size). Activation memory
# is that of one micro-batch; the price is k sequential passes instead of one
# parallel one.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-grad-accumulation.svg" alt="Four micro-batches each run forward and backward, their weighted gradients add up in .grad, then one optimizer step" width="100%">
#
# *What to notice:* there is exactly one `optimizer.step()` and one `zero_grad()` per big batch; the weights `nᵢ / N` are what make the sum equal the big-batch gradient.
#
# > **Intuition.** Totalling a big bill from the receipts of several shopping trips: the sum is
# > the same as one big trip, as long as each receipt is counted with the right weight. *Where it
# > breaks down:* BatchNorm only ever sees one trip at a time, so its statistics differ from a
# > real big batch.

# %%
model = new_model()
xb, yb = random_batch(batch=4 * 8, seed=1)

model.zero_grad()
_, big_loss = model(xb, yb)
big_loss.backward()
big_grads = [p.grad.clone() for p in model.parameters()]

model.zero_grad()
for xm, ym in zip(xb.split(8), yb.split(8)):  # 4 micro-batches of 8
    _, loss = model(xm, ym)
    (loss / 4).backward()  # equal sizes: weight 8/32 = 1/4
acc_grads = [p.grad.clone() for p in model.parameters()]

max_diff = max((a - b).abs().max().item() for a, b in zip(big_grads, acc_grads))
max_grad = max(g.abs().max().item() for g in big_grads)
print(f"max |grad difference| = {max_diff:.2e} (largest gradient entry {max_grad:.2e})")

# %% [markdown]
# **Expected:** a difference around 1e-9–1e-8 (our run: 5.6e-09), rounding error (the sums happen in a
# different order), relative to gradient entries around 1e-2.
#
# Exact equivalence breaks when the model has **BatchNorm** (statistics come
# from the micro-batch), when micro-batches have **unequal sizes** but you divide by
# `k`, or when the loss is not a plain mean over examples (e.g. a mean over
# *tokens* with variable-length padding: weight by tokens, not examples).
# `scaleml.train.train` does accumulation with `TrainConfig(grad_accum_steps=k)`.

# %% [markdown]
# > **Exercise 2.** Write `accumulated_step(model, batches, loss_fn)`. `batches` is a list of
# > `(x, y)` micro-batches of **possibly different sizes**; `loss_fn(model, x, y)` returns the
# > mean loss over that micro-batch. Clear any old gradients, accumulate so that every
# > `p.grad` equals the gradient of the mean loss over *all* examples, and return that mean
# > loss as a Python float. Do not call an optimizer.

# %%
def accumulated_step(model, batches, loss_fn):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(accumulated_step)

# %% [markdown]
# ## 2.4 · Activation checkpointing: trade compute for memory
#
# Checkpointing a block means: in forward, keep only the block's *input* and
# discard its internal activations; in backward, run the block's forward again
# to recreate them. Memory for the checkpointed blocks drops from
# "all internal activations of every layer" to "one input per layer + one layer's
# internals at a time"; compute rises by roughly one extra forward (~+30% of a
# training step, since backward ≈ 2× forward). Gradients are unchanged.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-activation-checkpointing.svg" alt="Two rows of four blocks: standard training keeps all internal activations of every block; checkpointing keeps only each block's input and recomputes the inside during backward" width="100%">
#
# *What to notice:* the red boxes disappear but the compute does not: every dashed 'recompute' box is a second forward pass through that block during backward.
#
# > **Intuition.** Instead of photocopying every page of your working, you keep the first page
# > of each chapter and redo a chapter when you need its details again. Worked numbers: a step
# > costs about 1 forward + 2 forwards' worth of backward = 3 units; recomputing adds 1, so
# > 4/3 ≈ +33% time, for memory that falls from "all layers" to "one input per layer + one layer".

# %%
def forward_checkpointed(model, idx, targets):
    """Same as model(idx, targets), but each decoder block is checkpointed."""
    pos = torch.arange(idx.shape[1], device=idx.device)
    h = model.drop(model.tok_emb(idx) + model.pos_emb(pos))
    for block in model.blocks:
        h = checkpoint(block, h, use_reentrant=False)  # use_reentrant=False: the recommended mode
    logits = model.head(model.ln_f(h))
    return logits, torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))


def fwd_bwd(model, fwd):
    model.zero_grad(set_to_none=True)
    _, loss = fwd(x, y)
    loss.backward()


results = {}
for name in ("plain", "checkpointed"):
    model = new_model()
    fwd = model if name == "plain" else (lambda i, t, m=model: forward_checkpointed(m, i, t))
    saved, loss = saved_bytes(model, x, y, forward=fwd)
    loss.backward()
    grads = [p.grad.clone() for p in model.parameters()]
    timing = prof.benchmark(lambda m=model, f=fwd: fwd_bwd(m, f), DEVICE, warmup=1, iters=5)
    peak = None
    if IS_CUDA:
        prof.reset_cuda_peak(DEVICE)
        fwd_bwd(model, fwd)
        peak = prof.cuda_peak_mb(DEVICE)
    results[name] = {"saved": saved, "s": timing["median_s"], "grads": grads, "peak_mb": peak}
    print(f"{name:13} saved activations {prof.human_bytes(saved):>9} | fwd+bwd {timing['median_s'] * 1e3:6.1f} ms"
          + (f" | CUDA peak {peak:.0f} MB" if peak else ""))

same = max((a - b).abs().max().item() for a, b in zip(results["plain"]["grads"], results["checkpointed"]["grads"]))
print(f"memory saved: {1 - results['checkpointed']['saved'] / results['plain']['saved']:.0%} | "
      f"time cost: {results['checkpointed']['s'] / results['plain']['s'] - 1:+.0%} | max grad diff {same:.1e}")

# %% [markdown]
# **Expected (CPU budget, measured):** saved activations fall from 82.0 MB to
# 13.9 MB (83% less: what remains is one input per block plus the
# logits), the forward+backward time rises by roughly 20–40% (our runs: +23% and +40%;
# noisy on a shared CPU), and gradients are identical (difference 0).
#
# The saved-tensor count above is what stays alive between forward and
# backward. The true *peak* with checkpointing also includes one block's
# activations while it is being recomputed. **On a T4 you should see** the CUDA
# peak drop by less than the saved-activation ratio suggests, for that reason
# (not measured in our verification runs).
#
# In practice you checkpoint every block (Hugging Face:
# `model.gradient_checkpointing_enable()`) or every k-th block, and spend the
# freed memory on a larger batch or longer sequences.
#
# ## 2.5 · `torch.compile`
#
# `torch.compile(model)` traces the Python code into a graph (TorchDynamo),
# then generates fused kernels for it (TorchInductor: Triton on GPU, C++/OpenMP on
# CPU). The first call is slow (compilation: tens of seconds to minutes); later calls
# with the same shapes reuse the compiled code. Gains come from fusing memory-bound
# elementwise ops and from removing Python overhead; matmuls already run at library speed.
# It can fail (unsupported Python, a missing C++ compiler, an old GPU for Triton),
# so production code keeps an eager fallback.
#
# On CPU, compiling this training step took about 25 s the first time in our runs
# (8 s on re-runs, when Inductor's on-disk cache already had the kernels). Set
# `TRY_COMPILE = False` to skip it.

# %%
TRY_COMPILE = True

if TRY_COMPILE:
    model = new_model()
    eager = prof.benchmark(lambda: fwd_bwd(model, model), DEVICE, warmup=2, iters=10)["median_s"]
    try:
        compiled = torch.compile(model)
        t0 = time.perf_counter()
        fwd_bwd(model, compiled)  # first call compiles forward and backward
        sync(DEVICE)
        compile_s = time.perf_counter() - t0
        comp = prof.benchmark(lambda: fwd_bwd(model, compiled), DEVICE, warmup=2, iters=10)["median_s"]
        breakeven = f"~{compile_s / (eager - comp):,.0f} steps" if comp < eager else "never (no speed-up)"
        print(f"compile time {compile_s:.1f} s | eager {eager * 1e3:.1f} ms/step | compiled {comp * 1e3:.1f} ms/step "
              f"| speed-up {eager / comp:.2f}x | break-even after {breakeven}")
    except Exception as exc:  # keep the notebook running; report why
        print(f"torch.compile failed ({type(exc).__name__}: {str(exc)[:200]}); falling back to eager.")
else:
    print("Skipped (TRY_COMPILE = False).")

# %% [markdown]
# **Expected (CPU, measured):** compilation takes ~10–60 s and, for this small
# model on CPU, the compiled step is about as fast as eager (our runs: 0.95×–1.09×). The matmuls dominate and were already efficient; there is little to fuse
# that matters at this size. "Break-even" is how many steps it takes to win back the
# compile time; with no speed-up, it is never: compile is not free.
#
# **On a T4 you should see** a larger gain for transformer training (often
# 1.2–2×, from fused LayerNorm/GELU/softmax and fewer launches), with a compile
# time of a minute or more (not measured in our verification runs). Changing input
# shapes (e.g. variable sequence lengths) triggers recompilation; pad to a few
# fixed lengths or use `dynamic=True`.
#
# ## 2.6 · An OOM debugging playbook
#
# When you hit `torch.OutOfMemoryError: CUDA out of memory`:
#
# 1. **Read the message.** It states how much was requested, how much is
#    allocated by PyTorch and how much is reserved but free. Reserved ≫ allocated
#    means *fragmentation*: try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
# 2. **Is it a leak or a peak?** Print `torch.cuda.memory_allocated()` at the
#    start of each step. If it *grows* step after step, something keeps tensors
#    alive (below). If it is flat and one step spikes, it is a peak: reduce it.
# 3. **Reduce the peak**, cheapest first: smaller micro-batch + gradient
#    accumulation (2.3) → mixed precision (halves activations) → shorter sequences
#    or truncation → activation checkpointing (2.4) → smaller/8-bit optimizer
#    state, LoRA (notebook 06) → sharding (FSDP/ZeRO, notebook 10).
# 4. **Evaluation OOM but training fine?** Missing `torch.no_grad()` /
#    `torch.inference_mode()`, so eval builds graphs; or eval batch size set larger.
# 5. **After an OOM in a notebook**, the failed step's tensors may still be
#    referenced (by the exception traceback, by `loss` variables). `del` them,
#    `gc.collect()`, `torch.cuda.empty_cache()`; if in doubt, restart the runtime.
#    `empty_cache` only returns *unused cached* blocks to the driver; it never frees
#    live tensors and does not make training use less memory.
# 6. **Find what is big:** `torch.cuda.memory_summary()`, or record a timeline with
#    `torch.cuda.memory._record_memory_history()` and view it at
#    `pytorch.org/memory_viz`.
#
# The classic leak, made visible on CPU: accumulating the loss *tensor* (with
# its graph) instead of a number. Each step's activations stay reachable through the
# graph and are never freed.

# %%
import gc
import weakref


def leaked_activation_bytes(keep_tensor: bool, steps: int = 5) -> int:
    model = new_model()
    alive = {}  # storage pointer -> bytes, removed when the tensor is freed
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}

    def pack(t):
        t = t.detach()  # an alias without grad_fn, so the hook itself creates no reference cycle
        ptr = t.untyped_storage().data_ptr()
        if ptr not in params and ptr not in alive:
            alive[ptr] = t.untyped_storage().nbytes()
            weakref.finalize(t, alive.pop, ptr, None)
        return t

    total = 0.0
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        for step in range(steps):  # an "evaluation" loop written without torch.no_grad()
            xs, ys = random_batch(batch=8, seed=step)  # small batch: this cell leaks on purpose
            loss = model(xs, ys)[1]  # don't keep the logits either: they reference the graph too
            total = total + (loss if keep_tensor else loss.item())
            del loss
    gc.collect()
    return sum(alive.values())


for keep in (False, True):
    held = leaked_activation_bytes(keep)
    print(f"total += {'loss       ' if keep else 'loss.item()'} -> activations still alive after 5 steps: "
          f"{prof.human_bytes(held)}")

# %% [markdown]
# **Expected (CPU budget, measured):** 0 B with `loss.item()`; about 100 MB with the
# tensor (5 steps × ~20 MB of activations for a batch of 8), because `total`
# references every step's graph. On a GPU
# this is the "memory grows every step until OOM" pattern. Same family: appending
# `outputs` or `loss` to a list for later, or storing predictions without
# `.detach().cpu()`.

# %% [markdown]
# > **Exercise 3.** You are fine-tuning a 1.3B-parameter decoder with AdamW on one 16 GB T4
# > at sequence length 1024 and it runs out of memory at batch size 1. (a) Using the table in
# > 2.1, how much memory do weights + gradients + optimizer state need, in fp32 and with
# > autocast? (b) Can *any* combination of the techniques in this notebook make full
# > fine-tuning fit? (c) What would you do instead? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1. Model-state memory.**
#
# $$
# M_{\text{state}} = \Psi \,\big(b_{w} + b_{g} + b_{\text{opt}}\big)
# $$
#
# In words: every parameter pays for its weight, its gradient and its optimizer state.
# Symbols: $\Psi$ = number of parameters; $b_w, b_g, b_{\text{opt}}$ = bytes per parameter for
# each. AdamW in fp32: $4 + 4 + 8 = 16$; mixed-precision Adam (ZeRO accounting): $2 + 2 + 12 = 16$;
# PyTorch autocast: $16 + 2 = 18$. Example: $7\times10^9 \times 16$ B = 112 GB; the CPU budget's
# 929,536 parameters × 16 B = 14.9 MB.
#
# **2. Activation memory of a transformer** (Korthikanti et al. 2022, fp16, no checkpointing).
#
# $$
# M_{\text{act}} \approx L \cdot B \cdot T \cdot \big(34\,h + 5\,a\,T\big) \;\text{bytes}
# $$
#
# In words: linear in layers, batch and hidden size, and **quadratic in sequence length**
# through the attention probabilities. Symbols: $L$ layers, $B$ batch, $T$ sequence length,
# $h$ hidden size, $a$ attention heads. Example: $L=12, h=768, a=12, T=1024, B=8$: per token per
# layer $34 \cdot 768 + 5 \cdot 12 \cdot 1024 = 87{,}552$ bytes, × 8,192 tokens × 12 layers ≈ 8.6 GB,
# 70% of it from the $T$-squared term.
#
# **3. Gradient accumulation.**
#
# $$
# g = \sum_{i=1}^{k} \frac{n_i}{N}\, \nabla_\theta \bar{\ell}_i, \qquad N = \sum_{i=1}^{k} n_i
# $$
#
# In words: weight each micro-batch's mean-loss gradient by its share of the examples. Symbols:
# $\bar\ell_i$ = mean loss of micro-batch $i$, $n_i$ its size. Example: micro-batches of 8 and 24
# get weights 0.25 and 0.75; dividing both by $k = 2$ (0.5 and 0.5) gives the wrong gradient.
#
# **4. Adam / AdamW with bias correction.**
#
# $$
# \begin{aligned}
# m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
# \hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
# \theta_t &= \theta_{t-1} - \eta \Big( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\, \theta_{t-1} \Big)
# \end{aligned}
# $$
#
# In words: $m$ is a running average of the gradient (momentum), $v$ of its square (scale); the
# step is momentum divided by the typical gradient size, so every parameter moves at a similar
# rate. The bias correction undoes the zero initialisation of $m$ and $v$. AdamW applies the
# weight decay $\lambda\theta$ directly ("decoupled"), not through the gradient. Symbols:
# $\eta$ learning rate, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon \approx 10^{-8}$.
# Example: at $t = 1$, $m_1 = 0.1\,g$ and $\hat m_1 = 0.1\,g / (1 - 0.9) = g$, likewise
# $\hat v_1 = g^2$, so the first update is $\eta \cdot \operatorname{sign}(g)$: **every weight moves by
# the full learning rate, whatever its gradient**, which is one reason for warm-up.
#
# **5. Activation checkpointing cost.**
#
# $$
# \text{step time} \approx F + 2F \;\longrightarrow\; F + 2F + F = \tfrac{4}{3} \times \text{(no checkpointing)}
# $$
#
# In words: backward costs about two forwards; recomputing adds one more. Example: +33% in
# theory; we measured +23% to +40% in 2.4, for 83% less saved activation memory.
#
# The next cell checks equation 4 against `torch.optim.AdamW` for three steps on a toy problem.

# %%
torch.manual_seed(0)
w = torch.randn(5, requires_grad=True)
lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.01
adamw = torch.optim.AdamW([w], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
theta, m, v = w.detach().clone(), torch.zeros(5), torch.zeros(5)
for t in range(1, 4):
    adamw.zero_grad()
    (w ** 2).sum().backward()          # loss = sum(w^2), so g = 2w
    adamw.step()
    g = 2 * theta                       # the same gradient, by hand
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
    step = lr * (m_hat / (v_hat.sqrt() + eps) + wd * theta)
    if t == 1:
        print("first-step |update| per weight:", step.abs().round(decimals=4).tolist())
    theta = theta - step
print("max |manual - torch.optim.AdamW| after 3 steps:", (theta - w.detach()).abs().max().item())

# %% [markdown]
# **Expected:** the first update is ~0.10 (the learning rate) for every weight, plus a tiny
# weight-decay term (our run: 0.1003–0.1022), whatever the size of its gradient; the manual and
# PyTorch weights agree to float32 rounding (our run: 2.4e-07).

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | `torch.OutOfMemoryError: CUDA out of memory` | peak over 15 GB; or a previous cell's model still on the GPU | follow 2.6; `del model, opt; gc.collect(); torch.cuda.empty_cache()`; restart the runtime |
# | Memory grows every step | storing tensors with graphs (`loss`, outputs) | `.item()`, `.detach()`, `torch.no_grad()` in eval |
# | Accumulated gradients differ from the big batch | dividing by the number of micro-batches with unequal sizes; BatchNorm; dropout with different random masks | weight by `n_i / N`; compare in `eval()` mode or with dropout off |
# | `torch.compile` errors (`InductorError`, missing `cl`/`gcc`, Triton errors) | no C++ compiler on CPU; unsupported GPU or Python code | `TRY_COMPILE = False`; the eager model is unchanged |
# | `torch.compile` slow every step | recompiling on every new input shape | fixed shapes (pad), `dynamic=True`, or check `TORCH_LOGS=recompiles` |
# | Checkpointing warning about `use_reentrant` | the old default | pass `use_reentrant=False` explicitly |
# | `max_memory_allocated` much lower than `nvidia-smi` | `nvidia-smi` shows reserved cache + CUDA context | compare like with like: `memory_reserved` |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. How much GPU memory does full fine-tuning of a 7B model with AdamW need for weights, gradients and optimizer state, in fp32 and with mixed precision? Which of these does mixed precision actually shrink?
# 2. What determines activation memory in a transformer, and how does it scale with batch size, sequence length, hidden size and number of layers?
# 3. Gradient accumulation: when is it exactly equivalent to a larger batch, and when is it not? What does it cost?
# 4. Explain activation checkpointing. What do you save, what do you pay, and where would you place checkpoints?
# 5. Your training job's memory grows every step and eventually OOMs. Walk through how you find the cause.
# 6. What does `torch.compile` do, when does it give a speed-up, and what are its operational risks?
#
# **Next chapter:** [`03_transformer_from_scratch.ipynb`](../ch41-transformer-from-scratch/notebook.ipynb) — build, train and sample from the decoder you have been measuring, and make generation fast with a KV cache.
