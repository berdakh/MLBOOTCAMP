# Solutions · Chapter 40 · Training efficiency & memory

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `training_memory_bytes`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
STATE_BYTES = {"sgd": 0, "sgd_momentum": 4, "adam": 8, "adamw": 8}

def training_memory_bytes(n_params, optimizer, mixed_precision):
    key = optimizer.lower()
    if key not in STATE_BYTES:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    return int((4 + 4 + STATE_BYTES[key] + (2 if mixed_precision else 0)) * n_params)
```

**Check against measurement.** In 40.1 the summed tensors after one AdamW step
were exactly 16.00 bytes/param (3.5 + 3.5 + 7.1 MB for 929,536 parameters).

**Variants an interviewer may ask for.** Plain-SGD inference-only: 4 B/param
(fp32) or 2 (fp16). The ZeRO paper's mixed-precision Adam: fp16 weights 2 +
fp16 grads 2 + fp32 master weights 4 + two fp32 moments 8 = 16 B/param.
8-bit Adam (bitsandbytes) stores moments in 1 byte each: 4 + 4 + 2 = 10 B/param in fp32.
None of these include activations, the CUDA context or fragmentation.

---

## Exercise 2 · `accumulated_step`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def accumulated_step(model, batches, loss_fn):
    model.zero_grad(set_to_none=True)
    total = sum(len(x) for x, _ in batches)
    mean_loss = 0.0
    for x, y in batches:
        loss = loss_fn(model, x, y) * (len(x) / total)
        loss.backward()
        mean_loss += loss.item()
    return float(mean_loss)
```

**Why the weight is `len(x) / total`.** The big-batch loss is
`(1/N) Σ_all ℓ_j = Σ_i (n_i/N) · mean_{j∈i} ℓ_j`. Gradients are linear, so
backpropagating each weighted micro-batch loss and letting `.grad` add up
gives exactly the big-batch gradient (up to floating-point summation order;
40.3 measured a 5.6e-09 difference). Dividing by the number of micro-batches
is only right when all have the same size.

**Common mistakes.** Not clearing old gradients first; dividing by
`len(batches)`; calling `.backward()` once on a summed loss after the loop
(correct gradients, but all micro-batch graphs stay alive at once, which defeats the
memory saving); returning a tensor instead of a float.

---

## Exercise 3 · Fitting a 1.3B model on a T4

[open-ended] (a) Weights + gradients + AdamW state: 1.3e9 × 16 B ≈ **20.8 GB** in fp32,
1.3e9 × 18 B ≈ **23.4 GB** with autocast. That is more than the T4's 16 GB *before any
activation*.

(b) **No, not for full fine-tuning with AdamW on one T4.** Gradient
accumulation, activation checkpointing, shorter sequences and mixed precision all
shrink *activations*, which were not the problem: even batch size 1 with zero
activations needs >20 GB. Only changes to the 16 B/param term help: a
stateless or 8-bit optimizer (SGD: 8 B/param = 10.4 GB, tight; 8-bit Adam: ~13 GB, still
too tight with activations and the CUDA context), or offloading optimizer state to CPU
RAM (ZeRO-Offload; slow on Colab's PCIe and 12 GB of host RAM).

(c) **Parameter-efficient fine-tuning** (LoRA, Chapter 44): freeze the base
weights (fp16: 2.6 GB, no gradients, no optimizer state), train adapters
with ~0.1–1% of the parameters; add activation checkpointing for seq 1024. Or
QLoRA (4-bit base weights, ~0.7 GB). Or pick a smaller model. The general lesson:
**do the arithmetic before picking the technique**: each technique shrinks a
specific term.

---

## Checkpoint answers

1. [expected] **7B with AdamW.** fp32: 7e9 × 16 B = **112 GB**; with autocast ≈ 126 GB
   (18 B/param); ZeRO-style fp16 + fp32 master: 112 GB. Mixed precision
   shrinks **activations** (fp16 intermediates) and speeds up matmuls; it does
   not shrink the fp32 optimizer state. That is why large-model training shards the state
   (ZeRO/FSDP) or uses PEFT.

2. [verified] **Activations** ≈ batch × seq × layers × (c·hidden + heads·seq) values: linear in
   batch, hidden and layers; linear in seq for most tensors and **quadratic** for
   attention probabilities unless a fused (flash) attention kernel is used. Measured in 40.2:
   18.0 MB per layer, matching `16h + a·s` values per token, plus 9.9 MB for logits and
   embeddings; activations were 5.8× the weights+grads+optimizer state for a 0.93M-parameter model.

3. [verified] **Accumulation** is exactly equivalent when the loss is a mean over
   independent examples and micro-batches are weighted by their share (40.3: difference
   5.6e-09, i.e. rounding). It is not equivalent with BatchNorm (per-micro-batch
   statistics), with per-token losses over variable lengths if weighted per example,
   with randomness that differs (dropout masks), or when the learning-rate schedule
   counts micro-steps instead of optimizer steps. Cost: k sequential passes, so
   less parallelism; the same FLOPs, but the GPU is less efficient with small batches.

4. [verified] **Checkpointing** keeps only block inputs in forward and recomputes the
   block in backward. Saves most activation memory (40.4: 82.0 → 13.9 MB, −83%)
   at the cost of one extra forward pass (40.4: +23% to +40% step time on CPU;
   ~+30% is the usual figure). Place them at transformer-block boundaries (cheap to store
   the residual stream, expensive inside); checkpoint every block for the biggest saving,
   or every k-th for a middle ground; avoid recomputing very expensive ops if memory allows
   (selective checkpointing: keep attention outputs, recompute the MLP).

5. [open-ended] **Growing memory.** Log `torch.cuda.memory_allocated()` per step to confirm
   a leak (grows) vs a peak (flat + spike). Look for tensors kept across steps:
   `total_loss += loss` instead of `.item()`, lists of outputs/predictions without
   `.detach().cpu()`, metrics computed on tensors with graphs, a growing cache
   (e.g. a dict keyed by step), hooks storing activations. Confirm with
   `torch.cuda.memory._record_memory_history()` + the memory visualiser, or
   `gc.get_objects()` filtered for tensors. 40.6 showed the effect: 0 B vs 102.6 MB of
   activations still alive after 5 steps.

6. [expected] **`torch.compile`** captures Python into FX graphs (TorchDynamo), then
   generates fused kernels (Inductor → Triton on GPU, C++ on CPU), optionally CUDA
   graphs (`mode="reduce-overhead"`). Speed-ups come from fusing memory-bound ops
   and removing Python/launch overhead, so they are large for overhead- or
   memory-bound models and small when big matmuls dominate (40.5 on CPU: 0.95–1.09×).
   Risks: compile time (≈25 s here on first compile, minutes for big models),
   recompilation on new shapes, graph breaks that silently fall back to eager,
   harder debugging and stack traces, version-specific bugs. Keep an eager
   fallback and test numerics against eager.
