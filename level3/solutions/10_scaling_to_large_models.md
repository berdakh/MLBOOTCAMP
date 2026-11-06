# Solutions · 10 · Scaling to large models

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## 10-1 · `zero_memory_per_gpu`

✅ Reference code: [`code/nb10.py`](code/nb10.py) (`ex1`), graded in CI.

```python
def zero_memory_per_gpu(n_params, world_size, stage):
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2 or 3")
    params, grads, optim = 2.0 * n_params, 2.0 * n_params, 12.0 * n_params
    if stage >= 1:
        optim /= world_size
    if stage >= 2:
        grads /= world_size
    if stage >= 3:
        params /= world_size
    return params + grads + optim
```

**Numbers (7B, 8 GPUs):** DDP 112 GB, ZeRO-1 38.5 GB, ZeRO-2 26.2 GB,
ZeRO-3 14.0 GB per GPU (printed in 10.1). ZeRO-1/2 plateau at 4Ψ/2Ψ bytes
because the fp16 parameters (and gradients) stay replicated.

**What the formula leaves out.** Activations (often the largest term; reduce
them with activation checkpointing), the all-gather buffers ZeRO-3 needs for
the layer currently being computed (one or two layers' full fp16 weights, more
with prefetching), communication buckets, the CUDA context and fragmentation.

---

## 10-2 · `train_flops_and_days`

✅ Reference code: [`code/nb10.py`](code/nb10.py) (`ex2`).

```python
def train_flops_and_days(n_params, tokens, gpu_tflops, mfu, n_gpus):
    flops = 6.0 * n_params * tokens
    seconds = flops / (n_gpus * gpu_tflops * 1e12 * mfu)
    return flops, seconds / 86_400
```

**Sanity checks.** 6 × 175e9 × 300e9 = 3.15e23 FLOPs, matching the
3.14e23 the GPT-3 paper reports. The 6 was measured in 10.4 as 6.33
FLOPs/token/parameter for a small TinyGPT (attention adds a bit).
**MFU vs HFU:** MFU counts only the model's 6ND FLOPs; *hardware* FLOPs
utilisation also counts recomputation from activation checkpointing (8ND
if you recompute the whole forward), so HFU > MFU for the same run.

---

## 10-3 · `optimal_checkpoint_interval`

✅ Reference code: [`code/nb10.py`](code/nb10.py) (`ex3`).

```python
import math

def optimal_checkpoint_interval(checkpoint_cost_s, mtbf_s):
    return math.sqrt(2.0 * checkpoint_cost_s * mtbf_s)
```

**Where it comes from.** Waste per unit time ≈ C/τ (writing) + τ/(2M) (on
average half an interval of work lost per failure). Setting the derivative
−C/τ² + 1/(2M) to zero gives τ* = √(2CM). The 10.6 simulation (job MTBF 9.4 h,
C = 5 min) found its best interval at 80 min against Young's 75 min, with
a flat optimum (60 and 80 min both +17%).

---

## 10-4 · Planning a 13B run

✍️ / 🔬 One reasonable answer (all assumptions stated):

**(a) Compute-optimal?** 260B / 13B = 20 tokens per parameter: exactly the
Chinchilla rule of thumb. If the model will be served heavily, consider
training longer (more tokens) for a better small model, or a smaller model on
the same budget.

**(b) Memory and parallelism.** Model state 16 × 13e9 = 208 GB: no single
80 GB GPU holds it, so plain DDP is out. Options:
* **FSDP / ZeRO-3 across all 512 GPUs**: 0.41 GB/GPU model state, plenty of room
  for activations, but every layer's parameters are all-gathered across 64
  nodes over InfiniBand in forward and backward.
* **Hybrid sharding (HSDP)**: shard within each 8-GPU node over NVLink
  (208 / 8 = 26 GB per GPU), replicate across nodes (data-parallel all-reduce
  of gradients across nodes). Usually faster than global ZeRO-3 at this size.
* **TP = 8 inside a node + ZeRO-1 data parallelism across nodes**: also fine;
  13B rarely needs pipeline parallelism on 80 GB GPUs.
Plus activation checkpointing and bf16 (A100 supports it: no loss scaling).

**(c) Time.** 6 × 13e9 × 260e9 = 2.03e22 FLOPs; at 512 × 312 TFLOP/s × 0.40,
≈ 3.7 days wall clock, ≈ 1,880 GPU-days. Add the failure overhead from (d) and
evaluation, data-loading stalls and restarts: plan ~4.5–5 days.

**(d) Checkpointing.** 64 nodes with a 30-day node MTBF → job MTBF
30 × 24 / 64 = 11.25 h. Young: √(2 × 240 s × 40,500 s) ≈ 4,400 s ≈ 73 min, so
checkpoint about hourly. Reduce waste further with asynchronous, sharded
checkpoints (`torch.distributed.checkpoint`), automatic restart with hot
spares, fast hang detection (timeouts, heartbeats), and keep the data loader
state in the checkpoint so resumption does not repeat or skip data.

---

## Checkpoint · 10

1. **7B memory.** ✅ 16 bytes/param × 7e9 = 112 GB of model state with
   mixed-precision Adam (plus activations). On 64 GPUs: ZeRO-1 29.3 GB,
   ZeRO-2 15.5 GB, ZeRO-3 1.75 GB per GPU (10.1). Stage 3 pays ~1.5× the
   communication of DDP (parameters all-gathered in forward and again in
   backward, gradients reduce-scattered), more latency-sensitive collectives
   per layer, and complexity (wrapping policy, prefetching, checkpoint format).
2. **Data vs tensor vs pipeline.** Data: split the batch; communicate
   gradients once per step (overlappable); works across nodes; needs the model
   state to fit (or ZeRO). Tensor: split each layer's matrices; all-reduce
   activations twice per layer per pass, on the critical path; inside an NVLink
   node. Pipeline: split layers into stages; send activations point-to-point
   between stages; cheap communication, works across nodes; costs the bubble
   and micro-batching complexity. Large runs combine all three (3D).
3. **Bubble.** With p stages and m micro-batches, the schedule takes
   (m + p − 1) slots per pass while each stage is busy for m of them, so the
   idle fraction is (p − 1)/(m + p − 1) (10.3 simulated 0.429 for p = m = 4).
   Make it small with m ≫ p (more micro-batches → larger global batch or
   smaller micro-batches with lower GPU efficiency), 1F1B scheduling to bound
   activation memory, and interleaved stages (more communication).
4. **7B on 1T tokens on 256 A100s.** 6 × 7e9 × 1e12 = 4.2e22 FLOPs; at
   312 TFLOP/s × 40% MFU × 256 GPUs ≈ 15 days (10.4). The MFU is the most
   uncertain input (it can be 25% or 55% depending on the parallelism,
   sequence length and kernels), followed by downtime from failures.
5. **Chinchilla.** Earlier practice (following Kaplan et al.) grew models much
   faster than data; Hoffmann et al. found that with properly tuned schedules
   parameters and tokens should scale equally, ~20 tokens/param, so many
   large models were under-trained (Chinchilla 70B beat Gopher 280B at similar
   compute). Production models are trained past this point because inference
   cost depends on model size: a smaller model trained on more tokens is cheaper
   to serve for the same quality, and serving dominates lifetime compute.
6. **fp16 vs bf16.** fp16 has a 5-bit exponent: max 65,504 and normal values
   down to 6e-5, so small gradients underflow to zero (10.7: 10.2% of the entries
   without scaling) and large activations overflow. Loss scaling shifts
   gradients into range. bf16 has fp32's 8-bit exponent, so no underflow and no
   scaling, at the cost of precision (eps 7.8e-3). On `inf`/`nan` the dynamic
   scaler skips that optimizer step and halves the scale; after N clean steps
   (2,000 by default) it doubles it again.
