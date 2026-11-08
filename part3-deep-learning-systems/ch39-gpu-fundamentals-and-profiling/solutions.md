# Solutions · Chapter 39 · GPU fundamentals & profiling

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## Exercise 1 · `time_fn`

✅ Reference code: [`code/nb01.py`](solutions.py) (`ex1`), graded in CI.

```python
def time_fn(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync(device)                     # earlier queued GPU work is not billed to fn
        start = time.perf_counter()
        fn()
        sync(device)                     # fn's own kernels have finished
        times.append(time.perf_counter() - start)
    return float(statistics.median(times))
```

**Why each piece.** Warm-up absorbs one-off costs (CUDA context, cuBLAS
heuristics, cuDNN autotuning with `benchmark=True`, allocator growth, CPU
caches). Synchronising *before* the clock starts matters as much as after:
otherwise the previous call's still-running kernels are charged to this one.
The median ignores the occasional slow call caused by something else on the
machine; report the spread (min, max or IQR) too when it matters.

**Common mistakes.** `time.time()` (lower resolution, can jump with NTP) instead
of `time.perf_counter()`; timing all iterations in one block without
synchronising (fine for throughput, but sync once at the end, never zero times);
including the warm-up; returning the mean.

**Alternative on GPU.** CUDA events (`torch.cuda.Event(enable_timing=True)`,
`record()` before/after, `elapsed_time()`) measure on the GPU's own clock and
avoid host-side jitter; `torch.utils.benchmark.Timer` does warm-up, sync and
statistics for you.

---

## Exercise 2 · `tflops`

✅ Reference code: [`code/nb01.py`](solutions.py) (`ex2`).

```python
def tflops(n, seconds):
    return float(2 * n**3 / seconds / 1e12)
```

An `n × n` result has `n²` entries; each is a length-`n` dot product = `n`
multiplies + `n` adds (one fused multiply-add counts as 2 FLOPs). For a
general `(m × k) @ (k × n)`: `2mkn`. For a Linear layer on a batch of `B`
tokens: `2·B·d_in·d_out` forward, about twice that backward, so ~`6 × params ×
tokens` for a full training step: the rule used for LLM compute budgets.

---

## Exercise 3 · `arithmetic_intensity`

✅ Reference code: [`code/nb01.py`](solutions.py) (`ex3`).

```python
def arithmetic_intensity(flops, bytes):
    if bytes <= 0:
        raise ValueError("bytes must be positive")
    return float(flops / bytes)
```

Worked values: fp32 `x + y` → 1 / 12; fp32 n×n matmul → 2n³ / 12n² = n/6
(1024 → 171 FLOP/byte); the same matmul in fp16 → n/3, because each element
is half the bytes. The bytes figure for a matmul is the *ideal* (each matrix
read or written once); real kernels move more, tiled through shared memory and
caches, so measured intensity at DRAM is lower.

---

## Exercise 4 · Reading a profiler table

✍️ A good answer, in the order an interviewer wants it:

**Bottleneck: the input pipeline (data loading on the CPU); the GPU is idle
most of the time.** Evidence:

* `enumerate(DataLoader)#_SingleProcessDataLoaderIter...` has 71% of self CPU
  time (6.9 s of 9.7 s). "SingleProcess" means `num_workers=0`: the main process
  loads every batch between steps.
* Self CUDA total is 1.18 s against 9.7 s of CPU time: the GPU did useful work
  ~12% of the wall time.
* The GPU kernels themselves look healthy (cuDNN convolutions forward and
  backward dominate CUDA time, as they should for a CNN), so speeding up the
  model would barely change the step time.
* `cudaStreamSynchronize` (~1 s) plus `aten::item`: the loop syncs every step,
  probably `loss.item()` for logging. Minor here, but it prevents the CPU from
  running ahead.

**Changes, in order.** (1) `num_workers` = number of available cores (2 on
Colab, more on a server), `pin_memory=True`, `persistent_workers=True`, and
`.to(device, non_blocking=True)`; re-profile. (2) If workers still cannot keep
up: make loading cheaper (pre-decode/resize images once and store them in a
fast format, cache the dataset in RAM, move augmentation to the GPU, e.g.
with torchvision v2 transforms on tensors or NVIDIA DALI). (3) Log `loss.item()` every N steps instead of every step.
Then measure again: the next bottleneck becomes visible only after this one is gone.

---

## Checkpoint answers

1. ✅ **Timing without sync.** CUDA calls are asynchronous: `model(x)` enqueues
   kernels and returns, so the measurement is Python + launch time. Fix: warm up,
   then `torch.cuda.synchronize()` before starting and before stopping the clock
   (or CUDA events), repeat, report the median. In 1.2 the CPU numbers were
   equal with and without sync because CPU ops are synchronous; that is exactly
   why the bug survives code that was first tested on a laptop.

2. 🔬 **Roofline.** Arithmetic intensity = FLOPs / bytes moved. Attainable FLOP/s =
   min(peak compute, intensity × bandwidth); the ridge point (peak / bandwidth,
   ~25 FLOP/byte fp32 and ~200 fp16 on a T4) separates the two regimes.
   LayerNorm does a handful of FLOPs per element read and written → intensity
   ~1, **memory-bound**: speed it up by fusing it with neighbours, not by more
   FLOP/s. A 4096² matmul has intensity ~680 (fp32) → **compute-bound**. Our CPU
   run showed the same split: an add at ~0.4–0.8 GFLOP/s vs a matmul at
   ~130–270 GFLOP/s on the same chip.

3. 🔬 **fp16 on a T4.** Tensor Cores (fp16 only on Turing) give ~8× the fp32
   peak, and fp16 halves memory traffic for memory-bound ops. fp16's
   exponent range is small (min normal ~6e-5), so small gradients underflow
   to zero: the GradScaler multiplies the loss by a large factor S (gradients
   scale by S), unscales before the step, and skips steps with inf/NaN while
   lowering S. bf16 has fp32's 8-bit exponent, so no scaling is needed (at the cost
   of fewer mantissa bits); T4s do not support it. Autocast keeps reductions,
   softmax and losses in fp32.

4. ✍️ **30% utilisation.** (a) Input pipeline: profile or time the loader alone;
   check `num_workers`, CPU usage per core (`top`/`htop`), disk/network reads.
   (b) Host-device syncs in the loop (`.item()`, `.cpu()`, printing, `nonzero`):
   profiler trace shows gaps after `cudaStreamSynchronize`. (c) Too-small work per
   kernel (small batch, small model): many short kernels with gaps; raise the
   batch size, use CUDA graphs / `torch.compile`. (d) Host-to-device copies not
   overlapped: pageable memory, no `non_blocking`. Also note that `nvidia-smi`
   "utilisation" is the fraction of time *any* kernel runs, not how busy the SMs
   are: 100% can still be inefficient.

5. 🔬 **Loader knobs.** `num_workers`: processes that run `__getitem__` + collate in
   parallel. `prefetch_factor`: batches queued per worker ahead of use.
   `pin_memory`: batches returned in page-locked memory so copies can be
   asynchronous DMA. `non_blocking=True`: the copy is enqueued and the CPU continues.
   Workers stop helping when the loader already outpaces the step (in 1.7, 2
   workers brought one epoch from 1.86 s to 0.80–0.83 s against a 0.60 s floor), when
   you run out of cores (2 on Colab), or when the shared resource is I/O or memory
   bandwidth.

6. ✍️ **Overhead-bound / launch-bound** (also "CPU-bound"). Fixes: fuse ops
   (`torch.compile`, fused optimizers `fused=True`, fused kernels like
   `scaled_dot_product_attention`), capture the step in **CUDA graphs** (replay
   with one launch; `torch.compile(mode="reduce-overhead")`), increase work per
   kernel (larger batch, fewer, larger layers), and remove Python-side work and
   syncs from the inner loop.
