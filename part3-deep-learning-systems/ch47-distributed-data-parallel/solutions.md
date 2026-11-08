# Solutions · Chapter 47 · Distributed data parallel

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `ring_allreduce_bytes`

[verified] Reference code: [`code/nb09.py`](solutions.py) (`ex1`), graded in CI.

```python
def ring_allreduce_bytes(n_params, bytes_per, world_size):
    return 2 * (world_size - 1) / world_size * n_params * bytes_per
```

**Derivation.** Split the S = `n_params × bytes_per` bytes into N chunks. In
the reduce-scatter phase, at each of N−1 steps every rank sends one chunk
(S/N) to its neighbour and adds the chunk it receives; afterwards each rank
owns the fully reduced version of one chunk. The all-gather phase circulates
those reduced chunks for another N−1 steps. Total sent per rank:
2(N−1) × S/N. For N = 1 it is 0; for large N it tends to 2S.

**Numbers to remember.** BERT-base (110M) with fp16 gradients on 8 GPUs:
2 × 7/8 × 220 MB = 385 MB per rank per step. A 7B model on 64 GPUs:
≈ 27.6 GB per rank per step (2 × 63/64 × 14 GB).

---

## Exercise 2 · `average_gradients`

[verified] Reference code: [`code/nb09.py`](solutions.py) (`ex2`).

```python
def average_gradients(grads_per_rank):
    import torch
    world = len(grads_per_rank)
    return [torch.stack([rank[i] for rank in grads_per_rank]).sum(0) / world
            for i in range(len(grads_per_rank[0]))]
```

**Check.** In 9.2, averaging 2 or 4 equal shards' gradients matched the
full-batch gradient to ~3e-08; an uneven 8 + 24 split was off by 6.3e-02,
because a plain mean of per-shard *mean* gradients weights each shard
equally rather than each example. With unequal shards, weight rank r by
`n_r / N` (all-reduce `grad × n_r` and the counts, then divide).

**Common mistakes.** Using `torch.stack(...).mean(0)` is fine; modifying
`grads_per_rank[0][i]` in place with `+=` is not (the grader checks inputs are
untouched; in real DDP the in-place all-reduce is fine because each rank owns
its buffer).

---

## Exercise 3 · `shard_indices`

[verified] Reference code: [`code/nb09.py`](solutions.py) (`ex3`).

```python
def shard_indices(n, world_size, rank, epoch, seed, shuffle):
    import math, torch
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
```

**Details the grader checks.** The repetition handles `n < world_size`
(e.g. 3 examples on 4 ranks pads with index list repeated). Rank r takes a
*strided* slice, not a contiguous block. The seed is `seed + epoch`, which is
exactly why forgetting `set_epoch` repeats the order every epoch.

---

## Exercise 4 · Diagnosing a 5.5× speed-up on 8 GPUs

[open-ended] Open-ended; a strong answer covers:

**(a) Where the missing 2.5× goes, and how to measure each.**
* *Communication not hidden*: profile one step (`torch.profiler` with CUDA
  activities; look for NCCL `allreduce` kernels after the last backward
  kernel). Compare step time with `world_size=1` vs 8 on the same per-GPU batch.
  Fixes: larger per-GPU batch, `no_sync()` accumulation, `bucket_cap_mb`, fp16/bf16 gradients.
* *Input pipeline*: 8 processes now share the node's CPU cores for decoding and
  feature transforms; check GPU utilisation gaps and time spent in `next(loader)`.
  Fix: more `num_workers` per rank only if cores allow, pre-processed data, pinned memory.
* *Stragglers*: one slow GPU (thermal throttling, a noisy neighbour, a slower
  PCIe slot) sets the pace for all. Log per-rank step times; look at the max.
* *Small per-GPU work*: kernels too small to fill the GPU; launch overhead dominates.
* *Embedding tables* (recsys): huge sparse gradients all-reduced densely.
  Use sparse gradients / sharded embeddings (TorchRec) instead.

**(b) Worse metric after the same number of epochs.** With the per-GPU batch
fixed, the global batch is 8× larger, so each epoch has 8× fewer optimizer
steps. Keep the learning-rate schedule in *steps* in mind: scale the learning
rate (linear scaling rule for SGD, roughly √ scaling or retuning for Adam), add
warm-up, and/or train for more epochs. Check `set_epoch` is called and that BatchNorm
statistics are not too noisy (SyncBatchNorm).

**(c) Rank-0-only evaluation with an all-reduce inside `if rank == 0:`.**
Only rank 0 calls the collective; the other ranks never do, so rank 0 waits
forever (a hang until the timeout) or, worse, the call matches a *different*
collective on the other ranks (the next DDP gradient all-reduce), corrupting
both. Either every rank evaluates its shard and every rank calls the
all-reduce, or rank 0 evaluates the whole set alone with *no* collective (and the
others wait at a `barrier()` that every rank calls).

---

## Checkpoint answers

1. **One DDP step.** [verified] Forward on the local shard (no communication; DDP
   may broadcast buffers like BatchNorm running stats from rank 0). `backward()`:
   autograd hooks mark each gradient ready; when a ~25 MB bucket is complete DDP
   launches an asynchronous all-reduce on it while backward continues with
   earlier layers (overlap). At the end of backward DDP waits for all buckets
   and divides by world size. `optimizer.step()` is local and identical on all
   ranks. Weights were broadcast once from rank 0 at construction. 9.3 showed
   the replicas bit-identical and ≈6e-08 from single-process training.
2. **Why it is equivalent.** The mean-loss gradient over the full batch equals
   the mean of per-shard mean gradients when shards are equal in size, and all
   replicas start equal and apply the same update. Breakers: BatchNorm
   (per-shard statistics), unequal shard sizes with plain averaging, different
   random augmentation/dropout is fine statistically but not bit-identical,
   losses normalised per rank by a variable count (tokens), and the effective
   batch size change itself (learning-rate schedule).
3. **Ring all-reduce volume.** 2(N−1)/N × S per GPU (09-1). It tends to 2S,
   so per-GPU traffic barely grows with N. DDP still stops scaling because
   (i) compute per GPU stays fixed while communication does not shrink, so
   large models (GB of gradients) are communication-bound on slow links (9.6:
   a 7B model on PCIe was ~34% efficient by our assumptions); (ii) the latency
   term grows with 2(N−1) steps; (iii) the slowest link/GPU sets the pace;
   (iv) large global batches eventually hurt optimisation (critical batch size).
4. **`DistributedSampler`.** [verified] Shuffles with seed + epoch, pads to a multiple
   of world size by repeating indices, and gives rank r indices r, r+N, …
   Without `set_epoch`, every epoch uses the same order (9.4: identical lists).
   Padding exists so every rank runs the same number of steps, otherwise the
   ranks with more data would wait forever in an all-reduce. It duplicates up
   to N−1 examples, which matters for evaluation metrics.
5. **Crash vs hang.** A crash: the process exits non-zero; peers get errors
   (`Connection closed by peer` in 9.5) or are killed by the torchrun agent,
   which reports the root-cause rank and exit code; the job exits non-zero.
   Recover by relaunching (scheduler requeue or `--max-restarts`) and resuming
   from the last checkpoint (9.5 resumed with a parameter difference of 0.0). A
   hang: GPUs at 0% (or 100% spinning in NCCL), no new log lines, no exit. Detect with
   heartbeats/step-time alarms, the NCCL watchdog and a deliberate process-group
   `timeout`; get stack traces with `py-spy dump` on each rank; usual causes are
   a collective skipped on some ranks, mismatched shapes, or a dead link.
6. **Global batch 4,096 on 8 × 64.** 8 × 64 = 512 per step, so accumulate 8
   micro-batches per optimizer step, wrapping the first 7 backward passes in
   `model.no_sync()` so DDP all-reduces only once. Retune the learning rate for
   the larger batch (linear scaling + warm-up for SGD; for Adam, tune), check
   steps-per-epoch-based schedules, and use LAMB/LARS if you go much larger.
   [expected] Whether 4,096 is beyond the critical batch size for this problem must be
   measured.
