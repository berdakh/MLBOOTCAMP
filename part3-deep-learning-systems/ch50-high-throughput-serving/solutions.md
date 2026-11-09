# Solutions · Chapter 50 · High-throughput serving

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `DynamicBatcher`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`, a class), graded in CI. The
production version with a worker thread and error handling is
`scaleml.serving.DynamicBatcher`.

```python
class DynamicBatcher:
    def __init__(self, process_batch, max_batch_size=16, max_wait_ms=5.0):
        self.process_batch = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = None

    def _q(self):
        if self.queue is None:
            self.queue = asyncio.Queue()
        return self.queue

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self._q().put((item, fut))
        return await fut

    async def run(self):
        q = self._q()
        while True:
            batch = [await q.get()]
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(q.get(), remaining))
                except asyncio.TimeoutError:
                    break
            results = self.process_batch([item for item, _ in batch])
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)
```

**Design points that interviewers commonly ask about.**
* The deadline starts at the *first* item, so no request waits more than
  `max_wait_ms` for company (plus the time of the batch ahead of it).
* Each caller gets *its* result because the future travels with the item.
* Calling a slow synchronous model directly inside `run()` blocks the event
  loop: no new requests are parsed while the model runs. Run it in a thread
  (`asyncio.to_thread`) or a separate process/GPU worker.
* If `process_batch` raises, set the exception on every future of that batch
  (otherwise those callers hang forever) and keep the loop alive.
* Add a bounded queue (`asyncio.Queue(maxsize=...)`) and reject with HTTP 503
  when it is full: rejecting excess load is preferable to unbounded latency.

---

## Exercise 2 · `cost_per_1k`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def cost_per_1k(requests_per_second, price_per_hour, utilization):
    return 1000 * price_per_hour / (requests_per_second * 3600 * utilization)
```

**In section 50.8 of the lab** (assumed $0.20/h, 40% utilisation) the measured CPU throughputs of
one run gave $0.00114 per 1,000 requests without batching (122 req/s) and
$0.00045 with dynamic batching (308 req/s); on a quieter machine the no-batching
plateau was higher (246 req/s), so the saving was smaller. [expected] Your numbers depend
on the machine. Halving the cost needs any of: 2× throughput,
half the price (spot/reserved instances), or 2× utilisation (autoscaling).

---

## Exercise 3 · Why the 7B chat service is expensive

[open-ended] One good answer:

**(a) Diagnosis.** With static batching the batch runs until its longest
response finishes; section 50.6 simulated only 26% of decode slots doing useful work
for realistic length distributions (3.7× more steps than continuous
batching). Memory is also reserved per sequence for the maximum length: section 50.7
found only 24% of reserved KV-cache memory holding real tokens. So the GPU is
mostly decoding padding, and memory that could hold more sequences sits
reserved but empty. Batch 8 is far below what a memory-bound decode step could
absorb at almost no extra time per step.

**(b) Changes, in order.**
1. A serving engine with **continuous batching + paged KV cache** (vLLM, TGI,
   TensorRT-LLM): watch tokens/s per GPU, running batch size, and p95
   time-to-first-token and inter-token latency.
2. Raise the max number of concurrent sequences until p95 inter-token latency
   approaches the SLO; enable prefix caching for the shared system prompt.
3. Shrink the KV cache: fp8/int8 KV cache, a model with grouped-query attention.
4. Shrink the model: int8/int4 weight-only quantisation (AWQ/GPTQ) or a
   distilled smaller model, checked on the quality evaluation set; speculative
   decoding with a small draft model for latency.
Watch cost per 1,000 output tokens and the quality metric at every step.

**(c) 2× longer context.** KV memory per sequence doubles (it is linear in
tokens: 512 KB/token for a 7B model without GQA, ~1 GB per 2,048 tokens), so
roughly half as many concurrent sequences fit, lowering throughput and raising
cost per token; prefill also gets slower (attention is quadratic in the prompt).
Mitigations: paged attention (only actual tokens use memory), KV quantisation,
GQA models, offloading or evicting idle conversations' caches, chunked prefill,
and pricing/limits that reflect the longer context.

---

## Checkpoint answers

1. **Why batching helps.** [verified] At batch 1 every weight is read from memory for one
   multiply-add (arithmetic intensity ~1 FLOP per 2–4 bytes), far below what the
   hardware needs to be compute-bound (a T4 needs ~200 FLOPs per byte at peak
   fp16). A batch of B reuses each weight B times for the same traffic, so time
   per call grows slowly: section 50.1 measured ~3 ms at batch 1 and ~15 ms at batch 32
   on CPU (~6× the throughput). It stops helping once compute-bound, when the
   batch no longer fits in memory, or when the wait to fill it breaks the latency SLO.
2. **Dynamic batcher.** Queue + background loop; parameters `max_batch_size`
   (largest batch the hardware finishes within the latency budget) and
   `max_wait_ms` (a small fraction of the SLO, a few ms). At low load batches
   stay small and the wait is pure added latency (section 50.4: p50 grew from 18 to 55 ms
   as the wait went 0 → 25 ms at 4 clients); at high load batches fill without
   waiting and throughput rises (section 50.4: ~310 req/s with batching vs a 110–245 req/s
   plateau without, depending on machine load).
3. **Load testing.** Report throughput and latency percentiles (p50, p95,
   p99) per offered load, error rate and batch-size distribution; means hide
   the tail that users and SLOs care about. Find capacity by increasing load
   (open-loop at fixed arrival rates, or closed-loop concurrency) until p99
   crosses the SLO or errors appear; take the highest throughput below that,
   with headroom. Run clients on separate machines, warm up first, and measure
   for long enough.
4. **Prefill vs decode.** Prefill processes the whole prompt in one pass:
   large matrix products, compute-bound, determines time-to-first-token.
   Decode generates one token per step: each step reads all weights and the KV
   cache for one token per sequence, memory-bound. The KV cache makes a decode
   step O(1) in the projections/MLP instead of re-running the whole prefix
   (section 50.5 measured 1.4–2.1× more tokens/s with the cache for 32 new tokens; more
   for longer outputs), at the cost of memory linear in sequence length.
5. **Continuous batching and paged attention.** Continuous batching schedules
   at every decode step: finished sequences leave, waiting requests join, so
   slots stay busy (section 50.6: 97% vs 26%). Paged attention stores KV caches in
   fixed-size blocks with a block table, so memory is allocated as tokens are
   generated rather than reserved for the maximum length (section 50.7: 98% vs 24%
   used), allowing more concurrent sequences and prefix sharing.
6. **Cost per 1,000 requests.** `1000 × price per hour / (requests/s × 3600 ×
   utilisation)`. Halve it by doubling throughput (dynamic/continuous batching,
   quantisation, distillation, compiled kernels), cutting the hourly price
   (spot or reserved capacity, cheaper accelerators, CPU for small models), or
   doubling utilisation (autoscaling, multiplexing models on one GPU, filling
   troughs with batch jobs).
