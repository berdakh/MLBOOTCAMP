# Solutions · Chapter 46 · Contrastive training & re-ranking

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `info_nce_loss`

[verified] Reference code: [`code/nb08.py`](solutions.py) (`ex1`), graded in CI.

```python
def info_nce_loss(q, d, temperature=0.05):
    logits = q @ d.T / temperature                  # (B, N): row i's positive is column i
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))
```

**Notes.** Hard negatives are just extra columns: with `d` of shape `(2B, dim)` (B
positives, then B hard negatives) every query is contrasted with all 2B documents. The
targets must be `arange(B)`, so the positives must come first and in query order. A
symmetric variant adds the document→query direction (`cross_entropy(logits.T[:B], …)`),
as CLIP does. With unit vectors and τ = 0.05 the logits span [−20, 20]; with τ = 1 the
softmax over cosines is nearly flat and gradients are tiny.

---

## Exercise 2 · `mine_hard_negatives`

[verified] Reference code: [`code/nb08.py`](solutions.py) (`ex2`).

```python
def mine_hard_negatives(ranked_ids, positives, n, skip_top=0):
    pos = set(positives)
    return [d for d in list(ranked_ids)[skip_top:] if d not in pos][:n]
```

`skip_top` skips *ranks* (positives included), not the first negatives: it is a
false-negative guard, and the documents at the very top are the likeliest unlabelled
positives. In the notebook, for 15 % of training claims BM25's top unlabelled document
was a judged positive of another claim (topically on target), which is why we skipped
the top 3.

---

## Exercise 3 · Fixing a RAG assistant's retrieval

[open-ended] A strong answer:

* **First measure where it fails.** Build a judged set from real questions (and the
  documents that answer them); compute Recall@k of the current first stage at the depth
  passed to the LLM (k = 5) and at a larger depth (50–100). If Recall@100 is high but
  Recall@5 low, it is an *ordering* problem (re-ranker); if Recall@100 is low, it is a
  *first-stage* problem (fine-tuning, hybrid, chunking).
* **(a) Fine-tune the bi-encoder on logged pairs.** Effect: Recall@k and nDCG can rise
  substantially in-domain ([expected] a few points or more with enough pairs; our tiny CPU run
  showed no significant change: +0.006 nDCG@10, CI −0.004 to +0.016). Costs: training, and
  **re-embedding 20M documents** for every model version (plan versioned indexes). Risks:
  click logs are position-biased (users click what was shown on top), unclicked ≠
  irrelevant (false negatives), feedback loops. Mitigate with de-biasing, hard-negative
  denoising, a held-out judged set.
* **(b) Add a cross-encoder re-ranker.** Effect: better ordering of the top k when the
  re-ranker matches the domain; it cannot improve recall beyond the first stage's depth.
  Cost: depth × a forward pass per query (in our CPU runs 24–41 ms per pair, 1.2–1.9 s for 50;
  [expected] tens of ms on a GPU), a GPU fleet at high QPS. Risk: **domain mismatch**; the
  MS MARCO re-ranker *lowered* SciFact nDCG@10 in our run (0.840 → 0.757 at depth 50 on 30
  queries). Fine-tune it on in-domain pairs before trusting it.
* **(c) Hybrid first stage.** Cheap (BM25 index + RRF), robust for identifiers, names,
  error codes; in notebook 07 RRF raised Recall@100 from 0.886 (BM25) / 0.919 (dense) to
  0.954. Usually the first thing to try.
* **Evaluation.** Offline: Recall@k and nDCG@10 on the judged set, per slice, with
  bootstrap CIs over queries, plus end-to-end answer correctness and citation accuracy.
  Online: A/B test on answer acceptance, follow-up/reformulation rate, latency p95/p99 and
  cost per query.

---

## Checkpoint answers

1. **InfoNCE.** `L = −mean_i log softmax(q_i · d_j / τ)_{j=i}` over a batch. The temperature
   sharpens the softmax: small τ concentrates the gradient on the hardest negatives (too
   small: unstable, over-penalises false negatives); large τ treats all negatives alike.
   Larger batches give more (and harder) negatives per query, a better approximation of
   the full softmax over the corpus, hence large-batch training, memory banks and
   GradCache.
2. **Hard negatives** are high-ranked non-positives from a retriever (BM25, the current
   model). **False negatives** are unlabelled relevant documents among them; training
   pushes them away. Mitigations: skip the top ranks or sample from a window; filter with a
   cross-encoder (drop candidates it scores as relevant); use fewer hard negatives per
   query; de-duplicate near-identical documents.
3. A cross-encoder needs one forward pass per (query, document) pair and has nothing to
   precompute, so scoring a corpus of N documents costs N passes per query: infeasible
   beyond a few hundred. Choose the depth from a quality-vs-latency table like 8.4's: the
   smallest depth after which nDCG stops rising (bounded by the first stage's
   Recall@depth), within the latency budget.
4. **Before shipping a better bi-encoder:** re-embed and re-index the whole corpus with the
   new model (old and new vectors are not comparable), keep queries and index on the same
   version, check latency and index size, verify per-slice quality and that no slice
   regressed, plan the backfill for new documents and a rollback path, then an online A/B
   test. The offline +3 points must also hold on fresh queries (not the tuning set).
5. **Training data from logs:** (query, clicked document) as positives, shown-but-skipped
   or random documents as negatives, dwell time or conversions as stronger signals.
   Biases: position bias (top results get clicks), presentation bias (only what the old
   system retrieved can be clicked: a feedback loop), popularity bias, and false negatives
   (relevant but unseen). Counter with randomised exploration traffic, inverse propensity
   weighting, and judged evaluation sets.
6. **Design: 100M docs, 1,000 QPS, 150 ms p99.** Query encoding (small bi-encoder on GPU,
   batched): ~10 ms; ANN search (sharded HNSW or IVF-PQ, top 200–500, with a BM25 leg in
   parallel and RRF): ~20–30 ms; feature fetch and network: ~20 ms; cross-encoder re-rank of
   the top 50–100 on GPUs: ~40–60 ms (at 1,000 QPS × 100 pairs = 100k pairs/s: the largest
   fleet; a distilled 6-layer model, fp16, short documents); remaining ~20 ms headroom for
   p99 tails and fallbacks (skip re-ranking under load). Justify each with measured
   per-stage latency and the quality gain of each stage on a judged set.
