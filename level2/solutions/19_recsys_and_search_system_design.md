# Solutions · 19 · Recommendation & search system design

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 19-1 · `embedding_memory_gb`

✅ Reference code: [`code/nb19.py`](code/nb19.py) (`ex1`), graded in CI.

```python
def embedding_memory_gb(n, dim, bytes_per):
    return float(n * dim * bytes_per / 1e9)
```

300M users × 128 dims × 4 bytes = 153.6 GB; in fp16, 76.8 GB. Say whether
you mean GB (10⁹) or GiB (2³⁰ = 7.4 % larger); both are fine if stated.
Remember the extras: optimizer state during training (Adam keeps 2 more
copies: ×3), index structures (HNSW links), and replicas.

---

## 19-2 · `qps`

✅ Reference code: [`code/nb19.py`](code/nb19.py) (`ex2`).

```python
def qps(dau, req_per_user, peak):
    return float(dau * req_per_user / 86_400 * peak)
```

Mental arithmetic: 86,400 ≈ 10⁵ s/day. 50M × 20 = 10⁹ requests/day →
~10⁴ average QPS (exactly 11,574) → ~3 × 10⁴ at peak ×2.5 (28,935). One
significant figure: "about 30,000 QPS at peak".

---

## 19-3 · Written design (prompt 1, 2 or 6)

✍️ Personal. Check your answer against this list:

* **10 steps signposted**, ~40 minutes when presented, the deep parts in
  candidate generation, ranking and serving.
* **Capacity table with labelled assumptions**: peak QPS from `qps`;
  candidates per stage (e.g. 2,000 → 500 → 50); ranker rows/s =
  QPS × candidates; embedding memory from `embedding_memory_gb`; servers from
  an *assumed, to-be-measured* per-core throughput.
* **Latency budget per stage** that leaves slack and names the fallback for
  each stage.
* **Two candidate sources with failure modes** (e.g. two-tower ANN: popularity
  bias from in-batch negatives, stale embeddings; co-visitation: no recall for
  new items, bot-click poisoning).
* **Labels and bias correction**: the exact label (watch-time completion,
  click, conversion), its delay, and position/exposure bias with an
  exploration bucket, IPS or position-as-feature.
* **Online experiment**: unit, primary metric, guardrails, duration,
  interference (marketplace/network effects), interleaving for fast ranking
  comparisons.
* **Two serious risks** with mitigations.

Prompt-specific must-haves: *short video* — watch-time bias toward long
videos (normalise by duration or use completion), real-time features and
fast retraining, creator cold start with exploration. *Ads CTR* —
calibration (bids × pCTR), log loss, negative down-sampling with
recalibration, delayed conversions, auction effects. *Similar items* —
substitutes vs complements, content embeddings for cold start, batch
precomputation of neighbours, evaluation by module add-to-cart.

---

## Checkpoint · 19

1. **Feed candidate generation.** Sources and sizes (e.g. two-tower ANN
   800, co-visitation from recent clicks 500, followed sellers/saved searches
   300, fresh items 200, trending 200), merged and deduped, filtered before
   ranking. Value per source: recall@K of eventually-engaged items, share of
   final slots won, and an ablation (turn the source off in an A/B or
   interleaving test). New items: a dedicated fresh source with an
   exploration budget, content-based embeddings available at listing time,
   smoothed priors for their engagement features.
2. **Estimate.** Peak QPS = 100M × 30 / 86,400 × (say) 2.5 ≈ 87,000. Ranker
   load = 87k × 1,000 ≈ 87M rows/s. Item embeddings = 500M × 256 × 4 B =
   512 GB (fp32), 256 GB fp16, ~36 GB with 64-byte PQ codes + ids. The
   embedding memory changes the architecture first (a sharded ANN service
   with compression and re-scoring), and the ranker load forces a pre-ranker
   and model distillation.
3. **Lexical vs semantic.** Lexical (BM25) fails on vocabulary mismatch,
   synonyms, misspellings, other languages and vague intents; it excels at
   exact tokens (brands, model numbers, sizes). Semantic retrieval fails on
   exact/rare tokens, negation and attribute constraints, and can return
   "topically close but wrong" items. Combine: run both in parallel, union
   candidates (or reciprocal rank fusion), and let the LTR model use both
   scores as features; apply extracted attribute filters to both.
4. **Position bias.** Users examine top positions more, so clicks reflect
   position as well as relevance. Estimate propensities with randomisation
   (shuffle the top k, swap adjacent pairs, or a random bucket) or by
   intervention harvesting across rankers. Correct with IPS-weighted training
   losses (each click weighted by 1/propensity) and IPS-weighted evaluation;
   or model position explicitly (feature/tower set to a constant at serving).
   Notebook 20: raw clicks preferred the old policy (0.760 vs 0.664 NDCG),
   IPS reversed it (+0.29 DCG), agreeing with the truth (+0.31).
5. **Marketplace A/B.** Treatment and control share inventory, so a better
   ranker in treatment can take scarce items from control (SUTVA violated),
   inflating the measured lift. Options: measure seller-side and
   inventory-level outcomes; cluster/geo randomisation or switchback designs
   where markets separate; budget-split or two-sided designs; at least
   check for contamination by comparing control to a pre-period or holdout.
   Interleaving avoids inventory competition only within the same user's page.
6. **Post-launch monitoring.** Per-stage latency and errors; fallback rates
   per stage; candidate-source shares; feature freshness and null rates;
   score distribution and calibration (predicted vs observed CTR by decile);
   coverage/diversity/seller concentration; business metrics vs the A/B
   estimate (online–offline gap). Page on: p99 latency or error-rate SLO
   breach, fallback rate above threshold (e.g. ranker fallback > 1 % for 10
   minutes), and stale features (newest value older than its SLA) or a
   collapse in CTR/score distribution. Everything else goes to dashboards and
   daily review.
