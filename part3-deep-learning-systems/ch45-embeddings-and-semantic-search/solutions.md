# Solutions · Chapter 45 · Embeddings & semantic search

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## Exercise 1 · `bm25_scores`

✅ Reference code: [`code/nb07.py`](solutions.py) (`ex1`), graded in CI.

```python
def bm25_scores(query_tokens, doc_tokens_list, k1=1.2, b=0.75):
    n = len(doc_tokens_list)
    avgdl = sum(len(d) for d in doc_tokens_list) / max(1, n)
    df = Counter(t for d in doc_tokens_list for t in set(d))     # documents containing t
    scores = []
    for doc in doc_tokens_list:
        tf = Counter(doc)
        s = 0.0
        for t in set(query_tokens):                               # distinct query terms
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(s)
    return scores
```

**Pitfalls.** Document frequency counts *documents* (`set(d)`), not occurrences. The
classic Robertson idf `ln((N − n + 0.5)/(n + 0.5))` goes negative for terms in more than
half the documents; Lucene's `ln(1 + …)` stays positive. `b = 0` turns length
normalisation off; `k1 → 0` makes BM25 binary (presence only). This loop is
O(|docs| × |query|) per query; `scaleml.retrieval.BM25Index` precomputes the weights into a
sparse matrix and answered a query in ~0.2 ms over 5,183 abstracts in the notebook.

---

## Exercise 2 · `reciprocal_rank_fusion`

✅ Reference code: [`code/nb07.py`](solutions.py) (`ex2`).

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])   # dicts keep insertion order; sort is stable
```

A document missing from a list simply gets no contribution from it, so truncated lists
(top-100 from each system) penalise documents only one system found, as intended. In
the notebook (CPU, 128-token documents) RRF of BM25 and MiniLM reached nDCG@10 0.681 and
Recall@100 0.954, vs 0.667 / 0.886 (BM25) and 0.622 / 0.919 (dense).

---

## Exercise 3 · Replacing BM25 product search with semantic search

✍️ A strong plan:

* **Offline evaluation.** Build a judged query set from logs (stratified: head, torso,
  tail queries; navigational queries with exact identifiers such as SKUs, model numbers,
  brand names; misspellings; long natural-language queries). Metrics: Recall@k of the
  first stage (k = what the ranker consumes), nDCG@10, and per-slice results. Expect
  dense to win on paraphrase/long-tail and lose on identifiers and rare terms (as BM25 beat
  MiniLM on SciFact's gene/drug-name claims: 0.667 vs 0.622 nDCG@10 in our run).
* **Index and memory.** 50M × 384 × 4 B = 77 GB of float32 vectors (+ HNSW graph,
  ~50M × 2M × 4 B ≈ 13 GB at M = 32): needs sharding or a big-memory host. IVF-PQ with
  48–64 byte codes cuts that to 2.4–3.2 GB at a recall cost you measure against exact
  search (as in 7.4); re-rank the ANN candidates with full vectors if needed.
* **Re-embedding cost.** Every encoder change re-embeds 50M descriptions: at our CPU's
  ~41 docs/s that is ~14 days on one machine; at 🔬 ~1–5k docs/s per GPU, hours on a
  small GPU fleet. Plan versioned indexes (old and new side by side), backfills and
  queries embedded with the *same* model version as the index.
* **Ship hybrid.** Keep BM25 for exact matching and fuse (RRF or a learned combination)
  with dense; on SciFact the hybrid was the best of the three on both metrics, and it is
  robust when one system fails on a slice. Validate with an online A/B test on
  search conversion, zero-result rate and latency.

---

## Checkpoint answers

1. **BM25.** idf weights rare terms up; `k1` controls term-frequency saturation; `b`
   controls length normalisation. BM25 wins on exact, rare terms (identifiers, names,
   numbers), out-of-domain vocabulary the encoder never saw, and when there is no
   training data; it is also cheap and interpretable. On SciFact (CPU run) it beat MiniLM
   on nDCG@10, 0.667 vs 0.622.
2. **Bi- vs cross-encoder.** A bi-encoder encodes documents independently of the query,
   so document vectors can be computed once and indexed; a cross-encoder's output
   depends on the (query, document) pair, so nothing can be precomputed: scoring a
   corpus means one forward pass per document *per query*.
3. **nDCG@10 vs Recall@100.** For a first stage feeding a re-ranker, optimise Recall@k at
   the re-ranking depth: what is not retrieved cannot be re-ranked; ordering inside the
   top k is the re-ranker's job. An unjudged relevant document counts as non-relevant,
   so both metrics are underestimated, and systems that retrieve different (unjudged)
   documents than those used to build the qrels are penalised (pooling bias).
4. **Exact vs HNSW vs IVF-PQ.** 1M: exact (on GPU) or HNSW, memory is fine. 100M: HNSW
   sharded, or IVF with PQ if memory-bound. 10B: IVF-PQ (compressed codes, possibly on
   disk, e.g. DiskANN-style), with re-ranking of candidates. `M` = graph degree (memory,
   recall, build time); `efSearch` = candidate list size at query time (latency vs
   recall: 0.88 → 0.996 recall@10 from 16 to 256 in our 50k test).
5. **RRF** uses ranks, which are comparable across systems, while raw scores are not
   (BM25 unbounded, cosines in [−1, 1], distributions differ per query). It can hurt when
   one system is much worse than the other (it adds noise) or when the score margins carry
   information (a very confident top hit); then a learned fusion helps.
6. **Internal-document search for an LLM assistant.** Stages: chunking (with titles and
   metadata), hybrid first stage (BM25 + dense, with permission filtering), cross-encoder
   re-ranking of the top 50–100, then the top few chunks to the LLM. Evaluation set:
   real employee questions with annotated supporting chunks (plus synthetic questions
   generated from documents, checked by people), measured with Recall@k and nDCG; then
   end-to-end answer quality. Online: answer acceptance/thumbs, citation clicks,
   reformulation and abandonment rates, latency, and the rate of "no answer found".
