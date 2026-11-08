# Solutions · Chapter 26 · Two-tower retrieval & ANN

Attempt each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `in_batch_softmax_loss`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI (needs PyTorch).

```python
def in_batch_softmax_loss(user_emb, item_emb, log_q=None):
    logits = user_emb @ item_emb.T                 # [B, B]: row = user, column = batch item
    if log_q is not None:
        logits = logits - log_q[None, :]           # correct each COLUMN (item) by its sampling prob
    target = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, target)         # the diagonal is the positive
```

**Why per column.** Column j is item j acting as a candidate for every user
in the batch; its chance of being in the batch is q_j, so the correction
belongs to the column. `log_q[:, None]` (per row) is a constant per row and
cancels out of each row's softmax: it silently does nothing (the grader
checks the values, so it fails).

**Common mistakes.** Building the target with `torch.ones`/one-hot and MSE;
converting to NumPy (breaks gradients); applying the temperature twice (the
lab divides the *user* embedding by τ before calling the loss).

**Push further.** Accidental hits: if item j equals the positive of row b
(same item twice in the batch), mask logit (b, j) with −∞;
`rt.in_batch_softmax_loss(..., item_ids=i)` does this.

---

## Exercise 2 · `recall_vs_exact`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def recall_vs_exact(approx_ids, exact_ids):
    vals = []
    for a, e in zip(np.asarray(approx_ids), np.asarray(exact_ids)):
        e_set = {int(x) for x in e if x >= 0}
        if e_set:
            vals.append(len({int(x) for x in a if x >= 0} & e_set) / len(e_set))
    return float(np.mean(vals)) if vals else float("nan")
```

**Notes.** Sets make order irrelevant (recall@k, not a ranking metric).
Ignoring −1 matters when an index returns fewer than k results (IVF with
small `nprobe` and nearly-empty cells). With ties in the exact scores
(duplicate vectors), "exact top-k" is itself ambiguous; benchmarks then
compare by *distance* threshold instead of ids.

---

## Exercise 3 · Why in-batch negatives over-penalise popular items (written)

[open-ended] One version with numbers:

* Item A gets 1% of all clicks, item B 0.01%. With batches of 1,024 clicks,
  A appears in a batch about 10 times on average, so for every user in the
  batch A is a negative about 10 times; B appears in about 1 batch in 10.
  Per training example, A is pushed down roughly **100× more often** than B,
  regardless of whether users like it less.
* The sampled softmax then estimates p(i | u) / q_i (q_i = the item's chance
  of being sampled): it learns "how much more than its base rate does u
  like i", a PMI-like score. Ranking by it buries popular items (lab:
  mean log-popularity of retrieved items 3.6 vs 5.3 corrected; recall on
  clicks 0.24 vs 0.41).
* Subtracting log q_i from item i's logit **during training** makes the
  softmax's optimum s(u, i) − log q_i ∝ log p(i|u) − log q_i, i.e.
  s(u, i) ∝ log p(i | u): the score itself is the unbiased one. At serving
  there is no sampling, so nothing to correct: use s(u, i) directly.
  Subtracting it again at serving would re-introduce the penalty.
* When to keep some penalty: when discovery is *wanted*, diversity or
  long-tail exposure (a "discover" surface, marketplace fairness to small
  sellers, or to counter a feedback loop where popular items get more
  popular). Then use a partial correction, log q_i scaled by α ∈ (0, 1), and
  tune α on a metric that values novelty, not only on next-week clicks
  (which reward popularity).

---

## Checkpoint answers

**1. Architecture.** User tower (user id, history, context features) → u;
item tower (item id, content features) → v_i; score = u·v_i. Because the
towers only meet in the final dot product, item vectors are precomputed and
indexed; per request only u is computed. Hence no user×item crosses
(e.g. "user's past CTR on this item's brand"); those go in the ranker,
which scores only a few hundred candidates and can afford them.

**2. In-batch negatives.** Cheap: the other positives in the batch serve as
negatives, so one B × B matrix multiply gives B − 1 negatives per example
without extra item-tower passes. Bias: negatives are sampled ∝ popularity
(over-penalising popular items); fix with logQ correction (estimate q_i
from counts or a streaming frequency estimator). Alternatives: uniform
random negatives from the catalogue (unbiased in sampling, but mostly easy
negatives; more item-tower passes), mixed negative sampling (in-batch +
uniform), hard negatives (items ranked high but not clicked: faster
learning, risk of false negatives), and impressed-but-not-clicked items
(real negatives but position-biased).

**3. Two-tower below MF.** (a) logQ correction missing or wrong sign
(check popularity of retrieved items); (b) target leakage through the
history (great train loss, poor validation); (c) temperature / normalisation
(loss stuck near log B); (d) evaluation mismatch: user vectors from one
model searched in item vectors of another, wrong id mapping, or
excluding/including previously clicked items differently from the baseline;
(e) under-training: too few epochs, learning rate, embedding dimension;
then features: are history and categorical features actually populated?

**4. IVF / HNSW / PQ.** IVF: k-means into `nlist` cells, search the `nprobe`
nearest cells; parameters `nlist`, `nprobe` (recall vs latency). HNSW: layered
proximity graph, greedy beam search; parameters `M` (memory, build time,
recall), `efConstruction` (build quality), `efSearch` (recall vs latency).
PQ: split vectors into m sub-vectors, quantise each with a small codebook,
store codes (e.g. 16 bytes per vector), compute approximate distances with
lookup tables; parameters m and bits (memory vs accuracy); combine as IVF-PQ and
re-rank the shortlist exactly.

**5. New items in minutes.** The item tower uses content features, so a
new item can be embedded on upload by the *current* item tower and added
incrementally to the index (IVF: assign to the nearest centroid; HNSW:
insert into the graph; rebuild periodically). Consistency: the user tower
and every vector in the index must come from the same model version (and
the same feature preprocessing); swap model and index together, or keep a
versioned index per model.

**6. −3% recall vs exact.** Measure end to end: does it change the *final*
ranked list and online metrics? Often the lost items are deep in the top
100 and the ranker would not have shown them. Check what is gained in exchange
(latency, cost), whether the loss is uniform or concentrated (new items,
small categories, cold users: fairness and freshness issues), and run an
A/B or at least an offline replay of the full cascade with exact vs ANN
retrieval.
