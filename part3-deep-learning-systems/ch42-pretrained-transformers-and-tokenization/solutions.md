# Solutions · Chapter 42 · Pretrained transformers & tokenisation

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `mean_pool`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def mean_pool(last_hidden, attention_mask):
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)        # (B, T, 1)
    return (last_hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
```

**Why.** Padded positions still have hidden states (the model computes them;
the mask only stops *other* tokens from attending to them), so `mean(dim=1)`
averages garbage that depends on how much padding the batch needed. In 4.3 the
naive mean of the same sentence had cosine 0.85 with its unpadded self; the masked
mean matched to 6e-07. `clamp` avoids 0/0 for an empty row. Under fp16, do the sum in
fp32 (`.float()`) for long sequences.

---

## Exercise 2 · `bucket_by_length`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def bucket_by_length(lengths, batch_size):
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
```

**Why it is (near-)optimal.** With a fixed number of full batches, grouping
neighbours in sorted order minimises Σ (max − length) within each batch: any batch
mixing a short and a long text can swap the short one with a longer text from a
batch of shorter texts without increasing either batch's maximum. Measured in 4.5:
padding fell from 50% (dynamic padding in data order) to 17%, and time from 8.4–8.8 s to 5.0 s.

**For training**, do not use the globally sorted order as is: shuffle, sort within
chunks of ~50–100 batches, cut batches, shuffle the batches (Hugging Face:
`group_by_length=True`; also "bucket iterators"). For **token-budget batching**, let
the batch size vary so that `batch × max_len` is roughly constant (more short texts per
batch, fewer long ones), which keeps memory use flat; the grader's fixed-count rule
excludes that variant on purpose.

---

## Exercise 3 · Reviewing a semantic-search pipeline

[open-ended] Problems and fixes:

1. **Naive mean over padding (correctness).** Offline embeddings include up to 500
   pad positions per text; online queries (batch of 1) have none, so documents and
   queries live in different spaces and document embeddings depend on batch
   composition. Fix: mask-aware mean pooling (Exercise 1), and a test that a text's
   embedding is identical alone and inside a padded batch.
2. **Pooling choice / model choice.** If `some-org/encoder` is a plain
   pretrained encoder (not trained for similarity), neither pooling gives good
   retrieval: use a sentence-embedding model trained with the pooling it expects
   (check its card), and evaluate recall@k on labelled queries (Chapter 45).
3. **Pad to 512 (cost).** Most documents are far shorter; padding to 512 wastes
   most of the compute (4.5: 59% wasted at max 128 on short news texts, 2.1–2.7× slower than
   sorted dynamic padding). Use `padding=True` plus length bucketing; restore the
   original order afterwards.
4. **Truncation.** Check `truncation=True` is set and how many documents exceed the
   model's limit; long documents need chunking (embed passages, then aggregate or
   index chunks), not silent truncation.
5. **Query/document asymmetry.** Some models expect prefixes (`"query: "`,
   `"passage: "`) or different max lengths; follow the card.
6. **No revision pin (reproducibility).** The repo owner can change weights or
   tokenizer at any time; the online service and the offline index could end up
   using different versions, silently breaking every similarity. Pin `revision=<commit>`
   for both the model and the tokenizer, record it with the index, and re-embed the
   corpus whenever it changes.
7. **Safety and licence.** Use `use_safetensors=True` (no pickle code execution),
   never `trust_remote_code=True` without review, and read the card's licence before
   deployment. Also: run in `eval()` + `torch.inference_mode()`, fp16 on GPU for throughput,
   and normalise embeddings if the index uses inner product.

---

## Checkpoint answers

1. [open-ended] Subwords bound the vocabulary (memory for the embedding and softmax) while
   covering any word, and keep sequences short. WordPiece: fixed vocabulary chosen by
   likelihood, `##` continuation pieces, typically lower-cased with an `[UNK]` token;
   byte-level BPE: merges over bytes, lossless and never unknown, `Ġ` marks spaces.
   Failure modes seen in 4.1: numbers split arbitrarily (`202`, `##4` vs `Ġ2024`), which hurts
   arithmetic; acronyms split oddly (`bc`, `##is`); non-English text and code use many
   more tokens per character (higher cost, shorter effective context).

2. [expected] The mask marks padding so attention gives it zero weight (−∞ before softmax),
   and pooling/loss ignore it. With **right** padding a decoder's last position for
   short rows is a pad token: generation would continue from padding, and positions of the
   new tokens are wrong. Pad on the **left** (and pass the mask / position ids) for batched generation.

3. [verified] CLS when the model was trained to put a summary there (BERT's NSP, or after
   fine-tuning a classifier on it); mean pooling for models trained with it
   (sentence-transformers) and as a more robust default for off-the-shelf encoders: in 4.3,
   mean pooling separated related from unrelated pairs ~3× more clearly (gap 0.20 vs 0.06).
   Mask-aware because padding states otherwise leak in and make the embedding depend on the
   batch (cosine 0.85 between the same sentence padded and unpadded).

4. [expected] Throughput depends on total token positions (including padding) × per-token
   cost, and on hardware utilisation. Maximise it with: length bucketing + dynamic
   padding (4.5: 2.1–2.7× on CPU); the largest batch that fits; fp16/bf16 (Tensor Cores) or int8
   (Chapter 49); truncating to the length you actually need; a smaller/distilled model;
   fused attention kernels; overlapping tokenisation (CPU workers) with inference; many GPUs in
   parallel (embarrassingly parallel). Do the arithmetic: 100M docs × ~200 tokens =
   2e10 tokens; at ~5e4 tokens/s per T4 (an order-of-magnitude guess, not measured here)
   that is ~4.6 GPU-days: worth optimising.

5. [open-ended] Pin model and tokenizer to a commit hash; load `safetensors` only; no
   `trust_remote_code` unless reviewed and pinned; mirror the artifacts into your own
   model registry (with the licence and card) instead of depending on the Hub at
   serving time; run an evaluation gate before promotion; record versions with every
   prediction log.

6. [open-ended] A tokenizer change (new normalisation, special tokens, `model_max_length`, fast vs
   slow tokenizer differences, padding/truncation side defaults), a model revision pulled from
   `main`, a changed default (e.g. attention implementation, dtype), or a pooling/head
   weights mismatch reported only as a warning ("newly initialised weights"). You catch it
   with pinned revisions, golden tests (fixed inputs → expected token ids and logits within a
   tolerance), an offline evaluation gate in CI before deploying, and dependency pinning
   (lock files).
