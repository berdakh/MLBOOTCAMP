# Solutions · Chapter 41 · A transformer from scratch

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `causal_self_attention`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def causal_self_attention(x, w_qkv, w_out, n_heads):
    B, T, C = x.shape
    D = C // n_heads
    q, k, v = (x @ w_qkv).split(C, dim=-1)
    q, k, v = (t.view(B, T, n_heads, D).transpose(1, 2) for t in (q, k, v))   # (B, H, T, D)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)                             # (B, H, T, T)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    weights = F.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
    y = (weights @ v).transpose(1, 2).reshape(B, T, C)                          # merge heads
    return y @ w_out
```

**Details interviewers check.** The scale uses the *head* dimension `D`, not
`C`. The mask is `triu(..., diagonal=1)`: position `i` may see `j ≤ i`
(including itself). `view(B, T, H, D).transpose(1, 2)`, not `view(B, H, T, D)`
(that would mix tokens across heads). After the transpose back, `reshape`
(or `.contiguous().view`) is required because the tensor is no longer contiguous.
Masking with `-inf` *before* the softmax, not zeroing weights after it (which
would leave rows that do not sum to 1). In production, use
`F.scaled_dot_product_attention(q, k, v, is_causal=True)`: same maths, fused
FlashAttention/memory-efficient kernels, no `T × T` matrix in memory.

---

## Exercise 2 · `top_k_sample`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def top_k_sample(logits, k, temperature, generator):
    logits = logits / temperature
    if k < logits.size(-1):
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]          # k-th largest per row
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)
```

**Notes.** Ties at the k-th value keep more than k tokens; that is the usual
convention (nanoGPT and Hugging Face both do it) and harmless. Temperature is applied before
the top-k cut (the order does not matter for which tokens survive, since dividing by
τ > 0 preserves order). **Top-p (nucleus)** instead keeps the smallest set whose
cumulative probability ≥ p: adaptive, so it keeps few tokens when the model is
confident and many when it is not. Temperature 0 is handled separately as `argmax`
(`tg.sample_next` does this), since dividing by zero is undefined.

---

## Exercise 3 · Attention complexity and the KV cache

[open-ended] A model answer:

(a) One forward pass over `n` tokens: computing Q, K, V and the output projection is
**O(n·d²)**; the scores `QKᵀ` and the weighted sum `AV` are **O(n²·d)**. The attention
matrix is `n × n` per head: **O(h·n²)** memory if materialised (FlashAttention avoids
storing it, but the FLOPs remain O(n²·d)).

(b) Generating `n` tokens one by one **without a cache**, step `t` re-runs the full
model on `t` tokens: attention O(t²·d) and projections/MLP O(t·d²) per step, so
**O(n³·d)** attention and **O(n²·d²)** for the rest in total. **With a cache**, step
`t` processes one token: projections/MLP O(d²) and attention over `t` cached keys
O(t·d), so **O(n·d²)** + **O(n²·d)** in total. The cache removes a whole factor of `n`
from both terms. (Measured in 3.7 on CPU: the cache was 2.1–2.2× faster at 64 tokens and
3.5–5.0× at 256–500 tokens.)

(c) **Cost:** memory, `2 × layers × n × d` values per sequence (K and V), growing
linearly with context and batch: 0.5 MB per token for a 7B model in fp16. **Why
bandwidth-bound:** each decode step does only ~2 FLOPs per parameter per token
(a matrix-vector product) but must read *all* weights and the whole KV cache from
memory: arithmetic intensity ~1 FLOP/byte, far left of the roofline ridge (Chapter
39). Batching many sequences raises intensity for the weights (read once, used for the
whole batch) but not for the KV cache (each sequence has its own), so long contexts
stay bandwidth-bound. Remedies: grouped-/multi-query attention (fewer K/V heads),
KV cache quantisation, paged attention to avoid fragmentation, sliding-window attention.

---

## Checkpoint answers

1. [open-ended] `softmax(QKᵀ/√d + M)V`. Q (what am I looking for), K (what do I contain), V
   (what I pass on) are linear projections of the input; `QKᵀ` scores every pair;
   `/√d` keeps the scores' variance ~1 so the softmax is not saturated (vanishing
   gradients); M (−∞ above the diagonal) enforces causality; the softmax turns
   scores into weights summing to 1; multiple heads attend to different
   relations in parallel subspaces at the same total cost, then a projection mixes them.

2. [open-ended] Post-LN: `x = LN(x + f(x))`: the residual path passes through every
   LayerNorm, gradients near the input can vanish/explode with depth, and training
   needs learning-rate warm-up and care. Pre-LN: `x = x + f(LN(x))`: an unobstructed
   identity path from output to input, so gradients flow, deep stacks train stably at
   higher learning rates; needs a final LayerNorm before the head (as in `TinyGPT.ln_f`).
   Trade-off reported in the literature: post-LN can reach slightly better quality when
   it trains at all.

3. [verified] A leaky mask (the model sees the token it must predict) or targets not
   shifted by one (`y == x`). Test: the perturbation test from 3.3 (change token
   `t`, logits before `t` must not change), check `y[:, :-1] == x[:, 1:]`, compare
   with the unigram baseline (3.347 here) and a known-good reference: a character model
   reaching 0.05 on Shakespeare is not plausible (ours: 1.83 validation after 1,500 steps).

4. [expected] Temperature divides logits: τ < 1 sharpens (more conservative, more
   repetitive), τ > 1 flattens (more diverse, more errors). Top-k keeps a fixed number
   of candidates; top-p keeps the smallest set covering probability p, adapting to
   the model's confidence; top-p is the common default for open-ended text, top-k (or a
   small k with low τ) when you want tight control; greedy/beam for tasks with a single
   correct answer. Greedy repeats because once a phrase is the most likely continuation of
   itself, the loop is self-reinforcing (3.6: "the stand the stand…"). Repetition penalties
   and sampling break it.

5. [expected] Stores, per layer, the K and V vectors of every token processed so far;
   each new step computes Q/K/V for one token and attends over the cache, saving the
   recomputation of all previous positions (O(n²d²) → O(nd²) projection work over a
   generation; measured 2–5× here on a toy model, much more for real LLMs). Costs
   `2 × layers × seq × d × bytes` per sequence and makes decoding bandwidth-bound. MQA
   shares one K/V head across all query heads; GQA shares one per group of heads:
   the cache shrinks by the number of heads (MQA) or the group size (GQA) with little
   quality loss.

6. [open-ended] Tying shares the `vocab × d` matrix: fewer parameters (large for big
   vocabularies and small models), and a consistent geometry where tokens that are
   embedded similarly are also predicted in similar contexts; it often improves
   perplexity on small data. You might not tie when the model is large (the
   embedding is a small fraction of parameters) and separate matrices give
   more capacity, or when input and output vocabularies differ.
