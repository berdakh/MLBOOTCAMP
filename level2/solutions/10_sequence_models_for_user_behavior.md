# Solutions · 10 · Sequence models for user behaviour

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 10-1 · `build_sequences`

✅ Reference code: [`code/nb10.py`](code/nb10.py) (`ex1`), graded in CI.

```python
def build_sequences(df, max_len):
    d = df.sort_values(["user_id", "ts"], kind="stable")
    users = np.sort(d["user_id"].unique())
    seqs = np.zeros((len(users), max_len), dtype=np.int64)
    for r, (_, items) in enumerate(d.groupby("user_id", sort=True)["item_id"]):
        last = items.to_numpy()[-max_len:] + 1
        seqs[r, max_len - len(last):] = last
    return users, seqs
```

**The details the grader checks.** A **stable** sort (`kind="stable"`), so two
clicks with the same timestamp keep their logged order (the default quicksort
does not promise that). Keep the **last** `max_len` items, not the first. Shift
by +1 so item 0 does not collide with padding. Write into the **right** end of the
row (left padding).

The loop is fine for 10,000 users. `shoprank.seqrec.build_sequences` does the
same without a Python loop: it computes each event's distance from the end of its
user's history and scatters the events directly into the array. Worth
knowing for 100 million users.

---

## 10-2 · `causal_mask`

✅ `torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)`.

`diagonal=1` excludes the diagonal: a position may attend to itself (it needs
its own item). Watch the convention: PyTorch's boolean `attn_mask` uses
`True` = *blocked*; some code (and `scaled_dot_product_attention`'s boolean mask) uses
`True` = *allowed*. Getting it backwards gives a model that sees *only* the future.
Test 1 in 10.4 catches that in one line.

---

## 10-3 · `padding_mask`

✅ `torch.arange(n)[None, :] < (n - lengths)[:, None]`.

With left padding, the first `n - length` positions are padding. A length-0
row is all padding (and, combined with the causal mask, would be all-NaN
without the diagonal fix); a length-`n` row has no padding.

---

## 10-4 · Before approving the A/B test (written)

✍️ Personal. A strong answer asks for checks that could *change the decision*:

1. **Full-catalogue metrics, same protocol for both models.** Sampled metrics
   compress differences and can reorder models (Krichene & Rendle, 2020; section
   10.6 showed sampled HR penalise a change the full metric was indifferent to).
   *Changes my mind:* the full-catalogue HR@10/NDCG@10 difference is within
   noise (bootstrap over users), or reverses.
2. **A time-respecting split and no leakage.** Trained only on data before the
   evaluation window; no item or user features computed over the evaluation
   window; the same candidate filtering as production (e.g. already-purchased items
   removed). *Changes my mind:* the gain shrinks or disappears under the
   time-respecting protocol, or most of it comes from re-recommending items the
   user just clicked (check the repeat rate of hits).
3. **The gain holds where the product is.** Slices that matter (new vs heavy
   users, cold-start items), coverage/popularity of recommendations, and
   serving cost (latency at production sequence lengths). *Changes my mind:*
   the model wins only for heavy users and loses for new users, collapses
   coverage to the head of the catalogue, or busts the latency budget.

Credit also: an offline replay with a debiased metric (IPS on exploration
traffic, notebook 05), or an interleaving test before a full A/B test (notebook 12).

---

## Checkpoint · 10

**1. Why a causal mask.** Training computes predictions for *every* position of the
sequence in one forward pass (teacher forcing); the causal mask makes position `t`'s output depend only
on items `≤ t`, which equals running a separate model on each prefix, at the
cost of one. Without it, position `t` attends to the item at `t + 1`, which is
its own target: the training loss collapses towards zero and the model is
useless at inference, where the future is not available.

**2. Left vs right padding.** Left: the latest item is always in the last
column, so the prediction is read at index `-1` for everyone; positional
embeddings can be aligned from the right (recency); but padding queries have fully
blocked rows, hence NaNs, hence the diagonal fix. Right: no blocked rows (each
real query sees itself), but the prediction position differs per user
(`length - 1`, a gather), and absolute positions count from the oldest kept item.
The padding mask changes accordingly: True at the end instead of the start.

**3. Leave-one-out leakage.** Holding out each user's last event but training
on all other events includes *other users' later behaviour*: trends, new
items' popularity, co-visitation from the future. The model is tested on
a moment it partially saw. Here: train only on days 0–41; targets are each
user's last click in days 42–48; inputs only clicks before the target.

**4. Sampled vs full metrics.** With 100 uniform negatives the target competes
with mostly obscure items, so the metric measures "beats random long-tail items",
not "beats the items that really compete for the top 10". Models differ mainly in
how they rank the strong competitors, which sampling rarely includes, so
differences shrink and can reverse (Krichene & Rendle 2020 show sampled metrics
need not preserve the ordering). Acceptable: when full ranking is truly
infeasible, the same negatives are used for every model, and conclusions are
checked on a full-catalogue subset, or with corrected estimators.

**5. 10 million items.** Training: the full softmax is too expensive per
position; use sampled softmax / in-batch negatives with a logQ correction
(notebook 08), or hierarchical/negative sampling, and mixed precision. The item
embedding table becomes the dominant memory cost (shard it). Serving: score
with ANN retrieval over item embeddings (FAISS, notebook 08) using the
sequence model's last hidden state as the query, then rerank a few hundred
candidates.

**6. Does it use order?** Ablate order at evaluation time: shuffle each
history (as in 10.7) and/or train a bag-of-items model (mean of item embeddings,
no positions) with the same capacity. If shuffled histories score as well,
order is not being used. Also probe: does swapping only the last two items change
the prediction? Section 10.7 measured 0.0700 → 0.0687 HR@10 with shuffled
histories (✅): on ShopStream, order carries little signal.
