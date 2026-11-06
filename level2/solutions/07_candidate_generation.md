# Solutions · 07 · Candidate generation

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 07-1 · `covisitation`

✅ Reference code: [`code/nb07.py`](code/nb07.py) (`ex1`), graded in CI.

```python
def covisitation(pairs_df, top_n):
    d = pairs_df[["session_id", "item_id"]].drop_duplicates()       # once per session
    p = d.merge(d, on="session_id", suffixes=("", "_nb"))            # all pairs in a session
    p = p[p["item_id"] != p["item_id_nb"]]                           # not yourself
    counts = p.groupby(["item_id", "item_id_nb"]).size().rename("n").reset_index()
    counts = counts.sort_values(["item_id", "n", "item_id_nb"], ascending=[True, False, True])
    top = counts.groupby("item_id").head(top_n)
    return {int(i): g.tolist() for i, g in top.groupby("item_id")["item_id_nb"]}
```

**Why it works.** The self-merge on `session_id` is the "shuffle by session,
emit all pairs" step of the classic MapReduce job; `groupby(...).size()` is
the reduce. `drop_duplicates` first makes the count "sessions in which both
appear", not "click pairs".

**Common mistakes.** Forgetting the de-duplication (a user clicking the same
item three times creates 3× the pairs); sorting with the default unstable
sort and no tie-breaker (the order of equal counts then changes between
runs); keeping self-pairs.

**At scale.** The pair count per session is O(len²): cap session length (or
sample), and handle very popular items specially (they appear in most
sessions and dominate the shuffle: skew). Normalise counts, e.g. by
√(pop_i · pop_j) (cosine) or by the item's popularity, or every item's top
neighbours become the global best-sellers.

---

## 07-2 · `recall_at_k_batch`

✅ Reference code: [`code/nb07.py`](code/nb07.py) (`ex2`).

```python
def recall_at_k_batch(candidates, truth, k):
    vals = []
    for u, rel in truth.items():
        rel = set(rel)
        if not rel:
            continue
        got = set(list(candidates.get(u, []))[:k])
        vals.append(len(got & rel) / len(rel))
    return float(np.mean(vals)) if vals else float("nan")
```

**Design choices to state when you report it.** It is a *macro* average
(every user counts equally; a micro average, Σ hits / Σ |truth|, weights
heavy users more). Users missing from `candidates` count as 0, so a
generator cannot look good by abstaining. Recall@K is capped below 1 when
|truth| > K: some teams use min(|truth|, K) in the denominator; say which.

---

## 07-3 · Retrieval for 50 million listings (written)

✍️ One strong answer, as bullets:

* **Budget split** (15 ms): generators run in parallel, each ≤ 8–10 ms,
  merge + de-dup ≤ 2 ms, leaving headroom for tail latency.
* **Main generator: two-tower embeddings + ANN** (notebook 08), ~300–500
  candidates. The item tower uses content features (category, title/image
  embeddings, price), so new listings get vectors at upload time; the index
  is refreshed incrementally (minutes) because 20% are new each week.
* **Item-to-item co-visitation** from the last session's clicks, ~100: fresh
  intent ("just looked at running shoes"), recomputed hourly, top-50
  neighbours per item stored in a key-value store.
* **Fresh / new-listing generator**, ~50: new items in the user's
  categories and locale, with an exploration quota so new items earn clicks.
* **Popularity by context** (country × category × time-of-day), ~50: the
  fallback for cold users and for outages of the other generators.
* **Merge**: quotas at first; later a light pre-ranker (small model on cheap
  features) that scores the union and passes ~500 to the ranker.
* **Offline evaluation**: recall@K per generator *and of the union* on a
  time-split week, on several truths (clicks, purchases, random-traffic
  clicks), sliced by user tenure and item age; coverage and popularity bias
  of the candidates; the *marginal* recall each generator adds to the union.
* **Online monitoring**: per-generator latency (p50/p99), timeout rate,
  empty-result rate, share of final impressions and clicks sourced from each
  generator, index freshness (age of the newest indexed item), new-item
  impression share.
* **Kill switches**: every generator can time out without failing the
  request; popularity is always available.

---

## Checkpoint · 07

**1. Recall vs NDCG.** Retrieval's job is to *not lose* good items; their
order is re-decided by the ranker, so the right metric is how many relevant
items survive the cut (recall@K). Tuning retrieval on final-list NDCG mixes
the two stages: a change can look bad because the ranker is not yet trained
on the new candidate distribution (and vice versa), and NDCG over a short
list ignores most of the K candidates. Still, *also* check the end-to-end
metric, because the two stages interact.

**2. Implicit ALS.** p_ui = 1 if the user interacted, else 0; c_ui = 1 + α
r_ui is the confidence (more clicks → more trust; zeros get weight 1, not
0, so they act as weak negatives). The loss is a weighted least squares,
quadratic in X for fixed Y and vice versa, so we alternate closed-form
solves. One user's solve is cheap because YᵀC_uY = YᵀY + Yᵀ(C_u − I)Y: YᵀY
is shared by all users, and (C_u − I) is non-zero only on that user's few
items: O(n_u f² + f³) per user.

**3. Co-visitation at 100 M sessions/day.** Map: per session, de-duplicate
items, cap length, emit (i, j) pairs (possibly weighted by time gap). Shuffle
by i (or by the pair). Reduce: sum counts per (i, j), then per i keep top-N
by (normalised) count. Skew: very popular items appear in a huge share of
sessions, so the reducer for that key is overloaded: salt hot keys, cap
pairs per item, or down-sample popular items; combine partial counts
map-side (combiners). Run incrementally: today's counts plus decayed
yesterday's.

**4. +8% recall, flat A/B.** (a) The ranker does not surface the new
candidates (trained on the old candidate distribution, or the extra hits
fall outside the shown top-10); (b) the offline recall was on exposure-
biased clicks: the generator found items the old system showed, which the
old system was already showing; (c) the metric moved on segments that
matter little (heavy users) or the effect is smaller than the test's
power; also latency added by the new generator can cost engagement.

**5. Cold start.** New item: content-based embedding (item tower on
metadata, text, image) and an exploration quota (boosted impressions, e.g.
a bandit with a prior from similar items); graduates when it has enough
interactions for collaborative signals (e.g. N clicks, or when its
collaborative embedding is stable). New user: context popularity
(country, device, referrer), onboarding choices, session-based generators
(co-visitation from the first clicks) and a user tower that uses features,
not only the id; graduates after a few sessions.

**6. Quotas.** Grid/coordinate search on validation recall of the union
(and the end-to-end ranked metric), subject to constraints like a minimum
fresh-item share; confirm with A/B tests. At scale, replace fixed quotas
with a learned merge: a light pre-ranker scoring the union of all
candidates with cheap features (including "which generators proposed it"
and their scores), or per-generator quotas adapted per user/context.
