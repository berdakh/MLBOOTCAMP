# Solutions · Chapter 38 · Final evaluation & mock loop

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `bootstrap_ci_grouped`

✅ Reference code: [`code/nb20.py`](solutions.py) (`ex1`), graded in CI.

```python
def bootstrap_ci_grouped(values, groups, n_boot, seed):
    v = np.asarray(values, dtype=float)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(g, weights=v)            # per-group sum
    counts = np.bincount(g).astype(float)       # per-group row count
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), (n_boot, len(sums)))   # groups drawn with replacement
    stats = sums[idx].sum(1) / counts[idx].sum(1)           # mean over rows of drawn groups
    return float(v.mean()), float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))
```

**Why sums and counts.** A replicate's statistic is the mean over all rows
of the drawn groups; with per-group sums and counts it is two fancy-index
sums, so 1,000 replicates over 8,000 users take well under a second.
(For very many groups, draw replicates in chunks to bound memory: the
module's `sv.grouped_bootstrap` does.)

**Common mistakes.** Averaging group means (weights a user with 1 request
like one with 50: a different estimand); resampling rows (intervals too
narrow when rows are correlated: the grader checks the width is > 2× a
row bootstrap on clustered data); a new RNG per replicate seeded `seed + i`
(works, but slow and easy to get wrong).

---

## Exercise 2 · Three-minute walkthrough (written)

✍️ Personal. A model script, with the numbers from our runs (use yours):

> *Problem.* A marketplace home feed: 10,000 users, 4,000 items, eight weeks
> of logged feed requests; the goal is to put the items each user would
> most likely click at the top. The catch is that the logs are produced by
> the current ranking policy and users mostly look at the top positions, so
> clicks are **position-biased**.
>
> *System.* A cascade: retrieval from three sources — FAISS over
> matrix-factorisation embeddings, co-visitation from recent clicks, and
> popularity — gives ~320 candidates with recall 0.85 of the user's true top
> 10. A LightGBM ranker on ten point-in-time features (features from
> earlier days only, position as a training feature fixed at serving) orders
> them; business rules dedupe, drop unlaunched items and cap three per
> category. Served with FastAPI with per-stage timings: about 6 ms p50 and
> 13 ms p99 single-client on a laptop CPU, with fallbacks to popularity if
> the ranker fails, which costs about 0.34 NDCG — so the fallback rate is
> monitored.
>
> *Evaluation.* I chose the system on the validation week, then wrote a
> pre-registration before touching the test week. Primary metric: NDCG@10
> against the simulator's true relevance over the whole catalogue: 0.715
> (95 % CI 0.709–0.721, bootstrap over users) vs 0.635 for embedding retrieval
> alone and 0.38 for popularity. On logged traffic, raw clicks favoured the
> incumbent policy by 0.10 NDCG — but inverse-propensity weighting with
> propensities estimated from randomised traffic reversed that to a 0.29 DCG
> gain for my ranker, matching the true 0.31.
>
> *Next.* The category cap costs 0.16 NDCG offline; I'd test it online on
> long-term engagement rather than decide it offline. And I'd replace the
> LightGBM ranker with a distilled MLP that was 25–100× faster for 0.02 NDCG.

Then check: under 3:30? One number per stage? The bias named before the
result? A "next" that shows judgement?

---

## Checkpoint answers

1. **Test once; what to pre-register.** Every look at test that influences a
   choice turns it into validation, and the reported number becomes
   optimistically biased (selection on noise). Pre-register: the system and
   configuration (incl. what is frozen and what is refreshed), data windows,
   the sample and seed, primary and secondary metrics with exact definitions
   (cutoff k, relevance source, caps), the uncertainty method (grouping unit,
   replicates), comparisons, success criterion, and the rule that nothing
   changes afterwards; save it with a timestamp/hash before scoring.
2. **Old policy wins on clicks.** The old policy decided positions, and
   users examine top positions more, so its top items got clicks largely
   *because* they were on top. Re-ranking that log with any other order moves
   clicked items down and looks worse (position bias / exposure bias). Decide
   with unbiased offline evaluation: IPS or doubly robust estimators using
   propensities from randomised traffic, evaluation on the exploration
   bucket only, or human relevance labels; then interleaving and A/B online.
3. **Propensities and IPS.** Estimate examination by position from
   randomised traffic (CTR by position on random lists ∝ examination;
   20.3 fitted eta 0.88 vs hidden 0.9), or with swap interventions/EM click
   models. IPS divides each click by the propensity of the position where it
   was logged, giving an unbiased estimate of relevance-based metrics.
   Weaknesses: high variance from small propensities (clip weights and report
   it), bias if propensities are wrong or depend on context (device, query),
   only items that were shown can be credited (no help for never-shown
   items), and it assumes the examination model (no trust/attractiveness
   effects).
4. **Bootstrap unit.** Requests of the same user are correlated (taste,
   activity), so the effective sample size is closer to the number of users
   than of requests; resampling requests treats them as independent and
   gives intervals that are too narrow → false confidence, especially for
   heavy users. Resample the unit that was randomised or that carries the
   correlation (user; query for search; session if users are anonymous).
5. **Recall 0.63 vs 0.30.** Logged clicks can only be on items the logging
   policy showed, drawn from its own candidate pool; a different retriever
   is penalised for disagreeing with the old system (exposure bias). True
   relevance has no such bias but only exists in simulation (or via human
   labels). Report the click-based recall with that caveat, prefer
   exposure-corrected or exploration-traffic versions, and validate with
   online tests (and, in ShopStream, the truth-based number).
6. **Walkthrough + weakest part.** See 20-2. Honest weaknesses: the primary
   metric uses simulator truth (unavailable in real life); one test week (no
   seasonality); the ranker was trained on logged candidates but serves
   retrieval candidates (distribution shift); propensities assume
   position-only examination; the latency numbers come from one process
   sharing the GIL with its load generator.
