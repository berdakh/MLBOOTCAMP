# Solutions · Chapter 23 · Implicit feedback & position bias

Attempt each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `estimate_propensities`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def estimate_propensities(random_imps):
    ctr = random_imps.groupby("position")["clicked"].mean().sort_index()
    return (ctr / ctr.iloc[0]).to_numpy()
```

**Why it works.** On exploration traffic the item at each position is a
uniformly random draw, so mean relevance is the same at every position and
CTR(p) ∝ θ_p. Dividing by CTR(0) removes the unknown constant (only ratios
are identifiable).

**Common mistakes.**

* Forgetting `sort_index()` and relying on row order. The grader shuffles rows.
* Using all traffic instead of `is_random`: the policy puts better items on
  top, so the curve comes out too steep (Section 5.3 of the lab: every position
  below the truth).
* Treating the ten ratios as independent measurements. They share one noisy
  denominator, CTR(0); in the lab that single number being about 2% low
  moved every ratio up together. Fitting a smooth curve to all positions
  (5.3) fixes both the noise and the shared-denominator problem.

**Push further.** Bootstrap over *requests* (the randomised unit). If the same
users contribute many requests, also try a bootstrap over *users* and compare
the widths.

---

## Exercise 2 · `ips_ctr_estimate`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def ips_ctr_estimate(clicks, positions, propensities, clip=None):
    w = 1.0 / np.asarray(propensities, dtype=float)[np.asarray(positions)]
    if clip is not None:
        w = np.minimum(w, clip)
    return float(np.mean(np.asarray(clicks, dtype=float) * w))
```

**Why it is unbiased (without clipping).** E[c | p] = θ_p · r, so
E[c / θ_p] = r. Averaging over impressions gives the mean relevance of the
shown items, i.e. the CTR they would have had if examined like position 0.

**Common mistakes.** Dividing the *sum of weighted clicks* by the *sum of
weights* (that is the self-normalised estimator, a different, biased but
lower-variance quantity: acceptable, but it should be stated). Clipping the propensity
from below instead of the weight from above is the same thing written
differently; the important point is to be consistent (`min(1/θ, M)` ⇔ `1/max(θ, 1/M)`).

---

## Exercise 3 · The cost of exploration traffic (written)

[expected] Numbers from our run (other runs will be close):

* Random traffic CTR ≈ 0.062 vs policy ≈ 0.141 per impression, i.e. about
  0.79 fewer clicks per exploration request (10 impressions).
* About 245,000 requests in 56 days ≈ 4,370 per day; 5% exploration ≈ 220
  requests per day ≈ **170 clicks lost per day**, out of about 6,000: roughly
  **3% of clicks** (and, proportionally, purchases).

[open-ended] The case for keeping it, in a form a product manager can act on:

1. **What it buys.** The only unbiased estimate of position propensities, and
   therefore unbiased offline evaluation and training (5.5–5.6). Without it,
   every offline comparison favours the current ranker (5.2): good models
   would be rejected and models that merely imitate production deployed.
2. **Size the value.** In 5.6 the naive click model gave about 1.63 true
   expected clicks per request and the IPS model about 1.72: about +5% on
   the 95% of traffic the model serves, against about 3% of clicks spent on
   exploration. And one avoided wrong launch decision (5.2's paradox) can be
   worth more than a year of exploration.
3. **Make it cheaper.**
   * Shrink it: propensities need far less than 5% once a power-law fit is
     used (try 1% and look at the bootstrap width).
   * Randomise less: swap *adjacent pairs* (RandPair) or shuffle only
     positions 3–10 instead of the whole list; users barely notice.
   * Randomise among good candidates only (top-20 of the policy), so the
     shown items are still reasonable.
   * Harvest interventions from A/B tests that are run anyway (different rankers
     put the same item at different positions).
4. **Guard rails.** Cap exploration per user per day; exclude new users and
   high-value surfaces (checkout); monitor its CTR as a health metric.

---

## Checkpoint answers

**1. Examination hypothesis and PBM.** A click requires examination and
relevance; the PBM makes examination depend only on position:
P(c) = θ_p · r_ui. Violations: users stop after a satisfying click (the
**cascade model** and its descendants: DCM, UBM, DBN, where examination of p
depends on clicks above); attractiveness of the snippet/image differs from
relevance; examination depends on the device and layout (grids, carousels),
and on the item itself (a striking image draws the eye).

**2. Loses offline, wins online.** The offline NDCG was computed on clicks
logged *at production's positions*. Any re-ranking moves clicked items away
from the highly examined top slots, so it is penalised for disagreeing with
production regardless of quality (5.2: true-relevance order 0.68 vs logged
0.76 on clicks; 1.84 vs 1.41 true expected clicks). The A/B test measures
clicks under the new positions, so it sees the real improvement. Fixes: IPS
or doubly-robust offline estimates; interleaving (Chapter 30).

**3. No exploration traffic.** Options: (a) start a small randomisation
(RandPair swaps, top-k shuffles), which costs some engagement; (b)
intervention harvesting from past A/B tests or ranker changes, which is free
but needs enough overlap and careful matching; (c) regression EM / click
models fit to ordinary logs, which cost no traffic but rest on modelling
assumptions that cannot be fully checked; (d) eye-tracking/user studies for a
first curve (expensive, small, but useful as a sanity check).

**4. IPS estimator.** V̂ = (1/N) Σ_requests Σ_i (θ_{new pos(i)} / θ_{logged pos(i)}) c_i.
Unbiased when the click model holds (PBM), the propensities are correct and
every position needed has θ > 0 in the logs. It can only evaluate
re-orderings of items that were *shown*: items never logged have no clicks
to re-weight (the support / coverage problem). Rankings far from the logging
policy get large weights and high variance.

**5. Clipping.** Caps the weights: lower variance, at the cost of a downward
bias concentrated exactly where weights are large (low positions, rankings
far from production). Choose the cap by measured RMSE: simulate (as in 5.5)
or bootstrap and inspect bias/variance, or use self-normalised / doubly
robust estimators instead of a hard cap. Report the cap that was used, since
the estimate depends on it.

**6. "Previous model's score" as a feature.** It encodes the *position* the
item was shown at, so a model trained on clicks learns examination, not
relevance, and an offline click metric rewards that (5.6: naive model 0.77
click-NDCG vs IPS 0.74, but 0.90 vs 0.94 NDCG on truth). Check: compare
against IPS-weighted training or position-aware training; evaluate on
exploration traffic or with IPS; look at how much of the gain disappears
when evaluating on randomised traffic; and watch for a feedback loop (the
new model copies the old one, which then trains the next one).
