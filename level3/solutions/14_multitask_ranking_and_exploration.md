# Solutions · 14 · Multi-task ranking & exploration

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## 14-1 · `second_price_auction`

✅ Reference code: [`code/nb14.py`](code/nb14.py) (`ex1`), graded in CI.

```python
def second_price_auction(bids, pctrs, reserve):
    bids, pctrs = np.asarray(bids, float), np.asarray(pctrs, float)
    eligible = np.flatnonzero(bids >= reserve)
    if len(eligible) == 0:
        return None, 0.0
    ecpm = bids[eligible] * pctrs[eligible]
    order = eligible[np.argsort(-ecpm, kind="stable")]     # stable: ties -> lower index first
    winner = int(order[0])
    if len(order) == 1:
        return winner, float(reserve)
    runner = order[1]
    return winner, float(max(reserve, bids[runner] * pctrs[runner] / pctrs[winner]))
```

**Why this price.** The winner keeps winning as long as `bid₁ · pCTR₁ ≥ bid₂ · pCTR₂`,
so the smallest winning bid is `bid₂ · pCTR₂ / pCTR₁`; charging exactly that per
click (only when a click happens) makes the expected payment per impression equal
to the runner-up's eCPM. It never exceeds `bid₁`. Production GSP adds a small
increment (one cent), quality scores beyond pCTR, and per-slot position effects.

---

## 14-2 · `thompson_sampling`

✅ Reference code: [`code/nb14.py`](code/nb14.py) (`ex2`).

```python
def thompson_sampling(true_ps, n_rounds, seed):
    p = np.asarray(true_ps, float)
    rng = np.random.default_rng(seed)
    alpha, beta = np.ones(len(p)), np.ones(len(p))
    regret, total = np.empty(n_rounds), 0.0
    for t in range(n_rounds):
        arm = int(np.argmax(rng.beta(alpha, beta)))
        reward = rng.random() < p[arm]
        alpha[arm] += reward
        beta[arm] += 1 - reward
        total += p.max() - p[arm]
        regret[t] = total
    return regret
```

**Notes.** Expected (pseudo-)regret `max p − p[arm]` is far less noisy than
realised regret (clicks lost), which is why simulations report it. Thompson
sampling's regret grows like `O(log T)` per suboptimal arm; the grader checks that
the last third of rounds adds less regret than the first third.

---

## 14-3 · Segment miscalibration and the exploration budget

✍️ Model answer:

(a) Only auctions where a video ad competes with non-video ads are affected. With
video pCTR ×1.1, a video ad's eCPM rises 10%: it now wins auctions where it was
within 10% of the best non-video ad (misallocation: the true best ad loses). Prices:
when a video ad wins against a non-video runner-up, its price per click
`bid₂·pCTR₂/pCTR₁` *falls* by ~9% (its own inflated pCTR is in the denominator);
when a non-video ad wins against a video runner-up, it pays ~10% *more*. Video vs
video auctions are unchanged (the factor cancels).

(b) Short term: video advertisers win more impressions, often at lower per-click
prices; their realised clicks are 10% below what the system expected, so platform
revenue per impression drops on the misallocated auctions (the 14.4 simulation:
a per-advertiser bias moved win share and value/cost while a uniform one barely
changed anything). Non-video advertisers lose volume and pay more. Long term,
advertisers' measured ROI shifts, they re-bid (video advertisers who get cheap
but low-converting traffic may lower bids; others raise or leave), and the
auction becomes less efficient: revenue and advertiser value both fall.

(c) Before launch: calibration (pred/obs) *per segment* (format, placement, country,
new vs old ads) on recent data, not just overall NE/AUC; a counterfactual auction
replay ("re-run yesterday's auctions with the new pCTR") to see shifts in win share,
price and revenue per segment. After launch: online pred/obs per segment from
real clicks, segment revenue and win share in the A/B test, advertiser ROI. Fix:
a calibration layer per segment (Platt/isotonic fitted on recent data, refreshed
often), find the root cause (feature drift for video, label pipeline).

(d) Treat exploration as an investment with a measurable return: run an experiment
with several budgets (e.g. 0.5%, 2%, 5%) and measure the short-term cost (revenue
and CTR loss on exploration traffic) against the long-term gain (how quickly new ads
reach their true eCPM, how many good new ads are found, advertiser retention for new
advertisers, catalogue freshness). Use adaptive exploration (Thompson/UCB with
priors from similar ads) so the budget concentrates on uncertain ads with high
upside, and cap per-ad exploration impressions.

---

## Checkpoint · 14

1. ✍️ MMoE addresses negative transfer in hard-sharing multi-task models: with one
   shared bottom, conflicting tasks pull the representation in different directions.
   MMoE has several expert networks and, per task, a softmax gate (a function of the
   input) that mixes experts per example, so tasks can share where they agree and
   specialise where they do not. Not worth it when tasks are highly aligned, data is
   small, or latency/compute budgets are tight; in 14.2 MMoE and shared-bottom were
   within ~0.002 NE of each other on our one-epoch CPU run, which is typical when the
   conflict is mild.

2. ✍️ CVR is only labelled on clicked impressions but applied to all impressions:
   training and serving distributions differ (SSB), plus conversions are sparse.
   ESMM models `pCTCVR = pCTR × pCVR` and trains pCTR (click labels) and pCTCVR
   (click-and-convert labels) on **all** impressions, so pCVR is learned over the
   entire space and shares embeddings with the data-rich CTR task. Weaknesses: pCVR
   gets gradient only through the product (weak where pCTR is small), and errors in
   pCTR propagate into pCVR; it does not fix the selection bias exactly (later work
   adds inverse-propensity or doubly-robust corrections, e.g. ESCM²). In our simulation
   (one epoch, CPU budget) ESMM improved CTR NE but its pCVR ranked worse than the
   naive multi-task model's: re-measure on your data rather than assuming the paper's gains.

3. 🔬 Score = `pCTR^a · pCVR^b · value` (or a weighted sum of calibrated expected values:
   `w₁·pCTR + w₂·pCTR·pCVR·value + …`). Choose weights by mapping the Pareto frontier
   offline (14.3) and then running online experiments at a few points; pick according
   to the business objective (e.g. maximise value subject to CTR ≥ x), revisit when
   models change. Products of calibrated probabilities keep a meaning (expected value
   per impression); exponents are tuning knobs and break that meaning.

4. ✅ eCPM = pCTR × bid × 1000, the expected revenue per thousand impressions.
   Second price: the winner pays per click the minimum needed to keep winning,
   `bid₂·pCTR₂/pCTR₁`; truthful bidding is dominant. A uniform factor `k` on all pCTRs
   multiplies every eCPM by `k`: the ranking and the per-click price (ratio of pCTRs)
   are unchanged; only comparisons with things not scaled (reserve in eCPM terms, other
   formats, organic content) shift. A segment error changes relative eCPMs, so winners
   change: the 14.4 simulation showed advertiser 0's win share and value/cost moving
   when only its pCTR was biased.

5. 🔬 ε-greedy: simple, explores uniformly forever (linear regret), wastes impressions on
   clearly bad ads. UCB: deterministic optimism, logarithmic regret, needs care with
   delayed feedback and many arms. Thompson: samples from the posterior, logarithmic
   regret, handles priors and delayed/batched feedback naturally, best in our 14.5
   runs. For cold start: Thompson with an informative prior from similar ads (category,
   advertiser history), with a capped exploration share or boost only for ads with few
   impressions, and logged propensities for later off-policy evaluation.

6. ✍️ Labels: conversion events with value, attribution window, delayed-feedback
   handling (13.8), deduplication. Models: pCTR, pCVR (ESMM/multi-task with shared
   embeddings), maybe a value regression model; calibration layers per segment. Ranking:
   `pCTR × pCVR × value` (or advertiser's target CPA/ROAS bidding: bid derived from value).
   Offline metrics: NE/AUC per task, calibration of pCVR and predicted value, a replayed
   auction simulation. Online: A/B test with revenue, advertiser conversions and ROI,
   CTR, user experience guardrails; ramp slowly because advertisers' bidding adapts.
