# Solutions · 12 · Advanced experimentation

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 12-1 · `cuped_adjust`

✅ Reference code: [`code/nb12.py`](code/nb12.py) (`ex1`), graded in CI.

```python
def cuped_adjust(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return y.copy()
    theta = np.cov(x, y, ddof=1)[0, 1] / var_x
    return y - theta * (x - x.mean())
```

**Why the mean of `x` is subtracted:** it keeps `mean(y_cv) = mean(y)`, so
the adjusted metric stays in the original units, and the difference between
arms is unchanged in expectation, because `x` is pre-treatment and so balanced across arms.
`θ` is the OLS slope of `y` on `x`, estimated on **both arms pooled** (per-arm θ's are also valid but slightly noisier;
they become a regression with a treatment-by-covariate interaction).

**Common mistakes.** Mixing `np.cov` (ddof=1 by default) with `np.var` (ddof=0
by default): θ is then off by `n/(n-1)`, a negligible error at large n but the grader
asks for consistency. Dividing by zero for constant `x` (new users with no
pre-period): return `y`.

Measured in 12.1 (✅): ρ = 0.71, variance ratio 0.50 vs predicted 0.49.

---

## 12-2 · `delta_method_ratio_var`

✅ Reference code: `ex2`.

```python
def delta_method_ratio_var(num, den):
    c, n = np.asarray(num, float), np.asarray(den, float)
    mc, mn = c.mean(), n.mean()
    cov = np.cov(c, n, ddof=1)
    return (cov[0, 0] / mn**2 - 2 * mc * cov[0, 1] / mn**3 + mc**2 * cov[1, 1] / mn**4) / len(c)
```

**Where it comes from.** `f(a, b) = a / b`, gradient `(1/μn, −μc/μn²)`; the
delta method says `Var f(c̄, n̄) ≈ ∇fᵀ Σ ∇f` with `Σ` the covariance matrix of the
*means* (`= cov(c, n) / N`). Expanding gives the three terms. A useful
equivalent: compute the per-user **linearised** values
`ℓ_i = (c_i − R n_i) / μn` with `R = c̄/n̄`; then `Var(R̂) ≈ var(ℓ)/N`, which also plugs
straight into CUPED (regress `ℓ` on a pre-period covariate).

Measured in 12.2 (✅): delta SE 0.00168, bootstrap SE 0.00170, true SD over
400 A/A experiments 0.00169, naive binomial SE 0.00061.

---

## 12-3 · `team_draft_interleave`

✅ Reference code: `ex3`.

```python
def team_draft_interleave(list_a, list_b, rng):
    k = len(list_a)
    items, teams, shown = [], [], set()
    picks = {"A": 0, "B": 0}
    ranked = {"A": list(list_a), "B": list(list_b)}
    while len(items) < k:
        if picks["A"] < picks["B"] or (picks["A"] == picks["B"] and rng.random() < 0.5):
            team = "A"
        else:
            team = "B"
        candidates = [x for x in ranked[team] if x not in shown]
        if not candidates:                      # cannot happen for two lists of length k,
            team = "B" if team == "A" else "A"  # but keeps the function total
            candidates = [x for x in ranked[team] if x not in shown]
            if not candidates:
                break
        items.append(candidates[0]); teams.append(team)
        shown.add(candidates[0]); picks[team] += 1
    return items, teams
```

**Properties the grader checks:** no duplicates; every item is its team's
best unused item at the moment of the pick (that is what makes the credit fair);
team sizes never differ by more than one, and are equal after every complete round; a fair coin
decides who picks first in each round; identical rankings give the same list.

**Credit.** A click on position `i` counts for `teams[i]`. Per user (or per
query), the ranker with more credited clicks wins; the test is a sign test or a
t-test on per-user credit differences (the notebook used the latter). Items that
both rankers ranked highly are credited to whichever team picked them, which adds noise but no bias,
thanks to the coin flips.

---

## 12-4 · Launch memo (written)

✍️ Personal. A strong memo, under the notebook's criteria:

* **Decision: do not ship as is.** The primary metric is positive and
  significant (+1.8%, CI +0.6% to +3.0%), and the SRM check passes (p = 0.21), but the
  **p95 latency guardrail is violated** (+7% against a 5% limit). Guardrails can
  block a launch, and this one does.
* **Purchases** (-0.4%, CI -1.3% to +0.5%): the lower bound is below the -1%
  non-inferiority margin, so non-inferiority is *not demonstrated*. That is a
  second reason not to ship, even before latency is fixed.
* **Secondary metrics**: two of eleven significant is weak evidence: 0.55 are
  expected by chance at α = 0.05, and two or more happen about 10% of the time
  even with no effect (independent metrics; correlated metrics make clusters more likely). "iOS in Germany +9%"
  is an unplanned segment result, exploratory only, a hypothesis for a future test.
* **Next steps:** profile the latency regression (feature fetches? a larger
  model?) and fix it, or distil the ranker (notebook 18); run a longer or
  larger test, powered for the purchases margin; keep the same pre-registered
  criteria.
* **To the team:** the ranker improves engagement, which is real, measured progress;
  it is blocked by cost and an unresolved commerce risk, not by its quality; a clear
  path to launch exists.

---

## Checkpoint · 12

**1. CUPED for a PM.** "We subtract from each user's result the part we could
already predict from their behaviour before the test, so the test only has to
see through the unpredictable part. Same answer, fewer users." Expected reduction:
variance × (1 − ρ²); ρ = 0.71 for clicks per user here, which halves the variance (✅).
Fails when the covariate barely predicts the metric (new users, per-user CTR with
ρ ≈ 0.35 here, rare conversions), or worse, when the "pre-period" covariate is
affected by treatment (bias).

**2. Pooled CTR and the t-test.** Impressions are not independent units:
randomisation is by user, and clicks within a user are correlated; and pooled
CTR is a ratio of two random sums, not a mean of i.i.d. values. The delta method
linearises `R = c̄ / n̄` around the means and gives its variance from per-*user*
sums, valid at the randomisation unit. Validate it with A/A tests (false-positive rate ≈ α:
measured 0.043 vs 0.507 for the naive test, ✅) and against a user-level bootstrap.

**3. +11% in one of 30 segments.** With 30 segments, one or two extreme
estimates are expected from noise; small segments are the noisiest. Section 12.3
showed a uniform true +2% effect produce segment estimates from +0.1% to +5.4%
with half of them "significant" (✅). Ask: was the segment pre-registered? Does
it survive multiple-testing correction? Is there a mechanism? If it is still
interesting, run a new experiment targeted at that segment, with its own pre-registered metric. Do not launch on it.

**4. Team-draft interleaving.** Each round, a coin decides which ranker picks first; each picks its
best unused item; clicks are credited to the picking team. It needs far fewer
users because each user compares both rankers on the same request, so
between-user variance (activity, propensity to click) cancels: the notebook
measured about 900 users for 80% power vs about 20,000 for an A/B test (✅,
about 22x). It cannot measure whole-page or long-term effects: revenue,
retention, latency, diversity of the final page, or effects when the two
rankers' items interact (e.g. a page of near-duplicates).

**5. Wins offline, loses online.** (a) **Position bias/feedback loop:** logged clicks
are concentrated where the old policy put items, so offline NDCG rewards
imitating the old policy: 12.5 showed the logging policy at 0.76 NDCG vs true
relevance at 0.68, while online CTR was the reverse (✅). (b) **Distribution
shift/selection:** offline data only contains items the old policy showed; a
new ranker surfacing unseen items gets no credit or blame for them. Others: a metric
mismatch (clicks vs purchases), novelty effects, latency. Fixes: IPS-weighted
metrics with exploration traffic (notebook 05), validating offline metrics
against a history of A/B outcomes, interleaving as an online filter.

**6. Launch criteria.** One primary decision metric with its MDE, power,
duration and analysis method (CUPED, delta method); validity checks (SRM, A/A
health); guardrails with explicit thresholds or non-inferiority margins;
decision rules for every combination; how secondary metrics and segments are
treated (exploratory, corrected). Written before the test, so the data cannot
choose the rule (the garden of forking paths), and so that stakeholders agree
on the trade-offs (e.g. how much latency a CTR gain may cost) while nobody yet
knows which side of them they will be on.
