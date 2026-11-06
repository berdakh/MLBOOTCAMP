# Solutions · 11 · A/B testing fundamentals

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 11-1 · `sample_size_per_arm`

✅ Reference code: [`code/nb11.py`](code/nb11.py) (`ex1`), graded in CI.

```python
def sample_size_per_arm(p0, mde_abs, alpha=0.05, power=0.8):
    p1 = p0 + mde_abs
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(math.ceil(z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / mde_abs**2))
```

`(1.96 + 0.84)² ≈ 7.85` is worth memorising: for 10% → 11%,
`7.85 × (0.09 + 0.0979) / 0.0001 ≈ 14,749` per arm. The notebook simulated
2,000 experiments at that size and measured a power of 0.805 (✅).

**Variants you may meet.** Some texts use the pooled variance `2 p̄(1-p̄)`
under the null for the `z_{α/2}` term; the numbers differ by well under 1%
for small effects. One-sided tests use `z_{1-α}` (1.645). All share the key
fact: `n ∝ 1/δ²`.

---

## 11-2 · `srm_pvalue`

✅ Reference code: `ex2`.

```python
def srm_pvalue(n_a, n_b, expected_ratio=0.5):
    total = n_a + n_b
    expected = [total * expected_ratio, total * (1 - expected_ratio)]
    return float(stats.chisquare([n_a, n_b], f_exp=expected).pvalue)
```

By hand: `χ² = Σ (observed − expected)² / expected`, 1 degree of freedom,
`p = stats.chi2.sf(χ², 1)`. Equivalent to a two-sided z-test of the
proportion `n_a / N` against `expected_ratio`.

**Sense of scale.** 50,400 / 49,600 (a 50.4% share) gives p ≈ 0.011 (✅ computed); the same
50.4% share of 2 million users gives p ≈ 1e-29. SRM checks become extremely sensitive at
scale, which is what you want: at 2 million users, a 0.4-point imbalance is
4,000 users that something lost or added.

---

## 11-3 · `aa_false_positive_rate`

✅ Reference code: `ex3`.

```python
def aa_false_positive_rate(n_experiments, n_users_per_arm, n_looks, alpha=0.05, p=0.1, seed=0):
    rng = np.random.default_rng(seed)
    per_look = n_users_per_arm // n_looks
    conv = rng.binomial(per_look, p, size=(n_experiments, 2, n_looks)).cumsum(axis=2)
    n = per_look * np.arange(1, n_looks + 1)
    pooled = (conv[:, 0] + conv[:, 1]) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * 2 / n)
    z = (conv[:, 1] - conv[:, 0]) / n / se
    pvals = 2 * stats.norm.sf(np.abs(z))
    return float((pvals < alpha).any(axis=1).mean())
```

The trick is to simulate **conversions per batch**, not individual users:
`binomial(per_look, p)` per arm per look, then `cumsum` gives the cumulative
counts every look sees. 4,000 experiments × 10 looks run in milliseconds.
Measured (✅): 0.047 with one look, 0.198 with ten looks, 0.009 at α = 0.01.

Why it rises: the cumulative z-statistics at successive looks are strongly
correlated but not identical, a random walk that gets several chances to
cross ±1.96. With infinitely many looks, the probability of eventually crossing is 1.

---

## Checkpoint · 11

**1. User randomisation, impression-level CI.** The binomial SE assumes
independent impressions, but impressions from one user share that user's
propensity to click (and heavy users contribute many). The true variance of the
pooled CTR is larger, so the CI is **too narrow** and the test fires too often.
Show it with an A/A test: the notebook measured a 50% false-positive rate for
the naive test vs 4% for a user-level test (✅), with the naive SE 2.8x too
small. Fix: per-user metric + t-test, or the delta method for the pooled ratio (notebook 12).

**2. Request / user / cluster.** *Request*: when the treatment has no memory
and outcomes are per request (e.g. a search ranking tweak on query-level CTR);
interference: carry-over (a user who saw B learns to scroll and behaves
differently under A), and an inconsistent experience. *User*: the default
for personalisation and any user-level outcome (retention, purchases);
interference: social features (shared content, referrals), and users sharing
devices. *Cluster* (city, seller, ad auction, time slot via switchbacks):
when users compete for shared resources (marketplace supply, ad budgets, drivers);
interference: a treatment that sends more buyers to some sellers takes them from
control's sellers, biasing the comparison (and fewer units mean more variance).

**3. 5% baseline, +0.25 pp.** `7.85 × (0.05·0.95 + 0.0525·0.9475) / 0.0025² ≈
122,000` users per arm (✅ `ex1(0.05, 0.0025)` = 122,121). Half the effect →
about **4x**: 482,856 per arm. Then discuss: run longer, use variance
reduction (CUPED), a more sensitive metric, or accept a larger MDE.

**4. 50.4% / 49.6% of 2 million.** Chi-square p ≈ 1e-29: an SRM. Do not
interpret the lift. Look for the cause: assignment vs logging mismatch, bot
filtering, redirects or crashes in one arm, a client version that fails to
log, users excluded post-assignment. Fix and rerun. Section 11.6 showed an A/A
test with a lazy-logging bug producing a +2% "lift" at p ≈ 1e-10 (✅).

**5. Daily peeking.** With 14 daily looks and stop-at-first-significance, the
A/A false-positive rate was 0.23 instead of 0.05 (✅ measured). Alternatives:
(a) a fixed horizon from a power calculation, looking only at guardrails for
safety; (b) a sequential design: group-sequential boundaries with alpha
spending (O'Brien–Fleming), or always-valid p-values / confidence sequences
(mSPRT), which allow continuous monitoring at a known error rate. Bonferroni
across looks is valid but very conservative (0.026 measured).

**6. 3 of 25 metrics up.** Under no effect at all you would expect about
1.25 significant metrics at α = 0.05, more with correlated metrics, so 3 is
weak evidence by itself. Decide with the **pre-registered primary metric**
and guardrails. Apply BH (or Bonferroni for a strict family) to the secondary metrics, check
direction consistency (do related metrics agree?), and treat surprises as
hypotheses for a follow-up test.
