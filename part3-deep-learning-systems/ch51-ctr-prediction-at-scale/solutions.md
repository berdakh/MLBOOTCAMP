# Solutions · Chapter 51 · CTR prediction at scale

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `normalized_entropy`

[verified] Reference code: [`code/nb13.py`](solutions.py) (`ex1`), graded in CI.

```python
def normalized_entropy(y, p, eps=1e-7):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    log_loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    b = np.clip(y.mean(), eps, 1 - eps)
    return float(log_loss / -(b * np.log(b) + (1 - b) * np.log(1 - b)))
```

**Notes.** The denominator is the log loss of the best *constant* predictor (the
base rate itself), so NE = 1 means "no information beyond the average CTR".
Use the base rate of the evaluation labels (that is the definition in He et al.
2014); some teams use the training base rate, which makes NE drift with the
test day's CTR. Report NE differences in relative terms ("−0.5% NE").

---

## Exercise 2 · `recalibrate_downsampled`

[verified] Reference code: [`code/nb13.py`](solutions.py) (`ex2`).

```python
def recalibrate_downsampled(p, w):
    p = np.asarray(p, dtype=float)
    return p / (p + (1 - p) / w)
```

**Derivation.** Keeping every positive and a random fraction `w` of negatives
multiplies the number of negatives (for any `x`) by `w`, so the odds the model
learns are `o' = p/(1−p) · 1/w`. Invert: `o = w·o'`, `q = o/(1+o) = w p'/(1 − p' + w p')
= p'/(p' + (1 − p')/w)`. In 13.7 this brought pred/obs from ~6.4 down to ~0.91 and NE from 1.85 to
0.86 with unchanged AUC; the remaining gap was the model's own under-confidence
(pred/obs ≈ 0.97 on a downsampled test set), removed by a Platt layer fitted on
non-downsampled validation data (pred/obs ≈ 1.01). It is exact only
if negatives were sampled independently of `x`; if you also downsample by
segment (e.g. per country), apply each segment's own `w`.

---

## Exercise 3 · `hash_feature`

[verified] Reference code: [`code/nb13.py`](solutions.py) (`ex3`).

```python
import hashlib

def hash_feature(field, value, n_buckets):
    digest = hashlib.md5(f"{field}={value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % n_buckets
```

**Notes.** `hash()` on strings is salted per interpreter (`PYTHONHASHSEED`), so
the trainer and each serving replica would disagree. MD5 is fine here (we
need spread, not cryptographic security), but it is slow; production uses
MurmurHash3/xxHash/CityHash in C++ or vectorised code (like
`ctr.hash_columns`, a 64-bit SplitMix). `mod n_buckets` with a 64-bit hash has
negligible bias. Putting the field in the key makes the hash *field-aware*; you can
also give each field its own table (no cross-field collisions, but more tables
to manage).

---

## Exercise 4 · Calibration shift after changing the downsampling rate

[open-ended] Model answer:

(a) The new model was trained with `w = 0.05`, so its raw predictions are inflated
by odds factor 20 instead of 10; if the serving path still re-calibrates with
production's `w = 0.1` (or a calibration layer fitted for the old model), predictions
come out too high. Check: recompute pred/obs after applying the correct
`recalibrate_downsampled(p, 0.05)`; plot calibration per bucket and per segment
(country, ad type, new vs old ads); confirm the `w` stored in the model's metadata.
Other suspects: a different training window (CTR seasonality), label/feature
pipeline changes, a feature with train/serve skew.

(b) In a second-price auction, inflated pCTR raises eCPM for every ad the model
scores; if it is uniform, winners and per-click prices barely change, but comparisons
with anything not scored by this model (reserve prices, other ad types, organic
content in a blended feed) shift in its favour: more ads shown, lower quality,
and advertisers paying for clicks that do not happen if pacing/bidding systems
use pCTR. If the inflation is uneven (segments), allocation changes and
advertisers in over-predicted segments overpay: their ROI drops, then their bids,
then revenue. Short-term revenue may even rise, which is why offline calibration
is a launch gate.

(c) Fix the re-calibration (correct `w`, then a post-hoc calibration layer such as
isotonic or Platt per segment trained on recent data), re-check pred/obs ≈ 1
overall and per segment, then A/B test. Watch: revenue (RPM), CTR, advertiser
value (conversions per spend, cost per click, ROI), online calibration
(predicted vs observed clicks per segment), ad load and user engagement
guardrails, auction metrics (clearing price, reserve-hit rate). Ramp slowly;
advertisers react to price changes over days.

---

## Checkpoint answers

1. [verified] NE = log loss / entropy of the base rate (= log loss of predicting the average).
   1 = no better than the average; lower is better; normalisation makes it comparable
   across days and products with different CTRs. Ads care about the *values* of the
   probabilities (auctions multiply pCTR by bids, pacing and billing use them), so a
   metric that rewards calibration (log loss/NE) plus pred/obs matters more than
   AUC, which only measures ranking and is unchanged by any monotone distortion. In
   13.7 the raw downsampled model had a near-normal AUC and an NE far above 1.

2. [verified] It removes the vocabulary: fixed memory, no id-assignment service, unseen
   values handled, training/serving agree by construction. Costs: collisions (two
   unrelated values share a weight), no way to inspect/delete a feature, and
   memory still grows with the buckets you allocate. Pick the bucket count by
   measuring (13.3: gains flattened past ~65k–500k buckets for ~426k distinct
   keys, i.e. once the frequent keys rarely collide); keys seen too rarely to learn do
   not need their own row. The hash must be stable across processes, languages and
   time, or serving looks up the wrong weights.

3. [expected] 10⁹ × 64 × 4 B = 256 GB of fp32 weights; Adam adds two slots (512 GB) → 768 GB.
   No single GPU holds that: shard the tables (row-wise for the huge ones, table-wise
   for the many small ones) across GPUs/hosts (model parallel), run the dense layers
   data-parallel, and exchange ids/embeddings with all-to-all (DLRM/TorchRec). Reduce
   it with row-wise Adagrad (one scalar per row), fp16/int8 storage, frequency-based
   admission and eviction, and a GPU cache of hot rows backed by host memory/SSD.

4. [open-ended] `x_{l+1} = x0 ⊙ (W x_l + b) + x_l`: each layer multiplies the current
   representation, element-wise, by a linear function of the input, raising the
   polynomial degree by one while the residual keeps the lower-order terms; `L`
   layers express all crosses up to degree `L+1`, with weights shared through `W`.
   LR is a sum of per-feature weights, so a cross like "this ad on this site" needs a
   hand-made feature. FMs model pairwise crosses as dot products of embeddings
   (`⟨v_i, v_j⟩`); DeepFM adds an MLP. DCN-v2 generalises the FM interaction (a
   full-rank or low-rank `W` instead of a fixed dot product) and stacks it to higher
   orders. 13.5: DCN improved NE from 0.878 (LR) to 0.854 on the same hashed features.

5. [verified] See 13-2 above. Forgetting it: predictions are inflated by an odds factor 1/w
   (pred/obs ≈ 6.4 and NE ≈ 1.85 in 13.7), so auctions over-value every ad, pacing/budgets
   mis-forecast, and NE is terrible even though AUC looks fine.

6. [verified] Conversions arrive days after the click; at training time recent clicks whose
   conversion has not happened yet are labelled negative, so the model learns a
   CVR that is too low (13.8: ~14% of conversions missing at the cutoff; naive
   pred/obs ≈ 0.88 against ≈ 1.04 for a model trained on the eventual labels). Fixes: (i) a wait window (train only on clicks older than
   W days): simple, but the model is W days stale and loses the freshest
   data, bad for new ads and fast-changing traffic; residual bias from delays > W.
   (ii) Model the delay: `P(observed) = pCVR · F(elapsed)` with F estimated from
   matured data (or learned jointly, Chapelle's DFM), keeps all data, needs a good
   delay model. (iii) Fake-negative weighting / re-ingesting positives when they
   arrive (streaming). Pick the attribution window with the business.
