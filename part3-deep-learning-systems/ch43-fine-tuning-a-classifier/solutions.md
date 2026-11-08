# Solutions · Chapter 43 · Fine-tuning a classifier

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `expected_calibration_error`

[verified] Reference code: [`code/nb05.py`](solutions.py) (`ex1`), graded in CI.

```python
def expected_calibration_error(probs, labels, n_bins=15):
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)  # (lo, hi]
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)
```

**Pitfalls.** `int(conf * n_bins)` puts a confidence of exactly 1.0 into a
non-existent bin `n_bins` (index error or silently dropped); the `ceil(...) - 1`
plus `clip` gives `(lo, hi]` bins with 1.0 in the last. Weight by bin *size*
(`in_bin.mean()`), not equally per bin. ECE is a *top-label* measure: it ignores how
the remaining probability mass is spread (class-wise ECE or the Brier score do not).
It is also biased by binning: with few examples per bin, even a perfectly calibrated
model shows a positive ECE.

---

## Exercise 2 · `fit_temperature`

[verified] Reference code: [`code/nb05.py`](solutions.py) (`ex2`).

```python
def fit_temperature(logits, labels):
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)   # T = exp(log_t) > 0
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())
```

**Why log T.** The NLL as a function of `T` is smooth and has one minimum, but `T`
must stay positive; optimising `log T` removes the constraint. A grid search over
`exp(linspace(log 0.05, log 20))` is a perfectly good alternative for one parameter.
**Fit on validation, never on test**: in the notebook (CPU run) `T = 0.59` fitted on 225
validation examples cut test ECE from 0.124 to 0.034 and test NLL from 0.487 to 0.427,
with accuracy unchanged at 0.872. [expected] A longer GPU fine-tune usually gives `T > 1`
(over-confidence), the direction reported by Guo et al. (2017).

---

## Exercise 3 · When a TF-IDF baseline is the right production choice

[open-ended] A strong answer weighs evidence, not fashion:

* **Accuracy gap and its uncertainty.** In our CPU run TF-IDF + LR scored 0.822 vs
  0.872 fine-tuned on 500 test examples (SE ≈ ±0.015 each): a real gap here, but on a
  larger training set the gap on AG News shrinks to a few points ([expected] ~0.90 vs ~0.94).
  Ask what each point is *worth* to the product (fewer misrouted articles? revenue?).
* **Cost and latency.** TF-IDF + LR fits in ~1 s and predicts in microseconds on a CPU;
  DistilBERT ran at ~36 texts/s on two CPU threads. At 50M articles/day (~580/s on
  average, more at peak) the transformer needs a GPU fleet (or heavy distillation and
  quantisation, notebook 11); the linear model needs a few CPU cores.
* **Data volume and drift.** With little labelled data, both baselines are competitive;
  with abundant data and stable labels, fine-tuning pays more. Under fast vocabulary
  drift (new names, events), cheap frequent retraining of a linear model can beat a
  stale transformer.
* **Interpretability and debugging.** LR weights per n-gram can be read, audited and
  hot-fixed; a transformer needs attribution tools and more careful error analysis.
* **Retraining cadence and ops.** Daily retrains in minutes on a CPU vs GPU jobs,
  model registries, larger artifacts (255 MB vs a few MB).
* **Calibration and thresholds.** Either model needs calibration before using its
  probabilities (TF-IDF + LR was badly under-confident here, ECE 0.21).

**Decision rule.** Ship TF-IDF when the measured gap is within the business tolerance
(or within noise), latency/cost budgets are tight, or explainability is required; ship
(or cascade to) the transformer when the gap is significant and valuable. A common
hybrid: the cheap model handles confident cases, the expensive one only the uncertain
ones (a cascade), which you can size from the reliability table. Evidence to collect:
a larger test set with a paired significance test, per-class errors, cost per 1k
predictions on the target hardware, and an online A/B test.

---

## Checkpoint answers

1. **Linear probe vs full fine-tuning vs TF-IDF.** TF-IDF + LR measures how much is
   solvable from surface vocabulary; the probe measures how much the frozen pretrained
   representation already encodes; full fine-tuning adapts the representation to the
   task. Start with TF-IDF (minutes), then a probe (one embedding pass), and fine-tune
   only if the gap justifies the cost. In our CPU run: 0.822 / 0.858 / 0.872.
2. **Small lr + warmup.** Pretrained weights sit in a good region; large steps (especially
   early, while the random head sends large, noisy gradients back) overwrite general
   knowledge: *catastrophic forgetting*. Warmup ramps the lr from ~0 so the head can
   settle first; typical fine-tuning lrs are 1e-5–1e-4, 10–100× below from-scratch.
3. **Early stopping/model selection data** must come from a validation split carved
   from the training data (or cross-validation). Using test makes the reported number
   an optimistically biased maximum over many choices; you no longer have an unbiased
   estimate of deployment performance.
4. **ECE** = size-weighted mean |accuracy − confidence| over confidence bins. Weaknesses:
   depends on the number and type of bins, is biased upward with few samples, looks
   only at the top label, and can be gamed (predicting the base rate is perfectly
   calibrated and useless). Temperature scaling divides all logits by the same `T > 0`,
   a monotonic transformation that never changes the argmax, so accuracy is unchanged.
5. **1.5 points on 500 examples** is about one standard error (±1.5 points at 87 %): not
   significant on its own. Use a *paired* test on the same examples (McNemar's test on
   the discordant pairs, or a paired bootstrap of the accuracy difference), report a
   confidence interval, and if it matters, label more test data. Also check variance
   across fine-tuning seeds, which is often ±0.5–1 point.
6. **Business ↔ Sci/Tech.** (a) Read 50 such errors and decide whether they are label
   ambiguity or model errors (our most confident mistakes were label noise);
   (b) fix the task definition: labelling guidelines, multi-label targets, or a merged
   class if the product does not need the distinction; (c) only then add targeted data
   for the confusable pair or route low-confidence predictions to review.
