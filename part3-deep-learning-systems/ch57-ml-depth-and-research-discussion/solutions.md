# Solutions · Chapter 57 · ML depth & research discussion

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `warmup_cosine_lr`

[verified] Reference code: [`code/nb19.py`](solutions.py) (`ex1`), graded in CI.

```python
def warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    if step < 0:
        raise ValueError("step must be >= 0")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps cannot exceed total_steps")
    if step < warmup_steps:
        return float(base_lr * (step + 1) / warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return float(min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress)))
```

**Details.** `(step + 1)` so the first step does not use LR 0 and the last
warmup step reaches `base_lr` (this matches `scaleml.train.linear_warmup_cosine`,
which `LambdaLR` evaluates at step 0, 1, …). `max(1, …)` avoids a division by
zero when there is no decay phase. The clip keeps the LR at `min_lr` if
training runs past `total_steps` (e.g. a resumed job with a miscounted
schedule), instead of the cosine rising again. A non-zero `min_lr` (often
~10% of the peak) is common for long pre-training runs; zero is common for
fine-tuning.

---

## Exercise 2 · `critique_flags`

[verified] Reference code: [`code/nb19.py`](solutions.py) (`ex2`): `{"A", "B", "C", "D", "E"}`.

* **A (real).** A random *trial* split puts every subject (and every session)
  in both training and test, so the model can learn subject-specific
  signatures. The claim is about *new users*; only a leave-subjects-out split
  tests it. In cross-subject EEG this is usually the largest inflation.
* **B (real).** Choosing 1 of 60 configurations by test accuracy makes the
  test set a validation set: the maximum of 60 noisy estimates is biased
  upward.
* **C (real).** Baselines at toolbox defaults and 10 epochs against a model
  tuned over 60 configurations: the gap partly measures tuning effort.
* **D (real).** "Best of five seeds" reports a maximum, not an expected
  result, with no spread and no interval over subjects (the independent unit).
* **E (real).** z-scoring with statistics of the full dataset leaks test
  information into training. With per-channel means and SDs over 12,000
  trials the effect is probably small, but it is a leak, and the same habit
  with ICA or feature selection can be large.
* **F (not a flaw).** With four *balanced* classes accuracy is a reasonable
  headline (chance = 25%); per-class recall and a confusion matrix would be
  nice additions, not a correction.
* **G (not a flaw).** Transformers apply to any sequence of vectors
  (time–frequency patches here).
* **H (not a flaw in itself).** 20 subjects is small and makes intervals
  wide, which is exactly why D matters; it does not forbid reporting results.
* **I (not a flaw).** Comparing models of different sizes is fine if tuned
  and evaluated fairly; parameter count is a cost to report, not a reason
  the comparison is invalid.

---

## Exercise 3 · The full critique

[open-ended] Open-ended. A strong answer, in template order:

1. **Claim:** calibration-free decoding for *unseen users*.
2. **Split:** random trials → the claim is untested (A). Expected effect:
   large and upward; cross-subject accuracy in motor imagery is commonly far
   below within-subject accuracy.
3. **Selection:** test-set tuning (B), best of seeds (D): upward.
4. **Baselines:** under-tuned (C): the reported gap is inflated.
5. **Uncertainty:** none; the unit should be subjects (20), so intervals
   will be wide.
6. **Preprocessing:** global z-scoring (E): small upward effect, wrong habit.
7. **Metric:** accuracy is acceptable for balanced classes; report Cohen's
   kappa or per-class recall too, and information transfer rate if the
   claim is about usable BCIs.
8. **Ablations:** which part matters (time–frequency patches? the
   transformer? the band selection?).
9. **Practicality:** latency per decision, electrodes needed, robustness to
   new sessions and hardware.
10. **Decisive experiment:** leave-one-subject-out (or k-fold by subject),
    hyperparameters chosen by nested cross-validation *within training
    subjects*, baselines tuned with the same budget, five seeds reported as
    mean ± spread, and a per-subject paired comparison with a bootstrap
    interval over subjects. Belief-changing result: the proposed model beats
    the tuned baselines on held-out subjects with an interval that excludes
    zero, and its accuracy on new users is high enough to be usable without
    calibration.

First question to the authors: *"What is the accuracy with a
leave-subjects-out split, with tuning inside the training subjects?"*

---

## Exercise 4 · Research-to-production story

[open-ended] Personal. A shape to aim for (fill with your own measured numbers; do
not use these placeholders as results):

> "In my PhD I built a cross-subject motor-imagery decoder: [N] subjects,
> leave-one-subject-out, [metric] of [your number ± spread] against
> [baseline number] for [baseline]. In a product, three things would break:
> new users differ from training users, the same user drifts across sessions,
> and headsets differ. I would ship a short calibration block with per-session
> re-centring of the covariance features and a small adapted head, a signal-
> quality gate that refuses to decode on high impedance or heavy artefacts,
> and a streaming loop with a [X] ms budget. I'd know it works from an online
> study with *new users*, pre-registered, with intervals over users, and I'd
> monitor input drift per session. Looking back, I would have [what you would
> change]."

Graders listen for: the split named next to the number, the specific
production failure modes (not generic "data drift"), a fallback, and an
evaluation unit that matches the claim.

---

## Checkpoint answers

1. **Warmup.** [verified] In our CPU run (TinyGPT, 300 steps, Adam, no clipping):
   at LR 3e-3 warmup made no difference (val loss 2.27 vs 2.31); at 1e-2 it
   was 2.53 without vs 2.24 with; at 3e-2, 2.75 vs 2.26. Mechanism: Adam's
   second-moment estimate is built from very few gradients at the start, so
   per-parameter steps are large and noisy just when the network (and,
   for post-LN transformers, the gradient scale near the output) is least
   stable; ramping the LR keeps early updates small until statistics settle.
   Warmup widens the usable peak-LR range.
2. **Batch 64 → 2,048.** [expected] Scale the LR (linear for SGD as a starting
   point, often √k for Adam), add or lengthen warmup, keep the total number
   of *examples* (not steps) and re-tune the schedule length; check
   per-replica batch statistics (BatchNorm) and use gradient clipping. You
   know you are past the critical batch size when doubling the batch no
   longer roughly halves the steps to reach a target loss, or when stability
   becomes seed-dependent. In our toy run, linear scaling to batch 2,048
   gave 0.55–0.75 accuracy across seeds against 0.92 for √k.
3. **100% train, 81% test, 20% noisy labels.** [verified] (our run: 1.000 train,
   0.813 final test, 0.928 early-stopped) The network memorised the
   flipped labels after first fitting the clean structure. First: early
   stopping on validation (even a noisy one ranks checkpoints), stronger
   regularisation; then measure the noise (relabel a sample), and if
   confirmed use noise-robust losses, label cleaning (confident learning,
   loss-based filtering) or better labelling.
4. **Double descent.** [verified] The peak is at the interpolation threshold
   (parameters ≈ training points: in our run, p = n = 100 gave a test MSE of
   ~194 against ~0.5 at p = 50). There, exactly one interpolating solution
   exists and it is badly conditioned, so it fits the noise with huge
   weights. With many more parameters there are infinitely many
   interpolating solutions and the minimum-norm one (which gradient descent
   from small init tends to find) is smooth: p = 3,000 gave ~0.30. Ridge
   removes the spike.
5. **Attention complexity.** [verified] (arithmetic) Compute O(T²·d) per layer,
   score memory O(T²) per head: 12 layers × 12 heads at fp16 is 1.2 GB per
   sequence at T = 2,048 and 19.3 GB at 8,192 if materialised.
   FlashAttention-style kernels compute the same result in tiles without
   storing the T × T matrix (memory becomes O(T), compute still O(T²), with
   far less memory traffic). A KV cache avoids recomputing keys and values
   for past tokens during generation (each new token costs O(T·d) instead of
   re-running the prefix) at the price of memory linear in T per layer.
6. **SpectraFormer in 90 seconds.** [expected] In order of likely inflation:
   (A) random trial split vs a new-user claim; (B) tuning on test and (D)
   best-of-five seeds; (C) under-tuned baselines; (E) global normalisation.
   Then the decisive experiment from 19-3.
