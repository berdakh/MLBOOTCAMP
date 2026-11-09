# Solutions · Chapter 49 · Quantisation & distillation

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `quantize_per_channel_int8`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI. The course's
version with a `bits` argument is `scaleml.serving.quantize_per_channel_int8`.

```python
def quantize_per_channel_int8(W):
    scale = W.abs().amax(dim=1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # all-zero rows
    q = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale
```

**Error analysis.** With `ŵ = q × scale`, rounding to the nearest integer gives
`|w − ŵ| ≤ scale / 2` for every element (the grader checks this bound). The
largest element of each row maps exactly to ±127, so nothing is clipped. In
section 49.3 of the lab, per-channel scales cut the relative weight error from 1.9e-2
(per-tensor) to 8.0e-3 on DistilBERT's first FFN matrix; int4 (`qmax = 7`)
raised it to 1.5e-1.

**Why −127 and not −128?** A symmetric range keeps zero exactly representable
and makes `q → −q` safe; using −128 would give one extra level on one side only.
Asymmetric (zero-point) quantisation is used for activations such as ReLU
outputs that are all positive.

**Common mistakes.** Truncating (`.to(torch.int8)` without `round`); one scale
for the whole matrix; dividing by zero on an all-zero row; computing the scale
over columns (`dim=0`) instead of output rows.

---

## Exercise 2 · `kd_loss`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

**Details.** `F.kl_div(input, target)` expects the *student's log-probabilities*
as `input` and the teacher's probabilities as `target`, and computes
KL(target ‖ input). `reduction="batchmean"` is the mathematically correct mean
over examples (`"mean"` would also divide by the number of classes). The
gradient of the soft term w.r.t. the logits scales as 1/T², so multiplying by
T² keeps its weight comparable across temperatures (Hinton et al., 2015).
With α = 1 and a student identical to the teacher the loss is 0.

---

## Exercise 3 · Choosing a model for a 20 ms p99 on CPUs

[open-ended] / [expected] Based on the table in section 49.5 of the lab (your numbers will differ):

**(a)** At batch 1 on one CPU thread, the fp32 teacher took ~45–55 ms (median) and
the int8 teacher ~15–20 ms; neither leaves room for a 20 ms *p99* (p99 is well above
the median, plus tokenisation and network). The TF-IDF student took ~1 ms. Batching
on a CPU reduced the per-example cost of the transformer only modestly in our
measurement (section 49.2, and it used two threads), and waiting to fill batches adds latency, so batching does not
rescue the teacher here; it would on a GPU. At 2,000 requests/s, even the int8
teacher would need ~40 single-threaded cores busy at all times, against roughly
2 cores for the student.

**(b) Before deploying the teacher on GPUs**, measure: cost per 1,000 requests
for each option (GPU instance price vs CPU fleet, at the real traffic shape);
whether 3 points of offline accuracy move the *business* metric (click-through,
complaint rate) in an A/B test; where the student's errors are (one class? rare
but important topics?) and whether a confidence threshold could route
low-confidence cases to the teacher (a cascade); latency including network hops
to a GPU service; and operational cost (a GPU fleet, model updates).

**(c) Improving the student, by expected payoff:** (1) much more transfer data
labelled by the teacher (cheap: unlabelled news is plentiful; section 49.4 showed the
transfer-set variants ahead); (2) a stronger student architecture that can
use soft targets well, e.g. a 2–4-layer transformer initialised from the
teacher's layers (as DistilBERT was), then quantised; (3) tune T and α and
train longer; then ensembling teachers for better soft targets.

---

## Checkpoint answers

1. **Symmetric int8.** [verified] `scale = max|w| / 127`, `q = round(w / scale)`,
   `ŵ = q × scale`, error ≤ scale/2. Per-channel uses one scale per output row,
   so rows with small weights are not forced onto a coarse grid set by the
   largest weight in the matrix (section 49.3: 2.4× lower error). Int4 has only 15
   levels; a single outlier in a row of thousands of weights makes the step
   huge for all of them, so scales per group of 64–128 weights (plus
   outlier-aware methods like AWQ/GPTQ) are needed, especially for LLMs.
2. **Kinds of quantisation.** *Dynamic*: weights int8 offline, activation
   scales computed per batch at run time; no calibration; CPU, Linear/LSTM-heavy
   models. *Static (post-training, calibrated)*: activation ranges measured on
   calibration data; fully int8 graphs, best CPU/edge speed (CNNs). *QAT*: fake
   quantisation during training so the model adapts; when PTQ loses too much
   accuracy (int4, small models). *Weight-only*: weights int8/int4, compute in
   fp16; memory-bound LLM decoding on GPUs, where reading fewer weight bytes is the win.
3. **Half, not a quarter.** [verified] Only `nn.Linear` weights became int8; the
   embedding matrix (23.4M of 67M parameters) stayed fp32, plus per-layer scales
   and biases (section 49.2: 268 → 139 MB). The speed-up comes from int8 matrix
   multiplies (more operations per instruction with VNNI) and moving fewer weight
   bytes. There may be none when the CPU lacks fast int8 instructions, when
   quantising/dequantising activations costs as much as it saves (small
   matrices), or when non-Linear ops dominate.
4. **Distillation loss.** `α T² KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1−α)
   CE(z_s, y)`. T > 1 softens both distributions so the small probabilities
   (which wrong classes are plausible) carry gradient; T² compensates for the
   1/T² shrinkage of the soft term's gradients so α keeps its meaning when T changes.
5. **Transfer set.** The teacher can label as much unlabelled in-domain text
   as you have, so the student learns from far more examples than were ever
   annotated: data, not just soft targets, is usually the main benefit. In section 49.4 the
   transfer-set variants beat the labelled-only ones by 1–1.5 points
   (0.849/0.843 vs 0.835), while soft targets on the labelled set alone did not
   help this bag-of-words student (0.827).
6. **4× cheaper with ≤ 1 point loss.** Measure first: latency/throughput at
   production batch and sequence sizes, cost per request, accuracy on a fixed
   evaluation set with confidence intervals. Then in order: serving
   improvements (batching, ONNX Runtime/TensorRT, fp16 on GPU), int8
   quantisation (nearly free, 2–3× on CPU in section 49.2), shorter max sequence length
   if the data allows, distillation into a smaller student with a large
   transfer set (and quantise the student), then a cascade (cheap model first,
   expensive model on low-confidence cases). Validate the final choice with an
   online A/B test on the business metric.
