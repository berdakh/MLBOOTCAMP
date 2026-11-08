# Solutions · Chapter 36 · Model optimisation for inference

Attempt each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `parity`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def parity(a, b, atol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    over = np.isnan(diff) | (diff > atol)
    max_abs = float("inf") if np.isnan(diff).any() else float(diff.max(initial=0.0))
    return {"max_abs_diff": max_abs, "n_over": int(over.sum()), "ok": bool(not over.any())}
```

**Why the strictness.** `(n,)` vs `(n, 1)` broadcasts to an `(n, n)`
difference matrix: an export that changed the output shape would "pass" with
a meaningless number. `NaN > atol` is `False`, so a naive check silently
passes NaN outputs. `diff.max(initial=0.0)` handles empty arrays.

**Choosing atol.** For a lossless float32 export, 1e-5 on logits is typical
(Chapter 36 measured 1.2e-6). Compare logits *and* probabilities: a
sigmoid hides logit differences at the extremes. For lossy steps (int8,
fp16, distillation), use rank-based checks instead (36.3).

---

## Exercise 2 · `distillation_loss`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`); the module's
`opt.distillation_loss` is the same.

```python
p = teacher_probs.clamp(1e-6, 1 - 1e-6)
soft_target = torch.sigmoid(torch.logit(p) / T)
soft = F.binary_cross_entropy_with_logits(student_logits / T, soft_target) * T * T
hard = F.binary_cross_entropy_with_logits(student_logits, labels.float())
return alpha * soft + (1 - alpha) * hard
```

**Details that matter.** Temper *both* sides (student logits and teacher
logits) by `T`. Use the `_with_logits` form: it is numerically stable
(log-sum-exp inside), unlike `BCE(sigmoid(z), y)`. The `T²` factor: the
gradient of the soft term w.r.t. the logits scales like `1/T²`, so
multiplying by `T²` keeps the soft and hard terms comparable when tuning
`T`. `alpha = 0` is plain click training; `alpha = 1, T = 1` is regression
on the teacher's probabilities.

**Observed.** [expected] Distilled 16-unit student: cascade NDCG@10 0.858 vs 0.843
on clicks only (teacher 0.875), +0.015 to +0.021 over seeds 0–2.

---

## Exercise 3 · Optimisation plan for a 45 ms transformer ranker (written)

[open-ended] A strong answer is ordered by expected gain per unit of effort and risk,
with a measurement gate at every step:

1. **Profile first.** Where do the 45 ms go: per-candidate cost (300 × the
   candidate part) or per-request cost (encoding the 50-action history)? If
   the user history encoder is shared, compute it **once per request** (or
   cache it per session) and only run a light cross part per candidate —
   often the biggest win, and lossless. Gate: exact parity (atol 1e-5).
2. **Runtime + graph optimisation.** Export to ONNX Runtime (or TorchScript /
   `torch.compile`), fuse attention, pre-pack weights, tune intra-op threads;
   batch all 500 candidates in one call. Lossless. Gate: parity atol 1e-5 on
   logits over 100k logged requests; latency p99 at the target batch.
3. **Precision.** Dynamic int8 for the linear layers (transformers are
   weight-heavy, unlike Chapter 36's tiny MLP, so int8 should actually
   help); fp16/bf16 if served on GPU. Gate: Spearman ≥ 0.99 with fp32
   scores, top-10 overlap ≥ 0.95, offline NDCG drop ≤ 0.002 with a paired
   bootstrap interval that includes 0 or a pre-agreed tolerance.
4. **Distillation / smaller architecture.** Fewer layers or a shorter
   history (50 → 20 actions), trained with the current model as teacher.
   Gate: same quality thresholds, then an online A/B (or interleaving) before
   full rollout, because offline NDCG and engagement can disagree.
5. **Cascade change.** A pre-ranker that cuts 500 → 300 for the heavy model.

Stop when the 500-candidate p99 fits (with slack), when the next step fails
its quality gate, or when the remaining latency is elsewhere (features,
network). Shadow-traffic the optimised model to compare scores on live
inputs before switching.

---

## Checkpoint answers

1. **ONNX Runtime vs PyTorch.** ORT runs a pre-planned, fused graph in C++
   with no Python dispatch per op; PyTorch eager pays per-op overhead
   (~µs each). It matters most for small models and small batches, where
   overhead dominates: 36.5 measured ~0.006 ms (ORT) vs ~0.025 ms (PyTorch) at
   batch 1, but only 0.18 vs 0.24 ms at batch 1,024.
2. **Dynamic vs static quantisation.** Dynamic: weights stored int8 ahead
   of time, activations quantised per batch at run time (no calibration).
   Static: activation ranges calibrated on representative data, so the whole
   path (incl. activation tensors) stays int8 with fused quantise ops —
   faster, needs calibration data and careful accuracy checks. Here int8 did
   not help because the model is tiny (10 inputs, ~10 K weights): matmuls are
   not the cost; quantise/dequantise overhead is. PyTorch int8 was slower
   (0.42 vs 0.24 ms at 1,024).
3. **3e-6 difference:** deploy (float32 rounding in a lossless export; set
   atol ~1e-5 and also check probabilities and shapes). **0.5 logit
   difference for int8:** don't judge by parity; judge by ranking agreement
   and the offline metric against thresholds agreed beforehand (Spearman
   0.997, NDCG unchanged in 36.3 → acceptable, then shadow and A/B).
4. **Distillation.** The student is trained on the teacher's probabilities
   (plus labels). Soft labels carry graded relevance ("0.3 vs 0.05") and the
   teacher's generalisation, smoothing click noise and position bias effects.
   Temperature softens both distributions so low-probability orderings also
   carry gradient. Expect most gain when the student is much smaller, labels
   are scarce/noisy, or the teacher sees extra information (more features,
   cross-attention, future signals) unavailable at serving time.
5. **Cutting a 45 % ranker.** (a) Serving/runtime work: batching, ONNX/
   compilation, threads, caching the user tower (low effort, low risk);
   (b) fewer candidates via a pre-ranker (medium effort; risk of recall
   loss); (c) quantisation/fp16 (medium; accuracy risk, operator support);
   (d) distillation or a smaller architecture (high effort; quality risk and
   a second model to maintain). Also: drop expensive, low-importance features.
6. **Safe rollout.** Offline: parity/ranking-agreement gates and NDCG on the
   frozen validation set with paired bootstrap intervals; latency benchmarks
   at production batch sizes. Shadow traffic: score live requests with both
   models, compare score distributions, latency, error rates. Canary A/B:
   1–5 % of users with guardrails (p99 latency, error/fallback rate, CTR and
   conversion, calibration). Roll back on guardrail breaches (latency,
   errors) immediately and on a significant drop in the primary metric.
