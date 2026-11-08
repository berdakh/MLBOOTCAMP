# Solutions · Chapter 27 · PyTorch training done right

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `train_one_epoch`

✅ Reference code: [`code/nb09.py`](solutions.py) (`ex1`), graded in CI.

```python
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total, count = 0.0, 0
    for *inputs, y in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(*inputs), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / max(count, 1)
```

**Why each line.** `model.train()` at the *start* of the epoch, because an
evaluation in between left the model in eval mode. `zero_grad` before
`backward` (anywhere after the previous `step` works). `loss.item()` turns the
loss into a float, so the epoch does not keep every step's autograd graph
alive. Multiplying by `len(y)` gives the **per-example** mean, which a smaller last
batch would otherwise distort.

**Common mistakes the grader catches.**

* `np.mean(batch_losses)`: over-weights a short last batch (the 32/32/16 test).
* `zero_grad` missing or after `step`: gradients accumulate, so the weights
  drift from the reference loop.
* Returning a tensor (`total` built from `loss` instead of `loss.item()`).

---

## Exercise 2 · `expected_initial_loss`

✅ Reference code: `ex2`.

```python
def expected_initial_loss(pos_rate):
    p = float(pos_rate)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))
```

The best constant prediction under log-loss is the base rate itself, and its loss
is the binary entropy. For ShopStream's 0.137 click rate that is 0.40 nats; for
a 1% ads CTR, 0.056. The notebook measured 0.395 with the output bias set to
`logit(p)` and 0.625 without it (✅). The edge cases use the convention
`0 · log 0 = 0`: a dataset with no positives has zero entropy.

**For softmax over C classes** the same reasoning gives `log C` for balanced
classes, or the entropy of the class prior in general.

---

## Exercise 3 · Find the bug

✅ Answer: **C, D, E**.

* **C** (line 16): `nn.CrossEntropyLoss` expects **logits**; it applies log-softmax
  itself. Feeding it probabilities squashes the "logits" into [0, 1], so the
  loss can never go below `log(1 + e^-1) ≈ 0.31` per example and the gradients are
  tiny. Symptom: a training loss stuck *above* the base-rate entropy.
* **D** (line 20): `model.eval()` at the end of epoch 0 is never undone, so
  epochs 1 and 2 train with dropout off.
* **E** (lines 3–5): a random permutation of seven weeks. The validation rows
  are not later than the training rows, so the score does not estimate
  performance on future traffic.

The distractors: **A** (`zero_grad` before the forward pass is correct), **B**
(`no_grad` in validation is correct), **F** (the seed is set on line 2, before the
model is built), **G** (`CrossEntropyLoss` takes `long` class indices, so `long` is
correct), **H** (3e-3 with Adam is fine for this model, as section 9.3 shows; the
stuck loss has another cause).

**Measured effect of each fix** (✅ our run, same seed, 3 epochs, train loss / valid AUC
after the last epoch; the time-split variant trains on days 28–41, validates on 42–48):

| Variant | train loss | valid AUC |
|---|---|---|
| as written (C, D, E present) | 0.4395 | 0.627 |
| fix C only (logits into the loss) | 0.3478 | 0.718 |
| fix D only (`model.train()` each epoch) | 0.4403 | 0.625 |
| fix E only (time split) | 0.4416 | 0.635 |
| fix C + D | 0.3533 | 0.720 |
| fix C + D + E | 0.3577 | 0.723 |

Read honestly: **C** does almost all the damage. **D** changes little *here*
(dropout 0.3 on a small MLP); with BatchNorm, or a model relying on dropout
for regularisation over many epochs, it matters more. **E** does not make the
score optimistic *on ShopStream*: users' tastes do not drift, and the
time-split AUC is even slightly higher. On real logs (trends, new items,
seasonality, users whose future clicks leak into training) a random split
typically overstates performance. The rule exists because you cannot know in
advance that your data is as stationary as a simulator's.

---

## Checkpoint answers

**1. One training step; where `zero_grad` goes.** Forward → loss → `backward()`
(accumulates into `.grad`) → `optimizer.step()` (reads `.grad`). `zero_grad`
must run between the previous `step()` and this `backward()`; the usual place
is the top of the loop. Not zeroing every step is deliberate in **gradient
accumulation**: sum the gradients of `n` micro-batches (each loss divided by `n`), then step,
to simulate a batch `n` times larger than fits in memory.

**2. `eval()` vs `no_grad()`.** `eval()` switches layer behaviour (dropout off,
BatchNorm uses running statistics); `no_grad()` stops autograd recording (less
memory, faster). Only `no_grad`: validation runs with dropout on, so metrics
are noisy and pessimistic, and BatchNorm updates its running stats on
validation data (a subtle leak). Only `eval`: correct numbers, but a full
graph is built and memory can blow up on large validation sets.

**3. 0.69 at step 0 with 2% positives.** Not a crash, but a bug in
initialisation: the correct starting loss is `H(0.02) ≈ 0.098`. The model
starts at 50% for everyone and spends its first steps learning the base rate,
which also distorts early loss curves and learning-rate warm-up. Fix: set the
final bias to `log(p / (1 - p)) ≈ -3.9`.

**4. Exact resume.** Model weights *and buffers*; optimizer state (Adam's two
moments per parameter, step counts); scheduler state; epoch/step counter; every
RNG state (Python, NumPy, torch CPU, CUDA, the sampler's generator, and
DataLoader workers' seeds, which derive from that generator); the AMP
`GradScaler` state when using mixed precision. Write atomically (temp file +
rename). The notebook showed a weights-only resume diverging from epoch 3 while the
full one matched bitwise (✅).

**5. Same seed, different GPU results.** Non-deterministic kernels (atomic adds
in scatter/index_add, some cuDNN convolution algorithms, cuBLAS workspace),
`cudnn.benchmark` picking different algorithms, a different number of
DataLoader workers or unseeded worker RNGs, a different GPU model or library
version. Fix: seed everything, a dedicated generator, `worker_init_fn`,
`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`cudnn.benchmark=False`. Cost: slower kernels (sometimes much slower). Not
worth it for large sweeps: there, report the mean and spread over seeds; keep
bitwise determinism for debugging and regression tests.

**6. First 15 minutes on "loss won't go down".** (1) Look at one batch: shapes,
dtypes, label range, a few raw examples. (2) Loss at step 0 vs the theoretical
value. (3) Overfit a single small batch with dropout off: must reach ~0.
(4) If not: check gradients per layer (all zero? NaN? a frozen or detached
part?), the loss input (logits vs probabilities), and that `optimizer` holds
the model's parameters. (5) If it overfits one batch: learning-rate sweep over
a few orders of magnitude, then data checks (shuffling, label noise, leakage),
then capacity.
