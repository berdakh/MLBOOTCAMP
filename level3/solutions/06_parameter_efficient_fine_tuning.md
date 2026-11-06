# Solutions · 06 · Parameter-efficient fine-tuning

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## 06-1 · `LoRALinear`

✅ Reference code: [`code/nb06.py`](code/nb06.py) (`ex1`), graded in CI. The course
version with dropout and merging is `scaleml.finetune.LoRALinear`.

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # W and b frozen
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: identity at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

**Details that matter.** Compute `(x Aᵀ) Bᵀ`, never `x (BA)ᵀ`: the first costs
`r·(d_in + d_out)` per token, the second materialises a full `d_out × d_in` matrix.
If *both* A and B started at zero, the gradient of each would be zero (each gradient
is proportional to the other), and nothing would ever train. On a GPU, create the
parameters on `base.weight.device` (the course version does).

---

## 06-2 · `merge_lora`

✅ Reference code: [`code/nb06.py`](code/nb06.py) (`ex2`).

```python
def merge_lora(W, A, B, alpha, r):
    return W + (alpha / r) * (B @ A)
```

`B @ A` is `(out, r) @ (r, in) = (out, in)`, the shape of `W`. The merged layer
computes `W'x = Wx + (α/r)·B(Ax)`, the LoRA output, up to float rounding: in the
notebook the merged model's logits differed by 1.1e-06 from the unmerged one, the
same as `peft`'s `merge_and_unload`. Merge in fp32 (or the training dtype): merging
into fp16 or int8 weights adds rounding error, and un-merging (`W' − (α/r)BA`) is only
exact in the dtype you merged in.

---

## 06-3 · Fine-tuning a 7B LLM for 40 customers

✍️ A strong answer:

* **Memory arithmetic (6.1).** Full fine-tuning in fp32 + AdamW: 16 B/param ≈ 112 GB
  (104 GiB) plus activations: impossible on 16–24 GB. LoRA with a frozen fp16 base:
  ~14 GB of weights + a few hundred MB of adapters and their optimiser state +
  activations: tight on 16 GB, feasible on 24 GB with gradient checkpointing and short
  sequences. **QLoRA** (4-bit NF4 base, ~3.5–4 GB, bf16/fp16 compute, paged optimisers)
  fits comfortably on a T4-class GPU. Choose QLoRA on T4s, LoRA on 24 GB+ GPUs.
* **Modules and rank.** Start with all attention projections (q, k, v, o), and add the
  MLP projections if quality lags (QLoRA paper: adapting all linear layers matters more
  than the rank). Start at r = 8–16, α = 2r; tune r on a validation split per customer
  only if the data justifies it (2–20k examples is little; rank rarely the bottleneck).
* **Serving 40 variants.** One base model per GPU with multi-adapter serving (vLLM /
  S-LoRA style batched LoRA kernels), adapters loaded on demand; merge only for a
  customer large enough to justify dedicated replicas. Keeps one fleet, independent
  adapter rollbacks and A/B tests.
* **Did LoRA lose quality?** Hold out a test set per customer; compare LoRA vs full
  fine-tuning on 2–3 representative customers where full FT is affordable (e.g. on rented
  larger GPUs), with paired tests and several seeds; check slices (long inputs, rare
  intents) and calibration, not only the average. Also compare against the base model
  with good prompting, the cheapest baseline.

---

## Checkpoint · 06

1. **Parameter count.** A `d × d` layer has `d²` weights; LoRA adds `A (r×d)` and
   `B (d×r)`: `2rd` (r = 8, d = 768: 12,288 vs 589,824, ~2 %). `B = 0` makes the adapted
   model *identical* to the pretrained one at step 0, so training starts from the good
   solution, not from a random perturbation; `A` is random so that `∂L/∂B = (∂L/∂h)(Ax)ᵀ`
   is non-zero. `α/r` keeps the update magnitude roughly independent of `r`, so changing
   the rank does not require re-tuning the learning rate.
2. **Memory saved / not saved.** Saved: gradients and optimiser state of frozen weights
   (16 → 4 B/param for them), and the saved inputs of frozen linear layers (in 6.3 we
   measured activations 739 → 505 MB). Not saved: the frozen weights themselves and most
   activations (attention, GELU, LayerNorm, the backward pass through every layer).
   QLoRA cuts the frozen weights to 4 bits; gradient checkpointing cuts activations by
   recomputing them in backward.
3. **Which layers, which rank.** Attention q and v is the paper's minimal choice; all
   linear layers usually do better at a similar budget. Pick the rank by a sweep (4, 8,
   16, 64) on validation, watching accuracy vs trainable parameters and time; stop when
   it saturates. Also measure the learning rate jointly: LoRA's best lr is ~10× full FT's.
4. **Probe vs LoRA vs full FT.** A probe suffices when the frozen features already
   separate the classes (our CPU run: 0.833 vs 0.847 for full FT, within noise), when
   many tasks share one embedding service, or when labelled data is tiny. LoRA when the
   representation must adapt but memory, storage or many-task serving matter. Full FT
   when you have lots of data, a big domain shift, and the budget, or need the last point.
5. **Merged vs unmerged.** Merged: no extra latency (in 6.5 the difference was within
   noise on CPU: 790–890 vs 800–810 ms per batch), but one task per model copy. Unmerged: small overhead, any number of
   adapters per base, hot-swap and roll back independently. For 1,000 customers: a shared
   base fleet with batched multi-adapter kernels (S-LoRA/Punica/vLLM), adapters (~3 MB
   each here) cached in GPU memory by popularity and paged from CPU/disk otherwise.
6. **Worse on long inputs.** The fine-tuning data or `max_len` was shorter than production
   inputs (truncation at training time: the adapters never saw long contexts), the
   benchmark lacks long examples, or low rank cannot capture position-dependent
   behaviour. Investigate by slicing evaluation by input length, checking the training
   length distribution and truncation rate, and re-training with longer sequences (or
   adapting more modules / a higher rank) to see whether the gap closes.
