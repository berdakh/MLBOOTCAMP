# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/06_parameter_efficient_fine_tuning.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 06 · Parameter-efficient fine-tuning
#
# **Day 3 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Full fine-tuning updates every weight, so every task needs its own copy of the model and
# training needs memory for gradients and optimiser state of *all* parameters. LoRA trains
# a tiny low-rank correction instead: well under 1 % of the parameters, adapters of a few
# megabytes, and one frozen base model that can serve many tasks. This notebook builds LoRA
# from scratch, measures what it saves (and what it does not), then checks the result
# against the `peft` library.
#
# By the end you will be able to:
#
# 1. derive LoRA (frozen $W$ plus $\frac{\alpha}{r} BA$, $B$ initialised to zero) and implement it as a module;
# 2. inject adapters into DistilBERT's attention projections and count trainable parameters;
# 3. compare a linear probe, LoRA (r = 4, 8) and full fine-tuning on one budget: accuracy, trainable parameters, memory, time per epoch;
# 4. reproduce the setup with `peft` (`LoraConfig`, `get_peft_model`) and verify it matches yours exactly;
# 5. merge adapters into the weights for zero-overhead serving, and verify identical outputs;
# 6. explain how one base model serves many adapters.
#
# **You will produce:** `artifacts/level3/nb06/peft_comparison.json` (the comparison table on
# your device) and `artifacts/level3/nb06/lora_r8_adapter.safetensors` (a trained adapter).
#
# **Data:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (licence on the card:
# *unknown*; check the card before reuse). **Model:**
# [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
# (Apache-2.0), pinned to a revision.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only and use an even smaller
# budget than notebook 05 (four training runs must fit in a few minutes), so accuracies are
# low and noisy. Numbers in **Expected:** lines were printed by a CPU run unless marked
# *"on a T4 you should see … (not measured in our verification runs)"*.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level3/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# Level 3 is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: data, tokenizer and model (downloaded once, then cached) ------
import copy
import json
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scaleml import finetune as ft
from scaleml import profiling as prof
from scaleml.device import count_parameters, seed_everything
from scaleml.grading import nb06 as grade
from scaleml.textdata import load_text_classification

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb06"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)
IS_CUDA = DEVICE.type == "cuda"

# Four training runs share this budget: on CPU it is smaller than notebook 05's.
N_TRAIN = BUDGET.train_examples // 2 if IS_CUDA else 640
N_EVAL = BUDGET.eval_examples if IS_CUDA else 300
MAX_LEN = BUDGET.max_seq_len
train_df, test_df, LABELS = load_text_classification("ag_news", n_train=N_TRAIN, n_eval=N_EVAL, seed=1)
tok = AutoTokenizer.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV)
enc_train, enc_test = ft.encode(tok, train_df["text"], MAX_LEN), ft.encode(tok, test_df["text"], MAX_LEN)
y_train, y_test = train_df["label"].to_numpy(), test_df["label"].to_numpy()
print("dataset card: https://huggingface.co/datasets/fancyzhx/ag_news (licence on the card: unknown; check it)")
print("model card:   https://huggingface.co/" + ft.DISTILBERT_ID)
print(f"train {len(train_df)} | test {len(test_df)} | max_len {MAX_LEN} | labels {LABELS}")


def new_classifier():
    """A fresh DistilBERT + randomly initialised head, identical every call (seeded)."""
    torch.manual_seed(0)
    return AutoModelForSequenceClassification.from_pretrained(
        ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV, num_labels=len(LABELS), use_safetensors=True)


# %% [markdown]
# **Expected:** `train 640 | test 300 | max_len 64` on the CPU budget (10,000 / 4,000 / 128
# with the `gpu` budget).
#
# ## 6.1 · Where fine-tuning memory goes
#
# Training memory = weights + gradients + optimiser state + activations (notebook 02).
# With AdamW in fp32 that is 4 + 4 + 8 = **16 bytes per trained parameter**, plus 4 bytes per
# *frozen* parameter (it must still be in memory for the forward pass). Freezing 99 % of the
# model removes almost all gradient and optimiser memory. It does **not** remove
# activations: the backward pass still has to flow through every layer to reach the
# adapters in the lower layers.

# %%
n_total = count_parameters(new_classifier())
print(f"DistilBERT classifier: {n_total / 1e6:.1f}M parameters")
for label, n_trained in [("full fine-tuning", n_total), ("LoRA, ~1% trained", n_total // 100)]:
    full = prof.memory_breakdown(n_trained, "adamw")
    frozen = 4 * (n_total - n_trained)
    print(f"{label:18}: weights {prof.human_bytes(full['weights'] + frozen):>9} | gradients "
          f"{prof.human_bytes(full['gradients']):>9} | AdamW state {prof.human_bytes(full['optimizer']):>9} | "
          f"total {prof.human_bytes(full['total'] + frozen):>9} (+ activations)")
n7b = 7e9
print(f"\na 7B model in fp32 + AdamW, full fine-tuning: {prof.human_bytes(16 * n7b)}; "
      f"LoRA with frozen fp16 weights: {prof.human_bytes(2 * n7b)} + adapters (+ activations)")

# %% [markdown]
# **Expected:** 67.0M parameters; full fine-tuning needs 1021.7 MB of state before
# activations, the LoRA-sized run 263.1 MB (almost all of it the frozen fp32 weights). For a
# 7B model the difference decides whether training fits on one GPU at all: 104 GB vs 13 GB of
# weights (QLoRA goes further and stores the frozen base in 4 bits, ~3.5 GB).
#
# ## 6.2 · LoRA from scratch
#
# Hu et al. (2021) observed that the weight *change* during fine-tuning has low intrinsic
# rank. LoRA therefore freezes the pretrained $W \in \mathbb{R}^{d_{out} \times d_{in}}$ and
# learns
#
# $$h = W x + b + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times d_{in}},\; B \in \mathbb{R}^{d_{out} \times r},\; r \ll d.$$
#
# * $B$ starts at **zero**, so at initialisation the model computes exactly what the
#   pretrained model did; $A$ starts random (Kaiming-uniform, as in `peft`), so gradients
#   reach $B$ immediately.
# * $\frac{\alpha}{r}$ keeps the update's scale roughly independent of $r$, so you can change
#   the rank without re-tuning the learning rate much.
# * Trainable parameters per layer: $r(d_{in} + d_{out})$ instead of $d_{in} d_{out}$:
#   for a 768 × 768 projection and $r = 8$, 12,288 instead of 589,824 (2 %).

# %% [markdown]
# > **Exercise 1.** Complete `LoRALinear`: wrap a pretrained `nn.Linear` called `base`, freeze its
# > weight and bias, create trainable `self.lora_A` of shape `(r, in_features)` (random, e.g.
# > `nn.init.kaiming_uniform_(…, a=math.sqrt(5))`) and `self.lora_B` of shape `(out_features, r)`
# > (zeros), and return `base(x) + (alpha / r) * x Aᵀ Bᵀ` in `forward`. The grader checks that
# > the output equals the base layer's at initialisation, the shapes, the scaling, and that only
# > `lora_A` and `lora_B` are trainable.

# %%
import math


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()
        self.base = base
        # YOUR CODE HERE: freeze base, create self.lora_A, self.lora_B, self.scaling
        raise NotImplementedError

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError


grade.ex1(LoRALinear)

# %% [markdown]
# The course version, used from here on, is `scaleml.finetune.LoRALinear`
# ([`src/scaleml/finetune.py`](../../src/scaleml/finetune.py)); `inject_lora` replaces
# chosen `nn.Linear` layers with it and freezes everything else except the new
# classification head. Following the LoRA paper we adapt the attention **query** and
# **value** projections (`q_lin`, `v_lin` in DistilBERT).

# %%
model = new_classifier()
wrapped = ft.inject_lora(model, targets=("q_lin", "v_lin"), r=8, alpha=16)
print(f"wrapped {len(wrapped)} layers, e.g. {wrapped[:2]}")
print(model.distilbert.transformer.layer[0].attention.q_lin)
summary = ft.trainable_summary(model)
lora_only = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
print(f"trainable {summary['trainable']:,} of {summary['total']:,} ({summary['percent']:.2f}%): "
      f"{lora_only:,} in adapters + {summary['trainable'] - lora_only:,} in the new head")

# At initialisation B = 0, so the wrapped model must compute exactly what the original does.
reference = new_classifier().eval()
batch = {k: v[:8] for k, v in enc_test.items()}
with torch.inference_mode():
    diff = (model.eval()(**batch).logits - reference(**batch).logits).abs().max().item()
print(f"max |logit difference| at initialisation: {diff:.1e}")
del reference

# %% [markdown]
# **Expected:** 12 wrapped layers (q and v in each of 6 blocks); trainable
# 741,124 of 67,104,004 parameters (1.10 %): 147,456 in the adapters (12 layers × 8 × (768 + 768))
# and 593,668 in the new head (`pre_classifier` 768 × 768 and `classifier` 768 × 4, with biases,
# which must be trained because they start random). Max logit difference at initialisation: 0.0e+00.
#
# ## 6.3 · Linear probe vs LoRA vs full fine-tuning, on one budget
#
# Same data, same sequence length, same number of steps for the three trained variants
# (one epoch at batch 32 on the CPU budget). What changes is *what* is trained and the
# learning rate: full fine-tuning 5e-5; LoRA 5e-4 (adapters start at zero and are
# low-dimensional, so they need larger steps, typically 10× full fine-tuning); the probe is a
# logistic regression on frozen mean-pooled embeddings (notebook 05).
#
# **Memory.** On a GPU we read `torch.cuda.max_memory_allocated`. On a CPU there is no such
# counter, so we *estimate*: parameter, gradient and AdamW-state bytes by arithmetic
# (`scaleml.profiling.memory_breakdown`) plus activation bytes **measured** for one training
# batch with `scaleml.profiling.SavedActivations` (what autograd saves for backward).

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from scaleml.train import TrainConfig, train


def estimated_memory(model, batch):
    """Bytes: all weights fp32 + grads and AdamW state of trainable params + measured activations."""
    n_all = count_parameters(model)
    n_tr = count_parameters(model, trainable_only=True)
    with prof.SavedActivations(model) as acts:
        model.train()
        model(**batch).loss.backward()
    model.zero_grad(set_to_none=True)
    return 4 * n_all + 16 * n_tr + acts.stats.bytes, acts.stats.bytes


def run_trained(name, model, lr):
    batches = ft.TokenBatches(enc_train, y_train, batch_size=BUDGET.batch_size, seed=0)
    first = next(iter(ft.TokenBatches(enc_train, y_train, batch_size=BUDGET.batch_size, shuffle=False)))
    if IS_CUDA:
        model.to(DEVICE)
        prof.reset_cuda_peak(DEVICE)
    else:
        mem_est, act_bytes = estimated_memory(model, first)
    cfg = TrainConfig(epochs=BUDGET.epochs, lr=lr, warmup_steps=max(1, len(batches) * BUDGET.epochs // 10),
                      max_grad_norm=1.0, log_every=10_000)
    res = train(model, batches, ft.classification_loss(DEVICE), DEVICE, cfg)
    acc = float((ft.predict_logits(model, enc_test, DEVICE).argmax(1) == y_test).mean())
    row = {"method": name, "trainable": count_parameters(model, trainable_only=True), "test_acc": acc,
           "s_per_epoch": res.seconds / BUDGET.epochs, "examples_per_s": res.examples_per_second}
    if IS_CUDA:
        row.update(memory="peak (measured)", memory_mb=prof.cuda_peak_mb(DEVICE))
    else:
        row.update(memory="estimate", memory_mb=mem_est / 2**20, activations_mb=act_bytes / 2**20)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return row


rows = []
# 1) Linear probe: frozen encoder, embeddings once, logistic regression on top.
t0 = time.time()
encoder = AutoModel.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV, use_safetensors=True).to(DEVICE)
if IS_CUDA:
    prof.reset_cuda_peak(DEVICE)
e_train = ft.frozen_embeddings(encoder, enc_train, DEVICE)
probe_s = time.time() - t0
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(e_train, y_train)
probe_acc = probe.score(ft.frozen_embeddings(encoder, enc_test, DEVICE), y_test)
rows.append({"method": "linear probe", "trainable": 768 * len(LABELS) + len(LABELS), "test_acc": float(probe_acc),
             "s_per_epoch": probe_s, "examples_per_s": len(y_train) / probe_s,
             "memory": "peak (measured)" if IS_CUDA else "estimate",
             "memory_mb": prof.cuda_peak_mb(DEVICE) if IS_CUDA else 4 * count_parameters(encoder) / 2**20})
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
del encoder

# 2-3) LoRA r=4 and r=8 on q, v (+ the new head); 4) full fine-tuning.
lora_models = {}
for r in (4, 8):
    m = new_classifier()
    ft.inject_lora(m, targets=("q_lin", "v_lin"), r=r, alpha=2 * r)
    rows.append(run_trained(f"LoRA r={r}", m, lr=5e-4))
    lora_models[r] = m
rows.append(run_trained("full fine-tuning", new_classifier(), lr=5e-5))

# %%
print(f"{'method':17} {'trainable':>10} {'test acc':>8} {'s/epoch':>8} {'memory MB':>10}")
for r in rows:
    print(f"{r['method']:17} {r['trainable']:>10,} {r['test_acc']:>8.3f} {r['s_per_epoch']:>8.1f} "
          f"{r['memory_mb']:>10.0f}  ({r['memory']})")
n = len(y_test)
print(f"(test set of {n}: one standard error is about ±{np.sqrt(0.25 / n):.3f} at 50 % accuracy, "
      f"±{np.sqrt(0.09 / n):.3f} at 90 %)")
_ = (OUT / "peft_comparison.json").write_text(json.dumps({"device": describe(DEVICE), "n_train": N_TRAIN,
                                                     "n_test": N_EVAL, "rows": rows}, indent=2, default=float))

# %% [markdown]
# **Expected (CPU budget, measured):** (one run on 2 shared CPU threads; time and accuracy vary
# between runs)
#
# | method | trainable | test acc | s/epoch | memory (estimate) |
# |---|---|---|---|---|
# | linear probe | 3,076 | 0.833 | 17.5 | 253 MB |
# | LoRA r=4 | 667,396 | 0.830 | 37.3 | 771 MB (505 MB activations) |
# | LoRA r=8 | 741,124 | 0.837 | 48.3 | 773 MB (505 MB activations) |
# | full fine-tuning | 66,956,548 | 0.847 | 82.1 | 2,016 MB (739 MB activations) |
#
# All four accuracies lie within about one standard error (±0.02 at 300 examples) of each
# other: this budget cannot rank them. LoRA trained ~1.7–2.2× faster per epoch than full
# fine-tuning here and needed ~38 % of its estimated memory.
#
# **What the table says, and what it does not.**
#
# * **Trainable parameters**: LoRA trains ~1 % of the model, and most of that is the new
#   classification head, not the adapters.
# * **Memory**: most of the saving is the gradient and AdamW state of the frozen weights
#   (~770 MB here). Activations also shrink, but far less (739 → 505 MB measured): a frozen
#   `nn.Linear` does not need its *input* saved, because no weight gradient is computed, yet the
#   backward pass still runs through every layer and the attention/GELU/LayerNorm activations
#   stay. The frozen weights still have to be stored. At 67M parameters the saving is modest;
#   at 7B it is the difference between fitting and not fitting (6.1).
# * **Time**: LoRA skips the weight-gradient matmuls of frozen layers (about half of the
#   backward FLOPs, a third of the step) and the AdamW update of 67M parameters, but still
#   computes activation gradients through every layer. On our CPU that gave 1.7–2.2× per epoch;
#   expect less on a GPU, where the optimiser step is cheap relative to the matmuls.
# * **Accuracy**: with this tiny budget the ranking is noisy. With more data, LoRA usually
#   gets within a point or so of full fine-tuning on classification tasks like this.
#
# **On a T4 you should see** (`gpu` budget, 10k examples, 128 tokens, fp16 autocast) LoRA and
# full fine-tuning both around 0.90–0.93 test accuracy and the probe lower, a peak-memory
# gap of a few hundred MB in LoRA's favour at batch 64 (activations dominate both), and
# LoRA epochs roughly 20–35 % faster (not measured in our verification runs).
#
# ## 6.4 · The same with `peft`
#
# Hugging Face `peft` implements LoRA (and many relatives: DoRA, IA³, prefix tuning) for any
# `transformers` model. The configuration maps one-to-one onto what we built:
# `r`, `lora_alpha`, `target_modules`, and `modules_to_save` for the new head (by default
# `peft` saves only `classifier` for sequence classification; DistilBERT's head also has
# `pre_classifier`, so we list both).

# %%
from peft import LoraConfig, TaskType, get_peft_model

config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.0,
                    target_modules=["q_lin", "v_lin"], modules_to_save=["pre_classifier", "classifier"])
peft_model = get_peft_model(new_classifier(), config)
peft_model.print_trainable_parameters()
n_peft = count_parameters(peft_model, trainable_only=True)
n_ours = count_parameters(lora_models[8], trainable_only=True)
print(f"trainable: ours {n_ours:,} | peft {n_peft:,} | equal: {n_ours == n_peft}")

# Copy our *trained* r=8 adapters and head into the peft model: identical outputs prove identical maths.
ours = lora_models[8].to(DEVICE).eval()
with torch.no_grad():
    for name, mod in ours.named_modules():
        if isinstance(mod, ft.LoRALinear):
            target = peft_model.base_model.model.get_submodule(name)
            target.lora_A["default"].weight.copy_(mod.lora_A)
            target.lora_B["default"].weight.copy_(mod.lora_B)
    for head in ("pre_classifier", "classifier"):
        dst = peft_model.base_model.model.get_submodule(head).modules_to_save["default"]
        dst.load_state_dict(ours.get_submodule(head).state_dict())
peft_model.to(DEVICE).eval()
batch = {k: v[:32].to(DEVICE) for k, v in enc_test.items()}
with torch.inference_mode():
    gap = (peft_model(**batch).logits - ours(**batch).logits).abs().max().item()
print(f"max |logit difference| ours vs peft with the same weights: {gap:.1e}")

# %% [markdown]
# **Expected:** `peft` reports `trainable params: 741,124 || all params: 67,697,672 ||
# trainable%: 1.0948`, and the counts are equal. Its *all params* is larger than our total
# by 593,668 because `modules_to_save` keeps a trainable **copy** of the head next to the
# frozen original. With identical weights the outputs agree exactly (max difference 0.0 in our run).
#
# ## 6.5 · Merging adapters: zero serving overhead
#
# After training, $W' = W + \frac{\alpha}{r} BA$ is an ordinary weight matrix of the original
# shape. Folding it in removes the two extra matmuls per adapted layer, so a merged model
# serves exactly as fast as the base model. The price: you can no longer switch adapters
# without un-merging.

# %% [markdown]
# > **Exercise 2.** Write `merge_lora(W, A, B, alpha, r)` returning the merged weight
# > `W + (alpha / r) · B @ A` (tensors: `W` is `(out, in)`, `A` is `(r, in)`, `B` is `(out, r)`).
# > Do not modify `W` in place.

# %%
def merge_lora(W, A, B, alpha, r):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(merge_lora)

# %%
from scaleml.profiling import benchmark

merged = ft.merge_lora_modules(ours).eval()
n_lora_layers = sum(isinstance(m, ft.LoRALinear) for m in merged.modules())
with torch.inference_mode():
    gap = (merged(**batch).logits - ours(**batch).logits).abs().max().item()
print(f"LoRALinear layers left after merging: {n_lora_layers} | max |logit difference|: {gap:.1e}")


def infer(m):
    return lambda: m(**batch)


with torch.inference_mode():
    t_unmerged = benchmark(infer(ours), DEVICE, warmup=1, iters=5)["median_s"]
    t_merged = benchmark(infer(merged), DEVICE, warmup=1, iters=5)["median_s"]
print(f"batch of 32: unmerged {t_unmerged * 1e3:.0f} ms | merged {t_merged * 1e3:.0f} ms")

peft_merged = peft_model.merge_and_unload()
with torch.inference_mode():
    gap_peft = (peft_merged(**batch).logits - ours(**batch).logits).abs().max().item()
print(f"peft merge_and_unload -> {type(peft_merged).__name__}, max |logit difference|: {gap_peft:.1e}")

# %% [markdown]
# **Expected (CPU, measured):** 0 `LoRALinear` layers left after merging and a max logit
# difference of ~1e-6 (float rounding: `W + s·BA` is summed in a different order than
# `Wx + s·B(Ax)`); `peft`'s `merge_and_unload` returns a plain `DistilBertForSequenceClassification`
# with the same ~1e-6 difference. A batch of 32 took 890 ms unmerged vs 811 ms merged on CPU
# (~10 %, noisy): the adapter matmuls are small but are extra kernel launches per layer, which
# matters more for small batches on a GPU.
#
# ## 6.6 · Serving many adapters on one base model
#
# An adapter is tiny. Save only the LoRA and head weights and compare with a full copy of the model:

# %%
from safetensors.torch import save_file

adapter = {n: p.detach().cpu().contiguous() for n, p in ours.named_parameters() if p.requires_grad}
path = OUT / "lora_r8_adapter.safetensors"
save_file(adapter, str(path))
full_bytes = sum(p.numel() * 4 for p in ours.parameters())
lora_bytes = sum(v.numel() * 4 for n, v in adapter.items() if "lora_" in n)
print(f"adapter file {path.name}: {path.stat().st_size / 2**20:.2f} MB "
      f"(LoRA matrices {lora_bytes / 2**20:.2f} MB + head) vs full model {full_bytes / 2**20:.0f} MB")
print(f"1,000 customer-specific models: full copies {1000 * full_bytes / 2**30:.0f} GB, "
      f"adapters {1000 * path.stat().st_size / 2**30:.1f} GB + one base")

# %% [markdown]
# **Expected:** an adapter file of 2.83 MB (0.56 MB of LoRA matrices; the rest is the 768 × 768
# `pre_classifier` head) against a 256 MB model: 1,000 tasks cost 250 GB as full copies vs
# 2.8 GB of adapters plus one base.
#
# How a multi-tenant service uses this:
#
# * **Swap per request batch** (simplest): keep the base in GPU memory and load the adapter
#   weights for the current tenant; `peft` does this with `load_adapter(…, adapter_name=…)` and
#   `set_adapter(name)`. Fine when traffic per tenant is bursty.
# * **Mixed-adapter batches** (S-LoRA, Punica, vLLM's multi-LoRA): run the shared base matmul
#   once for the whole batch, then a batched, *gathered* low-rank matmul where each row uses its
#   own tenant's $A$, $B$. Throughput stays close to the base model's while serving thousands
#   of adapters; adapters live in CPU memory and are paged into the GPU on demand.
# * **Merge** only when one adapter serves almost all the traffic (6.5).
#
# The trade-off to name in a design interview: merged = fastest per request, one task per
# replica; unmerged multi-adapter = slightly slower kernels, but one fleet for every tenant and
# adapters you can hot-swap, roll back and A/B test independently of the base model.

# %% [markdown]
# > **Exercise 3.** A product team asks you to fine-tune a 7B-parameter LLM for 40 enterprise
# > customers, each with 2–20k labelled examples, on T4/A10-class GPUs (16–24 GB). Using 6.1's
# > memory arithmetic, decide between full fine-tuning, LoRA and QLoRA; choose which modules to
# > adapt and a starting rank; and describe how you would serve the 40 variants (merged vs
# > multi-adapter) and evaluate whether LoRA lost quality against full fine-tuning. Write your
# > answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: peft` | not installed | `pip install peft` (it is in `level3/requirements.txt`) |
# | LoRA accuracy stays at chance | learning rate too low for adapters, or the new head was left frozen | lr ~10× full fine-tuning (1e-4–1e-3); train `classifier`/`pre_classifier` (`modules_to_save`) |
# | `ValueError: Target modules ... not found` | module names differ by architecture | print `model.named_modules()`; DistilBERT `q_lin`/`v_lin`, BERT `query`/`value`, Llama `q_proj`/`v_proj` |
# | Trainable count is 0 or 66M | forgot to freeze the base / to unfreeze the adapters | check `requires_grad` per parameter name |
# | Outputs differ after merging | merged in fp16, or forgot the `alpha / r` scaling | merge in fp32; `W + (alpha/r) B @ A` |
# | Memory barely drops with LoRA | activations dominate at this size | gradient checkpointing, shorter sequences, smaller batch + accumulation (notebook 02) |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Derive LoRA's parameter count for one `d × d` layer and explain why $B$ is initialised to zero and why the update is scaled by $\alpha / r$.
# 2. Which memory does LoRA save during training, and which does it not? How do QLoRA and gradient checkpointing complement it?
# 3. Which layers would you adapt in a transformer, and how would you choose the rank? What would you measure?
# 4. When is a linear probe enough, and when do you need LoRA or full fine-tuning?
# 5. Merged vs unmerged adapters at serving time: latency, flexibility, and how you would serve 1,000 per-customer adapters.
# 6. LoRA matches full fine-tuning on your benchmark but users report worse results on long inputs. What could explain it and how would you investigate?
#
# ➡️ **Next:** [`07_embeddings_and_semantic_search.ipynb`](07_embeddings_and_semantic_search.ipynb) — from classifying text to *retrieving* it: lexical, dense and hybrid search, evaluated properly.
