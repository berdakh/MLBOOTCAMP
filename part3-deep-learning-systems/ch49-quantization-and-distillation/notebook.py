# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 49 · Quantisation & distillation
#
# **Part III · Chapter 49 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# A model that is accurate but too slow or too expensive to serve is of limited practical use.
# The two standard ways to make a trained model cheaper are **quantisation** (store and compute
# with fewer bits) and **distillation** (train a small *student* to imitate a large
# *teacher*). Interview questions cover how they work, what they cost in accuracy, and when
# each is appropriate; this lab measures all three quantities (accuracy, latency, size) on one task.
#
# By the end of this lab you should be able to:
#
# 1. fine-tune a DistilBERT teacher on AG News and measure its accuracy, latency and size;
# 2. apply post-training dynamic int8 quantisation with `quantize_dynamic` and measure what changes;
# 3. quantise a weight matrix by hand (symmetric, per-channel int8) and analyse the error, including int4;
# 4. write the knowledge-distillation loss (temperature T, weight α) and distil the teacher into a small student;
# 5. build an accuracy / latency / size table and choose a model for a latency budget.
#
# **You will produce:** `artifacts/level3/nb11/compression_table.json` and the saved
# fp32 / int8 teacher weights (to compare file sizes) in `artifacts/level3/nb11/`.
# Helpers used here live in [`src/scaleml/serving.py`](../../src/scaleml/serving.py)
# (compression section); the training loop is [`src/scaleml/train.py`](../../src/scaleml/train.py).
#
# **Model:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
# (Apache-2.0), pinned to a revision. **Data:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news)
# (4 news topics; see its card for licence and terms; for research and teaching use).
#
# **Compute budget.** On a T4 the teacher is fully fine-tuned on 20k examples. On a CPU that
# would take hours, so the CPU budget fine-tunes only the **top 3 of 6 layers** on 1.5k
# examples for one epoch (~2 minutes): a weaker teacher, the same method. The student is
# deliberately tiny (a TF-IDF + MLP classifier) so that it trains in seconds anywhere.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*.

# %% [markdown]
# Read first: [Chapter 49 text](README.md), which explains quantisation and distillation in more detail than this lab.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

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
# --- Catch up: data and the pretrained model (nothing from earlier notebooks) -----
import copy
import json
import pickle
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scaleml import profiling as prof
from scaleml import serving as S
from scaleml.device import autocast, count_parameters, seed_everything
from grading import ch49 as grade
from scaleml.textdata import load_text_classification
from scaleml.train import TrainConfig, train

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb11"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)
CPU = torch.device("cpu")

BERT_ID, BERT_REV = "distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"
print(f"model card: https://huggingface.co/{BERT_ID} | data card: https://huggingface.co/datasets/fancyzhx/ag_news")
N_LABELLED = BUDGET.train_examples
N_TRANSFER = BUDGET.train_examples if DEVICE.type == "cuda" else 1000  # extra *unlabelled* text for distillation
MAX_LEN = BUDGET.max_seq_len
FROZEN_LAYERS = 3 if DEVICE.type == "cpu" else 0  # CPU budget: train only the top 3 of 6 layers

t0 = time.time()
train_all, eval_df, LABELS = load_text_classification("ag_news", N_LABELLED + N_TRANSFER, BUDGET.eval_examples)
lab_df = train_all.iloc[:N_LABELLED].reset_index(drop=True)
transfer_texts = list(train_all.text.iloc[N_LABELLED:])  # we pretend these have no labels
eval_texts, eval_y = list(eval_df.text), eval_df.label.to_numpy()
tok = AutoTokenizer.from_pretrained(BERT_ID, revision=BERT_REV)
print(f"labels: {LABELS} | labelled {len(lab_df)}, unlabelled transfer {len(transfer_texts)}, "
      f"eval {len(eval_df)} | loaded in {time.time() - t0:.0f} s")

# %% [markdown]
# **Expected:** the two card links, labels `['World', 'Sports', 'Business', 'Sci/Tech']`,
# 1,500 labelled + 1,000 transfer + 500 evaluation examples on the CPU budget (20,000 /
# 20,000 / 4,000 on a T4). The first run downloads AG News (~30 MB) and the tokenizer.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Quantisation | Storing (and computing with) numbers in fewer bits, e.g. 8-bit integers instead of 32-bit floats | DistilBERT's Linear layers converted to int8 |
# | Scale `s` | The real-valued size of one integer step; `x ≈ q × s` | `max(abs(W[row])) / 127` for one weight row |
# | Per-tensor vs per-channel | One scale for a whole matrix vs one scale per output row | per-channel cuts the weight error 2.4× in section 49.3 |
# | Dynamic quantisation | Weights stored as int8 once; activations quantised on the fly at run time, no calibration data | `quantize_dynamic(model, {nn.Linear})` |
# | Weight-only quantisation | Only the weights are low-bit; the arithmetic still runs in floating point | the int8/int4 simulation in section 49.3 |
# | Group-wise scales | One scale per small block of weights (e.g. 128), needed for int4 | GPTQ / AWQ 4-bit LLM loaders |
# | Knowledge distillation | Training a small *student* to imitate a large *teacher*'s outputs | DistilBERT teacher → TF-IDF + MLP student |
# | Temperature `T` | Divides the logits before the softmax; `T > 1` flattens the distribution | `T = 4` in the student runs |
# | Soft targets | The teacher's full probability vector, not just its top answer | "Sci/Tech 15%" for a Business article |
# | Transfer set | Unlabelled text the teacher labels so the student sees more examples | the 1,000 extra AG News texts |
# | Agreement | Share of examples where two models give the same prediction | int8 vs fp32 agree on ~96% |

# %% [markdown]
# ## 49.1 · The teacher: fine-tune DistilBERT
#
# A sequence-classification head (`AutoModelForSequenceClassification`, 4 labels) on
# DistilBERT, trained with the course's explicit loop (`scaleml.train.train`: AdamW,
# warm-up + cosine schedule, gradient clipping, fp16 autocast + `GradScaler` on a T4).

# %%
teacher = AutoModelForSequenceClassification.from_pretrained(
    BERT_ID, revision=BERT_REV, num_labels=len(LABELS), use_safetensors=True).to(DEVICE)
for p in teacher.distilbert.embeddings.parameters():
    p.requires_grad = FROZEN_LAYERS == 0
for layer in teacher.distilbert.transformer.layer[:FROZEN_LAYERS]:
    for p in layer.parameters():
        p.requires_grad = False
print(f"{count_parameters(teacher) / 1e6:.1f}M parameters, {count_parameters(teacher, trainable_only=True) / 1e6:.1f}M "
      f"trainable (frozen: embeddings + {FROZEN_LAYERS} layers)" if FROZEN_LAYERS else "all parameters trainable")


def batch_fn(df, batch_size, seed=0):
    calls = iter(range(10**6))

    def batches():  # a fresh shuffle per call (train() calls it once per epoch)
        d = df.sample(frac=1.0, random_state=seed + next(calls))
        out = []
        for i in range(0, len(d), batch_size):
            b = d.iloc[i:i + batch_size]
            enc = tok(list(b.text), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            out.append({**enc, "labels": torch.tensor(b.label.to_numpy())})
        return out
    return batches


def bert_loss(model, batch):
    return model(**{k: v.to(DEVICE) for k, v in batch.items()}).loss


@torch.no_grad()
def predict_logits(model, texts, device, batch_size=64):
    model.eval()
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to(device)
        with autocast(device):
            outs.append(model(**enc).logits.float().cpu())
    return torch.cat(outs)


def accuracy(logits, y):
    return float((logits.argmax(-1).numpy() == y).mean())


batches = batch_fn(lab_df, BUDGET.batch_size)
steps = len(batches()) * BUDGET.epochs
cfg = TrainConfig(epochs=BUDGET.epochs, lr=1e-4 if FROZEN_LAYERS else 5e-5, warmup_steps=max(1, steps // 10),
                  log_every=max(1, steps // 4))
t0 = time.time()
result = train(teacher, batches, bert_loss, DEVICE, cfg)
print(f"trained {result.steps} steps in {result.seconds:.0f} s ({result.examples_per_second:.1f} examples/s)")
for h in result.history:
    print(f"  step {h['step']:3d}: loss {h['loss']:.3f}")
teacher_logits_eval = predict_logits(teacher, eval_texts, DEVICE)
ACC_TEACHER = accuracy(teacher_logits_eval, eval_y)
print(f"teacher accuracy on {len(eval_y)} eval examples: {ACC_TEACHER:.3f}")

# %% [markdown]
# **Expected (CPU budget, measured):** 67.0M parameters, 21.9M trainable; 47 steps in
# ~100–160 s (10–15 examples/s; slower on a busy machine); the logged loss falls from ~1.15 to
# ~0.35; accuracy 0.878 in our final run (0.866 in an earlier run with a different random
# sample; 4 balanced classes, so chance is 0.25). Ignore the
# load report listing `vocab_*` as unexpected and `classifier`/`pre_classifier` as missing:
# the masked-language-model head is dropped and a new classification head is created. **On a T4 you should see** full
# fine-tuning on 20k examples reach ~0.92–0.94 in roughly 10 minutes (not measured in our
# verification runs).
#
# ## 49.2 · Post-training dynamic int8 quantisation
#
# **Dynamic quantisation** (`torch.ao.quantization.quantize_dynamic`) converts the weights
# of every `nn.Linear` to int8 *once*, and at run time quantises each layer's *activations*
# on the fly (a scale computed per batch), runs an int8 × int8 matmul with int32
# accumulation (fbgemm/oneDNN kernels on x86, qnnpack on ARM) and converts back to float.
# No calibration data, no retraining, one line. It is a **CPU** technique: on GPUs one uses
# weight-only int8/int4 kernels (bitsandbytes, GPTQ/AWQ) or TensorRT.
#
# `torch.ao.quantization` is deprecated in recent PyTorch (scheduled for removal in 2.10
# in favour of the `torchao` library); the cell suppresses the warning and reports cleanly
# if the API is missing.
#
# > **Intuition.** Dynamic quantisation is like a shop that keeps its price list (the weights)
# > in whole cents, but converts each customer's foreign currency (the activations) at today's rate
# > as they arrive. Nothing needs to be prepared in advance, at the cost of a conversion per request.
# > *Where the analogy breaks down:* only `nn.Linear` layers are converted; the embeddings (a third of
# > this model) stay fp32, which is why the file only halves.

# %%
teacher_cpu = copy.deepcopy(teacher).to(CPU).float().eval()
teacher_int8 = None
try:
    from torch.ao.quantization import quantize_dynamic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        teacher_int8 = quantize_dynamic(teacher_cpu, {torch.nn.Linear}, dtype=torch.qint8)
    print("quantised Linear layers:",
          sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in teacher_int8.modules()))
    print(teacher_int8.distilbert.transformer.layer[0].ffn)
except Exception as exc:  # e.g. a PyTorch version without torch.ao.quantization
    print(f"dynamic quantisation unavailable here ({type(exc).__name__}: {exc}); use torchao instead.")

size_fp32 = S.file_size_mb(teacher_cpu.state_dict(), OUT / "teacher_fp32.pt")
print(f"fp32 teacher on disk: {size_fp32:.0f} MB")
if teacher_int8 is not None:
    size_int8 = S.file_size_mb(teacher_int8.state_dict(), OUT / "teacher_int8.pt")
    print(f"int8 teacher on disk: {size_int8:.0f} MB ({size_int8 / size_fp32:.0%} of fp32)")
    int8_logits = predict_logits(teacher_int8, eval_texts, CPU)
    ACC_INT8 = accuracy(int8_logits, eval_y)
    agree = float((int8_logits.argmax(-1) == teacher_logits_eval.argmax(-1)).float().mean())
    print(f"accuracy fp32 {ACC_TEACHER:.3f} -> int8 {ACC_INT8:.3f} | same prediction on {agree:.1%} of examples")

# %% [markdown]
# **Expected (CPU, measured):** 38 Linear layers quantised (6 layers × (Q, K, V, output, 2 FFN)
# = 36, plus the pre-classifier and classifier); the file shrinks from 268 MB to 139 MB
# (52%), not to 25%, because the **embeddings** (23.4M of the 67M parameters) are not
# `nn.Linear` and stay fp32. Accuracy does not move (our run: 0.878 → 0.878), yet int8 and
# fp32 made the same prediction on only 95.6% of examples: some answers flipped both ways.
# Same accuracy is not the same model; check agreement (and calibration) too.
#
# Now latency. We time the model's forward pass alone (tokenisation excluded), for a single
# request on **one thread** (how CPU servers usually run: several single-threaded workers,
# one request each) and for a batch of 32 on all the notebook's threads. Dynamic quantisation
# is a CPU feature, so both models run on the CPU.

# %%
enc1 = tok(eval_texts[:1], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
enc32 = tok(eval_texts[:32], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
LAT = {}
N_THREADS = torch.get_num_threads()
with torch.inference_mode():
    for name, m in [("fp32", teacher_cpu), ("int8", teacher_int8)]:
        if m is None:
            continue
        torch.set_num_threads(1)
        r1 = prof.benchmark(lambda m=m: m(**enc1), CPU, warmup=3, iters=20)
        torch.set_num_threads(N_THREADS)
        r32 = prof.benchmark(lambda m=m: m(**enc32), CPU, warmup=1, iters=5)
        b1, b32 = r1["median_s"], r32["median_s"]
        LAT[name] = {"batch1_ms": b1 * 1e3, "batch32_ms": b32 * 1e3, "batch1_min_ms": r1["min_s"] * 1e3,
                     "batch32_min_ms": r32["min_s"] * 1e3}
        print(f"{name}: batch 1 median {b1 * 1e3:6.1f} ms (min {r1['min_s'] * 1e3:5.1f}) | batch 32 median "
              f"{b32 * 1e3:7.1f} ms (min {r32['min_s'] * 1e3:6.1f}; {b32 * 1e3 / 32:.1f} ms/example, "
              f"{N_THREADS} threads)")
if "int8" in LAT:
    print(f"int8 speed-up: {LAT['fp32']['batch1_ms'] / LAT['int8']['batch1_ms']:.2f}x at batch 1, "
          f"{LAT['fp32']['batch32_ms'] / LAT['int8']['batch32_ms']:.2f}x at batch 32")

# %% [markdown]
# **Expected (CPU, measured):** fp32 batch 1 ~45–55 ms on one thread, batch 32 ~0.7–1.5 s on
# two; int8 is 2–3× faster (our runs: 2.3× and 3.3× at batch 1, ~15–20 ms; 2.2× and 2.5× at
# batch 32). It depends on the CPU's int8 instructions (e.g. AVX-512 VNNI) and on machine
# load: on a heavily loaded machine we also saw multi-threaded int8 runs *slower* than fp32,
# which is why single requests are timed on one thread. On a CPU, batching cuts the cost per
# example only modestly (a 64-token sequence already makes the matrix products large); on a
# GPU it matters far more (Chapter 50).
#
# ## 49.3 · Quantising a weight matrix by hand
#
# Symmetric linear quantisation maps a real weight w to an integer
# `q = round(w / scale)` in `[-127, 127]` and back with `ŵ = q × scale`. The rounding error
# is at most `scale / 2` per weight. **Per-tensor**: one scale for the whole matrix
# (`max|W| / 127`), so a single large weight makes the step coarse for everybody.
# **Per-channel**: one scale per output row, adapting to each row's range. Int4 uses
# `[-7, 7]` and needs per-channel or finer *group-wise* scales (e.g. one per 128 weights,
# as in GPTQ/AWQ) to work at all.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-int8-quantization.svg" alt="Float values snapping to a 255-level integer grid, and per-tensor vs per-channel scales on three rows of different range" width="100%">
#
# *What to notice:* with one per-tensor scale the small-range rows get only one or two integer levels; per-channel scales give every row the whole grid.
#
# > **Intuition.** Quantising is measuring every weight with a ruler that has only 255 tick
# > marks. A per-tensor scale sizes the ruler for the *longest* object in the matrix, so short rows
# > are measured in very coarse steps; per-channel gives each row a ruler cut to its own length.
# > *Where the analogy breaks down:* a layer's output is a sum of many products, and independent
# > rounding errors partly cancel in that sum, so the output error is smaller than "ruler
# > precision" suggests (section 49.3 measures exactly this).

# %% [markdown]
# > **Exercise 1.** Write `quantize_per_channel_int8(W)` for a 2-D float tensor `W` (rows = output
# > channels). Return `(q, scale)`: `scale` has one entry per row, `max|W[row]| / 127` (use 1 for an
# > all-zero row so nothing divides by zero); `q` is `round(W / scale)` clamped to `[-127, 127]`, as
# > `torch.int8`. The grader dequantises with `q.float() * scale[:, None]` and checks the error is at most
# > `scale / 2` everywhere.

# %%
def quantize_per_channel_int8(W):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(quantize_per_channel_int8)

# %%
quant_fn = quantize_per_channel_int8
try:
    quant_fn(torch.ones(2, 2))
except NotImplementedError:
    quant_fn = S.quantize_per_channel_int8  # Exercise 1 not done yet: the course's version

lin1 = teacher_cpu.distilbert.transformer.layer[0].ffn.lin1  # 768 -> 3072, the first FFN matrix
W = lin1.weight.detach().float()
captured = {}


def grab_input(module, inputs, output):  # a forward hook; returning a value would REPLACE the output
    captured["x"] = inputs[0].detach()


hook = lin1.register_forward_hook(grab_input)
with torch.inference_mode():
    teacher_cpu(**enc32)
hook.remove()
X = captured["x"].reshape(-1, W.shape[1])  # real activations entering this layer
Y = X @ W.T


def report(name, W_hat):
    rel_w = ((W_hat - W).norm() / W.norm()).item()
    rel_y = ((X @ W_hat.T - Y).norm() / Y.norm()).item()
    print(f"{name:26} weight rel. error {rel_w:.2e} | max |error| {(W_hat - W).abs().max().item():.1e} "
          f"| layer-output rel. error {rel_y:.2e}")
    return rel_y


row_max = W.abs().amax(1)
print(f"W: {tuple(W.shape)}, |W| max {W.abs().max():.3f}; per-row max ranges {row_max.min():.3f}-{row_max.max():.3f}")
s_t = W.abs().max() / 127
errs = {"per-tensor int8": report("per-tensor int8", torch.round(W / s_t).clamp(-127, 127) * s_t)}
q, sc = quant_fn(W)
errs["per-channel int8"] = report("per-channel int8 (yours)" if quant_fn is not S.quantize_per_channel_int8
                                  else "per-channel int8", S.dequantize(q, sc))
q4, sc4 = S.quantize_per_channel_int8(W, bits=4)
errs["per-channel int4"] = report("per-channel int4", S.dequantize(q4, sc4))
print(f"memory: fp32 {W.numel() * 4 / 1e6:.2f} MB | int8 + scales {W.numel() / 1e6 + sc.numel() * 4 / 1e6:.2f} MB "
      f"| int4 packed {W.numel() / 2e6:.2f} MB")

# %% [markdown]
# **Expected (measured):** the matrix is 3072 × 768 and its rows' maxima range from 0.066 to
# 0.346. Per-tensor int8: relative weight error 1.9e-2; per-channel int8: 8.0e-3 (2.4× smaller,
# because small-range rows get finer steps); per-channel int4: 1.5e-1 (an int4 step is
# 127/7 ≈ 18× coarser). The relative error of the layer's *output* on real activations is
# of the same order (1.2e-2, 5.3e-3, 9.6e-2): rounding errors are roughly independent and
# partly average out in each dot product. Memory: 9.4 MB fp32 → 2.4 MB int8 → 1.2 MB int4.
#
# What matters is the effect on the *task*. Simulate weight-only quantisation of **every**
# Linear layer (quantise → dequantise in place, compute in fp32) and re-evaluate:

# %%
WO = {}
for bits in (8, 4):
    m = copy.deepcopy(teacher_cpu)
    n_layers = S.fake_quantize_linears(m, bits=bits)
    WO[bits] = accuracy(predict_logits(m.to(DEVICE), eval_texts, DEVICE), eval_y)  # fp math: any device
    print(f"weight-only int{bits} per-channel ({n_layers} Linear layers): accuracy {WO[bits]:.3f} "
          f"(fp32 {ACC_TEACHER:.3f})")
    del m

# %% [markdown]
# **Expected (CPU, measured):** weight-only int8 matches fp32 (our run: 0.880 vs 0.878), and,
# perhaps surprisingly, per-channel int4 without groups also holds up on this task (0.876)
# despite a ~15% weight error. A 4-class classifier with comfortable margins is robust;
# do not generalise from it. Generative LLMs compound errors over many tokens and are much more
# sensitive, which is why 4-bit LLM loaders use small groups (64–128 weights per scale) and
# calibration (GPTQ, AWQ), and why int8 is the "safe default". Always measure on your task.
#
# **Optional GPU extension.** On a T4 you can load the teacher in 8-bit or 4-bit with
# `bitsandbytes` (`BitsAndBytesConfig(load_in_8bit=True)` or `load_in_4bit=True`) and compare
# memory and speed. It is not needed for the rest of the notebook, needs `pip install
# bitsandbytes`, and works only on CUDA, so it is guarded by a flag and was not run in our
# verification.

# %%
RUN_GPU_EXTENSION = False

if RUN_GPU_EXTENSION and DEVICE.type == "cuda":
    try:
        from transformers import BitsAndBytesConfig

        teacher.save_pretrained(OUT / "teacher_hf")
        tok.save_pretrained(OUT / "teacher_hf")
        for cfg_name, bnb in [("8-bit", BitsAndBytesConfig(load_in_8bit=True)),
                              ("4-bit nf4", BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.float16))]:
            mq = AutoModelForSequenceClassification.from_pretrained(OUT / "teacher_hf", quantization_config=bnb)
            print(f"{cfg_name}: memory footprint {mq.get_memory_footprint() / 1e6:.0f} MB, accuracy "
                  f"{accuracy(predict_logits(mq, eval_texts, DEVICE), eval_y):.3f}")
            del mq
    except Exception as exc:
        print(f"bitsandbytes extension failed ({type(exc).__name__}: {exc}); pip install bitsandbytes")
else:
    print("Skipped (RUN_GPU_EXTENSION = False or no GPU).")

# %% [markdown]
# ## 49.4 · Knowledge distillation
#
# A teacher's output distribution carries more information than the label: for a
# Business article it might give Sci/Tech 15% and Sports 0.1%, which says which mistakes
# are reasonable ("dark knowledge", Hinton, Vinyals & Dean 2015). The student is trained to
# match the teacher's probabilities **softened by a temperature T** (`softmax(z / T)`; T > 1
# flattens the distribution and exposes the small probabilities), mixed with the ordinary
# cross-entropy on true labels:
#
# $$\mathcal{L} = \alpha \, T^2 \, \mathrm{KL}\big(\mathrm{softmax}(z_t/T) \,\|\, \mathrm{softmax}(z_s/T)\big) + (1-\alpha)\, \mathrm{CE}(z_s, y)$$
#
# The `T²` keeps the soft term's gradient scale independent of T. Distillation's other
# big advantage is data: the teacher can label **unlabelled** text (the *transfer set*), so
# the student can learn from far more examples than were ever annotated.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-distillation.svg" alt="Teacher and student both read the input; a KL term compares their softened probabilities and a cross-entropy term compares the student with the true label" width="100%">
#
# *What to notice:* the student gets two signals, the teacher's soft targets (weighted α·T²) and the true label (weighted 1 − α); without a label only the KL term remains.
#
# > **Intuition.** A hard label says "this is Business". The teacher's soft targets say "Business,
# > but it reads a bit like Sci/Tech and nothing like Sports", which tells the student *which*
# > mistakes are reasonable. Raising `T` is like turning up the contrast on those faint marks:
# > `softmax([4, 1, 0])` is `[0.94, 0.05, 0.02]` at `T = 1` but `[0.54, 0.26, 0.20]` at `T = 4`.
# > *Where the analogy breaks down:* too high a `T` flattens everything towards uniform and the
# > signal disappears; `T` between 2 and 8 is the usual range.

# %% [markdown]
# > **Exercise 2.** Write `kd_loss(student_logits, teacher_logits, labels, T, alpha)` returning the
# > scalar loss above. Use `F.kl_div(F.log_softmax(student / T), F.softmax(teacher / T),
# > reduction="batchmean")` for the KL term (note the argument order: `kl_div` takes the student's
# > *log*-probabilities first) and `F.cross_entropy` for the hard term.

# %%
def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(kd_loss)

# %% [markdown]
# The student: TF-IDF features (unigrams + bigrams, 5,000 features) and a one-hidden-layer
# MLP: ~640k parameters, about 100× fewer than the teacher, and no attention at all. The
# teacher labels the 1,500 labelled examples and the unlabelled transfer examples
# (forward passes only). Four ways to train the same student, 3 seeds each:
#
# * **A**: hard labels, labelled set only (no teacher);
# * **B**: distillation (T = 4, α = 0.7) on the labelled set;
# * **C**: hard *pseudo-labels* (the teacher's argmax) on labelled + transfer;
# * **D**: distillation on labelled + transfer (true labels where known, the teacher's argmax as
#   the hard target elsewhere).
#
# (Fixed TF-IDF features keep the student's training to seconds; on the CPU budget the transfer set
# is 1,000 texts.)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time.time()
teacher_logits_lab = predict_logits(teacher, list(lab_df.text), DEVICE)
teacher_logits_tr = predict_logits(teacher, transfer_texts, DEVICE)
print(f"teacher labelled {len(lab_df) + len(transfer_texts)} texts in {time.time() - t0:.0f} s; "
      f"its accuracy on the (hidden) transfer labels: "
      f"{accuracy(teacher_logits_tr, train_all.label.iloc[N_LABELLED:].to_numpy()):.3f}")

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
vec.fit(list(lab_df.text) + transfer_texts)  # fitting a vocabulary needs no labels


def featurize(texts):
    return torch.tensor(vec.transform(texts).toarray(), dtype=torch.float32)


X_lab, X_tr, X_eval = featurize(list(lab_df.text)), featurize(transfer_texts), featurize(eval_texts)
y_lab = torch.tensor(lab_df.label.to_numpy())


def new_student(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(X_lab.shape[1], 128), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                               torch.nn.Linear(128, len(LABELS)))


loss_impl = kd_loss
try:
    loss_impl(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), 2.0, 0.5)
except NotImplementedError:
    loss_impl = S.kd_loss  # Exercise 2 not done yet: the course's version


def train_student(X, y, t_logits, alpha, T=4.0, seed=0, epochs=12, lr=3e-3, bs=128):
    model = new_student(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(X), generator=g).split(bs):
            loss = loss_impl(model(X[idx]), t_logits[idx], y[idx], T, alpha)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model, accuracy(model(X_eval), eval_y)


pseudo_tr = teacher_logits_tr.argmax(-1)
X_all = torch.cat([X_lab, X_tr])
y_all = torch.cat([y_lab, pseudo_tr])
t_all = torch.cat([teacher_logits_lab, teacher_logits_tr])
variants = {
    "A hard labels": (X_lab, y_lab, teacher_logits_lab, 0.0),
    "B KD, labelled": (X_lab, y_lab, teacher_logits_lab, 0.7),
    "C pseudo-labels, +transfer": (X_all, y_all, t_all, 0.0),
    "D KD, +transfer": (X_all, y_all, t_all, 0.7),
}
STUDENT = {}
t0 = time.time()
for name, (X, y, tl, alpha) in variants.items():
    accs = []
    for seed in range(3):
        model, acc = train_student(X, y, tl, alpha, seed=seed)
        accs.append(acc)
    STUDENT[name] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "model": model}
    print(f"{name:28} accuracy {np.mean(accs):.3f} ± {np.std(accs):.3f}  (seeds: {[round(a, 3) for a in accs]})")
print(f"(12 students trained in {time.time() - t0:.0f} s) | teacher: {ACC_TEACHER:.3f}")

# %% [markdown]
# **Expected (CPU budget, measured):** the teacher labels 2,500 texts in about a minute and
# is right on ~90% of the transfer set (0.896). The students land below the teacher (a bag of
# n-grams cannot match a transformer) and their differences are small: compare the means
# with the seed spread (±0.002–0.003) before trusting any ranking. Our run: A 0.835,
# B 0.827, C 0.849, D 0.843. The transfer-set variants (C, D) gained 1–1.5 points over A
# because they see 1.7× as much text; soft targets alone on the labelled set (B) did not help
# this student (slightly worse), and with the transfer set plain pseudo-labels (C) did as
# well as full distillation (D). An earlier run with a different sample gave the same
# pattern (A 0.836, B 0.831, C 0.841, D 0.839). **More data labelled by
# the teacher is usually the largest benefit of distillation**, and soft targets matter more when
# the student is a smaller version of the teacher (same tokenizer, a few layers, e.g.
# DistilBERT itself was distilled from BERT) than for a very different model like ours.
#
# ## 49.5 · The trade-off table
#
# Accuracy, latency (batch 1, CPU, *including* featurisation for the student and excluding
# tokenisation for the transformers, which costs ~1 ms), and size on disk.

# %%
best_name = max(STUDENT, key=lambda k: STUDENT[k]["mean"])
student = STUDENT[best_name]["model"]
student_bytes = len(pickle.dumps(vec)) + sum(p.numel() * 4 for p in student.parameters())
torch.set_num_threads(1)  # same single-request setting as the teachers' batch-1 numbers
with torch.inference_mode():
    st1 = prof.benchmark(lambda: student(featurize(eval_texts[:1])), CPU, warmup=3, iters=30)["median_s"]
    torch.set_num_threads(N_THREADS)
    st32 = prof.benchmark(lambda: student(featurize(eval_texts[:32])), CPU, warmup=2, iters=10)["median_s"]
rows = [{"model": "DistilBERT teacher fp32", "accuracy": ACC_TEACHER, "batch1_ms": LAT["fp32"]["batch1_ms"],
         "batch32_ms": LAT["fp32"]["batch32_ms"], "size_mb": size_fp32}]
if teacher_int8 is not None:
    rows.append({"model": "DistilBERT dynamic int8", "accuracy": ACC_INT8, "batch1_ms": LAT["int8"]["batch1_ms"],
                 "batch32_ms": LAT["int8"]["batch32_ms"], "size_mb": size_int8})
rows.append({"model": f"student ({best_name})", "accuracy": STUDENT[best_name]["mean"], "batch1_ms": st1 * 1e3,
             "batch32_ms": st32 * 1e3, "size_mb": student_bytes / 1e6})
print(f"{'model':38} {'accuracy':>8} {'batch1 ms':>10} {'batch32 ms':>11} {'size MB':>8}")
for r in rows:
    print(f"{r['model']:38} {r['accuracy']:8.3f} {r['batch1_ms']:10.2f} {r['batch32_ms']:11.1f} {r['size_mb']:8.1f}")
(OUT / "compression_table.json").write_text(json.dumps(
    {"device": str(DEVICE), "budget": BUDGET.as_dict(), "rows": rows, "weight_only_accuracy": WO,
     "students": {k: {"mean": v["mean"], "std": v["std"]} for k, v in STUDENT.items()}}, indent=2))
print("wrote", OUT / "compression_table.json")

# %% [markdown]
# **Expected (CPU, measured; our final run):** the teacher at 0.878 accuracy, ~53 ms per
# request and 268 MB; int8 equally accurate, ~16 ms, 139 MB; the student 0.849 (3 points
# below), ~0.8 ms per request including TF-IDF (tens of times faster) and 2.8 MB. **On a T4 you should
# see** the teacher at ~0.93 accuracy and a few ms per request in fp16 (not measured in our
# verification runs): the GPU changes the latency column, not the ranking of sizes.
#
# How to choose: start from the **latency and cost budget** (p99 per request, requests per
# second, hardware), then choose the most accurate model that fits. Quantisation is nearly free
# accuracy-wise and should be tried first; distillation costs a training pipeline (and a
# teacher to maintain) but buys the largest speed-ups; the two compose (quantise the student).

# %% [markdown]
# > **Exercise 3.** Your team serves this topic classifier for a news feed: 2,000 requests per second at
# > peak, a p99 latency budget of 20 ms, CPU-only serving machines. Using *your* table, (a) which model(s)
# > meet the latency budget at batch 1, and how would batching change that answer? (b) The product team
# > says "the student is 3 points worse; deploy the teacher on GPUs instead". What would you measure before
# > agreeing (think: cost per request, the business metric, which classes the student gets wrong)?
# > (c) If you had to make the distilled student better, list three changes in order of expected payoff.
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1 · Symmetric int8 quantisation.**
#
# $$
# q = \operatorname{clamp}\!\big(\operatorname{round}(x / s),\,-127,\,127\big), \qquad s = \frac{\max_i \lvert x_i \rvert}{127}, \qquad \hat{x} = q\,s
# $$
#
# *In words:* divide by the step size, round to the nearest integer, and multiply back. *Symbols:*
# $x$ a real weight, $s$ the scale (one per tensor, per row, or per group), $q$ the stored int8,
# $\hat{x}$ the dequantised value. *Example:* $x = 0.21$, $\max\lvert x\rvert = 0.35$ gives
# $s = 0.35/127 = 0.00276$, $x/s = 76.2$, $q = 76$, $\hat{x} = 0.2094$.
#
# **2 · Rounding error bound.**
#
# $$
# \lvert x - \hat{x} \rvert \le \frac{s}{2}
# $$
#
# *In words:* rounding moves a value by at most half a step. *Example:* above, the error is
# $0.0006 \le s/2 = 0.0014$. Per-channel helps because each row's $s_r = \max_j \lvert W_{rj}\rvert / 127$
# is smaller than the global $s$; int4 uses 7 instead of 127, so its step is $127/7 \approx 18\times$ coarser.
#
# **3 · Memory.** $\text{bytes} = \text{parameters} \times \text{bits} / 8$ (+ one fp32 scale per row or group).
# *Example:* the $3072 \times 768$ FFN matrix has 2.36M weights: 9.4 MB in fp32, 2.4 MB in int8, 1.2 MB in int4.
#
# **4 · Distillation loss** (as in Exercise 2; some papers put $\alpha$ on the CE term instead, so always say which).
#
# $$
# \mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(p_t^{(T)} \,\Vert\, p_s^{(T)}\right) + (1 - \alpha)\, \mathrm{CE}(z_s, y), \qquad p^{(T)} = \operatorname{softmax}(z / T)
# $$
#
# *In words:* match the teacher's softened distribution (weight $\alpha$) and the true label (weight
# $1 - \alpha$). *Symbols:* $z_t, z_s$ teacher and student logits, $T$ the temperature, $\alpha \in [0, 1]$,
# $y$ the label. The $T^2$ undoes the $1/T^2$ shrinkage of the soft term's gradient, so changing $T$
# does not silently change its weight. *Example:* with $\alpha = 0$ this is ordinary cross-entropy
# (variants A and C of section 49.4).
#
# The cell below checks all four numerically.

# %%
# Key equations, checked numerically (CPU, instant).
x = torch.tensor([0.21, -0.35, 0.05, 0.349])
s = x.abs().max() / 127
q = torch.round(x / s).clamp(-127, 127)
err = (x - q * s).abs().max()
print(f"s = {s:.5f} | q = {q.int().tolist()} | max error {err:.5f} <= s/2 = {s / 2:.5f}: {bool(err <= s / 2)}")
print(f"3072 x 768 matrix: fp32 {3072 * 768 * 4 / 1e6:.1f} MB | int8 {3072 * 768 / 1e6:.1f} MB "
      f"| int4 {3072 * 768 / 2e6:.1f} MB")

zt = torch.tensor([[4.0, 1.0, 0.0]])
for T_ in (1.0, 4.0):
    print(f"T = {T_}: softmax(z_t / T) = {F.softmax(zt / T_, -1).squeeze().numpy().round(3)}")
zs, y_, T_, a_ = torch.tensor([[2.0, 1.5, 0.0]]), torch.tensor([0]), 4.0, 0.7
pt, log_ps = F.softmax(zt / T_, -1), F.log_softmax(zs / T_, -1)
by_hand = a_ * T_**2 * (pt * (pt.log() - log_ps)).sum() + (1 - a_) * F.cross_entropy(zs, y_)
print(f"distillation loss by hand {by_hand.item():.4f} | scaleml.serving.kd_loss "
      f"{S.kd_loss(zs, zt, y_, T_, a_).item():.4f}")

# %% [markdown]
# **Expected:** `s = 0.00276`, `q = [76, -127, 18, 127]`, an error below `s/2` (`True`); 9.4 / 2.4 /
# 1.2 MB; the two softmax vectors quoted above; and the same loss computed by hand and by the
# library (`0.6325` both).

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | Teacher training takes far longer than ~2 min on CPU | other processes using the cores; more threads than cores | close other notebooks; it uses 2 threads by design |
# | `DeprecationWarning: torch.ao.quantization is deprecated` | PyTorch 2.8+ | harmless here; for new projects use `torchao` |
# | `quantize_dynamic` model is not faster | the CPU lacks fast int8 instructions, or the model is dominated by non-Linear ops at short sequences | measure at your real batch/sequence sizes; consider ONNX Runtime int8 |
# | `RuntimeError: ... quantized::linear_dynamic ... CUDA` | running the dynamically quantised model on a GPU | dynamic quantisation is CPU-only; on GPU use bitsandbytes / TensorRT |
# | `kd_loss` is negative or `nan` | arguments to `F.kl_div` swapped (it expects log-probabilities first) or missing `log_softmax` | `F.kl_div(F.log_softmax(s/T), F.softmax(t/T), reduction="batchmean")` |
# | Distilled student no better than baseline | too little transfer data, teacher not better than student on that data, T or α badly set | add unlabelled transfer text; try T in 2–8, α in 0.5–0.9 |
# | bitsandbytes import error | not installed, or no CUDA | `pip install bitsandbytes`; keep `RUN_GPU_EXTENSION = False` on CPU |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Explain symmetric int8 quantisation of a weight matrix. Why is per-channel better than per-tensor, and why does int4 need group-wise scales?
# 2. What is the difference between dynamic quantisation, static (calibrated) quantisation, quantisation-aware training and weight-only quantisation? When would you use each?
# 3. Dynamic int8 halved the model file, not quartered it. Why? Where does the speed-up come from, and why might there be none?
# 4. Write the distillation loss. What do the temperature and the T² factor do?
# 5. Why does distillation with an unlabelled transfer set often help more than soft targets alone?
# 6. You must cut inference cost by 4× with at most 1 point of accuracy loss. Walk through your plan and the measurements you would make.
#
# **Next chapter:** [Chapter 50 · High-throughput serving](../ch50-high-throughput-serving/notebook.ipynb) — put a model behind an API and make it fast: dynamic batching, latency vs throughput curves, KV caches and continuous batching for generation, and cost per 1,000 requests.
