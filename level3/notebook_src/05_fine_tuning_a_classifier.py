# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 05 · Fine-tuning a classifier
#
# **Day 3 · ~90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# "We fine-tuned BERT and got 94 %" is not a result until you know what a TF-IDF model
# gets on the same split, what the frozen encoder gets with a linear head, whether the
# validation set was carved from the training data, and whether the probabilities mean
# anything. This notebook fine-tunes DistilBERT the way you would defend it in a design
# review: against honest baselines, with proper splits, calibration and error analysis.
#
# By the end you will be able to:
#
# 1. build two strong cheap baselines: TF-IDF + logistic regression and a linear probe on frozen embeddings;
# 2. fully fine-tune a pretrained encoder with a sequence-classification head (AMP on a GPU, warmup + cosine, gradient clipping, early stopping on a validation split carved from *train*);
# 3. evaluate properly: accuracy, macro-F1, a confusion matrix;
# 4. measure calibration (expected calibration error, reliability table) and fix it with temperature scaling fitted on validation data;
# 5. do error analysis: the most frequent confusions and the most confident mistakes, with examples;
# 6. argue when the cheap baseline is the right production choice.
#
# **You will produce:** `artifacts/level3/nb05/metrics.json` (all models' test metrics,
# calibration and timings on your device).
#
# **Data:** [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) (`fancyzhx/ag_news`):
# 120k training and 7.6k test news snippets in 4 topics. Its card lists the licence as
# *unknown*; the original corpus page describes it as provided for research purposes:
# check the card before any reuse outside this course.
# **Model:** [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
# (Apache-2.0, 66M parameters), pinned to a revision.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only, with the `cpu`
# budget (1.5k training examples, 64 tokens, 1 epoch). That budget shows the *method*; it
# under-states what fine-tuning achieves. Numbers in **Expected:** lines were printed by a CPU
# run unless marked *"on a T4 you should see … (not measured in our verification runs)"*.

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
# --- Catch up: data and model (downloaded once, then cached) -----------------
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from scaleml import finetune as ft
from scaleml.device import count_parameters, seed_everything
from scaleml.grading import nb05 as grade
from scaleml.textdata import load_text_classification

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb05"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)

print("dataset card: https://huggingface.co/datasets/fancyzhx/ag_news (licence on the card: unknown; check it)")
print("model card:   https://huggingface.co/" + ft.DISTILBERT_ID)
t0 = time.time()
train_full, test_df, LABELS = load_text_classification("ag_news", n_train=BUDGET.train_examples,
                                                       n_eval=BUDGET.eval_examples, seed=0)
# Validation is carved from TRAIN (stratified). The test split is touched only for final numbers.
train_df, val_df = train_test_split(train_full, test_size=0.15, stratify=train_full["label"], random_state=0)
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
print(f"loaded in {time.time() - t0:.0f} s | labels: {LABELS}")
print(f"train {len(train_df)} | validation {len(val_df)} (from train) | test {len(test_df)} (from the test split)")
print("train label counts:", train_df["label"].value_counts().sort_index().tolist())
print("example:", train_df.loc[0, "text"][:120], "->", LABELS[train_df.loc[0, "label"]])

# %% [markdown]
# **Expected (CPU budget):** `train 1275 | validation 225 | test 500`, four balanced classes
# (`World`, `Sports`, `Business`, `Sci/Tech`; ~319 each in train). With the `gpu` budget:
# 17,000 / 3,000 / 4,000.
#
# **Why three splits.** Every decision you make by looking at a number (the learning rate,
# when to stop, the temperature, *which model to ship*) is fitted to the data that produced
# the number. Those decisions use **validation**, carved from the training data. The test split
# is read once, at the end, for the numbers you report. Choosing the epoch by test accuracy
# is the most common way a fine-tuning result is quietly inflated.
#
# ## 5.1 · Baseline 1: TF-IDF + logistic regression
#
# Word and bigram counts, re-weighted by inverse document frequency, into a linear
# classifier. It trains in seconds on a CPU, serves in microseconds, and is the number any
# deep model must beat by enough to pay for itself. We pick the regularisation strength `C`
# on **validation**.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

y_train, y_val, y_test = (d["label"].to_numpy() for d in (train_df, val_df, test_df))


def scores(y_true, probs):
    pred = probs.argmax(1)
    return {"accuracy": accuracy_score(y_true, pred), "macro_f1": f1_score(y_true, pred, average="macro")}


results, timings = {}, {}
t0 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train, X_val, X_test = tfidf.fit_transform(train_df["text"]), tfidf.transform(val_df["text"]), tfidf.transform(test_df["text"])
val_acc = {}
for C in [1.0, 3.0, 10.0, 30.0]:
    val_acc[C] = LogisticRegression(C=C, max_iter=2000).fit(X_train, y_train).score(X_val, y_val)
best_C = max(val_acc, key=val_acc.get)
tfidf_lr = LogisticRegression(C=best_C, max_iter=2000).fit(X_train, y_train)
timings["tfidf_lr_fit_s"] = time.time() - t0
probs = {"tfidf_lr": {"val": tfidf_lr.predict_proba(X_val), "test": tfidf_lr.predict_proba(X_test)}}
results["tfidf_lr"] = scores(y_test, probs["tfidf_lr"]["test"])
print(f"vocabulary {X_train.shape[1]:,} | validation accuracy by C: { {c: round(a, 3) for c, a in val_acc.items()} }")
print(f"TF-IDF + LR (C={best_C}): test {results['tfidf_lr']} | fit {timings['tfidf_lr_fit_s']:.1f} s")

# %% [markdown]
# **Expected (CPU budget, measured):** validation accuracy 0.86–0.87 for every `C` (we got
# `C=3.0`), a vocabulary of ~8k uni- and bigrams, and **0.822 test accuracy (macro-F1 0.821)** after
# under a second of fitting. With 17k training examples (`gpu` budget) TF-IDF + LR typically
# reaches ~0.89–0.90 on AG News.
#
# ## 5.2 · Baseline 2: a linear probe on frozen DistilBERT embeddings
#
# Keep the pretrained encoder **frozen**, embed every text once (mask-aware mean pooling,
# notebook 04), and train only a logistic regression on top. This tells you how much of
# the task the pretrained representation already solves. It is also a common production
# pattern: one shared embedding service, many cheap heads.

# %%
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV)
MAX_LEN = BUDGET.max_seq_len
enc = {name: ft.encode(tok, d["text"], MAX_LEN) for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]}
truncated = (enc["train"]["attention_mask"][:, -1] == 1).float().mean().item()
print(f"max_len {MAX_LEN}: {truncated:.0%} of training texts are truncated")

t0 = time.time()
encoder = AutoModel.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV, use_safetensors=True).to(DEVICE)
emb = {name: ft.frozen_embeddings(encoder, e, DEVICE, batch_size=2 * BUDGET.batch_size) for name, e in enc.items()}
timings["probe_embed_s"] = time.time() - t0
del encoder

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(emb["train"], y_train)
probs["probe"] = {"val": probe.predict_proba(emb["val"]), "test": probe.predict_proba(emb["test"])}
results["probe"] = scores(y_test, probs["probe"]["test"])
n_texts = sum(len(e["input_ids"]) for e in enc.values())
print(f"embedded {n_texts} texts in {timings['probe_embed_s']:.0f} s ({n_texts / timings['probe_embed_s']:.0f} texts/s)")
print(f"linear probe: validation acc {probe.score(emb['val'], y_val):.3f} | test {results['probe']}")

# %% [markdown]
# **Expected (CPU budget, measured):** 20 % of texts are truncated at 64 tokens; embedding
# 2,000 texts took 56 s (36 texts/s on 2 otherwise idle CPU threads; 148 s on a heavily shared machine); the probe reached 0.893 validation and
# **0.858 test accuracy (macro-F1 0.858)**: a frozen, never-fine-tuned encoder plus a linear
# layer already beats TF-IDF on this small training set, because the representation was
# learned from far more text than our 1,275 examples.
#
# ## 5.3 · Full fine-tuning
#
# `AutoModelForSequenceClassification` puts a freshly initialised head (`pre_classifier`
# → ReLU → dropout → `classifier`, on the `[CLS]` position for DistilBERT) on the pretrained
# encoder; the load report lists those weights as missing, which is expected. **Every**
# weight is then trained with the course's loop, `scaleml.train.train`
# ([`src/scaleml/train.py`](../../src/scaleml/train.py)):
#
# * **AdamW, lr 5e-5**: pretrained weights need small steps (1e-5 to 1e-4); a from-scratch
#   learning rate (1e-3) destroys what pre-training learned ("catastrophic forgetting");
# * **linear warmup (10 % of steps) then cosine decay**: the head starts random, and its
#   large early gradients would otherwise flow into the encoder at full learning rate;
# * **gradient clipping** at norm 1.0; **fp16 autocast + GradScaler on a CUDA GPU**
#   (a no-op on CPU);
# * **early stopping** on validation accuracy (`scaleml.finetune.EarlyStopping`): evaluate
#   every epoch, keep the best weights, stop after `patience` epochs without improvement.
#   With the CPU budget's single epoch it only records and restores; with the GPU budget
#   (2 epochs) it can act.

# %%
from scaleml.train import TrainConfig, train

seed_everything(0)
model = AutoModelForSequenceClassification.from_pretrained(
    ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV, num_labels=len(LABELS), use_safetensors=True)
print(f"{type(model).__name__}: {count_parameters(model, trainable_only=True) / 1e6:.1f}M trainable parameters")

batches = ft.TokenBatches(enc["train"], y_train, batch_size=BUDGET.batch_size, seed=0)
steps = len(batches) * BUDGET.epochs
cfg = TrainConfig(epochs=BUDGET.epochs, lr=5e-5, weight_decay=0.01, warmup_steps=max(1, steps // 10),
                  max_grad_norm=1.0, mixed_precision=True, log_every=max(1, len(batches) // 4))


def val_metric(m):
    acc = float((ft.predict_logits(m, enc["val"], DEVICE).argmax(1) == y_val).mean())
    return {"score": acc, "val_accuracy": acc}


stopper = ft.EarlyStopping(val_metric, patience=1)
result = train(model, stopper.wrap(batches), ft.classification_loss(DEVICE), DEVICE, cfg, evaluate=stopper)
stopper.restore(model)
timings["finetune_s"], timings["finetune_examples_per_s"] = result.seconds, result.examples_per_second
print(f"{result.steps} steps in {result.seconds:.0f} s ({result.examples_per_second:.1f} examples/s); "
      f"best epoch {stopper.best_epoch}, validation accuracy {stopper.best:.3f}")
for row in result.history:
    print({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()})

# %% [markdown]
# **Expected (CPU budget, measured):** 67.0M trainable parameters; 40 steps in ~135 s
# (~9 examples/s on 2 idle CPU threads; 443 s when other jobs shared the CPU: time is the least
# reproducible number in this notebook); the logged training loss falls from ~1.27 (first 10 steps) to
# ~0.46 (last 10) and validation accuracy after the single epoch is ~0.90. One epoch on 1,275
# short examples is far from converged: this is the regime where fine-tuning is *under-stated*.
#
# **On a T4 you should see** the `gpu` budget (17k examples, 128 tokens, 2 epochs, ~530
# steps) finish in a few minutes with fp16 autocast, with the loss falling well below 0.3
# and validation accuracy around 0.92–0.94 (not measured in our verification runs).

# %%
logits = {"val": ft.predict_logits(model, enc["val"], DEVICE), "test": ft.predict_logits(model, enc["test"], DEVICE)}
probs["finetuned"] = {split: ft.softmax(z) for split, z in logits.items()}
results["finetuned"] = scores(y_test, probs["finetuned"]["test"])

print(f"{'model':12} {'test acc':>8} {'macro-F1':>8}")
for name, r in results.items():
    print(f"{name:12} {r['accuracy']:8.3f} {r['macro_f1']:8.3f}")
n = len(y_test)
se = np.sqrt(0.9 * 0.1 / n)
print(f"(test set of {n}: a standard error of about ±{se:.3f} at 90 % accuracy, so differences under ~{2 * se:.2f} are noise)")

cm = confusion_matrix(y_test, probs["finetuned"]["test"].argmax(1))
print("\nfine-tuned confusion matrix (rows = true, columns = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

# %% [markdown]
# **Expected (CPU budget, measured):** TF-IDF + LR 0.822, linear probe 0.858, fine-tuned
# **0.872** test accuracy (macro-F1 equal to accuracy to 3 decimals: the classes are balanced). The largest
# off-diagonal cells are Business → Sci/Tech (19) and World → Business (13); Sports is almost
# never confused.
#
# **Read the comparison honestly.** The test set's standard error is about ±0.013 (500
# examples). The gap between TF-IDF and the two DistilBERT models (~0.04–0.05) is several
# standard errors; the gap between the probe and fine-tuning (0.014) is about one, so this CPU
# run cannot say that fine-tuning beats the probe (a paired test on the same examples, such as
# McNemar's, or a bootstrap, is the right tool). Fine-tuning's advantage grows with data and sequence length;
# with the `gpu` budget you should see the fine-tuned model clearly ahead
# (around 0.93–0.94 test accuracy vs ~0.88–0.90 for TF-IDF + LR and the probe), not measured
# in our verification runs (CPU only). The confusion matrix shows *where* errors sit: AG News
# mostly confuses **Business** with **Sci/Tech** (technology companies) and **World** with
# **Business** (economics and politics), while **Sports** is nearly separable.
#
# ## 5.4 · Calibration: do the probabilities mean anything?
#
# A classifier is **calibrated** if, among the examples where it says 0.9, about 90 % are
# right. Calibration matters whenever a probability is *used*: thresholds for automation vs
# human review, expected-value ranking, combining models, or abstaining when unsure.
#
# **Expected calibration error (ECE).** Split predictions into equal-width bins by
# confidence (the max probability), and average the gap between accuracy and mean
# confidence, weighted by bin size:
#
# $$\mathrm{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{n}\,\bigl|\,\mathrm{acc}(S_b) - \mathrm{conf}(S_b)\bigr|$$

# %% [markdown]
# > **Exercise 1.** Write `expected_calibration_error(probs, labels, n_bins)`: `probs` is an
# > `(n, C)` array of class probabilities, `labels` an `(n,)` array of true classes. Use
# > equal-width bins over confidence `(0, 1/B], (1/B, 2/B], …` (a confidence of exactly
# > 1.0 goes in the last bin), skip empty bins, and return a Python float.

# %%
def expected_calibration_error(probs, labels, n_bins=15):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(expected_calibration_error)

# %% [markdown]
# Now compare the three models' calibration on the test split, and look at the reliability
# table of the fine-tuned model (the data behind a reliability diagram).

# %%
N_BINS = 10
calib = {name: {"ece": ft.expected_calibration_error(p["test"], y_test, N_BINS),
                "mean_confidence": float(p["test"].max(1).mean()),
                "accuracy": results[name]["accuracy"]} for name, p in probs.items()}
print(f"{'model':12} {'accuracy':>8} {'mean conf':>9} {'ECE':>6}")
for name, c in calib.items():
    print(f"{name:12} {c['accuracy']:8.3f} {c['mean_confidence']:9.3f} {c['ece']:6.3f}")
print("\nfine-tuned reliability table (test):")
print(pd.DataFrame(ft.reliability_table(probs["finetuned"]["test"], y_test, N_BINS)).round(3).to_string(index=False))

# %% [markdown]
# **Expected (CPU budget, measured):** the three models are miscalibrated in
# *different directions*. TF-IDF + LR is strongly **under**-confident (mean confidence 0.62 vs
# accuracy 0.82, ECE 0.21: L2 regularisation shrinks its weights); the probe is slightly
# **over**-confident (0.93 vs 0.86, ECE 0.08); the briefly fine-tuned model is **under**-confident
# (0.75 vs 0.87, ECE 0.12): in the reliability table every bin from 0.7 up is more accurate than
# its confidence (e.g. confidence 0.83 → accuracy 0.98). With 500 test examples, ECE itself
# has a sampling error of a few hundredths.
#
# **Temperature scaling** (Guo et al., 2017) is the standard one-parameter fix: divide the
# logits by a scalar $T > 0$ before the softmax, choosing $T$ to minimise the negative
# log-likelihood on **validation** data. $T > 1$ softens over-confident predictions,
# $T < 1$ sharpens under-confident ones; the argmax, and therefore accuracy, never changes.

# %% [markdown]
# > **Exercise 2.** Write `fit_temperature(logits, labels)`: return the temperature `T > 0`
# > (a Python float) that minimises the mean NLL (cross-entropy) of `softmax(logits / T)`.
# > Hint: optimise `log T` with `torch.optim.LBFGS` (or any 1-D search); the grader compares
# > with a reference to within 2 %.

# %%
def fit_temperature(logits, labels):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(fit_temperature)

# %%
T = ft.fit_temperature(logits["val"], y_val)  # fitted on VALIDATION only
cal_test = ft.softmax(logits["test"], T)
temp = {"temperature": T,
        "test_nll_before": ft.nll(logits["test"], y_test), "test_nll_after": ft.nll(logits["test"], y_test, T),
        "test_ece_before": calib["finetuned"]["ece"],
        "test_ece_after": ft.expected_calibration_error(cal_test, y_test, N_BINS),
        "test_accuracy_after": float((cal_test.argmax(1) == y_test).mean())}
print(f"fitted T = {T:.2f} on {len(y_val)} validation examples")
print(f"test NLL {temp['test_nll_before']:.3f} -> {temp['test_nll_after']:.3f} | "
      f"test ECE {temp['test_ece_before']:.3f} -> {temp['test_ece_after']:.3f} | "
      f"accuracy {results['finetuned']['accuracy']:.3f} -> {temp['test_accuracy_after']:.3f} (unchanged)")

# %% [markdown]
# **Expected (CPU budget, measured):** fitted **T = 0.59** (< 1: sharpen, because this
# under-trained model is under-confident); test NLL 0.487 → 0.427 and test ECE 0.124 → 0.034,
# accuracy unchanged at 0.872.
#
# **On a T4 you should see** the opposite direction more often: a model fine-tuned for
# several epochs to near-zero training loss is typically **over**-confident (fitted $T > 1$),
# the classic finding of Guo et al. (not measured in our verification runs). Either way the
# recipe is the same: fit $T$ on validation, check ECE and NLL on test, and re-fit after
# every retrain.
#
# ## 5.5 · Error analysis
#
# Aggregate metrics say *how much*; error analysis says *what to do next*. Two views: the
# most frequent confusions (which class pairs, with examples), and the most confident
# mistakes, which are often label noise or genuinely ambiguous inputs rather than model failures.

# %%
pred = probs["finetuned"]["test"].argmax(1)
conf = cal_test.max(1)
errors = test_df.assign(pred=pred, conf=conf)[pred != y_test]
pairs = errors.groupby(["label", "pred"]).size().sort_values(ascending=False)
print(f"{len(errors)} errors out of {len(test_df)}. Top confusions (true -> predicted):")
for (t, p), count in pairs.head(3).items():
    print(f"\n{LABELS[t]} -> {LABELS[p]}: {count}")
    for text in errors[(errors.label == t) & (errors.pred == p)]["text"].head(2):
        print("   -", text[:150].replace("\n", " "))
print("\nMost confident mistakes (calibrated confidence):")
for _, row in errors.sort_values("conf", ascending=False).head(3).iterrows():
    print(f"  [{row.conf:.2f}] true {LABELS[row.label]}, predicted {LABELS[row.pred]}: {row.text[:120]}")

# %% [markdown]
# **Expected:** 64 errors out of 500 (CPU run); the top confusions are
# Business → Sci/Tech (Microsoft, Dell and Cisco stories), World → Business and Sci/Tech →
# Business. The most confident "mistakes" (calibrated confidence 0.97) are *Olympic swimming
# and cricket stories labelled World*: the model is right and the label is debatable or wrong. Read
# them as a reviewer would: many are articles that are *about* two topics (a company's
# earnings from a new chip is both Business and Sci/Tech), so the label is a judgement
# call. The fixes that follow from error analysis are rarely "a bigger model": clearer
# labelling guidelines, a multi-label formulation, more data for the confusable pair, or a
# "review" route for low-confidence predictions.

# %%
summary = {"device": describe(DEVICE), "budget": BUDGET.as_dict(), "results": results, "calibration": calib,
           "temperature_scaling": temp, "timings": timings, "confusion_matrix": cm.tolist(), "labels": LABELS}
(OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
print("wrote", OUT / "metrics.json")

# %% [markdown]
# > **Exercise 3.** Your team must ship a news-topic classifier for 50 million articles a day.
# > Using the numbers from this notebook (accuracy, training and inference time, model size)
# > and what you expect on a GPU, argue **when the TF-IDF + logistic regression baseline is the
# > right production choice** and when fine-tuning is worth it. Name at least four factors
# > (e.g. accuracy gap and its uncertainty, latency and cost per prediction, data volume and
# > drift, interpretability and debugging, retraining cadence, hardware) and the evidence you
# > would collect to decide. Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | Load report lists `pre_classifier` / `classifier` as **MISSING** | the classification head is new | expected; the head is trained during fine-tuning |
# | Loss stays at ~1.39 (= ln 4) | learning rate far too high or too low, or labels misaligned with texts | lr 2e-5–1e-4 with warmup; print a few (text, label) pairs from a batch |
# | Accuracy collapses to ~0.25 after a few steps | divergence (lr too high, no warmup) or fp16 overflow | lower the lr, add warmup; keep the GradScaler on a T4 |
# | `CUDA out of memory` | batch × sequence length too large | halve `batch_size`, use gradient accumulation (notebook 02), or shorten `max_len` |
# | Validation accuracy much higher than test | tuned on validation repeatedly, or validation leaked from test | carve validation from train; read test once |
# | Temperature fitted on test looks "perfect" | you fitted on the data you evaluate on | fit `T` on validation only |
# | CPU run is slow | DistilBERT on 2 CPU cores | expected; the `cpu` budget is the minimum that shows the method |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Linear probe vs full fine-tuning vs a TF-IDF baseline: what does each tell you, and how would you choose for a new text-classification product?
# 2. Why use a small learning rate with warmup when fine-tuning a pretrained transformer? What is catastrophic forgetting?
# 3. Where must the early-stopping and model-selection data come from, and what goes wrong if it is the test set?
# 4. What is ECE, what are its weaknesses (binning, class-wise vs top-label), and why does temperature scaling not change accuracy?
# 5. Your fine-tuned model is 1.5 points better than TF-IDF on a 500-example test set. Is that significant? How would you find out?
# 6. Error analysis shows most mistakes are Business ↔ Sci/Tech. What are your next three actions, in order?
#
# ➡️ **Next:** [`06_parameter_efficient_fine_tuning.ipynb`](06_parameter_efficient_fine_tuning.ipynb) — train 1 % of the parameters with LoRA and compare memory, time and accuracy against this full fine-tune.
