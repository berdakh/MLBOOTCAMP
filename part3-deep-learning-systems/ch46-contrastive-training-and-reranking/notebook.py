# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 46 · Contrastive training & re-ranking
#
# **Part III · Chapter 46 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Chapter 45 showed a general-purpose embedding model performing worse than BM25 on scientific claims.
# The two standard fixes are (1) **fine-tune the bi-encoder** on in-domain (query, relevant
# document) pairs with a contrastive loss, and (2) add a **cross-encoder re-ranker** that reads
# query and document together for the top candidates. Both are standard parts of production search
# and retrieval-augmented generation (RAG) systems, and both have costs that must be quantified.
#
# After this lab you should be able to:
#
# 1. write the InfoNCE loss with in-batch negatives and explain the role of the temperature;
# 2. fine-tune a bi-encoder on the benchmark's *train* queries and measure nDCG@10 before/after on the *test* queries;
# 3. mine hard negatives with BM25, and explain false negatives and how to limit them;
# 4. re-rank the top-50 with a public cross-encoder and build a quality-vs-latency table;
# 5. reason quantitatively about the two-stage (retrieve → re-rank) trade-off.
#
# **You will produce:** `artifacts/level3/nb08/two_stage_results.json` (metrics before/after
# training, re-ranking quality and latency on your device).
#
# **Data:** BEIR SciFact ([`BeIR/scifact`](https://huggingface.co/datasets/BeIR/scifact),
# [`BeIR/scifact-qrels`](https://huggingface.co/datasets/BeIR/scifact-qrels); licence on the
# cards: CC BY-SA 4.0). We train only on queries of the qrels **train** split and evaluate on the
# **test** split; the two query sets are disjoint. **Models:**
# [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
# (bi-encoder) and [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
# (re-ranker trained on MS MARCO), both Apache-2.0 and pinned.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. This notebook embeds the
# corpus three times (base model and two fine-tuned ones), so the CPU budget searches a
# **1,200-document sub-corpus** (every abstract relevant to a test query plus random others,
# the one Chapter 45 used to show BM25's inflation) with documents cut at 128 tokens. Fewer
# distractors make every score higher than on the full corpus: compare systems within this
# notebook only. The CPU budget also trains on a subset of the training pairs for one epoch and
# re-ranks for a subset of test queries; with a GPU the notebook uses the full corpus, all
# pairs and all test queries.
# Numbers in **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*.

# %% [markdown]
# Read first: [Chapter 46 text](README.md) explains the ideas behind this lab (Contrastive training & re-ranking) in more depth; [solutions](solutions.md) has worked answers.

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
# --- Catch up: benchmark, BM25 and the base bi-encoder (rebuilt from scratch) --
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from scaleml import retrieval as R
from scaleml.device import seed_everything, sync
from grading import ch46 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb08"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)
IS_CUDA = DEVICE.type == "cuda"

full_corpus, queries, qrels = R.load_scifact()
train_qrels, test_qrels = qrels["train"], qrels["test"]
N_DOCS = None if IS_CUDA else 1200  # CPU: a sub-corpus (see the note above)
DOC_LEN = 256 if IS_CUDA else 128
corpus = full_corpus if N_DOCS is None else R.subsample_corpus(full_corpus, test_qrels, N_DOCS, seed=0)
doc_ids = list(corpus)
doc_texts = [corpus[d] for d in doc_ids]
test_qids = list(test_qrels)
print("dataset cards: https://huggingface.co/datasets/BeIR/scifact | https://huggingface.co/datasets/BeIR/scifact-qrels"
      " (licence on the cards: cc-by-sa-4.0)")
for mid in (R.MINILM_ID, R.CROSS_ENCODER_ID):
    print("model card:", f"https://huggingface.co/{mid}")

# Leakage checks: train and test queries must be disjoint.
assert not set(train_qrels) & set(test_qrels), "train and test queries overlap"
shared_docs = {d for r in train_qrels.values() for d in r} & {d for r in test_qrels.values() for d in r}
print(f"train queries {len(train_qrels)} | test queries {len(test_qrels)} | overlap 0 | "
      f"relevant docs shared by a train and a test query: {len(shared_docs)}")
print(f"searching {len(doc_ids):,} of {len(full_corpus):,} documents, first {DOC_LEN} tokens")

bm25 = R.BM25Index([R.tokenize(t) for t in doc_texts])
runs = {"BM25": {q: [doc_ids[j] for j in bm25.search(R.tokenize(queries[q]), 100)[0]] for q in test_qids}}
tok = AutoTokenizer.from_pretrained(R.MINILM_ID, revision=R.MINILM_REV)


def fresh_encoder():
    model = AutoModel.from_pretrained(R.MINILM_ID, revision=R.MINILM_REV, use_safetensors=True)
    return R.Encoder(model=model, tokenizer=tok, device=DEVICE)


q_texts = [queries[q] for q in test_qids]


def dense_run(encoder):
    """Embed the corpus with ``encoder`` (the expensive part) and search exactly; returns (run, seconds)."""
    sync(DEVICE)
    t0 = time.time()
    d = encoder.encode(doc_texts, batch_size=BUDGET.batch_size, max_len=DOC_LEN)
    sync(DEVICE)
    seconds = time.time() - t0
    q = encoder.encode(q_texts, batch_size=BUDGET.batch_size, max_len=64)
    top = np.argsort(-(q @ d.T), axis=1, kind="stable")[:, :100]
    return {qid: [doc_ids[j] for j in row] for qid, row in zip(test_qids, top)}, seconds


runs["dense: base MiniLM"], embed_s = dense_run(fresh_encoder())
metrics = {name: R.evaluate_run(run, test_qrels) for name, run in runs.items()}
for name, m in metrics.items():
    print(f"{name:20} nDCG@10 {m['ndcg@10']:.3f} | Recall@100 {m['recall@100']:.3f}")
print(f"(embedding the corpus took {embed_s:.0f} s: paid again after every training run)")

# %% [markdown]
# **Expected (CPU, measured):** 300 test and 809 train queries with no overlap, but
# 181 relevant documents are shared between a train and a test query (claims about the same
# paper: not leakage of test *labels*, but a reason why in-domain training transfers). On the
# 1,200-document sub-corpus: BM25 nDCG@10 0.770 / Recall@100 0.937, base MiniLM 0.760 / 0.980;
# embedding the sub-corpus took ~21 s.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Contrastive learning | training by pulling matching pairs together and pushing other pairs apart | (claim, abstract) pairs in 8.2 |
# | Positive pair | a query and a document known to be relevant to it | every judged pair in the train qrels |
# | In-batch negatives | the other queries' positives in the same batch, used as negatives for free | 15 negatives per query at batch 16 |
# | Hard negative | a document a retriever ranks high that is not labelled relevant | BM25's top unlabelled hit, after `SKIP_TOP` |
# | False negative | a 'negative' that is actually relevant but was never judged | 15% of mined negatives are another claim's positive |
# | InfoNCE | the contrastive loss: cross-entropy over similarity scores, target = the true pair | Exercise 1 |
# | Temperature τ | divides the cosine similarities before the softmax; small τ = sharp | τ = 0.05 (scale 20) |
# | Cross-encoder | one transformer reads query and document together and outputs a score | `ms-marco-MiniLM-L-6-v2` in 8.4 |
# | Re-ranking depth | how many first-stage candidates the re-ranker scores | top 50 |
# | Bootstrap CI | a confidence interval from re-sampling the test queries many times | the 95% intervals in 8.3 |
# | Distillation | training a cheap model to imitate an expensive one's scores | cross-encoder → bi-encoder (8.5) |

# %% [markdown]
# ## 8.1 · Contrastive learning with in-batch negatives
#
# A bi-encoder is trained so that a query's embedding is closer to its relevant document than
# to other documents. With a batch of $B$ (query, positive) pairs, every *other* query's
# positive serves as a negative for free. **InfoNCE** is a $B$-way classification: for query
# $i$, pick its positive out of the $B$ documents,
#
# $$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathbf{q}_i \cdot \mathbf{d}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{q}_i \cdot \mathbf{d}_j / \tau)}$$
#
# with unit-norm embeddings (so $\mathbf{q}\cdot\mathbf{d}$ is a cosine in [−1, 1]) and a
# **temperature** $\tau$ (0.05 here, i.e. scale 20, as `all-MiniLM-L6-v2` was trained). Without
# it, logits in [−1, 1] give a softmax too flat to learn from; a small $\tau$ focuses the loss on
# the hardest negatives. Extra **hard negatives** are simply appended as rows $B+1 \ldots N$
# of the document matrix. Larger batches give more negatives, which is why contrastive
# training at scale uses batches of thousands (and tricks like GradCache).

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-infonce.svg" alt="A similarity matrix with four query rows and seven document columns: the diagonal of the first four columns holds the positives, the rest of them are in-batch negatives, and three extra columns hold hard negatives" width="100%">
#
# *What to notice:* every row is its own classification problem with one green target; the batch size sets how many grey distractors you get for free, the hard-negative columns make the distractors harder.
#
# > **Intuition.** InfoNCE is a multiple-choice test: for each claim, pick its abstract from all
# > the abstracts in the batch. A bigger batch means more options, a harder test and more to
# > learn from each step. *Where it breaks down:* when two options are both correct (a false
# > negative), the test marks the right answer wrong, and the model learns to push away a
# > relevant document.
#
# > **The temperature, in numbers.** Two candidates with cosines 0.8 (positive) and 0.3
# > (negative). At $\tau = 1$ the logits are 0.8 and 0.3 and the loss is
# > $\ln(1 + e^{-0.5}) = 0.47$: the model is still being pushed. At $\tau = 0.05$ the logits are
# > 16 and 6 and the loss is $\ln(1 + e^{-10}) \approx 5 \times 10^{-5}$: this pair is "solved",
# > and the gradient goes to the rows where the negatives are close.

# %% [markdown]
# > **Exercise 1.** Write `info_nce_loss(q, d, temperature)`: `q` is `(B, dim)`, `d` is `(N, dim)` with
# > `N ≥ B` (both already L2-normalised); `d[i]` is the positive for `q[i]` and every other row of
# > `d` is a negative. Return the mean cross-entropy as a scalar tensor (it must be differentiable).

# %%
def info_nce_loss(q, d, temperature=0.05):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(info_nce_loss)

# %% [markdown]
# ## 8.2 · Fine-tuning the bi-encoder on SciFact train queries
#
# Training pairs: (claim, relevant abstract) for every judged pair in the **train** qrels. The
# documents come from the full corpus; the test queries are never seen. We train with the
# course loop (`scaleml.train.train`): AdamW, lr 2e-5 with warmup and cosine decay, gradient
# clipping, fp16 autocast on a GPU.
#
# One trap of in-batch negatives: if two queries in a batch share a positive document, that
# document is a "negative" for one of them while being its answer. We avoid putting the same
# document twice in a batch.

# %%
from scaleml.train import TrainConfig, train

pairs = [(q, d) for q, rel in train_qrels.items() for d in rel if d in full_corpus]
random.Random(0).shuffle(pairs)
N_PAIRS = len(pairs) if IS_CUDA else 320
EPOCHS = 2 if IS_CUDA else 1
TRAIN_BS = 64 if IS_CUDA else 16
pairs = pairs[:N_PAIRS]


def make_batches(pairs, batch_size, hard_negs=None, seed=0):
    """Batches of {'q': [...], 'd': [...]} with no document repeated inside a batch."""
    rng = random.Random(seed)

    def gen():
        todo = pairs[:]
        rng.shuffle(todo)
        out = []
        while todo:
            batch, used, rest = [], set(), []
            for q, d in todo:
                negs = hard_negs.get(q, [])[:1] if hard_negs else []
                if len(batch) < batch_size and d not in used and not used & set(negs):
                    batch.append((q, d, negs))
                    used |= {d, *negs}
                else:
                    rest.append((q, d))
            todo = rest
            b = {"q": [queries[q] for q, _, _ in batch], "d": [full_corpus[d] for _, d, _ in batch]}
            b["d"] += [full_corpus[n] for _, _, negs in batch for n in negs]  # hard negatives after the positives
            out.append(b)
        return out

    batches = gen()
    return lambda: batches


def contrastive_loss(encoder, temperature=0.05):
    def loss_fn(model, batch):
        q = encoder.embed(batch["q"], max_len=64)
        d = encoder.embed(batch["d"], max_len=DOC_LEN)
        return R.info_nce_loss(q, d, temperature)
    return loss_fn


def finetune(batches):
    seed_everything(0)
    encoder = fresh_encoder()
    n_steps = len(batches()) * EPOCHS
    cfg = TrainConfig(epochs=EPOCHS, lr=2e-5, warmup_steps=max(1, n_steps // 10), max_grad_norm=1.0,
                      log_every=max(1, n_steps // 4))
    res = train(encoder.model, batches, contrastive_loss(encoder), DEVICE, cfg,
                batch_size_of=lambda b: len(b["q"]))
    print(f"{res.steps} steps in {res.seconds:.0f} s ({res.examples_per_second:.1f} pairs/s) | loss: "
          + " -> ".join(f"{h['loss']:.3f}" for h in res.history if "loss" in h))
    return encoder, res


inbatch = make_batches(pairs, TRAIN_BS)
print(f"{len(pairs)} training pairs from {len({q for q, _ in pairs})} train queries | {len(inbatch())} batches of <= {TRAIN_BS}")
enc_inbatch, res_inbatch = finetune(inbatch)
runs["dense: + in-batch FT"], _ = dense_run(enc_inbatch)
metrics["dense: + in-batch FT"] = R.evaluate_run(runs["dense: + in-batch FT"], test_qrels)
print("test:", metrics["dense: + in-batch FT"])
del enc_inbatch

# %% [markdown]
# **Expected (CPU, measured):** 320 pairs from 301 train queries in 20 batches of 16;
# 20 steps in 31–68 s (two runs; the slower one on a shared machine); the logged loss falls from ~0.42 to ~0.17 (the base model already separates
# most in-batch pairs: random negatives are easy); test nDCG@10 0.765, Recall@100 0.983.
#
# ## 8.3 · Hard negatives mined with BM25
#
# Random in-batch negatives are mostly *easy* (an abstract about astrophysics vs a claim about
# T cells): the loss is already near zero for them and they teach little. **Hard negatives**
# are documents a retriever ranks high but that are not labelled relevant. Mining them with
# BM25 adds the lexical near-misses that a semantic model must learn to tell apart.
#
# **False negatives.** "Not labelled relevant" is not "irrelevant": qrels are incomplete, and
# the documents a retriever ranks *highest* are the most likely to be unlabelled positives.
# Training to push them away teaches the model the wrong thing. Common mitigations: skip the
# top few ranks (`skip_top`), sample from a rank window (e.g. 10–100) instead of the top, drop
# candidates a cross-encoder scores as relevant ("denoised" hard negatives, as in RocketQA),
# or use only one hard negative per query.

# %% [markdown]
# > **Exercise 2.** Write `mine_hard_negatives(ranked_ids, positives, n, skip_top=0)`: skip the first
# > `skip_top` entries of `ranked_ids` (best first), then return the first `n` remaining ids that are
# > not in `positives`, in rank order (fewer if the list runs out).

# %%
def mine_hard_negatives(ranked_ids, positives, n, skip_top=0):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(mine_hard_negatives)

# %%
# Mine over the FULL corpus (cheap with BM25) for every training query.
full_ids = list(full_corpus)
bm25_full = R.BM25Index([R.tokenize(full_corpus[d]) for d in full_ids])
SKIP_TOP = 3
train_queries = sorted({q for q, _ in pairs})
hard_negs, top1_unlabelled = {}, []
for q in train_queries:
    ranked = [full_ids[j] for j in bm25_full.search(R.tokenize(queries[q]), 30)[0]]
    hard_negs[q] = R.mine_hard_negatives(ranked, train_qrels[q], n=1, skip_top=SKIP_TOP)
    top1_unlabelled.append(R.mine_hard_negatives(ranked, train_qrels[q], n=1)[0])
judged = {d for split in qrels.values() for rel in split.values() for d in rel}
print(f"mined 1 hard negative per query for {len(hard_negs)} train queries (skipping BM25's top {SKIP_TOP})")
print(f"the highest-ranked unlabelled document is relevant to some OTHER claim in {np.mean([d in judged for d in top1_unlabelled]):.0%}"
      " of queries")
q = train_queries[0]
print("\nexample claim:", queries[q])
print("  positive:     ", full_corpus[next(iter(train_qrels[q]))][:150], "...")
print("  hard negative:", full_corpus[hard_negs[q][0]][:150], "...")

hard = make_batches(pairs, TRAIN_BS, hard_negs=hard_negs)
print(f"\n{len(hard())} batches: {TRAIN_BS} positives + up to {TRAIN_BS} hard negatives each")
enc_hard, res_hard = finetune(hard)
runs["dense: + hard-neg FT"], _ = dense_run(enc_hard)
metrics["dense: + hard-neg FT"] = R.evaluate_run(runs["dense: + hard-neg FT"], test_qrels)
print("test:", metrics["dense: + hard-neg FT"])

# %% [markdown]
# **Expected (CPU, measured):** one hard negative for each of 301 train queries; for
# 15 % of queries BM25's highest-ranked *unlabelled* document is a judged positive of some other
# claim, i.e. topically on target, the false-negative risk that `SKIP_TOP` limits. Each batch
# now carries 16 extra documents, so a step is slower (20 steps in 51–91 s) and the loss is
# higher (~0.89 → ~0.48): hard negatives *are* harder. Test nDCG@10 0.760, Recall@100 0.977.

# %%
nd = {name: R.per_query_ndcg(run, test_qrels) for name, run in runs.items()}
rng = np.random.default_rng(0)
boot = rng.integers(0, len(test_qids), (2000, len(test_qids)))
print(f"{'system':22} {'nDCG@10':>8} {'Recall@100':>10}   change in nDCG@10 vs base (95% CI)")
for name, m in metrics.items():
    d = nd[name] - nd["dense: base MiniLM"]
    lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
    ci = "" if name == "dense: base MiniLM" else f"   {d.mean():+.3f} ({lo:+.3f} to {hi:+.3f})"
    print(f"{name:22} {m['ndcg@10']:8.3f} {m['recall@100']:10.3f}{ci}")

# %% [markdown]
# **Expected (CPU, measured):**
#
# | system | nDCG@10 | Recall@100 | Δ nDCG@10 vs base (95 % CI) |
# |---|---|---|---|
# | BM25 | 0.770 | 0.937 | +0.010 (−0.033 to +0.052) |
# | dense: base MiniLM | 0.760 | 0.980 | |
# | dense: + in-batch FT | 0.765 | 0.983 | +0.006 (−0.004 to +0.016) |
# | dense: + hard-neg FT | 0.760 | 0.977 | +0.000 (−0.011 to +0.012) |
#
# **On the CPU budget neither fine-tune measurably changed retrieval quality**; every
# interval includes zero. Twenty steps on 320 pairs with a small learning rate barely move a
# model pre-trained on a billion pairs, and the sub-corpus leaves little headroom. This is the
# result to expect from a budget this small, not evidence that contrastive fine-tuning does
# not work.
#
# **On a T4 you should see** (full corpus, all ~900 training pairs, 2 epochs, batch 64) a
# clearer gain from in-domain contrastive fine-tuning, typically a few nDCG@10 points over the
# base model, with hard negatives adding a further point or two when false negatives are
# controlled (not measured in our verification runs). Whatever the size of the gain, the
# workflow is the important part: train on train queries, measure on held-out queries, report
# a confidence interval over queries.
#
# ## 8.4 · Cross-encoder re-ranking
#
# A **cross-encoder** feeds `[CLS] query [SEP] document [SEP]` through one transformer and
# outputs a relevance score. Query and document tokens attend to each other in every layer,
# which is usually more accurate than comparing two independent vectors *when it was trained
# for the same notion of relevance*; but nothing can be
# precomputed, so it costs one forward pass **per (query, candidate) pair**. It can only
# re-rank a short list produced by a cheap first stage.
#
# First stage here: RRF fusion of BM25 and the hard-negative bi-encoder (chosen in advance,
# not by looking at test scores). We re-rank its top 50.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-bi-vs-cross-encoder.svg" alt="Left: a bi-encoder encodes query and document separately, documents offline once, and compares vectors by cosine; right: a cross-encoder reads the query and document together in one pass and outputs a score" width="100%">
#
# *What to notice:* the two arrows into the cosine box come from independent encoders, so the document side can be precomputed; the cross-encoder has a single input, so every (query, document) pair costs a full forward pass.
#
# > **Intuition.** A bi-encoder compares two CVs that were written separately; a cross-encoder
# > interviews the two candidates together, so it can notice how their answers relate. The
# > joint interview is more informative, but one cannot interview a million candidates per query.

# %%
ce_tok = AutoTokenizer.from_pretrained(R.CROSS_ENCODER_ID, revision=R.CROSS_ENCODER_REV)
ce = AutoModelForSequenceClassification.from_pretrained(R.CROSS_ENCODER_ID, revision=R.CROSS_ENCODER_REV,
                                                        use_safetensors=True).to(DEVICE).eval()
CE_LEN = 64 + DOC_LEN  # query + document tokens
first_stage = {q: R.reciprocal_rank_fusion([runs["BM25"][q], runs["dense: + hard-neg FT"][q]])[:100] for q in test_qids}
N_RERANK = len(test_qids) if IS_CUDA else 30
rerank_qids = test_qids[:N_RERANK]
sub_qrels = {q: test_qrels[q] for q in rerank_qids}
DEPTH = 50

ce_scores, ce_seconds = {}, 0.0
for q in rerank_qids:
    cands = first_stage[q][:DEPTH]
    sync(DEVICE)
    t0 = time.perf_counter()
    ce_scores[q] = R.cross_encoder_scores(ce, ce_tok, queries[q], [corpus[d] for d in cands], DEVICE, max_len=CE_LEN)
    sync(DEVICE)
    ce_seconds += time.perf_counter() - t0
ms_per_pair = ce_seconds / (N_RERANK * DEPTH) * 1e3


def reranked(q, depth):
    cands = first_stage[q]
    order = np.argsort(-ce_scores[q][:depth], kind="stable")
    return [cands[i] for i in order] + cands[depth:]


# Latency per query, measured directly at each depth on a few queries (batch of `depth` pairs).
timing_q = rerank_qids[:5]
rows = [{"depth": 0, **R.evaluate_run({q: first_stage[q] for q in rerank_qids}, sub_qrels), "ms_per_query": 0.0}]
for depth in (10, 20, 50):
    sync(DEVICE)
    t0 = time.perf_counter()
    for q in timing_q:
        R.cross_encoder_scores(ce, ce_tok, queries[q], [corpus[d] for d in first_stage[q][:depth]], DEVICE,
                               max_len=CE_LEN, batch_size=depth)
    sync(DEVICE)
    ms = (time.perf_counter() - t0) / len(timing_q) * 1e3
    rows.append({"depth": depth, **R.evaluate_run({q: reranked(q, depth) for q in rerank_qids}, sub_qrels),
                 "ms_per_query": ms})
ceiling = np.mean([R.recall_at_k(first_stage[q], test_qrels[q], DEPTH) for q in rerank_qids])
print(f"re-ranked {N_RERANK} test queries x top-{DEPTH}: {ms_per_pair:.1f} ms per (query, doc) pair on {DEVICE.type}")
print(f"first-stage Recall@{DEPTH} on these queries (the re-ranker's ceiling): {ceiling:.3f}")
print(f"{'re-rank depth':>13} {'nDCG@10':>8} {'ms/query':>9}")
for r in rows:
    print(f"{r['depth']:>13} {r['ndcg@10']:8.3f} {r['ms_per_query']:9.0f}")

# %% [markdown]
# **Expected (CPU, measured):** 24–41 ms per (query, document) pair on 2 CPU
# threads (two runs; the machine was shared in the second); the first stage already has every relevant document in its top 50 for these 30
# queries (ceiling 1.000), and:
#
# | re-rank depth | nDCG@10 | ms/query |
# |---|---|---|
# | 0 (first stage only) | 0.840 | 0 |
# | 10 | 0.773 | 250–640 |
# | 20 | 0.768 | 480–850 |
# | 50 | 0.757 | 1,230–1,910 |
#
# **The re-ranker made the ranking worse**, and the deeper it re-ranks, the worse (with only 30
# queries, each nDCG value has a standard error of several points, but the direction was the
# same in a separate check: re-ranking BM25's top 20 on the full corpus for 60 test queries with
# 192 or 512 tokens also lowered nDCG@10, 0.764 → 0.727 / 0.743). We verified the model on the
# standard sanity pair from its card (a relevant passage scores +8.8, an irrelevant one −11.2),
# so this is not a bug. `ms-marco-MiniLM-L-6-v2` was trained on web search questions and
# passages; SciFact pairs scientific *claims* with abstracts that support **or refute** them,
# a different notion of relevance. An off-the-shelf re-ranker is a hypothesis to test on your
# own judged queries, not a guaranteed gain; fine-tuning it on in-domain pairs (the train qrels,
# exactly as in 8.2) is the usual fix. Latency grows linearly with depth.
#
# **On a T4 you should see** the re-ranker score a 50-candidate batch in a few tens of
# milliseconds with fp16 (well under 1 ms per pair), i.e. 20–50× faster than our CPU; the quality
# question is the same and must be measured on all 300 test queries (not measured in our
# verification runs).
#
# ## 8.5 · The two-stage trade-off
#
# | | Bi-encoder (stage 1) | Cross-encoder (stage 2) |
# |---|---|---|
# | Input | query, document separately | the (query, document) pair together |
# | Precompute | all document vectors, offline | nothing |
# | Cost per query | 1 encoder call + ANN search (~ms), independent of corpus size | depth × one forward pass |
# | Quality | good recall, coarse ordering | best ordering of a short list, if trained in-domain |
# | Retraining cost | re-embed the whole corpus | none for the corpus |
#
# The re-ranking depth is the dial: for a re-ranker that helps (measure it, as 8.4 shows),
# quality rises with depth only while the first stage still has relevant documents to surface
# (its Recall@depth is the ceiling), and latency rises linearly. Stated with numbers: *at a 100 ms p99 budget with ~1 ms per
# pair on a GPU, a 50-deep re-rank fits; at 1,000 QPS that is 50,000 pair inferences per
# second, so the re-ranker's GPU fleet, not the ANN index, dominates cost*. Distilling the
# cross-encoder into the bi-encoder (training stage 1 on stage 2's scores) is the usual way to
# recover some of that quality without the latency.

# %%
summary = {"device": describe(DEVICE), "n_docs": len(doc_ids), "doc_len": DOC_LEN, "n_train_pairs": len(pairs),
           "epochs": EPOCHS, "batch_size": TRAIN_BS, "metrics": metrics,
           "train_seconds": {"in_batch": res_inbatch.seconds, "hard_negatives": res_hard.seconds},
           "corpus_embed_seconds": embed_s, "rerank": {"queries": N_RERANK, "depth": DEPTH,
                                                       "ms_per_pair": ms_per_pair, "rows": rows,
                                                       "first_stage_recall@50": ceiling}}
(OUT / "two_stage_results.json").write_text(json.dumps(summary, indent=2, default=float))
print("wrote", OUT / "two_stage_results.json")

# %% [markdown]
# > **Exercise 3.** Your RAG assistant retrieves 5 passages from 20M internal documents for an LLM.
# > Users complain that answers miss the right document. Using this notebook's measurements, lay
# > out how you would decide between (a) fine-tuning the bi-encoder on logged (query, clicked
# > document) pairs, (b) adding a cross-encoder re-ranker and (c) a hybrid first stage. For each:
# > the expected effect on Recall@k and nDCG, the latency and cost, the risk (false negatives in
# > click logs, position bias, re-embedding 20M documents), and how you would evaluate it offline
# > and online. Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1. InfoNCE with in-batch and hard negatives.**
#
# $$
# \mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathbf q_i \cdot \mathbf d_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf q_i \cdot \mathbf d_j / \tau)}
# $$
#
# In words: for every query, a softmax over all $N$ documents in the batch (its positive, the
# other $B - 1$ positives and $N - B$ hard negatives) and the cross-entropy of picking its own.
# Symbols: unit vectors $\mathbf q_i, \mathbf d_j$; $\tau$ temperature. Example: cosines 0.8
# (positive) and 0.3 (one negative): loss 0.47 at $\tau = 1$, $5 \times 10^{-5}$ at $\tau = 0.05$.
#
# **2. A random-guess baseline for the loss.**
#
# $$
# \mathcal{L}_{\text{uniform}} = \ln N
# $$
#
# Example: batch 16 with 16 hard negatives ($N = 32$): $\ln 32 = 3.47$. Our logged losses of
# 0.2–0.9 show the base model already separates most pairs.
#
# **3. Bootstrap confidence interval for a metric difference.**
#
# $$
# \Delta^{\star(b)} = \frac{1}{n} \sum_{i \in \text{sample } b} \big(m_A(i) - m_B(i)\big), \qquad
# \text{CI}_{95\%} = \big[\, \text{2.5th},\; \text{97.5th percentile of } \Delta^{\star} \big]
# $$
#
# In words: resample the $n$ test queries with replacement many times, recompute the paired
# difference each time, and read off the middle 95%. If the interval contains 0, the change is
# not established. Example: 2,000 resamples of 300 queries in 8.3.
#
# **4. Re-ranking cost.**
#
# $$
# \text{latency} \approx \text{depth} \times t_{\text{pair}} / \text{parallelism}, \qquad \text{pair inferences/s} = \text{QPS} \times \text{depth}
# $$
#
# Example: depth 50, 1,000 QPS: 50,000 cross-encoder passes per second, whatever the ANN index
# costs.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Loss stuck at ln(N) | embeddings not normalised with a tiny temperature, or labels misaligned (`d[i]` not the positive of `q[i]`) | normalise; target = `arange(B)`; print a batch |
# | nDCG drops after fine-tuning | too high a learning rate (forgetting), too many steps on few pairs, or false negatives | lr 1e-5–3e-5; fewer epochs; skip the top mined ranks |
# | Duplicate positives in a batch | several queries share a document | build batches without repeated documents (as here) |
# | Test score improves suspiciously | trained on test queries or tuned on test | train qrels only; choose settings on a held-out part of train |
# | Cross-encoder scores look random | wrong input format (query and doc concatenated as one string) or wrong head | pass `(query, doc)` as a text pair; use the model's own head |
# | Re-ranking is too slow | depth × pairs on CPU, long documents | GPU + fp16; lower depth; truncate documents; distil into the bi-encoder |
# | `CUDA out of memory` while training | batch of 64 + hard negatives at 256 tokens | halve the batch or `DOC_LEN`; gradient checkpointing |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. Write down the InfoNCE loss. What does the temperature do, and why do larger batches help contrastive training?
# 2. What are hard negatives, how do you mine them, and what are false negatives? Give two mitigations.
# 3. Why can a cross-encoder not be used as a first-stage retriever? How do you pick the re-ranking depth?
# 4. You fine-tuned the bi-encoder and offline nDCG@10 went up 3 points. What else must happen before you deploy it (re-indexing, A/B test, latency, freshness of embeddings, backward compatibility of vectors)?
# 5. How would you build training data for a retriever from search logs, and what biases does it carry?
# 6. Design: a two-stage retrieval system for 100M documents at 1,000 QPS with a 150 ms p99 budget. Allocate the budget across stages and justify the numbers.
#
# **Next chapter:** [Chapter 47 · Distributed data parallel](../ch47-distributed-data-parallel/notebook.ipynb) — the first of two chapters on distributed training: train on several processes with `torch.distributed`, by hand.
