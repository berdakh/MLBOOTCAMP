# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 45 · Embeddings & semantic search
#
# **Part III · Chapter 45 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# Retrieval is the first stage of search, recommendations, question answering and every
# retrieval-augmented LLM: it decides which few hundred items out of millions the expensive
# models ever see. This lab builds the three standard first stages, **lexical (BM25)**,
# **dense (a bi-encoder + FAISS)** and **hybrid (rank fusion)**, and evaluates them on a
# public benchmark with relevance judgements, so that the claim "semantic search is better"
# can be checked with a number.
#
# After this lab you should be able to:
#
# 1. implement BM25 from its formula and as a sparse-matrix index;
# 2. embed text with a sentence encoder from plain `transformers` (mean pooling + normalisation) and measure throughput;
# 3. search with FAISS: exact (`IndexFlatIP`) vs approximate (HNSW), and measure the recall/latency trade-off;
# 4. evaluate retrieval with nDCG@10 and Recall@100 against qrels;
# 5. fuse lexical and dense rankings with reciprocal rank fusion and state accurately which system is better, and where.
#
# **You will produce:** `artifacts/level3/nb07/retrieval_results.json` (metrics, latencies and
# throughput on your device).
#
# **Data:** BEIR **SciFact** ([`BeIR/scifact`](https://huggingface.co/datasets/BeIR/scifact),
# [`BeIR/scifact-qrels`](https://huggingface.co/datasets/BeIR/scifact-qrels)): 5,183 scientific
# abstracts, 1,109 claims as queries, relevance judgements (qrels) split into 809 train and 300
# test queries. Licence on both cards: **CC BY-SA 4.0** (attribution, share-alike).
# **Model:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
# (Apache-2.0; 22.7M parameters, 384-dimensional embeddings), loaded with `transformers`
# only. Both pinned to a revision.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. The CPU budget embeds only
# the first **128 tokens** of each abstract (most abstracts are longer: on a CPU, 256 tokens
# would take several minutes); with a GPU the notebook uses 256 tokens. Both search the full
# 5,183-document corpus. Numbers in **Expected:** lines were printed
# by a CPU run unless marked *"on a T4 you should see … (not measured in our verification runs)"*.

# %% [markdown]
# Read first: [Chapter 45 text](README.md) explains the ideas behind this lab (Embeddings & semantic search) in more depth; [solutions](solutions.md) has worked answers.

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
# --- Catch up: the benchmark and the encoder (downloaded once, then cached) ---
import json
import time

import numpy as np
import torch

from scaleml import retrieval as R
from scaleml.device import count_parameters, seed_everything, sync
from grading import ch45 as grade

try:
    import faiss
except ImportError:  # Colab does not ship FAISS
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "faiss-cpu"], check=True)
    import faiss

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    faiss.omp_set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb07"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)
IS_CUDA = DEVICE.type == "cuda"

t0 = time.time()
full_corpus, queries, qrels = R.load_scifact()
test_qrels = qrels["test"]
DOC_LEN = 256 if IS_CUDA else 128  # tokens per document embedded
corpus = full_corpus
doc_ids = list(corpus)
doc_texts = [corpus[d] for d in doc_ids]
test_qids = list(test_qrels)
print("dataset cards: https://huggingface.co/datasets/BeIR/scifact | https://huggingface.co/datasets/BeIR/scifact-qrels"
      " (licence on the cards: cc-by-sa-4.0)")
print(f"loaded in {time.time() - t0:.0f} s | corpus {len(corpus):,} docs | "
      f"queries {len(queries):,} | qrels: train {len(qrels['train'])} queries, test {len(test_qrels)} queries")
print(f"relevant docs per test query: mean {np.mean([len(v) for v in test_qrels.values()]):.2f}, "
      f"max {max(len(v) for v in test_qrels.values())}")
q0 = test_qids[0]
print("\nexample claim:", queries[q0])
print("relevant abstract:", corpus[next(iter(test_qrels[q0]))][:200], "...")

# %% [markdown]
# **Expected:** 5,183 documents, 1,109 queries, qrels for
# 809 train and 300 test queries, ~1.1 relevant documents per test query (max 5). Queries are
# short scientific *claims*; documents are abstracts that support or refute them, often with
# different wording ("aspirin" vs "acetylsalicylic acid"), which is what makes this hard for
# purely lexical matching, and specialised vocabulary is what makes it hard for a general
# embedding model.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Corpus / qrels | the documents to search / the judged (query, relevant document) pairs | 5,183 abstracts; ~1.1 relevant per test query |
# | Lexical retrieval | matching the exact words of the query | BM25 |
# | idf | inverse document frequency: rare terms count more than common ones | "prion" vs "protein" |
# | BM25 | a word-matching score with saturating term counts and length normalisation | nDCG@10 0.667 on SciFact |
# | Dense retrieval / bi-encoder | embed queries and documents separately into vectors, search by similarity | `all-MiniLM-L6-v2`, 384 dimensions |
# | L2 normalisation | scaling each vector to length 1, so dot product = cosine | all norms 1.0000 in 7.3 |
# | ANN | approximate nearest-neighbour search: trade a little recall for much less latency | HNSW, recall@10 0.975 at efSearch 64 |
# | HNSW | a layered graph of vectors that a query walks greedily toward its neighbours | `IndexHNSWFlat`, `M`, `efSearch` |
# | nDCG@10 | how well the top 10 is ordered, 1.0 = every relevant document on top | BEIR's headline metric |
# | Recall@100 | the share of relevant documents found anywhere in the top 100 | what a re-ranker can at best recover |
# | RRF | reciprocal rank fusion: combine rankings by summing 1 / (k + rank) | the hybrid in 7.6 |
# | Hybrid retrieval | combining lexical and dense results | best on both metrics here |

# %% [markdown]
# ## 7.1 · Retrieval metrics
#
# For each query, a system returns a ranked list; the **qrels** say which documents are
# relevant (here a grade of 1; graded qrels use 0–3).
#
# * **nDCG@10**: discounted cumulative gain of the top 10, $\sum_{i=1}^{10} \frac{rel_i}{\log_2(i+1)}$,
#   divided by the best possible value for that query. 1.0 means every relevant document is at
#   the top, in order of relevance. This is BEIR's headline metric: it rewards putting the
#   answer *first*, which is what a user (or an LLM context window) sees.
# * **Recall@100**: the fraction of relevant documents anywhere in the top 100. This is what
#   matters for a **first stage** whose output goes to a re-ranker (Chapter 46): a document
#   that is not retrieved cannot be re-ranked up.
#
# Unjudged documents count as non-relevant, so these metrics are lower bounds; Chapter 46
# returns to that ("false negatives").
#
# ## 7.2 · Lexical retrieval: BM25
#
# BM25 scores a document by the query terms it contains, with two corrections over raw
# counts: **term-frequency saturation** (the 10th "protein" adds less than the 1st, controlled
# by $k_1$) and **length normalisation** (long documents match more words by chance,
# controlled by $b$):
#
# $$\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{idf}(t)\,\frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgdl}}\right)}, \qquad \mathrm{idf}(t) = \ln\!\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right)$$
#
# where $f(t,d)$ is the count of $t$ in $d$, $|d|$ the document length in tokens, avgdl the
# average length, $N$ the number of documents and $n_t$ how many contain $t$.

# %% [markdown]
# > **Exercise 1.** Write `bm25_scores(query_tokens, doc_tokens_list, k1=1.2, b=0.75)`: return a list
# > with one BM25 score per document, using the formula above (Lucene's idf). Sum over the
# > **distinct** query terms; a term absent from a document contributes 0; terms absent from the
# > whole collection contribute 0.

# %%
def bm25_scores(query_tokens, doc_tokens_list, k1=1.2, b=0.75):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(bm25_scores)

# %% [markdown]
# Scoring every document in a Python loop is fine for a test, not for a corpus. The course
# version (`scaleml.retrieval.BM25Index`, [`src/scaleml/retrieval.py`](../../src/scaleml/retrieval.py))
# precomputes every (term, document) weight once into a sparse matrix, so a query costs one
# sparse row-slice and a sum: an inverted index in miniature. Our tokenizer is deliberately
# simple (lower-case alphanumerics minus stopwords, no stemming).

# %%
def run_from_scores(score_matrix, k=100):
    """{query id: [doc ids, best first]} from a (queries x docs) score matrix."""
    top = np.argsort(-score_matrix, axis=1, kind="stable")[:, :k]
    return {q: [doc_ids[j] for j in row] for q, row in zip(test_qids, top)}


t0 = time.time()
doc_tokens = [R.tokenize(t) for t in doc_texts]
bm25 = R.BM25Index(doc_tokens)
index_s = time.time() - t0
t0 = time.time()
bm25_scores_m = np.stack([bm25.scores(R.tokenize(queries[q])) for q in test_qids])
bm25_query_ms = (time.time() - t0) / len(test_qids) * 1e3
runs = {"BM25": run_from_scores(bm25_scores_m)}
metrics = {"BM25": R.evaluate_run(runs["BM25"], test_qrels)}
print(f"BM25 index: {bm25.matrix.shape[0]:,} terms x {bm25.matrix.shape[1]:,} docs, {bm25.matrix.nnz:,} non-zeros, "
      f"built in {index_s:.1f} s | {bm25_query_ms:.2f} ms/query")
print("BM25 on this corpus:", metrics["BM25"])
# A smaller corpus (every abstract relevant to a test query + random others) has fewer distractors:
sub = R.subsample_corpus(full_corpus, test_qrels, 1200, seed=0)
sub_ids = list(sub)
bm25_sub = R.BM25Index([R.tokenize(sub[d]) for d in sub_ids])
run_sub = {q: [sub_ids[j] for j in bm25_sub.search(R.tokenize(queries[q]), 100)[0]] for q in test_qids}
print("BM25 on a 1,200-doc sub-corpus:", R.evaluate_run(run_sub, test_qrels))

# %% [markdown]
# **Expected (CPU, measured):** a vocabulary of ~36k terms, the index built in
# under a second, ~0.2 ms per query; **nDCG@10 0.667, Recall@100 0.886** on the full corpus,
# close to the published BEIR BM25 figure for SciFact (0.665), although our tokenizer is
# simpler. The same BM25 on the 1,200-document sub-corpus scores 0.770 / 0.937: removing
# distractors inflates the metric by 10 nDCG points. Never compare numbers across corpora.
#
# ## 7.3 · Dense retrieval: a bi-encoder
#
# A **bi-encoder** maps queries and documents *independently* to vectors, so documents are
# embedded once, offline, and a query costs one encoder call plus a nearest-neighbour search.
# `all-MiniLM-L6-v2` was trained contrastively on ~1B sentence pairs to put related texts
# close in cosine similarity. Its `sentence-transformers` pipeline is just: transformer →
# mask-aware **mean pooling** (Chapter 42) → **L2 normalisation**. We rebuild it with
# `AutoModel` (`scaleml.retrieval.Encoder`), so nothing is hidden and there is one less
# dependency. With unit vectors, inner product = cosine similarity.

# %%
enc = R.Encoder(R.MINILM_ID, R.MINILM_REV, device=DEVICE)
print(f"model card: https://huggingface.co/{R.MINILM_ID} | {count_parameters(enc.model) / 1e6:.1f}M parameters, "
      f"dim {enc.model.config.hidden_size}")
n_tok = [len(x) for x in enc.tokenizer(doc_texts, truncation=True, max_length=DOC_LEN)["input_ids"]]
sync(DEVICE)
t0 = time.time()
doc_emb = enc.encode(doc_texts, batch_size=BUDGET.batch_size, max_len=DOC_LEN)
sync(DEVICE)
embed_s = time.time() - t0
t0 = time.time()
q_emb = enc.encode([queries[q] for q in test_qids], batch_size=BUDGET.batch_size, max_len=64)
q_embed_ms = (time.time() - t0) / len(test_qids) * 1e3
throughput = {"docs_per_s": len(doc_texts) / embed_s, "tokens_per_s": sum(n_tok) / embed_s,
              "doc_len": DOC_LEN, "query_ms_batched": q_embed_ms}
print(f"embedded {len(doc_texts):,} docs (<= {DOC_LEN} tokens) in {embed_s:.0f} s: "
      f"{throughput['docs_per_s']:.0f} docs/s, {throughput['tokens_per_s']:,.0f} tokens/s | "
      f"queries {q_embed_ms:.1f} ms each (batched)")
print(f"norms: {np.linalg.norm(doc_emb, axis=1).min():.4f}-{np.linalg.norm(doc_emb, axis=1).max():.4f}")
print(f"{np.mean(np.array(n_tok) >= DOC_LEN):.0%} of documents were truncated at {DOC_LEN} tokens")

# %% [markdown]
# **Expected (CPU, measured):** 22.7M parameters, 384 dimensions; 5,183 documents
# embedded in about 2 minutes (41–48 docs/s, 5,200–6,100 tokens/s on 2 CPU threads in two runs; 98 % of abstracts were
# cut at 128 tokens); a query costs ~4 ms (batched); all norms 1.0000.
#
# **On a T4 you should see** the full 5,183-document corpus at 256 tokens embedded in roughly
# 10–30 seconds with fp16 autocast, i.e. a few hundred to a thousand documents/s (not measured
# in our verification runs). Embedding is the offline cost of dense retrieval and is paid
# again for the *whole corpus* every time the encoder changes (Chapter 46).
#
# ## 7.4 · Exact vs approximate nearest neighbours with FAISS
#
# * **`IndexFlatIP`**: exact; compares the query with every vector ($O(N d)$ per query). Perfect
#   recall; fine up to a few million vectors on a GPU, or when latency is not critical.
# * **HNSW** (`IndexHNSWFlat`): a multi-layer proximity graph; a query greedily walks the graph
#   ($\approx O(\log N)$). `M` (links per node) trades memory for recall; `efSearch` (candidate
#   list size) trades latency for recall at query time. No training step; the graph is built
#   as vectors are added.
#
# On our small corpus both are instant, so we also time them on a larger synthetic collection
# (our embeddings plus noise, re-normalised).

# %%
def time_search(index, xq, k=10, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        _, ids = index.search(xq, k)
        best = min(best, time.perf_counter() - t0)
    return ids, best / len(xq) * 1e3


flat = R.faiss_flat_index(doc_emb)
hnsw = R.faiss_hnsw_index(doc_emb, m=32)
ids_flat, ms_flat = time_search(flat, q_emb, 100)
ids_hnsw, ms_hnsw = time_search(hnsw, q_emb, 100)
overlap = np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(ids_flat, ids_hnsw)])
runs["dense (MiniLM)"] = {q: [doc_ids[j] for j in row] for q, row in zip(test_qids, ids_flat)}
metrics["dense (MiniLM)"] = R.evaluate_run(runs["dense (MiniLM)"], test_qrels)
metrics["dense, HNSW"] = R.evaluate_run({q: [doc_ids[j] for j in row] for q, row in zip(test_qids, ids_hnsw)}, test_qrels)
print(f"real corpus ({len(doc_ids):,}): flat {ms_flat:.3f} ms/query | HNSW {ms_hnsw:.3f} ms/query | "
      f"HNSW top-10 overlap with exact: {overlap:.3f}")
print("dense, exact:", metrics["dense (MiniLM)"])
print("dense, HNSW: ", metrics["dense, HNSW"])

# A larger synthetic collection: perturbed copies of the real embeddings.
N_SYN = 200_000 if IS_CUDA else 50_000
rng = np.random.default_rng(0)
syn = doc_emb[rng.integers(0, len(doc_emb), N_SYN)] + 0.05 * rng.standard_normal((N_SYN, doc_emb.shape[1])).astype(np.float32)
syn /= np.linalg.norm(syn, axis=1, keepdims=True)
t0 = time.time()
flat_syn = R.faiss_flat_index(syn)
flat_build = time.time() - t0
t0 = time.time()
hnsw_syn = R.faiss_hnsw_index(syn, m=32, ef_construction=64)
hnsw_build = time.time() - t0
exact_ids, ms_exact = time_search(flat_syn, q_emb, 10)
ann_rows = []
for ef in (16, 64, 256):
    hnsw_syn.hnsw.efSearch = ef
    ids, ms = time_search(hnsw_syn, q_emb, 10)
    rec = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(exact_ids, ids)])
    ann_rows.append({"efSearch": ef, "ms_per_query": ms, "recall@10_vs_exact": rec})
print(f"\nsynthetic {N_SYN:,} x {syn.shape[1]}: build flat {flat_build:.1f} s, HNSW {hnsw_build:.1f} s | "
      f"exact search {ms_exact:.2f} ms/query")
for r in ann_rows:
    print(f"  HNSW efSearch={r['efSearch']:>3}: {r['ms_per_query']:.3f} ms/query, recall@10 vs exact {r['recall@10_vs_exact']:.3f}")

# %% [markdown]
# **Expected (CPU, measured):** on the real corpus both searches take well under 1 ms per
# query (exact 0.14 ms, HNSW 0.06 ms) and HNSW's top 10 agrees with exact search 99 % of the time,
# with the same nDCG@10 to two decimals (0.622 vs 0.619). On 50,000 synthetic vectors the HNSW
# graph took ~6 s to build (a flat index: nothing to build); exact search costs 1.6–2.0 ms per query,
# while HNSW costs 0.04 / 0.10 / 0.33 ms at `efSearch` 16 / 64 / 256 with recall@10 against exact
# search of 0.88 / 0.975 / 0.996 (latencies vary ±20 % between runs). Exact search grows linearly with N; HNSW grows slowly.
#
# The general lesson: an ANN index is a *recall vs latency vs memory* dial, and
# you must measure its recall against exact search on your own data. HNSW costs memory (the
# graph adds ~`2·M` neighbour ids per vector) and build time; IVF-PQ indexes compress vectors
# (product quantisation) for billion-scale collections at a further recall cost.
#
# ## 7.5 · Lexical vs dense: where each is better
#
# Averages hide structure. Compare the two systems query by query.

# %%
nd_bm25 = R.per_query_ndcg(runs["BM25"], test_qrels)
nd_dense = R.per_query_ndcg(runs["dense (MiniLM)"], test_qrels)
print(f"per-query nDCG@10: BM25 better on {(nd_bm25 > nd_dense).sum()} queries, dense better on "
      f"{(nd_dense > nd_bm25).sum()}, tied on {(nd_bm25 == nd_dense).sum()} (of {len(test_qids)})")
for name, i in [("BM25 wins most", int(np.argmax(nd_bm25 - nd_dense))), ("dense wins most", int(np.argmax(nd_dense - nd_bm25)))]:
    q = test_qids[i]
    print(f"\n{name}: nDCG@10 BM25 {nd_bm25[i]:.2f} vs dense {nd_dense[i]:.2f}")
    print("  claim:   ", queries[q])
    print("  relevant:", corpus[next(iter(test_qrels[q]))][:160], "...")

# %% [markdown]
# **Expected:** BM25 better on ~80 queries, dense better on ~70, tied on ~150
# of 300 (mostly where both are perfect or both fail). In our run BM25's largest advantage was on a claim
# with a rare exact term (*"1/2000 in UK have abnormal PrP positivity"*: "PrP"/"prion" anchors
# the match); dense retrieval's largest advantage was on a claim that paraphrased the abstract (*"methionine restriction may activate
# miRNAs"* vs an abstract on "microRNAs" in "stress responses"). Typically BM25 is better when the claim
# shares rare, exact terms with its abstract (gene names, drug names, numbers); dense is better
# when the claim paraphrases the abstract. That complementarity is why hybrids work.
#
# ## 7.6 · Hybrid retrieval: reciprocal rank fusion
#
# BM25 scores are unbounded; cosine similarities live in [−1, 1]: adding them needs a
# per-query normalisation that is fragile. **Reciprocal rank fusion** (RRF, Cormack et al.
# 2009) uses only ranks:
#
# $$\mathrm{RRF}(d) = \sum_{\text{systems } s} \frac{1}{k + \mathrm{rank}_s(d)}, \qquad k = 60 \text{ by convention}.$$
#
# A document ranked high by either system gets a large contribution; agreement between
# systems is rewarded; $k$ damps the dominance of the very top ranks.

# %% [markdown]
# > **Exercise 2.** Write `reciprocal_rank_fusion(rankings, k=60)`: `rankings` is a list of ranked
# > lists of document ids (best first; ranks start at **1**; lists may have different lengths and
# > contents). Return all ids that appear in any list, sorted by fused score, highest first.
# > Break ties by first appearance (iterate the lists in order, and sort stably).

# %%
def reciprocal_rank_fusion(rankings, k=60):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(reciprocal_rank_fusion)

# %%
runs["hybrid (RRF)"] = {q: R.reciprocal_rank_fusion([runs["BM25"][q], runs["dense (MiniLM)"][q]], k=60)[:100]
                        for q in test_qids}
metrics["hybrid (RRF)"] = R.evaluate_run(runs["hybrid (RRF)"], test_qrels)
nd_hybrid = R.per_query_ndcg(runs["hybrid (RRF)"], test_qrels)
diff = nd_hybrid - np.maximum(nd_bm25, nd_dense)
boot = np.random.default_rng(0)
idx = boot.integers(0, len(test_qids), (2000, len(test_qids)))
for other, nd in [("BM25", nd_bm25), ("dense", nd_dense)]:
    d = nd_hybrid - nd
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    print(f"hybrid - {other:5}: {d.mean():+.3f} nDCG@10 (95% bootstrap CI over queries {lo:+.3f} to {hi:+.3f})")
print(f"\n{'system':16} {'nDCG@10':>8} {'Recall@100':>10}")
for name in ["BM25", "dense (MiniLM)", "dense, HNSW", "hybrid (RRF)"]:
    m = metrics[name]
    print(f"{name:16} {m['ndcg@10']:8.3f} {m['recall@100']:10.3f}")

results = {"device": describe(DEVICE), "n_docs": len(doc_ids), "doc_len": DOC_LEN, "metrics": metrics,
           "embedding_throughput": throughput, "bm25_ms_per_query": bm25_query_ms,
           "faiss": {"real_flat_ms": ms_flat, "real_hnsw_ms": ms_hnsw, "real_hnsw_overlap@10": overlap,
                     "synthetic_n": N_SYN, "synthetic_exact_ms": ms_exact, "synthetic_hnsw": ann_rows}}
(OUT / "retrieval_results.json").write_text(json.dumps(results, indent=2, default=float))
print("wrote", OUT / "retrieval_results.json")

# %% [markdown]
# **Expected (CPU, measured):**
#
# | system | nDCG@10 | Recall@100 |
# |---|---|---|
# | BM25 | 0.667 | 0.886 |
# | dense (MiniLM, 128 tokens) | 0.622 | 0.919 |
# | dense, HNSW | 0.619 | 0.916 |
# | hybrid (RRF) | **0.681** | **0.954** |
#
# **Interpretation.** On nDCG@10, BM25 beats the general-purpose dense model here (by 0.045); dense has
# the better Recall@100. The hybrid is best on both metrics, but its nDCG@10 gain over BM25
# (+0.014) has a 95 % bootstrap interval that includes zero (−0.011 to +0.037); its gain over dense
# (+0.059, interval +0.028 to +0.089) and its Recall@100 (0.954) are clear. For a first stage
# feeding a re-ranker, that recall gain is what counts.
#
# **On a T4 you should see** the dense model improve with 256-token documents (most abstracts
# exceed 128 tokens): published results for `all-MiniLM-L6-v2` on SciFact are around 0.64–0.65
# nDCG@10, still at or below BM25, with the hybrid a little above both (not measured in our
# verification runs).
#
# **What to take away.** No single first stage is best everywhere: on scientific claims, exact
# terms matter, and a general-purpose embedding model has not seen this domain's vocabulary.
# Measure on *your* queries before replacing a lexical system. Chapter 46 shows the two
# levers that move dense retrieval: **fine-tuning the bi-encoder on in-domain pairs** and a
# **cross-encoder re-ranker** on top.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-two-stage-retrieval.svg" alt="A pipeline: corpus embedded offline, stage 1 BM25 plus dense fused by RRF returning the top 50, stage 2 a cross-encoder re-ranking to the top 5 for the user or an LLM" width="100%">
#
# *What to notice:* this lab built the blue box; Chapter 46 adds the yellow one. Stage 1 is optimised for recall because anything it misses is gone for good.
#
# > **Intuition.** BM25 matches *spelling*, a dense model matches *meaning*. "PrP" in a claim and
# > "PrP" in an abstract is a sure hit for BM25 and just another token for a general embedding
# > model; "methionine restriction may activate miRNAs" vs "microRNAs in stress responses" is the
# > reverse. A hybrid asks both and rewards agreement.
#
# > **Intuition.** HNSW is like asking for directions in an unfamiliar country: first take
# > the motorway to the right region (the sparse top layer), then main roads, then side streets
# > (the dense bottom layer). `efSearch` is how many alternative routes you keep in mind; more
# > routes, better answer, slower trip.

# %% [markdown]
# > **Exercise 3.** You are asked to replace the BM25 search over 50 million product
# > descriptions with "semantic search". Using this notebook's measurements, write a short plan:
# > what you would evaluate offline (metrics, query sets, which slices: head vs tail queries,
# > exact identifiers like SKUs), the index choice and its memory at 384 dimensions (float32 vs
# > product quantisation), the re-embedding cost when the model changes, and whether you would
# > deploy dense-only or hybrid, and why. Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1. BM25.**
#
# $$
# \operatorname{BM25}(q, d) = \sum_{t \in q} \operatorname{idf}(t)\, \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1 \big(1 - b + b\, \lvert d \rvert / \text{avgdl}\big)}, \qquad
# \operatorname{idf}(t) = \ln\!\Big(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\Big)
# $$
#
# In words: rare query terms (high idf) that appear in the document count most; repeated
# occurrences saturate ($k_1$); long documents are penalised ($b$). Example: $N = 1000$ documents,
# the term is in $n_t = 10$, appears $f = 3$ times in a document of average length, $k_1 = 1.2$:
# $\operatorname{idf} = \ln(1 + 990.5/10.5) = 4.56$ and the term contributes
# $4.56 \times 3 \cdot 2.2 / (3 + 1.2) = 7.16$. Ten times as many occurrences ($f = 30$) give
# $4.56 \times 2.12 = 9.64$: only 35% more (saturation; the ceiling is $\operatorname{idf} \cdot (k_1 + 1) = 10.0$).
#
# **2. Cosine similarity (unit vectors).**
#
# $$
# \operatorname{sim}(\mathbf q, \mathbf d) = \frac{\mathbf q \cdot \mathbf d}{\lVert \mathbf q \rVert\, \lVert \mathbf d \rVert} = \mathbf q \cdot \mathbf d \quad \text{if } \lVert \mathbf q \rVert = \lVert \mathbf d \rVert = 1
# $$
#
# Example: exact search over $N = 50{,}000$ vectors of $d = 384$ costs $N d \approx 1.9 \times 10^7$
# multiply-adds per query, which is why it grows linearly with the corpus.
#
# **3. nDCG@k.**
#
# $$
# \operatorname{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}, \qquad \operatorname{nDCG@}k = \frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}
# $$
#
# In words: gains discounted by position, divided by the best possible ordering. Example: one
# relevant document, found at rank 3: $1/\log_2 4 = 0.5$, ideal is $1$, so nDCG@10 = 0.5; at rank
# 1 it would be 1.0.
#
# **4. Recall@k.**
#
# $$
# \operatorname{Recall@}k = \frac{\lvert \text{relevant} \cap \text{top-}k \rvert}{\lvert \text{relevant} \rvert}
# $$
#
# Example: 2 relevant documents, 1 of them in the top 100: Recall@100 = 0.5.
#
# **5. Reciprocal rank fusion.**
#
# $$
# \operatorname{RRF}(d) = \sum_{s} \frac{1}{k + \operatorname{rank}_s(d)}, \qquad k = 60
# $$
#
# Example: ranks 1 and 3 give $1/61 + 1/63 = 0.03227$; ranks 2 and 2 give $2/62 = 0.03226$:
# nearly a tie, so consistent agreement is worth as much as one first place.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: faiss` | FAISS is not preinstalled in Colab | the catch-up cell installs `faiss-cpu`; restart if the import still fails |
# | nDCG is ~0 for every system | id types differ (int vs str) between run and qrels | compare as strings, as `scaleml.retrieval.load_scifact` does |
# | Dense results worse than random | embeddings not normalised (inner product ≠ cosine) or naive mean over padding | L2-normalise; mask-aware mean pooling |
# | HNSW recall low | `efSearch` too small, or metric mismatch (L2 index, IP vectors) | raise `efSearch`; build with `METRIC_INNER_PRODUCT` for cosine |
# | Embedding the corpus is very slow | CPU, long documents, unsorted batches | GPU + fp16; sort by length; truncate (measure what truncation costs) |
# | Metrics much higher than published | evaluated on a sub-corpus (fewer distractors), or test queries leaked into tuning | compare only systems run on the same corpus and queries |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. Walk through BM25: what do idf, $k_1$ and $b$ each do, and when does BM25 beat a dense retriever?
# 2. Bi-encoder vs cross-encoder: why can only one of them index a corpus offline?
# 3. nDCG@10 vs Recall@100: which would you optimise for a first-stage retriever feeding a re-ranker, and why? What does an unjudged document do to both?
# 4. Exact vs HNSW vs IVF-PQ: how do you choose for 1M, 100M and 10B vectors? What do `M` and `efSearch` control?
# 5. Why does reciprocal rank fusion work without score normalisation, and when could it hurt?
# 6. Design: semantic search over a company's internal documents for an LLM assistant. Name the retrieval stages, the evaluation set you would build, and the online metrics you would watch.
#
# **Next chapter:** [Chapter 46 · Contrastive training & re-ranking](../ch46-contrastive-training-and-reranking/notebook.ipynb) — fine-tune this bi-encoder contrastively on in-domain pairs and add a cross-encoder re-ranker.
