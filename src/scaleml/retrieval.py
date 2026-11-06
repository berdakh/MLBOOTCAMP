"""Retrieval: BM25, dense bi-encoders, FAISS, fusion, metrics, contrastive training, re-ranking.

Introduced in Level 3 notebooks 07 (embeddings & semantic search) and 08
(contrastive training & re-ranking). Small and readable on purpose:

* ``load_scifact`` / ``subsample_corpus``: the BEIR SciFact benchmark from the Hub;
* ``tokenize`` / ``bm25_scores`` / ``BM25Index``: lexical retrieval (Okapi BM25)
  written out, then vectorised with a sparse matrix;
* ``Encoder``: a sentence encoder from plain ``transformers`` (``AutoModel`` +
  mask-aware mean pooling + L2 normalisation), no ``sentence-transformers`` needed;
* ``ndcg_at_k`` / ``recall_at_k`` / ``evaluate_run``: IR metrics with graded qrels;
* ``reciprocal_rank_fusion``: hybrid lexical + dense retrieval;
* ``info_nce_loss`` / ``mine_hard_negatives``: contrastive bi-encoder training;
* ``cross_encoder_scores``: a re-ranker that reads query and document together.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

import numpy as np
import torch

from .device import autocast
from .finetune import mean_pool

# --------------------------------------------------------------------------- data

#: BEIR SciFact (Wadden et al., 2020; Thakur et al., 2021): 5,183 abstracts, 1,109 claims.
#: Cards: https://huggingface.co/datasets/BeIR/scifact (cc-by-sa-4.0 on the card)
#:        https://huggingface.co/datasets/BeIR/scifact-qrels (cc-by-sa-4.0 on the card)
SCIFACT_ID, SCIFACT_REV = "BeIR/scifact", "b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_QRELS_ID, SCIFACT_QRELS_REV = "BeIR/scifact-qrels", "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"

#: Models (Apache-2.0 on their cards), pinned.
MINILM_ID, MINILM_REV = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CROSS_ENCODER_ID, CROSS_ENCODER_REV = "cross-encoder/ms-marco-MiniLM-L-6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def load_scifact():
    """Return ``(corpus, queries, qrels)``.

    ``corpus``: doc id -> "title. abstract"; ``queries``: query id -> claim;
    ``qrels``: {"train": {qid: {doc_id: relevance}}, "test": {...}}. Ids are strings.
    """
    from datasets import load_dataset

    c = load_dataset(SCIFACT_ID, "corpus", revision=SCIFACT_REV)["corpus"]
    q = load_dataset(SCIFACT_ID, "queries", revision=SCIFACT_REV)["queries"]
    r = load_dataset(SCIFACT_QRELS_ID, revision=SCIFACT_QRELS_REV)
    corpus = {str(i): (f"{t}. {x}" if t else x).strip() for i, t, x in zip(c["_id"], c["title"], c["text"])}
    queries = {str(i): t for i, t in zip(q["_id"], q["text"])}
    qrels: dict[str, dict[str, dict[str, int]]] = {}
    for split in r:
        d: dict[str, dict[str, int]] = {}
        for qid, did, s in zip(r[split]["query-id"], r[split]["corpus-id"], r[split]["score"]):
            if int(s) > 0:
                d.setdefault(str(qid), {})[str(did)] = int(s)
        qrels[split] = d
    return corpus, queries, qrels


def subsample_corpus(corpus: dict, keep_qrels: dict, n_docs: int, seed: int = 0) -> dict:
    """Every document relevant to a query in ``keep_qrels``, plus random others, up to ``n_docs``.

    A smaller corpus makes retrieval *easier* (fewer distractors), so metrics on a
    subsample are higher than on the full corpus: compare systems on the same corpus only.
    """
    if n_docs >= len(corpus):
        return dict(corpus)
    must = sorted({d for rel in keep_qrels.values() for d in rel if d in corpus})
    rest = sorted(set(corpus) - set(must))
    random.Random(seed).shuffle(rest)
    keep = must + rest[:max(0, n_docs - len(must))]
    return {d: corpus[d] for d in sorted(keep)}


# --------------------------------------------------------------------------- lexical: BM25

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with "
    "we our their there these those not no than then can may also into".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stopwords. Real systems add stemming (Porter, Krovetz)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def bm25_idf(df: int, n_docs: int) -> float:
    """Lucene's BM25 idf: ln(1 + (N - df + 0.5) / (df + 0.5)), always positive."""
    return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(query_tokens, doc_tokens_list, k1: float = 1.2, b: float = 0.75) -> list[float]:
    """BM25 score of every document for one query, written as the formula reads.

    score(q, d) = sum over *distinct* query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    """
    n = len(doc_tokens_list)
    avgdl = sum(len(d) for d in doc_tokens_list) / max(1, n)
    tfs = [Counter(d) for d in doc_tokens_list]
    df = Counter(t for tf in tfs for t in tf)
    out = []
    for tf, doc in zip(tfs, doc_tokens_list):
        s = 0.0
        for t in set(query_tokens):
            if tf[t]:
                norm = tf[t] + k1 * (1 - b + b * len(doc) / avgdl)
                s += bm25_idf(df[t], n) * tf[t] * (k1 + 1) / norm
        out.append(s)
    return out


class BM25Index:
    """The same BM25, precomputed as a sparse (vocabulary x documents) matrix of term weights.

    A query's scores are then one sparse row-slice and a sum: this is, in
    miniature, what an inverted index (Lucene, Elasticsearch) does.
    """

    def __init__(self, doc_tokens_list, k1: float = 1.2, b: float = 0.75):
        from scipy import sparse

        self.k1, self.b = k1, b
        self.vocab: dict[str, int] = {}
        rows, cols, vals = [], [], []
        lengths = np.array([len(d) for d in doc_tokens_list], dtype=np.float64)
        avgdl = lengths.mean() if len(lengths) else 1.0
        for j, doc in enumerate(doc_tokens_list):
            for t, tf in Counter(doc).items():
                rows.append(self.vocab.setdefault(t, len(self.vocab)))
                cols.append(j)
                vals.append(tf)
        tf = np.array(vals, dtype=np.float64)
        rows_a, cols_a = np.array(rows), np.array(cols)
        df = np.bincount(rows_a, minlength=len(self.vocab))
        n = len(doc_tokens_list)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        norm = tf + k1 * (1 - b + b * lengths[cols_a] / avgdl)
        w = idf[rows_a] * tf * (k1 + 1) / norm
        self.matrix = sparse.csr_matrix((w, (rows_a, cols_a)), shape=(len(self.vocab), n))
        self.n_docs = n

    def scores(self, query_tokens) -> np.ndarray:
        ids = [self.vocab[t] for t in set(query_tokens) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs)
        return np.asarray(self.matrix[ids].sum(axis=0)).ravel()

    def search(self, query_tokens, k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Indices and scores of the top-``k`` documents."""
        s = self.scores(query_tokens)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]


# --------------------------------------------------------------------------- dense: bi-encoder


class Encoder:
    """A sentence encoder from plain ``transformers``: tokenise, encode, mean-pool, L2-normalise.

    This is exactly what ``sentence-transformers`` does for
    ``all-MiniLM-L6-v2`` (its ``modules.json``: Transformer -> Pooling(mean) -> Normalize).
    """

    def __init__(self, model_id: str = MINILM_ID, revision: str | None = MINILM_REV, device=None,
                 model=None, tokenizer=None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or torch.device("cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = (model or AutoModel.from_pretrained(model_id, revision=revision, use_safetensors=True))
        self.model.to(self.device)

    def embed(self, texts, max_len: int = 128) -> torch.Tensor:
        """Differentiable embeddings (use inside a training step)."""
        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=max_len,
                             return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state
        return torch.nn.functional.normalize(mean_pool(h.float(), enc["attention_mask"]), dim=-1)

    @torch.inference_mode()
    def encode(self, texts, batch_size: int = 64, max_len: int = 128) -> np.ndarray:
        """float32 (n, dim) unit vectors. Batches are formed after sorting by length (less padding)."""
        texts = list(texts)
        self.model.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            idx = order[s:s + batch_size]
            with autocast(self.device):
                e = self.embed([texts[i] for i in idx], max_len)
            out[idx] = e.float().cpu().numpy()
        return out


def faiss_flat_index(vectors: np.ndarray):
    """Exact inner-product search (= cosine on unit vectors)."""
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def faiss_hnsw_index(vectors: np.ndarray, m: int = 32, ef_construction: int = 80, ef_search: int = 64):
    """Approximate search on an HNSW graph (``m`` links per node), inner-product metric."""
    import faiss

    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    index.hnsw.efSearch = ef_search
    return index


# --------------------------------------------------------------------------- metrics


def ndcg_at_k(ranked_ids, relevant: dict, k: int = 10) -> float:
    """nDCG@k with linear gains (as ``pytrec_eval`` and BEIR compute it)."""
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked_ids[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant: dict, k: int = 100) -> float:
    rel = {d for d, g in relevant.items() if g > 0}
    return len(rel & set(ranked_ids[:k])) / len(rel) if rel else 0.0


def evaluate_run(run: dict, qrels: dict, ndcg_k: int = 10, recall_k: int = 100) -> dict:
    """Mean nDCG@k and Recall@k over the queries in ``qrels`` (a missing ranking scores 0)."""
    nd = [ndcg_at_k(run.get(q, []), rel, ndcg_k) for q, rel in qrels.items()]
    rc = [recall_at_k(run.get(q, []), rel, recall_k) for q, rel in qrels.items()]
    return {f"ndcg@{ndcg_k}": float(np.mean(nd)), f"recall@{recall_k}": float(np.mean(rc)), "queries": len(nd)}


def per_query_ndcg(run: dict, qrels: dict, k: int = 10) -> np.ndarray:
    return np.array([ndcg_at_k(run.get(q, []), rel, k) for q, rel in qrels.items()])


# --------------------------------------------------------------------------- fusion


def reciprocal_rank_fusion(rankings, k: int = 60) -> list:
    """Fuse ranked lists: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Uses only ranks, so it needs no score calibration between systems
    (Cormack et al., 2009). Ties are broken by first appearance.
    """
    scores: dict = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + r)
    return sorted(scores, key=lambda d: -scores[d])


# --------------------------------------------------------------------------- contrastive training


def info_nce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """InfoNCE with in-batch negatives.

    ``q``: (B, dim) query embeddings; ``d``: (N, dim) document embeddings with
    N >= B, where ``d[i]`` is the positive for ``q[i]`` and every other row
    (other queries' positives, plus any extra hard negatives at rows B..N-1)
    is a negative. Cross-entropy over the (B, N) similarity matrix / temperature.
    """
    logits = q @ d.T / temperature
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(q), device=q.device))


def mine_hard_negatives(ranked_ids, positives, n: int, skip_top: int = 0) -> list:
    """The first ``n`` ids of ``ranked_ids`` that are not positives, after skipping the top ``skip_top`` ranks.

    Skipping the very top reduces *false negatives*: unlabelled documents that are
    actually relevant, which a retriever ranks highest of all.
    """
    pos = set(positives)
    return [d for d in list(ranked_ids)[skip_top:] if d not in pos][:n]


# --------------------------------------------------------------------------- re-ranking


@torch.inference_mode()
def cross_encoder_scores(model, tokenizer, query: str, docs, device, max_len: int = 256,
                         batch_size: int = 32) -> np.ndarray:
    """Relevance logits of (query, doc) pairs from a cross-encoder (one logit per pair)."""
    model.eval()
    out = []
    for s in range(0, len(docs), batch_size):
        chunk = list(docs[s:s + batch_size])
        enc = tokenizer([query] * len(chunk), chunk, padding=True, truncation="only_second",
                        max_length=max_len, return_tensors="pt").to(device)
        with autocast(device):
            out.append(model(**enc).logits[:, 0].float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)
