"""Self-checks for Level 3 notebook 07 · Embeddings & semantic search."""

from __future__ import annotations

import math
import random
from collections import Counter

from churnlab.checks import run_check


def _bm25_reference(query_tokens, docs, k1, b):
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df = Counter(t for d in docs for t in set(d))
    out = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in set(query_tokens):
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


def ex1(bm25_scores):
    def test():
        rng = random.Random(0)
        vocab = [f"w{i}" for i in range(25)]
        docs = [[rng.choice(vocab) for _ in range(rng.randint(2, 30))] for _ in range(40)]
        docs.append([])  # an empty document must score 0, not crash
        query = ["w1", "w3", "w3", "w20", "unseen"]
        for k1, b in [(1.2, 0.75), (0.9, 0.4), (2.0, 0.0)]:
            got = list(map(float, bm25_scores(list(query), [list(d) for d in docs], k1=k1, b=b)))
            want = _bm25_reference(query, docs, k1, b)
            assert len(got) == len(docs), f"expected {len(docs)} scores, got {len(got)}"
            if k1 == 1.2:
                naive = [sum(_bm25_reference([t], docs, k1, b)[j] for t in query) for j in range(len(docs))]
                if any(abs(g - w) > 1e-6 for g, w in zip(got, want)) and all(
                        math.isclose(g, w, abs_tol=1e-6) for g, w in zip(got, naive)):
                    raise AssertionError("you summed over repeated query terms: use the DISTINCT query terms")
            bad = [j for j, (g, w) in enumerate(zip(got, want)) if not math.isclose(g, w, rel_tol=1e-6, abs_tol=1e-9)]
            assert not bad, (f"k1={k1}, b={b}: document {bad[0]} scored {got[bad[0]]:.4f}, expected {want[bad[0]]:.4f}"
                             " (check idf = ln(1 + (N - n_t + 0.5)/(n_t + 0.5)) and the length normalisation)")

    return run_check("Exercise 1 · bm25_scores", test,
                     hint="per doc: sum over set(query) of idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*len(d)/avgdl))")


def _rrf_reference(rankings, k):
    scores, order = {}, []
    for ranking in rankings:
        for r, d in enumerate(ranking, start=1):
            if d not in scores:
                order.append(d)
                scores[d] = 0.0
            scores[d] += 1.0 / (k + r)
    return sorted(order, key=lambda d: -scores[d])


def ex2(reciprocal_rank_fusion):
    def test():
        got = list(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60))
        assert sorted(got) == ["a", "b", "c", "d"], f"every id must appear exactly once, got {got}"
        assert got == ["a", "c", "b", "d"], f"expected ['a', 'c', 'b', 'd'], got {got} (ranks start at 1)"
        got = list(reciprocal_rank_fusion([["x", "y"], ["y", "x"]], k=1))
        assert got == ["x", "y"], f"a tie must keep first appearance: expected ['x', 'y'], got {got}"
        rng = random.Random(1)
        ids = list(range(50))
        for k in (1, 10, 60):
            rankings = [rng.sample(ids, rng.randint(5, 40)) for _ in range(3)]
            want = _rrf_reference(rankings, k)
            got = list(reciprocal_rank_fusion([list(r) for r in rankings], k=k))
            assert got == want, f"k={k}: fused order differs from the reference at position " \
                                f"{next(i for i, (g, w) in enumerate(zip(got, want)) if g != w)}"

    return run_check("Exercise 2 · reciprocal_rank_fusion", test,
                     hint="scores[d] += 1 / (k + rank) with rank from 1; a dict keeps insertion order; "
                          "sorted(scores, key=lambda d: -scores[d]) is stable")
