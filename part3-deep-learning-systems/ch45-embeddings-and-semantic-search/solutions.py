"""Reference solutions for Level 3 notebook 07. Graded by tests/test_solutions.py."""

from __future__ import annotations

import math
from collections import Counter


def ex1(query_tokens, doc_tokens_list, k1=1.2, b=0.75):
    n = len(doc_tokens_list)
    avgdl = sum(len(d) for d in doc_tokens_list) / max(1, n)
    df = Counter(t for d in doc_tokens_list for t in set(d))
    scores = []
    for doc in doc_tokens_list:
        tf = Counter(doc)
        s = 0.0
        for t in set(query_tokens):
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(s)
    return scores


def ex2(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, start=1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])
