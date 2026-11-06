"""Ranking and retrieval metrics, written to be read.

Conventions: higher score = ranked higher; ties broken by the order given
(stable sort), so results are deterministic. ``k`` truncates the list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevance_in_rank_order, k: int) -> float:
    rel = np.asarray(relevance_in_rank_order, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0 ** rel - 1.0) * discounts))


def ndcg_at_k(relevance_in_rank_order, k: int) -> float:
    """NDCG@k with exponential gain. 0 when the list has no relevant item."""
    ideal = dcg_at_k(np.sort(np.asarray(relevance_in_rank_order, dtype=float))[::-1], k)
    return dcg_at_k(relevance_in_rank_order, k) / ideal if ideal > 0 else 0.0


def reciprocal_rank(relevance_in_rank_order) -> float:
    hits = np.flatnonzero(np.asarray(relevance_in_rank_order) > 0)
    return 1.0 / (hits[0] + 1) if hits.size else 0.0


def recall_at_k(ranked_ids, relevant_ids, k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return float("nan")
    return len(relevant.intersection(list(ranked_ids)[:k])) / len(relevant)


def grouped_ranking_metrics(df: pd.DataFrame, group: str, score: str, label: str,
                            k: int = 10, skip_all_negative: bool = True) -> dict[str, float]:
    """Mean NDCG@k and MRR over groups (e.g. requests), ranking by ``score``.

    Groups with no positive label carry no ranking information; by default
    they are skipped (and counted), which is the usual convention - say so
    when you report the number.
    """
    ndcgs, rrs, skipped = [], [], 0
    order = df.sort_values([group, score], ascending=[True, False], kind="stable")
    for _, g in order.groupby(group, sort=False):
        rel = g[label].to_numpy()
        if skip_all_negative and rel.max() <= 0:
            skipped += 1
            continue
        ndcgs.append(ndcg_at_k(rel, k))
        rrs.append(reciprocal_rank(rel))
    return {f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else float("nan"),
            "mrr": float(np.mean(rrs)) if rrs else float("nan"),
            "groups": len(ndcgs), "groups_skipped": skipped}


def catalog_coverage(recommended_lists, n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    seen = set()
    for lst in recommended_lists:
        seen.update(lst)
    return len(seen) / n_items
