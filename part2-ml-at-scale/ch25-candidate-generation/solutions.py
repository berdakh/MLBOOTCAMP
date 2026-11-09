"""Reference solutions for Level 2 notebook 07 (graded in CI by tests/test_solutions.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(pairs_df: pd.DataFrame, top_n: int) -> dict:
    d = pairs_df[["session_id", "item_id"]].drop_duplicates()
    p = d.merge(d, on="session_id", suffixes=("", "_nb"))
    p = p[p["item_id"] != p["item_id_nb"]]
    counts = p.groupby(["item_id", "item_id_nb"]).size().rename("n").reset_index()
    counts = counts.sort_values(["item_id", "n", "item_id_nb"], ascending=[True, False, True])
    top = counts.groupby("item_id").head(top_n)
    return {int(i): g.tolist() for i, g in top.groupby("item_id")["item_id_nb"]}


def ex2(candidates: dict, truth: dict, k: int) -> float:
    vals = []
    for u, rel in truth.items():
        rel = set(rel)
        if not rel:
            continue
        got = set(list(candidates.get(u, []))[:k])
        vals.append(len(got & rel) / len(rel))
    return float(np.mean(vals)) if vals else float("nan")
