"""Candidate generation and retrieval for ShopStream (notebooks 07 and 08).

A ranker can only order what retrieval hands it. These are the classic
first-stage generators, written in plain NumPy/pandas so you can read every
line: global and per-category popularity, item-to-item co-visitation, and
implicit-feedback matrix factorisation (ALS). Plus the evaluation helpers:
recall@K against a set of held-out items, and blending.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp


# --------------------------------------------------------------------------- data
def click_matrix(clicks: pd.DataFrame, n_users: int, n_items: int) -> sp.csr_matrix:
    """User x item matrix of click counts (CSR)."""
    m = sp.coo_matrix((np.ones(len(clicks), dtype=np.float32),
                       (clicks["user_id"].to_numpy(), clicks["item_id"].to_numpy())),
                      shape=(n_users, n_items))
    return m.tocsr()


def truth_sets(clicks: pd.DataFrame) -> dict[int, set[int]]:
    """{user: set of items the user clicked} for an evaluation window."""
    return {int(u): set(map(int, g)) for u, g in clicks.groupby("user_id")["item_id"]}


# --------------------------------------------------------------------------- generators
def top_items(scores: np.ndarray, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k largest scores (descending; ties broken by lower index)."""
    s = np.asarray(scores, dtype=float).copy()
    if exclude is not None:
        s[exclude] = -np.inf
    k = min(k, s.size)
    part = np.argpartition(-s, k - 1)[:k]
    return part[np.lexsort((part, -s[part]))]


def popularity_scores(clicks: pd.DataFrame, n_items: int) -> np.ndarray:
    return np.bincount(clicks["item_id"].to_numpy(), minlength=n_items).astype(float)


def category_popularity(users_hist: sp.csr_matrix, pop: np.ndarray, item_cat: np.ndarray,
                        users: np.ndarray, k: int, n_cats: int | None = None) -> dict[int, list[int]]:
    """Per user: popular items from the categories they clicked, in proportion to their clicks.

    Users with no history fall back to global popularity.
    """
    n_cats = n_cats or int(item_cat.max()) + 1
    by_cat = [top_items(np.where(item_cat == c, pop, -np.inf), k) for c in range(n_cats)]
    global_top = top_items(pop, k)
    cat_onehot = sp.csr_matrix((np.ones(item_cat.size), (np.arange(item_cat.size), item_cat)),
                               shape=(item_cat.size, n_cats))
    ucat = (users_hist @ cat_onehot).toarray()
    out = {}
    for u in users:
        w = ucat[u]
        if w.sum() == 0:
            out[int(u)] = global_top[:k].tolist()
            continue
        quota = np.floor(k * w / w.sum()).astype(int)
        quota[np.argmax(w)] += k - quota.sum()
        lst = [int(i) for c in np.flatnonzero(quota) for i in by_cat[c][:quota[c]]]
        out[int(u)] = lst
    return out


def covisitation(pairs_df: pd.DataFrame, top_n: int) -> dict[int, list[int]]:
    """Item -> its top_n co-visited items.

    ``pairs_df`` has one row per click with columns ``session_id`` and
    ``item_id``. Two items co-visit once per session in which both were clicked
    (duplicates inside a session count once). Neighbours are sorted by count,
    descending, ties broken by smaller item id; an item is never its own
    neighbour.
    """
    d = pairs_df[["session_id", "item_id"]].drop_duplicates()
    pairs = d.merge(d, on="session_id", suffixes=("", "_nb"))
    pairs = pairs[pairs["item_id"] != pairs["item_id_nb"]]
    counts = pairs.groupby(["item_id", "item_id_nb"]).size().rename("n").reset_index()
    counts = counts.sort_values(["item_id", "n", "item_id_nb"], ascending=[True, False, True])
    top = counts.groupby("item_id").head(top_n)
    return {int(i): g.tolist() for i, g in top.groupby("item_id")["item_id_nb"]}


def covisitation_candidates(neighbours: dict[int, list[int]], user_recent: dict[int, list[int]],
                            k: int, fallback: np.ndarray) -> dict[int, list[int]]:
    """Score items by how often they co-visit the user's recent clicks; pad with ``fallback``."""
    out = {}
    for u, recent in user_recent.items():
        score: dict[int, float] = defaultdict(float)
        for rank, item in enumerate(recent):
            for j, nb in enumerate(neighbours.get(item, [])):
                score[nb] += 1.0 / (1 + j) / (1 + 0.1 * rank)  # closer neighbours and newer clicks count more
        ranked = sorted(score, key=lambda i: (-score[i], i))[:k]
        if len(ranked) < k:
            seen = set(ranked)
            ranked += [int(i) for i in fallback if int(i) not in seen][:k - len(ranked)]
        out[int(u)] = ranked
    return out


def implicit_als(R: sp.csr_matrix, factors: int = 32, reg: float = 0.1, alpha: float = 20.0,
                 iterations: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Implicit-feedback ALS (Hu, Koren & Volinsky 2008) in NumPy.

    Preference p_ui = 1 if r_ui > 0; confidence c_ui = 1 + alpha * r_ui.
    Each half-step solves, per row u,
    (Y^T Y + Y^T (C_u - I) Y + reg I) x_u = Y^T C_u p_u
    using only the row's non-zeros (the Y^T Y term is shared).
    """
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    X = rng.normal(0, 0.01, (n_u, factors))
    Y = rng.normal(0, 0.01, (n_i, factors))
    Rt = R.T.tocsr()

    def solve(M: sp.csr_matrix, fixed: np.ndarray, out: np.ndarray) -> None:
        gram = fixed.T @ fixed + reg * np.eye(factors)
        for row in range(M.shape[0]):
            lo, hi = M.indptr[row], M.indptr[row + 1]
            if lo == hi:
                out[row] = 0.0
                continue
            idx, r = M.indices[lo:hi], M.data[lo:hi]
            Yi = fixed[idx]
            conf = alpha * r  # (c - 1)
            A = gram + (Yi.T * conf) @ Yi
            b = Yi.T @ (1.0 + conf)
            out[row] = np.linalg.solve(A, b)

    for _ in range(iterations):
        solve(R, Y, X)
        solve(Rt, X, Y)
    return X.astype(np.float32), Y.astype(np.float32)


def score_topk(user_vecs: np.ndarray, item_vecs: np.ndarray, users, k: int,
               batch: int = 2048, item_bias: np.ndarray | None = None) -> dict[int, list[int]]:
    """Exact top-k by dot product for each user in ``users``."""
    users = np.asarray(users)
    out = {}
    for lo in range(0, len(users), batch):
        u = users[lo:lo + batch]
        s = user_vecs[u] @ item_vecs.T
        if item_bias is not None:
            s = s + item_bias[None, :]
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        top = np.take_along_axis(part, order, axis=1)
        out.update({int(uu): row.tolist() for uu, row in zip(u, top)})
    return out


# --------------------------------------------------------------------------- evaluation
def recall_at_k_batch(candidates: dict, truth: dict, k: int) -> float:
    """Mean over users with a non-empty truth set of |top-k candidates ∩ truth| / |truth|.

    A user missing from ``candidates`` scores 0 (the generator returned nothing).
    """
    vals = []
    for u, rel in truth.items():
        if not rel:
            continue
        got = set(list(candidates.get(u, []))[:k])
        vals.append(len(got & set(rel)) / len(rel))
    return float(np.mean(vals)) if vals else float("nan")


def blend(lists: list[dict], quotas: list[int], users) -> dict[int, list[int]]:
    """Merge generators: take each generator's next items up to its quota, skipping duplicates."""
    out = {}
    for u in users:
        seen: list[int] = []
        seen_set: set[int] = set()
        for cand, q in zip(lists, quotas):
            taken = 0
            for i in cand.get(int(u), []):
                if taken >= q:
                    break
                if i not in seen_set:
                    seen.append(int(i))
                    seen_set.add(int(i))
                    taken += 1
        out[int(u)] = seen
    return out


def true_top_items(world, users, k: int, eligible: np.ndarray | None = None, batch: int = 256) -> dict[int, set[int]]:
    """{user: the k items with the highest TRUE click probability} - the simulation's ground truth.

    Computed in batches so memory stays small (users x items x dim would not fit).
    """
    users = np.asarray(users)
    n_items = len(world.items)
    all_items = np.arange(n_items)
    out = {}
    for lo in range(0, len(users), batch):
        u = users[lo:lo + batch]
        logit = world.true_logit(u[:, None], all_items[None, :])
        if eligible is not None:
            logit[:, ~eligible] = -np.inf
        top = np.argpartition(-logit, k - 1, axis=1)[:, :k]
        out.update({int(uu): set(row.tolist()) for uu, row in zip(u, top)})
    return out


# --------------------------------------------------------------------------- two-tower helpers (notebook 08)
def in_batch_softmax_loss(user_emb, item_emb, log_q=None, item_ids=None):
    """Sampled-softmax loss with in-batch negatives (needs PyTorch).

    Row b of ``user_emb`` is paired with row b of ``item_emb`` (its clicked
    item); every other item in the batch is a negative. Logits are
    ``user_emb @ item_emb.T``; with ``log_q`` (log sampling probability of each
    batch item, shape [B]) the logQ correction subtracts ``log_q[j]`` from
    column j. With ``item_ids``, columns holding the *same* item as the
    row's positive (an accidental duplicate in the batch) are masked out.
    """
    import torch
    import torch.nn.functional as F

    logits = user_emb @ item_emb.T
    if log_q is not None:
        logits = logits - log_q[None, :]
    if item_ids is not None:
        same = item_ids[:, None] == item_ids[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, torch.finfo(logits.dtype).min)
    target = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, target)


def recall_vs_exact(approx_ids, exact_ids) -> float:
    """Mean over queries of |approx ∩ exact| / |exact|, ignoring FAISS's -1 padding."""
    vals = []
    for a, e in zip(np.asarray(approx_ids), np.asarray(exact_ids)):
        e_set = {int(x) for x in e if x >= 0}
        if not e_set:
            continue
        vals.append(len({int(x) for x in a if x >= 0} & e_set) / len(e_set))
    return float(np.mean(vals)) if vals else float("nan")
