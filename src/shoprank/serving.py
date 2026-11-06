"""Serving a ranking cascade: retrieval -> features -> ranker -> business rules.

Notebook 17 builds this step by step; notebooks 18 and 20 reuse it. It is a
deliberately small, readable version of what a production recommender runs
for every feed request:

    retrieval   FAISS inner-product search over matrix-factorisation item
                embeddings, item-item co-visitation from the user's recent
                clicks, and a popularity list as the fallback for everyone
    features    an in-memory "online store" (user rows behind a simulated
                network round trip, item columns in arrays) with an LRU cache
                in front, and defaults when a row is missing
    ranker      LightGBM on a handful of point-in-time features, called with
                a timeout; on error or timeout the candidates are served in
                popularity order instead
    rules       dedupe, drop items not launched yet, cap items per category

Point-in-time discipline: an :class:`OfflineState` is computed from a fixed
window of days (``feature_days``) and is only ever used to serve or label
*later* days. Counts are normalised per day, so a state built from a longer
window (refreshed for serving) has the same feature scale as the one the
ranker was trained on.

Everything here runs on a CPU in seconds at the ``small`` scale.
"""

from __future__ import annotations

import concurrent.futures as cf
import socket
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

FEATURES = [
    "mf_score", "covis_score", "item_ctr", "item_log_imps_per_day", "item_log_price",
    "item_age_days", "user_ctr", "user_log_imps_per_day", "user_cat_share", "position",
]


# ---------------------------------------------------------------------------
# Offline state: everything computed from a window of past days
# ---------------------------------------------------------------------------
@dataclass
class OfflineState:
    feature_days: tuple[int, int]          # inclusive window the state was computed from
    item_cat: np.ndarray
    item_created: np.ndarray
    item_log_price: np.ndarray
    item_ctr: np.ndarray                   # smoothed click-through rate
    item_log_imps_per_day: np.ndarray
    popular: np.ndarray                    # item ids, most clicked first
    user_ctr: np.ndarray
    user_log_imps_per_day: np.ndarray
    user_cat_share: np.ndarray             # (n_users, n_categories), smoothed share of clicks
    user_emb: np.ndarray                   # (n_users, dim) float32
    item_emb: np.ndarray                   # (n_items, dim) float32
    user_covis: sp.csr_matrix = field(repr=False)  # (n_users, n_items): sum of co-visitation from recent clicks
    user_seen: np.ndarray = field(repr=False)      # bool (n_users,): had any impression in the window
    global_ctr: float = 0.0
    n_categories: int = 0

    @property
    def n_users(self) -> int:
        return len(self.user_ctr)

    @property
    def n_items(self) -> int:
        return len(self.item_ctr)


def build_state(imps: pd.DataFrame, items: pd.DataFrame, n_users: int, feature_days: tuple[int, int],
                dim: int = 32, recent: int = 5, covis_window: int = 3, covis_top: int = 30,
                smoothing: float = 20.0, seed: int = 0) -> OfflineState:
    """Compute features, embeddings and co-visitation from impressions in ``feature_days``."""
    lo, hi = feature_days
    n_items = len(items)
    n_days = hi - lo + 1
    win = imps[(imps["day"] >= lo) & (imps["day"] <= hi)]
    u = win["user_id"].to_numpy()
    i = win["item_id"].to_numpy()
    c = win["clicked"].to_numpy().astype(np.float64)
    global_ctr = float(c.mean())

    item_imps = np.bincount(i, minlength=n_items).astype(float)
    item_clicks = np.bincount(i, weights=c, minlength=n_items)
    user_imps = np.bincount(u, minlength=n_users).astype(float)
    user_clicks = np.bincount(u, weights=c, minlength=n_users)
    item_ctr = (item_clicks + smoothing * global_ctr) / (item_imps + smoothing)
    user_ctr = (user_clicks + smoothing * global_ctr) / (user_imps + smoothing)

    item_cat = items["category"].to_numpy()
    n_cat = int(item_cat.max()) + 1
    cat_prior = np.bincount(item_cat[i[c > 0]], minlength=n_cat) + 1.0
    cat_prior = cat_prior / cat_prior.sum()
    uc = np.bincount(u[c > 0] * n_cat + item_cat[i[c > 0]], minlength=n_users * n_cat).reshape(n_users, n_cat)
    m = 5.0
    user_cat_share = ((uc + m * cat_prior) / (uc.sum(1, keepdims=True) + m)).astype(np.float32)

    # Matrix factorisation of the (user x item) click matrix by truncated SVD.
    clicks = win.loc[win["clicked"], ["user_id", "item_id", "ts"]]
    mat = sp.csr_matrix((np.ones(len(clicks)), (clicks["user_id"].to_numpy(), clicks["item_id"].to_numpy())),
                        shape=(n_users, n_items))
    mat.data = np.log1p(mat.data)
    from scipy.sparse.linalg import svds

    rng = np.random.default_rng(seed)
    U, S, Vt = svds(mat, k=dim, v0=rng.random(min(mat.shape)))
    root = np.sqrt(S)
    user_emb = (U * root).astype(np.float32)
    item_emb = (Vt.T * root).astype(np.float32)

    # Item-item co-visitation: consecutive clicks of the same user (within `covis_window`).
    clicks = clicks.sort_values(["user_id", "ts"], kind="stable")
    cu = clicks["user_id"].to_numpy()
    ci = clicks["item_id"].to_numpy()
    rows, cols = [], []
    for lag in range(1, covis_window + 1):
        same = cu[lag:] == cu[:-lag]
        a, b = ci[:-lag][same], ci[lag:][same]
        keep = a != b
        rows += [a[keep], b[keep]]
        cols += [b[keep], a[keep]]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    co = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_items, n_items))
    co.sum_duplicates()
    pop = np.asarray(co.sum(0)).ravel() + 1.0
    co = sp.csr_matrix(co.multiply(1.0 / np.sqrt(pop)[None, :]))  # damp very popular neighbours
    co = _top_per_row(co, covis_top)

    # Each user's most recent clicks -> one sparse row; user_covis = recent @ co.
    last = clicks.groupby("user_id", sort=False).tail(recent)
    recent_mat = sp.csr_matrix((np.ones(len(last)), (last["user_id"].to_numpy(), last["item_id"].to_numpy())),
                               shape=(n_users, n_items))
    user_covis = (recent_mat @ co).tocsr()
    user_covis.sort_indices()

    return OfflineState(
        feature_days=(lo, hi), item_cat=item_cat, item_created=items["created_day"].to_numpy(),
        item_log_price=np.log(items["price"].to_numpy()), item_ctr=item_ctr,
        item_log_imps_per_day=np.log1p(item_imps / n_days), popular=np.argsort(-item_clicks, kind="stable"),
        user_ctr=user_ctr, user_log_imps_per_day=np.log1p(user_imps / n_days),
        user_cat_share=user_cat_share, user_emb=user_emb, item_emb=item_emb, user_covis=user_covis,
        user_seen=user_imps > 0, global_ctr=global_ctr, n_categories=n_cat)


def _top_per_row(mat: sp.csr_matrix, top: int) -> sp.csr_matrix:
    mat = mat.tocsr()
    rows, cols, vals = [], [], []
    for r in range(mat.shape[0]):
        s, e = mat.indptr[r], mat.indptr[r + 1]
        if e == s:
            continue
        d, idx = mat.data[s:e], mat.indices[s:e]
        keep = np.argsort(-d, kind="stable")[:top]
        rows.append(np.full(len(keep), r))
        cols.append(idx[keep])
        vals.append(d[keep])
    if not rows:
        return sp.csr_matrix(mat.shape)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=mat.shape)


def pair_features(state: OfflineState, user_idx, item_idx, day, position=0) -> np.ndarray:
    """Feature matrix (n, len(FEATURES)) for (user, item) pairs - vectorised, for offline use."""
    u = np.asarray(user_idx)
    i = np.asarray(item_idx)
    day = np.broadcast_to(np.asarray(day), u.shape)
    pos = np.broadcast_to(np.asarray(position, dtype=float), u.shape)
    mf = np.einsum("nd,nd->n", state.user_emb[u], state.item_emb[i])
    covis = np.asarray(state.user_covis[u, i]).ravel()
    cols = [mf, covis, state.item_ctr[i], state.item_log_imps_per_day[i], state.item_log_price[i],
            np.clip(day - state.item_created[i], 0, 60), state.user_ctr[u], state.user_log_imps_per_day[u],
            state.user_cat_share[u, state.item_cat[i]], pos]
    return np.column_stack(cols).astype(np.float32)


def train_ranker(imps: pd.DataFrame, items: pd.DataFrame, n_users: int,
                 feature_days: tuple[int, int] = (0, 34), label_days: tuple[int, int] = (35, 41),
                 max_rows: int = 300_000, seed: int = 0, n_estimators: int = 150):
    """Point-in-time ranker: features from ``feature_days``, click labels from ``label_days``.

    ``position`` is a training feature (it explains part of every click); at
    serving time it is fixed to 0, asking "how likely is a click in the top slot".
    Returns (state used for labels, fitted LGBMClassifier, training frame).
    """
    import lightgbm as lgb

    assert feature_days[1] < label_days[0], "labels must come strictly after the feature window"
    state = build_state(imps, items, n_users, feature_days, seed=seed)
    lab = imps[(imps["day"] >= label_days[0]) & (imps["day"] <= label_days[1])]
    if len(lab) > max_rows:
        req = lab["request_id"].unique()
        keep = np.random.default_rng(seed).choice(req, max_rows // 10, replace=False)
        lab = lab[lab["request_id"].isin(keep)]
    X = pair_features(state, lab["user_id"].to_numpy(), lab["item_id"].to_numpy(), lab["day"].to_numpy(),
                      lab["position"].to_numpy())
    y = lab["clicked"].to_numpy()
    model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
                               min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.9, random_state=seed, n_jobs=2, verbose=-1)
    model.fit(pd.DataFrame(X, columns=FEATURES), y)
    frame = pd.DataFrame(X, columns=FEATURES).assign(clicked=y, request_id=lab["request_id"].to_numpy())
    return state, model, frame


def score_with(model, X: np.ndarray) -> np.ndarray:
    """Raw scores (log-odds) from an LGBMClassifier, single-threaded (serving style)."""
    return model.booster_.predict(X, raw_score=True, num_threads=1)


# ---------------------------------------------------------------------------
# Small serving building blocks
# ---------------------------------------------------------------------------
class LRUCache:
    """Least-recently-used cache with hit statistics. O(1) get/put (OrderedDict)."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return default

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_EXECUTOR = cf.ThreadPoolExecutor(max_workers=32, thread_name_prefix="ranker")


def call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Run ``fn`` in a worker thread; return (result, status) with status ok/error/timeout.

    A timed-out call keeps running in its thread (Python cannot kill a thread):
    the *request* is protected, the CPU is not. Real services also cap the
    work itself (smaller candidate sets, cheaper model) for that reason.
    """
    future = _EXECUTOR.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s), "ok"
    except cf.TimeoutError:
        return None, "timeout"
    except Exception:
        return None, "error"


def business_rules(ranked: np.ndarray, item_cat: np.ndarray, item_created: np.ndarray, day: int,
                   max_per_category: int, k: int) -> tuple[np.ndarray, dict]:
    """Dedupe, drop unlaunched items, cap per category (backfilling if the cap empties the list)."""
    seen: set[int] = set()
    per_cat: dict[int, int] = {}
    out, capped = [], []
    counts = {"duplicate": 0, "unlaunched": 0, "category_cap": 0}
    for item in ranked.tolist():
        if item in seen:
            counts["duplicate"] += 1
            continue
        seen.add(item)
        if item_created[item] > day:
            counts["unlaunched"] += 1
            continue
        cat = int(item_cat[item])
        if per_cat.get(cat, 0) >= max_per_category:
            counts["category_cap"] += 1
            capped.append(item)
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        out.append(item)
        if len(out) == k:
            break
    if len(out) < k:
        out += capped[: k - len(out)]
    return np.array(out, dtype=np.int64), counts


class OnlineStore:
    """A stand-in for a key-value feature store (Redis, DynamoDB, a feature-store API).

    User rows sit behind a simulated network round trip (``latency_s`` per
    call, default 1 ms); item features are one batched call per request.
    Rows are plain dicts, as they would be after deserialising a response.
    """

    def __init__(self, state: OfflineState, latency_s: float = 0.001):
        self.latency_s = latency_s
        self.calls = 0
        self.users = {
            int(u): {"ctr": float(state.user_ctr[u]), "log_imps_per_day": float(state.user_log_imps_per_day[u]),
                     "cat_share": state.user_cat_share[u], "emb": state.user_emb[u]}
            for u in np.flatnonzero(state.user_seen)
        }
        self.items = {"ctr": state.item_ctr, "log_imps_per_day": state.item_log_imps_per_day,
                      "log_price": state.item_log_price, "created": state.item_created, "cat": state.item_cat}

    def _wait(self):
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)

    def get_user(self, user_id: int) -> dict | None:
        self._wait()
        return self.users.get(int(user_id))

    def get_items(self, item_ids: np.ndarray) -> dict:
        self._wait()
        return {k: v[item_ids] for k, v in self.items.items()}


# ---------------------------------------------------------------------------
# The cascade
# ---------------------------------------------------------------------------
class Cascade:
    """One request in, top-k items out, with per-stage timings and fallbacks recorded."""

    STAGES = ("retrieve", "features", "rank", "rules")

    def __init__(self, state: OfflineState, ranker=None, store: OnlineStore | None = None,
                 n_ann: int = 200, n_covis: int = 100, n_pop: int = 100, k: int = 10,
                 max_per_category: int = 3, rank_timeout_s: float = 0.05, user_cache_size: int = 2048,
                 use_ranker: bool = True):
        import faiss

        self.state, self.ranker = state, ranker
        self.store = store or OnlineStore(state)
        self.n_ann, self.n_covis, self.n_pop, self.k = n_ann, n_covis, n_pop, k
        self.max_per_category = max_per_category
        self.rank_timeout_s = rank_timeout_s
        self.use_ranker = use_ranker and ranker is not None
        self.index = faiss.IndexFlatIP(state.item_emb.shape[1])
        self.index.add(np.ascontiguousarray(state.item_emb))
        self.user_cache = LRUCache(user_cache_size)
        self.popular_cache = LRUCache(64)
        self.pop_rank = np.empty(state.n_items)
        self.pop_rank[state.popular] = np.arange(state.n_items)
        self.default_user = {"ctr": state.global_ctr, "log_imps_per_day": 0.0,
                             "cat_share": np.full(state.n_categories, 1.0 / state.n_categories, np.float32),
                             "emb": None}
        self.faults = {"ranker_error": False, "ranker_sleep_s": 0.0, "store_down": False}
        self.counters = {"requests": 0, "ranker_ok": 0, "ranker_error": 0, "ranker_timeout": 0,
                         "user_default": 0}
        self._lock = threading.Lock()

    # -- stages ---------------------------------------------------------------
    def _user(self, user_id: int) -> tuple[dict, bool]:
        row = self.user_cache.get(user_id)
        if row is None:
            row = None if self.faults["store_down"] else self.store.get_user(user_id)
            if row is not None:
                self.user_cache.put(user_id, row)
        if row is None:
            return self.default_user, True
        return row, False

    def _popular(self, day: int) -> np.ndarray:
        cands = self.popular_cache.get(day)
        if cands is None:
            pop = self.state.popular
            cands = pop[self.state.item_created[pop] <= day][: self.n_pop]
            self.popular_cache.put(day, cands)
        return cands

    def retrieve(self, user_id: int, day: int, user: dict) -> tuple[np.ndarray, dict]:
        parts = []
        sources = {}
        if user["emb"] is not None and float(np.abs(user["emb"]).sum()) > 0:
            _, ids = self.index.search(user["emb"][None, :], self.n_ann)
            parts.append(ids[0][ids[0] >= 0])
            sources["ann"] = len(parts[-1])
        if 0 <= user_id < self.state.n_users:
            row = self.state.user_covis[user_id]
            if row.nnz:
                top = row.indices[np.argsort(-row.data, kind="stable")[: self.n_covis]]
                parts.append(top)
                sources["covis"] = len(top)
        parts.append(self._popular(day))
        sources["popular"] = len(parts[-1])
        cands = np.concatenate(parts)
        _, first = np.unique(cands, return_index=True)
        return cands[np.sort(first)], sources

    def features(self, user_id: int, user: dict, cands: np.ndarray, day: int) -> np.ndarray:
        it = self.store.get_items(cands)
        n = len(cands)
        emb = user["emb"]
        mf = self.state.item_emb[cands] @ emb if emb is not None else np.zeros(n, np.float32)
        if 0 <= user_id < self.state.n_users:
            covis = np.asarray(self.state.user_covis[np.full(n, user_id), cands]).ravel()
        else:
            covis = np.zeros(n)
        cols = [mf, covis, it["ctr"], it["log_imps_per_day"], it["log_price"],
                np.clip(day - it["created"], 0, 60), np.full(n, user["ctr"]),
                np.full(n, user["log_imps_per_day"]), user["cat_share"][it["cat"]], np.zeros(n)]
        return np.column_stack(cols).astype(np.float32)

    def _rank(self, X: np.ndarray) -> np.ndarray:
        if self.faults["ranker_sleep_s"]:
            time.sleep(self.faults["ranker_sleep_s"])
        if self.faults["ranker_error"]:
            raise RuntimeError("injected ranker failure")
        return score_with(self.ranker, X)

    # -- one request ----------------------------------------------------------
    def recommend(self, user_id: int, day: int, k: int | None = None) -> dict:
        k = k or self.k
        t = {}
        start = time.perf_counter()
        user, defaulted = self._user(user_id)
        cands, sources = self.retrieve(user_id, day, user)
        t["retrieve"] = time.perf_counter()
        X = self.features(user_id, user, cands, day)
        t["features"] = time.perf_counter()
        status = "skipped"
        if self.use_ranker:
            scores, status = call_with_timeout(self._rank, self.rank_timeout_s, X)
        if status != "ok":  # fallback: popularity order of the same candidates
            scores = -self.pop_rank[cands]
        order = cands[np.argsort(-scores, kind="stable")]
        t["rank"] = time.perf_counter()
        items, dropped = business_rules(order, self.state.item_cat, self.state.item_created, day,
                                        self.max_per_category, k)
        t["rules"] = time.perf_counter()
        timings, prev = {}, start
        for stage in self.STAGES:
            timings[stage] = (t[stage] - prev) * 1000
            prev = t[stage]
        timings["total"] = (t["rules"] - start) * 1000
        with self._lock:
            self.counters["requests"] += 1
            self.counters["user_default"] += int(defaulted)
            if status != "skipped":
                self.counters[f"ranker_{status}"] += 1
        return {"user_id": int(user_id), "day": int(day), "items": items.tolist(),
                "n_candidates": int(len(cands)), "sources": sources, "ranker": status,
                "user_features": "default" if defaulted else "store", "dropped": dropped,
                "timings_ms": {s: round(v, 3) for s, v in timings.items()}}


# ---------------------------------------------------------------------------
# HTTP service
# ---------------------------------------------------------------------------
class RecRequest(BaseModel):
    """Body of POST /recommend. Module level on purpose: with ``from __future__ import
    annotations`` FastAPI resolves annotations by name, so a class (or ``Response``)
    imported inside ``create_app`` would silently become a required query parameter."""

    user_id: int = Field(ge=0)
    day: int = Field(ge=0)
    k: int = Field(default=10, ge=1, le=50)


def create_app(cascade: Cascade):
    """App factory: a FastAPI service around one :class:`Cascade` (no global state)."""
    from collections import deque

    app = FastAPI(title="ShopStream ranking cascade", version="1.0.0")
    app.state.cascade = cascade
    lat: dict[str, deque] = {s: deque(maxlen=20_000) for s in (*Cascade.STAGES, "total")}
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/recommend")
    def recommend(req: RecRequest, response: Response):
        out = cascade.recommend(req.user_id, req.day, req.k)
        tm = out["timings_ms"]
        with lock:
            for s, v in tm.items():
                lat[s].append(v)
        # W3C Server-Timing: browsers' dev tools and many proxies display it.
        response.headers["server-timing"] = ", ".join(f"{s};dur={v:.2f}" for s, v in tm.items())
        response.headers["x-ranker-status"] = out["ranker"]
        return out

    @app.get("/metrics")
    def metrics():
        with lock:
            pct = {s: {q: (round(float(np.percentile(list(v), q)), 3) if v else None) for q in (50, 95, 99)}
                   for s, v in lat.items()}
        return {"latency_ms": pct, "counters": dict(cascade.counters),
                "user_cache_hit_rate": round(cascade.user_cache.hit_rate, 4),
                "popular_cache_hit_rate": round(cascade.popular_cache.hit_rate, 4)}

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class BackgroundServer:
    """Run a uvicorn server in a daemon thread (works inside Jupyter). Use as a context manager."""

    def __init__(self, app, port: int | None = None, log_level: str = "warning"):
        import uvicorn

        self.port = port or free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                                    log_level=log_level, access_log=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self, timeout_s: float = 10.0) -> BackgroundServer:
        self.thread.start()
        deadline = time.time() + timeout_s
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ---------------------------------------------------------------------------
# Offline evaluation against the hidden truth
# ---------------------------------------------------------------------------
def dcg_rows(rel: np.ndarray) -> np.ndarray:
    """Row-wise DCG with exponential gain for a (n, k) relevance matrix (k = list length)."""
    disc = 1.0 / np.log2(np.arange(2, rel.shape[1] + 2))
    return ((2.0 ** rel - 1.0) * disc).sum(1)


def true_prob_matrix(world, user_ids, days) -> np.ndarray:
    """(n_requests, n_items) true click probabilities; items not launched on that day get -inf."""
    u = np.asarray(user_ids)
    d = np.asarray(days)
    created = world.items["created_day"].to_numpy()
    logits = 1.6 * (world.user_vec[u] @ world.item_vec.T)
    it = world.items
    logits += (0.5 * it["quality"].to_numpy() - 0.25 * np.log(it["price"].to_numpy()) - 1.2)[None, :]
    p = 1.0 / (1.0 + np.exp(-logits))
    # same formula as World.true_logit, vectorised over the catalogue; checked against it:
    assert np.allclose(p[0, :5], world.true_click_prob(np.full(5, u[0]), np.arange(5)))
    return np.where(created[None, :] <= d[:, None], p, -np.inf)


def catalogue_ndcg(P: np.ndarray, rec_lists, k: int = 10) -> np.ndarray:
    """NDCG@k of each recommended list against the best k LAUNCHED items in the whole catalogue.

    ``P`` comes from :func:`true_prob_matrix` (position-free truth, one row per
    request). 1.0 means the list is that user's true top-k. Unlaunched or
    missing slots count as relevance 0.
    """
    ideal = -np.sort(-P, axis=1)[:, :k]
    got = np.zeros((len(P), k))
    for j, rec in enumerate(rec_lists):
        rec = np.asarray(rec[:k], dtype=int)
        if len(rec):
            got[j, : len(rec)] = np.maximum(P[j, rec], 0.0)
    return dcg_rows(got) / dcg_rows(ideal)


def ndcg_by_request(scores: np.ndarray, rel: np.ndarray, list_len: int = 10, k: int = 10) -> np.ndarray:
    """NDCG@k for logged requests stored contiguously, ``list_len`` rows each.

    Re-ranks each request's rows by ``scores`` (stable) and grades with ``rel``
    (clicks or true probabilities). Requests whose relevance is all zero get NaN.
    """
    s = np.asarray(scores, float).reshape(-1, list_len)
    r = np.asarray(rel, float).reshape(-1, list_len)
    order = np.argsort(-s, axis=1, kind="stable")
    got = np.take_along_axis(r, order, axis=1)[:, :k]
    ideal = -np.sort(-r, axis=1)[:, :k]
    num, den = dcg_rows(got), dcg_rows(ideal)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def grouped_bootstrap(values, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Mean of ``values`` with a percentile CI from resampling whole ``groups`` (users, requests).

    Rows of one group are correlated (one user's requests), so resampling rows
    would understate the uncertainty. Returns (mean, lo, hi). Vectorised: each
    replicate is sum(values in drawn groups) / count(rows in drawn groups).
    """
    v = np.asarray(values, dtype=float)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(g, weights=v)
    counts = np.bincount(g).astype(float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for lo in range(0, n_boot, 200):  # chunks keep memory small for many groups
        hi = min(lo + 200, n_boot)
        idx = rng.integers(0, len(sums), (hi - lo, len(sums)))
        stats[lo:hi] = sums[idx].sum(1) / counts[idx].sum(1)
    return float(v.mean()), float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
