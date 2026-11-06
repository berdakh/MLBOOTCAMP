"""ShopStream: a synthetic marketplace with ranked feeds, clicks and purchases.

Why synthetic (again)
---------------------
Real click logs have one fatal teaching flaw: you never know the truth. You
cannot check whether a position-bias correction recovered the *real*
relevance, or whether an offline metric agrees with what users actually
prefer. ShopStream keeps the truth: every user and item has latent vectors, and
:class:`World` exposes the true click probability of any (user, item) pair.
So Level 2 can grade methods against reality, not just against other methods.

The generating process
----------------------
* **Items** belong to categories, have a price, a quality and a latent vector
  near their category centroid. Some appear only later (cold start).
* **Users** have a latent taste vector, an activity rate and a home segment.
* Each **request** (a feed load) shows ``K = 10`` items chosen by the
  production *logging policy*: a noisy mix of popularity and a weak affinity
  model. A small fraction of requests (``random_fraction``) is *exploration
  traffic*: items shown in uniformly random order - the only unbiased data.
* A user **examines** position ``p`` with probability ``(1 + p) ** -eta``
  (position bias) and **clicks** an examined item with its true relevance
  probability. A click leads to a **purchase** with a price-dependent
  probability.

Nothing here describes a real company or real people.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

K = 10  # items per feed request


@dataclass(frozen=True)
class WorldConfig:
    n_users: int = 10_000
    n_items: int = 4_000
    n_categories: int = 20
    dim: int = 16
    n_days: int = 56
    requests_per_user_day: float = 0.35
    candidates_per_request: int = 200
    random_fraction: float = 0.05
    eta: float = 0.9               # position-bias strength
    seed: int = 2025
    start: str = "2025-03-03"      # a Monday

    @classmethod
    def scale(cls, name: str) -> WorldConfig:
        """Presets: 'tiny' for tests, 'small' default (~2.4M impressions), 'large' for Day 1."""
        presets = {
            "tiny": dict(n_users=600, n_items=300, n_days=14),
            "small": {},
            "large": dict(n_users=200_000, n_items=20_000, requests_per_user_day=0.3),
        }
        return cls(**presets[name])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class World:
    """The hidden truth behind the logs. Build with :func:`make_world`."""

    config: WorldConfig
    users: pd.DataFrame
    items: pd.DataFrame
    user_vec: np.ndarray = field(repr=False)
    item_vec: np.ndarray = field(repr=False)

    def true_logit(self, user_idx, item_idx) -> np.ndarray:
        """Relevance logit of (user, item) pairs - position-free, the ground truth."""
        u = self.user_vec[np.asarray(user_idx)]
        v = self.item_vec[np.asarray(item_idx)]
        it = self.items
        quality = it["quality"].to_numpy()[np.asarray(item_idx)]
        logprice = np.log(it["price"].to_numpy()[np.asarray(item_idx)])
        return 1.6 * np.sum(u * v, axis=-1) + 0.5 * quality - 0.25 * logprice - 1.2

    def true_click_prob(self, user_idx, item_idx) -> np.ndarray:
        """P(click | examined) - what a perfectly debiased model would predict."""
        return _sigmoid(self.true_logit(user_idx, item_idx))

    def examination_prob(self, position) -> np.ndarray:
        return (1.0 + np.asarray(position, dtype=float)) ** (-self.config.eta)


def make_world(config: WorldConfig | None = None) -> World:
    cfg = config or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    cats = np.arange(cfg.n_categories)
    centroids = rng.normal(0, 1, (cfg.n_categories, cfg.dim)) / np.sqrt(cfg.dim) * 1.6

    item_cat = rng.choice(cats, size=cfg.n_items, p=rng.dirichlet(np.ones(cfg.n_categories) * 2))
    item_vec = centroids[item_cat] + rng.normal(0, 0.35, (cfg.n_items, cfg.dim)) / np.sqrt(cfg.dim) * 1.6
    base_price = np.exp(rng.normal(3.0, 0.6, cfg.n_categories))
    price = np.round(base_price[item_cat] * np.exp(rng.normal(0, 0.35, cfg.n_items)), 2)
    quality = rng.normal(0, 1, cfg.n_items)
    # 15% of items launch during the window: cold-start items.
    created = np.where(rng.random(cfg.n_items) < 0.15, rng.integers(7, cfg.n_days, cfg.n_items), 0)
    items = pd.DataFrame({
        "item_id": np.arange(cfg.n_items),
        "category": item_cat,
        "price": price,
        "quality": quality,
        "created_day": created,
    })

    segment = rng.integers(0, 5, cfg.n_users)
    seg_pref = rng.dirichlet(np.ones(cfg.n_categories) * 0.3, size=5)
    # Users mix 2-3 favourite categories of their segment, plus individual noise.
    fav = np.array([rng.choice(cats, size=3, p=seg_pref[s]) for s in segment])
    w = rng.dirichlet(np.ones(3), size=cfg.n_users)
    user_vec = np.einsum("uk,ukd->ud", w, centroids[fav]) + rng.normal(0, 0.4, (cfg.n_users, cfg.dim)) / np.sqrt(cfg.dim)
    activity = np.exp(rng.normal(0, 0.8, cfg.n_users))
    users = pd.DataFrame({
        "user_id": np.arange(cfg.n_users),
        "segment": segment,
        "country": rng.choice(["US", "GB", "DE", "IN", "BR"], cfg.n_users, p=[0.4, 0.15, 0.15, 0.2, 0.1]),
        "device": rng.choice(["ios", "android", "web"], cfg.n_users, p=[0.35, 0.45, 0.2]),
        "signup_day": np.where(rng.random(cfg.n_users) < 0.2, rng.integers(0, cfg.n_days, cfg.n_users), 0),
        "activity": activity.round(4),
    })
    return World(cfg, users, items, user_vec, item_vec)


def simulate_logs(world: World, chunk_requests: int = 20_000, seed_offset: int = 1):
    """Yield impression DataFrames chunk by chunk (one row per shown item).

    Columns: request_id, ts, day, user_id, item_id, position, is_random,
    policy_score, examined (hidden truth - drop before training!), clicked,
    purchased.
    """
    cfg = world.config
    rng = np.random.default_rng(cfg.seed + seed_offset)
    start = pd.Timestamp(cfg.start)
    items = world.items
    popularity = _sigmoid(items["quality"].to_numpy()) * np.exp(rng.normal(0, 0.5, cfg.n_items))
    # The production policy's own (weak, noisy) view of user taste.
    policy_user = world.user_vec + rng.normal(0, 0.5, world.user_vec.shape) / np.sqrt(cfg.dim)

    # Requests per user per day ~ Poisson(rate * activity), after signup.
    lam = cfg.requests_per_user_day * world.users["activity"].to_numpy()
    counts = rng.poisson(lam[:, None], (cfg.n_users, cfg.n_days))
    counts[np.arange(cfg.n_days)[None, :] < world.users["signup_day"].to_numpy()[:, None]] = 0
    req_user, req_day = np.nonzero(counts)
    reps = counts[req_user, req_day]
    req_user, req_day = np.repeat(req_user, reps), np.repeat(req_day, reps)
    order = np.lexsort((rng.random(len(req_day)), req_day))
    req_user, req_day = req_user[order], req_day[order]
    seconds = rng.integers(0, 86_400, len(req_day))
    n_req = len(req_user)

    created = items["created_day"].to_numpy()
    for lo in range(0, n_req, chunk_requests):
        hi = min(lo + chunk_requests, n_req)
        u, d = req_user[lo:hi], req_day[lo:hi]
        n = hi - lo
        # Candidate pool: random eligible items (launched by that day).
        cand = rng.integers(0, cfg.n_items, (n, cfg.candidates_per_request))
        eligible = created[cand] <= d[:, None]
        affinity = np.einsum("nd,ncd->nc", policy_user[u], world.item_vec[cand])
        score = np.log(popularity[cand]) + 0.8 * affinity + rng.gumbel(0, 0.7, cand.shape)
        score[~eligible] = -np.inf
        top = np.argsort(-score, axis=1)[:, :K]
        shown = np.take_along_axis(cand, top, axis=1)
        pscore = np.take_along_axis(score, top, axis=1)
        is_random = rng.random(n) < cfg.random_fraction
        if is_random.any():
            # Exploration: K eligible items in uniformly random order.
            r = np.flatnonzero(is_random)
            rnd_score = np.where(eligible[r], rng.random((len(r), cfg.candidates_per_request)), -1.0)
            rtop = np.argsort(-rnd_score, axis=1)[:, :K]
            shown[r] = np.take_along_axis(cand[r], rtop, axis=1)
            pscore[r] = np.nan
        pos = np.broadcast_to(np.arange(K), (n, K))
        uu = np.repeat(u, K).reshape(n, K)
        exam = rng.random((n, K)) < world.examination_prob(pos)
        p_click = world.true_click_prob(uu, shown)
        clicked = exam & (rng.random((n, K)) < p_click)
        price = items["price"].to_numpy()[shown]
        p_buy = _sigmoid(-0.6 - 0.35 * np.log(price) + 0.4 * items["quality"].to_numpy()[shown])
        purchased = clicked & (rng.random((n, K)) < p_buy)
        ts = start + pd.to_timedelta(d * 86_400 + seconds[lo:hi], unit="s")
        yield pd.DataFrame({
            "request_id": np.repeat(np.arange(lo, hi), K),
            "ts": np.repeat(ts.values, K),
            "day": np.repeat(d, K).astype(np.int16),
            "user_id": uu.ravel().astype(np.int32),
            "item_id": shown.ravel().astype(np.int32),
            "position": pos.ravel().astype(np.int8),
            "is_random": np.repeat(is_random, K),
            "policy_score": pscore.ravel().astype(np.float32),
            "examined": exam.ravel(),
            "clicked": clicked.ravel(),
            "purchased": purchased.ravel(),
        })


HIDDEN_COLUMNS = ["examined"]  # simulation truth; never a feature, never in a "real" log


def write_dataset(world: World, directory: Path, partition_by_day: bool = False) -> Path:
    """Write users, items and impressions as Parquet (impressions optionally partitioned)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    world.users.to_parquet(directory / "users.parquet", index=False)
    world.items.to_parquet(directory / "items.parquet", index=False)
    imp_dir = directory / "impressions"
    imp_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(simulate_logs(world)):
        if partition_by_day:
            for day, part in chunk.groupby("day"):
                pdir = imp_dir / f"day={int(day)}"  # unpadded: DuckDB infers an integer
                pdir.mkdir(exist_ok=True)
                part.drop(columns="day").to_parquet(pdir / f"part-{i:04d}.parquet", index=False)
        else:
            chunk.to_parquet(imp_dir / f"part-{i:04d}.parquet", index=False)
    return directory


def load_impressions(directory: Path, columns=None) -> pd.DataFrame:
    """Read all impression parts (works for flat or day-partitioned layouts)."""
    import duckdb

    pattern = str(Path(directory) / "impressions" / "**" / "*.parquet")
    cols = "*" if columns is None else ", ".join(columns)
    con = duckdb.connect()
    try:
        df = con.execute(
            f"SELECT {cols} FROM read_parquet('{pattern}', hive_partitioning=true) ORDER BY request_id, position"
        ).df()
    finally:
        con.close()
    if "day" in df.columns:
        df["day"] = df["day"].astype("int16")
    return df
