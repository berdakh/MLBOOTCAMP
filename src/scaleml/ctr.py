"""A synthetic, Criteo-like ads world: impressions, clicks and delayed conversions.

Real click logs (Criteo, Avazu) cannot tell you the *true* click probability
of an impression, so you can never say how close a model is to the best
possible one. This generator can. It draws impressions with

* 13 dense count features ``I1..I13`` (heavy-tailed, ~10% missing), and
* 26 categorical fields ``C1..C26`` whose cardinalities follow a power law from a
  handful of values up to ``max_card`` (default 1,000,000); values within a
  field are Zipf-distributed (a few head values, a very long tail), and are
  reported as opaque 32-bit ids, like Criteo's hashed hex strings;
* a timestamp over ``n_days`` days (with a time-of-day effect);

and a hidden logit made of a bias, non-linear dense effects, per-category
effects (only some fields matter), pairwise **feature crosses** (field × field
and dense × field) that a linear model cannot represent, and noise-free
Bernoulli clicks. The bias is solved so the average CTR is ``base_ctr`` (~3%).

Clicked impressions convert with a second, partly shared logit (base rate
``base_cvr``) after an exponentially distributed **delay**, which is what
makes "recent clicks look like non-converting clicks" (delayed feedback).

Per-category effects are not stored: they are computed on the fly from a
deterministic 64-bit hash of (seed, field, value), so a field with a million
values costs no memory and the same value always has the same effect.

    world = CTRWorld(CTRConfig(seed=0))
    df = world.sample(200_000, seed=1)        # pandas DataFrame
    df[["true_ctr", "click", "true_cvr", "conversion", "conv_delay_h"]]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DENSE = [f"I{i}" for i in range(1, 14)]
CATEGORICAL = [f"C{i}" for i in range(1, 27)]
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


# --------------------------------------------------------------------------- hashing
def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser: a fast, well-mixed, *stable* 64-bit hash.

    Unlike Python's ``hash()`` on strings (salted per process via
    ``PYTHONHASHSEED``), the output depends only on the input, on every machine.
    """
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _combine(*parts) -> np.ndarray:
    h = np.uint64(0)
    with np.errstate(over="ignore"):
        for p in parts:
            h = splitmix64(np.asarray(p, dtype=np.uint64) ^ (h * np.uint64(0x100000001B3)))
    return h


def _unit_normal(h: np.ndarray) -> np.ndarray:
    """Map 64-bit hashes to N(0, 1) draws (Box-Muller on the two 32-bit halves)."""
    u1 = ((h >> np.uint64(32)).astype(np.float64) + 0.5) / 2**32
    u2 = ((h & np.uint64(0xFFFFFFFF)).astype(np.float64) + 0.5) / 2**32
    return np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)


def hash_columns(cat: np.ndarray, n_buckets: int, field_aware: bool = True, seed: int = 0) -> np.ndarray:
    """The hashing trick, vectorised: raw ids ``(n, F)`` -> bucket ids in ``[0, n_buckets)``.

    ``field_aware`` mixes the field index into the hash so that value 123 in
    field C1 and value 123 in field C2 are different features (they usually
    land in different buckets of the one shared table).
    """
    cat = np.asarray(cat, dtype=np.uint64)
    fields = np.arange(cat.shape[1], dtype=np.int64)[None, :] if field_aware else -1
    return hash_field(cat, fields, n_buckets, seed)


def hash_field(values, field, n_buckets: int, seed: int = 0) -> np.ndarray:
    """Bucket ids of raw ``values`` of field number ``field`` (0-based; broadcastable) - as in ``hash_columns``."""
    f = np.asarray(field, dtype=np.int64) + 1
    h = _combine(np.uint64(seed), f.astype(np.uint64), np.asarray(values, dtype=np.uint64))
    return (h % np.uint64(n_buckets)).astype(np.int64)


def collision_rate(cat: np.ndarray, n_buckets: int) -> float:
    """Share of distinct (field, value) keys in ``cat`` that share their bucket with another key."""
    buckets = np.concatenate([hash_field(np.unique(cat[:, f]), f, n_buckets) for f in range(cat.shape[1])])
    _, inverse, counts = np.unique(buckets, return_inverse=True, return_counts=True)
    return float(np.mean(counts[inverse] > 1))


# --------------------------------------------------------------------------- the world
@dataclass(frozen=True)
class CTRConfig:
    seed: int = 0
    n_dense: int = 13
    n_cat: int = 26
    min_card: int = 4
    max_card: int = 1_000_000
    zipf: float = 1.15            # popularity exponent within a field (bigger = more head-heavy)
    base_ctr: float = 0.03
    base_cvr: float = 0.10        # P(conversion | click)
    mean_delay_days: float = 2.0  # mean click -> conversion delay
    n_days: int = 14
    signal: float = 1.0           # scales every feature effect (bigger = more predictable)
    cvr_ctr_corr: float = 0.3     # how much of the CTR signal the CVR logit shares (can be negative)

    def as_dict(self) -> dict:
        return asdict(self)


class CTRWorld:
    """Ground truth for a synthetic ads log. ``sample()`` draws impressions."""

    def __init__(self, config: CTRConfig | None = None):
        self.cfg = cfg = config or CTRConfig()
        rng = np.random.default_rng(cfg.seed)
        # power-law cardinalities from min_card to max_card, in random field order
        cards = np.unique(np.round(np.logspace(np.log10(cfg.min_card), np.log10(cfg.max_card), cfg.n_cat)))
        cards = np.resize(cards, cfg.n_cat).astype(np.int64)
        self.cardinalities = rng.permutation(cards)
        # per-field effect scales: about half the fields carry signal
        scale = rng.choice([0.0, 0.15, 0.35, 0.6], size=cfg.n_cat, p=[0.35, 0.25, 0.25, 0.15])
        self.cat_scale = scale * cfg.signal
        self.dense_mu = rng.uniform(0.0, 3.0, cfg.n_dense)
        self.dense_sigma = rng.uniform(0.6, 1.4, cfg.n_dense)
        self.dense_w = rng.normal(0, 0.25, cfg.n_dense) * cfg.signal
        self.dense_missing = rng.uniform(0.0, 0.2, cfg.n_dense)
        informative = np.flatnonzero(scale > 0.2)
        pick = rng.permutation(informative)
        # crosses: (field a, field b, scale) and (dense j, field c, scale)
        self.cat_crosses = [(int(pick[0]), int(pick[1]), 0.5 * cfg.signal),
                            (int(pick[2]), int(pick[3 % len(pick)]), 0.4 * cfg.signal),
                            (int(pick[0]), int(pick[4 % len(pick)]), 0.3 * cfg.signal)]
        self.dense_crosses = [(int(rng.integers(cfg.n_dense)), int(pick[1]), 0.35 * cfg.signal),
                              (int(rng.integers(cfg.n_dense)), int(pick[2]), 0.3 * cfg.signal)]
        self.dense_pair = (0, 1, 0.25 * cfg.signal)
        # the conversion logit: shares part of the CTR signal, plus its own fields
        self.cvr_scale = rng.choice([0.0, 0.3, 0.6], size=cfg.n_cat, p=[0.6, 0.25, 0.15]) * cfg.signal
        self.cvr_dense_w = rng.normal(0, 0.2, cfg.n_dense) * cfg.signal
        # value of a conversion (e.g. basket size) depends on one field
        self.value_field = int(rng.integers(cfg.n_cat))
        self.bias = 0.0
        self.cvr_bias = 0.0
        calib = self.sample(200_000, seed=cfg.seed + 10_000, _calibrating=True)
        self.bias = _solve_bias(calib["_ctr_logit"].to_numpy(), cfg.base_ctr)
        self.cvr_bias = _solve_bias(calib["_cvr_logit"].to_numpy(), cfg.base_cvr,
                                    weights=_sigmoid(calib["_ctr_logit"].to_numpy() + self.bias))

    # -- feature effects -----------------------------------------------------
    def _cat_effect(self, f: int, values: np.ndarray, salt: int = 1) -> np.ndarray:
        return _unit_normal(_combine(np.uint64(self.cfg.seed), np.uint64(salt), np.uint64(f), values))

    def _draw_categoricals(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        out = np.empty((n, cfg.n_cat), dtype=np.uint64)
        for f, K in enumerate(self.cardinalities):
            u = rng.random(n)
            s = cfg.zipf
            # inverse CDF of a continuous power law on [1, K+1), floored: rank 1 = most popular
            rank = np.floor((((float(K) + 1) ** (1 - s) - 1) * u + 1) ** (1 / (1 - s)))
            rank = np.clip(rank.astype(np.int64), 1, K)
            # opaque 32-bit ids (like Criteo's hashed strings)
            out[:, f] = _combine(np.uint64(cfg.seed + 7), np.uint64(f), rank.astype(np.uint64)) & np.uint64(0xFFFFFFFF)
        return out

    def sample(self, n: int, seed: int = 0, start_day: float = 0.0, n_days: float | None = None,
               _calibrating: bool = False) -> pd.DataFrame:
        """Draw ``n`` impressions (sorted by time) with their clicks and conversions.

        Columns: ``ts`` (seconds since day 0), ``day``, ``I1..I13`` (float, NaN =
        missing), ``C1..C26`` (uint32 ids as int64), ``true_ctr``, ``click``,
        ``true_cvr`` (P(conversion | click), defined for every impression),
        ``conversion`` (only possible if clicked), ``conv_delay_h`` (hours from
        click to conversion; inf if none) and ``value`` (of a conversion).
        """
        cfg = self.cfg
        rng = np.random.default_rng([cfg.seed, seed, 1])
        n_days = cfg.n_days if n_days is None else n_days
        ts = np.sort(rng.uniform(start_day, start_day + n_days, n)) * 86_400.0
        hour = (ts / 3600.0) % 24
        counts = np.floor(np.exp(rng.normal(self.dense_mu, self.dense_sigma, (n, cfg.n_dense))))
        missing = rng.random((n, cfg.n_dense)) < self.dense_missing
        z = (np.log1p(counts) - np.log1p(np.exp(self.dense_mu))) / (self.dense_sigma + 0.5)
        z_eff = np.where(missing, 0.0, z)
        cat = self._draw_categoricals(n, rng)

        logit = np.tanh(z_eff) @ self.dense_w + 0.3 * np.sin(2 * np.pi * (hour - 15) / 24) * cfg.signal
        a, b, s = self.dense_pair
        logit += s * np.tanh(z_eff[:, a] * z_eff[:, b])
        for f in range(cfg.n_cat):
            if self.cat_scale[f] > 0:
                logit += self.cat_scale[f] * self._cat_effect(f, cat[:, f])
        for fa, fb, s in self.cat_crosses:
            pair = _combine(cat[:, fa], cat[:, fb])
            logit += s * self._cat_effect(1000 + fa * 64 + fb, pair, salt=2)
        for j, fc, s in self.dense_crosses:
            logit += s * np.tanh(z_eff[:, j]) * self._cat_effect(2000 + fc, cat[:, fc], salt=3)

        cvr_logit = cfg.cvr_ctr_corr * logit + np.tanh(z_eff) @ self.cvr_dense_w
        for f in range(cfg.n_cat):
            if self.cvr_scale[f] > 0:
                cvr_logit += self.cvr_scale[f] * self._cat_effect(f, cat[:, f], salt=4)

        df = pd.DataFrame({"ts": ts, "day": (ts // 86_400).astype(np.int64)})
        dense = np.where(missing, np.nan, counts)
        for j in range(cfg.n_dense):
            df[DENSE[j]] = dense[:, j]
        for f in range(cfg.n_cat):
            df[CATEGORICAL[f]] = cat[:, f].astype(np.int64)
        if _calibrating:
            df["_ctr_logit"], df["_cvr_logit"] = logit, cvr_logit
            return df
        p = _sigmoid(logit + self.bias)
        q = _sigmoid(cvr_logit + self.cvr_bias)
        click = rng.random(n) < p
        conv = click & (rng.random(n) < q)
        delay = np.where(conv, rng.exponential(cfg.mean_delay_days * 24.0, n), np.inf)
        vmult = np.exp(0.5 * self._cat_effect(self.value_field, cat[:, self.value_field], salt=5))
        df["true_ctr"] = p
        df["click"] = click.astype(np.int8)
        df["true_cvr"] = q
        df["conversion"] = conv.astype(np.int8)
        df["conv_delay_h"] = delay
        df["value"] = np.round(20.0 * vmult, 2)
        return df


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_bias(logit: np.ndarray, target: float, weights: np.ndarray | None = None) -> float:
    """Find b such that the (weighted) mean of sigmoid(logit + b) equals ``target`` (bisection)."""
    w = np.ones_like(logit) if weights is None else weights
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.average(_sigmoid(logit + mid), weights=w) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------------------------------- features
def dense_features(df: pd.DataFrame) -> np.ndarray:
    """``log1p`` of the counts, missing -> 0, plus a missing-indicator per column: ``(n, 26)`` float32."""
    x = df[DENSE].to_numpy(dtype=np.float64)
    miss = np.isnan(x)
    return np.concatenate([np.log1p(np.nan_to_num(x, nan=0.0)), miss], axis=1).astype(np.float32)


def table_bytes(rows: int, dim: int, bytes_per_value: int = 4, optimizer_slots: int = 0) -> int:
    """Memory of an embedding table: rows × dim × bytes, plus optimizer state (e.g. Adam = 2 slots)."""
    return int(rows) * int(dim) * bytes_per_value * (1 + optimizer_slots)


# --------------------------------------------------------------------------- metrics
def log_loss(y, p, eps: float = 1e-7) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def normalized_entropy(y, p, eps: float = 1e-7) -> float:
    """NE (He et al. 2014, Facebook): log loss divided by the entropy of the base rate. Lower is better; 1.0 = predicting the average."""
    base = float(np.clip(np.mean(y), eps, 1 - eps))
    h = -(base * np.log(base) + (1 - base) * np.log(1 - base))
    return log_loss(y, p, eps) / h


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of the prediction: mean predicted vs observed rate and their ratio."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = [{"bin": i, "n": len(b), "mean_pred": p[b].mean(), "observed": y[b].mean()} for i, b in enumerate(bins)]
    out = pd.DataFrame(rows)
    out["pred/obs"] = out["mean_pred"] / out["observed"].where(out["observed"] > 0)
    return out


def recalibrate_downsampled(p, w: float):
    """Undo negative downsampling at rate ``w`` (fraction of negatives kept): q = p / (p + (1 - p) / w)."""
    p = np.asarray(p, dtype=np.float64)
    return p / (p + (1 - p) / w)


# --------------------------------------------------------------------------- click streams
def sample_clicks(world: CTRWorld, n_clicks: int, seed: int = 0, start_day: float = 0.0,
                  n_days: float | None = None, chunk: int = 200_000) -> pd.DataFrame:
    """Only the clicked impressions (what a conversion model trains on), about ``n_clicks`` of them.

    Draws impressions in chunks that each span the whole time window and keeps
    the clicks, so the result is uniform in time; sorted by timestamp.
    """
    parts, got, i = [], 0, 0
    while got < n_clicks:
        df = world.sample(chunk, seed=seed * 1000 + i, start_day=start_day, n_days=n_days)
        df = df[df["click"] == 1]
        parts.append(df)
        got += len(df)
        i += 1
    return pd.concat(parts).sort_values("ts", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- models (PyTorch)
try:  # torch is a Level 3 dependency; keep the data generator importable without it
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def to_tensors(df: pd.DataFrame, n_buckets: int, dense_stats: tuple | None = None, label: str = "click"):
    """Hashed categorical ids, standardised dense features and the label, as tensors.

    Returns ``(data, dense_stats)``: ``data`` has ``idx`` (n, 26) int64,
    ``dense`` (n, 26) float32 and ``y`` (n,) float32. Pass the *training*
    ``dense_stats`` when converting validation/test data (no leakage).
    """
    dense = dense_features(df)
    if dense_stats is None:
        dense_stats = (dense.mean(0), dense.std(0) + 1e-6)
    mu, sd = dense_stats
    data = {"idx": torch.from_numpy(hash_columns(df[CATEGORICAL].to_numpy(), n_buckets)),
            "dense": torch.from_numpy(((dense - mu) / sd).astype(np.float32))}
    if label in df:
        data["y"] = torch.from_numpy(df[label].to_numpy(dtype=np.float32))
    return data, dense_stats


class Batches:
    """Shuffled mini-batches of a dict of tensors; a new order every epoch (``len()`` works)."""

    def __init__(self, data: dict, batch_size: int, seed: int = 0, shuffle: bool = True):
        self.data, self.batch_size, self.seed, self.shuffle, self.epoch = data, batch_size, seed, shuffle, 0
        self.n = len(next(iter(data.values())))

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(self.n, generator=g) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.batch_size):
            b = order[i:i + self.batch_size]
            yield {k: v[b] for k, v in self.data.items()}


if nn is not None:

    class HashedLogReg(nn.Module):
        """Logistic regression on hashed one-hot features: one weight per bucket (a ``dim=1`` embedding)."""

        def __init__(self, n_buckets: int, n_dense: int = 26, base_rate: float | None = None):
            super().__init__()
            self.w = nn.Embedding(n_buckets, 1)
            nn.init.zeros_(self.w.weight)
            self.dense = nn.Linear(n_dense, 1)
            if base_rate is not None:  # start at the base rate's log-odds, not at p = 0.5
                nn.init.constant_(self.dense.bias, float(np.log(base_rate / (1 - base_rate))))

        def forward(self, idx, dense):
            return self.w(idx).sum(dim=(1, 2)) + self.dense(dense).squeeze(-1)

    class DCNv2(nn.Module):
        """Deep & Cross Network v2 (Wang et al. 2021), "lite": full-rank cross layers in parallel with an MLP.

        x0 = [embeddings of the 26 fields, flattened; dense features]
        cross layer:  x_{l+1} = x0 * (W_l x_l + b_l) + x_l      (explicit, bounded-degree feature crosses)
        deep tower:   MLP(x0)                                   (implicit interactions)
        logit = w · [x_L ; MLP(x0)] + b
        """

        def __init__(self, n_buckets: int, emb_dim: int = 8, n_fields: int = 26, n_dense: int = 26,
                     n_cross: int = 2, hidden: tuple = (128, 64), base_rate: float | None = None):
            super().__init__()
            self.emb = nn.Embedding(n_buckets, emb_dim)  # ONE shared table; field-aware hashing separates fields
            nn.init.normal_(self.emb.weight, std=0.01)
            d = n_fields * emb_dim + n_dense
            self.cross = nn.ModuleList(nn.Linear(d, d) for _ in range(n_cross))
            layers, prev = [], d
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            self.deep = nn.Sequential(*layers)
            self.head = nn.Linear(d + prev, 1)
            if base_rate is not None:
                nn.init.constant_(self.head.bias, float(np.log(base_rate / (1 - base_rate))))

        def forward(self, idx, dense):
            x0 = torch.cat([self.emb(idx).flatten(1), dense], dim=1)
            x = x0
            for layer in self.cross:
                x = x0 * layer(x) + x
            return self.head(torch.cat([x, self.deep(x0)], dim=1)).squeeze(-1)


def predict_proba(model, data: dict, device, batch_size: int = 65_536) -> np.ndarray:
    """Probabilities for every row of ``data`` (no grad, eval mode, fp32 output)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data["idx"]), batch_size):
            logits = model(data["idx"][i:i + batch_size].to(device), data["dense"][i:i + batch_size].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(out).astype(np.float64)


# --------------------------------------------------------------------------- multi-task models
if nn is not None:

    def _mlp(d_in: int, hidden: tuple) -> tuple:
        layers, prev = [], d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        return nn.Sequential(*layers), prev

    def _tower(d_in: int, base_rate: float | None) -> nn.Sequential:
        t = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 1))
        if base_rate is not None:
            nn.init.constant_(t[-1].bias, float(np.log(base_rate / (1 - base_rate))))
        return t

    class _EmbedInput(nn.Module):
        def __init__(self, n_buckets: int, emb_dim: int, n_fields: int = 26, n_dense: int = 26):
            super().__init__()
            self.emb = nn.Embedding(n_buckets, emb_dim)
            nn.init.normal_(self.emb.weight, std=0.01)
            self.d = n_fields * emb_dim + n_dense

        def forward(self, idx, dense):
            return torch.cat([self.emb(idx).flatten(1), dense], dim=1)

    class SharedBottom(nn.Module):
        """Hard parameter sharing: embeddings + one MLP shared by all tasks, one small tower per task."""

        def __init__(self, n_buckets: int, emb_dim: int = 8, bottom: tuple = (128,), base_rates=(None, None)):
            super().__init__()
            self.inp = _EmbedInput(n_buckets, emb_dim)
            self.bottom, d = _mlp(self.inp.d, bottom)
            self.towers = nn.ModuleList(_tower(d, b) for b in base_rates)

        def forward(self, idx, dense):
            h = self.bottom(self.inp(idx, dense))
            return [t(h).squeeze(-1) for t in self.towers]

    class MMoE(nn.Module):
        """Multi-gate Mixture-of-Experts (Ma et al. 2018), lite: shared experts, one softmax gate per task.

        task k's input = sum_e gate_k(x)[e] * expert_e(x); each task can weight the
        experts differently, so conflicting tasks need not share one representation.
        """

        def __init__(self, n_buckets: int, emb_dim: int = 8, n_experts: int = 3, expert: tuple = (64,),
                     base_rates=(None, None)):
            super().__init__()
            self.inp = _EmbedInput(n_buckets, emb_dim)
            built = [_mlp(self.inp.d, expert) for _ in range(n_experts)]
            self.experts = nn.ModuleList(b[0] for b in built)
            d = built[0][1]
            self.gates = nn.ModuleList(nn.Linear(self.inp.d, n_experts) for _ in base_rates)
            self.towers = nn.ModuleList(_tower(d, b) for b in base_rates)
            self.last_gates = None

        def forward(self, idx, dense):
            x = self.inp(idx, dense)
            ex = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, d)
            outs, gates = [], []
            for gate, tower in zip(self.gates, self.towers):
                g = torch.softmax(gate(x), dim=-1)  # (B, E)
                gates.append(g.detach())
                outs.append(tower((g.unsqueeze(-1) * ex).sum(1)).squeeze(-1))
            self.last_gates = gates
            return outs


def predict_tasks(model, data: dict, device, batch_size: int = 65_536) -> list[np.ndarray]:
    """Per-task probabilities from a multi-task model returning a list of logits."""
    model.eval()
    outs = None
    with torch.no_grad():
        for i in range(0, len(data["idx"]), batch_size):
            logits = model(data["idx"][i:i + batch_size].to(device), data["dense"][i:i + batch_size].to(device))
            probs = [torch.sigmoid(z.float()).cpu().numpy() for z in logits]
            outs = [[p] for p in probs] if outs is None else [o + [p] for o, p in zip(outs, probs)]
    model.train()
    return [np.concatenate(o).astype(np.float64) for o in outs]
