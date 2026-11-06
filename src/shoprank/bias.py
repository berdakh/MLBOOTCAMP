"""Position bias, propensities and inverse propensity scoring (notebook 05).

The position-based click model (PBM) behind ShopStream:

    P(click | user, item, position p) = P(examine p) * P(click | examined)
                                      =      theta_p   *   relevance(user, item)

Clicks therefore mix two things: *where* an item was shown (theta) and
*whether the user liked it* (relevance). Everything here is about separating
the two:

* estimate theta from exploration traffic, where items land in random
  positions so relevance is independent of position;
* re-weight clicks by 1 / theta (inverse propensity scoring, IPS) to recover
  relevance-level quantities: offline evaluation of a new ranking and
  training a click model.

All matrix helpers assume a frame sorted by (request_id, position) with
exactly ``K`` rows per request, which is how :func:`shoprank.store.load`
returns impressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K = 10


# --------------------------------------------------------------------------- propensities
def ctr_by_position(imps: pd.DataFrame) -> pd.DataFrame:
    """Click rate by position, separately for policy and exploration (random) traffic."""
    t = imps.groupby(["position", "is_random"])["clicked"].mean().unstack("is_random")
    return t.rename(columns={False: "policy", True: "random"})[["policy", "random"]]


def estimate_propensities(random_imps: pd.DataFrame) -> np.ndarray:
    """Examination probability by position, relative to position 0, from RANDOM traffic.

    On exploration traffic every item is equally likely at every position, so
    mean relevance is the same at each position and CTR(p) / CTR(0) =
    theta_p / theta_0.
    """
    ctr = random_imps.groupby("position")["clicked"].mean().sort_index()
    return (ctr / ctr.iloc[0]).to_numpy()


def power_law_propensities(random_imps: pd.DataFrame, k: int = K) -> tuple[float, np.ndarray]:
    """Fit CTR(p) = c * (1 + p) ** -eta on exploration traffic; return (eta, theta with theta_0 = 1).

    Weighted least squares on the log scale (weights = sqrt(clicks) per position):
    one parameter instead of k - 1 noisy ratios, and no reliance on CTR(0) alone.
    """
    g = random_imps.groupby("position")["clicked"].agg(["mean", "sum"]).sort_index()
    slope = np.polyfit(np.log1p(g.index.to_numpy(dtype=float)), np.log(g["mean"].to_numpy()), 1,
                       w=np.sqrt(g["sum"].to_numpy()))[0]
    eta = float(-slope)
    return eta, (1.0 + np.arange(k)) ** -eta


def ips_weights(positions, propensities, clip: float | None = None) -> np.ndarray:
    """1 / theta_position, optionally capped at ``clip``."""
    w = 1.0 / np.asarray(propensities, dtype=float)[np.asarray(positions)]
    return np.minimum(w, clip) if clip is not None else w


def ips_ctr_estimate(clicks, positions, propensities, clip: float | None = None) -> float:
    """Position-debiased click rate: mean of click / theta_position (weights capped at ``clip``).

    Under the PBM this is unbiased (without clipping) for the click rate the
    same items would get *if every impression were examined like position 0*.
    """
    c = np.asarray(clicks, dtype=float)
    return float(np.mean(c * ips_weights(positions, propensities, clip)))


# --------------------------------------------------------------------------- per-request matrices
def matrix(values, k: int = K) -> np.ndarray:
    """Reshape a request-sorted column to (n_requests, k)."""
    return np.asarray(values).reshape(-1, k)


def rank_positions(scores: np.ndarray) -> np.ndarray:
    """New position (0 = top) of every item when each row is sorted by score, descending (stable)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.arange(scores.shape[1])[None, :].repeat(len(scores), 0), axis=1)
    return pos


def ndcg_matrix(scores: np.ndarray, gains: np.ndarray, k: int = K) -> np.ndarray:
    """Per-row NDCG@k (exponential gain 2**g - 1, like :func:`shoprank.metrics.ndcg_at_k`).

    Rows whose gains are all zero are dropped (they carry no ranking information).
    Vectorised: ~100x faster than looping over requests in pandas.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    g = np.take_along_axis(gains.astype(float), order, axis=1)[:, :k]
    disc = 1.0 / np.log2(np.arange(2, g.shape[1] + 2))
    dcg = ((2.0 ** g - 1.0) * disc).sum(1)
    ideal_g = -np.sort(-gains.astype(float), axis=1)[:, :k]
    ideal = ((2.0 ** ideal_g - 1.0) * disc).sum(1)
    ok = ideal > 0
    return dcg[ok] / ideal[ok]


def expected_clicks(scores: np.ndarray, true_p: np.ndarray, theta: np.ndarray) -> float:
    """TRUE expected clicks per request if each row were shown in score order.

    Only computable in a simulation (it needs the true relevance): this is what
    an online A/B test of the ranking would measure, minus the noise.
    """
    pos = rank_positions(scores)
    return float((np.asarray(theta)[pos] * true_p).sum(1).mean())


def ips_policy_value(scores: np.ndarray, clicks: np.ndarray, logged_pos: np.ndarray,
                     theta: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Per-request IPS estimate of expected clicks if the logged items were re-ranked by ``scores``.

    Under the PBM, click / theta_logged estimates the item's relevance, and the
    new ranking would show it at theta_new: value = sum theta_new * click / theta_logged.
    ``clip`` caps the importance weight theta_new / theta_logged.
    """
    theta = np.asarray(theta, dtype=float)
    w = theta[rank_positions(scores)] / theta[logged_pos]
    if clip is not None:
        w = np.minimum(w, clip)
    return (w * clicks).sum(1)


# --------------------------------------------------------------------------- features for notebook 05
def click_features(history: pd.DataFrame, rows: pd.DataFrame, items: pd.DataFrame,
                   theta: np.ndarray | None = None, smoothing: float = 20.0) -> pd.DataFrame:
    """Four simple features for ``rows``, computed from ``history`` only (no leakage).

    * ``item_ctr``  smoothed item click rate (log)       - debiased if ``theta`` is given
    * ``ucat_ctr``  smoothed user x category click rate (log) - debiased if ``theta`` is given
    * ``log_price``
    * ``policy_score`` the production ranker's score for this impression (0 if missing)

    With ``theta``, each impression counts as theta_position "examinations"
    instead of one: clicks over expected examinations (COEC).
    """
    h = history[["user_id", "item_id", "position", "clicked"]].copy()
    h["category"] = items["category"].to_numpy()[h["item_id"].to_numpy()]
    h["n"] = 1.0 if theta is None else np.asarray(theta, dtype=float)[h["position"].to_numpy()]
    prior = float(h["clicked"].sum() / h["n"].sum())

    gi = h.groupby("item_id")[["clicked", "n"]].sum()
    item_ctr = (gi["clicked"] + smoothing * prior) / (gi["n"] + smoothing)
    gu = h.groupby(["user_id", "category"])[["clicked", "n"]].sum()
    ucat_ctr = (gu["clicked"] + 5 * prior) / (gu["n"] + 5)

    r_item = rows["item_id"].to_numpy()
    r_cat = items["category"].to_numpy()[r_item]
    idx = pd.MultiIndex.from_arrays([rows["user_id"].to_numpy(), r_cat])
    out = pd.DataFrame({
        "item_ctr": np.log(item_ctr.reindex(r_item).fillna(prior).to_numpy()),
        "ucat_ctr": np.log(ucat_ctr.reindex(idx).fillna(prior).to_numpy()),
        "log_price": np.log(items["price"].to_numpy()[r_item]),
        "policy_score": rows["policy_score"].fillna(0.0).to_numpy(),
    }, index=rows.index)
    return out
