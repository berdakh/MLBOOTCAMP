"""Simulated online experiments on ShopStream users (notebooks 11 and 12).

Real A/B tests take weeks and you never learn the true effect. Simulated
ones take milliseconds and you always do, so you can *measure* a test's
false-positive rate, its power, or how much CUPED shrinks its variance,
instead of taking the textbook's word for it.

The population comes from the ShopStream world: every user keeps their
activity rate (how many feed requests they make) and gets a baseline CTR
computed from their true click model (:func:`user_population`). An
experiment draws users from that population, simulates their requests and
clicks in a pre-period and an experiment period, and lets a treatment
multiply each user's CTR by ``1 + lift`` (:func:`simulate_users`).

Also here: the statistics notebooks 11-12 build by hand first
(:func:`welch_ttest`, :func:`two_proportion_ztest`, :func:`bootstrap_diff_ci`,
:func:`benjamini_hochberg`), ranking helpers for interleaving
(:func:`team_draft_interleave`) and a vectorised NDCG over fixed-size lists
(:func:`ndcg_rows`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import K, World

# ---------------------------------------------------------------------------
# Population and simulation
# ---------------------------------------------------------------------------


def user_population(world: World, n_candidates: int = 200, policy_noise: float = 3.0,
                    seed: int = 0) -> pd.DataFrame:
    """One row per ShopStream user: activity, segment, device, country, base_ctr.

    ``base_ctr`` = expected clicks per impression on a 10-item feed: a noisy
    ranker picks the top 10 of ``n_candidates`` random items by
    ``true_logit + N(0, policy_noise)``, and each position contributes
    ``examination_prob(position) * true_click_prob``. The default noise gives a
    mean CTR of about 0.14, close to the logging policy's CTR in the logs.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = world.config.n_users, world.config.n_items
    ctr = np.empty(n_users)
    exam = world.examination_prob(np.arange(K))
    for lo in range(0, n_users, 2000):
        u = np.arange(lo, min(lo + 2000, n_users))
        cand = rng.integers(0, n_items, (len(u), n_candidates))
        logit = world.true_logit(u[:, None], cand)
        noisy = logit + rng.normal(0, policy_noise, logit.shape)
        top = np.argsort(-noisy, axis=1)[:, :K]
        p = 1 / (1 + np.exp(-np.take_along_axis(logit, top, axis=1)))
        ctr[u] = (p * exam).mean(axis=1)
    out = world.users[["user_id", "segment", "country", "device", "activity"]].copy()
    out["base_ctr"] = ctr
    return out


def simulate_users(pop: pd.DataFrame, n_users: int, rng: np.random.Generator, lift: float = 0.0,
                   days: int = 14, pre_days: int = 14, requests_per_day: float = 0.35,
                   drift_sd: float = 0.3, treat_share: float = 0.5,
                   lift_by: dict | None = None) -> pd.DataFrame:
    """Simulate ``n_users`` users (drawn with replacement from ``pop``) in a
    user-randomised A/B test.

    Each user makes Poisson(activity * requests_per_day * days) requests of K
    impressions in the experiment period (and the same in a pre-period of
    ``pre_days``, with no treatment). Their CTR in each period is
    ``base_ctr * exp(N(0, drift_sd))`` (tastes drift a little between periods),
    multiplied by ``1 + lift`` in treatment. ``lift_by={"device": {"ios": 0.05}}``
    overrides the lift for a subgroup (heterogeneous effects).

    Returns one row per user: arm (0 = control, 1 = treatment), requests,
    impressions, clicks, pre_requests, pre_impressions, pre_clicks, and the
    population columns. Users with zero requests are kept (filter on
    ``impressions > 0`` for exposure-based analyses).
    """
    idx = rng.integers(0, len(pop), n_users)
    df = pop.iloc[idx].reset_index(drop=True)
    arm = (rng.random(n_users) < treat_share).astype(np.int8)
    rate = df["activity"].to_numpy() * requests_per_day
    base = df["base_ctr"].to_numpy()
    effect = np.full(n_users, float(lift))
    if lift_by:
        for col, mapping in lift_by.items():
            for value, val_lift in mapping.items():
                effect[df[col].to_numpy() == value] = val_lift
    p_pre = np.clip(base * np.exp(rng.normal(0, drift_sd, n_users)), 0, 1)
    p_exp = np.clip(base * np.exp(rng.normal(0, drift_sd, n_users)) * (1 + effect * arm), 0, 1)
    pre_req = rng.poisson(rate * pre_days)
    req = rng.poisson(rate * days)
    df["arm"] = arm
    df["pre_requests"], df["requests"] = pre_req, req
    df["pre_impressions"], df["impressions"] = pre_req * K, req * K
    df["pre_clicks"] = rng.binomial(pre_req * K, p_pre)
    df["clicks"] = rng.binomial(req * K, p_exp)
    return df


def simulate_request_randomized(pop: pd.DataFrame, n_users: int, rng: np.random.Generator,
                                lift: float = 0.0, days: int = 14, requests_per_day: float = 0.35,
                                drift_sd: float = 0.3) -> pd.DataFrame:
    """Same users and click model as :func:`simulate_users`, but every REQUEST
    is randomised: each user's requests are split between arms by a fair coin.

    Returns one row per user with impressions/clicks in each arm
    (``imps_0, clicks_0, imps_1, clicks_1``).
    """
    idx = rng.integers(0, len(pop), n_users)
    df = pop.iloc[idx].reset_index(drop=True)
    rate = df["activity"].to_numpy() * requests_per_day
    p = np.clip(df["base_ctr"].to_numpy() * np.exp(rng.normal(0, drift_sd, n_users)), 0, 1)
    req = rng.poisson(rate * days)
    req1 = rng.binomial(req, 0.5)
    req0 = req - req1
    df["imps_0"], df["imps_1"] = req0 * K, req1 * K
    df["clicks_0"] = rng.binomial(req0 * K, p)
    df["clicks_1"] = rng.binomial(req1 * K, np.clip(p * (1 + lift), 0, 1))
    return df


# ---------------------------------------------------------------------------
# Tests and intervals
# ---------------------------------------------------------------------------


def welch_ttest(a, b) -> dict:
    """Difference in means (b - a), its standard error and the two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    p = stats.ttest_ind(b, a, equal_var=False).pvalue
    return {"diff": float(diff), "se": float(se), "p": float(p)}


def two_proportion_ztest(x_a, n_a, x_b, n_b) -> dict:
    """Pooled two-proportion z-test. Correct only if the n trials are independent."""
    p_a, p_b = x_a / n_a, x_b / n_b
    pooled = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    return {"diff": p_b - p_a, "se": se, "p": 2 * stats.norm.sf(np.abs(z))}


def bootstrap_diff_ci(a, b, n_boot: int = 2000, alpha: float = 0.05, rng=None) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(b) - mean(a), resampling units within each arm."""
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = np.empty(n_boot)
    for lo in range(0, n_boot, 50):  # chunks keep memory small for large arms
        m = min(50, n_boot - lo)
        diffs[lo:lo + m] = (b[rng.integers(0, len(b), (m, len(b)))].mean(axis=1)
                            - a[rng.integers(0, len(a), (m, len(a)))].mean(axis=1))
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def benjamini_hochberg(pvalues, q: float = 0.05) -> np.ndarray:
    """Boolean array: which hypotheses BH rejects at false-discovery rate ``q``."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        k = np.max(np.flatnonzero(passed))     # largest i with p_(i) <= q i / m
        reject[order[:k + 1]] = True
    return reject


# ---------------------------------------------------------------------------
# Rankers: interleaving and offline metrics
# ---------------------------------------------------------------------------


def team_draft_interleave(list_a, list_b, rng: np.random.Generator):
    """Team-draft interleaving (Radlinski, Kurup & Joachims, 2008).

    In each round a coin decides which team picks first; each team then adds
    its highest-ranked item not yet shown. Returns ``(items, teams)`` of length
    ``len(list_a)``, where ``teams[i]`` is ``"A"`` or ``"B"``.
    """
    k = len(list_a)
    items, teams, shown = [], [], set()
    count = {"A": 0, "B": 0}
    lists = {"A": list(list_a), "B": list(list_b)}
    ptr = {"A": 0, "B": 0}
    while len(items) < k:
        if count["A"] < count["B"] or (count["A"] == count["B"] and rng.random() < 0.5):
            team = "A"
        else:
            team = "B"
        lst = lists[team]
        while ptr[team] < len(lst) and lst[ptr[team]] in shown:
            ptr[team] += 1
        if ptr[team] >= len(lst):          # this team has nothing new left: the other picks
            team = "B" if team == "A" else "A"
            lst = lists[team]
            while ptr[team] < len(lst) and lst[ptr[team]] in shown:
                ptr[team] += 1
            if ptr[team] >= len(lst):
                break
        item = lst[ptr[team]]
        items.append(item)
        teams.append(team)
        shown.add(item)
        count[team] += 1
    return items, teams


def ndcg_rows(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> np.ndarray:
    """NDCG@k per row for fixed-size lists: ``scores`` and binary/graded
    ``labels`` have shape (n_lists, list_len). Rows without a positive -> NaN.
    Ties broken by original column order (stable), like ``shoprank.metrics``."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    gains = 2.0 ** np.take_along_axis(labels, order, axis=1) - 1
    disc = 1 / np.log2(np.arange(2, gains.shape[1] + 2))
    dcg = (gains * disc).sum(axis=1)
    best = -np.sort(-labels, axis=1)[:, :k]            # labels in descending order
    ideal = ((2.0 ** best - 1) * disc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ideal > 0, dcg / ideal, np.nan)
