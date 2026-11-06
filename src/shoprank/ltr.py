"""Learning to rank on ShopStream: point-in-time features, groups, pairwise metrics.

Introduced in notebook 06. Everything here is small and readable on purpose;
production feature pipelines do the same arithmetic with more plumbing.

Point-in-time rule
------------------
A row logged on day ``d`` may only use statistics from days ``< d``. The
features are built from per-day aggregates with a cumulative sum *minus the
current day*, which is exactly "everything known at the start of day ``d``"
(a daily batch feature job). Same-day clicks would leak the label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K = 10

#: Features the ranker sees. ``position`` is deliberately absent (see notebook 06).
FEATURES = [
    "u_cat_ctr", "u_cat_imps", "u_cat_clicks", "u_ctr", "u_imps",
    "i_ctr", "i_coec", "i_imps", "i_age", "log_price", "price_vs_cat",
    "category", "segment", "device",
]
CATEGORICAL = ["category", "segment", "device"]


def _prior_day_sums(df: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    """Per (keys, day): sums of ``values`` over all EARLIER days (strictly before)."""
    daily = df.groupby(keys + ["day"], sort=True, observed=True)[values].sum()
    cum = daily.groupby(level=list(range(len(keys))), sort=False).cumsum() - daily
    return cum.add_prefix("prior_").reset_index()


def build_features(imps: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame,
                   propensities: np.ndarray, smoothing: float = 20.0,
                   prior_ctr: float = 0.13) -> pd.DataFrame:
    """Point-in-time ranking features for every impression row of ``imps``.

    ``imps`` must hold the full history you want statistics from (e.g. all
    days up to the last day you will score). ``propensities[p]`` is the
    examination probability of position ``p``; it is used for the debiased
    "clicks over expected clicks" (COEC) features. Returns a frame aligned
    with ``imps`` (same index) holding :data:`FEATURES`.

    ``prior_ctr`` (the smoothing target) is a fixed constant on purpose: a
    prior computed from ``imps`` itself would include future days, and every
    feature would then change when later data arrives (a subtle leak).
    """
    it = items.set_index("item_id")
    df = imps[["day", "user_id", "item_id", "position", "clicked"]].copy()
    df["category"] = it["category"].to_numpy()[df["item_id"].to_numpy()]
    df["clicks"] = df["clicked"].astype(np.float32)
    df["imps"] = np.float32(1.0)
    df["exp_clicks"] = np.asarray(propensities, dtype=np.float32)[df["position"].to_numpy()]

    item_s = _prior_day_sums(df, ["item_id"], ["clicks", "imps", "exp_clicks"])
    user_s = _prior_day_sums(df, ["user_id"], ["clicks", "imps"])
    ucat_s = _prior_day_sums(df, ["user_id", "category"], ["clicks", "imps"])
    df = (df.merge(item_s, on=["item_id", "day"], how="left")
            .merge(user_s, on=["user_id", "day"], how="left", suffixes=("", "_u"))
            .merge(ucat_s, on=["user_id", "category", "day"], how="left", suffixes=("", "_uc")))
    df.index = imps.index

    global_ctr = prior_ctr
    a = smoothing
    out = pd.DataFrame(index=imps.index)
    out["u_cat_ctr"] = (df["prior_clicks_uc"] + 5 * global_ctr) / (df["prior_imps_uc"] + 5)
    out["u_cat_imps"] = np.log1p(df["prior_imps_uc"])
    out["u_cat_clicks"] = np.log1p(df["prior_clicks_uc"])
    out["u_ctr"] = (df["prior_clicks_u"] + 5 * global_ctr) / (df["prior_imps_u"] + 5)
    out["u_imps"] = np.log1p(df["prior_imps_u"])
    out["i_ctr"] = (df["prior_clicks"] + a * global_ctr) / (df["prior_imps"] + a)
    # COEC: clicks / clicks expected from the positions the item was shown at.
    exp_rate = global_ctr / float(np.mean(propensities))  # CTR per unit of examination, positions uniform
    out["i_coec"] = (df["prior_clicks"] + a * global_ctr) / (df["prior_exp_clicks"] * exp_rate + a * global_ctr)
    out["i_imps"] = np.log1p(df["prior_imps"])
    out["i_age"] = df["day"].to_numpy() - it["created_day"].to_numpy()[df["item_id"].to_numpy()]
    price = it["price"].to_numpy()[df["item_id"].to_numpy()]
    out["log_price"] = np.log(price)
    cat_med = items.groupby("category")["price"].median()
    out["price_vs_cat"] = np.log(price / cat_med.reindex(df["category"]).to_numpy())
    out["category"] = df["category"].to_numpy()
    uu = users.set_index("user_id")
    out["segment"] = uu["segment"].to_numpy()[df["user_id"].to_numpy()]
    out["device"] = pd.Categorical(uu["device"].to_numpy()[df["user_id"].to_numpy()],
                                   categories=["android", "ios", "web"]).codes
    for c in CATEGORICAL:
        out[c] = out[c].astype("int32")
    return out[FEATURES].astype({c: "float32" for c in FEATURES if c not in CATEGORICAL})


def make_groups(df: pd.DataFrame, group_col: str):
    """Stable-sort ``df`` by ``group_col``; return (sorted frame, group sizes in that order)."""
    out = df.sort_values(group_col, kind="stable")
    sizes = out.groupby(group_col, sort=False).size().to_numpy()
    return out, sizes


def pairwise_accuracy(scores, labels, groups) -> float:
    """Share of within-group pairs with different labels that the scores order correctly.

    A tie in score counts as half a correct pair. Pairs across groups never count.
    Returns NaN when there is no such pair. O(sum of group_size**2).
    """
    s, y, g = (np.asarray(x) for x in (scores, labels, groups))
    order = np.argsort(g, kind="stable")
    s, y, g = s[order], y[order], g[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    good, total = 0.0, 0
    for lo, hi in zip(starts, ends):
        yg, sg = y[lo:hi], s[lo:hi]
        pairs = (yg[:, None] - yg[None, :]) > 0
        n = int(pairs.sum())
        if n == 0:
            continue
        diff = (sg[:, None] - sg[None, :])[pairs]
        total += n
        good += float((diff > 0).sum() + 0.5 * (diff == 0).sum())
    return good / total if total else float("nan")
