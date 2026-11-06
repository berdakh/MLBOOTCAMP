"""Monitoring a model after it ships.

Three layers, in the order you can compute them:

1. **Service health** - errors, latency, throughput. Available immediately.
2. **Data and prediction drift** - do today's inputs and scores look like the
   training reference? Available immediately, but drift is a *symptom*, not a
   verdict: inputs can shift without hurting accuracy, and accuracy can drop
   with no visible input shift.
3. **Performance** - did the predictions come true? For this model the label
   arrives 30 days later, so this layer always lags. Plan for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index between two numeric samples.

    Bins are quantiles of the *reference*, so each holds ~10% of it. Rule of
    thumb (a convention, not a law): < 0.1 stable, 0.1-0.25 moderate shift,
    > 0.25 large shift. Always look at the distributions, not just the number.
    """
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def categorical_psi(reference, current, eps: float = 1e-4) -> float:
    ref = pd.Series(reference).astype(str).value_counts(normalize=True)
    cur = pd.Series(current).astype(str).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    r = np.clip(ref.reindex(cats, fill_value=0).to_numpy(), eps, None)
    c = np.clip(cur.reindex(cats, fill_value=0).to_numpy(), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str],
                 categorical: list[str] = ()) -> pd.DataFrame:
    rows = []
    for col in numeric:
        rows.append({"feature": col, "kind": "numeric", "psi": psi(reference[col], current[col]),
                     "ref_mean": reference[col].mean(), "cur_mean": current[col].mean(),
                     "cur_missing": current[col].isna().mean()})
    for col in categorical:
        unseen = ~current[col].astype(str).isin(set(reference[col].astype(str)))
        rows.append({"feature": col, "kind": "categorical",
                     "psi": categorical_psi(reference[col], current[col]),
                     "unseen_rate": unseen.mean()})
    out = pd.DataFrame(rows)
    out["status"] = pd.cut(out["psi"], [-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "large"])
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class AlertRule:
    name: str
    metric: str
    op: str  # ">" or "<"
    threshold: float
    severity: str = "warn"  # "warn" -> ticket, "page" -> wake someone up

    def fires(self, value: float) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        return value > self.threshold if self.op == ">" else value < self.threshold


DEFAULT_RULES = [
    AlertRule("error rate high", "error_rate", ">", 0.01, "page"),
    AlertRule("p95 latency high", "latency_p95_ms", ">", 200.0, "page"),
    AlertRule("feature drift large", "max_psi", ">", 0.25, "warn"),
    AlertRule("score mean shifted", "score_mean_ratio", ">", 1.5, "warn"),
    AlertRule("score mean collapsed", "score_mean_ratio", "<", 0.5, "warn"),
    AlertRule("PR-AUC dropped", "pr_auc_ratio", "<", 0.8, "warn"),
]


def evaluate_rules(values: dict[str, float], rules=DEFAULT_RULES) -> pd.DataFrame:
    return pd.DataFrame([
        {"rule": r.name, "metric": r.metric, "value": values.get(r.metric), "threshold": f"{r.op} {r.threshold}",
         "fires": r.fires(values.get(r.metric)), "severity": r.severity}
        for r in rules
    ])


def latency_summary(latencies_ms) -> dict[str, float]:
    a = np.asarray(latencies_ms, dtype=float)
    return {"count": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}
