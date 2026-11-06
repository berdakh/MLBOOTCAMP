"""Reference solutions for notebook 14. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ex1(reference, current, bins: int = 10) -> float:
    ref = pd.Series(reference, dtype=float).dropna().to_numpy()
    cur = pd.Series(current, dtype=float).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ex2(candidate: dict, baseline: dict, policy: dict) -> tuple[bool, list[str]]:
    reasons = []
    cand_pr, base_pr = candidate.get("pr_auc"), baseline.get("pr_auc")
    if cand_pr is not None and base_pr is not None:
        floor = base_pr * (1 - policy["max_pr_auc_drop"])
        if cand_pr < floor:
            reasons.append(f"pr_auc {cand_pr:.3f} below {floor:.3f} (baseline {base_pr:.3f} "
                           f"minus {policy['max_pr_auc_drop']:.0%})")
    if candidate["error_rate"] > policy["max_error_rate"]:
        reasons.append(f"error_rate {candidate['error_rate']:.4f} above {policy['max_error_rate']}")
    if candidate["p95_ms"] > policy["max_p95_ms"]:
        reasons.append(f"p95_ms {candidate['p95_ms']:.1f} above {policy['max_p95_ms']}")
    return bool(reasons), reasons
