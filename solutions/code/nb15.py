"""Reference solutions for notebook 15. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from churnlab.evaluation import bootstrap_ci, confusion_at


def ex1(y, scores, threshold: float, groups, seed: int = 0) -> dict:
    y = np.asarray(y)
    scores = np.asarray(scores)
    pr_auc, lo, hi = bootstrap_ci(y, scores, metric=average_precision_score, groups=groups, seed=seed)
    c = confusion_at(y, scores, threshold)
    flagged = c["tp"] + c["fp"]
    positives = c["tp"] + c["fn"]
    return {
        "pr_auc": pr_auc,
        "pr_auc_ci": (lo, hi),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": c["tp"] / flagged if flagged else float("nan"),
        "recall": c["tp"] / positives if positives else float("nan"),
        "flagged": int(flagged),
    }


def _fmt(value) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return f"[{value[0]:.3f}, {value[1]:.3f}]"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def ex2(report: dict) -> str:
    lines = ["| metric | value |", "|---|---|"]
    lines += [f"| {key} | {_fmt(value)} |" for key, value in report.items()]
    return "\n".join(lines) + "\n"
