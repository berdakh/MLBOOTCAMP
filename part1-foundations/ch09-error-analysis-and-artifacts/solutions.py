"""Reference solutions for notebook 08. Graded by tests/test_solutions.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TENURE_EDGES = [0, 30, 90, 180, 365, 730, np.inf]
TENURE_LABELS = ["0-30d", "30-90d", "90-180d", "180-365d", "1-2y", "2y+"]


def ex1(tenure_days: pd.Series) -> pd.Series:
    return pd.cut(tenure_days, bins=TENURE_EDGES, labels=TENURE_LABELS, right=False, ordered=True)


def ex2(report: pd.DataFrame, metric: str, min_rows: int, n: int) -> pd.DataFrame:
    eligible = report[report["rows"] >= min_rows].dropna(subset=[metric])
    return eligible.sort_values(metric, kind="stable").head(n)


def ex3(meta: dict, path) -> Path:
    path = Path(path)
    m = meta.get("metrics", {}).get("validation", {})
    data = meta.get("training_data", {})
    lines = [
        f"# Model card · {meta['model_name']} {meta['version']}",
        "",
        "## Intended use",
        "",
        "Rank active TaskFlow accounts by their risk of cancelling in the next 30 days, so the",
        "customer-success team can decide whom to contact. Not for pricing, credit or any decision",
        "made about a customer without a human in the loop.",
        "",
        "## Training data",
        "",
        f"- rows: {data.get('rows')}",
        f"- snapshots: {data.get('first_snapshot')} to {data.get('last_snapshot')}",
        f"- fingerprint: `{data.get('fingerprint')}`",
        f"- features: {', '.join(meta.get('features', []))}",
        "",
        "## Evaluation",
        "",
        "Validation period only (the test period is reserved).",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in m.items() if isinstance(v, (int, float))],
        "",
        "## Decision threshold",
        "",
        f"Flag when the score is at least {meta['threshold']:.3f}.",
        "",
        "## Limitations",
        "",
        "- Synthetic data; validated on three months only.",
        "",
        "## Ethical considerations",
        "",
        "- A flag triggers a call to a customer; it must never reduce service or change prices.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
