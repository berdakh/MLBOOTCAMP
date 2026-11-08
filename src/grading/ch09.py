"""Self-checks for Chapter 9 · Error analysis & artifacts."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from churnlab.checks import run_check

TENURE_LABELS = ["0-30d", "30-90d", "90-180d", "180-365d", "1-2y", "2y+"]

MODEL_CARD_SECTIONS = ["## Intended use", "## Training data", "## Evaluation",
                       "## Decision threshold", "## Limitations", "## Ethical considerations"]


def ex1(tenure_bucket):
    def test():
        days = pd.Series([0, 29, 30, 89.5, 90, 179, 180, 364, 365, 729, 730, 5000, np.nan],
                         index=list("abcdefghijklm"))
        want = ["0-30d", "0-30d", "30-90d", "30-90d", "90-180d", "90-180d", "180-365d", "180-365d",
                "1-2y", "1-2y", "2y+", "2y+", None]
        got = tenure_bucket(days)
        assert isinstance(got, pd.Series), f"return a pandas Series, got {type(got).__name__}"
        assert got.index.equals(days.index), "the result must keep the input's index"
        assert isinstance(got.dtype, pd.CategoricalDtype), "return a categorical Series (pd.cut does this)"
        assert list(got.cat.categories) == TENURE_LABELS, (
            f"categories must be {TENURE_LABELS} in that order, got {list(got.cat.categories)}")
        assert got.cat.ordered, "the categories should be ordered (young < old)"
        for x, w, g in zip(days, want, got):
            if w is None:
                assert pd.isna(g), "a missing tenure must stay missing (NaN), not get a bucket"
            else:
                assert g == w, f"tenure {x} -> expected {w!r}, got {g!r} (intervals are [left, right))"

    return run_check("Exercise 1 · tenure_bucket", test,
                     hint="pd.cut(s, bins=[0, 30, 90, 180, 365, 730, np.inf], labels=..., right=False)")


def _report():
    return pd.DataFrame({
        "plan": ["basic", "team", "business", "enterprise", "legacy", "trial"],
        "rows": [3000, 2500, 900, 40, 12, 400],
        "pr_auc": [0.15, 0.12, 0.18, 0.02, 0.01, np.nan],
        "roc_auc": [0.72, 0.70, 0.75, 0.55, 0.40, np.nan],
    })


def ex2(worst_slices):
    def test():
        report = _report()
        before = report.copy()
        got = worst_slices(report, "pr_auc", 50, 2)
        assert isinstance(got, pd.DataFrame), "return a DataFrame"
        pd.testing.assert_frame_equal(report, before, obj="the input report (do not modify it)")
        assert list(got["plan"]) == ["team", "basic"], (
            f"expected ['team', 'basic'] (lowest pr_auc among slices with >= 50 rows), got {list(got['plan'])}")
        got = worst_slices(report, "roc_auc", 10, 3)
        assert list(got["plan"]) == ["legacy", "enterprise", "team"], (
            f"min_rows=10 should keep the small slices: got {list(got['plan'])}")
        got = worst_slices(report, "pr_auc", 1000, 5)
        assert list(got["plan"]) == ["team", "basic"], "slices without a metric (NaN) must be skipped"

    return run_check("Exercise 2 · worst_slices", test,
                     hint="report[report.rows >= min_rows].dropna(subset=[metric]).nsmallest(n, metric)")


def _meta():
    return {
        "model_name": "logreg",
        "version": "v7",
        "threshold": 0.1898,
        "features": ["plan", "region", "seats", "tenure_days"],
        "training_data": {"rows": 25278, "first_snapshot": "2023-03-01", "last_snapshot": "2024-03-01",
                          "fingerprint": "abc123"},
        "metrics": {"validation": {"pr_auc": 0.16331, "roc_auc": 0.73317, "brier": 0.04672}},
    }


def ex3(write_model_card):
    def test():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "card" / "model_card.md"
            path.parent.mkdir()
            out = write_model_card(_meta(), path)
            assert out is not None and Path(out) == path, "return the path you wrote"
            assert path.exists(), "the file was not written"
            text = path.read_text(encoding="utf-8")
            assert text.lstrip().startswith("# Model card"), "the card must start with a '# Model card' title"
            for section in MODEL_CARD_SECTIONS:
                assert any(line.strip() == section for line in text.splitlines()), f"missing heading {section!r}"
            meta = _meta()
            for needle in ("logreg", "v7", "25278" if "25278" in text else "25,278"):
                assert needle in text, f"the card should mention {needle!r}"
            for name, value in meta["metrics"]["validation"].items():
                assert f"{value:.3f}" in text, f"validation {name} ({value:.3f}) does not appear in the card"
            assert "0.190" in text, "the decision threshold (0.190, 3 decimals) does not appear in the card"

    return run_check("Exercise 3 · write_model_card", test,
                     hint="build a list of lines: title, then each '## ...' section; f'{value:.3f}' for numbers")
