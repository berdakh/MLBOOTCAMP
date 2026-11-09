"""Self-checks for Chapter 16 · Final evaluation & documentation."""

from __future__ import annotations

import re

import numpy as np

from churnlab.checks import run_check


def _synthetic(seed: int = 0, n_groups: int = 120, per_group: int = 20):
    """Grouped data where rows of one customer are strongly correlated."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"C{g:04d}" for g in range(n_groups)], per_group)
    risk = np.repeat(rng.normal(size=n_groups), per_group)
    y = (rng.random(len(groups)) < 1 / (1 + np.exp(-(risk * 1.5 - 2.5)))).astype(int)
    scores = 1 / (1 + np.exp(-(risk + rng.normal(scale=0.7, size=len(groups)) - 2.0)))
    return y, scores, groups


def reference_final_report(y, scores, threshold, groups, seed=0) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    from churnlab.evaluation import bootstrap_ci, confusion_at

    y = np.asarray(y)
    scores = np.asarray(scores)
    _, lo, hi = bootstrap_ci(y, scores, metric=average_precision_score, groups=groups, seed=seed)
    c = confusion_at(y, scores, threshold)
    flagged = c["tp"] + c["fp"]
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "pr_auc_ci": (lo, hi),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": c["tp"] / flagged if flagged else float("nan"),
        "recall": c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else float("nan"),
        "flagged": int(flagged),
    }


def ex1(final_report):
    def test():
        from sklearn.metrics import average_precision_score

        from churnlab.evaluation import bootstrap_ci

        y, scores, groups = _synthetic()
        threshold = 0.2
        got = final_report(y, scores, threshold, groups, 0)
        want = reference_final_report(y, scores, threshold, groups, 0)
        assert isinstance(got, dict), f"expected a dict, got {type(got).__name__}"
        for key in want:
            assert key in got, f"missing key {key!r}; expected keys {sorted(want)}"
        for key in ("pr_auc", "roc_auc", "precision", "recall"):
            assert abs(float(got[key]) - want[key]) < 1e-6, f"{key}: expected {want[key]:.4f}, got {got[key]}"
        assert int(got["flagged"]) == want["flagged"], f"flagged: expected {want['flagged']}, got {got['flagged']}"
        ci = got["pr_auc_ci"]
        assert isinstance(ci, tuple) and len(ci) == 2, f"pr_auc_ci must be a (low, high) tuple, got {ci!r}"
        lo, hi = (float(v) for v in ci)
        assert lo <= want["pr_auc"] <= hi, f"the CI ({lo:.3f}, {hi:.3f}) does not contain the point estimate"
        assert abs(lo - want["pr_auc_ci"][0]) < 0.03 and abs(hi - want["pr_auc_ci"][1]) < 0.03, (
            f"CI ({lo:.3f}, {hi:.3f}) is far from the reference "
            f"({want['pr_auc_ci'][0]:.3f}, {want['pr_auc_ci'][1]:.3f}); use bootstrap_ci with groups= and seed=")
        _, rlo, rhi = bootstrap_ci(y, scores, metric=average_precision_score, groups=None, seed=0)
        assert (hi - lo) > 1.3 * (rhi - rlo), (
            "your CI is about as narrow as a ROW bootstrap: resample whole customers (groups=...)")
        again = final_report(y, scores, threshold, groups, 0)
        assert tuple(again["pr_auc_ci"]) == tuple(ci), "same seed must give the same CI"

    return run_check("Exercise 1 · final_report", test,
                     hint="average_precision_score, bootstrap_ci(y, scores, groups=groups, seed=seed), "
                          "roc_auc_score, confusion_at(y, scores, threshold)")


def parse_markdown_table(text: str) -> dict[str, str]:
    """Return {first cell: second cell} for every body row of the first markdown table."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip().startswith("|")]
    assert len(lines) >= 3, "expected a markdown table: a header row, a |---| separator, then one row per metric"
    assert re.fullmatch(r"\|(\s*:?-{3,}:?\s*\|)+", lines[1].replace(" ", "")), (
        f"second table line must be a separator like |---|---|, got {lines[1]!r}")
    rows = {}
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) >= 2:
            rows[cells[0].strip("`* ")] = cells[1]
    return rows


def ex2(results_markdown):
    def test():
        report = {"pr_auc": 0.19667, "pr_auc_ci": (0.171753, 0.224094), "roc_auc": 0.7399396,
                  "precision": 0.2866, "recall": 0.12623, "flagged": 314, "positive_rate": 0.0587944}
        text = results_markdown(report)
        assert isinstance(text, str), "return the markdown as a string"
        rows = parse_markdown_table(text)
        for key, value in report.items():
            assert key in rows, f"no table row whose first cell is {key!r}; rows found: {list(rows)}"
            cell = rows[key]
            if isinstance(value, tuple):
                want = [f"{v:.3f}" for v in value]
                assert all(w in cell for w in want), f"{key}: expected both bounds {want} in {cell!r}"
                assert cell.index(want[0]) < cell.index(want[1]), f"{key}: low bound must come first"
            elif isinstance(value, int):
                assert re.search(rf"(?<![\d.]){value}(?![\d.])", cell.replace(",", "")), (
                    f"{key}: expected the integer {value} in {cell!r}")
            else:
                assert f"{value:.3f}" in cell, f"{key}: expected {value:.3f} (3 decimals) in {cell!r}"
        other = results_markdown({"pr_auc": 0.5, "pr_auc_ci": (0.4, 0.6)})
        rows = parse_markdown_table(other)
        assert set(rows) == {"pr_auc", "pr_auc_ci"}, "the table must follow the dict passed in, not fixed keys"

    return run_check("Exercise 2 · results_markdown", test,
                     hint="'| metric | value |\\n|---|---|\\n' + one '| key | formatted |' line per item; "
                          "f'{v:.3f}' for floats, f'[{lo:.3f}, {hi:.3f}]' for the CI tuple")
