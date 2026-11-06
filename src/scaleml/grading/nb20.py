"""Self-checks for Level 3 notebook 20 · Full interview loop & portfolio."""

from __future__ import annotations

from churnlab.checks import close, run_check

TBD = "TBD — fill from your run"


def _raises_value_error(fn, *args) -> bool:
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def ex1(loop_score):
    def test():
        scores = {"coding": {"problem_solving": 3, "verification": 2, "communication": 3},
                  "design": {"trade_offs": 4, "operations": 2, "communication": 2},
                  "behavioural": {"ownership": 3, "communication": 4}}
        weights = {"coding": 1.0, "design": 2.0, "behavioural": 1.0, "unused_round": 5.0}
        before = {k: dict(v) for k, v in scores.items()}
        out = loop_score(scores, weights)
        assert scores == before, "do not modify the inputs"
        assert isinstance(out, dict), f"return a dict, got {type(out).__name__}"
        need = {"overall", "by_round", "by_signal", "weakest_signal"}
        assert need <= set(out), f"missing keys: {sorted(need - set(out))}"
        assert close(out["by_round"]["coding"], 8 / 3) and close(out["by_round"]["behavioural"], 3.5), (
            f"by_round is the mean of each round's signals, got {out['by_round']}")
        assert set(out["by_round"]) == set(scores), "by_round needs exactly the rounds that were scored"
        want = (1 * 8 / 3 + 2 * 8 / 3 + 1 * 3.5) / 4
        if close(out["overall"], (8 / 3 + 8 / 3 + 3.5) / 3):
            raise AssertionError("overall ignores the weights: use the weighted mean of per-round means")
        if close(out["overall"], (8 / 3 * 1 + 8 / 3 * 2 + 3.5 * 1) / 9):
            raise AssertionError("normalise weights over the rounds PRESENT (ignore 'unused_round')")
        assert close(out["overall"], want), f"overall: expected {want:.4f}, got {out['overall']}"
        assert close(out["by_signal"]["communication"], 3.0), "communication = mean(3, 2, 4) = 3.0 across rounds"
        assert close(out["by_signal"]["verification"], 2.0), "verification only appears in coding: 2.0"
        assert len(out["by_signal"]) == 6, f"six distinct signals were scored, got {sorted(out['by_signal'])}"
        assert out["weakest_signal"] == "operations", (
            f"verification and operations tie at 2.0; break ties alphabetically -> 'operations', "
            f"got {out['weakest_signal']!r}")
        single = loop_score({"r": {"a": 4}}, {"r": 0.3})
        assert close(single["overall"], 4.0) and single["weakest_signal"] == "a", "one round, one signal"
        bad_cases = [
            ({}, {"r": 1}, "no rounds"),
            ({"r": {}}, {"r": 1}, "a round with no scores"),
            ({"r": {"a": 5}}, {"r": 1}, "score 5"),
            ({"r": {"a": 2.5}}, {"r": 1}, "score 2.5"),
            ({"r": {"a": True}}, {"r": 1}, "score True (a bool is not a score)"),
            ({"r": {"a": 3}}, {"other": 1}, "missing weight"),
            ({"r": {"a": 3}, "s": {"a": 3}}, {"r": -1, "s": 2}, "negative weight"),
            ({"r": {"a": 3}}, {"r": 0}, "zero total weight"),
        ]
        for s, w, why in bad_cases:
            assert _raises_value_error(loop_score, s, w), f"{why} must raise ValueError"

    return run_check("Exercise 1 · loop_score", test,
                     hint="validate; by_round = {r: mean(s.values())}; overall = sum(w[r]*by_round[r]) / sum(w[r]) "
                          "over rounds present; collect each signal's scores across rounds for by_signal; "
                          "weakest = min(sorted(by_signal), key=by_signal.get)")


def _project():
    return {"title": "Ranking cascade", "problem": "Rank items for users.", "data": "Simulated logs.",
            "approach": "Two-tower retrieval + GBDT ranker.",
            "results": [{"metric": "NDCG@10", "value": 0.812345, "split": "test week, users bootstrapped"},
                        {"metric": "p99 latency (ms)", "value": None, "split": "load test"},
                        {"metric": "requests", "value": 1200, "split": "test week"}],
            "limitations": ["simulated world", "offline only"], "next_steps": ["interleaving test"]}


def ex2(project_brief):
    def test():
        p = _project()
        before = repr(p)
        md = project_brief(p)
        assert repr(p) == before, "do not modify the project dict"
        assert isinstance(md, str), f"return a Markdown string, got {type(md).__name__}"
        lines = md.splitlines()
        assert lines and lines[0].strip() == "# Ranking cascade", f"first line must be '# Ranking cascade', got {lines[:1]}"
        heads = ["## Problem", "## Data", "## Approach", "## Results", "## Limitations", "## Next steps"]
        pos = []
        for h in heads:
            assert h in lines, f"missing section heading {h!r} (on its own line)"
            pos.append(lines.index(h))
        assert pos == sorted(pos), f"sections must appear in the order {heads}"
        assert "Two-tower retrieval + GBDT ranker." in md, "include the approach text"
        assert any(ln.replace(" ", "") == "|Metric|Value|Split|" for ln in lines), \
            "the results table needs the header | Metric | Value | Split |"
        row = next((ln for ln in lines if ln.startswith("|") and "NDCG@10" in ln), "")
        assert "0.8123" in row and "test week, users bootstrapped" in row, \
            f"format floats with :.4g and include the split; got row {row!r}"
        row = next((ln for ln in lines if ln.startswith("|") and "p99 latency" in ln), "")
        assert TBD in row, f"a None value must read {TBD!r}; got row {row!r}"
        row = next((ln for ln in lines if ln.startswith("|") and "requests" in ln), "")
        assert "1200" in row, f"integers are written as they are; got row {row!r}"
        assert "- simulated world" in lines and "- offline only" in lines, "limitations as '- ' bullets"
        assert "- interleaving test" in lines, "next steps as '- ' bullets"
        assert "## Links" not in md, "no Links section when the project has no links"
        q = _project()
        q["results"] = []
        q["links"] = {"code": "https://example.org/repo"}
        md2 = project_brief(q)
        res = md2.split("## Results", 1)[1].split("## Limitations", 1)[0]
        assert TBD in res, "with no results, the Results section says 'TBD — fill from your run'"
        assert "- [code](https://example.org/repo)" in md2.splitlines(), "links as '- [name](url)' bullets"
        assert md2.index("## Links") > md2.index("## Next steps"), "Links comes last"
        r = _project()
        r["results"][0]["split"] = ""
        assert _raises_value_error(project_brief, r), "a number without a split must raise ValueError"
        r = _project()
        del r["results"][2]["split"]
        assert _raises_value_error(project_brief, r), "a result with a value and no 'split' key must raise ValueError"
        r = _project()
        del r["data"]
        assert _raises_value_error(project_brief, r), "a missing required key must raise ValueError"

    return run_check("Exercise 2 · project_brief", test,
                     hint="check required keys; build a list of lines: '# title', '', '## Problem', '', text, ...; "
                          "results rows f'| {metric} | {value} | {split} |' with None -> TBD and floats f'{v:.4g}'")
