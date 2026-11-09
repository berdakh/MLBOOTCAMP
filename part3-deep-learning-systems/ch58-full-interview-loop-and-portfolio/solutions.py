"""Reference solutions for Level 3 notebook 20. Graded by tests/test_solutions.py."""

from __future__ import annotations

TBD = "TBD — fill from your run"
REQUIRED = ("title", "problem", "data", "approach", "results", "limitations", "next_steps")


def ex1(scores_by_round, weights):
    if not scores_by_round:
        raise ValueError("no rounds scored")
    by_round, per_signal = {}, {}
    for rnd, scores in scores_by_round.items():
        if not scores:
            raise ValueError(f"round {rnd!r} has no scores")
        for sig, v in scores.items():
            if isinstance(v, bool) or v not in (1, 2, 3, 4):
                raise ValueError(f"{rnd}.{sig}: scores are whole numbers 1-4, got {v!r}")
            per_signal.setdefault(sig, []).append(v)
        if rnd not in weights:
            raise ValueError(f"no weight for round {rnd!r}")
        if weights[rnd] < 0:
            raise ValueError(f"negative weight for round {rnd!r}")
        by_round[rnd] = sum(scores.values()) / len(scores)
    total = sum(weights[r] for r in by_round)
    if total <= 0:
        raise ValueError("weights of the scored rounds sum to 0")
    by_signal = {s: sum(v) / len(v) for s, v in per_signal.items()}
    return {
        "overall": sum(weights[r] * m for r, m in by_round.items()) / total,
        "by_round": by_round,
        "by_signal": by_signal,
        "weakest_signal": min(sorted(by_signal), key=by_signal.get),
    }


def _fmt(v):
    if v is None:
        return TBD
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def ex2(project):
    missing = [k for k in REQUIRED if k not in project]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    lines = [f"# {project['title']}", ""]
    for head, key in (("Problem", "problem"), ("Data", "data"), ("Approach", "approach")):
        lines += [f"## {head}", "", str(project[key]), ""]
    lines += ["## Results", ""]
    if project["results"]:
        lines += ["| Metric | Value | Split |", "|---|---|---|"]
        for r in project["results"]:
            split = r.get("split") or ""
            if r.get("value") is not None and not split.strip():
                raise ValueError(f"{r.get('metric')!r} has a value but no split: every number names its split")
            lines.append(f"| {r['metric']} | {_fmt(r.get('value'))} | {split} |")
    else:
        lines.append(TBD)
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in project["limitations"]]
    lines += ["", "## Next steps", ""] + [f"- {x}" for x in project["next_steps"]]
    if project.get("links"):
        lines += ["", "## Links", ""] + [f"- [{name}]({url})" for name, url in project["links"].items()]
    return "\n".join(lines) + "\n"
