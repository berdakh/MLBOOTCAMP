# Solutions · Chapter 58 · Full interview loop & portfolio

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run or hardware · [open-ended] personal / open-ended.

---

## Exercise 1 · `loop_score`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def loop_score(scores_by_round, weights):
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
    return {"overall": sum(weights[r] * m for r, m in by_round.items()) / total,
            "by_round": by_round, "by_signal": by_signal,
            "weakest_signal": min(sorted(by_signal), key=by_signal.get)}
```

**Details.** `isinstance(v, bool)` first, because `True == 1` would pass the
range check. Averaging per-round means (not all scores pooled) stops a round
with five signals outweighing one with three. `min(sorted(...), key=...)`
breaks ties alphabetically because `min` returns the first minimum it meets.
With the notebook's EXAMPLE attempt and illustrative weights this gives an
overall of **2.77** and *verification* (2.0 in all four coding rounds) as the
weakest signal [verified].

**What the number is not.** Real hiring decisions are not a weighted mean:
a single round with a strong negative signal can decide the outcome, and
different interviewers calibrate differently. Use `by_round` and
`weakest_signal` to choose practice; use `overall` only to see a trend.

---

## Exercise 2 · `project_brief`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
TBD = "TBD — fill from your run"
REQUIRED = ("title", "problem", "data", "approach", "results", "limitations", "next_steps")

def project_brief(project):
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
                raise ValueError(f"{r.get('metric')!r} has a value but no split")
            v = r.get("value")
            v = TBD if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)
            lines.append(f"| {r['metric']} | {v} | {split} |")
    else:
        lines.append(TBD)
    lines += ["", "## Limitations", ""] + [f"- {x}" for x in project["limitations"]]
    lines += ["", "## Next steps", ""] + [f"- {x}" for x in project["next_steps"]]
    if project.get("links"):
        lines += ["", "## Links", ""] + [f"- [{n}]({u})" for n, u in project["links"].items()]
    return "\n".join(lines) + "\n"
```

With Chapter 57's `depth_demos.json` in the same artifact directory, the
third brief quotes the val losses at LR 3e-2 (2.751 without warmup, 2.261
with) and the 20%-noise test accuracies (0.813 final, 0.9277 early-stopped),
each with its split [verified] (checked by running the notebook with this solution).
In a fresh runtime every value is TBD, which is the correct output.

The `ValueError` for a value without a split is the code version of the
course's honesty rule. Numbers harvested generically from Part I/II files
get the split `TBD: name the split (from <file>)`; replace it with the real
description before publishing.

---

## Exercise 3 · "Tell me about yourself"

[open-ended] Personal. Structure and a calibrated example (bracketed parts are yours
to fill with measured facts; do not invent them):

> "I'm an ML engineer with a background in [your field, e.g. time-series
> modelling and signal processing]. [In my thesis / last role] I built
> [model] for [task]; on a [named split, e.g. leave-subjects-out across N
> subjects] evaluation it reached [metric ±
> spread], against [baseline]. Since then I've focused on production ML: I
> built [a ranking cascade on simulated marketplace logs] with a
> pre-registered held-out evaluation that corrected for position bias, and
> [fine-tuned / served] [model] on a T4 with [latency or cost number]. What
> I bring is rigorous evaluation (I look for the leak before I trust the
> number) and the engineering to deploy it: tests, serving, monitoring. I'm
> looking for an MLE role where models face real users at scale, which is
> why [this team's problem] interests me."

Check: under 2:15 spoken; one line of identity; two proof points with your
role and a measured number with its split; strengths tied to the job;
a forward-looking reason, no complaints about past roles.

---

## Checkpoint answers

1. **Tell me about yourself.** [open-ended] See Exercise 3. Interviewers usually probe the
   most impressive number first ("how was that evaluated?") and the
   transition ("why leave research?"). Prepare a 30-second answer to both.
2. **Coding: mid vs senior when both solve it.** [expected] Generic signals: the
   senior candidate reaches the efficient approach without hints and says
   why it is optimal, writes clean code with edge cases handled unprompted,
   tests with cases chosen to break the code (empty, duplicates, extremes),
   states complexity precisely, and handles the follow-up (a changed
   constraint, concurrency, scale) by adapting the design rather than
   restarting.
3. **Staff vs senior in design (ads).** [expected] Two concrete additions: (a) a
   marketplace view with long-term advertiser ROI/retention metrics and
   *budget-split* experiments because user-split tests leak through shared
   advertiser budgets; (b) organisational contracts: which team owns
   calibration monitoring and the recalibration fix, how ranking, auction and
   pacing changes are sequenced and rolled back, and what to build first.
   Numbers that drive decisions (e.g. 3,969 ranker replicas → cascade and
   distillation first) also read as staff.
4. **Verification as the weakest signal.** [open-ended] Plan: for two weeks, in every
   coding and ML-coding practice, write two tests *before* running the code
   (one edge case, one property or reference comparison), say them aloud, and
   log them; do two ML-coding prompts that require a reference check
   (attention vs `scaled_dot_product_attention`, LoRA merge equivalence).
   Evidence it worked: `verification` ≥ 3 in the next two logged loops on
   fresh prompts, and no interviewer-prompted "how would you test this?".
5. **Strongest project in three minutes.** [open-ended] Problem and user (20 s); the
   one number with its split and interval (20 s); the architecture and
   where it is hard (60 s); the main trade-off with the alternative and
   evidence (40 s); limits and next step (40 s). Every number from your own
   run.
6. **Serving cost at 1,000 QPS.** [expected] Arithmetic with assumptions stated:
   measure per-replica throughput at the target p99 (e.g., say a T4 replica
   sustains R requests/s at batch B within the latency SLO, Chapter 50);
   replicas = ⌈1,000 / (R × (1 − headroom))⌉, at least 2, more for zones
   (`serving_fleet_size` from Chapter 56); monthly cost = replicas × price per
   replica-hour × 730; cost per 1k requests = monthly cost / (average QPS ×
   3,600 × 730) × 1,000 (`ds.cost_per_1k_requests`). Then name the levers:
   batching, quantisation/distillation (Chapter 49), caching, autoscaling to
   the average instead of the peak.
