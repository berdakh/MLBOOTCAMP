"""Diagrams for the handbook and guides (docs/) - prefix ``hb-``.

See tools/make_diagrams.py for the conventions. One idea per figure, a
left-to-right or top-to-bottom flow, and one takeaway box at the bottom.
"""

from __future__ import annotations

from svgkit import Canvas, diagram

# ---------------------------------------------------------------------------
# Chapter 01 - the ML engineering job
# ---------------------------------------------------------------------------


@diagram
def job_map():
    c = Canvas(880, 400)
    c.title("Three titles, one shared core",
            "Same skills, different centre of gravity. Read the job description, not the title.")
    roles = [
        (40, "ML Engineer", "Does it work reliably in production, and keep working? "
         "Tested on coding, SQL, system design, operations.", "accent"),
        (315, "Applied Scientist", "Which method solves this business problem, and how do we "
         "know? Tested on ML depth, statistics, experiments.", "default"),
        (590, "Algorithm Engineer", "Is this algorithm correct, fast and robust on the target "
         "system? Tested on algorithms, maths, numerics.", "default"),
    ]
    for x, name, sub, role in roles:
        c.box(x, 80, 250, 110, name, sub, role=role)
        c.arrow(x + 125, 192, x + 125, 236, accent=role == "accent")
    c.box(40, 240, 800, 64, "Shared core: what this course concentrates on",
          "code other people can run  ·  evaluation that does not fool you  ·  "
          "decisions explained to a non-specialist", role="good")
    c.box(40, 326, 800, 50, "Where researchers usually struggle",
          "MLE: testing, APIs, deployment  ·  AS: business decisions, SQL  ·  "
          "AE: systems-level performance", role="warn")
    return "hb-job-map.svg", c.render(
        "Three role boxes, ML Engineer, Applied Scientist and Algorithm Engineer, each with "
        "its core question and interview focus, all pointing down to a shared core of "
        "runnable code, honest evaluation and clear decisions."
    ), "ML Engineer vs Applied Scientist vs Algorithm Engineer: different centres, one shared core"


@diagram
def research_to_production():
    c = Canvas(880, 400)
    c.title("What a researcher already brings",
            "The discipline transfers; only the names change.")
    c.text(200, 78, "Research habit", "lbl")
    c.text(680, 78, "Engineering name for it", "lbl")
    rows = [
        ("Leave-one-subject-out", "evaluate on people never seen",
         "Group split", "prevents entity leakage"),
        ("Untouched final session", "opened only at the very end",
         "Locked test set", "one-time final evaluation"),
        ("Mean ± CI across seeds", "a spread, not one number",
         "Bootstrap intervals", "do not ship noise"),
        ("Filters fitted on train only", "no test statistics leak in",
         "Pipeline discipline", "no preprocessing leakage"),
    ]
    for i, (a, a_sub, b, b_sub) in enumerate(rows):
        y = 92 + i * 56
        c.box(40, y, 320, 46, a, a_sub, role="default")
        c.arrow(364, y + 23, 516, y + 23, accent=True)
        c.box(520, y, 320, 46, b, b_sub, role="good")
    c.box(40, 330, 800, 50, "Say both names in interviews",
          "the research version is your evidence; the engineering name shows you know where it applies",
          role="accent")
    return "hb-research-to-production.svg", c.render(
        "Four research habits on the left mapped by arrows to their engineering names on the "
        "right: leave-one-subject-out to group split, untouched final session to locked test "
        "set, confidence intervals across seeds to bootstrap intervals, filters fitted on "
        "training data to pipeline discipline."
    ), "Research habits and their engineering names"


# ---------------------------------------------------------------------------
# Chapter 02 - reproducibility
# ---------------------------------------------------------------------------


@diagram
def repro_layers():
    c = Canvas(880, 380)
    c.title("Four levels of reproducibility",
            "Each level is stricter than the last. Decisions need the last one.")
    levels = [
        ("Re-runnable", "runs top to bottom with no manual steps", "exploration", "default"),
        ("Repeatable", "same machine and environment, same number", "debugging", "default"),
        ("Reproducible", "another machine, pinned environment, same number",
         "reports, releases", "accent"),
        ("Robust", "other seeds and perturbations, same conclusion", "decisions", "good"),
    ]
    for i, (name, sub, enough, role) in enumerate(levels):
        x = 40 + i * 210
        c.box(x, 84, 170, 96, name, sub, role=role)
        c.text(x + 85, 202, f"enough for: {enough}", "small")
        if i < 3:
            c.arrow(x + 174, 132, x + 206, 132, accent=True)
    c.box(40, 226, 800, 62, "Every level answers four questions: which code, data, config, environment?",
          "commit hash  ·  SHA-256 checksum or fingerprint  ·  saved configuration  ·  "
          "lock file - all written into one run record", role="default")
    c.box(40, 308, 800, 48, "A model comparison that flips when you change the seed was never a comparison",
          role="warn")
    return "hb-repro-layers.svg", c.render(
        "Four boxes left to right, re-runnable, repeatable, reproducible and robust, each with "
        "what it is enough for, above a box listing the four questions every result must "
        "answer: code, data, configuration and environment."
    ), "Re-runnable, repeatable, reproducible, robust: what each level buys"


# ---------------------------------------------------------------------------
# Chapter 03 - data, SQL and leakage
# ---------------------------------------------------------------------------


@diagram
def leakage_map():
    c = Canvas(880, 400)
    c.title("Three ways the answer leaks into training",
            "Each one inflates the validation score; each has a standard fix.")
    kinds = [
        ("Label leakage", "a feature encodes the outcome: status = cancelled, a "
         "cancellation reason, a retention-offer flag",
         "Would this value exist at prediction time?", "if not, drop it or rebuild it point-in-time"),
        ("Temporal leakage", "a random split lets the model learn from the future of the "
         "period it is scored on",
         "Time split + embargo", "train on the past; drop rows whose label window crosses a boundary"),
        ("Group leakage", "the same customer on both sides, so the score partly measures "
         "memorisation",
         "Group split", "every row of one customer, device or site on one side"),
    ]
    for i, (name, how, fix, fix_sub) in enumerate(kinds):
        x = 40 + i * 275
        c.box(x, 82, 250, 98, name, how, role="bad")
        c.arrow(x + 125, 182, x + 125, 216, accent=True)
        c.text(x + 135, 203, "fix", "small t-good", anchor="start")
        c.box(x, 220, 250, 82, fix, fix_sub, role="good")
    c.box(40, 326, 800, 50, "Symptom of all three: a score that is too good",
          "measure it - the leaky score minus the honest score is the size of the leak",
          role="warn")
    return "hb-leakage-map.svg", c.render(
        "Three columns: label leakage, temporal leakage and group leakage, each with how it "
        "gets in and an arrow down to its fix: a point-in-time check, a time split with an "
        "embargo, and a group split."
    ), "Leakage map: label, temporal and group leakage, and the fix for each"


# ---------------------------------------------------------------------------
# Chapter 04 - evaluation and decisions
# ---------------------------------------------------------------------------


@diagram
def which_metric():
    c = Canvas(880, 440)
    c.title("Which metric?", "Start from the decision the score drives, then pick the metric.")
    c.box(290, 76, 300, 58, "What does the score drive?",
          "write down action, capacity and costs", role="accent")
    c.arrow(440, 136, 440, 176)
    c.path("M 440 156 L 165 156 L 165 176")
    c.path("M 440 156 L 715 156 L 715 176")
    branches = [
        ("A fixed number of actions", "200 calls a week: take the top k",
         "Precision@k, lift@k", "measured at the team's capacity"),
        ("An action per case, by cost", "act when expected value is positive",
         "Calibrated p + threshold", "p > C / (V + C); Brier, log loss"),
        ("Ranking, no action yet", "compare models and pick one",
         "PR-AUC", "against the positive rate, not 0.5"),
    ]
    for i, (q, q_sub, metric, m_sub) in enumerate(branches):
        x = 40 + i * 275
        c.box(x, 180, 250, 60, q, q_sub, role="default")
        c.arrow(x + 125, 242, x + 125, 276, accent=True)
        c.box(x, 280, 250, 64, metric, m_sub, role="good")
    c.box(40, 368, 800, 50, "Whatever you pick, report it with",
          "a baseline  ·  an interval from the right resampling unit  ·  the split and period",
          role="warn")
    return "hb-which-metric.svg", c.render(
        "A decision tree: what does the score drive? A fixed number of actions leads to "
        "precision at k; a cost-based action per case leads to calibrated probabilities and a "
        "threshold; ranking without an action yet leads to PR-AUC against the positive rate."
    ), "Which metric? A decision tree from the action to the metric"


@diagram
def threshold_value():
    c = Canvas(880, 400)
    c.title("Thresholds from costs",
            "Expected value of 'call everyone above t' (illustrative shape).")
    # Plot area: t in [0, 0.6] -> x in [80, 520]; value 0 at y = 200.
    c.rule(80, 80, 80, 300)
    c.rule(80, 200, 520, 200)
    c.text(88, 78, "expected value", "small", anchor="start")
    c.text(520, 190, "0", "small", anchor="end")
    c.curve("M 80 292 C 110 240, 160 118, 202 118 C 244 118, 330 178, 520 197",
            tone="accent", width=2.6)
    c.path("M 202 126 L 202 300", dashed=True, arrow=False)
    c.circle(202, 118, 5, role="good")
    c.text(212, 108, "best: t = p* = 1/6", "small t-good", anchor="start")
    c.text(112, 270, "t too low:", "small t-bad", anchor="start")
    c.text(112, 284, "wasted calls", "small t-bad", anchor="start")
    c.text(516, 222, "t too high: missed churners", "small t-warn", anchor="end")
    for x, label in ((80, "0"), (202, "1/6"), (520, "0.6")):
        c.text(x, 318, label, "small")
    c.text(380, 318, "threshold t", "small")
    c.box(580, 80, 260, 80, "Break-even: p > 1/6",
          "a call pays when 150p > 30(1 − p)", role="accent")
    c.box(580, 176, 260, 62, "Needs calibrated p",
          "if 0.3 really means 0.1, it is wrong", role="warn")
    c.box(580, 254, 260, 62, "Capacity can override",
          "200 calls a week: take the top 200", role="default")
    c.box(40, 336, 800, 44, "The threshold follows from stated costs. Change the costs and it moves.",
          role="good")
    return "hb-threshold-value.svg", c.render(
        "A curve of expected value against the threshold: negative when almost everyone is "
        "called, peaking at the break-even probability one sixth, and falling towards zero "
        "when almost nobody is called. Boxes state the break-even rule, the need for "
        "calibration and the capacity override."
    ), "Expected value against threshold: the peak sits at the break-even probability"


# ---------------------------------------------------------------------------
# Chapter 05 - serving
# ---------------------------------------------------------------------------


@diagram
def retry_timeline():
    c = Canvas(880, 360)
    c.title("A well-behaved client",
            "Time out, retry what might not repeat, back off with jitter. Never retry a validation error.")
    attempts = [
        (40, "Attempt 1", "timeout: no answer", "bad"),
        (340, "Attempt 2", "503 service unavailable", "bad"),
        (640, "Attempt 3", "200 OK, same request id", "good"),
    ]
    for x, name, sub, role in attempts:
        c.box(x, 84, 200, 62, name, sub, role=role)
    c.arrow(244, 115, 336, 115, accent=True, label="wait 0.2 s", label_dy=-10)
    c.arrow(544, 115, 636, 115, accent=True, label="wait 0.4 s", label_dy=-10)
    c.box(40, 176, 390, 70, "422 invalid input: do not retry",
          "fails the same way every time: return the error", role="warn")
    c.box(450, 176, 390, 70, "Is the call idempotent?",
          "scoring: safe to retry  ·  'create a task': needs a key",
          role="accent")
    c.box(40, 272, 800, 62, "Worst case = (retries + 1) timeouts + all the waits",
          "always set a timeout: many HTTP libraries wait forever by default", role="default")
    return "hb-retry-timeline.svg", c.render(
        "Three attempts left to right: a timeout, a 503, then success, separated by waits "
        "that double with jitter. Below, a 422 validation error is never retried, and only "
        "idempotent calls are safe to retry."
    ), "Retries with timeout, exponential backoff and idempotence"


# ---------------------------------------------------------------------------
# Chapter 06 - operations and monitoring
# ---------------------------------------------------------------------------


@diagram
def release_ladder():
    c = Canvas(880, 384)
    c.title("Safer releases: evidence before exposure",
            "Each step shows the new model to more users and tells you more about it.")
    steps = [
        ("Offline eval", "locked test set", "default"),
        ("Shadow", "live traffic, answers unused", "accent"),
        ("Canary", "1-5% of traffic, then more", "accent"),
        ("A/B test", "long enough for outcomes", "accent"),
        ("Full rollout", "new version is CURRENT", "good"),
    ]
    for i, (name, sub, role) in enumerate(steps):
        x = 40 + i * 165
        c.box(x, 84, 140, 90, name, sub, role=role)
        if i < 4:
            c.arrow(x + 142, 129, x + 163, 129, accent=True)
    for x in (370, 535, 700):
        c.arrow(x + 70, 176, x + 70, 236, dashed=True)
    c.text(436, 210, "metrics drop?", "small", anchor="end")
    c.box(330, 240, 470, 56, "Rollback: point CURRENT at the previous version",
          "immutable bundles, one atomic write - seconds, not hours", role="bad")
    c.box(40, 316, 800, 48, "Roll back first, then investigate",
          "undoing a deploy should be faster than understanding what went wrong", role="warn")
    return "hb-release-ladder.svg", c.render(
        "Five steps left to right: offline evaluation, shadow, canary, A/B test and full "
        "rollout. Dashed arrows from canary, A/B test and full rollout lead down to a "
        "rollback box that points CURRENT at the previous version."
    ), "Release ladder: offline, shadow, canary, A/B, full - with rollback from every live step"


# ---------------------------------------------------------------------------
# Chapter 07 - ML system design
# ---------------------------------------------------------------------------


@diagram
def design_framework():
    c = Canvas(880, 360)
    c.title("The system-design interview in 40 minutes",
            "Eight steps; box width is the time to spend on each.")
    steps = [("Clarify", 4), ("Success", 3), ("Labels & data", 6), ("Baseline & model", 5),
             ("Offline eval", 5), ("Serving", 6), ("Monitoring", 6), ("Risks", 3)]
    per_min = 800 / sum(m for _, m in steps)
    x = 40.0
    edges = []
    for i, (name, minutes) in enumerate(steps):
        w = minutes * per_min
        role = "accent" if i < 3 else ("default" if i < 6 else "good")
        c.text(x + w / 2, 80, str(i + 1), "small")
        c.box(x, 88, w - 3, 70, name, f"{minutes} min", role=role)
        edges.append(x)
        x += w
    edges.append(x)
    groups = [
        (0, 3, "Frame the problem", "decision, success, labels", "accent"),
        (3, 6, "Build and evaluate", "baseline, split, serving", "default"),
        (6, 8, "Operate", "monitoring, risks", "good"),
    ]
    for a, b, name, sub, role in groups:
        mins = sum(m for _, m in steps[a:b])
        c.box(edges[a], 182, edges[b] - edges[a] - 3, 62, name, f"{sub}  ·  {mins} min", role=role)
    c.box(40, 270, 797, 64, "Say each step's name as you start it",
          "framing and communication count more than the model you name; "
          "leave two minutes for questions",
          role="warn")
    return "hb-design-framework.svg", c.render(
        "A horizontal bar of eight steps whose widths are proportional to minutes: clarify, "
        "success, labels and data, baseline and model, offline evaluation, serving, "
        "monitoring and risks, grouped into framing, building and operating."
    ), "The eight-step ML system-design framework, sized by minutes"
