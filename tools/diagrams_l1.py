"""Diagrams for Level 1 - see tools/make_diagrams.py for the conventions.

The six original figures (roadmap, architecture, point-in-time, splits,
serving, monitoring) keep their historical file names because README.md, the
docs and the notebooks embed them. Everything new is prefixed ``l1-``.
"""

from __future__ import annotations

import math

from svgkit import Canvas, diagram

W = 880  # standard canvas width


def _row_arrows(c: Canvas, xs: list[float], w: float, y: float, accent: bool = True) -> None:
    """Arrows between consecutive boxes of width ``w`` starting at ``xs``."""
    for a, b in zip(xs, xs[1:]):
        c.arrow(a + w + 2, y, b - 4, y, accent=accent)


def _takeaway(c: Canvas, y: float, text: str, role: str = "ghost", h: float = 40) -> None:
    c.box(30, y, W - 60, h, text, role=role)


def _axes(c: Canvas, x0: float, y0: float, w: float, h: float, xlabel: str, ylabel: str) -> None:
    """L-shaped axes with origin at (x0, y0 + h)."""
    c.rule(x0, y0 + h, x0 + w, y0 + h)
    c.rule(x0, y0, x0, y0 + h)
    c.text(x0 + w / 2, y0 + h + 22, xlabel, "small")
    c.raw(f'<text x="{x0 - 26:.1f}" y="{y0 + h / 2:.1f}" font-family="sans-serif" font-size="11.5" '
          f'fill="#656d76" text-anchor="middle" transform="rotate(-90 {x0 - 26:.1f} {y0 + h / 2:.1f})">'
          f'{ylabel}</text>')


def _poly(points: list[tuple[float, float]]) -> str:
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


# ---------------------------------------------------------------------------
# The six original figures, redrawn (same file names)
# ---------------------------------------------------------------------------


@diagram
def roadmap():
    c = Canvas(W, 300)
    c.title("Five days, one project", "TaskFlow churn prediction, from a raw extract to a defended result.")
    days = [
        ("Day 1", "Reproducible projects", "env · contracts · SQL", "splits · baseline", "01-04"),
        ("Day 2", "Evaluate and improve", "pipelines · CV", "thresholds · artifacts", "05-08"),
        ("Day 3", "Serve predictions", "FastAPI · contracts", "clients · tests", "09-11"),
        ("Day 4", "Deploy and operate", "config · logs · latency", "monitoring · rollback", "12-14"),
        ("Day 5", "Demonstrate", "held-out test · docs", "drills · mock interview", "15-17"),
    ]
    xs = [30 + i * 166 for i in range(5)]
    for i, (x, (day, sub, t1, t2, nbs)) in enumerate(zip(xs, days)):
        c.box(x, 80, 140, 80, day, sub, role="good" if i == 4 else "accent")
        c.text(x + 70, 72, f"notebooks {nbs}", "small t-accent")
        c.text(x + 70, 184, t1, "small")
        c.text(x + 70, 202, t2, "small")
    _row_arrows(c, xs, 140, 120)
    _takeaway(c, 230, "Then: five independent capstones on public data, one at a time.")
    return "roadmap.svg", c.render(
        "Five-day roadmap: Day 1 reproducible projects, Day 2 evaluation, Day 3 serving, "
        "Day 4 deployment and operations, Day 5 held-out test and interview practice, then capstones."
    ), "Five-day roadmap of Level 1"


@diagram
def architecture():
    c = Canvas(W, 400)
    c.title("The running project, end to end")
    c.text(30, 76, "Days 1-2 · data → model", "small t-accent", anchor="start")
    steps = [("Raw extract", "4 tables, checksummed", "default"),
             ("Validate", "contracts + cleaning", "default"),
             ("SQL features", "point-in-time", "accent"),
             ("Split", "time + embargo", "default"),
             ("Pipeline", "preprocess + model", "accent"),
             ("Registry", "v1 v2 v3 + CURRENT", "good")]
    xs = [30 + i * 138 for i in range(6)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 90, 122, 64, lbl, sub, role=role)
    _row_arrows(c, xs, 122, 122)

    c.text(30, 266, "Days 3-4", "lbl", anchor="start")
    c.text(30, 286, "model → service → operations", "small t-accent", anchor="start")
    c.box(444, 250, 122, 64, "Client", "timeouts · retries")
    c.box(582, 250, 122, 64, "FastAPI service", "/predict /ready", role="accent")
    c.box(720, 250, 122, 64, "Monitoring", "drift · latency · labels", role="warn")
    c.arrow(568, 282, 578, 282, accent=True)
    c.arrow(706, 282, 716, 282, accent=True)
    c.path("M 781 156 L 781 200 L 643 200 L 643 246", accent=True)
    c.text(712, 192, "loads CURRENT", "small")

    c.rule(30, 340, 850, 340)
    c.text(30, 368, "The saved pipeline travels as one object; rolling back = pointing CURRENT at the previous version.",
           "small", anchor="start")
    return "architecture.svg", c.render(
        "Project architecture: raw extract, validation, SQL features, time split, pipeline and registry; "
        "the FastAPI service loads the CURRENT model, a client calls it, and monitoring watches it."
    ), "The Level 1 project end to end"


@diagram
def point_in_time():
    c = Canvas(W, 350)
    c.title("One snapshot T: features look back, the label looks forward")
    T = 500
    c.path(f"M {T} 70 L {T} 240", dashed=True, arrow=False)
    c.box(130, 92, T - 130, 36, "ticket and payment windows: [T − 90d, T)", role="muted", label_cls="small")
    c.box(250, 146, T - 10 - 250, 46, "usage: completed weeks", "week_start + 7d ≤ T", role="accent")
    c.box(T + 10, 146, 250, 46, "label", "churned in [T, T + 30d)", role="good")
    c.arrow(40, 214, 850, 214)
    c.text(850, 236, "time", "small", anchor="end")
    c.text(130, 236, "T − 90d", "small")
    c.text(760, 236, "T + 30d", "small")
    c.box(415, 246, 170, 46, "open week", "ends after T: excluded", role="bad")
    c.text(T, 64, "snapshot T", "lbl")
    c.text(30, 330, "Leakage = any feature built from rows at or after T, including a 'status' column exported today.",
           "small", anchor="start")
    return "point-in-time.svg", c.render(
        "Timeline around snapshot T. Ticket and payment features use the 90 days before T, usage features use "
        "only weeks completed before T, the label is churn in the 30 days from T, and a week that ends after T "
        "is excluded."
    ), "Point-in-time features and labels around a snapshot"


@diagram
def splits():
    c = Canvas(W, 330)
    c.title("Same rows, three splits, three questions")
    c.arrow(220, 72, 676, 72)
    c.text(220, 64, "monthly snapshots, oldest → newest", "small", anchor="start")
    rows = [
        ("Random rows", "bad", ["accent", "accent", "good", "accent", "warn", "accent", "accent"],
         "mixes past and future;", "same customer on both sides"),
        ("Group (customer)", "default", None, "customers never seen;", "says nothing about next month"),
        ("Time + embargo", "good", None, "how well next month?", "this project's question"),
    ]
    for r, (name, role, pattern, l1, l2) in enumerate(rows):
        y = 90 + r * 66
        c.box(30, y, 170, 48, name, role=role)
        for i in range(21):
            if r == 0:
                cell = pattern[(i * 5) % 7]
            elif r == 1:
                cell = "good" if i % 5 in (1, 3) else "accent"
            else:
                cell = "accent" if i < 12 else ("ghost" if i == 12 else ("good" if i < 16 else "warn"))
            c.box(220 + i * 22, y + 6, 18, 36, role=cell, radius=3)
        tone = "small t-good" if r == 2 else "small"
        c.text(694, y + 20, l1, tone, anchor="start")
        c.text(694, y + 37, l2, tone, anchor="start")
    legend = [("accent", "train"), ("good", "validation"), ("warn", "test (once, nb 15)"),
              ("ghost", "embargo gap (label window)")]
    x = 220
    for role, name in legend:
        c.box(x, 298, 14, 14, role=role, radius=3)
        c.text(x + 20, 310, name, "small", anchor="start")
        x += 20 + len(name) * 6.4 + 22
    return "splits.svg", c.render(
        "Three ways to split monthly snapshots: random rows mix past and future, a group split keeps whole "
        "customers apart, and a time split trains on early months, leaves an embargo gap, then validates and tests "
        "on later months."
    ), "Random, group and time splits"


@diagram
def serving():
    c = Canvas(W, 350)
    c.title("Inside the prediction service")
    c.box(30, 150, 120, 60, "Caller", "JSON request")
    c.box(190, 70, 280, 220, role="ghost")
    c.text(330, 92, "FastAPI app", "lbl")
    c.box(210, 106, 240, 50, "Middleware", "request id · timing · log line")
    c.box(210, 166, 240, 50, "Pydantic contract", "bad input → 422", role="accent")
    c.box(210, 226, 240, 50, "Endpoints", "/predict /ready /health /metrics")
    c.box(520, 110, 170, 70, "ModelHolder", "pipeline in memory, swapped under a lock", role="accent")
    c.box(730, 110, 120, 70, "Registry", "v1 v2 v3 + CURRENT", role="good")
    c.box(520, 220, 330, 60, "Logs and metrics", "JSON lines · p50 / p95 latency · recent inputs for drift",
          role="warn")
    c.arrow(152, 180, 186, 180, accent=True)
    c.arrow(472, 145, 516, 145, accent=True, label="score")
    c.arrow(728, 145, 694, 145, label="load")
    c.arrow(472, 250, 516, 250)
    c.text(30, 326, "The saved object includes preprocessing, so the service cannot transform inputs differently "
           "from training.", "small", anchor="start")
    return "serving.svg", c.render(
        "Inside the prediction service: a caller sends JSON to the FastAPI app, whose middleware logs and times "
        "the request and whose Pydantic contract rejects bad input with 422. The model holder keeps the pipeline "
        "loaded from the registry, and logs and metrics record latency and recent inputs."
    ), "Inside the prediction service"


@diagram
def monitoring():
    c = Canvas(W, 330)
    c.title("What you can know, and when", "Three layers, three clocks.")
    cols = [("Service health", "available now", "errors · latency · readiness", "good"),
            ("Data and score drift", "available now", "PSI per feature · score mean · unseen categories", "accent"),
            ("Model performance", "30+ days later", "PR-AUC and calibration on matured labels", "warn")]
    for i, (lbl, when, sub, role) in enumerate(cols):
        x = 30 + i * 280
        c.box(x, 80, 260, 90, lbl, sub, role=role)
        c.text(x + 130, 194, when, f"small t-{'warn' if i == 2 else 'good'}")
        c.arrow(x + 130, 204, x + 130, 236)
    c.box(30, 240, 820, 60, "Decision", "page → roll back (move the CURRENT pointer) · warn → investigate or retrain")
    return "monitoring.svg", c.render(
        "Monitoring layers: service health and drift are available immediately, model performance only after "
        "the 30-day label delay. All three feed one decision: roll back on a page, investigate on a warning."
    ), "What you can monitor, and when"


# ---------------------------------------------------------------------------
# 01 - the problem and reproducibility
# ---------------------------------------------------------------------------


@diagram
def churn_decision_loop():
    c = Canvas(W, 380)
    c.title("The monthly churn loop", "What the model is for: a ranked list a team can act on.")
    y = 90
    steps = [("Snapshot", "1st of the month", "default"),
             ("Score", "every active account", "accent"),
             ("Rank", "highest risk first", "accent"),
             ("Contact top N", "retention team's capacity", "good")]
    xs = [30 + i * 210 for i in range(4)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, y, 170, 60, lbl, sub, role=role)
    _row_arrows(c, xs, 170, y + 30)
    y2 = 210
    back = [("Retrain / monitor", "next month's model", "warn"),
            ("Label", "churned in the 30 days?", "default"),
            ("Outcome", "observed 30 days later", "default")]
    xs2 = [240, 450, 660]
    for x, (lbl, sub, role) in zip(xs2, back):
        c.box(x, y2, 170, 60, lbl, sub, role=role)
    c.arrow(745, 152, 745, 206, accent=True)
    c.arrow(658, y2 + 30, 624, y2 + 30, accent=True)
    c.arrow(448, y2 + 30, 414, y2 + 30, accent=True)
    c.path(f"M 238 {y2 + 30} L 115 {y2 + 30} L 115 154", accent=True, dashed=True)
    c.text(176, y2 + 22, "next month", "small")
    _takeaway(c, 310, "The model ranks; people act; value = customers kept − cost of effort.", role="good")
    return "l1-churn-decision-loop.svg", c.render(
        "The monthly churn loop: on the first of the month every active account is scored and ranked, the "
        "retention team contacts the top N it has capacity for, the outcome is observed 30 days later and becomes "
        "the label, which feeds retraining and monitoring for the next month."
    ), "The monthly churn decision loop: snapshot, score, rank, contact, outcome, label, retrain."


@diagram
def four_pins():
    c = Canvas(W, 330)
    c.title("Four pins make a number reproducible", "Plus a fifth that no tool can pin for you.")
    pins = [("Code", "git commit, clean tree", "accent"),
            ("Data", "SHA-256 of every file", "accent"),
            ("Environment", "lock file: exact versions", "accent"),
            ("Randomness", "random_state everywhere", "accent"),
            ("Decisions", "written next to the result", "muted")]
    xs = [30 + i * 166 for i in range(5)]
    for x, (lbl, sub, role) in zip(xs, pins):
        c.box(x, 80, 148, 64, lbl, sub, role=role)
        c.arrow(x + 74, 146, x + 74, 196, accent=role == "accent", dashed=role == "muted")
    c.box(30, 200, 812, 54, "The same number, on another machine, months later", role="good")
    c.text(30, 294, "Miss one pin and you will eventually get a different number with no idea why. "
           "A seed makes a run repeatable, not reliable.", "small", anchor="start")
    return "l1-four-pins.svg", c.render(
        "Code, data, environment and randomness must all be pinned for a result to reproduce; the decisions "
        "behind it are written down next to the result."
    ), "The four pins (code, data, environment, randomness) plus written decisions."


@diagram
def run_record():
    c = Canvas(W, 360)
    c.title("Anatomy of a run record", "A config that is plain data can be hashed, and the hash names the run.")
    steps = [("config dict", "every knob, one object", "default"),
             ("canonical JSON", "sort_keys=True", "accent"),
             ("SHA-256", "64 hex characters", "accent"),
             ("run_id", "first 12 hex", "good")]
    xs = [30 + i * 210 for i in range(4)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 80, 170, 56, lbl, sub, role=role)
    _row_arrows(c, xs, 170, 108)
    c.arrow(745, 138, 745, 174, accent=True)
    c.box(450, 178, 400, 100, role="ghost")
    c.text(466, 200, "runs/<run_id>.json", "lbl", anchor="start")
    fields = ["run_id · config · metrics", "created_at · git commit (code)",
              "data fingerprint · environment versions"]
    for i, f in enumerate(fields):
        c.text(466, 224 + i * 20, f, "mono", anchor="start")
    c.text(30, 196, "Same config, any key order", "lbl", anchor="start")
    c.text(30, 216, "→ same id → the same file is overwritten,", "small", anchor="start")
    c.text(30, 234, "not a new one piled up.", "small", anchor="start")
    c.text(30, 262, "Change any value → a new id.", "small", anchor="start")
    c.text(30, 320, "This is all an experiment tracker does at its core: config, versions, metrics, under an id.",
           "small", anchor="start")
    return "l1-run-record.svg", c.render(
        "A configuration dictionary is serialised as canonical JSON with sorted keys, hashed with SHA-256, and "
        "the first 12 hex characters become the run id; the run record stores config, metrics, time, code, data "
        "and environment versions under that id."
    ), "How a config becomes a run id, and what a run record holds."


# ---------------------------------------------------------------------------
# 02 - data acquisition and validation
# ---------------------------------------------------------------------------


@diagram
def validation_flow():
    c = Canvas(W, 330)
    c.title("From download to trusted tables", "Each step either proves something or records a decision.")
    steps = [("Download", "to name.part", "default"),
             ("Verify", "SHA-256 matches?", "accent"),
             ("Read", "only empty = missing", "accent"),
             ("Contract", "types, ranges, keys", "accent"),
             ("Investigate", "evidence per defect", "warn"),
             ("Clean", "one stated decision each", "accent"),
             ("Re-validate", "relaxed only where cleaned", "good")]
    xs = [30 + i * 118 for i in range(7)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 90, 104, 76, lbl, sub, role=role)
    _row_arrows(c, xs, 104, 128)
    c.arrow(xs[1] + 52, 168, xs[1] + 52, 206, label="no", label_dy=0)
    c.box(xs[1] - 20, 210, 144, 56, "Refuse", "delete .part, raise", role="bad")
    c.arrow(xs[3] + 52, 168, xs[3] + 52, 206)
    c.box(xs[3] - 20, 210, 144, 56, "Report", "lists issues, never raises", role="warn")
    c.text(30, 306, "A contract says that something is wrong; only an investigation says what to do about it.",
           "small", anchor="start")
    return "l1-validation-flow.svg", c.render(
        "Data pipeline: download to a .part file, verify the checksum (refuse on mismatch), read with only empty "
        "fields as missing, check the data contract (which reports issues), investigate each defect, clean with "
        "one stated decision each, and re-validate."
    ), "Download, verify, read, contract, investigate, clean, re-validate."


@diagram
def na_trap():
    c = Canvas(W, 330)
    c.title("The \"NA\" trap", "One region code, two readers, two different datasets.")
    c.box(30, 120, 170, 76, "C00017,NA,pro", "the raw bytes: NA = North America", mono=True)
    c.path("M 202 158 L 220 158 L 220 108 L 236 108")
    c.path("M 202 158 L 220 158 L 220 228 L 236 228")
    c.box(240, 80, 220, 56, "pd.read_csv(path)", "default NA strings", role="bad", mono=True)
    c.box(240, 200, 220, 56, "keep_default_na=False", "na_values=[\"\"]", role="good", mono=True)
    c.box(500, 80, 140, 56, "region = NaN", role="bad")
    c.box(500, 200, 140, 56, "region = \"NA\"", role="good")
    c.box(680, 80, 170, 56, "≈ 42% missing", "and no NA region at all", role="bad")
    c.box(680, 200, 170, 56, "≈ 2,005 NA", "only 87 truly missing", role="good")
    for y in (108, 228):
        c.arrow(462, y, 496, y)
        c.arrow(642, y, 676, y)
    c.text(30, 300, "When a number surprises you (42% missing in a required field), look at the raw bytes.",
           "small", anchor="start")
    return "l1-na-trap.svg", c.render(
        "The same raw line read two ways: with pandas defaults the North America code NA becomes NaN and 42% "
        "of regions look missing; with keep_default_na=False only empty fields are missing and 87 regions are "
        "truly missing."
    ), "pandas reads the region code NA as missing unless told otherwise."


# ---------------------------------------------------------------------------
# 03 - SQL
# ---------------------------------------------------------------------------


@diagram
def join_fanout():
    c = Canvas(W, 370)
    c.title("The join fan-out bug", "One customer: 41 logged weeks, 7 tickets.")
    c.text(30, 76, "Wrong: join two one-to-many tables, then aggregate", "small t-bad", anchor="start")
    c.box(30, 86, 170, 34, "usage: 41 rows", label_cls="small")
    c.box(30, 130, 170, 34, "tickets: 7 rows", label_cls="small")
    c.box(250, 86, 160, 78, "JOIN both", "on customer_id", role="bad")
    c.box(460, 86, 170, 78, "287 rows", "every (week, ticket) pair", role="bad")
    c.box(680, 86, 170, 78, "1,484 sessions", "true value × 7", role="bad")
    c.arrow(202, 103, 246, 113)
    c.arrow(202, 147, 246, 137)
    c.arrow(412, 125, 456, 125)
    c.arrow(632, 125, 676, 125)

    c.text(30, 216, "Right: aggregate each table to one row per key, then join", "small t-good", anchor="start")
    c.box(30, 226, 170, 34, "usage → 1 row", label_cls="small")
    c.box(30, 270, 170, 34, "tickets → 1 row", label_cls="small")
    c.box(250, 226, 160, 78, "LEFT JOIN", "1 × 1 per customer", role="good")
    c.box(460, 226, 170, 78, "1 row", "per customer", role="good")
    c.box(680, 226, 170, 78, "212 sessions", "and 7 tickets", role="good")
    c.arrow(202, 243, 246, 253, accent=True)
    c.arrow(202, 287, 246, 277, accent=True)
    c.arrow(412, 265, 456, 265, accent=True)
    c.arrow(632, 265, 676, 265, accent=True)
    c.text(30, 344, "No error and no warning: 1,484 = 212 × 7. Check row counts after every join.",
           "small", anchor="start")
    return "l1-join-fanout.svg", c.render(
        "Joining usage (41 rows) and tickets (7 rows) to a customer before aggregating produces 287 row pairs and "
        "inflates sessions sevenfold to 1,484; aggregating each table to one row per customer first gives the true "
        "212 sessions and 7 tickets."
    ), "Join fan-out: aggregate each table first, then join."


@diagram
def rows_vs_range():
    c = Canvas(W, 330)
    c.title("ROWS vs RANGE window frames", "\"Sessions in the last 4 weeks\" for the Apr 22 row; Apr 1 was never logged.")
    weeks = ["Mar 11", "Mar 18", "Mar 25", "Apr 1", "Apr 8", "Apr 15", "Apr 22"]
    x0, step, w = 210, 92, 80
    for i, wk in enumerate(weeks):
        x = x0 + i * step
        if wk == "Apr 1":
            c.box(x, 80, w, 50, wk, "no row", role="ghost")
        elif wk == "Apr 22":
            c.box(x, 80, w, 50, wk, "current", role="accent")
        else:
            c.box(x, 80, w, 50, wk, "logged")
    c.text(30, 110, "week_start", "small", anchor="start")
    # ROWS 3 PRECEDING: Mar 25, Apr 8, Apr 15, Apr 22
    c.text(30, 172, "ROWS 3 PRECEDING", "mono", anchor="start")
    c.box(x0 + 2 * step, 150, 5 * step - 12, 36, "4 rows, but 5 weeks of time", role="bad", label_cls="small")
    c.text(30, 226, "RANGE 21 DAY", "mono", anchor="start")
    c.box(x0 + 3 * step, 204, 4 * step - 12, 36, "4 weeks of time, 3 rows", role="good", label_cls="small")
    c.text(30, 290, "ROWS counts rows; RANGE measures time. They agree only when no week is missing.",
           "small", anchor="start")
    c.text(30, 308, "For a feature defined in time (\"last 4 weeks\"), RANGE is the correct frame.",
           "small", anchor="start")
    return "l1-rows-vs-range.svg", c.render(
        "Weekly rows with a missing week: a ROWS 3 PRECEDING frame takes four rows and reaches five weeks back, "
        "while a RANGE 21 DAY PRECEDING frame stays within four weeks and takes three rows."
    ), "ROWS frames count rows; RANGE frames measure time."


# ---------------------------------------------------------------------------
# 04 - splits and leakage
# ---------------------------------------------------------------------------


@diagram
def embargo():
    c = Canvas(W, 340)
    c.title("Why the split has an embargo gap", "Each snapshot's label looks 30 days ahead. Boundary moved to mid-March.")
    B = 425
    c.box(40, 70, B - 46, 28, "train", role="accent", label_cls="small")
    c.box(B + 6, 70, 840 - B - 6, 28, "validation", role="good", label_cls="small")
    c.path(f"M {B} 104 L {B} 270", dashed=True, arrow=False)
    months = [("Jan 1", 100), ("Feb 1", 230), ("Mar 1", 360), ("Apr 1", 490), ("May 1", 620), ("Jun 1", 750)]
    for name, x in months:
        c.circle(x, 240, 5, role="accent" if x < B else "good")
        c.text(x, 262, name, "small")
    c.arrow(40, 240, 850, 240)
    for name, x in months[:5]:
        role = "accent" if x + 120 < B else ("bad" if x < B else "good")
        lbl = "crosses: dropped" if role == "bad" else f"{name} + 30d"
        c.box(x, 160 if role != "bad" else 116, 120, 36, lbl, role=role, label_cls="small")
        c.path(f"M {x} {196 if role != 'bad' else 152} L {x} 232", dashed=True, arrow=False)
    c.text(B, 286, "train_end = Mar 15", "small t-bad")
    c.text(30, 322, "Keep a training row only if T + gap ≤ train_end: its label must be settled before validation starts.",
           "small", anchor="start")
    return "l1-embargo.svg", c.render(
        "Monthly snapshots with 30-day label windows and a train boundary on March 15: the March 1 snapshot's "
        "label window crosses the boundary, so the embargo gap drops it from training."
    ), "The embargo gap: drop training rows whose label window crosses the boundary."


@diagram
def leakage_types():
    c = Canvas(W, 340)
    c.title("Four kinds of leakage, measured on this project", "Validation PR-AUC, honest → leaky.")
    heads = [("Target leak", "feature contains the label"),
             ("Temporal leak", "uses rows from after T"),
             ("Group leak", "same customer both sides"),
             ("Preprocessing leak", "statistics fit on eval rows")]
    rows = [("Example", ["cust_rate_full", "status at export", "random row split", "scaler fit on all rows"]),
            ("Measured", ["0.16 → 0.58", "0.12 → 0.99", "0.12 → 0.13-0.15", "0.1632 → 0.1633"]),
            ("Defence", ["only closed, past labels", "every feature < T", "time or group split",
                         "fit inside the Pipeline"])]
    for i, (h, s) in enumerate(heads):
        c.box(150 + i * 176, 70, 164, 56, h, s, role="bad")
    for r, (name, cells) in enumerate(rows):
        y = 158 + r * 40
        c.text(30, y, name, "lbl", anchor="start")
        for i, cell in enumerate(cells):
            tone = "small t-good" if name == "Defence" else ("mono" if name == "Measured" else "small")
            c.text(232 + i * 176, y, cell, tone)
        c.rule(30, y + 14, 850, y + 14)
    _takeaway(c, 284, "Feature leaks fool every split; only point-in-time construction prevents them.", role="warn")
    return "l1-leakage-types.svg", c.render(
        "Four kinds of leakage with their measured effect on this project: a target leak (0.16 to 0.58), a "
        "temporal leak from export-time columns (0.12 to 0.99), a group leak from a random split (a small rise) "
        "and a preprocessing leak (negligible), each with its defence."
    ), "Target, temporal, group and preprocessing leakage, with measured sizes and defences."


# ---------------------------------------------------------------------------
# 05 - pipelines and CV
# ---------------------------------------------------------------------------


@diagram
def pipeline_inside():
    c = Canvas(W, 330)
    c.title("Preprocessing inside the model", "The fitted object is the whole recipe, so evaluation and serving cannot diverge.")
    c.text(30, 82, "Wrong: preprocess everything, then split", "small t-bad", anchor="start")
    top = [("All rows", "train + validation", "default"),
           ("Fit imputer, scaler", "sees eval rows", "bad"),
           ("Split, fit model", "on transformed arrays", "default"),
           ("Serve", "re-implement each step", "bad")]
    xs = [30 + i * 210 for i in range(4)]
    for x, (lbl, sub, role) in zip(xs, top):
        c.box(x, 92, 180, 56, lbl, sub, role=role)
    _row_arrows(c, xs, 180, 120, accent=False)
    c.text(30, 192, "Right: split first, fit one Pipeline on train", "small t-good", anchor="start")
    bot = [("Split by time", "train | validation", "default"),
           ("Pipeline.fit(train)", "impute → scale → model", "accent"),
           ("One saved object", "model.joblib", "good"),
           ("Serve that object", "predict_proba(raw rows)", "good")]
    for x, (lbl, sub, role) in zip(xs, bot):
        c.box(x, 202, 180, 56, lbl, sub, role=role)
    _row_arrows(c, xs, 180, 230)
    c.text(30, 300, "Leak size here was tiny (0.1632 vs 0.1633); the skew it prevents in serving is not.",
           "small", anchor="start")
    return "l1-pipeline-inside.svg", c.render(
        "Wrong: fitting preprocessing on all rows before splitting lets evaluation rows shape it, and serving has to "
        "re-implement it. Right: split first, fit one Pipeline on train, save it, and serve the same object."
    ), "Preprocessing inside the Pipeline vs fitted separately."


@diagram
def rolling_origin():
    c = Canvas(W, 330)
    c.title("Rolling-origin cross-validation", "Inside the train period: expand the past, validate on the next two months.")
    n = 13
    x0, step = 150, 52
    c.text(x0, 70, "train-period snapshots, Mar 2023 → Mar 2024", "small", anchor="start")
    starts = [6, 8, 9, 11]
    for f, s in enumerate(starts):
        y = 84 + f * 44
        c.text(30, y + 22, f"fold {f + 1}", "lbl", anchor="start")
        for i in range(n):
            if i < s - 1:
                role = "accent"
            elif i == s - 1:
                role = "ghost"
            elif i < s + 2:
                role = "good"
            else:
                role = "muted"
            c.box(x0 + i * step, y + 4, 44, 30, role=role, radius=4)
    legend = [("accent", "fit"), ("ghost", "gap (label window)"), ("good", "validate"), ("muted", "unused")]
    x = x0
    for role, name in legend:
        c.box(x, 268, 14, 14, role=role, radius=3)
        c.text(x + 20, 280, name, "small", anchor="start")
        x += 20 + len(name) * 6.4 + 26
    c.text(30, 314, "Never a future month in the fit. The validation period (Apr-Jun 2024) plays no part.",
           "small", anchor="start")
    return "l1-rolling-origin.svg", c.render(
        "Four rolling-origin folds over the training snapshots: each fold fits on all earlier months, leaves a gap "
        "for the label window, and validates on the next two months; training windows grow from fold to fold."
    ), "Rolling-origin (expanding-window) CV folds."


# ---------------------------------------------------------------------------
# 06 - model selection
# ---------------------------------------------------------------------------


@diagram
def bootstrap_units():
    c = Canvas(W, 360)
    c.title("Bootstrap: resample rows or customers?", "Three customers, three monthly rows each. Draw with replacement.")
    roles = {"A": "accent", "B": "good", "C": "warn"}

    def cells(x, y, seq):
        for i, k in enumerate(seq):
            c.box(x + i * 40, y, 34, 34, k, role=roles[k], radius=4, label_cls="small")

    for px, title, out, note in [
        (30, "Rows", "ACBAACCBC", "rows treated as independent"),
        (460, "Customers (cluster)", "AAACCCAAA", "whole customers: A, C, A"),
    ]:
        c.text(px, 84, title, "lbl", anchor="start")
        cells(px, 96, "AAABBBCCC")
        c.arrow(px + 176, 134, px + 176, 170, label="resample", label_dy=4)
        c.text(px + 196, 156, "", "small")
        cells(px, 178, out)
        c.text(px, 236, note, "small", anchor="start")
    c.rule(440, 80, 440, 240)
    c.text(30, 278, "Copies are not new evidence. Triple every row: the row interval shrinks by √3,", "small",
           anchor="start")
    c.text(30, 296, "the customer interval does not. Resample the unit that is independent.", "small", anchor="start")
    _takeaway(c, 310, "Report the customer-level interval by default.", role="good", h=36)
    return "l1-bootstrap-units.svg", c.render(
        "Row bootstrap draws individual rows and mixes customers; the cluster bootstrap draws whole customers with "
        "all their rows, which is the honest unit when a customer's rows are correlated."
    ), "Row vs cluster (customer) bootstrap resampling."


# ---------------------------------------------------------------------------
# 07 - thresholds and calibration
# ---------------------------------------------------------------------------


@diagram
def pr_vs_roc():
    c = Canvas(W, 360)
    c.title("ROC and PR curves at a 5% positive rate", "Same model, two views. Only one has a baseline that moves with the positive rate.")
    # ROC
    x0, y0, w, h = 80, 80, 300, 200
    _axes(c, x0, y0, w, h, "false positive rate", "true positive rate")
    c.curve(_poly([(x0, y0 + h), (x0 + w, y0)]), tone="muted_fg", dashed=True, width=1.6)
    pts = [(x0 + w * t, y0 + h - h * (1 - (1 - t) ** 2.6)) for t in [i / 40 for i in range(41)]]
    c.curve(_poly(pts), tone="accent")
    c.text(x0 + 200, y0 + 150, "random = 0.5", "small")
    c.text(x0 + 70, y0 + 16, "ROC-AUC ≈ 0.73", "small t-accent")
    # PR
    x1 = 520
    _axes(c, x1, y0, w, h, "recall", "precision")
    base = y0 + h - h * 0.05
    c.curve(_poly([(x1, base), (x1 + w, base)]), tone="muted_fg", dashed=True, width=1.6)
    pr = [(x1 + w * r, y0 + h - h * (0.05 + 0.45 * (1 - r) ** 1.5 * math.exp(-1.5 * r))) for r in
          [0.01 + i * 0.99 / 40 for i in range(41)]]
    c.curve(_poly(pr), tone="accent")
    c.text(x1 + 60, base - 8, "random = 0.05", "small")
    c.text(x1 + 190, y0 + 60, "PR-AUC ≈ 0.16", "small t-accent")
    c.text(30, 334, "ROC is flattered by thousands of easy negatives; PR shows the top of the ranking, "
           "where the calls are made.", "small", anchor="start")
    return "l1-pr-vs-roc.svg", c.render(
        "Two plots of the same model: the ROC curve with the random diagonal at 0.5, and the precision-recall "
        "curve whose random baseline is the positive rate, 0.05."
    ), "ROC vs precision-recall curves on a 5% positive rate."


@diagram
def threshold_value():
    c = Canvas(W, 360)
    c.title("Choosing a threshold from costs", "Expected value of calling everyone above t, with VALUE_TP = $150 and COST_FP = $30.")
    x0, y0, w, h = 90, 80, 520, 210
    _axes(c, x0, y0, w, h, "threshold t", "expected value ($)")
    zero = y0 + 140
    c.curve(_poly([(x0, zero), (x0 + w, zero)]), tone="stroke", width=1.4)
    c.text(x0 - 8, zero + 4, "0", "small", anchor="end")

    def ev(t):  # schematic: negative at low t, broad top near 1/6, decays to 0
        return 120 * (1 - math.exp(-(t - 0.07) * 18)) * math.exp(-(t - 0.07) * 4.5)

    pts = [(x0 + w * t / 0.6, zero - ev(t)) for t in [0.05 + i * 0.005 for i in range(111)]]
    c.curve(_poly(pts), tone="accent")
    px = x0 + w * (1 / 6) / 0.6
    c.curve(f"M {px:.1f} {y0} L {px:.1f} {y0 + h}", tone="good", dashed=True, width=1.6)
    c.text(px, y0 - 6, "p* = 30 / (150 + 30) = 1/6", "small t-good")
    c.text(px + 14, zero + 30, "← left of p*: too many calls,", "small t-bad", anchor="start")
    c.text(px + 14, zero + 46, "each wasted one costs $30", "small t-bad", anchor="start")
    c.text(x0 + w - 10, zero - 80, "far right: too few calls,", "small t-warn", anchor="end")
    c.text(x0 + w - 10, zero - 64, "churners left on the table", "small t-warn", anchor="end")
    c.box(650, 90, 200, 120, "Call when p > 1/6", "p · 150 > (1 − p) · 30. Valid only for calibrated p.",
          role="good")
    c.text(650, 240, "Only the ratio cost/value", "small", anchor="start")
    c.text(650, 258, "sets p*; the flat top means", "small", anchor="start")
    c.text(650, 276, "0.17 vs 0.19 is noise.", "small", anchor="start")
    return "l1-threshold-value.svg", c.render(
        "Expected value against threshold: negative when too many low-probability accounts are called, a broad "
        "maximum near the break-even probability 1/6, and falling towards zero when too few are called."
    ), "Expected value vs threshold, and the break-even probability p* = cost/(value+cost)."


@diagram
def calibration():
    c = Canvas(W, 360)
    c.title("Reading a reliability diagram", "Bin by predicted score; compare the mean score with the observed churn rate.")
    x0, y0, w, h = 90, 70, 240, 240
    _axes(c, x0, y0, w, h, "mean predicted probability", "observed churn rate")
    c.curve(_poly([(x0, y0 + h), (x0 + w, y0)]), tone="muted_fg", dashed=True, width=1.6)
    good = [(0.02, 0.02), (0.08, 0.07), (0.15, 0.16), (0.25, 0.24), (0.38, 0.40), (0.55, 0.53)]
    bad = [(0.10, 0.01), (0.25, 0.03), (0.40, 0.05), (0.58, 0.09), (0.75, 0.15), (0.9, 0.22)]
    for pts, tone in ((good, "good"), (bad, "bad")):
        xy = [(x0 + w * p, y0 + h - h * o) for p, o in pts]
        c.curve(_poly(xy), tone=tone)
        for x, y in xy:
            c.circle(x, y, 3.5, role=tone)
    c.text(x0 + 150, y0 + 70, "perfect", "small")
    c.box(380, 80, 220, 64, "Plain logreg", "on the diagonal · Brier ≈ 0.047", role="good")
    c.box(380, 160, 220, 64, "class_weight=balanced", "far below it · Brier ≈ 0.19", role="bad")
    c.box(630, 80, 220, 144, "Same ranking", "PR-AUC 0.163 vs 0.161. Re-weighting moved the probabilities, not the order.",
          role="ghost")
    c.text(380, 270, "Below the diagonal = over-confident. A monotone recalibration (sigmoid) fixes the",
           "small", anchor="start")
    c.text(380, 288, "probabilities and leaves PR-AUC unchanged. Fit it on later rows, never validation.",
           "small", anchor="start")
    return "l1-calibration.svg", c.render(
        "Reliability diagram: a calibrated logistic regression follows the diagonal, while a class-weighted model "
        "sits far below it, predicting several times the true churn rate, with almost the same ranking quality."
    ), "Reliability diagram: calibrated vs over-confident scores."


# ---------------------------------------------------------------------------
# 08 - bundle and registry
# ---------------------------------------------------------------------------


@diagram
def bundle_registry():
    c = Canvas(W, 350)
    c.title("A registry: immutable versions and one pointer",
            "artifacts/registry/ - every vN/ folder holds model.joblib + metadata.json.")
    vers = [("v1", "previous · rollback target", "default"),
            ("v2", "live", "good"),
            ("v3", "candidate, not promoted", "muted")]
    for i, (v, sub, role) in enumerate(vers):
        c.box(30 + i * 220, 80, 200, 76, v, sub, role=role)
    c.box(690, 80, 160, 196, "history.jsonl", "append-only: register v1, register v2, promote v2, register v3, …")
    c.box(250, 216, 200, 60, "CURRENT", "the file contains: v2", role="accent")
    c.arrow(350, 214, 350, 160, accent=True)
    c.path("M 248 246 L 130 246 L 130 160", dashed=True)
    c.text(140, 268, "rollback = write v1", "small t-bad", anchor="start")
    c.text(30, 320, "Versions are never overwritten. Deploying and rolling back only rewrite one tiny file, "
           "and both are logged.", "small", anchor="start")
    return "l1-bundle-registry.svg", c.render(
        "A model registry: immutable version folders v1, v2 and v3, each with a model file and metadata, a CURRENT "
        "file pointing at the live version, an append-only history log, and rollback as rewriting CURRENT."
    ), "Model bundles in a registry, with a CURRENT pointer and rollback."


# ---------------------------------------------------------------------------
# 09 - 11 serving
# ---------------------------------------------------------------------------


@diagram
def request_lifecycle():
    c = Canvas(W, 340)
    c.title("The life of a /predict request", "Every rejection happens before any model code runs.")
    main = [("Request", "JSON body", "default"),
            ("Validate", "Pydantic schema", "accent"),
            ("Size check", "batch ≤ 1000", "accent"),
            ("Model ready?", "CURRENT loaded", "accent"),
            ("Score", "the saved pipeline", "accent"),
            ("200", "score, decision, version", "good")]
    xs = [30 + i * 138 for i in range(6)]
    for x, (lbl, sub, role) in zip(xs, main):
        c.box(x, 90, 122, 64, lbl, sub, role=role)
    _row_arrows(c, xs, 122, 122)
    fails = [(1, "422", "names the bad field", "invalid"),
             (2, "413", "chunk the batch", "too big"),
             (3, "503", "retry later", "no model")]
    for i, code, sub, why in fails:
        x = xs[i]
        c.arrow(x + 61, 156, x + 61, 214, label=why, label_dy=0)
        c.box(x, 218, 122, 56, code, sub, role="bad")
    c.text(30, 314, "A 4xx is the caller's fault and must not be retried; a 503 may succeed later.",
           "small", anchor="start")
    return "l1-request-lifecycle.svg", c.render(
        "A prediction request is validated against the schema (422 on failure), checked for batch size (413), "
        "checked for a loaded model (503), then scored and answered with 200 plus the model version."
    ), "Request lifecycle with its 422, 413 and 503 exits."


@diagram
def serving_skew():
    c = Canvas(W, 360)
    c.title("Training/serving skew", "Two implementations of \"the same\" preprocessing are two different models.")
    c.text(30, 80, "Wrong: preprocessing written twice", "small t-bad", anchor="start")
    c.box(30, 90, 170, 46, "training rows", label_cls="small")
    c.box(240, 90, 220, 46, "notebook transform", "log1p · medians · scaling")
    c.box(30, 146, 170, 46, "API request", label_cls="small")
    c.box(240, 146, 220, 46, "rewritten in service", "one step forgotten", role="bad")
    c.box(500, 90, 150, 102, "same coefficients", role="default")
    c.box(690, 90, 160, 102, "different scores", "validated ≠ served", role="bad")
    c.arrow(202, 113, 236, 113)
    c.arrow(202, 169, 236, 169)
    c.arrow(462, 113, 496, 125)
    c.arrow(462, 169, 496, 157)
    c.arrow(652, 141, 686, 141)
    c.text(30, 236, "Right: one fitted object, trained and served", "small t-good", anchor="start")
    right = [("train rows", "default"), ("Pipeline.fit", "accent"), ("bundle", "good"), ("service loads it", "good")]
    xs = [30 + i * 210 for i in range(4)]
    for x, (lbl, role) in zip(xs, right):
        c.box(x, 246, 180, 46, lbl, role=role)
    _row_arrows(c, xs, 180, 269)
    c.text(30, 334, "Guard it with a parity test: offline predict_proba and the running API must agree on the same rows.",
           "small", anchor="start")
    return "l1-serving-skew.svg", c.render(
        "Wrong: preprocessing written once in a notebook and again in the service, with one step forgotten, gives "
        "different scores from the same coefficients. Right: one fitted pipeline is saved and served."
    ), "Training/serving skew and the one-object fix."


@diagram
def retry_policy():
    c = Canvas(W, 400)
    c.title("Which failures to retry", "Retry only what can succeed next time, with growing waits.")
    c.box(30, 150, 130, 60, "Call", "timeout = 5 s", role="accent")
    c.box(200, 150, 150, 60, "Classify", "the outcome", role="accent")
    c.arrow(162, 180, 196, 180, accent=True)
    outs = [(70, "2xx", "return the result", "good"),
            (150, "429 · 502 · 503 · 504", "or timeout / connection", "warn"),
            (230, "400 · 404 · 413 · 422", "raise now: never retry", "bad")]
    for y, lbl, sub, role in outs:
        c.box(410, y, 230, 60, lbl, sub, role=role)
        c.path(f"M 352 180 L 380 180 L 380 {y + 30} L 406 {y + 30}")
    c.box(680, 150, 170, 60, "Back off", "0.2 · 2ⁿ s, + jitter", role="warn")
    c.arrow(642, 180, 676, 180)
    c.path("M 765 212 L 765 320 L 95 320 L 95 214", dashed=True)
    c.text(430, 312, "retry (at most 3 times), then give up", "small")
    c.text(30, 354, "Waits: 0.2 s, 0.4 s, 0.8 s. A 500 is usually a bug: retry at most once.",
           "small", anchor="start")
    c.text(30, 372, "Retrying a 422 sends the same bad request four times and fails four times.",
           "small", anchor="start")
    return "l1-retry-policy.svg", c.render(
        "A client call is classified: success returns, rate-limit, gateway, timeout and connection errors back off "
        "exponentially and retry up to three times, and client errors such as 422 are raised at once."
    ), "Retry decision: which status codes to retry, with exponential backoff."


# ---------------------------------------------------------------------------
# 12 - 13 configuration and health
# ---------------------------------------------------------------------------


@diagram
def config_env():
    c = Canvas(W, 340)
    c.title("One image, configured by its environment", "Code is identical everywhere; only CHURN_* variables change.")
    c.box(30, 140, 130, 70, "Same image", "code + deps", role="default")
    envs = [(70, "dev", "CHURN_LOG_LEVEL=DEBUG"), (150, "staging", "CHURN_MODEL_VERSION=v3"),
            (230, "production", "CHURN_LOG_LEVEL=INFO")]
    for y, name, var in envs:
        c.box(200, y, 180, 60, name, var, mono=False)
        c.path(f"M 162 175 L 180 175 L 180 {y + 30} L 196 {y + 30}")
        c.path(f"M 382 {y + 30} L 400 {y + 30} L 400 175 L 416 175")
    c.box(420, 140, 160, 70, "Settings.from_env", "parse + validate once", role="accent")
    c.box(630, 80, 220, 64, "Start serving", "frozen, typed settings", role="good")
    c.box(630, 206, 220, 64, "Refuse to start", "name the variable, value, allowed", role="bad")
    c.path("M 582 160 L 606 160 L 606 112 L 626 112", accent=True)
    c.path("M 582 190 L 606 190 L 606 238 L 626 238")
    c.text(30, 320, "Fail at start-up: a bad deploy dies in seconds and the old version keeps serving. "
           "Secrets arrive the same way, never in the image.", "small", anchor="start")
    return "l1-config-env.svg", c.render(
        "One container image runs in dev, staging and production; each environment sets CHURN_ variables, "
        "Settings.from_env parses and validates them once, and the service either starts or refuses to start "
        "with a message naming the bad variable."
    ), "Configuration from environment variables, validated at start-up."


@diagram
def liveness_readiness():
    c = Canvas(W, 340)
    c.title("Liveness vs readiness", "Two probes, two questions, two different reactions.")
    cols = [(30, "GET /health", "liveness", "Can the process answer HTTP?",
             "Restart the process", "bad"),
            (460, "GET /ready", "readiness", "Is a model loaded and able to predict?",
             "Stop sending traffic, keep it running", "warn")]
    for x, ep, kind, q, act, role in cols:
        c.box(x, 76, 390, 54, ep, kind, role="accent")
        c.arrow(x + 195, 132, x + 195, 156)
        c.box(x, 160, 390, 46, q, label_cls="small")
        c.arrow(x + 195, 208, x + 195, 232)
        c.text(x + 205, 224, "if it fails", "small", anchor="start")
        c.box(x, 236, 390, 46, act, role=role)
    c.text(30, 318, "Liveness must never depend on anything outside the process: a slow model load would "
           "otherwise cause a restart loop.", "small", anchor="start")
    return "l1-liveness-readiness.svg", c.render(
        "The liveness probe asks whether the process answers HTTP and a failure restarts it; the readiness probe "
        "asks whether a model is loaded and a failure only stops traffic to it."
    ), "Liveness restarts; readiness stops traffic."


@diagram
def latency_percentiles():
    c = Canvas(W, 360)
    c.title("Latency is a distribution", "Right-skewed: the mean hides the tail that users notice.")
    x0, y0, w, h = 70, 80, 560, 200
    _axes(c, x0, y0, w, h, "latency (ms)", "share of requests")
    mu, sig = math.log(5), 0.6
    xmax = 30.0

    def pdf(x):
        return math.exp(-((math.log(x) - mu) ** 2) / (2 * sig * sig)) / (x * sig * math.sqrt(2 * math.pi))

    peak = max(pdf(0.2 + i * 0.05) for i in range(600))
    pts = [(x0 + w * x / xmax, y0 + h - (h - 20) * pdf(x) / peak) for x in [0.3 + i * 0.1 for i in range(297)]]
    c.curve(_poly(pts), tone="accent")
    marks = [("p50", 50, 0.0, "good"), ("mean", None, None, "muted_fg"), ("p95", 95, 1.645, "warn"),
             ("p99", 99, 2.326, "bad")]
    for name, _, z, tone in marks:
        val = math.exp(mu + sig * sig / 2) if z is None else math.exp(mu + sig * z)
        x = x0 + w * val / xmax
        c.curve(f"M {x:.1f} {y0 + 10} L {x:.1f} {y0 + h}", tone=tone, dashed=True, width=1.5)
        anchor = {"p50": "end", "mean": "start"}.get(name, "middle")
        dx = {"p50": -4, "mean": 4}.get(name, 0)
        c.text(x + dx, y0 + 4, f"{name} {val:.0f} ms", f"small t-{tone}" if tone != "muted_fg" else "small",
               anchor=anchor)
    for v in (0, 10, 20, 30):
        c.text(x0 + w * v / xmax, y0 + h + 14, str(v), "small")
    c.box(660, 80, 190, 96, "The tail", "1 in 20 requests is slower than p95; 1 in 100 slower than p99", role="warn")
    c.box(660, 188, 190, 92, "A page with 20 calls", "hits a slower-than-p95 call 64% of the time", role="bad")
    c.text(30, 340, "Report p50, p95, p99 and the request count, on a named machine.", "small", anchor="start")
    return "l1-latency-percentiles.svg", c.render(
        "A right-skewed latency distribution with the median, mean, 95th and 99th percentiles marked: the tail "
        "lies far to the right of the typical request."
    ), "Latency percentiles (p50, p95, p99) on a skewed distribution."


# ---------------------------------------------------------------------------
# 14 - monitoring and rollback
# ---------------------------------------------------------------------------


@diagram
def psi_bins():
    c = Canvas(W, 380)
    c.title("Population Stability Index, bin by bin", "Reference cut into quintiles (20% each); this month's share in each bin.")
    ref = [0.20] * 5
    cur = [0.30, 0.25, 0.20, 0.15, 0.10]
    x0, base, scale = 70, 250, 500
    for i, (e, a) in enumerate(zip(ref, cur)):
        x = x0 + i * 110
        c.box(x, base - e * scale, 36, e * scale, role="muted", radius=2)
        c.box(x + 40, base - a * scale, 36, a * scale, role="accent", radius=2)
        contrib = (a - e) * math.log(a / e)
        c.text(x + 38, base + 18, f"bin {i + 1}", "small")
        c.text(x + 38, base + 38, f"{contrib:.3f}", "mono")
    c.rule(60, base, 610, base)
    c.text(30, base + 38, "term", "small", anchor="start")
    c.box(x0, 300, 14, 14, role="muted", radius=2)
    c.text(x0 + 20, 312, "reference e", "small", anchor="start")
    c.box(x0 + 120, 300, 14, 14, role="accent", radius=2)
    c.text(x0 + 140, 312, "current a", "small", anchor="start")
    c.box(640, 80, 210, 72, "PSI = Σ (a − e) ln(a / e)", "every term ≥ 0", role="accent")
    c.box(640, 166, 210, 50, "PSI = 0.135", "moderate shift", role="warn")
    bands = [("< 0.1 stable", "good"), ("0.1-0.25 moderate", "warn"), ("> 0.25 large", "bad")]
    for i, (b, role) in enumerate(bands):
        c.box(640, 230 + i * 34, 210, 28, b, role=role, label_cls="small")
    c.text(30, 360, "Bands are conventions, not laws; a calendar artifact can score above 3 while nothing is wrong.",
           "small", anchor="start")
    return "l1-psi-bins.svg", c.render(
        "Reference and current shares in five quintile bins; each bin contributes (a - e) ln(a / e), and the "
        "terms sum to a PSI of 0.135, a moderate shift."
    ), "PSI computed bin by bin, with the usual bands."


@diagram
def canary_rollback():
    c = Canvas(W, 340)
    c.title("From retrain to promote, or roll back", "Every new version goes through the same gates.")
    steps = [("Retrain", "monthly job", "default"),
             ("Register vN", "immutable bundle", "default"),
             ("Gate", "validation vs CURRENT", "accent"),
             ("Shadow / canary", "live traffic, matured slice", "accent"),
             ("Compare", "PR-AUC · errors · p95", "warn")]
    xs = [30 + i * 138 for i in range(5)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 136, 122, 72, lbl, sub, role=role)
    _row_arrows(c, xs, 122, 172)
    c.box(740, 70, 110, 64, "Promote", "CURRENT = vN", role="good")
    c.box(740, 210, 110, 64, "Roll back", "CURRENT = previous", role="bad")
    c.path("M 706 160 L 722 160 L 722 102 L 736 102", accent=True)
    c.path("M 706 186 L 722 186 L 722 242 L 736 242")
    c.text(30, 312, "Rollback is a pointer write: seconds, no retraining. Labels lag 30 days, so shadow mode catches "
           "crashes first.", "small", anchor="start")
    return "l1-canary-rollback.svg", c.render(
        "A retrained model is registered, gated against the current model on validation, run in shadow or canary "
        "mode, compared on PR-AUC, error rate and p95 latency, and then promoted or rolled back by moving CURRENT."
    ), "Release flow: gate, shadow or canary, then promote or roll back."


# ---------------------------------------------------------------------------
# 17 - interview
# ---------------------------------------------------------------------------


@diagram
def design_framework():
    c = Canvas(W, 320)
    c.title("ML system design in nine steps", "Same order every time; about 35-45 minutes.")
    steps = [("1 Clarify", "goal, users, scale"), ("2 Metrics", "offline, online, guard"),
             ("3 Data", "labels and delay"), ("4 Features", "point-in-time"), ("5 Model", "baseline first"),
             ("6 Serving", "batch or online"), ("7 Evaluate", "time split, A/B"), ("8 Monitor", "drift, rollback"),
             ("9 Risks", "loops, fairness")]
    for i, (lbl, sub) in enumerate(steps):
        row, col = divmod(i, 5)
        x = 30 + col * 166
        y = 80 + row * 110
        role = "good" if i == 0 else ("warn" if i == 8 else "accent")
        c.box(x, y, 148, 60, lbl, sub, role=role)
        if col < 4 and i < 8:
            c.arrow(x + 150, y + 30, x + 162, y + 30, accent=True)
    c.path("M 778 142 L 778 162 L 104 162 L 104 186", accent=True)
    c.text(30, 300, "Start from the decision the model supports, state assumptions with numbers, and name the "
           "trade-off every time you choose.", "small", anchor="start")
    return "l1-design-framework.svg", c.render(
        "Nine steps of an ML system design answer: clarify, metrics, data, features, model, serving, evaluation, "
        "monitoring and risks."
    ), "The nine-step ML system design framework."


@diagram
def star():
    c = Canvas(W, 250)
    c.title("STAR: where the time goes", "Most of a behavioural answer is the Action: your decisions and why.")
    parts = [("S", "Situation", "context", 0.10, "default"), ("T", "Task", "your job", 0.10, "default"),
             ("A", "Action", "what you did, alternatives rejected", 0.60, "accent"),
             ("R", "Result", "number + lesson", 0.20, "good")]
    x, total = 30, 820
    for key, name, sub, share, role in parts:
        w = total * share
        c.box(x, 80, w - 4, 70, f"{key} · {int(share * 100)}%", name, role=role)
        c.text(x + (w - 4) / 2, 172, sub, "small")
        x += w
    c.text(30, 220, "Say \"I\", not \"we\". Real stories only, with a number where one honestly exists.",
           "small", anchor="start")
    return "l1-star.svg", c.render(
        "STAR answer structure as a bar: Situation 10%, Task 10%, Action 60%, Result 20%."
    ), "STAR: Situation, Task, Action, Result, with time shares."
