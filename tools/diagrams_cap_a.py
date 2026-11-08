"""Capstone walkthrough figures (cap_a) - see tools/make_diagrams.py for conventions.

Capstone 1 (C-MAPSS predictive maintenance, prefix ``cap1-``) and
Capstone 2 (ERCOT load forecasting, prefix ``cap2-``).
"""

from __future__ import annotations

from svgkit import Canvas, diagram  # noqa: F401


def _takeaway(c: Canvas, y: float, text: str, h: float = 40) -> None:
    c.box(30, y, c.width - 60, h, text, role="good", label_cls="lbl")


# --------------------------------------------------------------------------
# Capstone 1 · C-MAPSS
# --------------------------------------------------------------------------

@diagram
def cap1_engine_split():
    c = Canvas(880, 380)
    c.title("Split by engine, not by window",
            "Engines play the role of subjects: the same rule as cross-subject EEG evaluation")
    # Column headers
    c.text(230, 92, "Random window split", "lbl t-bad")
    c.text(230, 110, "leaks: one engine on both sides", "small")
    c.text(650, 92, "Group split by engine", "lbl t-good")
    c.text(650, 110, "honest: each engine on one side", "small")
    c.rule(440, 80, 440, 290)

    def row(x0: float, y: float, name: str, roles: list[str]) -> None:
        c.text(x0, y + 22, name, "small", anchor="start")
        for k, role in enumerate(roles):
            label = "train" if role == "accent" else "val"
            c.box(x0 + 70 + k * 56, y, 50, 34, label, role=role, radius=6)

    left = ["accent", "bad", "accent", "accent", "bad", "accent"]
    left2 = ["bad", "accent", "accent", "bad", "accent", "accent"]
    row(40, 140, "engine 7", left)
    row(40, 200, "engine 12", left2)
    row(460, 140, "engine 7", ["accent"] * 6)
    row(460, 200, "engine 12", ["good"] * 6)
    c.text(230, 270, "neighbouring windows share 29 of 30 cycles", "small")
    c.text(650, 270, "GroupKFold over engine IDs", "small")
    _takeaway(c, 310, "A random window split measures memory of an engine, not prognosis of a new one.")
    return ("cap1-engine-split.svg",
            c.render("Two panels. Left: windows of the same engine are assigned to both train and "
                     "validation, which leaks. Right: every window of an engine goes to one side."),
            "Capstone 1: random window split versus engine-level (group) split, the "
            "cross-subject rule applied to engines.")


@diagram
def cap1_rul_cap():
    c = Canvas(880, 400)
    c.title("The piecewise-linear RUL label",
            "Early in life the sensors look healthy, so we stop asking for a precise number there")
    x0, y0, x1, y1 = 90, 300, 640, 90          # plot box: origin bottom-left
    c.rule(x0, y0, x1 + 20, y0)                 # x axis
    c.rule(x0, y0, x0, y1 - 10)                 # y axis
    c.text((x0 + x1) / 2, y0 + 30, "cycle (failure at the right edge)", "small")
    c.text(x0 - 12, y1, "RUL", "small", anchor="end")
    # true (uncapped) RUL: a straight line from (x0, y1) to (x1, y0)
    c.curve(f"M {x0} {y1} L {x1} {y0}", tone="muted_fg", dashed=True, width=1.8)
    # capped RUL: flat at R_max then falling at slope -1 to failure
    cap_y = 190
    knee_x = x0 + (cap_y - y1) / (y0 - y1) * (x1 - x0)
    c.curve(f"M {x0} {cap_y} L {knee_x:.1f} {cap_y} L {x1} {y0}", tone="accent", width=2.6)
    c.text(x0 + 8, cap_y - 10, "R_max = 125 (a choice)", "small t-accent", anchor="start")
    c.text(x0 + 70, y1 + 4, "true RUL = last cycle − cycle", "small", anchor="start")
    c.circle(x1, y0, 5, role="bad")
    c.text(x1, y0 + 30, "failure", "small t-bad")
    # side notes
    c.box(670, 110, 190, 60, "Flat part", "wear not visible yet", role="muted")
    c.box(670, 200, 190, 60, "Sloped part", "the model must track wear", role="accent")
    _takeaway(c, 340, "Train on the capped label; report error against the uncapped truth, and test 2-3 caps.")
    return ("cap1-rul-cap.svg",
            c.render("A plot of remaining useful life against cycle. The true RUL is a straight "
                     "falling line; the capped label is flat at 125 and then falls to zero at failure."),
            "Capstone 1: the piecewise-linear (capped) RUL training label versus the true RUL.")


@diagram
def cap1_cnn():
    c = Canvas(880, 300)
    c.title("A compact 1D-CNN for RUL",
            "Convolutions slide along time; pooling makes the output independent of position")
    names = [("Window", "30 × C sensors", "default"),
             ("Conv1d + ReLU", "32 ch, kernel 5", "accent"),
             ("Conv1d + ReLU", "32 ch, kernel 5", "accent"),
             ("Global avg pool", "→ 32 numbers", "accent"),
             ("Linear head", "→ RUL in cycles", "good")]
    w, gap, x, y = 148, 28, 30, 110
    for k, (lab, sub, role) in enumerate(names):
        c.box(x + k * (w + gap), y, w, 70, lab, sub, role=role)
        if k:
            c.arrow(x + k * (w + gap) - gap + 2, y + 35, x + k * (w + gap) - 2, y + 35, accent=True)
    c.text(440, 214, "loss: mean squared error on capped RUL · early stopping on validation engines", "small")
    _takeaway(c, 240, "About 7,500 weights: trains on a laptop CPU in well under a minute per run.")
    return ("cap1-cnn.svg",
            c.render("Five boxes left to right: input window, two 1D convolution layers, global "
                     "average pooling, and a linear head that outputs RUL."),
            "Capstone 1: the compact 1D-CNN used as the temporal model.")


# --------------------------------------------------------------------------
# Capstone 2 · ERCOT
# --------------------------------------------------------------------------

@diagram
def cap2_backtest():
    c = Canvas(880, 400)
    c.title("Rolling-origin backtest (expanding window)",
            "Retrain monthly; every day's forecast uses only data up to the previous midnight")
    x0, x1 = 150, 820
    # year ticks: 7 years over the axis, last year = test year
    years = ["Y−6", "Y−5", "Y−4", "Y−3", "Y−2", "Y−1", "Y"]
    step = (x1 - x0) / len(years)
    for k, lab in enumerate(years):
        c.text(x0 + k * step + step / 2, 92, lab, "small")
    rows = [("val fold 1", 5.00, "warn"), ("val fold 2", 5.25, "warn"),
            ("val fold 12", 5.92, "warn"), ("test month 1", 6.00, "bad"),
            ("test month 12", 6.92, "bad")]
    for r, (name, start, role) in enumerate(rows):
        y = 110 + r * 34
        c.text(x0 - 12, y + 18, name, "small", anchor="end")
        train_w = start * step
        c.box(x0, y, train_w, 24, role="accent", radius=5)
        c.box(x0 + train_w, y, step / 12 * 1.0 + 6, 24, role=role, radius=5)
    for k, (lab, role) in enumerate([("train: all days before the fold", "accent"),
                                     ("validation month", "warn"), ("test month (run once)", "bad")]):
        lx = 150 + k * 230
        c.box(lx, 288, 22, 16, role=role, radius=4)
        c.text(lx + 30, 301, lab, "small", anchor="start")
    _takeaway(c, 330, "Choose features and settings on the validation year; touch the test year once.")
    return ("cap2-backtest.svg",
            c.render("Rows of horizontal bars. Each row has a training bar that grows to the right and "
                     "a one-month evaluation block after it; the last rows fall in the held-out test year."),
            "Capstone 2: rolling-origin backtest with monthly retraining, validation year and test year.")


@diagram
def cap2_day_ahead():
    c = Canvas(880, 370)
    c.title("What is known at the forecast origin",
            "Day-ahead: issue at the end of day D, forecast every hour of day D+1")
    y = 120
    boxes = [("Day D−6", "same weekday as D+1", "accent", 40),
             ("Day D−1", "lag 2 days", "accent", 240),
             ("Day D", "lag 1 day", "accent", 420),
             ("Day D+1", "target: 23-25 hours", "warn", 660)]
    for lab, sub, role, x in boxes:
        c.box(x, y, 170, 64, lab, sub, role=role)
    c.text(225, y + 38, "…", "lbl")
    c.rule(625, 96, 625, 290)
    c.text(625, 88, "origin: midnight", "small t-accent")
    c.box(40, 230, 550, 50, "Allowed: lags of 1, 2 and 7 days", "and statistics of day D and the week before it",
          role="accent")
    c.box(660, 230, 190, 50, "Lag 1 hour", "not known at origin", role="bad")
    _takeaway(c, 310, "Every feature must come from day D or earlier; a test enforces it.")
    return ("cap2-day-ahead.svg",
            c.render("A row of days. Days D-6, D-1 and D feed the features; a vertical line marks "
                     "the origin at midnight; day D+1 is the target. A red box says a 1-hour lag is not known."),
            "Capstone 2: the day-ahead information set and which lags are allowed.")


@diagram
def cap2_dst():
    c = Canvas(880, 360)
    c.title("Daylight saving: 23, 24 or 25 hours in a day",
            "Store time in UTC; derive local date and hour only for calendar features")
    c.box(40, 90, 220, 56, "Spring-forward day", "23 hour-ending rows", role="warn")
    c.box(40, 160, 220, 56, "Ordinary day", "24 rows, HE 01:00-24:00", role="default")
    c.box(40, 230, 220, 56, "Fall-back day", "25 rows, one hour repeated", role="warn")
    c.box(340, 160, 220, 56, "Continuous UTC index", "local midnight + k hours", role="accent")
    c.box(640, 120, 200, 56, "Local date and hour", "peak window, weekday", role="default")
    c.box(640, 200, 200, 56, "Check row counts", "23 / 24 / 25 per day", role="good")
    for yy, ty in ((118, 172), (188, 188), (258, 204)):
        c.arrow(260, yy, 338, ty, accent=True)
    c.arrow(560, 180, 638, 148)
    c.arrow(560, 196, 638, 228)
    _takeaway(c, 300, "Never drop or sum the repeated hour: count rows per local day and assert.")
    return ("cap2-dst.svg",
            c.render("Three boxes for spring, ordinary and fall days feed a continuous UTC index, which "
                     "then gives local date and hour and a row-count check."),
            "Capstone 2: handling daylight-saving days with a UTC index and row-count checks.")
