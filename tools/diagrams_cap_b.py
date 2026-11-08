"""Capstone walkthrough figures (cap_b) - see tools/make_diagrams.py for conventions.

Capstone 3 (retail demand forecasting, prefix ``cap3-``) and Capstone 5 (content moderation
triage, prefix ``cap5-``).
"""

from __future__ import annotations

from svgkit import Canvas, diagram  # noqa: F401

W = 880


def _row_arrows(c: Canvas, xs: list[float], w: float, y: float, accent: bool = True) -> None:
    for a, b in zip(xs, xs[1:]):
        c.arrow(a + w + 2, y, b - 4, y, accent=accent)


def _takeaway(c: Canvas, y: float, text: str, role: str = "ghost", h: float = 40) -> None:
    c.box(30, y, W - 60, h, text, role=role)


# --------------------------------------------------------------------------- capstone 3


@diagram
def cap3_rolling_origin():
    c = Canvas(W, 400)
    c.title("Rolling-origin evaluation", "Weekly origins: features read weeks up to the origin, targets are the next four weeks.")
    c.box(30, 70, 430, 60, "Training weeks", "fit the model · choose scope and segments", role="accent")
    c.box(470, 70, 185, 60, "Validation", "12 weeks · choose methods", role="warn")
    c.box(665, 70, 185, 60, "Test", "12 weeks · scored once", role="good")

    c.text(30, 172, "One origin, zoomed in", "lbl", anchor="start")
    xs = [150 + i * 80 for i in range(8)]
    labels = ["t−3", "t−2", "t−1", "t", "t+1", "t+2", "t+3", "t+4"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        c.box(x, 190, 70, 44, lab, role="accent" if i < 4 else "good")
    c.path("M 465 180 L 465 290", dashed=True, arrow=False)
    c.text(465, 306, "origin: end of week t", "small t-accent")
    c.text(300, 260, "features: lags, means, price, returns", "small")
    c.text(620, 260, "targets: horizons h = 1 … 4", "small t-good")
    c.box(30, 190, 100, 44, "t+h−52", "last year", role="muted")

    _takeaway(c, 330, "Nothing after the origin is read; the model is retrained once per period, not at every origin.")
    return "cap3-rolling-origin.svg", c.render(
        "Timeline with training weeks, 12 validation weeks and 12 test weeks. Zoomed origin: weeks t-3 to t are "
        "features, weeks t+1 to t+4 are targets; the same week last year is an extra feature."
    ), "Capstone 3: rolling-origin evaluation and the features available at a forecast origin"


@diagram
def cap3_cleaning_funnel():
    c = Canvas(W, 380)
    c.title("From invoice lines to a weekly panel", "Each rule is SQL with a hand-made test; every removal is counted.")
    steps = [("Raw lines", "two Excel sheets", "default"),
             ("De-duplicate", "sheet overlap", "default"),
             ("Product codes", "no postage, fees", "default"),
             ("Valid sales", "qty and price > 0", "accent"),
             ("Quick cancels", "undone in 24 h", "warn"),
             ("Weekly panel", "zero weeks kept", "good")]
    xs = [30 + i * 140 for i in range(6)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 80, 124, 66, lbl, sub, role=role)
    _row_arrows(c, xs, 124, 113)

    c.box(310, 210, 124, 56, "Returns table", "C invoices", role="muted")
    c.arrow(372, 146, 372, 206)
    c.path("M 434 238 L 792 238 L 792 150")
    c.text(613, 230, "returns rate feature", "small")

    _takeaway(c, 300, "The label is gross units on valid sales; cancellations become a separate returns signal.")
    return "cap3-cleaning-funnel.svg", c.render(
        "Pipeline: raw lines, de-duplicate the sheet overlap, keep product codes, keep valid sales, drop sales "
        "cancelled within 24 hours, aggregate to a product by week panel with zeros. Cancellations feed a returns table."
    ), "Capstone 3: cleaning rules from raw invoice lines to the product-by-week panel"


@diagram
def cap3_demand_segments():
    c = Canvas(W, 420)
    c.title("Four demand segments", "Syntetos–Boylan classification, computed on the training window only.")
    x0, y0, w, h = 160, 70, 560, 250
    c.rule(x0, y0 + h, x0 + w, y0 + h)
    c.rule(x0, y0, x0, y0 + h)
    c.text(x0 + w / 2, y0 + h + 24, "ADI: average weeks between sales", "small")
    c.text(x0 - 12, y0 + 8, "CV²", "small", anchor="end")
    c.text(x0 - 12, y0 + 24, "size variation", "small", anchor="end")
    xc, yc = x0 + 240, y0 + 130
    c.path(f"M {xc} {y0} L {xc} {y0 + h}", dashed=True, arrow=False)
    c.path(f"M {x0} {yc} L {x0 + w} {yc}", dashed=True, arrow=False)
    c.text(xc, y0 - 8, "ADI = 1.32", "small t-accent")
    c.text(x0 - 12, yc + 4, "CV² = 0.49", "small t-accent", anchor="end")
    c.box(x0 + 20, yc + 20, 200, 90, "Smooth", "sells most weeks, steady size", role="good")
    c.box(x0 + 20, y0 + 10, 200, 100, "Erratic", "sells most weeks, size varies", role="warn")
    c.box(xc + 20, yc + 20, 280, 90, "Intermittent", "many zero weeks, steady size: Croston/SBA", role="accent")
    c.box(xc + 20, y0 + 10, 280, 100, "Lumpy", "many zeros and variable size: hardest", role="bad")
    _takeaway(c, 360, "The segment decides which simple method is hard to beat; choose methods per segment on validation.")
    return "cap3-demand-segments.svg", c.render(
        "ADI on the x axis and CV squared on the y axis, split at 1.32 and 0.49 into smooth, erratic, intermittent "
        "and lumpy demand."
    ), "Capstone 3: demand segments by ADI and CV²"


# --------------------------------------------------------------------------- capstone 5


@diagram
def cap5_triage_flow():
    c = Canvas(W, 380)
    c.title("Triage for human review", "The model ranks comments; capacity sets the threshold; people decide.")
    steps = [("Comment", "text only", "default"),
             ("Score", "TF-IDF + LR", "accent"),
             ("Calibrate", "on validation", "accent"),
             ("Threshold", "top q of traffic", "warn")]
    xs = [30 + i * 150 for i in range(4)]
    for x, (lbl, sub, role) in zip(xs, steps):
        c.box(x, 110, 130, 64, lbl, sub, role=role)
    _row_arrows(c, xs, 130, 142)
    c.box(660, 60, 190, 56, "Auto-approve", "published; audit sample", role="muted")
    c.box(660, 170, 190, 56, "Review queue", "a person decides", role="good")
    c.path("M 612 132 L 630 132 L 630 88 L 656 88")
    c.path("M 612 152 L 630 152 L 630 198 L 656 198", accent=True)
    c.text(624, 100, "below", "small", anchor="end")
    c.text(624, 190, "above", "small t-accent", anchor="end")
    c.path("M 755 226 L 755 262 L 95 262 L 95 178", dashed=True)
    c.text(425, 254, "reviewer decisions become new labels (only for queued comments)", "small")
    _takeaway(c, 300, "No automatic removal: the score only orders the queue, and the threshold is stored with the model.")
    return "cap5-triage-flow.svg", c.render(
        "Comment, score, calibration and a capacity threshold route each comment to a human review queue or to "
        "auto-approve; reviewer decisions flow back as labels."
    ), "Capstone 5: score, calibrate, threshold by capacity, human review"


@diagram
def cap5_capacity_threshold():
    c = Canvas(W, 420)
    c.title("Review capacity sets the operating point", "Shape of precision and recall as the review budget grows (illustrative).")
    x0, y0, w, h = 110, 70, 600, 240
    c.rule(x0, y0 + h, x0 + w, y0 + h)
    c.rule(x0, y0, x0, y0 + h)
    c.text(x0 + w / 2, y0 + h + 26, "review budget q: share of all comments sent to people", "small")
    c.text(x0 - 10, y0 + 6, "1.0", "small", anchor="end")
    c.text(x0 - 10, y0 + h, "0", "small", anchor="end")
    c.curve(f"M {x0} {y0 + 30} C {x0 + 150} {y0 + 50}, {x0 + 300} {y0 + 120}, {x0 + w} {y0 + 205}", tone="accent")
    c.curve(f"M {x0} {y0 + h} C {x0 + 60} {y0 + 120}, {x0 + 220} {y0 + 40}, {x0 + w} {y0 + 15}", tone="good")
    c.text(x0 + w + 8, y0 + 205, "precision", "small t-accent", anchor="start")
    c.text(x0 + w + 8, y0 + 20, "recall", "small t-good", anchor="start")
    for xq, lab in [(x0 + 40, "1%"), (x0 + 80, "2%"), (x0 + 200, "5%")]:
        c.path(f"M {xq} {y0} L {xq} {y0 + h}", dashed=True, arrow=False)
        c.text(xq, y0 + h + 12, lab, "small t-warn")
    c.text(x0 + 330, y0 + 200, "a larger queue finds more toxic comments", "small")
    c.text(x0 + 330, y0 + 216, "but a smaller share of it is toxic", "small")
    _takeaway(c, 350, "Budget q → threshold = the (1 − q) quantile of validation scores; report precision and recall there.")
    return "cap5-capacity-threshold.svg", c.render(
        "Illustrative curves: as the review budget grows, recall rises and precision falls; dashed lines mark 1, 2 "
        "and 5 percent budgets."
    ), "Capstone 5: review budget, threshold, precision and recall"


@diagram
def cap5_split_protocol():
    c = Canvas(W, 360)
    c.title("What each split is for", "Official WILDS splits; duplicates removed from train; identity labels for evaluation only.")
    c.box(30, 80, 250, 80, "Train", "de-duplicated against val/test · stratified subsample · fit", role="accent")
    c.box(315, 80, 250, 80, "Validation", "calibration · thresholds · model choice", role="warn")
    c.box(600, 80, 250, 80, "Test", "scored once · CIs · subgroups", role="good")
    c.arrow(282, 120, 311, 120, accent=True)
    c.arrow(567, 120, 596, 120, accent=True)
    c.box(30, 200, 380, 60, "Model inputs", "comment text only", role="default")
    c.box(470, 200, 380, 60, "Never inputs", "identity columns · other toxicity sub-labels", role="bad")
    c.arrow(155, 162, 155, 196)
    c.path("M 725 162 L 725 196", dashed=True)
    c.text(735, 184, "used to slice results", "small", anchor="start")
    _takeaway(c, 290, "A threshold or calibration fitted on test, or a duplicate across splits, makes every test number optimistic.")
    return "cap5-split-protocol.svg", c.render(
        "Train is de-duplicated and subsampled to fit; validation sets calibration, thresholds and model choice; test "
        "is scored once. Only text is a model input; identity columns are used only to slice evaluation."
    ), "Capstone 5: the role of each split and of the identity labels"
