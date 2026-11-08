"""Capstone walkthrough figures (cap_c) - see tools/make_diagrams.py for conventions."""

from __future__ import annotations

from svgkit import Canvas, diagram  # noqa: F401


# ---------------------------------------------------------------------------
# Capstone 4 - movie recommendations (MovieLens). Names start with "cap4-".
# ---------------------------------------------------------------------------

_W4 = 880


def _cap4_timeline(c, y, xs, roles, r=7):
    """One user's ratings on a time axis: a thin line with one circle per event."""
    c.rule(230, y, 850, y)
    for x, role in zip(xs, roles):
        c.circle(x, y, r, role=role)


@diagram
def cap4_time_split():
    c = Canvas(_W4, 430)
    c.title("Three ways to hold out a rating, and what each one leaks",
            "Two users, A and B, on the same time axis. Circles are ratings; the test circle is the held-out positive.")
    legend = [("accent", "used for training"), ("good", "test target"), ("bad", "future seen in training"),
              ("ghost", "not used")]
    for i, (role, text) in enumerate(legend):
        x = 250 + i * 160
        c.circle(x, 80, 7, role=role)
        c.text(x + 14, 84, text, "small", anchor="start")
    rows = [
        ("Random holdout", "leaks the user's own future", "bad",
         [260, 320, 380, 440, 500, 560, 620, 680], ["accent"] * 3 + ["good"] + ["bad"] * 4,
         [300, 400, 480, 560, 660, 760], ["accent"] * 6),
        ("Leave-last-out", "leaks other users' future", "warn",
         [260, 320, 380, 440, 500, 560, 620, 700], ["accent"] * 6 + ["good", "ghost"],
         [300, 400, 480, 560, 660, 760], ["accent"] * 4 + ["bad"] * 2),
        ("Global split at T", "no event after T is used", "good",
         [260, 320, 380, 440, 500, 640, 700, 760], ["accent"] * 5 + ["good", "ghost", "ghost"],
         [300, 400, 480, 560, 660, 760], ["accent"] * 4 + ["ghost"] * 2),
    ]
    for k, (name, sub, role, xa, ra, xb, rb) in enumerate(rows):
        y = 130 + k * 80
        c.box(24, y - 4, 180, 58, name, sub, role=role)
        c.text(222, y + 13, "A", "small", anchor="end")
        c.text(222, y + 45, "B", "small", anchor="end")
        _cap4_timeline(c, y + 9, xa, ra)
        _cap4_timeline(c, y + 41, xb, rb)
    c.raw('<line x1="600" y1="282" x2="600" y2="362" stroke="#9a6700" stroke-width="1.6" stroke-dasharray="5 4"/>')
    c.text(606, 294, "T", "small t-warn", anchor="start")
    c.text(850, 370, "time →", "small", anchor="end")
    c.box(24, 380, _W4 - 48, 40,
          "Primary protocol: leave-last-out with time filters; robustness check: the global split.", role="default")
    return "cap4-time-split.svg", c.render(
        "Three splits on a time axis for two users. Random holdout: user A's held-out rating has later "
        "ratings of A in training. Leave-last-out: A's last positive is the test target, but user B's "
        "ratings after A's test time are in training. Global split at time T: everything before T is "
        "training, A's first positive after T is the target, nothing after T is used."
    ), "Capstone 4: random holdout, leave-last-out and a global temporal split, and what each leaks."


@diagram
def cap4_implicit_models():
    c = Canvas(_W4, 400)
    c.title("Three collaborative models on one binary matrix",
            "Same input, same masking, same full-ranking evaluation; only the scoring rule differs.")
    c.box(24, 160, 180, 92, "Binary matrix X", "users × movies; 1 if rated 4 or more before the request",
          role="accent")
    models = [("Item-kNN", "cosine of item columns; score = x·S"),
              ("EASE", "closed form: one items × items inverse"),
              ("Implicit ALS", "user and item vectors, alternating solves")]
    for k, (name, sub) in enumerate(models):
        y = 80 + k * 90
        c.box(270, y, 240, 70, name, sub, role="default")
        c.arrow(206, 206, 266, y + 35, accent=True)
        c.arrow(512, y + 35, 562, 190 + k * 16, accent=True)
    c.box(566, 166, 110, 80, "Scores", "every movie", role="default")
    c.box(700, 160, 156, 92, "Mask, then top 10", "not yet released; already rated", role="good")
    c.arrow(678, 206, 696, 206, accent=True)
    c.box(24, 340, _W4 - 48, 44,
          "A model only changes the scores: candidate filters are shared, so the comparison is fair.",
          role="default")
    return "cap4-implicit-models.svg", c.render(
        "The binary user by movie matrix feeds three models: item-kNN (cosine similarity of item "
        "columns), EASE (closed-form item-to-item weights) and implicit ALS (user and item vectors). "
        "Each produces a score for every movie; movies that did not exist yet or were already rated "
        "are masked before taking the top 10."
    ), "Capstone 4: item-kNN, EASE and implicit ALS share one input matrix and one masking step."


@diagram
def cap4_serving():
    c = Canvas(_W4, 400)
    c.title("Serving: precompute nightly, merge and filter per request",
            "The request path never fits a model; it reads the bundle and always reports its strategy.")
    c.text(24, 84, "Nightly batch (precomputed)", "small", anchor="start")
    c.box(24, 94, 150, 64, "All ratings", "up to last night", role="default")
    c.box(194, 94, 150, 64, "Fit and precompute", "selected model", role="accent")
    c.box(364, 94, 492, 64, "Model bundle",
          "top-200 per user · item neighbours · recent popularity · model version", role="accent")
    c.arrow(176, 126, 190, 126, accent=True)
    c.arrow(346, 126, 360, 126, accent=True)
    c.text(24, 220, "Per request (online)", "small", anchor="start")
    steps = [("Request", "user, k, recent, exclude", "default"),
             ("Validate", "k 1-100, no duplicates", "warn"),
             ("Base list", "model top-200, or popularity if cold", "accent"),
             ("Merge and filter", "recent items (RRF); drop rated, excluded", "accent"),
             ("Response", "top k, strategy, model version", "good")]
    for k, (name, sub, role) in enumerate(steps):
        x = 24 + k * 170
        c.box(x, 230, 150, 76, name, sub, role=role)
        if k:
            c.arrow(x - 18, 268, x - 4, 268, accent=True)
    for x in (437, 607):
        c.arrow(x, 160, x, 226)
        c.text(x + 8, 198, "read", "small", anchor="start")
    c.box(24, 336, _W4 - 48, 44,
          "Freshness costs latency: today's ratings reach the list tonight, unless the request carries them.",
          role="default")
    return "cap4-serving.svg", c.render(
        "Serving design. Nightly batch: all ratings are used to fit the selected model and write a bundle "
        "with the top 200 candidates per user, item neighbours, recent popularity and the model version. "
        "Per request: validate, choose the model's list or the popularity fallback for cold users, merge "
        "recent items with reciprocal-rank fusion, drop rated and excluded movies, and return the top k "
        "with the strategy and model version."
    ), "Capstone 4: precomputed candidates and a short online path with a popularity fallback."
