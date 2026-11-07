"""Diagrams for Level 2, notebooks 11-20 - see tools/make_diagrams.py for the conventions.

Every name starts with ``l2-x-`` so it cannot collide with notebooks 01-10.
Grouped by notebook: experimentation (11-12), ML from scratch (13-14), coding
patterns (15-16), serving and optimisation (17-18), design and evaluation (19-20).
"""

from __future__ import annotations

import math
import random

from svgkit import LIGHT, Canvas, diagram

W = 880


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _row(c, x0, y, w, h, gap, items, accent=True):
    """A left-to-right row of (label, sub, role) boxes joined by arrows."""
    for i, (label, sub, role) in enumerate(items):
        x = x0 + i * (w + gap)
        c.box(x, y, w, h, label, sub, role=role)
        if i:
            c.arrow(x - gap + 2, y + h / 2, x - 3, y + h / 2, accent=accent)


def _takeaway(c, y, text, h=44):
    c.box(24, y, W - 48, h, text, role="default")


def _poly(c, points, tone, opacity):
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z"
    c.raw(f'<path d="{d}" fill="{LIGHT[tone]}" fill-opacity="{opacity}" stroke="none"/>')


def _line(points):
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def _dash(c, x1, y1, x2, y2, tone="muted_fg"):
    c.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
          f'stroke="{LIGHT[tone]}" stroke-width="1.3" stroke-dasharray="4 4"/>')


def _swatch(c, x, y, tone, opacity=0.35):
    c.raw(f'<rect x="{x}" y="{y}" width="14" height="14" rx="3" fill="{LIGHT[tone]}" '
          f'fill-opacity="{opacity}" stroke="{LIGHT[tone]}" stroke-width="1"/>')


def _node(c, x, y, label, role="accent", r=18):
    c.circle(x, y, r, role=role)
    c.text(x, y + 5, label, "lbl")


# ---------------------------------------------------------------------------
# 11 - A/B testing fundamentals
# ---------------------------------------------------------------------------


@diagram
def ab_lifecycle():
    c = Canvas(W, 330)
    c.title("The life of an A/B test",
            "Six steps. Steps 1-4 happen before any result is visible, which is what keeps it honest.")
    items = [
        ("Hypothesis", "one primary metric, written down", "default"),
        ("Power", "n per arm from pre-period σ", "default"),
        ("Randomise", "by user: hash(user, salt)", "accent"),
        ("SRM check", "χ² test on arm counts", "warn"),
        ("Analyse", "at the unit, once, at the end", "accent"),
        ("Decide", "primary up, guardrails hold", "good"),
    ]
    _row(c, 24, 92, 120, 76, 24, items)
    skipped = [("20 metrics read", "→ false wins"), ("too few users", "→ noise, no answer"),
               ("wrong unit", "→ SE far too small"), ("broken logging", "→ biased arms"),
               ("daily peeking", "→ FPR 0.05 → 0.2"), ("metric shopping", "→ ship noise")]
    c.text(24, 196, "If you skip it:", "small t-bad", anchor="start")
    for i, (a, b) in enumerate(skipped):
        x = 24 + i * 144 + 60
        c.text(x, 218, a, "small t-bad")
        c.text(x, 234, b, "small t-bad")
    _takeaway(c, 262, "The statistics are the easy part: most bad launch decisions come from a skipped step, not a wrong formula.")
    return "l2-x-ab-test-lifecycle.svg", c.render(
        "An A/B test in six steps: hypothesis with one primary metric, power calculation, "
        "randomisation by user, sample-ratio-mismatch check, analysis at the randomisation "
        "unit, and the decision. Under each step, what goes wrong if it is skipped."
    ), "The six steps of an A/B test, and what goes wrong when each is skipped."


@diagram
def randomisation_unit():
    c = Canvas(W, 380)
    c.title("Choosing the randomisation unit",
            "Smaller units give less variance; bigger units protect against interference.")
    cols = [
        ("Request", "each page view flips a coin", "default",
         ("Variance: lowest", "a user's own CTR cancels out", "good"),
         ("Risk: carry-over", "B teaches habits that leak into A; no user-level metrics", "bad")),
        ("User", "every request of a user, one arm", "accent",
         ("Variance: higher", "analyse per user, not per impression", "warn"),
         ("Risk: low", "the default for rankers and feeds", "good")),
        ("Cluster", "city, seller, time slot", "default",
         ("Variance: highest", "few units, wide intervals", "bad"),
         ("Risk: handled", "marketplace spillover stays inside a cluster", "good")),
    ]
    for i, (head, sub, role, var, risk) in enumerate(cols):
        x = 24 + i * 284
        c.box(x, 84, 264, 56, head, sub, role=role)
        c.box(x, 160, 264, 62, var[0], var[1], role=var[2])
        c.box(x, 236, 264, 62, risk[0], risk[1], role=risk[2])
    _takeaway(c, 318, "Randomise at the unit whose behaviour the treatment changes, and compute the variance at that same unit.")
    return "l2-x-randomization-unit.svg", c.render(
        "Three randomisation units compared. Request: lowest variance but carry-over between "
        "arms. User: higher variance, analysed per user, the default. Cluster: highest "
        "variance but contains marketplace spillover."
    ), "Request, user or cluster randomisation: variance against interference."


@diagram
def power_curves():
    c = Canvas(W, 400)
    c.title("Power: two distributions of the test statistic",
            "Under H₀ the statistic centres on 0; under H₁ it centres on the true effect δ.")
    y0, sd, peak = 300, 70, 165
    m0, m1 = 250, 480
    crit = m0 + 1.96 * sd

    def f(x, m):
        return y0 - peak * math.exp(-0.5 * ((x - m) / sd) ** 2)

    def area(m, a, b):
        xs = [a + (b - a) * i / 60 for i in range(61)]
        return [(a, y0)] + [(x, f(x, m)) for x in xs] + [(b, y0)]

    _poly(c, area(m1, m1 - 4 * sd, crit), "bad", 0.18)
    _poly(c, area(m1, crit, m1 + 4 * sd), "good", 0.18)
    _poly(c, area(m0, crit, m0 + 4 * sd), "warn", 0.55)
    xs = [40 + i * 2 for i in range(321)]
    c.curve(_line([(x, f(x, m0)) for x in xs if x <= m0 + 4.2 * sd]), tone="muted_fg")
    c.curve(_line([(x, f(x, m1)) for x in xs if x >= m1 - 4.2 * sd]), tone="accent")
    c.rule(40, y0, 700, y0)
    _dash(c, crit, y0, crit, 104, tone="fg")
    c.text(crit, 96, "reject H₀ beyond z = 1.96", "small")
    c.text(m0, y0 - peak - 10, "H₀: no effect", "lbl")
    c.text(m1, y0 - peak - 10, "H₁: true effect δ", "lbl t-accent")
    c.arrow(m0, y0 + 22, m1, y0 + 22)
    c.text((m0 + m1) / 2, y0 + 40, "δ = the minimum detectable effect, in standard errors", "small")
    lx = 690
    for i, (tone, op, text) in enumerate([("warn", 0.55, "α = 0.05: false alarm"),
                                          ("bad", 0.18, "β = 0.2: missed effect"),
                                          ("good", 0.18, "1 − β = 0.8: power")]):
        _swatch(c, lx, 132 + i * 30, tone, op)
        c.text(lx + 22, 144 + i * 30, text, "small", anchor="start")
    _takeaway(c, 346, "n per arm ≈ 2σ²(z₁₋α/₂ + z₁₋β)² / δ²: halve δ and you need four times the users.")
    return "l2-x-power.svg", c.render(
        "Two bell curves of the test statistic: under the null hypothesis centred at zero, "
        "under the alternative centred at the true effect delta. The critical value at 1.96 "
        "cuts off alpha in the null's right tail; the alternative's mass left of the line is "
        "beta, the rest is power."
    ), "Power, α, β and the MDE as two overlapping distributions."


@diagram
def peeking():
    c = Canvas(W, 440)
    c.title("Peeking: every look is another chance for noise",
            "One A/A test (no true effect), z-statistic recomputed every day for 14 days.")
    x0, x1, yc, s = 80, 600, 240, 38

    def X(day):
        return x0 + (day - 1) * (x1 - x0) / 13

    def Y(z):
        return yc - z * s

    c.rule(x0, Y(-2.6), x1, Y(-2.6))
    _dash(c, x0, Y(0), x1, Y(0))
    for z in (1.96, -1.96):
        c.raw(f'<line x1="{x0}" y1="{Y(z):.1f}" x2="{x1}" y2="{Y(z):.1f}" stroke="{LIGHT["warn"]}" '
              f'stroke-width="1.5" stroke-dasharray="6 4"/>')
        c.text(x0 - 8, Y(z) + 4, f"{z:+.2f}", "small", anchor="end")
    c.text(x0 - 8, Y(0) + 4, "0", "small", anchor="end")
    obf = [(X(d / 4), Y(2.0 * math.sqrt(14 / (d / 4)))) for d in range(14, 57)]
    c.curve(_line(obf), tone="good", dashed=True)
    c.text(X(7.5), 122, "sequential boundary (O'Brien-Fleming)", "small t-good", anchor="start")
    walk = [0.3, 0.9, 1.4, 1.7, 2.15, 1.8, 1.3, 1.0, 0.6, 0.8, 0.5, 0.2, 0.4, 0.3]
    c.curve(_line([(X(i + 1), Y(z)) for i, z in enumerate(walk)]), tone="accent")
    c.circle(X(5), Y(2.15), 6, role="bad")
    c.text(88, 140, "day 5: p < 0.05, stop and 'ship'", "small t-bad", anchor="start")
    c.circle(X(14), Y(0.3), 6, role="good")
    c.text(X(14), Y(0.3) + 26, "day 14: p = 0.76", "small t-good", anchor="end")
    for d in (1, 7, 14):
        c.text(X(d), Y(-2.6) + 18, f"day {d}", "small")
    c.text(650, 110, "A/A false-positive rate", "lbl", anchor="start")
    rows = [("look once, at the end", "0.05", "good"), ("peek daily, stop at p < 0.05", "≈ 0.20", "bad"),
            ("daily, Bonferroni α/14", "≈ 0.02", "warn"), ("daily, sequential boundary", "0.05", "good")]
    for i, (a, b, tone) in enumerate(rows):
        c.text(650, 140 + i * 40, a, "small", anchor="start")
        c.text(650, 157 + i * 40, b, f"lbl t-{tone}", anchor="start")
    _takeaway(c, 380, "Fix the duration in advance and read the primary metric once, or use a test designed for peeking.")
    return "l2-x-peeking.svg", c.render(
        "A random walk of the z-statistic of an A/A test over 14 days crosses the 1.96 line on "
        "day 5 and ends near zero on day 14. A sequential boundary starts high and narrows. "
        "False-positive rates: 0.05 for one look, about 0.20 for daily peeking, 0.02 with "
        "Bonferroni, 0.05 with a sequential boundary."
    ), "Why daily peeking inflates false positives, and the sequential-boundary fix."


# ---------------------------------------------------------------------------
# 12 - advanced experimentation
# ---------------------------------------------------------------------------


@diagram
def cuped():
    c = Canvas(W, 400)
    c.title("CUPED: subtract what the pre-period already predicts",
            "Heavy users stay heavy. Remove that predictable part and the noise shrinks.")
    rnd = random.Random(3)
    px0, py0 = 60, 300
    data = []
    for _ in range(46):
        x = rnd.uniform(0.05, 1.0)
        y = min(max(0.1 + 0.75 * x + rnd.gauss(0, 0.12), 0.03), 1.0)
        data.append((x, y))
    pts = [(px0 + x * 300, py0 - y * 200) for x, y in data]
    c.rule(px0, py0, px0 + 320, py0)
    c.rule(px0, py0, px0, 86)
    for x, y in pts:
        c.raw(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{LIGHT["muted_fg"]}" fill-opacity="0.55"/>')
    c.curve(_line([(px0, py0 - 0.1 * 200), (px0 + 320, py0 - (0.1 + 0.75 * 320 / 300) * 200)]), tone="accent")
    k = max(range(len(data)), key=lambda i: data[i][1] - (0.1 + 0.75 * data[i][0]))
    hx, hy = pts[k]
    fy = py0 - (0.1 + 0.75 * data[k][0]) * 200
    c.raw(f'<line x1="{hx:.1f}" y1="{hy:.1f}" x2="{hx:.1f}" y2="{fy:.1f}" stroke="{LIGHT["bad"]}" stroke-width="2.2"/>')
    c.text(px0 + 160, py0 + 22, "pre-period clicks x (before assignment)", "small")
    c.text(px0 + 4, 80, "experiment clicks y", "small", anchor="start")
    c.text(px0 + 320, 266, "slope θ = cov(x, y) / var(x)", "small t-accent", anchor="end")
    c.text(px0 + 320, 284, "red: the residual CUPED keeps", "small t-bad", anchor="end")

    bx, base, sd = 660, 290, 70

    def bell(sd_, h):
        xs = [bx - 3.4 * sd_ + i * 6.8 * sd_ / 80 for i in range(81)]
        return [(x, base - h * math.exp(-0.5 * ((x - bx) / sd_) ** 2)) for x in xs]

    c.curve(_line(bell(sd, 95)), tone="muted_fg")
    c.curve(_line(bell(sd * 0.71, 134)), tone="accent")
    c.rule(bx - 220, base, bx + 200, base)
    c.text(bx, 136, "y − θ(x − x̄): spread × √(1 − ρ²)", "small t-accent")
    c.text(bx + 140, 236, "raw y", "small")
    c.text(bx, base + 22, "same centre (x is independent of the arm)", "small")
    _takeaway(c, 334, "With ρ = corr(x, y) = 0.71 the variance halves (1 − ρ² ≈ 0.5): the same power with half the users.")
    return "l2-x-cuped.svg", c.render(
        "Left: a scatter of experiment clicks against pre-period clicks with a regression line "
        "of slope theta; the vertical residual is what CUPED keeps. Right: the distribution of "
        "the raw metric and the narrower CUPED-adjusted one, with the same centre."
    ), "CUPED as regression on a pre-period covariate: same mean, smaller variance."


@diagram
def team_draft():
    c = Canvas(W, 400)
    c.title("Team-draft interleaving",
            "Like picking playground teams: each round a coin decides who picks first.")
    a = [("a1", "accent"), ("s1", "ghost"), ("a2", "accent"), ("s2", "accent"), ("a3", "ghost")]
    b = [("s1", "warn"), ("b1", "warn"), ("s2", "ghost"), ("b2", "ghost"), ("a1", "ghost")]
    merged = [("a1", "picked by A", "accent", "round 1 · A first"),
              ("s1", "picked by B", "warn", ""),
              ("b1", "picked by B", "warn", "round 2 · B first"),
              ("a2", "picked by A", "accent", ""),
              ("s2", "picked by A", "accent", "round 3 · A first")]
    c.text(74, 88, "Ranker A", "lbl")
    c.text(186, 88, "Ranker B", "lbl")
    c.text(480, 88, "Shown to the user", "lbl")
    c.text(660, 88, "Clicks", "lbl")
    clicks = {3: True, 4: True}
    for i in range(5):
        y = 100 + i * 46
        c.box(24, y, 100, 38, a[i][0], role=a[i][1])
        c.box(136, y, 100, 38, b[i][0], role=b[i][1])
        lab, sub, role, rnd_ = merged[i]
        c.box(400, y, 160, 38, lab, sub, role=role)
        if rnd_:
            c.text(390, y + 24, rnd_, "small", anchor="end")
        if clicks.get(i):
            c.text(660, y + 24, "click → team A", "small t-accent")
        else:
            c.text(660, y + 24, "no click", "small")
    c.box(742, 170, 114, 76, "A 2 : B 0", "A wins this request", role="accent")
    _takeaway(c, 340, "One user judges both rankers on the same request, so between-user variance cancels: ~20× fewer users here.")
    return "l2-x-team-draft.svg", c.render(
        "Two rankings, A and B, are merged by team draft: in each round a coin decides which "
        "ranker picks first and each picks its best item not yet shown. Clicks are credited to "
        "the team that picked the item; here A gets two clicks and B none."
    ), "Team-draft interleaving: merge two rankings, credit each click to a team."


@diagram
def offline_online():
    c = Canvas(W, 370)
    c.title("The offline-online gap, in one flip",
            "Validation-week requests, re-ranked four ways (notebook 12.5). Ranks, best on top.")
    left = [("logging policy · 0.76", "bad"), ("true relevance · 0.68", "good"),
            ("item popularity", "default"), ("random", "ghost")]
    right = [("true relevance · 0.18", "good"), ("item popularity · 0.15", "default"),
             ("logging policy · 0.14", "bad"), ("random", "ghost")]
    c.text(170, 94, "Offline: NDCG@10 on logged clicks", "lbl")
    c.text(710, 94, "Online: CTR, true click model", "lbl")
    for i in range(4):
        y = 110 + i * 50
        c.box(40, y, 260, 36, f"{i + 1}. {left[i][0]}", role=left[i][1])
        c.box(580, y, 260, 36, f"{i + 1}. {right[i][0]}", role=right[i][1])
    links = [(0, 2, "bad"), (1, 0, "good"), (2, 1, "muted_fg"), (3, 3, "stroke")]
    for li, ri, tone in links:
        c.curve(_line([(304, 128 + li * 50), (576, 128 + ri * 50)]), tone=tone, width=2.4)
    _takeaway(c, 316, "Logged clicks reward agreeing with the old policy. Only an online test measures the decision itself.")
    return "l2-x-offline-online-gap.svg", c.render(
        "A slope chart: offline NDCG on logged clicks ranks the logging policy first and true "
        "relevance second; online CTR under the true click model ranks true relevance first, "
        "popularity second and the logging policy third."
    ), "Offline NDCG on logged clicks and online CTR rank the same rankers in different orders."


# ---------------------------------------------------------------------------
# 13 - ML from scratch I
# ---------------------------------------------------------------------------


@diagram
def logreg_gd():
    c = Canvas(W, 390)
    c.title("Logistic regression: one gradient, one bowl",
            "The loss is convex, so gradient descent reaches the single minimum from any start.")
    cx, cy = 230, 196
    for rx, ry in [(176, 90), (120, 62), (66, 35), (24, 13)]:
        c.raw(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" transform="rotate(-22 {cx} {cy})" '
              f'fill="none" stroke="{LIGHT["stroke"]}" stroke-width="1.6"/>')
    path = [(84, 140), (154, 262), (178, 166), (222, 230), (214, 182), (236, 206), (230, 196)]
    c.curve(_line(path), tone="accent")
    for x, y in path[:-1]:
        c.raw(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{LIGHT["accent"]}"/>')
    c.circle(cx, cy, 6, role="good")
    c.text(84, 128, "w = 0", "small", anchor="middle")
    c.text(cx, 314, "minimum: unique (convex loss)", "small t-good")
    c.text(cx, 332, "zig-zag: features on different scales → standardise", "small")
    items = [("z = Xw + b", "logits, shape (n,)", "default"),
             ("p = σ(z)", "predicted probabilities", "default"),
             ("∂L/∂z = (p − y) / n", "prediction minus target", "accent"),
             ("∇w = Xᵀ(p − y)/n + λw", "then w ← w − lr · ∇w", "good")]
    for i, (lab, sub, role) in enumerate(items):
        y = 84 + i * 64
        c.box(500, y, 330, 48, lab, sub, role=role)
        if i:
            c.arrow(665, y - 15, 665, y - 3, accent=True)
    _takeaway(c, 362 - 6, "The same “prediction minus target” gradient appears for softmax and for squared loss.", h=28)
    return "l2-x-logreg-gd.svg", c.render(
        "Left: elliptical contours of the logistic loss with gradient-descent steps zig-zagging "
        "to the unique minimum. Right: the four lines of the model - logits, sigmoid, the "
        "gradient p minus y over n, and the weight gradient with L2."
    ), "Logistic regression: convex loss contours and the p − y gradient."


@diagram
def kmeans_steps():
    c = Canvas(W, 400)
    c.title("k-means: assign, move, repeat",
            "Lloyd's algorithm. Both steps can only lower the objective, so it stops, at a local minimum.")
    rnd = random.Random(7)
    blobs = [(70, 70, "accent"), (190, 80, "good"), (130, 160, "warn")]
    pts = []
    for bx, by, role in blobs:
        for _ in range(8):
            pts.append((bx + rnd.gauss(0, 18), by + rnd.gauss(0, 16), role))
    init = [(pts[2][0], pts[2][1]), (pts[9][0], pts[9][1]), (pts[20][0], pts[20][1])]
    panels = [("1 · Initialise (k-means++)", 24), ("2 · Assign to nearest centre", 312),
              ("3 · Move centre to the mean", 600)]
    for title, x in panels:
        c.box(x, 96, 256, 200, role="ghost", radius=8)
        c.text(x + 128, 86, title, "lbl")
    for px, py, role in pts:
        c.circle(24 + 8 + px, 96 + 10 + py, 4, role="default")
        c.circle(312 + 8 + px, 96 + 10 + py, 4, role=role)
        c.circle(600 + 8 + px, 96 + 10 + py, 4, role=role)
    for (ix, iy), (_, _, role) in zip(init, blobs):
        c.circle(24 + 8 + ix, 96 + 10 + iy, 8, role=role)
        c.circle(312 + 8 + ix, 96 + 10 + iy, 8, role=role)
        mx = sum(p[0] for p in pts if p[2] == role) / 8
        my = sum(p[1] for p in pts if p[2] == role) / 8
        c.circle(600 + 8 + ix, 96 + 10 + iy, 8, role="ghost")
        c.arrow(600 + 8 + ix, 96 + 10 + iy, 600 + 8 + mx, 96 + 10 + my)
        c.circle(600 + 8 + mx, 96 + 10 + my, 8, role=role)
    c.arrow(282, 196, 308, 196, accent=True)
    c.arrow(570, 196, 596, 196, accent=True)
    c.path("M 728 296 L 728 318 L 440 318 L 440 300", accent=True)
    c.text(584, 312, "repeat until the labels stop changing", "small")
    _takeaway(c, 334, "J = Σ ‖xᵢ − μ_c(i)‖². k-means++ spreads the first centres (P ∝ distance²); keep the best of several runs.")
    return "l2-x-kmeans.svg", c.render(
        "Three panels: k-means++ picks three spread-out initial centres; each point is assigned "
        "to its nearest centre; each centre moves to the mean of its points. Assign and move "
        "repeat until the labels stop changing."
    ), "k-means++ initialisation, then Lloyd's assign and update steps."


# ---------------------------------------------------------------------------
# 14 - ML from scratch II
# ---------------------------------------------------------------------------


def _sse(v):
    if not v:
        return 0.0
    m = sum(v) / len(v)
    return sum((t - m) ** 2 for t in v)


@diagram
def tree_split():
    c = Canvas(W, 464)
    c.title("Finding the best split: try every gap, keep the lowest SSE",
            "A regression stump on one feature. Sort once; prefix sums give every split in O(1).")
    ys = [1.0, 1.2, 0.8, 1.1, 3.0, 3.2, 2.9, 3.1]

    def X(i):
        return 110 + i * 70

    def Y(v):
        return 240 - v * 40

    c.rule(80, 240, 640, 240)
    c.text(360, 258, "feature x, sorted", "small")
    for i in range(7):
        gx = (X(i) + X(i + 1)) / 2
        if i == 3:
            c.raw(f'<line x1="{gx}" y1="92" x2="{gx}" y2="240" stroke="{LIGHT["accent"]}" stroke-width="2.2"/>')
        else:
            _dash(c, gx, 110, gx, 240)
    c.curve(_line([(X(0) - 20, Y(1.025)), (X(3) + 20, Y(1.025))]), tone="good", width=2)
    c.curve(_line([(X(4) - 20, Y(3.05)), (X(7) + 20, Y(3.05))]), tone="good", width=2)
    for i, v in enumerate(ys):
        c.circle(X(i), Y(v), 5, role="accent")
    c.text(X(3) + 35, 86, "best threshold", "small t-accent")
    c.text(X(1) + 35, Y(1.025) + 26, "left mean", "small t-good")
    c.text(X(5) + 35, Y(3.05) - 14, "right mean", "small t-good")
    losses = [_sse(ys[:i + 1]) + _sse(ys[i + 1:]) for i in range(7)]
    top = max(losses)
    pts = [((X(i) + X(i + 1)) / 2, 360 - 64 * v / top) for i, v in enumerate(losses)]
    c.rule(80, 372, 640, 372)
    c.curve(_line(pts), tone="muted_fg")
    for i, (x, y) in enumerate(pts):
        c.circle(x, y, 4.5, role="good" if i == 3 else "default")
    c.text(84, 284, "SSE(left) + SSE(right) at each gap", "small", anchor="start")
    c.text(pts[3][0], 388, f"minimum: {losses[3]:.2f}", "small t-good")
    c.box(668, 100, 188, 66, "SSE = S₂ − S₁²/m", "S₁ = Σy, S₂ = Σy², m = size", role="accent")
    c.box(668, 180, 188, 66, "Prefix sums", "cumsum(y), cumsum(y²) give every left child", role="default")
    c.box(668, 260, 188, 66, "Only between distinct x", "equal values must stay together", role="warn")
    _takeaway(c, 404, "Gain = SSE(parent) − SSE(left) − SSE(right), the variance reduction. Sort in O(n log n), all splits in O(n).")
    return "l2-x-tree-split.svg", c.render(
        "Eight points sorted by a feature, with dashed candidate thresholds between each pair; "
        "the best threshold separates a low group from a high group. Below, the summed SSE of "
        "the two children at each candidate, lowest at the best threshold."
    ), "A regression-tree split chosen by the lowest summed SSE, computed with prefix sums."


@diagram
def gradient_boosting():
    c = Canvas(W, 350)
    c.title("Gradient boosting: each tree fits what is still wrong",
            "Squared loss: the negative gradient at the current prediction is the residual.")
    items = [("F₀ = mean(y)", "the best constant", "default"),
             ("r₁ = y − F₀", "residuals = −gradient", "warn"),
             ("fit stump h₁", "to r₁ with best_split", "default"),
             ("F₁ = F₀ + ν·h₁", "ν = lr shrinks the step", "accent"),
             ("r₂ = y − F₁", "smaller residuals", "warn"),
             ("… F_M", "M rounds; stop on validation", "good")]
    _row(c, 24, 92, 120, 72, 22, items)
    pts = [(60 + i * 26, 196 + 64 * (1 - math.exp(-i / 6))) for i in range(30)]
    c.rule(56, 272, 840, 272)
    c.curve(_line(pts), tone="accent")
    c.text(840, 254, "training MSE never goes up", "small t-accent", anchor="end")
    c.text(66, 196, "loss", "small", anchor="start")
    _takeaway(c, 290, "Other losses change only the pseudo-residuals and leaf values (XGBoost, LightGBM: a Newton step Σg / Σh).", h=40)
    return "l2-x-gradient-boosting.svg", c.render(
        "Gradient boosting as a sequence: start from the mean, compute residuals, fit a stump to "
        "them, add it with a shrinkage factor, recompute residuals, and repeat for M rounds. "
        "A curve shows the training loss falling with each round."
    ), "Gradient boosting as sequential residual fitting: F_m = F_(m−1) + ν·h_m."


@diagram
def mlp_backprop():
    c = Canvas(W, 384)
    c.title("Backprop through a two-layer MLP",
            "Forward: keep every intermediate. Backward: the same graph in reverse, one local rule per box.")
    fwd = [("X", "(n, d)", "default"), ("z₁ = XW₁ + b₁", "(n, h)", "accent"),
           ("h = relu(z₁)", "(n, h)", "accent"), ("z₂ = hW₂ + b₂", "logits (n, C)", "accent"),
           ("L = CE(P, y)", "P = softmax(z₂); mean", "good")]
    bwd = [("dW₁ = Xᵀ dz₁", "db₁ = Σ dz₁ · shape of W₁", "warn"),
           ("dz₁ = dh·[z₁>0]", "ReLU passes or blocks", "warn"),
           ("dh = dz₂ W₂ᵀ", "(n,C)(C,h) → (n,h)", "warn"),
           ("dz₂ = (P − Y) / n", "dW₂ = hᵀ dz₂, db₂ = Σ dz₂", "warn"),
           ("dL = 1", "start here", "ghost")]
    c.text(24, 84, "forward →", "small t-accent", anchor="start")
    c.text(24, 220, "← backward (chain rule)", "small t-warn", anchor="start")
    for i in range(5):
        x = 22 + i * 170
        c.box(x, 92, 156, 56, fwd[i][0], fwd[i][1], role=fwd[i][2])
        c.box(x, 228, 156, 74, bwd[i][0], bwd[i][1], role=bwd[i][2])
        _dash(c, x + 78, 150, x + 78, 200)
        if i:
            c.arrow(x - 13, 120, x - 3, 120, accent=True)
            c.arrow(x - 2, 265, x - 12, 265)
    _takeaway(c, 320, "Every gradient has the shape of its variable: dW₁ is d × h like W₁, so the matrix product is forced.")
    return "l2-x-mlp-backprop.svg", c.render(
        "Top row, the forward pass of a two-layer MLP: X, linear, ReLU, linear, softmax "
        "cross-entropy. Bottom row, the backward pass in reverse: dz2 = (P - Y)/n, dh = dz2 W2 "
        "transposed, dz1 = dh masked by z1 > 0, dW1 = X transposed dz1."
    ), "Forward and backward passes of a two-layer MLP with each local gradient."


@diagram
def attention_lookup():
    c = Canvas(W, 420)
    c.title("Attention is a soft dictionary lookup",
            "One query scores every key; softmax turns scores into weights; the output mixes the values.")
    c.box(24, 170, 120, 70, "query q", "what this position looks for", role="accent")
    rows = [("key k₁", "2.0", 0.63, "default"), ("key k₂", "1.0", 0.23, "default"),
            ("key k₃", "0.5", 0.14, "default"), ("key k₄", "masked → −∞", 0.0, "ghost")]
    c.text(250, 82, "keys", "lbl")
    c.text(410, 82, "score q·k / √d", "lbl")
    c.text(600, 82, "weight (softmax)", "lbl")
    c.text(800, 82, "values", "lbl")
    for i, (key, score, w, role) in enumerate(rows):
        y = 94 + i * 64
        c.box(196, y, 108, 44, key, role=role)
        c.arrow(146, 205, 192, y + 22)
        tone = "small t-bad" if w == 0 else "mono"
        c.text(410, y + 27, score, tone)
        if w:
            c.raw(f'<rect x="510" y="{y + 12}" width="{w * 220:.1f}" height="20" rx="3" '
                  f'fill="{LIGHT["good"]}" fill-opacity="0.35" stroke="{LIGHT["good"]}"/>')
        c.text(510 + max(w * 220, 0) + 8, y + 27, f"{w:.2f}", "mono", anchor="start")
        c.box(756, y, 96, 44, f"v{'₁₂₃₄'[i]}", role=role)
    c.text(438, 368, "output = 0.63·v₁ + 0.23·v₂ + 0.14·v₃ + 0·v₄", "mono")
    c.text(438, 392, "If every key in a row is masked, the convention is zero weights and a zero output, never NaN.",
           "small t-bad")
    return "l2-x-attention-lookup.svg", c.render(
        "A query is compared with four keys. Scores 2.0, 1.0 and 0.5 and a masked fourth key "
        "become softmax weights 0.63, 0.23, 0.14 and 0; the output is the weighted sum of the "
        "values."
    ), "Attention as a weighted lookup: softmax(QKᵀ/√d)V with a masked key."


@diagram
def batchnorm_modes():
    c = Canvas(W, 340)
    c.title("Two layers, two modes",
            "Batch norm and dropout behave differently in training and at serving.")
    c.text(345, 90, "Training  ·  model.train()", "lbl")
    c.text(705, 90, "Inference  ·  model.eval()", "lbl")
    c.box(24, 104, 140, 74, "BatchNorm", "normalise each feature", role="default")
    c.box(24, 194, 140, 74, "Dropout", "inverted", role="default")
    c.box(190, 104, 310, 74, "batch mean and biased variance",
          "update running stats: 0.9 · old + 0.1 · batch (unbiased var)", role="warn")
    c.box(550, 104, 310, 74, "running mean and variance",
          "output does not depend on the rest of the batch", role="good")
    c.box(190, 194, 310, 74, "zero each unit with prob. p",
          "scale survivors by 1 / (1 − p), so E[output] = input", role="warn")
    c.box(550, 194, 310, 74, "identity", "nothing to undo: the scaling already happened", role="good")
    c.arrow(502, 141, 546, 141, dashed=True)
    _takeaway(c, 284, "Left in training mode at serving, predictions change with the batch they arrive in, and between calls.")
    return "l2-x-batchnorm-modes.svg", c.render(
        "A two-by-two grid. Batch norm in training normalises with batch statistics and updates "
        "running averages; in eval it uses the running averages. Dropout in training zeroes units "
        "with probability p and rescales by 1/(1-p); in eval it is the identity."
    ), "Batch norm and dropout in training versus inference mode."


# ---------------------------------------------------------------------------
# 15 - coding patterns I
# ---------------------------------------------------------------------------


@diagram
def interview_protocol():
    c = Canvas(W, 330)
    c.title("The seven-step coding protocol",
            "Same steps every time, out loud. Minutes are for a 45-minute round.")
    items = [("1 Clarify", "2-4 min: sizes, types, edge inputs", "default"),
             ("2 Examples", "2-3 min: one normal, one edge", "default"),
             ("3 Brute force", "1-2 min: say it and its cost", "default"),
             ("4 Optimise", "3-8 min: bottleneck → pattern", "accent"),
             ("5 Code", "10-15 min: narrate invariants", "accent"),
             ("6 Test", "4-6 min: trace by hand", "warn"),
             ("7 Complexity", "1-2 min: time and space", "good")]
    _row(c, 20, 120, 110, 86, 11, items)
    xc, xt = 20 + 4 * 121 + 55, 20 + 5 * 121 + 55
    c.path(f"M {xt} 118 L {xt} 98 L {xc} 98 L {xc} 116", dashed=True)
    c.text((xc + xt) / 2, 92, "bug? fix it out loud", "small t-warn")
    c.text(440, 236, "Interviewers score: problem solving · coding · verification · communication", "small")
    _takeaway(c, 262, "A narrated solution with a bug you then catch can outscore a silent correct one.")
    return "l2-x-interview-protocol.svg", c.render(
        "Seven boxes left to right: clarify, examples, brute force, optimise, code, test, "
        "complexity, with suggested minutes. A dashed loop goes from test back to code."
    ), "The seven-step communication protocol for a coding round."


@diagram
def pattern_chooser():
    c = Canvas(W, 500)
    c.title("Which pattern? Ask in this order",
            "The first “yes” is usually the pattern. Then state its cost before you code.")
    rows = [("Sorted input, or pairs in a sorted array?", "Two pointers / binary search", "O(n), O(log n)"),
            ("Contiguous subarray with a constraint?", "Sliding window", "O(n)"),
            ("Seen it before? Count occurrences?", "Hash map", "O(n), O(1) lookups"),
            ("Top-k, or next item by priority?", "Heap of size k", "O(n log k)"),
            ("Ranges that overlap?", "Sort, then sweep", "O(n log n)"),
            ("Graph, grid or dependencies?", "BFS / DFS / topological sort", "O(V + E)"),
            ("Overlapping subproblems?", "Dynamic programming", "O(states × transition)")]
    for i, (q, a, cost) in enumerate(rows):
        y = 80 + i * 52
        c.box(30, y, 340, 36, q, role="default")
        c.arrow(372, y + 18, 432, y + 18, accent=True)
        c.text(402, y + 12, "yes", "small t-accent")
        c.box(436, y, 250, 36, a, role="accent")
        c.text(700, y + 23, cost, "small", anchor="start")
        if i < len(rows) - 1:
            c.arrow(110, y + 37, 110, y + 51)
            c.text(118, y + 48, "no", "small", anchor="start")
    _takeaway(c, 446, "Negative numbers break sliding windows (use prefix sums); weights break BFS (use Dijkstra).", h=40)
    return "l2-x-pattern-chooser.svg", c.render(
        "A decision ladder of seven questions. Sorted input leads to two pointers or binary "
        "search; a contiguous window to a sliding window; seen-before or counts to a hash map; "
        "top-k to a heap; overlapping ranges to sort-and-sweep; graphs to BFS, DFS or "
        "topological sort; overlapping subproblems to dynamic programming."
    ), "A decision ladder from problem signals to the coding pattern that fits."


# ---------------------------------------------------------------------------
# 16 - coding patterns II
# ---------------------------------------------------------------------------


def _tree(c, cx, order, role):
    pos = {"A": (cx, 110), "B": (cx - 100, 180), "C": (cx + 100, 180),
           "D": (cx - 150, 250), "E": (cx - 50, 250), "F": (cx + 50, 250), "G": (cx + 150, 250)}
    for p, kids in [("A", "BC"), ("B", "DE"), ("C", "FG")]:
        for k in kids:
            x1, y1 = pos[p]
            x2, y2 = pos[k]
            c.rule(x1, y1 + 18, x2, y2 - 18)
    for name, (x, y) in pos.items():
        _node(c, x, y, str(order.index(name) + 1), role=role)


@diagram
def bfs_dfs():
    c = Canvas(W, 390)
    c.title("BFS and DFS on the same tree",
            "Numbers are the visit order. Only the container differs: a queue or a stack.")
    c.text(260, 82, "BFS · queue (FIFO)", "lbl t-accent")
    c.text(650, 82, "DFS · stack (LIFO)", "lbl t-good")
    for i, y in enumerate((110, 180, 250)):
        c.text(30, y + 4, f"layer {i}", "small", anchor="start")
    _tree(c, 260, "ABCDEFG", "accent")
    _tree(c, 650, "ABDECFG", "good")
    c.text(260, 300, "visits layer by layer: the first visit", "small")
    c.text(260, 316, "is along a path with the fewest edges", "small")
    c.text(650, 300, "goes deep first: cycle checks,", "small")
    c.text(650, 316, "topological order, flood fill", "small")
    _takeaway(c, 334, "Mark a node seen when you enqueue it, use collections.deque, and write DFS with an explicit stack in Python.")
    return "l2-x-bfs-vs-dfs.svg", c.render(
        "The same seven-node tree traversed twice. BFS with a queue visits A, B, C, then D, E, "
        "F, G layer by layer; DFS with a stack visits A, B, D, E, then C, F, G."
    ), "BFS versus DFS visit order on the same tree."


@diagram
def dijkstra_fig():
    c = Canvas(W, 380)
    c.title("Why BFS is not enough when edges have weights",
            "Fewest edges is not cheapest. Dijkstra expands the cheapest frontier node first.")
    n = {0: (90, 200), 1: (560, 200), 2: (240, 300), 3: (410, 300)}
    x0, y0 = n[0]
    x1, y1 = n[1]
    c.arrow(x0 + 20, y0, x1 - 22, y1)
    c.text((x0 + x1) / 2, y0 - 10, "weight 10", "small t-bad")
    c.text((x0 + x1) / 2, y0 - 40, "BFS: 1 hop, cost 10", "lbl t-bad")
    for a, b in [(0, 2), (2, 3), (3, 1)]:
        xa, ya = n[a]
        xb, yb = n[b]
        dx, dy = xb - xa, yb - ya
        L = math.hypot(dx, dy)
        c.arrow(xa + 20 * dx / L, ya + 20 * dy / L, xb - 22 * dx / L, yb - 22 * dy / L, accent=True)
        c.text((xa + xb) / 2 - 14 * dy / L, (ya + yb) / 2 + 14 * dx / L + 4, "1", "lbl t-accent")
    for k, (x, y) in n.items():
        _node(c, x, y, str(k), role="good" if k == 1 else "accent", r=20)
    c.text(325, 346, "Dijkstra: 3 hops, cost 3", "lbl t-good")
    c.box(626, 84, 230, 220, role="ghost", radius=8)
    c.text(741, 104, "heap of (distance, node)", "lbl")
    trace = ["pop (0,0): push (10,1), (1,2)", "pop (1,2): push (2,3)", "pop (2,3): push (3,1)",
             "pop (3,1): dist[1] = 3", "pop (10,1): stale, skip"]
    for i, t in enumerate(trace):
        c.text(638, 136 + i * 30, t, "mono", anchor="start")
    c.text(741, 286, "no decrease-key: skip stale entries", "small t-warn")
    c.text(660, 346, "O((V + E) log V); weights must be ≥ 0", "small")
    return "l2-x-dijkstra.svg", c.render(
        "A four-node graph: a direct edge from 0 to 1 of weight 10, and a path 0-2-3-1 with "
        "three edges of weight 1. BFS takes the one-hop edge at cost 10; Dijkstra finds the "
        "three-hop path at cost 3. A heap trace shows the pops, including a stale entry skipped."
    ), "BFS minimises hops; Dijkstra's heap finds the cheapest weighted path."


@diagram
def union_find():
    c = Canvas(W, 440)
    c.title("Union-find: path compression flattens the tree",
            "find(x) returns the root of x's set. Compression points every visited node at the root.")
    ys = [96, 156, 216, 276, 336]
    labels = ["r", "3", "2", "1", "x"]
    for i in range(1, 5):
        c.arrow(200, ys[i] - 18, 200, ys[i - 1] + 18)
    for y, lab in zip(ys, labels):
        _node(c, 200, y, lab, role="good" if lab == "r" else "accent", r=17)
    c.text(240, 340, "find(x) walks 4 steps", "small t-bad", anchor="start")
    c.text(240, 100, "root", "small t-good", anchor="start")
    c.arrow(330, 216, 470, 216, accent=True, label="after find(x)")
    rx, ry = 660, 130
    kids = [(540, 300, "x"), (620, 300, "1"), (700, 300, "2"), (780, 300, "3")]
    for kx, ky, _ in kids:
        dx, dy = rx - kx, ry - ky
        L = math.hypot(dx, dy)
        c.arrow(kx + 17 * dx / L, ky + 17 * dy / L, rx - 19 * dx / L, ry - 19 * dy / L)
    _node(c, rx, ry, "r", role="good", r=17)
    for kx, ky, lab in kids:
        _node(c, kx, ky, lab, r=17)
    c.text(660, 346, "next find from any of them: 1 step", "small t-good")
    _takeaway(c, 376, "Path compression plus union by size: O(α(n)) amortised per operation, effectively constant.")
    return "l2-x-union-find.svg", c.render(
        "Left: a chain of parent pointers from x up to the root r, four steps long. Right: after "
        "path compression every node on that path points directly at the root."
    ), "Path compression turns a chain into a star: near-O(1) find."


# ---------------------------------------------------------------------------
# 17 - serving a ranking cascade
# ---------------------------------------------------------------------------


@diagram
def ranking_cascade():
    c = Canvas(W, 364)
    c.title("A ranking cascade and its latency budget",
            "Cheap stages shrink the set so the expensive stage only sees a few hundred items.")
    items = [("Catalogue", "4,000 items (real: 10⁶-10⁹)", "default"),
             ("Retrieval", "ANN + co-visitation + popular → ~300", "accent"),
             ("Feature fetch", "300 × F from the online store", "accent"),
             ("Ranker", "LightGBM scores 300 rows", "accent"),
             ("Rules", "dedupe, category cap → top 10", "good")]
    _row(c, 18, 86, 148, 72, 26, items)
    c.text(24, 206, "p99 budget for a 100 ms SLO (notebook 17.8)", "lbl", anchor="start")
    segs = [("retrieve 20 ms", 20, "accent", 0.18), ("features 20 ms", 20, "accent", 0.30),
            ("rank 30 ms", 30, "accent", 0.45), ("5", 5, "muted_fg", 0.25),
            ("outside the stages 25 ms", 25, "warn", 0.35)]
    x = 24.0
    for label, ms, tone, op in segs:
        w = ms * 8.32
        c.raw(f'<rect x="{x:.1f}" y="218" width="{w:.1f}" height="40" fill="{LIGHT[tone]}" '
              f'fill-opacity="{op}" stroke="#ffffff" stroke-width="2"/>')
        c.text(x + w / 2, 243, label, "small")
        x += w
    c.text(24 + 72.5 * 8.32 + 20.8, 276, "rules", "small")
    c.text(24 + 87.5 * 8.32, 276, "HTTP, queueing, GIL", "small t-warn")
    _takeaway(c, 300, "Retrieval sets the ceiling (recall); the ranker spends the most; leave slack for what you did not instrument.")
    return "l2-x-ranking-cascade.svg", c.render(
        "Five stages left to right: catalogue, retrieval to about 300 candidates, feature fetch, "
        "LightGBM ranker, business rules to a top 10. Below, a stacked bar splits a 100 ms p99 "
        "budget: retrieval 20, features 20, ranking 30, rules 5 and 25 ms outside the stages."
    ), "The four-stage ranking cascade with a per-stage p99 latency budget."


@diagram
def lru_cache():
    c = Canvas(W, 420)
    c.title("An LRU cache: a hash map plus a doubly linked list",
            "The map finds a node in O(1); the list keeps recency order and evicts from the back in O(1).")
    c.box(40, 84, 800, 48, "dict: key → node", "OrderedDict in Python is exactly this pair", role="accent")
    nodes = [("user 7", "most recent", "good"), ("user 3", "", "default"),
             ("user 9", "", "default"), ("user 1", "least recent: evicted next", "bad")]
    for i, (lab, sub, role) in enumerate(nodes):
        x = 60 + i * 210
        c.box(x, 196, 150, 48, lab, role=role)
        c.arrow(x + 75, 134, x + 75, 192)
        if sub:
            c.text(x + 75, 266, sub, f"small t-{'good' if i == 0 else 'bad'}")
        if i:
            c.arrow(x - 58, 212, x - 3, 212)
            c.arrow(x - 3, 230, x - 58, 230)
    c.box(40, 290, 390, 56, "get(key)", "hit: move node to the front · miss: count it", role="default")
    c.box(450, 290, 390, 56, "put(key, value)", "insert at the front; if full, drop the back", role="default")
    _takeaway(c, 362, "Hit rate is capped by repeats: 1 − 3,177 / 4,483 ≈ 0.29 in one day of ShopStream requests.")
    return "l2-x-lru-cache.svg", c.render(
        "A dictionary maps each key to a node in a doubly linked list ordered by recency: user 7 "
        "most recent at the front, user 1 least recent at the back and evicted next. get moves a "
        "hit to the front; put inserts at the front and evicts the back when full."
    ), "LRU cache as a hash map over a doubly linked list."


@diagram
def degradation():
    c = Canvas(W, 400)
    c.title("Graceful degradation: always return something sensible",
            "Healthy: p50 ≈ 5 ms, catalogue NDCG@10 0.72. Each failure has a cheaper fallback.")
    c.text(150, 92, "What breaks", "lbl")
    c.text(470, 92, "What users get", "lbl")
    c.text(740, 92, "Cost (notebook 17 run)", "lbl")
    rows = [("Ranker raises", "candidates in popularity order", "NDCG 0.38, faster"),
            ("Ranker over 50 ms", "same fallback, at the timeout", "p50 ≈ 54 ms, NDCG 0.40"),
            ("Feature store down", "default features; co-visitation still works", "NDCG 0.45"),
            ("Retrieval fails", "cached popular list", "availability only")]
    for i, (a, b, cost) in enumerate(rows):
        y = 106 + i * 60
        c.box(40, y, 220, 46, a, role="bad")
        c.arrow(262, y + 23, 316, y + 23)
        c.box(320, y, 300, 46, b, role="warn")
        c.text(740, y + 28, cost, "small")
    _takeaway(c, 352 - 6, "Count every fallback and alert on it: silently serving popularity for a week is an outage nobody sees.")
    return "l2-x-graceful-degradation.svg", c.render(
        "Four failure rows: ranker error and ranker timeout fall back to popularity order; a "
        "feature-store outage uses default features; a retrieval failure serves a cached "
        "popular list. Each row lists its measured cost."
    ), "The fallback ladder of a ranking service and what each fallback costs."


# ---------------------------------------------------------------------------
# 18 - model optimisation for inference
# ---------------------------------------------------------------------------


@diagram
def inference_pipeline():
    c = Canvas(W, 360)
    c.title("Three ways to make a ranker cheaper, and how to test each",
            "Lossless steps get exact parity; lossy steps get ranking-quality thresholds.")
    c.text(24, 86, "Lossless", "lbl t-good", anchor="start")
    _row(c, 24, 96, 186, 64, 30, [("PyTorch MLP fp32", "9,729 parameters", "default"),
                                  ("Export to ONNX", "dynamic batch dimension", "accent"),
                                  ("ONNX Runtime", "C++ engine, no Python", "accent"),
                                  ("Exact parity", "max |Δ| ≈ 1e-6 ≤ atol", "good")])
    c.text(24, 196, "Lossy", "lbl t-warn", anchor="start")
    _row(c, 24, 206, 186, 64, 30, [("LightGBM teacher", "soft labels p_t", "default"),
                                   ("Distil a student", "16 hidden units, 193 params", "accent"),
                                   ("Dynamic int8", "4× smaller weights", "accent"),
                                   ("Ranking checks", "Spearman, top-k, NDCG", "warn")])
    _takeaway(c, 294, "One table decides: here int8 did not speed up a tiny model, while changing the model class did.")
    return "l2-x-inference-optimization.svg", c.render(
        "Top row, lossless: PyTorch model exported to ONNX and run in ONNX Runtime, tested with "
        "exact parity. Bottom row, lossy: a LightGBM teacher distilled into a small student and "
        "quantised to int8, tested with Spearman correlation, top-k overlap and NDCG."
    ), "ONNX export, distillation and int8 quantisation, each with its own test."


@diagram
def quantization():
    c = Canvas(W, 380)
    c.title("Affine int8 quantisation: a scale and a zero-point",
            "Map the float range [min, max] onto the 256 integers [−128, 127].")
    x0, x1 = 100, 780

    def F(v):
        return x0 + (v + 1.2) / 3.5 * (x1 - x0)

    c.rule(x0, 120, x1, 120)
    c.rule(x0, 220, x1, 220)
    c.text(x0 - 12, 124, "float", "small", anchor="end")
    c.text(x0 - 12, 224, "int8", "small", anchor="end")
    pairs = [(-1.2, "−1.2 (min)", "−128"), (0.0, "0.0", "−41 = z"), (1.0, "1.0", "32"), (2.3, "2.3 (max)", "127")]
    for v, ft, it in pairs:
        x = F(v)
        c.raw(f'<line x1="{x:.1f}" y1="114" x2="{x:.1f}" y2="126" stroke="{LIGHT["fg"]}" stroke-width="1.6"/>')
        c.raw(f'<line x1="{x:.1f}" y1="214" x2="{x:.1f}" y2="226" stroke="{LIGHT["fg"]}" stroke-width="1.6"/>')
        c.arrow(x, 130, x, 208, accent=(v == 0.0))
        c.text(x, 104, ft, "mono")
        c.text(x, 246, it, "mono t-accent" if v == 0.0 else "mono")
    rows = [("scale", "s = (max − min) / 255 = 3.5 / 255 ≈ 0.0137", ""),
            ("zero-point", "z = round(−128 − min / s) = −41", "0.0 is stored exactly"),
            ("quantise", "q = clip(round(x / s) + z, −128, 127)", "1.0 → 73 − 41 = 32"),
            ("dequantise", "x̂ = s · (q − z)", "32 → 0.0137 × 73 = 1.002")]
    for i, (name, formula, example) in enumerate(rows):
        y = 290 + i * 20
        c.text(24, y, name, "small", anchor="start")
        c.text(120, y, formula, "mono", anchor="start")
        c.text(560, y, example, "mono t-accent", anchor="start")
    return "l2-x-quantization.svg", c.render(
        "A float number line from -1.2 to 2.3 maps onto the int8 line from -128 to 127. Zero "
        "maps exactly to the zero-point -41 and 1.0 maps to 32. The scale is 3.5/255."
    ), "Int8 quantisation with a scale and zero-point, on a worked example."


# ---------------------------------------------------------------------------
# 19 - recsys and search system design
# ---------------------------------------------------------------------------


@diagram
def recsys_design():
    c = Canvas(W, 400)
    c.title("Design A: a home feed for 50M daily users",
            "Five candidate sources feed one cascade. Numbers are design assumptions, stated aloud.")
    srcs = [("Two-tower ANN", "800"), ("Co-visitation", "500"), ("Followed sellers", "300"),
            ("Fresh items", "200 · cold start"), ("Trending", "200 · fallback")]
    for i, (lab, sub) in enumerate(srcs):
        y = 84 + i * 50
        c.box(24, y, 190, 42, lab, sub, role="default")
        c.arrow(216, y + 21, 272, 181 + i * 12)
    _row(c, 276, 170, 130, 70, 22, [("Merge + filter", "~2,000, drop sold-out", "accent"),
                                    ("Pre-ranker", "2,000 → 500, tiny MLP", "accent"),
                                    ("Heavy ranker", "multi-task: click, cart, buy", "accent"),
                                    ("Re-rank", "diversity, caps → 20", "good")])
    for i, ms in enumerate(["5 ms", "10 ms", "45 ms", "10 ms"]):
        c.text(276 + i * 152 + 65, 258, ms, "small")
    c.text(119, 350, "25 ms, in parallel", "small")
    c.text(560, 300, "peak ≈ 28,900 QPS · 14.5M heavy-ranker rows/s ≈ 38 servers", "small")
    c.text(560, 318, "14.5M item-feature reads/s → local in-memory cache", "small")
    c.box(276, 336, 580, 44, "Say the numbers: they decide the architecture.", role="default")
    return "l2-x-recsys-design.svg", c.render(
        "Five candidate sources - two-tower ANN, co-visitation, followed sellers, fresh items and "
        "trending - merge into about 2,000 candidates, then a pre-ranker to 500, a multi-task "
        "heavy ranker and a re-ranking step to a page of 20, with per-stage latency."
    ), "The home-feed design: candidate sources, pre-ranker, heavy ranker, re-ranking."


@diagram
def search_design():
    c = Canvas(W, 360)
    c.title("Design B: product search ranking",
            "Understand the query, retrieve two ways, rank twice. p99 300 ms end to end.")
    c.box(24, 134, 84, 72, "Query", "“red nike shoes”", role="default")
    c.box(128, 134, 136, 72, "Understand", "spelling, category, attributes", role="accent")
    c.box(284, 78, 172, 64, "Lexical: BM25", "top 1,000, exact terms", role="accent")
    c.box(284, 198, 172, 64, "Semantic two-tower", "top 500, “sofa” = “couch”", role="accent")
    c.box(476, 134, 136, 72, "L1 LambdaMART", "~100 features → 200", role="accent")
    c.box(634, 134, 110, 72, "L2 cross-encoder", "top 50", role="accent")
    c.box(764, 134, 92, 72, "Rules", "stock, ads", role="good")
    c.arrow(110, 170, 124, 170, accent=True)
    c.arrow(266, 156, 280, 128, accent=True)
    c.arrow(266, 184, 280, 212, accent=True)
    c.arrow(458, 128, 472, 156, accent=True)
    c.arrow(458, 212, 472, 184, accent=True)
    c.arrow(614, 170, 630, 170, accent=True)
    c.arrow(746, 170, 760, 170, accent=True)
    for x, t in [(196, "20 ms"), (368, "60 ms, in parallel"), (544, "40 ms"), (689, "60 ms"), (810, "20 ms")]:
        c.text(x, 284, t, "small")
    c.text(440, 304, "+ 70 ms network and slack", "small")
    _takeaway(c, 318 - 2, "Labels: human relevance grades (unbiased, scarce) plus clicks (plentiful, position-biased → IPS).", h=36)
    return "l2-x-search-design.svg", c.render(
        "A query passes through query understanding, then lexical BM25 and semantic two-tower "
        "retrieval in parallel, a LambdaMART first-stage ranker, a cross-encoder second stage on "
        "the top 50, and business rules, with a latency budget per stage."
    ), "The product-search design: query understanding, hybrid retrieval, two ranking stages."


# ---------------------------------------------------------------------------
# 20 - final evaluation and mock loop
# ---------------------------------------------------------------------------


@diagram
def prereg_flow():
    c = Canvas(W, 330)
    c.title("Open the test set once, under a plan written first",
            "The pre-registration's hash goes into the results file, so anyone can check the order.")
    items = [("Choose", "on validation, days 42-48", "default"),
             ("Pre-register", "metrics, sample, CI method; save + hash", "accent"),
             ("Open test once", "days 49-55, nothing tuned", "warn"),
             ("Bootstrap by user", "1,000 replicates, paired diffs", "accent"),
             ("Report", "number, interval, plan hash", "good")]
    _row(c, 24, 96, 148, 76, 25, items)
    x_choose, x_test = 24 + 74, 24 + 2 * 173 + 74
    c.path(f"M {x_test} 174 L {x_test} 210 L {x_choose} 210 L {x_choose} 176", dashed=True)
    c.text((x_choose + x_test) / 2, 228, "never: look at test, then re-choose", "small t-bad")
    _takeaway(c, 262, "Every look at test that changes a choice turns the test set into a second validation set.")
    return "l2-x-preregistration.svg", c.render(
        "Five steps: choose the system on validation, write and hash a pre-registration, open "
        "the test week once, bootstrap by user, and report with the plan's hash. A dashed "
        "forbidden path goes from the test set back to choosing."
    ), "Pre-register, open the test set once, bootstrap by user, report."


@diagram
def mock_loop():
    c = Canvas(W, 364)
    c.title("The mock loop, and the loop around it",
            "Four timed rounds, scored at once, logged, and the weakest one practised next.")
    top = [("Coding", "45 min · notebooks 15-16", "accent"),
           ("ML coding", "45 min · notebooks 13-14", "accent"),
           ("System design", "45 min · notebooks 17-19", "accent"),
           ("Behavioural", "30 min · STAR stories", "accent")]
    bottom = [("Next mock", "fresh prompts, a week later", "good"),
              ("Practise", "the lowest column first", "default"),
              ("Log it", "progress_log.csv grows", "default"),
              ("Score in 10 min", "rubric 1-4 per dimension", "warn")]
    _row(c, 40, 88, 182, 64, 24, top)
    for i, (lab, sub, role) in enumerate(bottom):
        x = 40 + i * 206
        c.box(x, 216, 182, 64, lab, sub, role=role)
        if i < 3:
            c.arrow(x + 182 + 22, 248, x + 185, 248, accent=True)
    c.arrow(40 + 3 * 206 + 91, 154, 40 + 3 * 206 + 91, 212, accent=True)
    c.arrow(131, 214, 131, 156, accent=True)
    _takeaway(c, 300, "Aim for 3+ in every round, on fresh prompts, before booking real interviews.")
    return "l2-x-mock-loop.svg", c.render(
        "Top row: four timed rounds - coding, ML coding, system design, behavioural. Bottom row, "
        "right to left: score within ten minutes, log it, practise the lowest column, and run "
        "the next mock, which leads back to the first round."
    ), "The four-round mock loop and the practise-log-repeat cycle around it."
