"""Diagrams for Level 2 notebooks 01-10 (and the Level 2 README).

See tools/make_diagrams.py for the conventions. Every function returns
``(filename, svg, caption)``; the caption doubles as alt text in the index.
Coordinates sit on a 10-pixel grid wherever possible, flows run left to
right, and each figure ends with one takeaway box.
"""

from __future__ import annotations

import math

from svgkit import Canvas, diagram

W = 880  # standard canvas width


def takeaway(c: Canvas, y: float, text: str, role: str = "good", h: float = 40) -> None:
    """The one-line lesson at the bottom of every figure."""
    c.box(24, y, W - 48, h, text, role=role)


# ---------------------------------------------------------------------------
# README + notebook 01
# ---------------------------------------------------------------------------


@diagram
def shopstream_at_a_glance():
    c = Canvas(W, 360)
    c.title("ShopStream at a glance",
            "How one feed request becomes log rows, and what the log can never contain.")
    steps = [
        ("Users & items", "10k users, 4k items, latent tastes", "default"),
        ("Request", "one feed load, ~245k in 8 weeks", "default"),
        ("Logging policy", "picks 10 of 200; 5% random order", "accent"),
        ("Examination", "sees position p with prob. θ_p", "warn"),
        ("Click", "P = θ_p · r", "accent"),
        ("Purchase", "some clicks, price-dependent", "accent"),
    ]
    x, y, w, h, gap = 24, 80, 122, 76, 20
    for i, (label, sub, role) in enumerate(steps):
        c.box(x, y, w, h, label, sub, role=role)
        if i < len(steps) - 1:
            c.arrow(x + w + 2, y + h / 2, x + w + gap - 2, y + h / 2, accent=True)
        x += w + gap
    # Bottom: what is logged vs what only the simulation knows.
    c.box(24, 196, 400, 76, "Logged: what you get",
          "request_id · user_id · item_id · ts · position · is_random · clicked · purchased",
          role="good")
    c.box(456, 196, 400, 76, "Hidden truth: graders only",
          "true relevance r(u, i) · examined flag · θ_p",
          role="bad")
    # Examination and click feed the hidden box from directly above.
    c.arrow(509, 158, 509, 194, dashed=True)
    c.arrow(651, 158, 651, 194, dashed=True)
    takeaway(c, 296, "Policy CTR falls from 0.47 (position 0) to 0.05 (position 9): "
                     "mostly examination, not relevance.", role="warn")
    return "l2-shopstream-at-a-glance.svg", c.render(
        "ShopStream's generating process: users and items, a feed request, a logging "
        "policy that picks 10 items (5% of requests in random order), examination that "
        "depends on position, a click with probability theta_p times relevance, and "
        "sometimes a purchase. The log records ids, timestamp, position, the random flag, "
        "clicks and purchases; the true relevance, the examined flag and theta_p stay hidden."
    ), "ShopStream: users and items, a logging policy, position-biased examination, clicks and purchases; what is logged vs the hidden truth."


@diagram
def scan_skipping():
    c = Canvas(W, 380)
    c.title("Three ways a query reads less",
            "Partition pruning, row-group statistics and column projection, outermost first.")
    # Panel 1: partition folders.
    c.text(24, 86, "1 · Partition pruning", "lbl", anchor="start")
    c.text(24, 104, "WHERE day BETWEEN 42 AND 48", "mono", anchor="start")
    cell, gap = 16, 3
    for d in range(56):
        r, k = divmod(d, 14)
        role = "accent" if 42 <= d <= 48 else "ghost"
        c.box(24 + k * (cell + gap), 124 + r * (cell + gap), cell, cell, role=role, radius=3)
    c.text(24, 220, "56 folders day=0 … day=55", "small", anchor="start")
    c.text(24, 238, "7/56 files opened, from the path alone", "small t-accent", anchor="start")

    # Panel 2: row groups with min/max.
    c.text(316, 86, "2 · Row-group statistics", "lbl", anchor="start")
    c.text(316, 104, "WHERE request_id BETWEEN …", "mono", anchor="start")
    ranges = ["0 – 12k", "12k – 24k", "24k – 37k", "37k – 49k", "49k – 61k"]
    for i, rg in enumerate(ranges):
        role = "accent" if i == 2 else "ghost"
        note = "may match: read" if i == 2 else "skipped by min/max"
        c.box(316, 120 + i * 22, 250, 18, role=role, radius=3)
        c.text(324, 133 + i * 22, f"rg{i}  {rg}", "small", anchor="start")
        c.text(558, 133 + i * 22, note, "small t-accent" if i == 2 else "small", anchor="end")
    c.text(316, 238, "file sorted by the filter column", "small t-accent", anchor="start")

    # Panel 3: column projection.
    c.text(612, 86, "3 · Column projection", "lbl", anchor="start")
    c.text(612, 104, "SELECT position, clicked", "mono", anchor="start")
    cols = ["rid", "uid", "item", "ts", "pos", "rnd", "clk", "buy"]
    for i, name in enumerate(cols):
        role = "accent" if name in ("pos", "clk") else "ghost"
        x = 612 + i * 31
        c.box(x, 120, 24, 86, role=role, radius=3)
        c.text(x + 12, 222, name, "small", anchor="middle")
    c.text(612, 238, "2 of 8 columns decompressed", "small t-accent", anchor="start")

    c.rule(24, 262, 856, 262)
    c.text(24, 288, "CSV can do none of this: one text row after another, every byte parsed, "
                    "types re-guessed on every read.", "small", anchor="start")
    takeaway(c, 310, "Filter on the partition column, sort by the next filter column, "
                     "select only the columns you need.")
    return "l2-scan-skipping.svg", c.render(
        "Three levels of skipping in a Parquet scan. Partition pruning opens 7 of 56 day "
        "folders from the path alone. Row-group min/max statistics skip row groups whose "
        "request_id range cannot match. Column projection decompresses only position and "
        "clicked, 2 of 8 columns."
    ), "Partition pruning, row-group statistics and column projection: three ways a columnar scan reads less."


# ---------------------------------------------------------------------------
# Notebook 02
# ---------------------------------------------------------------------------


@diagram
def map_shuffle_reduce():
    c = Canvas(W, 380)
    c.title("Map → shuffle → reduce, with a combiner",
            "SELECT item_id, count(*), sum(clicked) … GROUP BY item_id on 4 machines.")
    ys = [86, 156, 226]
    for i, y in enumerate(ys):
        c.box(24, y, 130, 54, f"Split {i + 1}", "a few Parquet files")
        c.arrow(156, y + 27, 186, y + 27, accent=True)
        c.box(190, y, 160, 54, "Map + combine", "one row per key seen", role="accent")
        c.arrow(352, y + 27, 400, y + 27, accent=True)
    c.box(404, 86, 140, 194, "Shuffle",
          "send (key, partial) to reducer hash(key) % R · network, disk, sort", role="warn")
    for i, y in enumerate(ys):
        c.arrow(546, y + 27, 586, y + 27, accent=True)
        c.box(590, y, 140, 54, f"Reducer {i}", "merge partials per key", role="accent")
        c.arrow(732, y + 27, 762, y + 27, accent=True)
    c.box(766, 86, 90, 194, "Result", "one row per item", role="good")
    c.text(276, 300, "2,446,900 rows in", "small", anchor="middle")
    c.text(474, 300, "~15,000 rows moved", "small t-warn", anchor="middle")
    c.text(474, 316, "(160× fewer than without it)", "small", anchor="middle")
    takeaway(c, 330, "Combiners shrink algebraic aggregates (count, sum, avg); "
                     "count(DISTINCT) and medians do not shrink.")
    return "l2-map-shuffle-reduce.svg", c.render(
        "A distributed GROUP BY: each input split is mapped and pre-aggregated by a "
        "combiner, the partial rows are shuffled to reducer hash(key) mod R, and each "
        "reducer merges the partials for its keys. The combiner cuts the shuffled rows "
        "from 2.4 million to about 15 thousand."
    ), "Map, combine, shuffle by hash(key), reduce: where a distributed GROUP BY spends its time."


@diagram
def skew_salting():
    c = Canvas(W, 400)
    c.title("Skew, and salting",
            "Distinct users per seller on 8 reducers: one hot key makes one straggler.")
    base, unit = 250, 1.0

    def panel(x0, loads, role_of, heading, key_text, ratio):
        c.text(x0, 86, heading, "lbl", anchor="start")
        c.text(x0, 104, key_text, "mono", anchor="start")
        mean = sum(loads) / len(loads)
        for i, load in enumerate(loads):
            h = load * unit
            c.box(x0 + i * 46, base - h, 34, h, role=role_of(load, mean), radius=3)
        y_mean = base - mean * unit
        c.raw(f'<line x1="{x0 - 6}" y1="{y_mean:.1f}" x2="{x0 + 8 * 46 - 6}" y2="{y_mean:.1f}" '
              f'stroke="#656d76" stroke-width="1.2" stroke-dasharray="4 3"/>')
        c.text(x0 + 8 * 46 - 4, y_mean - 5, "mean", "small", anchor="end")
        c.rule(x0 - 6, base, x0 + 8 * 46 - 6, base)
        c.text(x0, base + 20, "reducer 0 … 7", "small", anchor="start")
        c.text(x0 + 8 * 46 - 6, base + 20, ratio, "small t-accent", anchor="end")

    panel(24, [126, 22, 18, 20, 16, 21, 19, 17],
          lambda load, m: "bad" if load > 2 * m else "default",
          "Without salt", "key = seller", "max / mean ≈ 3.9")
    panel(476, [34, 32, 33, 31, 35, 32, 34, 33],
          lambda load, m: "good",
          "With salt S = 8", "key = (seller, hash(user) % S)", "max / mean ≈ 1.05")
    c.text(476, 300, "stage 2: add the S partial counts per seller (exact,", "small", anchor="start")
    c.text(476, 316, "because one user always gets the same salt)", "small", anchor="start")
    c.text(24, 300, "the whole job waits for the hot reducer:", "small", anchor="start")
    c.text(24, 316, "\"99% of tasks finished an hour ago\"", "small t-bad", anchor="start")
    takeaway(c, 340, "Salt only the hot keys, then re-combine in a second, tiny stage.")
    return "l2-skew-salting.svg", c.render(
        "Reducer loads for distinct users per seller. Without salting one reducer holds "
        "the top seller's pairs and runs about four times the mean load. With the key "
        "salted as seller plus hash of user mod 8 the loads are nearly equal, and a second "
        "small stage adds the partial counts exactly."
    ), "Key skew makes one reducer a straggler; salting the hot key spreads it and a second stage re-combines exactly."


@diagram
def join_strategies():
    c = Canvas(W, 380)
    c.title("Broadcast vs shuffle hash join",
            "Who moves over the network decides the cost.")
    # Left: broadcast.
    c.text(24, 86, "Broadcast hash join", "lbl", anchor="start")
    c.box(24, 156, 140, 60, "items", "small: copy everywhere", role="accent")
    for i, y in enumerate([100, 160, 220]):
        c.box(230, y, 176, 50, f"impressions part {i + 1}", "⋈ local copy of items")
        c.arrow(166, 186, 226, y + 25, accent=True)
    c.text(24, 300, "the big side never moves: no shuffle", "small t-good", anchor="start")

    # Right: shuffle hash.
    c.text(456, 86, "Shuffle hash join", "lbl", anchor="start")
    c.box(456, 110, 120, 56, "impressions", "big")
    c.box(456, 210, 120, 56, "clicks", "big")
    c.arrow(578, 138, 618, 160, accent=True)
    c.arrow(578, 238, 618, 216, accent=True)
    c.box(622, 110, 100, 156, "Shuffle both", "by hash(key)", role="warn")
    for i, y in enumerate([110, 166, 222]):
        c.arrow(724, y + 22, 752, y + 22, accent=True)
        c.box(756, y, 100, 44, f"partition {i}", "build + probe")
    c.text(456, 300, "both sides cross the network; a hot key → one huge partition",
           "small t-warn", anchor="start")
    takeaway(c, 320, "Small side → broadcast it. Both large → shuffle hash or sort-merge. "
                     "Skewed side → aggregate first.")
    return "l2-join-strategies.svg", c.render(
        "Left: a broadcast hash join copies the small items table to every partition of "
        "the big impressions table, so the big side is never shuffled. Right: a shuffle "
        "hash join repartitions both big tables by hash of the key, so both cross the "
        "network and a hot key produces one huge partition."
    ), "Broadcast hash join (copy the small side) versus shuffle hash join (repartition both sides)."


@diagram
def asof_join():
    c = Canvas(W, 380)
    c.title("As-of join: the price when the item was shown",
            "For each event: the latest record with the same key and ts ≤ event ts.")
    x0, scale = 60, 28
    axis_y = 200

    def tx(day):
        return x0 + day * scale

    c.arrow(40, axis_y, 850, axis_y)
    c.text(850, axis_y + 22, "time (days)", "small", anchor="end")
    prices = [(3, "$20"), (10, "$18"), (20, "$15")]
    for day, p in prices:
        c.box(tx(day) - 50, 92, 100, 54, f"price {p}", f"valid from day {day}", role="accent")
        c.arrow(tx(day), 148, tx(day), axis_y - 4, dashed=True)
    events = [(5, "$20"), (12, "$18"), (25, "$15")]
    for (day, p), (pday, _) in zip(events, prices):
        c.circle(tx(day), axis_y, 5, role="good")
        c.box(tx(day) - 60, 240, 120, 54, f"shown day {day}", f"gets {p}", role="good")
        c.arrow(tx(day), 238, tx(day), axis_y + 8)
        # Look back along the timeline to the latest earlier record.
        c.path(f"M {tx(day) - 6} {axis_y - 12} L {tx(pday) + 8} {axis_y - 12}", accent=True)
    c.text(tx(22.5), axis_y - 20, "look back", "small t-accent", anchor="middle")
    # "Today": the naive join uses the last price for everything.
    c.circle(tx(27.5), axis_y, 5, role="bad")
    c.text(tx(27.5), axis_y - 14, "today: $15", "small t-bad", anchor="middle")
    takeaway(c, 316, "A plain join to the current price gives every row $15: wrong for 55% of "
                     "impressions, and it leaks the future.", role="bad")
    return "l2-asof-join.svg", c.render(
        "A timeline with price records valid from day 3 ($20), day 10 ($18) and day 20 "
        "($15), and impressions on days 5, 12 and 25. The as-of join gives each impression "
        "the latest price at or before its time: $20, $18 and $15. A plain join to today's "
        "price would give every impression $15."
    ), "As-of join on a timeline: each event takes the latest record at or before its timestamp."


# ---------------------------------------------------------------------------
# Notebook 03
# ---------------------------------------------------------------------------


@diagram
def feature_store():
    c = Canvas(W, 400)
    c.title("Offline and online feature stores",
            "One feature definition; two stores; the same TTL in both.")
    c.box(24, 160, 130, 64, "Event log", "impressions, clicks")
    c.arrow(156, 192, 186, 192, accent=True)
    c.box(190, 160, 150, 64, "Daily snapshots", "window [d−7, d−1], ts = d 00:00",
          role="accent")
    # Up: offline. Down: online.
    c.path("M 342 182 L 366 182 L 366 116 L 386 116", accent=True)
    c.path("M 342 202 L 366 202 L 366 268 L 386 268", accent=True)
    c.box(390, 86, 160, 60, "Offline store", "every value + ts")
    c.arrow(552, 116, 582, 116, accent=True)
    c.box(586, 86, 140, 60, "PIT join", "ts > snapshot_ts", role="accent")
    c.arrow(728, 116, 752, 116, accent=True)
    c.box(756, 86, 100, 60, "Training", "rows as of then", role="good")

    c.box(390, 238, 160, 60, "Online store", "latest, loaded daily")
    c.arrow(552, 268, 582, 268, accent=True)
    c.box(586, 238, 140, 60, "get(key, now)", "expired after TTL", role="accent")
    c.arrow(728, 268, 752, 268, accent=True)
    c.box(756, 238, 100, 60, "Serving", "values as of now", role="good")

    c.path("M 806 148 L 806 236", dashed=True, arrow=False)
    c.text(798, 196, "parity", "small t-warn", anchor="end")
    c.text(798, 210, "test", "small t-warn", anchor="end")
    takeaway(c, 330, "Training sees each feature as it was just before the event; "
                     "serving sees the latest.")
    return "l2-feature-store.svg", c.render(
        "The event log is aggregated into daily snapshots over the window d-7 to d-1. "
        "Snapshots go to the offline store, which a point-in-time as-of join turns into "
        "training rows, and are materialised daily into the online store, which serves the "
        "latest value per key with a TTL. A parity test compares training and serving values."
    ), "Offline store plus point-in-time join for training; online store with TTL for serving; a parity test between them."


# ---------------------------------------------------------------------------
# Notebook 04
# ---------------------------------------------------------------------------


@diagram
def event_time_watermark():
    c = Canvas(W, 420)
    c.title("Event time, processing time and the watermark",
            "Each dot is one event: when it happened (up) vs when it arrived (right).")
    ox, oy, sx, sy = 80, 330, 46, 22

    def pt(arrival, event):
        return ox + arrival * sx, oy - event * sy

    c.arrow(ox, oy, ox + 10.4 * sx, oy)
    c.arrow(ox, oy, ox, oy - 10.8 * sy)
    c.text(ox + 10.4 * sx, oy + 22, "processing time (arrival) →", "small", anchor="end")
    c.text(ox - 10, oy - 10.8 * sy - 8, "event time ↑", "small", anchor="start")
    x1, y1 = pt(0, 0)
    x2, y2 = pt(10, 10)
    c.curve(f"M {x1} {y1} L {x2} {y2}", tone="stroke", dashed=True, width=1.6)
    c.text(x2 - 4, y2 - 8, "no delay", "small", anchor="end")

    events = [(1.3, 1), (2.2, 2), (3.1, 2.6), (3.8, 3.5), (4.9, 4.2), (5.4, 5),
              (6.6, 6), (7.2, 6.8), (8.1, 7.5), (9.0, 8.6)]
    # Watermark = max event time seen so far minus allowed lateness (1 unit).
    lateness, d, best, prev_y = 1.0, "", -1.0, 0.0
    for a, e in events:
        best = max(best, e)
        wx, wy = pt(a, best - lateness)
        if not d:
            d = f"M {wx:.1f} {wy:.1f}"
        else:
            d += f" L {wx:.1f} {prev_y:.1f} L {wx:.1f} {wy:.1f}"
        prev_y = wy
    d += f" L {pt(10, 0)[0]:.1f} {prev_y:.1f}"
    c.curve(d, tone="warn", width=2.4)
    c.text(pt(10, 0)[0] + 6, prev_y + 4, "watermark", "small t-warn", anchor="start")
    for a, e in events:
        c.circle(*pt(a, e), 5, role="accent")
    lx, ly = pt(8.4, 3)
    c.circle(lx, ly, 6, role="bad")
    c.text(lx, ly + 24, "late event", "small t-bad", anchor="middle")

    c.box(620, 90, 236, 56, "Event time", "when it happened (features)")
    c.box(620, 158, 236, 56, "Processing time", "when it arrived: network, retries")
    c.box(620, 226, 236, 70, "Watermark W", "max event time seen − allowed lateness; "
          "window closes when W ≥ its end", role="warn")
    takeaway(c, 356, "More allowed lateness: fewer late events, but later results "
                     "and more open windows.")
    return "l2-event-time-watermark.svg", c.render(
        "A scatter of events with arrival time on the horizontal axis and event time on "
        "the vertical axis, all below the no-delay diagonal. The watermark is a staircase "
        "that trails the maximum event time seen by the allowed lateness. One event that "
        "happened early but arrived late falls below the watermark: it is late."
    ), "Event time vs processing time, the watermark staircase, and a late event below it."


@diagram
def stream_windows():
    c = Canvas(W, 400)
    c.title("Tumbling, sliding and session windows",
            "The same ten events, grouped three ways.")
    x0, s = 150, 44

    def tx(t):
        return x0 + t * s

    events = [1, 2, 2.5, 4.5, 7, 7.5, 8.5, 11, 11.5, 14.5]
    c.rule(tx(0), 96, tx(16), 96)
    for t in range(0, 17, 4):
        c.text(tx(t), 84, f"{t}m", "small", anchor="middle")
    for t in events:
        c.circle(tx(t), 96, 5, role="accent")
    c.text(24, 100, "events", "lbl", anchor="start")

    def count(lo, hi):
        return sum(lo <= t < hi for t in events)

    # Tumbling: size 4.
    c.text(24, 140, "Tumbling", "lbl", anchor="start")
    c.text(24, 156, "size 4, no overlap", "small", anchor="start")
    for lo in range(0, 16, 4):
        c.box(tx(lo) + 2, 126, s * 4 - 4, 36, f"{count(lo, lo + 4)} event" + ("" if count(lo, lo + 4) == 1 else "s"), role="accent")
    # Sliding: size 8, hop 4 -> each event in two windows; staggered rows.
    c.text(24, 200, "Sliding", "lbl", anchor="start")
    c.text(24, 216, "size 8, hop 4", "small", anchor="start")
    for k, lo in enumerate(range(-4, 16, 4)):
        lo_c, hi_c = max(lo, 0), min(lo + 8, 16)
        y = 180 if k % 2 == 0 else 214
        role = "muted" if lo < 0 or lo + 8 > 16 else "accent"
        c.box(tx(lo_c) + 2, y, (hi_c - lo_c) * s - 4, 28, f"{count(lo, lo + 8)}", role=role)
    # Session: gap 2.
    c.text(24, 276, "Session", "lbl", anchor="start")
    c.text(24, 292, "split at a gap > 2", "small", anchor="start")
    sessions, cur = [], [events[0]]
    for t in events[1:]:
        if t - cur[-1] > 2:
            sessions.append(cur)
            cur = [t]
        else:
            cur.append(t)
    sessions.append(cur)
    for sess in sessions:
        lo, hi = sess[0], sess[-1]
        w = max((hi - lo) * s, 0) + 32
        c.box(tx(lo) - 16, 266, w, 36, f"{len(sess)}", role="good")
    takeaway(c, 330, "Sliding windows count each event size/hop times; "
                     "session windows have data-dependent lengths.")
    return "l2-stream-windows.svg", c.render(
        "Ten events on a timeline grouped three ways: tumbling windows of 4 minutes with "
        "no overlap; sliding windows of 8 minutes advancing every 4, so each event falls "
        "in two windows; and session windows that close after a gap longer than 2 minutes "
        "and so have data-dependent lengths."
    ), "Tumbling, sliding (hopping) and session windows over the same events."


# ---------------------------------------------------------------------------
# Notebook 05
# ---------------------------------------------------------------------------


@diagram
def position_bias_ips():
    c = Canvas(W, 400)
    c.title("Position bias and its inverse",
            "Examination θ_p = (1 + p)^−0.9, and the IPS weight 1/θ_p that undoes it.")
    theta = [(1 + p) ** -0.9 for p in range(10)]
    base = 280

    # Left: theta_p bars.
    c.text(24, 86, "P(examined | position p)", "lbl", anchor="start")
    for p, t in enumerate(theta):
        h = t * 160
        c.box(30 + p * 38, base - h, 28, h, role="accent", radius=3)
        c.text(44 + p * 38, base + 18, str(p), "small", anchor="middle")
    c.text(44, base - 166, "1.00", "small", anchor="middle")
    c.text(386, base - 26, "0.13", "small", anchor="middle")
    c.rule(24, base, 410, base)
    c.text(24, base + 38, "relevance is flat on random traffic; CTR drops ~8×",
           "small", anchor="start")

    # Right: 1/theta_p bars with a clip line at M = 4.
    c.text(470, 86, "IPS weight 1/θ_p (clip at M = 4)", "lbl", anchor="start")
    unit, clip = 20, 4
    for p, t in enumerate(theta):
        wgt = 1 / t
        x = 476 + p * 38
        kept = min(wgt, clip) * unit
        c.box(x, base - kept, 28, kept, role="warn", radius=3)
        if wgt > clip:
            extra = (wgt - clip) * unit
            c.box(x, base - kept - extra, 28, extra, role="ghost", radius=3)
        c.text(x + 14, base + 18, str(p), "small", anchor="middle")
    cy = base - clip * unit
    c.raw(f'<line x1="470" y1="{cy}" x2="856" y2="{cy}" stroke="#cf222e" '
          f'stroke-width="1.6" stroke-dasharray="6 4"/>')
    c.text(470, cy - 6, "M = 4", "small t-bad", anchor="start")
    c.text(818, base - 7.94 * unit - 6, "7.9", "small", anchor="middle")
    c.rule(470, base, 856, base)
    c.text(470, base + 38, "dashed part is cut by the clip: less variance, more bias",
           "small", anchor="start")
    takeaway(c, 336, "E[click / θ_p] = θ_p · r / θ_p = r: unbiased, "
                     "but noisy where θ_p is small.")
    return "l2-position-bias-ips.svg", c.render(
        "Left: bars of the examination probability by position, from 1.0 at position 0 "
        "down to 0.13 at position 9. Right: the inverse propensity weights 1/theta, from 1 "
        "up to about 7.9, with a clip at 4 cutting the weights of positions 4 to 9."
    ), "Examination probability by position, and the inverse-propensity weights (with clipping) that correct it."


# ---------------------------------------------------------------------------
# Notebook 06
# ---------------------------------------------------------------------------


@diagram
def ltr_objectives():
    c = Canvas(W, 420)
    c.title("Pointwise, pairwise, listwise",
            "Three losses for the same request of 10 items; only the order inside it matters.")
    cols = [
        ("Pointwise", "logistic regression, GBDT classifier",
         "each item alone: predict its click", "log loss per item",
         "calibrated; blind to the group", "default"),
        ("Pairwise (RankNet)", "every pair with y_i > y_j",
         "push s_i above s_j", "log(1 + e^−(s_i − s_j))",
         "a swap at 9/10 costs as much as at 1/2", "default"),
        ("Listwise (LambdaMART)", "RankNet pair gradients",
         "× |ΔNDCG| of swapping i and j", "λ_ij weighted by the metric",
         "top mistakes get big gradients", "accent"),
    ]
    for k, (head, what, idea, loss, note, role) in enumerate(cols):
        x = 24 + k * 284
        c.box(x, 80, 264, 44, head, role=role)
        c.box(x, 136, 264, 66, idea, what, role=role)
        c.box(x, 214, 264, 40, loss, mono=True, role="ghost")
        c.text(x + 132, 274, note, "small", anchor="middle")
    c.rule(24, 296, 856, 296)
    c.text(24, 320, "Why |ΔNDCG|: move the one relevant item from rank 2 to rank 1 → "
                    "ΔNDCG = 1 − 1/log2(3) = 0.37;", "small", anchor="start")
    c.text(24, 338, "from rank 10 to rank 9 → 1/log2(10) − 1/log2(11) = 0.01. "
                    "Same pair loss, 30× different weight.", "small", anchor="start")
    takeaway(c, 360, "LambdaMART = LambdaRank gradients + boosted trees: "
                     "a smooth stand-in for NDCG.")
    return "l2-ltr-objectives.svg", c.render(
        "Three columns. Pointwise: predict each item's click with log loss; calibrated "
        "but blind to the group. Pairwise RankNet: for pairs with different labels, "
        "minimise log(1 + exp(-(s_i - s_j))); all swaps cost the same. Listwise "
        "LambdaMART: RankNet gradients weighted by the change in NDCG from swapping the "
        "pair, so mistakes at the top dominate."
    ), "Pointwise vs pairwise (RankNet) vs listwise (LambdaMART), and why LambdaRank weights pairs by ΔNDCG."


# ---------------------------------------------------------------------------
# Notebooks 07 and 08
# ---------------------------------------------------------------------------


@diagram
def retrieval_cascade():
    c = Canvas(W, 360)
    c.title("The two-stage recommender cascade",
            "Each stage sees fewer items and can afford a richer model per item.")
    stages = [
        ("Catalogue", "10^6 – 10^9 items", 180, "default"),
        ("Retrieval", "several cheap generators, ~10 ms", 150, "accent"),
        ("Candidates", "10^2 – 10^3, merged by quota", 120, "default"),
        ("Ranking", "rich features, ~20–50 ms", 96, "accent"),
        ("Rules", "diversity, dedupe, business", 76, "warn"),
        ("Shown", "top 10–50, logged", 64, "good"),
    ]
    x, mid, w, gap = 24, 170, 122, 20
    for i, (label, sub, h, role) in enumerate(stages):
        c.box(x, mid - h / 2, w, h, label, sub, role=role)
        if i < len(stages) - 1:
            c.arrow(x + w + 2, mid, x + w + gap - 2, mid, accent=True)
        x += w + gap
    c.text(166, 280, "judged by recall@K", "small t-accent", anchor="middle")
    c.text(166, 296, "(did the good items survive?)", "small", anchor="middle")
    c.text(450, 280, "judged by NDCG@k", "small t-accent", anchor="middle")
    c.text(450, 296, "(is the best on top?)", "small", anchor="middle")
    takeaway(c, 310, "An item retrieval misses can never be shown: no ranker can recover it.")
    return "l2-retrieval-cascade.svg", c.render(
        "A left-to-right funnel: a catalogue of millions to billions of items, retrieval "
        "with several cheap generators in about 10 ms, hundreds to thousands of candidates, "
        "a ranking model with rich features, business and diversity rules, and the 10 to 50 "
        "items shown and logged. Retrieval is judged by recall at K, ranking by NDCG."
    ), "Catalogue → retrieval → candidates → ranking → rules → shown, with the metric that judges each stage."


@diagram
def matrix_factorization():
    c = Canvas(W, 360)
    c.title("Matrix factorisation for implicit feedback",
            "A sparse click matrix approximated by user vectors times item vectors.")
    cell = 26
    clicks = {(0, 1), (0, 4), (1, 0), (1, 1), (2, 5), (2, 7), (3, 2), (3, 6), (4, 4),
              (4, 1), (5, 3), (5, 7)}
    rx, ry = 40, 104
    for r in range(6):
        for k in range(8):
            role = "accent" if (r, k) in clicks else "ghost"
            c.box(rx + k * cell, ry + r * cell, cell - 3, cell - 3, role=role, radius=3)
    c.text(rx + 4 * cell, ry - 12, "R: users × items (clicks)", "small", anchor="middle")
    c.text(rx + 4 * cell, ry + 6 * cell + 18, "blank = not clicked, maybe never seen",
           "small", anchor="middle")
    c.text(282, ry + 3 * cell + 8, "≈", "title", anchor="middle")
    ux = 310
    for r in range(6):
        for k in range(2):
            c.box(ux + k * cell, ry + r * cell, cell - 3, cell - 3, role="good", radius=3)
    c.text(ux + cell, ry - 12, "X: k = 2", "small", anchor="middle")
    c.text(ux + cell, ry + 6 * cell + 18, "x_u per user", "small", anchor="middle")
    c.text(386, ry + 3 * cell + 8, "×", "title", anchor="middle")
    yx = 410
    for r in range(2):
        for k in range(8):
            c.box(yx + k * cell, ry + 52 + r * cell, cell - 3, cell - 3, role="warn", radius=3)
    c.text(yx + 4 * cell, ry + 40, "Yᵀ: y_i per item", "small", anchor="middle")
    c.box(660, 96, 196, 66, "score(u, i) = x_u · y_i", "precomputable, indexable",
          role="accent")
    c.box(660, 176, 196, 76, "ALS: alternate", "fix Y, solve every x_u in closed form; "
          "then fix X", role="default")
    takeaway(c, 296, "Implicit ALS: preference p_ui ∈ {0, 1} with confidence "
                     "c_ui = 1 + α·r_ui on every cell.")
    return "l2-matrix-factorization.svg", c.render(
        "A sparse 6 by 8 user-item click matrix approximated by a 6 by 2 user factor "
        "matrix times a 2 by 8 item factor matrix. The score of a user-item pair is the dot "
        "product of their vectors. ALS alternates between solving for the user vectors and "
        "the item vectors in closed form."
    ), "Matrix factorisation: the click matrix ≈ user vectors × item vectors; ALS alternates closed-form solves."


@diagram
def two_tower_in_batch():
    c = Canvas(W, 400)
    c.title("Two towers, trained with in-batch negatives",
            "B (user, clicked item) pairs give a B × B score matrix for free.")
    c.box(24, 90, 140, 60, "User features", "id, last 50 clicks, device")
    c.arrow(166, 120, 196, 120, accent=True)
    c.box(200, 90, 120, 60, "User tower", "MLP → L2-norm", role="accent")
    c.box(24, 220, 140, 60, "Item features", "id, category, price")
    c.arrow(166, 250, 196, 250, accent=True)
    c.box(200, 220, 120, 60, "Item tower", "MLP → L2-norm", role="accent")
    c.path("M 322 120 L 350 120 L 350 170 L 376 170", accent=True)
    c.path("M 322 250 L 350 250 L 350 200 L 376 200", accent=True)
    c.box(380, 155, 110, 60, "u · v / τ", "τ = 0.05", role="accent")
    c.arrow(492, 185, 552, 185, accent=True)
    cell, mx, my = 40, 560, 86
    for i in range(5):
        for j in range(5):
            role = "good" if i == j else "ghost"
            c.box(mx + j * cell, my + i * cell, cell - 4, cell - 4, role=role, radius=4)
    c.text(mx + 98, my + 5 * cell + 16, "rows: users b · columns: items j", "small",
           anchor="middle")
    c.text(780, 106, "diagonal =", "small t-good", anchor="start")
    c.text(780, 122, "positive", "small t-good", anchor="start")
    c.text(780, 186, "rest =", "small", anchor="start")
    c.text(780, 202, "negatives", "small", anchor="start")
    c.text(780, 266, "− log q_j", "small t-warn", anchor="start")
    c.text(780, 282, "per column", "small t-warn", anchor="start")
    takeaway(c, 336, "Popular items are negatives more often: subtract log q_j "
                     "in training only (logQ).", role="warn")
    return "l2-two-tower-in-batch.svg", c.render(
        "A user tower and an item tower each map features to an L2-normalised vector; "
        "the score is their dot product divided by a temperature. For a batch of B "
        "user-item pairs the B by B score matrix has the positives on the diagonal and "
        "every other item in the row as a negative. The logQ correction subtracts the log "
        "sampling probability of each column's item during training."
    ), "Two-tower model and the B × B in-batch softmax: diagonal positives, off-diagonal free negatives, logQ correction."


@diagram
def ann_tradeoff():
    c = Canvas(W, 420)
    c.title("ANN search: recall against latency",
            "Each point is one knob setting; 100,000 items, real user-vector queries.")
    ox, oy, pw, ph = 80, 320, 460, 220

    def pt(ms, rec):
        lx = (math.log10(ms) - math.log10(0.008)) / (math.log10(0.6) - math.log10(0.008))
        return ox + lx * pw, oy - (rec - 0.4) / 0.62 * ph

    c.arrow(ox, oy, ox + pw + 10, oy)
    c.arrow(ox, oy, ox, oy - ph - 16)
    c.text(ox + pw + 10, oy + 22, "latency per query (log scale) →", "small", anchor="end")
    c.text(ox - 8, oy - ph - 22, "recall@100 vs exact ↑", "small", anchor="start")
    for rec in (0.5, 0.75, 1.0):
        _, y = pt(0.01, rec)
        c.text(ox - 8, y + 4, f"{rec:.2f}", "small", anchor="end")
    ivf = [(0.013, 0.50, "nprobe 1"), (0.018, 0.92, "4"), (0.027, 0.997, "16")]
    hnsw = [(0.02, 0.63, "ef 32"), (0.045, 0.90, "128"), (0.14, 0.99, "512")]
    for pts, tone, role, dy in ((ivf, "accent", "accent", -10), (hnsw, "warn", "warn", 18)):
        d = " ".join(("M" if k == 0 else "L") + f" {pt(m, r)[0]:.1f} {pt(m, r)[1]:.1f}"
                     for k, (m, r, _) in enumerate(pts))
        c.curve(d, tone=tone)
        for m, r, lab in pts:
            x, y = pt(m, r)
            c.circle(x, y, 5, role=role)
            c.text(x + 8, y + dy, lab, f"small t-{tone}", anchor="start")
    ex, ey = pt(0.4, 1.0)
    c.circle(ex, ey, 6, role="good")
    c.text(ex, ey - 12, "exact", "small t-good", anchor="middle")
    c.box(600, 90, 256, 70, "IVF · blue", "k-means cells; scan the nprobe nearest cells",
          role="accent")
    c.box(600, 172, 256, 70, "HNSW · amber", "graph of item neighbours; greedy beam of width "
          "efSearch", role="warn")
    c.box(600, 254, 256, 56, "Exact (IndexFlatIP)", "every dot product: the recall reference",
          role="good")
    takeaway(c, 356, "Benchmark ANN with your real queries: user vectors are not "
                     "item vectors.")
    return "l2-ann-tradeoff.svg", c.render(
        "Recall at 100 against query latency on a log scale for 100,000 items. IVF climbs "
        "from 0.5 recall at nprobe 1 to 0.997 at nprobe 16 while staying far faster than "
        "exact search. HNSW needs efSearch 512 to reach 0.99 on user-vector queries, at a "
        "higher latency. Exact search is the slowest point at recall 1."
    ), "ANN recall–latency curves for IVF (nprobe) and HNSW (efSearch) against exact search."


# ---------------------------------------------------------------------------
# Notebook 09
# ---------------------------------------------------------------------------


@diagram
def training_loop_checkpoint():
    c = Canvas(W, 380)
    c.title("One training step, and what a checkpoint must hold",
            "Resume is exact only if every piece of training state comes back.")
    steps = [
        (24, 110, "Batch", "seeded sampler", "default"),
        (154, 120, "Forward", "model.train()", "accent"),
        (294, 120, "Loss", "BCEWithLogits", "accent"),
        (434, 130, "Backward", "zero_grad first", "accent"),
        (584, 110, "Clip", "norm ≤ 1.0", "warn"),
        (714, 142, "Step", "optimizer + scheduler", "good"),
    ]
    y, h = 86, 60
    for i, (x, w, label, sub, role) in enumerate(steps):
        c.box(x, y, w, h, label, sub, role=role)
        if i < len(steps) - 1:
            nx = steps[i + 1][0]
            c.arrow(x + w + 2, y + h / 2, nx - 2, y + h / 2, accent=True)
    c.path("M 785 148 L 785 180 L 79 180 L 79 150", accent=True)
    c.text(432, 172, "next batch · every epoch: eval() + no_grad(), then train() again", "small")
    c.box(24, 206, 404, 92, "Full checkpoint",
          "model · optimizer (Adam m, v) · scheduler · torch, NumPy and sampler RNG states · "
          "epoch → resumed run equals the uninterrupted one, bitwise", role="good")
    c.box(452, 206, 404, 92, "Weights only",
          "warmup restarts at peak LR, Adam moments start empty, the shuffle order "
          "changes → the curve departs", role="bad")
    takeaway(c, 318, "Save atomically: write a temporary file, then rename it.")
    return "l2-training-loop-checkpoint.svg", c.render(
        "The training step: batch, forward in train mode, loss, zero_grad and backward, "
        "gradient clipping, optimizer and scheduler step, then the next batch. A full "
        "checkpoint stores model, optimizer moments, scheduler, all RNG states and the "
        "epoch, so resume is bitwise exact; a weights-only checkpoint restarts warmup with "
        "empty Adam moments and a new shuffle."
    ), "The PyTorch training step, and a full checkpoint versus a weights-only one."


@diagram
def debug_flowchart():
    c = Canvas(W, 380)
    c.title("The loss won't go down: cheapest check first",
            "Each check rules out a whole class of bugs before you touch hyper-parameters.")
    checks = [
        ("Step-0 loss OK?", "≈ base-rate entropy, or log C"),
        ("Overfit one batch?", "loss → ~0 on a fixed batch"),
        ("Gradient norms?", "per layer, at step 0"),
    ]
    fails = [
        ("Labels / loss / scale", "wrong dtype, softmax twice, bad bias init"),
        ("Model or loop bug", "detached tensor, frozen layer, no step"),
        ("Graph or numerics", "0: broken graph · NaN or huge: LR, op, inputs"),
    ]
    x, w, gap = 24, 176, 44
    for (q, qs), (f, fs) in zip(checks, fails):
        c.box(x, 86, w, 60, q, qs, role="accent")
        c.arrow(x + w + 2, 116, x + w + gap - 2, 116, accent=True, label="yes")
        c.arrow(x + w / 2, 148, x + w / 2, 190, label="no", label_dy=4)
        c.box(x, 194, w, 76, f, fs, role="bad")
        x += w + gap
    c.box(x, 86, 856 - x, 60, "Then tune", "LR sweep, data checks, capacity", role="good")
    takeaway(c, 300, "Know the step-0 number: −p log p − (1−p) log(1−p) "
                     "= 0.40 for p = 0.137.", role="warn")
    return "l2-debug-flowchart.svg", c.render(
        "A left-to-right flowchart. First check the loss at step 0 against the base-rate "
        "entropy; if wrong, suspect labels, loss or output scale. Then overfit one batch; "
        "if it cannot, suspect the model or loop. Then inspect gradient norms per layer; "
        "zero means a broken graph, NaN or huge means numerics. Only then tune the learning "
        "rate, data and capacity."
    ), "Debugging a model that won't learn: step-0 loss, overfit one batch, gradient norms, then tune."


# ---------------------------------------------------------------------------
# Notebook 10
# ---------------------------------------------------------------------------


@diagram
def causal_padding_mask():
    c = Canvas(W, 480)
    c.title("Causal + padding mask for a left-padded sequence",
            "Rows: query position i. Columns: key position j. Green = may attend.")
    toks = ["PAD", "PAD", "i17", "i4", "i9", "i2"]
    n, cell, gx, gy = 6, 46, 110, 110
    for j, t in enumerate(toks):
        c.text(gx + j * cell + cell / 2 - 2, gy - 10, t, "small", anchor="middle")
    for i, t in enumerate(toks):
        c.text(gx - 12, gy + i * cell + cell / 2 + 2, t, "small", anchor="end")
        for j in range(n):
            pad_q, pad_k = i < 2, j < 2
            if j > i:
                role = "ghost"
            elif pad_q:
                role = "warn" if i == j else "bad"
            else:
                role = "bad" if pad_k else "good"
            c.box(gx + j * cell, gy + i * cell, cell - 4, cell - 4, role=role, radius=4)
    c.text(gx + 3 * cell, gy + n * cell + 18, "key j →", "small", anchor="middle")
    legend = [
        ("good", "may attend", "j ≤ i and key is a real item"),
        ("ghost", "causal block", "j > i: no peeking at the future"),
        ("bad", "padding block", "key is PAD"),
        ("warn", "unblocked diagonal", "a PAD row would be all −∞ → NaN"),
    ]
    for k, (role, head, sub) in enumerate(legend):
        y = 100 + k * 64
        c.box(440, y, 30, 30, role=role, radius=4)
        c.text(484, y + 13, head, "lbl", anchor="start")
        c.text(484, y + 30, sub, "small", anchor="start")
    c.box(440, 360, 416, 30, "softmax(QKᵀ/√d + M) V,  M = −∞ where blocked", mono=True,
          role="ghost")
    takeaway(c, 420, "Test the mask: changing future items must not change "
                     "earlier outputs.")
    return "l2-causal-padding-mask.svg", c.render(
        "A 6 by 6 attention mask for a sequence left-padded with two PAD tokens. Real "
        "query positions may attend to real keys at or before them. Keys after the query "
        "are blocked by the causal mask, PAD keys are blocked by the padding mask, and the "
        "diagonal of the PAD rows is unblocked so no row is entirely minus infinity."
    ), "Causal plus padding attention mask for a left-padded sequence, with the unblocked diagonal that avoids NaN."
