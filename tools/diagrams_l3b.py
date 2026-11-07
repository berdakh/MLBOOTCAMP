"""Diagrams for l3b (Level 3 notebooks 11-20) - see tools/make_diagrams.py for the conventions.

Every file here is named ``l3-x-*.svg`` so it never collides with the figures
for notebooks 01-10 (``diagrams_l3a.py``).
"""

from __future__ import annotations

import math

from svgkit import Canvas, diagram


def _takeaway(c: Canvas, y: float, text: str, sub: str = "", x0: float = 24, x1: float | None = None) -> None:
    """A rule and one bold takeaway line (plus an optional small line) at the bottom."""
    x1 = c.width - 24 if x1 is None else x1
    c.rule(x0, y, x1, y)
    c.text(x0, y + 24, text, "lbl", anchor="start")
    if sub:
        c.text(x0, y + 43, sub, "small", anchor="start")


def _axes(c: Canvas, x0: float, y0: float, x1: float, y1: float, xlabel: str, ylabel: str) -> None:
    """Plot axes with the origin at (x0, y0) (bottom left) and the top right at (x1, y1)."""
    c.path(f"M {x0} {y0} L {x1} {y0}", arrow=True)
    c.path(f"M {x0} {y0} L {x0} {y1}", arrow=True)
    c.text((x0 + x1) / 2, y0 + 24, xlabel, "small")
    c.text(x0 - 10, y1 + 4, ylabel, "small", anchor="end")


def _poly(points) -> str:
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


# ---------------------------------------------------------------------------
# 11 - quantisation and distillation
# ---------------------------------------------------------------------------


@diagram
def int8_quantization():
    c = Canvas(880, 450)
    c.title("Symmetric int8 quantisation",
            "q = round(x / s) with s = max|x| / 127; dequantise with x̂ = q · s; error ≤ s / 2")

    # Float axis and integer grid.
    x0, x1 = 120, 760
    c.text(x0 - 16, 104, "float x", "small", anchor="end")
    c.text(x0 - 16, 174, "int8 q", "small", anchor="end")
    c.rule(x0, 100, x1, 100)
    c.rule(x0, 170, x1, 170)
    steps = 16
    for i in range(steps + 1):
        x = x0 + (x1 - x0) * i / steps
        c.rule(x, 164, x, 176)
    for x, lab in [(x0, "−max|x|"), ((x0 + x1) / 2, "0"), (x1, "+max|x|")]:
        c.rule(x, 94, x, 106)
        c.text(x, 88, lab, "small")
    for x, lab in [(x0, "−127"), ((x0 + x1) / 2, "0"), (x1, "+127")]:
        c.text(x, 194, lab, "small")
    # A few values snapping to their nearest grid point.
    step = (x1 - x0) / steps
    for xv in (205, 492, 668):
        snapped = x0 + round((xv - x0) / step) * step
        c.circle(xv, 100, 4, role="accent")
        c.arrow(xv, 106, snapped, 162, accent=True)
    c.text(x1 + 14, 140, "÷ s, round", "small", anchor="start")

    # Per-tensor vs per-channel.
    rows = [("row 1", 1.0), ("row 2", 0.45), ("row 3", 0.2)]
    for px, title, per_channel in [(40, "Per-tensor: one scale", False), (470, "Per-channel: one scale per row", True)]:
        c.text(px, 236, title, "lbl", anchor="start")
        cx, half = px + 210, 150
        for i, (name, rng) in enumerate(rows):
            y = 254 + i * 34
            c.text(px, y + 13, name, "small", anchor="start")
            w = half * rng
            c.box(cx - w, y, 2 * w, 18, role="good" if per_channel else ("bad" if rng < 0.5 else "accent"),
                  radius=4)
            spacing = (w if per_channel else half) / 4
            k = 0
            while k * spacing <= w + 0.1:
                for sign in (-1, 1):
                    c.rule(cx + sign * k * spacing, y - 3, cx + sign * k * spacing, y + 21)
                k += 1
        note = ("small rows get only 1-2 levels" if not per_channel else "every row uses its full grid")
        c.text(px, 368, note, "small t-bad" if not per_channel else "small t-good", anchor="start")

    _takeaway(c, 390, "Per-channel int8 is the safe default; int4 needs per-group scales (64-128 weights each).",
              "Tiny example: x = 0.21, max|x| = 0.35 → s = 0.00276, q = round(76.2) = 76, x̂ = 0.2094, error 0.0006 ≤ s/2.")
    return "l3-x-int8-quantization.svg", c.render(
        "Symmetric int8 quantisation. Real values on a float axis snap to the nearest point of a 255-level "
        "integer grid from -127 to 127 via q = round(x/s). Below, per-tensor quantisation uses one step size "
        "for all rows so rows with a small range get only one or two levels, while per-channel quantisation "
        "gives each row its own scale so every row uses the full grid."
    ), "Symmetric int8 quantisation, and why per-channel scales beat one per-tensor scale."


@diagram
def distillation():
    c = Canvas(880, 470)
    c.title("Knowledge distillation",
            "The student learns from the teacher's softened probabilities and from the true label.")
    c.box(30, 160, 120, 60, "Input text", "labelled or not")
    c.box(190, 90, 150, 60, "Teacher", "large, frozen")
    c.box(190, 230, 150, 60, "Student", "small, trained", role="accent")
    c.arrow(150, 180, 186, 128)
    c.arrow(150, 200, 186, 252, accent=True)
    c.box(390, 90, 170, 60, "Soft targets", "softmax(z_t / T)", role="good")
    c.box(390, 230, 170, 60, "Student logits z_s", "softmax(z_s / T)", role="accent")
    c.box(390, 340, 170, 50, "True label y", "if known")
    c.arrow(340, 120, 386, 120)
    c.arrow(340, 260, 386, 260, accent=True)
    c.box(610, 160, 130, 60, "KL term", "× α · T²", role="warn")
    c.box(610, 300, 130, 60, "CE term", "× (1 − α)", role="warn")
    c.arrow(560, 128, 606, 172)
    c.arrow(560, 250, 606, 206, accent=True)
    c.arrow(560, 274, 606, 316, accent=True)
    c.arrow(560, 362, 606, 344)
    c.box(780, 220, 80, 80, "Loss", "sum", role="accent")
    c.arrow(740, 196, 776, 240)
    c.arrow(740, 324, 776, 284)
    _takeaway(c, 410, "L = α · T² · KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1 − α) · CE(z_s, y)",
              "T > 1 exposes the small probabilities (“dark knowledge”); T² keeps the soft term's gradient "
              "size independent of T.")
    return "l3-x-distillation.svg", c.render(
        "Knowledge distillation. The same input goes to a frozen teacher and a trainable student. The teacher's "
        "logits softened by temperature T give soft targets; the student's softened logits are compared with them "
        "by a KL term weighted alpha times T squared, and the student's logits are compared with the true label "
        "by a cross-entropy term weighted one minus alpha. The two terms are summed into the loss."
    ), "Knowledge distillation: a KL term on softened teacher probabilities plus cross-entropy on the label."


# ---------------------------------------------------------------------------
# 12 - high-throughput serving
# ---------------------------------------------------------------------------


@diagram
def dynamic_batching():
    c = Canvas(880, 440)
    c.title("A dynamic batcher",
            "Collect requests until the batch is full or the first one has waited long enough.")
    y = 80
    c.box(30, y, 120, 64, "Clients", "one request each")
    c.box(180, y, 120, 64, "Queue", "(input, future)", role="accent")
    c.box(330, y, 230, 64, "Batch closes when", "32 items OR 2 ms after the first", role="warn")
    c.box(590, y, 120, 64, "Model", "one call on B", role="accent")
    c.box(740, y, 120, 64, "Split", "row i → caller i", role="good")
    c.arrow(150, y + 32, 176, y + 32, accent=True)
    c.arrow(300, y + 32, 326, y + 32, accent=True)
    c.arrow(560, y + 32, 586, y + 32, accent=True)
    c.arrow(710, y + 32, 736, y + 32, accent=True)
    c.path(f"M 800 {y + 64} L 800 174 L 90 174 L 90 {y + 68}", accent=False)
    c.text(452, 168, "each awaiting caller receives its own answer", "small")

    # Timeline of arrivals and one batching window.
    ty = 290
    c.text(30, 226, "One batch on a timeline", "lbl", anchor="start")
    c.box(150, 244, 260, 64, role="ghost")
    c.text(280, 260, "max_wait window", "small")
    c.rule(60, ty, 820, ty)
    for x in (150, 196, 255, 300, 372):
        c.circle(x, ty, 6, role="accent")
    for x in (600, 680, 760):
        c.circle(x, ty, 6, role="muted")
    c.text(150, ty + 26, "first arrival t0", "small")
    c.text(418, ty + 26, "t0 + max_wait: run B = 5", "small t-warn", anchor="start")
    c.rule(410, 240, 410, 312)
    c.text(680, ty - 16, "later arrivals: next batch", "small")

    _takeaway(c, 350, "Under light load the wait is pure added latency; under heavy load batches fill instantly.",
              "Set max_wait to a small fraction of the latency SLO; set max_batch to what the hardware runs within the SLO.")
    return "l3-x-dynamic-batching.svg", c.render(
        "A dynamic batcher. Clients put requests with a future on a queue; the batcher closes a batch when it "
        "holds 32 items or 2 milliseconds have passed since the first one, runs the model once on the batch, and "
        "splits the results so each caller receives its own answer. A timeline shows five requests collected "
        "within one max-wait window and later arrivals starting the next batch."
    ), "Dynamic batching: a queue, a max batch size and a max wait, one model call per batch."


@diagram
def throughput_latency():
    c = Canvas(880, 430)
    c.title("The throughput–latency curve",
            "Latency stays flat, then explodes as load approaches a replica's capacity.")
    x0, y0, x1, y1 = 80, 330, 560, 80
    _axes(c, x0, y0, x1, y1, "throughput (requests / s)", "p95 ms")

    def to_px(rps, ms):
        return x0 + rps / 360 * (x1 - x0 - 20), y0 - ms / 160 * (y0 - y1 - 10)

    for cap, base, tone in [(240, 8.0, "muted_fg"), (320, 11.0, "accent")]:
        pts = []
        for i in range(0, 200):
            rps = cap * i / 200
            ms = base / (1 - rps / cap)
            if ms > 160:
                break
            pts.append(to_px(rps, ms))
        c.curve(_poly(pts), tone=tone)
    slo = 60
    _, ys = to_px(0, slo)
    c.curve(f"M {x0} {ys:.1f} L {x1 - 20} {ys:.1f}", tone="warn", dashed=True, width=1.6)
    c.text(x1 - 20, ys - 8, "SLO: p95 ≤ 60 ms", "small t-warn", anchor="end")
    c.text(362, 100, "no batching", "small", anchor="end")
    c.text(468, 100, "dynamic batching", "small t-accent", anchor="start")
    for cap, base in [(240, 8.0), (320, 11.0)]:
        rps = cap * (1 - base / slo)
        px, py = to_px(rps, slo)
        c.circle(px, py, 5, role="warn")
        c.text(px, y0 - 8, f"{rps:.0f}", "small")

    bx = 600
    c.box(bx, 90, 250, 70, "Little's law: L = λ · W", "in flight = arrival rate × time",
          role="accent")
    c.box(bx, 180, 250, 70, "300 req/s × 0.05 s = 15", "requests in flight at once")
    c.box(bx, 270, 250, 60, "Capacity = load at the SLO", "the marked points, not the peak", role="warn")
    _takeaway(c, 360, "Batching moves the curve right: more requests per second at the same latency.",
              "Plan capacity at the highest load whose p95 (or p99) still meets the SLO, then add headroom.")
    return "l3-x-throughput-latency.svg", c.render(
        "Throughput against p95 latency for a server without batching and with dynamic batching. Both curves are "
        "flat at low load and rise steeply near capacity; the batching curve sits further right. A dashed SLO line "
        "at 60 ms marks each configuration's usable capacity. Side boxes state Little's law, L equals lambda "
        "times W, with the example 300 requests per second times 0.05 seconds equals 15 requests in flight."
    ), "Throughput vs latency: capacity is the load at which p95 still meets the SLO; Little's law L = λW."


def _gantt(c: Canvas, x0: float, y0: float, rows, n_steps: int, idle_role: str) -> None:
    """rows: per slot, a list of (letter, start, length). Busy cells accent, idle cells idle_role."""
    pitch, cell = 28, 26
    for s, seqs in enumerate(rows):
        busy = {}
        for letter, start, length in seqs:
            for t in range(start, start + length):
                busy[t] = letter if t == start else ""
        for t in range(n_steps):
            x, y = x0 + t * pitch, y0 + s * pitch
            if t in busy:
                c.box(x, y, cell, 24, busy[t], role="accent", radius=4)
            else:
                c.box(x, y, cell, 24, role=idle_role, radius=4)


@diagram
def continuous_batching():
    c = Canvas(880, 380)
    c.title("Static vs continuous batching",
            "4 slots, 8 requests of different output lengths; one column = one decode step.")
    static = [[("a", 0, 3), ("e", 9, 4)], [("b", 0, 9), ("f", 9, 2)],
              [("c", 0, 2), ("g", 9, 5)], [("d", 0, 5), ("h", 9, 3)]]
    cont = [[("a", 0, 3), ("f", 3, 2), ("g", 5, 5)], [("b", 0, 9)],
            [("c", 0, 2), ("e", 2, 4)], [("d", 0, 5), ("h", 5, 3)]]
    c.text(40, 92, "Static: the batch waits for its longest sequence", "lbl", anchor="start")
    _gantt(c, 40, 106, static, 14, "bad")
    c.text(40, 236, "done at step 14 · 33 of 56 slot-steps busy (59%)", "small t-bad", anchor="start")
    c.text(460, 92, "Continuous: refill a slot the step it frees", "lbl", anchor="start")
    _gantt(c, 460, 106, cont, 14, "ghost")
    c.text(460, 236, "done at step 10 · free slots take the next requests", "small t-good", anchor="start")
    c.box(40, 262, 26, 24, role="accent", radius=4)
    c.text(74, 279, "decoding a token", "small", anchor="start")
    c.box(220, 262, 26, 24, role="bad", radius=4)
    c.text(254, 279, "idle slot (padding)", "small", anchor="start")
    c.box(420, 262, 26, 24, role="ghost", radius=4)
    c.text(454, 279, "free for a new request", "small", anchor="start")
    _takeaway(c, 316, "A decode step costs about the same whatever the batch holds, so fewer steps = more throughput.",
              "The notebook's simulation (512 requests, batch 16): 26% vs 97% of slots busy, 3.7× fewer decode steps.")
    return "l3-x-continuous-batching.svg", c.render(
        "Two Gantt charts of four batch slots over fourteen decode steps. With static batching the first batch "
        "runs until its longest sequence ends at step 9, leaving many idle padding cells, and all eight requests "
        "finish at step 14. With continuous batching a waiting request takes a slot the step it frees, and the "
        "same eight requests finish at step 10."
    ), "Static vs continuous batching for generation: refilling slots every step removes idle padding."


@diagram
def paged_kv_cache():
    c = Canvas(880, 400)
    c.title("Paged attention: KV cache in blocks",
            "Reserve-the-maximum wastes memory; fixed-size blocks allocated on demand do not.")
    c.text(40, 92, "Contiguous: reserve max length per request", "lbl", anchor="start")
    for i, (name, used) in enumerate([("A", 130), ("B", 60), ("C", 210)]):
        y = 108 + i * 46
        c.text(40, y + 22, name, "lbl", anchor="start")
        c.box(64, y, used, 32, role="accent", radius=4)
        c.box(64 + used, y, 320 - used, 32, "reserved, unused" if 320 - used > 130 else "", role="bad", radius=4)
    c.text(40, 262, "most of the reservation is never written", "small t-bad", anchor="start")

    c.text(470, 92, "Paged: 16-token blocks + a block table", "lbl", anchor="start")
    c.box(470, 108, 140, 40, "A → 0, 5, 9", role="accent", mono=True)
    c.box(470, 156, 140, 40, "B → 2, 12", role="good", mono=True)
    c.box(470, 204, 140, 40, "C → 1, 6, 7, 14", role="warn", mono=True)
    c.arrow(612, 176, 640, 176)
    owner = {0: "A", 5: "A", 9: "A", 2: "B", 12: "B", 1: "C", 6: "C", 7: "C", 14: "C"}
    roles = {"A": "accent", "B": "good", "C": "warn"}
    for k in range(16):
        r, col = divmod(k, 4)
        x, y = 646 + col * 54, 108 + r * 36
        who = owner.get(k)
        c.box(x, y, 48, 30, f"{who}" if who else str(k), role=roles.get(who, "ghost"), radius=4,
              label_cls="lbl" if who else "small")
    c.text(751, 262, "free blocks serve any sequence", "small t-good")
    _takeaway(c, 300, "KV bytes per token = 2 × layers × KV heads × head dim × bytes per value",
              "GPT-2 small, fp16: 2 × 12 × 12 × 64 × 2 = 36,864 B ≈ 36 KB per token; a 7B model without GQA: 512 KB.")
    c.text(24, 372, "The KV cache, not the weights, limits how many sequences fit; paging lets many more fit "
           "and lets sequences share a common prefix.", "small", anchor="start")
    return "l3-x-paged-kv-cache.svg", c.render(
        "Left: contiguous allocation reserves the maximum length for each of three sequences, and most of each "
        "reservation is never used. Right: paged attention stores each sequence's KV cache in 16-token blocks "
        "scattered across GPU memory, found through a per-sequence block table; free blocks can serve any "
        "sequence. The KV cache per token is 2 times layers times KV heads times head dimension times bytes."
    ), "Paged attention: a block table maps each sequence to fixed-size KV blocks allocated on demand."


# ---------------------------------------------------------------------------
# 13 - CTR prediction
# ---------------------------------------------------------------------------


@diagram
def ctr_dcn():
    c = Canvas(880, 420)
    c.title("A CTR model: hashed embeddings + DCN-v2",
            "Sparse ids become vectors, a cross network learns feature crosses, a sigmoid gives pCTR.")
    c.box(30, 90, 120, 60, "Sparse ids", "26 fields")
    c.box(180, 90, 130, 60, "Stable hash", "of (field, value)", role="warn")
    c.box(340, 90, 120, 60, "Embedding table", "n rows × dim", role="accent")
    c.box(30, 220, 120, 60, "Dense counts", "13 fields")
    c.box(180, 220, 130, 60, "log(1 + x)", "then standardise")
    c.arrow(150, 120, 176, 120, accent=True)
    c.arrow(310, 120, 336, 120, accent=True)
    c.arrow(150, 250, 176, 250)
    c.box(490, 90, 90, 190, "x0", "concat of all vectors", role="accent")
    c.arrow(460, 120, 486, 120, accent=True)
    c.arrow(310, 250, 486, 250)
    c.box(610, 90, 140, 60, "Cross layers", "explicit crosses", role="accent")
    c.box(610, 220, 140, 60, "Deep MLP", "implicit crosses")
    c.arrow(580, 120, 606, 120, accent=True)
    c.arrow(580, 250, 606, 250)
    c.box(780, 120, 80, 130, "σ", "pCTR", role="good")
    c.arrow(750, 130, 776, 150, accent=True)
    c.arrow(750, 240, 776, 222)
    _takeaway(c, 310, "Almost all parameters live in the embedding table (93% in the notebook's model).",
              "Cross layer: x_{l+1} = x0 ⊙ (W x_l + b) + x_l. Each raises the interaction degree by one; + x_l keeps lower orders.")
    c.text(24, 390, "Loss: log loss on clicks. Output bias starts at log(CTR / (1 − CTR)) ≈ −3.5 so the model is "
           "calibrated from step one.", "small", anchor="start")
    return "l3-x-ctr-dcn.svg", c.render(
        "A CTR model. Sparse categorical ids are hashed with a stable hash of field and value modulo n and looked up "
        "in an embedding table; dense counts are log-transformed and standardised. All vectors are concatenated "
        "into x0, which feeds a cross network and a deep MLP in parallel; their outputs are combined and passed "
        "through a sigmoid to give pCTR."
    ), "CTR model architecture: hashing, embedding tables, dense features, DCN-v2 cross layers, sigmoid."


@diagram
def hashing_trick():
    c = Canvas(880, 440)
    c.title("The hashing trick",
            "bucket = hash(field, value) mod n: no vocabulary, fixed memory, and some collisions.")
    keys = [("C1 = nike", 90), ("C2 = nike", 150), ("C1 = adidas", 210), ("C7 = 12345", 270)]
    for label, y in keys:
        c.box(30, y, 150, 44, label, mono=True)
        c.arrow(180, y + 22, 296, 192 + (y - 180) / 6, accent=True)
    c.box(300, 160, 180, 76, "Stable hash", "MurmurHash / xxHash, field in the key", role="accent")
    buckets_y = [86 + i * 34 for i in range(8)]
    targets = {90: 1, 150: 3, 210: 5, 270: 5}
    for i, by in enumerate(buckets_y):
        role = "bad" if i == 5 else ("accent" if i in (1, 3) else "default")
        c.box(620, by, 120, 28, f"bucket {i}", role=role, radius=4, label_cls="small")
    for (_, y), off in zip(keys, (-12, -4, 4, 12)):
        b = targets[y]
        c.arrow(480, 198 + off, 616, buckets_y[b] + 14, accent=True)
    c.text(752, buckets_y[5] + 4, "collision:", "small t-bad", anchor="start")
    c.text(752, buckets_y[5] + 19, "two keys share", "small t-bad", anchor="start")
    c.text(752, buckets_y[5] + 34, "one weight", "small t-bad", anchor="start")
    c.text(105, 336, "C1 = nike ≠ C2 = nike", "small", anchor="middle")
    _takeaway(c, 370, "P(a key collides) = 1 − (1 − 1/n)^(k−1) ≈ 1 − e^(−k/n)",
              "k = 425,677 keys into n = 2^19 = 524,288 buckets: 1 − e^(−0.81) ≈ 56% of keys share a bucket. "
              "Never use Python's salted hash().")
    return "l3-x-hashing-trick.svg", c.render(
        "The hashing trick. Four keys, each a field and value pair, go through a stable hash modulo n into a "
        "table of eight buckets. Including the field keeps C1=nike and C2=nike apart, but C1=adidas and C7=12345 "
        "land in the same bucket and share one weight: a collision. The expected collision rate is about one "
        "minus e to the minus k over n."
    ), "The hashing trick: stable hash of (field, value) into n buckets, and the collision rate 1 − e^(−k/n)."


@diagram
def negative_downsampling():
    c = Canvas(880, 420)
    c.title("Negative downsampling and re-calibration",
            "Keep every click and a fraction w of the non-clicks, then undo the bias exactly.")
    bx, bw = 200, 560
    c.text(30, 104, "Full log", "lbl", anchor="start")
    c.text(30, 120, "100% of rows", "small", anchor="start")
    c.box(bx, 90, bw * 0.03 + 8, 34, role="good", radius=3)
    c.box(bx + bw * 0.03 + 8, 90, bw * 0.97 - 8, 34, "non-clicks 97%", role="muted", radius=3)
    c.text(bx - 6, 112, "clicks 3%", "small t-good", anchor="end")
    c.text(30, 164, "Downsampled", "lbl", anchor="start")
    c.text(30, 180, "w = 0.1 · 12.7% of rows", "small", anchor="start")
    cl = bw * 0.236
    c.box(bx, 150, cl, 34, "clicks 24%", role="good", radius=3)
    c.box(bx + cl, 150, bw - cl, 34, "10% of the non-clicks", role="muted", radius=3)

    y = 230
    c.box(30, y, 170, 66, "Train on sample", "~8× fewer rows", role="accent")
    c.box(240, y, 190, 66, "Raw prediction q", "pred/obs ≈ 6.4", role="bad")
    c.box(470, y, 200, 66, "p = q / (q + (1 − q)/w)", "odds × w", role="warn")
    c.box(710, y, 150, 66, "Calibrated p", "AUC unchanged", role="good")
    c.arrow(200, y + 33, 236, y + 33, accent=True)
    c.arrow(430, y + 33, 466, y + 33, accent=True)
    c.arrow(670, y + 33, 706, y + 33, accent=True)
    _takeaway(c, 326, "Sampling multiplies the odds by 1/w, so multiplying them by w undoes it exactly.",
              "q = 0.236, w = 0.1: p = 0.236 / (0.236 + 0.764/0.1) = 0.236 / 7.876 = 0.030, the true base rate.")
    c.text(24, 394, "Log w with the model: correcting with a different w than training used shifts every prediction.",
           "small t-bad", anchor="start")
    return "l3-x-negative-downsampling.svg", c.render(
        "Negative downsampling. The full log is 3 percent clicks; keeping all clicks and 10 percent of non-clicks "
        "gives a training set that is 24 percent clicks. A model trained on it predicts far too high; the "
        "re-calibration p = q / (q + (1 - q) / w) multiplies the odds by w and restores calibrated probabilities "
        "without changing the ranking."
    ), "Negative downsampling at rate w and the exact re-calibration p = q / (q + (1 − q)/w)."


@diagram
def delayed_feedback():
    c = Canvas(880, 440)
    c.title("Delayed feedback",
            "At training time, recent clicks have not had time to convert, so they look like negatives.")
    x0, x1 = 80, 800
    cut = x0 + (x1 - x0) * 12 / 14
    c.rule(x0, 300, x1, 300)
    for d in range(0, 15, 2):
        x = x0 + (x1 - x0) * d / 14
        c.rule(x, 296, x, 304)
        c.text(x, 320, f"day {d}", "small")
    c.curve(f"M {cut} 76 L {cut} 300", tone="warn", dashed=True, width=1.8)
    c.text(cut, 72, "training cutoff T", "small t-warn")
    rows = [(1.5, 3.5, "converted, seen", "good"), (5, 8.5, "converted, seen", "good"),
            (9.5, 13.2, "converts after T, so labelled 0", "bad"), (11.2, None, "no conversion yet", "muted")]
    for i, (click, conv, note, role) in enumerate(rows):
        y = 100 + i * 46
        cx = x0 + (x1 - x0) * click / 14
        c.circle(cx, y, 6, role="accent")
        if conv is not None:
            vx = x0 + (x1 - x0) * conv / 14
            c.curve(f"M {cx + 6} {y} L {vx - 6} {y}", tone="muted_fg" if role == "good" else "bad",
                    dashed=role == "bad", width=1.8)
            c.circle(vx, y, 6, role=role)
            if role == "bad":
                c.text(cx - 14, y + 4, note, "small t-bad", anchor="end")
            else:
                c.text(vx + 12, y + 4, note, "small t-good", anchor="start")
        else:
            c.text(cx + 14, y + 4, note, "small", anchor="start")
    c.circle(x0 + 6, 266, 6, role="accent")
    c.text(x0 + 18, 270, "click", "small", anchor="start")
    c.circle(x0 + 76, 266, 6, role="good")
    c.text(x0 + 88, 270, "conversion (delay ~ exponential, mean 2 days)", "small", anchor="start")
    _takeaway(c, 344, "Fix 1: wait window (train on clicks ≥ W days old). Fix 2: delay-aware likelihood:",
              "P(y_obs = 1 | x, e) = pCVR(x) · F(e), F(e) = 1 − e^(−e/τ). A 12-hour-old click, τ = 48 h: F = 1 − e^(−0.25) = 22%.")
    c.text(24, 412, "Delayed feedback is mostly a calibration problem: the naive model under-predicts most on the "
           "freshest traffic.", "small", anchor="start")
    return "l3-x-delayed-feedback.svg", c.render(
        "A timeline of clicks and their conversions relative to a training cutoff T. Early clicks have converted "
        "before T and are labelled correctly; a recent click converts after T and is wrongly labelled negative; "
        "another has not converted yet. Two fixes: a wait window, or a delay-aware likelihood that multiplies "
        "pCVR by the delay CDF F of the elapsed time."
    ), "Delayed feedback: conversions after the training cutoff are mislabelled as negatives."


# ---------------------------------------------------------------------------
# 14 - multi-task ranking and exploration
# ---------------------------------------------------------------------------


@diagram
def mmoe():
    c = Canvas(880, 450)
    c.title("Shared-bottom vs MMoE",
            "Two ways for one network to predict clicks and conversions.")
    # Shared-bottom.
    c.text(40, 90, "Shared-bottom", "lbl", anchor="start")
    c.box(140, 106, 140, 50, "Input embeddings")
    c.box(140, 186, 140, 56, "Shared MLP", "one representation", role="accent")
    c.box(60, 280, 130, 56, "CTR tower", "pCTR", role="good")
    c.box(230, 280, 130, 56, "CVR tower", "pCVR", role="good")
    c.arrow(210, 156, 210, 182, accent=True)
    c.arrow(180, 242, 140, 276, accent=True)
    c.arrow(240, 242, 280, 276, accent=True)
    c.text(210, 362, "cheap; tasks fight over one representation", "small t-bad")
    # MMoE.
    c.text(470, 90, "MMoE", "lbl", anchor="start")
    c.box(590, 106, 140, 50, "Input embeddings")
    for i, x in enumerate((480, 610, 740)):
        c.box(x, 186, 110, 46, f"Expert {i + 1}", role="accent")
        c.arrow(660, 156, x + 55, 182, accent=True)
    c.box(480, 252, 370, 30, "per-task softmax gates mix the experts", role="warn", label_cls="small")
    for x in (535, 665, 795):
        c.arrow(x, 232, x, 248)
    c.box(510, 300, 130, 50, "CTR tower", role="good")
    c.box(690, 300, 130, 50, "CVR tower", role="good")
    c.arrow(575, 282, 575, 296)
    c.arrow(755, 282, 755, 296)
    c.text(665, 372, "tasks can use different experts", "small t-good")
    _takeaway(c, 386, "MMoE: y_k = tower_k( Σ_i g_k,i(x) · E_i(x) ), with g_k = softmax(W_k x) for each task k",
              "Worth it only if tasks conflict; in the notebook the gates stayed near uniform (≈ 0.33 each).")
    return "l3-x-mmoe.svg", c.render(
        "Left, shared-bottom: input embeddings feed one shared MLP, which feeds a CTR tower and a CVR tower. "
        "Right, MMoE: input embeddings feed three expert networks; a softmax gate per task mixes the experts "
        "before each task's tower, so tasks that conflict can rely on different experts."
    ), "Shared-bottom vs multi-gate mixture-of-experts (MMoE) for multi-task ranking."


@diagram
def esmm():
    c = Canvas(880, 450)
    c.title("Sample selection bias and ESMM",
            "CVR labels exist only after a click, but pCVR is used on every impression.")
    # Nested populations.
    c.box(30, 86, 360, 260, role="ghost")
    c.text(46, 108, "All impressions: where pCVR is served", "small", anchor="start")
    c.box(70, 130, 280, 190, role="accent")
    c.text(86, 152, "Clicked (~8%): CVR labels exist", "small", anchor="start")
    c.box(120, 190, 180, 100, "Converted", "~15% of clicks", role="good")
    c.text(210, 370, "naive CVR: trained inside, served outside", "small t-bad")
    # ESMM network.
    c.box(440, 86, 170, 56, "Shared embeddings", role="accent")
    c.box(440, 182, 170, 56, "CTR tower", "pCTR")
    c.box(670, 182, 180, 56, "CVR tower", "pCVR")
    c.arrow(525, 142, 525, 178, accent=True)
    c.arrow(610, 120, 756, 178, accent=True)
    c.box(560, 280, 180, 60, "pCTCVR = pCTR × pCVR", role="good", label_cls="small")
    c.arrow(525, 238, 596, 276)
    c.arrow(760, 238, 704, 276)
    c.text(525, 268, "loss: click", "small", anchor="end")
    c.text(650, 366, "loss: click AND convert, on all impressions", "small t-good")
    _takeaway(c, 386, "ESMM trains pCTR and pCTCVR on the entire impression space; pCVR is their ratio, never fitted on clicks alone.",
              "Example: pCTR = 0.08, pCVR = 0.15 → pCTCVR = 0.012 conversions per impression. Weak where pCTR is small.")
    return "l3-x-esmm.svg", c.render(
        "Left: nested populations. All impressions contain the clicked impressions, which contain the converted "
        "ones; a naive CVR model trains on clicked impressions but is served on all of them, which is sample "
        "selection bias. Right: ESMM shares embeddings between a CTR tower and a CVR tower and multiplies them into "
        "pCTCVR, trained with a click loss and a click-and-convert loss on all impressions."
    ), "Sample selection bias in CVR modelling and the ESMM fix: pCTCVR = pCTR × pCVR over all impressions."


@diagram
def second_price_auction():
    c = Canvas(880, 400)
    c.title("A second-price auction with eCPM",
            "Rank by expected revenue per 1,000 impressions; the winner pays just enough to keep winning.")
    ads = [("Ad A", "bid $2.00 · pCTR 3.0%", 60, "good"), ("Ad B", "bid $4.00 · pCTR 1.2%", 48, "accent"),
           ("Ad C", "bid $1.00 · pCTR 4.0%", 40, "default")]
    for i, (name, sub, ecpm, role) in enumerate(ads):
        y = 90 + i * 64
        c.box(30, y, 200, 50, name, sub, role=role)
        c.box(250, y + 8, ecpm * 4.5, 34, role=role, radius=4)
        c.text(250 + ecpm * 4.5 + 10, y + 30, f"eCPM ${ecpm}", "lbl", anchor="start")
    c.text(410, 300, "eCPM = pCTR × bid × 1000", "mono")
    c.box(640, 110, 210, 120, "A pays $1.60 per click", "= 48 / (3.0% × 1000), below its own $2.00 bid",
          role="good")
    _takeaway(c, 320, "price per click = max(reserve, bid₂ · pCTR₂ / pCTR₁)",
              "A uniform calibration error cancels in this ratio; an error for one segment moves wins and money.")
    return "l3-x-second-price-auction.svg", c.render(
        "Three ads with bids and predicted click-through rates, ranked by eCPM equals pCTR times bid times 1000: "
        "A at 60, B at 48, C at 40 dollars. A wins and pays per click the runner-up's eCPM divided by its own pCTR "
        "times 1000, 48 over 30 equals 1.60 dollars, below its 2 dollar bid."
    ), "Second-price auction: rank by eCPM = pCTR × bid × 1000; the winner pays bid₂·pCTR₂/pCTR₁ per click."


@diagram
def bandits():
    c = Canvas(880, 460)
    c.title("Explore–exploit: bandits",
            "Each round: pick an ad, see a click or not, update the beliefs, repeat.")
    c.box(40, 90, 150, 56, "Choose arm", "policy below", role="accent")
    c.box(250, 90, 150, 56, "Show the ad", "one impression")
    c.box(250, 196, 150, 56, "Observe", "click r ∈ {0, 1}")
    c.box(40, 196, 150, 56, "Update", "α += r, β += 1 − r", role="good")
    c.arrow(190, 118, 246, 118, accent=True)
    c.arrow(325, 146, 325, 192, accent=True)
    c.arrow(250, 224, 194, 224, accent=True)
    c.arrow(115, 196, 115, 150, accent=True)
    for i, (name, rule) in enumerate([("ε-greedy", "best mean; random arm with prob ε"),
                                       ("UCB1", "argmax mean + √(2 ln t / n_arm)"),
                                       ("Thompson", "θ_k ~ Beta(α_k, β_k); play argmax θ")]):
        y = 290 + i * 24
        c.text(40, y, name, "lbl", anchor="start")
        c.text(130, y, rule, "small", anchor="start")
    # Regret curves.
    x0, y0, x1, y1 = 500, 330, 850, 90
    _axes(c, x0, y0, x1, y1, "rounds t", "regret")
    w, h = x1 - x0 - 20, y0 - y1 - 20

    def curve(fn, n=60):
        return _poly([(x0 + w * i / n, y0 - h * fn(i / n)) for i in range(n + 1)])

    c.curve(curve(lambda t: 0.95 * t), tone="muted_fg")
    c.curve(curve(lambda t: 0.085 + 0.3 * t if t > 0.05 else 2.0 * t), tone="warn")
    c.curve(curve(lambda t: 0.2 * math.log1p(20 * t) / math.log(21)), tone="accent")
    c.text(x0 + w - 4, y0 - h * 0.95 - 8, "uniform: linear", "small", anchor="end")
    c.text(x0 + w, y0 - h * 0.385 - 10, "ε-greedy: slope ε × gap", "small t-warn", anchor="end")
    c.text(x0 + w, y0 - h * 0.2 + 20, "Thompson: flattens", "small t-accent", anchor="end")
    _takeaway(c, 380, "Regret(T) = Σ_t ( p* − p_{a_t} ): expected clicks lost against always showing the best ad.",
              "Thompson: ad with 3 clicks in 100 shows → Beta(1 + 3, 1 + 97) = Beta(4, 98), mean 4/102 ≈ 3.9%.")
    return "l3-x-bandits.svg", c.render(
        "Left: the bandit loop of choosing an arm, showing the ad, observing a click and updating the Beta "
        "posterior, with the selection rules of epsilon-greedy, UCB1 and Thompson sampling. Right: cumulative "
        "regret over rounds; uniform play grows linearly and steeply, epsilon-greedy grows linearly with a small "
        "slope, Thompson sampling flattens out."
    ), "Explore–exploit: the bandit loop, three policies, and how their regret grows."


# ---------------------------------------------------------------------------
# 15 - fairness and privacy
# ---------------------------------------------------------------------------


def _bump(c: Canvas, cx: float, base: float, width: float, height: float, tone: str, dashed: bool = False) -> None:
    pts = [(cx + width * (i / 20 - 1.5), base - height * math.exp(-0.5 * ((i / 20 - 1.5) * 3) ** 2))
           for i in range(61)]
    c.curve(_poly(pts), tone=tone, dashed=dashed)


@diagram
def bpsn_bnsp():
    c = Canvas(880, 440)
    c.title("Subgroup AUC, BPSN and BNSP",
            "Score distributions of toxic (+) and non-toxic (−) comments, with and without an identity mention.")
    x0, x1 = 120, 560
    for y, label in [(170, "Background"), (290, "Subgroup g")]:
        c.rule(x0, y, x1, y)
        c.text(x0 - 12, y - 20, label, "lbl", anchor="end")
    _bump(c, 230, 170, 60, 70, "good")
    _bump(c, 450, 170, 60, 70, "bad")
    c.text(230, 92, "BN (−)", "small t-good")
    c.text(450, 92, "BP (+)", "small t-bad")
    _bump(c, 230, 290, 60, 70, "good", dashed=True)
    _bump(c, 360, 290, 60, 70, "good")
    _bump(c, 470, 290, 60, 70, "bad")
    c.text(360, 212, "SN (−), shifted up", "small t-good")
    c.text(480, 212, "SP (+)", "small t-bad")
    c.arrow(256, 254, 330, 254, dashed=True)
    c.text(x0, 312, "score 0", "small", anchor="start")
    c.text(x1, 312, "score 1", "small", anchor="end")
    bx = 600
    c.box(bx, 86, 250, 62, "Subgroup AUC", "SN vs SP: separation within g")
    c.box(bx, 162, 250, 62, "BPSN AUC: BP vs SN", "low → identity term causes false positives",
          role="bad")
    c.box(bx, 238, 250, 62, "BNSP AUC: BN vs SP", "low → toxic posts about g are missed", role="good")
    _takeaway(c, 344, "AUC(A vs B) = P(score of a random positive in A > score of a random negative in B)",
              "Notebook signature for 'black', 'muslim', 'gay': BPSN 0.67–0.75 but BNSP 0.91–0.95: benign mentions score like toxicity.")
    c.text(24, 412, "No threshold is involved: these AUCs show where the ranking fails before you pick an operating point.",
           "small", anchor="start")
    return "l3-x-bpsn-bnsp.svg", c.render(
        "Score distributions on two rows. Background comments: non-toxic BN on the left, toxic BP on the right. "
        "Subgroup comments: non-toxic SN shifted towards higher scores by the identity term, and toxic SP. "
        "Subgroup AUC compares SN with SP, BPSN compares BP with SN and is low when SN is shifted up, and BNSP "
        "compares BN with SP and is low when toxic posts about the group are missed."
    ), "Subgroup AUC, BPSN and BNSP as comparisons between four score distributions."


@diagram
def dp_sgd():
    c = Canvas(880, 460)
    c.title("One DP-SGD step",
            "Bound each example's influence, then hide it in noise.")
    y = 86
    c.box(30, y, 140, 62, "Per-example", "gradients g_1 … g_n", role="accent")
    c.box(200, y, 140, 62, "Clip each", "to norm ≤ C", role="warn")
    c.box(370, y, 140, 62, "Sum", "Σ_i clipped g_i", role="accent")
    c.box(540, y, 140, 62, "Add noise", "N(0, σ² C² I)", role="warn")
    c.box(710, y, 140, 62, "Average, step", "θ −= η · sum / n", role="good")
    for x in (170, 340, 510, 680):
        c.arrow(x, y + 31, x + 26, y + 31, accent=True)
    # Clipping picture.
    cx, cy, r = 270, 260, 60
    c.circle(cx, cy, r, role="ghost")
    c.text(cx, cy + r + 22, "radius C", "small")
    for ang, ln in [(-30, 40), (40, 52), (160, 110), (230, 35)]:
        a = math.radians(ang)
        end = min(ln, r - 4)
        c.arrow(cx, cy, cx + end * math.cos(a), cy - end * math.sin(a), accent=ln <= r)
        if ln > r:
            c.curve(f"M {cx + r * math.cos(a):.1f} {cy - r * math.sin(a):.1f} "
                    f"L {cx + ln * math.cos(a):.1f} {cy - ln * math.sin(a):.1f}", tone="bad", dashed=True, width=1.8)
    c.text(110, 296, "a large gradient is", "small t-bad", anchor="middle")
    c.text(110, 311, "cut back to length C", "small t-bad", anchor="middle")
    c.text(270, 186, "g · min(1, C / ‖g‖)", "mono")
    c.text(610, 210, "noise std on the summed gradient: σ C", "small", anchor="middle")
    c.text(610, 232, "on the mean: σ C / n", "small", anchor="middle")
    c.text(610, 254, "bigger batches → less noise per step", "small t-good", anchor="middle")
    _takeaway(c, 360, "(ε, δ)-DP: P[A(D) ∈ S] ≤ e^ε · P[A(D′) ∈ S] + δ for datasets differing in one record",
              "C = 1, σ = 2, n = 256: noise std on the mean gradient = 2 · 1 / 256 ≈ 0.0078 per coordinate. "
              "ε comes from an accountant (σ, sampling rate, steps).")
    return "l3-x-dp-sgd.svg", c.render(
        "One DP-SGD step: compute per-example gradients, clip each to L2 norm at most C, sum them, add Gaussian "
        "noise with standard deviation sigma times C, then average and take a step. A circle of radius C shows a "
        "long gradient cut back to the boundary while short ones are unchanged. Noise on the mean gradient is "
        "sigma C over n."
    ), "DP-SGD: per-example clipping to norm C plus Gaussian noise σC on the summed gradient."


# ---------------------------------------------------------------------------
# 16 - LLM evaluation and guardrails
# ---------------------------------------------------------------------------


@diagram
def llm_eval_harness():
    c = Canvas(880, 350)
    c.title("An LLM evaluation harness",
            "Every prompt, model or parser change goes through the same pipeline on the same items.")
    y = 90
    boxes = [("Eval set", "versioned, gold labels", "default"), ("Prompt", "template × item", "accent"),
             ("Model", "greedy, fixed config", "accent"), ("Parser", "JSON → fields", "warn"),
             ("Metrics", "exact match, field F1", "accent"), ("Paired CI", "bootstrap Δ", "good")]
    x = 30
    for i, (lab, sub, role) in enumerate(boxes):
        c.box(x, y, 124, 64, lab, sub, role=role)
        if i < len(boxes) - 1:
            c.arrow(x + 124, y + 32, x + 142, y + 32, accent=True)
        x += 142
    c.box(170, 200, 540, 50, "Log everything: generations.jsonl (prompt, reply, parsed fields, score)",
          role="muted", label_cls="small")
    c.arrow(440, 154, 440, 196, dashed=True)
    _takeaway(c, 280, "Read the replies: format failures, parser misses and real errors look the same in one number.",
              "Paired bootstrap: resample items, recompute mean(B) − mean(A); 95% CI = 2.5th–97.5th percentiles.")
    return "l3-x-llm-eval-harness.svg", c.render(
        "An LLM evaluation pipeline: a versioned evaluation set with gold labels, a prompt template, the model "
        "with a fixed decoding configuration, a parser that turns replies into fields, metrics such as exact match "
        "and field F1, and a paired bootstrap confidence interval for the difference between variants. All "
        "generations and scores are logged."
    ), "LLM evaluation harness: dataset → prompt → model → parser → metrics → paired bootstrap CI."


@diagram
def judge_position_bias():
    c = Canvas(880, 420)
    c.title("Testing an LLM judge for position bias",
            "Ask about the same pair twice, with the order swapped.")
    for i, (first, second, verdict, role, means) in enumerate([
            ("X (correct)", "Y (corrupted)", "“A”", "good", "picks X"),
            ("Y (corrupted)", "X (correct)", "“A”", "bad", "picks Y")]):
        y = 90 + i * 100
        c.text(30, y + 36, f"order {i + 1}", "lbl", anchor="start")
        c.box(110, y, 170, 34, f"A: {first}", role="default", label_cls="small")
        c.box(110, y + 40, 170, 34, f"B: {second}", role="default", label_cls="small")
        c.box(330, y + 7, 130, 60, "Judge", "which is better?", role="accent")
        c.arrow(280, y + 37, 326, y + 37, accent=True)
        c.box(510, y + 7, 120, 60, verdict, means, role=role)
        c.arrow(460, y + 37, 506, y + 37, accent=True)
    c.box(670, 100, 190, 160, "Same letter twice", "the position decided, not the content: inconsistent",
          role="bad")
    c.arrow(630, 134, 666, 160)
    c.arrow(630, 234, 666, 210)
    _takeaway(c, 300, "position-bias rate = share of pairs where both orders return the same letter",
              "Debias: average the log-prob margin log P(A) − log P(B) for each candidate over its two positions.")
    c.text(24, 380, "The notebook's 0.5B judge said “A” every time (rate 1.00); debiased, it preferred the "
           "correct answer in 75% of pairs.", "small", anchor="start")
    return "l3-x-judge-position-bias.svg", c.render(
        "A position-bias test. The same pair of answers is shown to an LLM judge twice: first with the correct "
        "answer X in slot A, then with it in slot B. The judge answers A both times, so it picked the correct "
        "answer once and the corrupted one once: the position decided. The position-bias rate is the share of "
        "pairs that get the same letter in both orders."
    ), "LLM-as-judge position-bias test: judge (X, Y) then (Y, X); the same letter twice means position decided."


@diagram
def prompt_injection():
    c = Canvas(880, 420)
    c.title("Prompt injection and layered defences",
            "User text is data, but the model reads it in the same channel as its instructions.")
    y = 120
    c.box(30, y, 140, 70, "Ticket text", "“ignore previous instructions…”", role="bad")
    c.box(200, y, 130, 70, "Input guardrail", "rules + classifier", role="warn")
    c.box(360, y, 150, 70, "LLM", "system prompt marks the ticket untrusted", role="accent")
    c.box(540, y, 140, 70, "Output check", "schema + business rules", role="warn")
    c.box(710, y, 140, 70, "Action", "least privilege; human for refunds", role="good")
    for x in (170, 330, 510, 680):
        c.arrow(x, y + 35, x + 26, y + 35, accent=True)
    c.text(265, 106, "block or flag", "small t-warn")
    c.text(610, 106, "reject bad output", "small t-warn")
    c.path("M 100 190 L 100 250 L 780 250 L 780 194", dashed=True, accent=False)
    c.text(440, 244, "without the layers, the attacker's text controls the action", "small t-bad")
    c.box(250, 270, 380, 40, "Red-team continuously: track attack success rate (ASR)", role="muted",
          label_cls="small")
    _takeaway(c, 330, "No single layer fixes injection; each one lowers the attack success rate.",
              "ASR = attacks where the targeted field took the attacker's value ÷ attacks. Notebook: 8/12 = 0.67 (terse) vs 6/12 = 0.50 (schema).")
    return "l3-x-prompt-injection.svg", c.render(
        "The path of a prompt injection: ticket text containing an instruction passes an input guardrail, an LLM "
        "whose system prompt marks the ticket as untrusted, an output check against a schema and business rules, "
        "and finally an action with least privilege and human approval for refunds. A dashed path shows that "
        "without these layers the attacker's text controls the action. Red-teaming tracks the attack success rate."
    ), "Prompt injection: the attack path and the guardrail layers that each lower the attack success rate."


# ---------------------------------------------------------------------------
# 17-18 - platform and design cases
# ---------------------------------------------------------------------------


@diagram
def ml_platform():
    c = Canvas(880, 440)
    c.title("ML platform reference architecture",
            "Five components, each owned by a team, joined by contracts.")
    for i, lab in enumerate(["Event streams", "Databases", "Batch tables"]):
        c.box(30, 120 + i * 50, 120, 40, lab, label_cls="small")
        c.arrow(150, 140 + i * 50, 186, 170 + (i - 1) * 16)
    c.box(190, 70, 670, 30, "Metadata · lineage · access control (shared by every component)", role="ghost",
          label_cls="small")
    cols = [(190, "Feature platform", "definitions in code; offline + online stores", "accent"),
            (360, "Training platform", "versioned data, GPU quota, tracking", "accent"),
            (530, "Registry + deploy", "signature, CI/CD/CT, shadow, canary", "accent"),
            (700, "Serving", "online API · batch scoring", "good")]
    for x, lab, sub, role in cols:
        c.box(x, 120, 150, 130, lab, sub, role=role)
    for x in (340, 510, 680):
        c.arrow(x, 185, x + 16, 185, accent=True)
    c.box(190, 290, 660, 50, "Monitoring: data quality · drift · delayed-label performance · cost", role="warn")
    c.arrow(775, 250, 775, 286)
    c.path("M 435 290 L 435 254", dashed=True)
    c.text(445, 276, "retrain trigger", "small", anchor="start")
    _takeaway(c, 362, "Build the registry, deploy path and monitoring first; adopt a tracker; grow the feature store.",
              "Measure the platform like a product: time from new repo to first shadow deployment, share on the golden path.")
    return "l3-x-ml-platform.svg", c.render(
        "An ML platform. Event streams, databases and batch tables feed a feature platform, then a training "
        "platform, a registry with deployment, and serving for online and batch use. A metadata, lineage and "
        "access-control layer spans the components, and a monitoring platform beneath them watches data quality, "
        "drift, delayed-label performance and cost, and can trigger retraining."
    ), "ML platform reference architecture: features, training, registry and deployment, serving, monitoring."


@diagram
def design_cases():
    c = Canvas(880, 510)
    c.title("Four design cases, one skeleton",
            "Each case narrows, scores and decides; each has its own signature difficulty.")
    cases = [("Ads ranking", ["Targeting filters", "Retrieve ~1k ads", "pCTR × pCVR model", "Auction + pacing"],
              "calibration × bids"),
             ("Video recs", ["Sources → ~5k", "Pre-ranker", "Multi-task ranker", "Blend + diversity"],
              "watch time vs long-term"),
             ("LLM assistant", ["Input guardrail", "Retrieve (RAG)", "LLM with citations", "Check or hand off"],
              "evaluating generated text"),
             ("Fraud", ["Velocity features", "Rules", "GBDT / NN score", "Approve / step-up"],
              "adversaries, late labels")]
    for i, (name, stages, hard) in enumerate(cases):
        x = 30 + i * 212
        c.text(x + 90, 92, name, "lbl")
        for j, st in enumerate(stages):
            y = 106 + j * 66
            c.box(x, y, 180, 46, st, role="good" if j == 3 else "accent")
            if j < 3:
                c.arrow(x + 90, y + 46, x + 90, y + 62, accent=True)
        c.box(x, 372, 180, 60, hard, "signature difficulty", role="warn")
    _takeaway(c, 450, "Same 12-part structure every time; spend the deep dives on the signature difficulty.",
              "Fleet size: replicas = max(2, ⌈peak QPS / (per-replica QPS × (1 − headroom))⌉).")
    return "l3-x-design-cases.svg", c.render(
        "Four design-case skeletons side by side. Ads: targeting filters, retrieve about a thousand ads, a pCTR "
        "times pCVR model, then auction and pacing; signature difficulty calibration times bids. Video: several "
        "sources to five thousand candidates, pre-ranker, multi-task ranker, blending with diversity; watch time "
        "versus long-term value. LLM assistant: input guardrail, retrieval, LLM with citations, check or hand off; "
        "evaluating generated text. Fraud: velocity features, rules, a model score, approve or step-up; "
        "adversaries and late labels."
    ), "Four design-case skeletons (ads, video, LLM assistant, fraud) with their signature difficulties."


# ---------------------------------------------------------------------------
# 19 - depth: double descent
# ---------------------------------------------------------------------------


@diagram
def double_descent():
    c = Canvas(880, 420)
    c.title("Double descent",
            "Test error vs model size: the peak sits at the interpolation threshold, p ≈ n.")
    x0, y0, x1, y1 = 80, 320, 600, 80
    _axes(c, x0, y0, x1, y1, "number of features p (log scale)", "test MSE")
    w, h = x1 - x0 - 20, y0 - y1 - 20
    xp = x0 + w * 0.4

    def mn(t):
        d = abs(t - 0.4)
        if t < 0.4:
            return min(0.95, 0.55 - 0.5 * t + 0.03 / (d + 0.02))
        return min(0.95, 0.22 + 0.25 * math.exp(-4 * (t - 0.4)) + 0.03 / (d + 0.02) - 0.04)

    c.curve(_poly([(x0 + w * i / 120, y0 - h * mn(i / 120)) for i in range(121)]), tone="accent")
    c.curve(_poly([(x0 + w * i / 60, y0 - h * (0.56 - 0.36 * (1 - math.exp(-3 * i / 60)))) for i in range(61)]),
            tone="good", dashed=True)
    c.curve(f"M {xp:.1f} {y0} L {xp:.1f} {y1 + 10}", tone="warn", dashed=True, width=1.6)
    c.text(xp, y1 + 2, "p = n (interpolation threshold)", "small t-warn")
    c.text(x0 + 20, y0 - h * 0.62, "under-parameterised", "small", anchor="start")
    c.text(480, 196, "over-parameterised:", "small t-accent")
    c.text(480, 211, "min-norm fit gets smoother", "small t-accent")
    c.text(xp + 26, 112, "→ ~190, off the chart", "small", anchor="start")
    c.text(x1 - 30, y0 - h * 0.12, "ridge: no peak", "small t-good", anchor="end")
    bx = 630
    c.box(bx, 90, 220, 66, "Notebook (n = 100)", "test MSE: p=50 0.51 · p=100 ~190 · p=3000 0.30",
          role="accent")
    c.box(bx, 176, 220, 66, "Why the peak", "exactly determined, tiny singular values amplify noise", role="warn")
    c.box(bx, 262, 220, 56, "What removes it", "ridge, early stopping, more data", role="good")
    _takeaway(c, 346, "Min-norm least squares: w = X⁺ y, the interpolating solution with the smallest ‖w‖.",
              "The peak comes from fitting noise exactly with a badly conditioned system, not from 'too many parameters'.")
    return "l3-x-double-descent.svg", c.render(
        "Test error against the number of random features on a log scale. The minimum-norm fit's error falls, "
        "spikes at the interpolation threshold where the number of features equals the number of training points, "
        "then descends again below its earlier minimum. A ridge-regularised fit has no peak. Side boxes give the "
        "notebook's numbers and the reasons."
    ), "Double descent: test error peaks at the interpolation threshold p = n; regularisation removes the peak."


# ---------------------------------------------------------------------------
# 20 - the interview loop
# ---------------------------------------------------------------------------


@diagram
def interview_loop():
    c = Canvas(880, 420)
    c.title("The full interview loop",
            "Seven rounds, each scored 1–4 on a few signals; the debrief looks for recurring patterns.")
    row1 = [("Phone screen", "problem solving, code", "default"), ("Coding 1", "code quality, verification", "default"),
            ("Coding 2", "different pattern", "default"), ("ML coding", "fundamentals, verification", "accent")]
    row2 = [("ML system design", "quantification, trade-offs", "accent"), ("ML depth", "fundamentals, evidence", "accent"),
            ("Behavioural", "ownership, leadership", "default"), ("Debrief", "per round + per signal", "good")]
    for i, (lab, sub, role) in enumerate(row1):
        x = 30 + i * 212
        c.box(x, 90, 180, 64, lab, sub, role=role)
        if i < 3:
            c.arrow(x + 180, 122, x + 208, 122, accent=True)
    c.path("M 726 154 L 726 180 L 120 180 L 120 206", accent=True)
    for i, (lab, sub, role) in enumerate(row2):
        x = 30 + i * 212
        c.box(x, 210, 180, 64, lab, sub, role=role)
        if i < 3:
            c.arrow(x + 180, 242, x + 208, 242, accent=True)
    c.text(120, 298, "45–60 min", "small")
    c.text(332, 298, "45 min, drills 'why?'", "small")
    c.text(544, 298, "4–5 STAR stories", "small")
    c.text(756, 298, "a weak round stands out", "small t-warn")
    _takeaway(c, 326, "overall = Σ_r w_r · mean(scores_r) / Σ_r w_r; also average each signal across the rounds that score it",
              "Example: design 3.0 (w 1.5), coding 2.25 (w 1), phone 3.0 (w 0.5) → (4.5 + 2.25 + 1.5) / 3 = 2.75.")
    c.text(24, 394, "The weakest signal (e.g. verification, low in four coding rounds) is next week's practice plan.",
           "small", anchor="start")
    return "l3-x-interview-loop.svg", c.render(
        "The interview loop as a sequence of rounds: phone screen, two coding rounds, ML coding, ML system design, "
        "ML depth, behavioural, then the debrief. Each round lists the signals it mainly scores. The overall score "
        "is a weighted mean of round means, and each signal is also averaged across the rounds that score it."
    ), "The full interview loop: rounds, the signals each scores, and how a loop is summarised."
