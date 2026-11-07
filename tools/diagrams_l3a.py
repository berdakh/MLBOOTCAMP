"""Diagrams for Level 3, notebooks 01-10 (prefix ``l3-``).

See tools/make_diagrams.py for the conventions. Each function returns
``(filename, svg, caption)``; the caption doubles as alt text.
"""

from __future__ import annotations

import math

from svgkit import Canvas, diagram  # noqa: F401

W = 880


def _takeaway(c: Canvas, y: float, text: str, width: float = W, role: str = "good") -> None:
    """One takeaway box across the bottom of the figure."""
    c.box(24, y, width - 48, 40, text, role=role)


def _vline(c: Canvas, x: float, y1: float, y2: float, tone: str = "stroke", dashed: bool = False,
           width: float = 1.4) -> None:
    c.curve(f"M {x:.1f} {y1:.1f} L {x:.1f} {y2:.1f}", tone=tone, dashed=dashed, width=width)


def _hline(c: Canvas, x1: float, x2: float, y: float, tone: str = "stroke", dashed: bool = False,
           width: float = 1.4) -> None:
    c.curve(f"M {x1:.1f} {y:.1f} L {x2:.1f} {y:.1f}", tone=tone, dashed=dashed, width=width)


def _bracket(c: Canvas, x1: float, x2: float, y: float, label: str, tone: str) -> None:
    """A horizontal measuring bracket with end ticks and a label under it."""
    _hline(c, x1, x2, y, tone=tone, width=2)
    _vline(c, x1, y - 7, y + 7, tone=tone, width=2)
    _vline(c, x2, y - 7, y + 7, tone=tone, width=2)
    c.text((x1 + x2) / 2, y + 20, label, f"small t-{tone}")


# ---------------------------------------------------------------------------
# 01 - GPU fundamentals
# ---------------------------------------------------------------------------


@diagram
def async_launch():
    c = Canvas(W, 380)
    c.title("GPU kernels launch asynchronously",
            "The CPU only queues the work and races ahead; the GPU runs it later.")

    c.text(24, 112, "CPU (Python)", "lbl", anchor="start")
    c.text(24, 192, "GPU stream", "lbl", anchor="start")

    # CPU lane: three quick launches, then the wrong clock read, then a sync.
    for i, x in enumerate((150, 240, 330)):
        c.box(x, 90, 80, 40, f"launch {'ABC'[i]}", role="muted", label_cls="small")
    c.box(420, 90, 110, 40, "t1 = clock()", role="bad", label_cls="small")
    c.box(540, 90, 150, 40, "synchronize: wait", role="warn", label_cls="small")
    c.box(700, 90, 110, 40, "t1 = clock()", role="good", label_cls="small")

    # GPU lane: the kernels actually run here, one after another.
    for i, x in enumerate((190, 370, 550)):
        c.box(x, 170, 170, 40, f"kernel {'ABC'[i]} runs", role="accent")
    for x_cpu, x_gpu in ((190, 200), (280, 380), (370, 560)):
        c.arrow(x_cpu, 132, x_gpu, 166, dashed=True)
    c.arrow(690, 190, 696, 132, dashed=True)

    _bracket(c, 150, 530, 246, "timed without sync: launch cost only (~0.1 ms)", "bad")
    _bracket(c, 150, 810, 290, "synchronize before and after: the real work", "good")

    _takeaway(c, 324, "Warm up, synchronize before and after, repeat, report the median.")
    return "l3-async-launch.svg", c.render(
        "Two timelines. On the CPU, Python launches kernels A, B and C in microseconds and moves on; "
        "on the GPU the kernels run later, one after another. Reading the clock right after the "
        "launches measures only launch cost; synchronizing before reading the clock measures the "
        "real work."
    ), "Asynchronous kernel launches: why GPU timing needs synchronize."


@diagram
def roofline():
    c = Canvas(W, 440)
    c.title("The roofline model",
            "Attainable FLOP/s = min(peak, arithmetic intensity × memory bandwidth). T4, fp32.")

    x0, x1, y0, y1 = 90, 520, 80, 330          # plot box
    ai_lo, ai_hi = math.log2(1 / 16), math.log2(1024)
    f_lo, f_hi = math.log10(0.01), math.log10(30)  # TFLOP/s

    def px(ai):
        return x0 + (math.log2(ai) - ai_lo) / (ai_hi - ai_lo) * (x1 - x0)

    def py(tf):
        return y1 - (math.log10(tf) - f_lo) / (f_hi - f_lo) * (y1 - y0)

    _hline(c, x0, x1, y1)
    _vline(c, x0, y0, y1)
    for ai, lab in ((1 / 16, "1/16"), (1, "1"), (16, "16"), (256, "256")):
        c.text(px(ai), y1 + 16, lab, "small")
    for tf, lab in ((0.01, "0.01"), (0.1, "0.1"), (1, "1"), (10, "10")):
        c.text(x0 - 8, py(tf) + 4, lab, "small", anchor="end")
    c.text((x0 + x1) / 2, y1 + 36, "arithmetic intensity (FLOP per byte, log scale)", "small")
    c.text(x0 - 8, y0 - 10, "TFLOP/s", "small", anchor="end")

    peak, bw = 8.1, 0.32  # TFLOP/s, TB/s
    ridge = peak / bw
    c.curve(f"M {px(1 / 16):.1f} {py(bw / 16):.1f} L {px(ridge):.1f} {py(peak):.1f} "
            f"L {px(1024):.1f} {py(peak):.1f}", tone="accent", width=2.6)
    _vline(c, px(ridge), py(peak), y1, tone="muted_fg", dashed=True, width=1.2)
    c.text(px(ridge) + 6, y1 - 8, "ridge ≈ 25", "small", anchor="start")
    c.text(px(0.5), py(0.5 * bw) - 22, "memory-bound", "small t-warn", anchor="end")
    c.text(px(200), py(peak) - 10, "compute-bound", "small t-good")

    for ai, tf, lab, role in ((1 / 12, bw / 12, "x + y", "bad"),
                              (1.0, 0.32, "LayerNorm", "warn"),
                              (170, 6.0, "matmul n=1024", "good")):
        c.circle(px(ai), py(tf), 6, role=role)
        c.text(px(ai) + 10, py(tf) + 18, lab, "small", anchor="start")

    # Right-hand panel: the three numbers to remember.
    c.box(570, 80, 286, 64, "Arithmetic intensity", "FLOPs ÷ bytes moved to and from memory",
          role="default")
    c.box(570, 160, 286, 64, "Ridge point", "peak ÷ bandwidth = 8.1 T ÷ 320 G ≈ 25 FLOP/byte",
          role="default")
    c.box(570, 240, 286, 64, "matmul n×n", "2n³ FLOPs, 3n²·4 bytes → n/6 FLOP/byte",
          role="default")

    _takeaway(c, 384, "Left of the ridge, buy bandwidth (fuse kernels); right of it, buy FLOPs (fp16 Tensor Cores).")
    return "l3-roofline.svg", c.render(
        "Log-log roofline for a T4 in fp32: attainable throughput rises with arithmetic intensity "
        "along the memory-bandwidth roof until the ridge point at about 25 FLOP per byte, then "
        "stays flat at the 8.1 TFLOP/s compute peak. An elementwise add (intensity 1/12) and "
        "LayerNorm sit on the memory-bound slope; a 1024 by 1024 matmul (intensity about 170) is "
        "compute-bound."
    ), "Roofline: memory-bound vs compute-bound operations on a T4."


@diagram
def dataloader_overlap():
    c = Canvas(W, 380)
    c.title("Overlap loading with compute",
            "Same model, same data. Only the input pipeline changes.")

    x0 = 190
    load, step = 86, 45

    c.text(24, 104, "num_workers=0", "lbl", anchor="start")
    c.text(24, 122, "load, then train", "small", anchor="start")
    x = x0
    for i in range(4):
        c.box(x, 90, load, 34, f"load {i + 1}", role="warn", label_cls="small", radius=5)
        c.box(x + load, 90, step, 34, f"{i + 1}", role="accent", label_cls="small", radius=5)
        x += load + step
    c.text(x + 10, 112, "4 steps", "small t-bad", anchor="start")

    c.text(24, 178, "num_workers=2", "lbl", anchor="start")
    c.text(24, 196, "workers prefetch", "small", anchor="start")
    c.text(x0 - 10, 166, "worker 1", "small", anchor="end")
    c.text(x0 - 10, 204, "worker 2", "small", anchor="end")
    c.text(x0 - 10, 242, "train loop", "small", anchor="end")
    for k in range(3):
        c.box(x0 + k * load, 150, load - 2, 28, f"load {2 * k + 1}", role="warn", label_cls="small", radius=5)
        c.box(x0 + k * load, 188, load - 2, 28, f"load {2 * k + 2}", role="warn", label_cls="small", radius=5)
    for i in range(6):
        c.box(x0 + load + i * step, 226, step - 2, 28, f"{i + 1}", role="accent", label_cls="small", radius=5)
    c.text(x0 + load + 6 * step + 10, 246, "6 steps, GPU never idle", "small t-good", anchor="start")

    c.box(x0, 280, 18, 16, role="warn", radius=3)
    c.text(x0 + 26, 293, "load + collate a batch (CPU)", "small", anchor="start")
    c.box(x0 + 250, 280, 18, 16, role="accent", radius=3)
    c.text(x0 + 276, 293, "training step (GPU)", "small", anchor="start")

    _takeaway(c, 320, "Workers + prefetch hide loading; pin_memory + non_blocking=True also hides the copy.")
    return "l3-dataloader-overlap.svg", c.render(
        "Two timelines over the same time window. With num_workers=0 the loop alternates loading a "
        "batch and training on it, finishing 4 steps. With two workers, batches are loaded in "
        "parallel ahead of time and the training loop runs 6 steps back to back without waiting."
    ), "DataLoader workers and prefetching overlap loading with compute."


# ---------------------------------------------------------------------------
# 02 - training memory
# ---------------------------------------------------------------------------


@diagram
def training_memory():
    c = Canvas(W, 400)
    c.title("What fills memory in training, per parameter",
            "Bytes per parameter for AdamW, before activations.")

    x0, scale = 250, 26  # px per byte
    rows = [
        ("fp32", "the plain recipe", [("weights 4", 4, "accent"), ("grads 4", 4, "warn"),
                                      ("Adam m, v  8", 8, "good")], "16 B"),
        ("autocast (this course)", "fp32 state + fp16 copy",
         [("weights 4", 4, "accent"), ("grads 4", 4, "warn"), ("Adam m, v  8", 8, "good"),
          ("+2", 2, "muted")], "18 B"),
        ("fp16 + fp32 master", "ZeRO paper accounting",
         [("w 2", 2, "accent"), ("g 2", 2, "warn"), ("master + m + v  12", 12, "good")], "16 B"),
    ]
    y = 84
    for name, sub, segs, total in rows:
        c.text(24, y + 20, name, "lbl", anchor="start")
        c.text(24, y + 37, sub, "small", anchor="start")
        x = x0
        for lab, nbytes, role in segs:
            c.box(x, y + 4, nbytes * scale, 38, lab, role=role, label_cls="small", radius=4)
            x += nbytes * scale
        c.text(x + 12, y + 28, total, "lbl", anchor="start")
        y += 64

    c.box(24, 284, 400, 54, "+ activations",
          "batch × seq × hidden × layers: 5.8× the state above in 2.2", role="bad")
    c.box(456, 284, 400, 54, "7B params × 16 B = 112 GB",
          "before a single activation: why LoRA and ZeRO exist", role="warn")

    _takeaway(c, 350, "Mixed precision speeds up matmuls and halves activations; it does not shrink model state.")
    return "l3-training-memory.svg", c.render(
        "Stacked bars of bytes per parameter. fp32 AdamW: 4 for weights, 4 for gradients, 8 for the "
        "two Adam moments, 16 in total. PyTorch autocast keeps the same fp32 state and adds a 2-byte "
        "fp16 copy, 18 in total. The ZeRO accounting uses fp16 weights and gradients plus 12 bytes "
        "of fp32 master weights and moments, 16 in total. Activations come on top and often dominate."
    ), "Training memory per parameter: weights, gradients, Adam states, activations."


@diagram
def grad_accumulation():
    c = Canvas(W, 360)
    c.title("Gradient accumulation = a big batch in small pieces",
            "k micro-batches, backward on each, ONE optimizer step.")

    ys = (80, 136, 192, 248)
    for i, y in enumerate(ys):
        c.box(40, y, 220, 44, f"micro-batch {i + 1}", "forward + backward", role="accent")
        c.arrow(264, y + 22, 396, 186, accent=True)
    c.text(330, 104, "loss × nᵢ / N", "small t-warn")

    c.box(400, 150, 200, 72, "grads add up in .grad", "no zero_grad in between", role="warn")
    c.arrow(604, 186, 646, 186, accent=True)
    c.box(650, 150, 206, 72, "optimizer.step()", "then zero_grad()", role="good")

    _takeaway(c, 306, "Same gradient as batch k × b; activation memory of one micro-batch; k passes in sequence.")
    return "l3-grad-accumulation.svg", c.render(
        "Four micro-batches each run forward and backward; their gradients, each weighted by its "
        "share of the examples, add up in the .grad buffers; then one optimizer step and "
        "zero_grad. The result equals the gradient of the full batch at the activation memory of "
        "one micro-batch."
    ), "Gradient accumulation: k micro-batches, one optimizer step."


@diagram
def activation_checkpointing():
    c = Canvas(W, 384)
    c.title("Activation checkpointing: trade compute for memory",
            "Keep only each block's input; recompute the inside during backward.")

    xs = (170, 290, 410, 530)
    for row, (y, name, sub) in enumerate(((84, "Standard", "keep everything"),
                                          (204, "Checkpointed", "keep inputs only"))):
        c.text(24, y + 24, name, "lbl", anchor="start")
        c.text(24, y + 42, sub, "small", anchor="start")
        for i, x in enumerate(xs):
            c.box(x, y, 100, 36, f"block {i + 1}", role="accent", label_cls="small")
            if i < 3:
                c.arrow(x + 100, y + 18, x + 118, y + 18, accent=True)
            if row == 0:
                c.box(x, y + 44, 100, 40, "all internals", role="bad", label_cls="small", radius=4)
            else:
                c.box(x, y + 44, 30, 40, role="good", radius=4)
                c.box(x + 34, y + 44, 66, 40, "recompute", role="muted", label_cls="small", radius=4)
    c.text(660, 112, "82 MB saved", "lbl t-bad", anchor="start")
    c.text(660, 130, "for backward (2.4)", "small", anchor="start")
    c.text(660, 232, "14 MB saved", "lbl t-good", anchor="start")
    c.text(660, 250, "+20-40% step time", "small", anchor="start")
    c.text(660, 268, "gradients identical", "small", anchor="start")
    c.text(170, 306, "green = the block's input, kept · dashed = rebuilt during backward", "small", anchor="start")

    _takeaway(c, 324, "Memory: one input per block + one block at a time. Cost: about one extra forward pass.")
    return "l3-activation-checkpointing.svg", c.render(
        "Two rows of four transformer blocks. In standard training every block keeps all of its "
        "internal activations for backward (82 MB here). With checkpointing each block keeps only "
        "its small input and recomputes its internals during backward (14 MB here), costing about "
        "one extra forward pass."
    ), "Activation checkpointing: keep block inputs, recompute internals."


# ---------------------------------------------------------------------------
# 03 - transformer from scratch
# ---------------------------------------------------------------------------


@diagram
def causal_attention():
    c = Canvas(W, 420)
    c.title("Causal self-attention as a soft lookup",
            "Each query scores every key, the mask hides the future, softmax turns scores into weights.")

    toks = ["the", "cat", "sat", "on", "mat"]
    gx, gy, cell = 110, 110, 44
    for j, t in enumerate(toks):
        c.text(gx + j * cell + cell / 2, gy - 10, t, "small")
        c.text(gx - 10, gy + j * cell + cell / 2 + 4, t, "small", anchor="end")
    c.text(gx + 2.5 * cell, gy - 30, "keys (what each token offers)", "small t-accent")
    c.text(gx - 66, gy + 2.5 * cell, "queries", "small t-accent", anchor="middle")
    weights = {2: [0.2, 0.5, 0.3]}
    for i in range(5):
        for j in range(5):
            x, y = gx + j * cell, gy + i * cell
            if j <= i:
                lab = f"{weights[i][j]:.1f}" if i in weights else ""
                c.box(x + 2, y + 2, cell - 4, cell - 4, lab, role="accent", label_cls="small", radius=4)
            else:
                c.box(x + 2, y + 2, cell - 4, cell - 4, "−∞", role="ghost", label_cls="small", radius=4)
    c.text(gx + 2.5 * cell, gy + 5 * cell + 22, "row \"sat\" sums to 1 over the past only", "small")

    bx, bw = 440, 300
    steps = [("scores = Q Kᵀ / √d", "how well each query matches each key", "default"),
             ("+ causal mask M", "0 on and below the diagonal, −∞ above", "warn"),
             ("softmax over each row", "weights ≥ 0 that sum to 1", "default"),
             ("weights × V", "a weighted average of value vectors", "good")]
    for k, (lab, sub, role) in enumerate(steps):
        y = 84 + k * 68
        c.box(bx, y, bw, 48, lab, sub, role=role)
        if k < 3:
            c.arrow(bx + bw / 2, y + 48, bx + bw / 2, y + 66, accent=True)

    _takeaway(c, 364, "A query asks, keys advertise, values are what you get back; the mask forbids peeking ahead.")
    return "l3-causal-attention.svg", c.render(
        "A five-by-five attention grid for the tokens 'the cat sat on mat'. Cells on and below the "
        "diagonal are allowed; cells above are masked with minus infinity. The row for 'sat' has "
        "weights 0.2, 0.5, 0.3 over 'the', 'cat', 'sat'. Beside it, the four steps: scaled scores "
        "QK transpose over root d, add the causal mask, softmax each row, multiply by V."
    ), "Causal attention: scores, mask, softmax, weighted sum of values."


@diagram
def decoder_block():
    c = Canvas(920, 360)
    c.title("One pre-LayerNorm decoder block",
            "Normalise inside each branch; the residual stream passes straight through.")

    y, h = 170, 52
    cy = y + h / 2
    c.box(24, y, 92, h, "x", "residual stream", role="default")
    c.arrow(116, cy, 146, cy, accent=True)
    c.box(150, y, 80, h, "LayerNorm", role="muted", label_cls="small")
    c.arrow(230, cy, 256, cy, accent=True)
    c.box(260, y, 150, h, "Attention", "causal, H heads", role="accent")
    c.arrow(410, cy, 424, cy, accent=True)
    c.circle(440, cy, 14, role="good")
    c.text(440, cy + 5, "+", "lbl")
    c.arrow(454, cy, 476, cy, accent=True)
    c.box(480, y, 80, h, "LayerNorm", role="muted", label_cls="small")
    c.arrow(560, cy, 586, cy, accent=True)
    c.box(590, y, 150, h, "MLP", "C → 4C → C, GELU", role="accent")
    c.arrow(740, cy, 754, cy, accent=True)
    c.circle(770, cy, 14, role="good")
    c.text(770, cy + 5, "+", "lbl")
    c.arrow(784, cy, 806, cy, accent=True)
    c.box(810, y, 86, h, "out", "to next block", role="default")

    # Residual (skip) paths above the row.
    c.path(f"M 133 {cy} L 133 116 L 440 116 L 440 {cy - 16}", accent=False)
    c.path(f"M 466 {cy} L 466 116 L 770 116 L 770 {cy - 16}", accent=False)
    c.text(286, 108, "residual: h = x + Attn(LN(x))", "small")
    c.text(618, 108, "residual: out = h + MLP(LN(h))", "small")

    c.text(460, 260, "Stack N blocks, then a final LayerNorm and a Linear to the vocabulary"
           " (weight-tied to the token embedding).", "small")
    _takeaway(c, 300, "Pre-LN keeps a clean identity path from output to input, so deep stacks train stably.",
              width=920)
    return "l3-decoder-block.svg", c.render(
        "A pre-LayerNorm decoder block read left to right: x goes through LayerNorm and causal "
        "multi-head attention and is added back to x; the result goes through LayerNorm and an MLP "
        "and is added back again. Two residual skip paths run above the row."
    ), "A pre-LN decoder block: attention and MLP branches added to the residual stream."


@diagram
def kv_cache():
    c = Canvas(W, 420)
    c.title("The KV cache: prefill once, then decode one token at a time",
            "Past keys and values never change under a causal mask, so store them.")

    # Prefill (left)
    c.text(40, 92, "1 · Prefill: the whole prompt in one pass", "lbl", anchor="start")
    for i in range(5):
        c.box(40 + i * 64, 106, 56, 34, f"t{i + 1}", role="default", label_cls="small")
    c.arrow(190, 144, 190, 168, accent=True)
    c.box(40, 172, 312, 44, "all layers, all positions at once", "big matmuls: compute-bound",
          role="accent")
    c.arrow(190, 220, 190, 244, accent=True)
    for i in range(5):
        c.box(40 + i * 64, 248, 56, 26, "K", role="good", label_cls="small", radius=4)
        c.box(40 + i * 64, 278, 56, 26, "V", role="good", label_cls="small", radius=4)

    # Decode (right)
    c.text(470, 92, "2 · Decode: one new token per step", "lbl", anchor="start")
    c.box(470, 106, 56, 34, "t6", role="warn", label_cls="small")
    c.arrow(530, 123, 556, 123, accent=True)
    c.box(560, 106, 296, 34, "q, k, v for ONE token", role="accent", label_cls="small")
    c.arrow(818, 144, 818, 244, accent=True)
    c.text(808, 196, "append the new k, v", "small", anchor="end")
    c.text(808, 212, "then attend q over all 6", "small", anchor="end")
    for i in range(6):
        role = "warn" if i == 5 else "good"
        c.box(470 + i * 64, 248, 56, 26, "K", role=role, label_cls="small", radius=4)
        c.box(470 + i * 64, 278, 56, 26, "V", role=role, label_cls="small", radius=4)

    c.text(W / 2, 334, "cache bytes = 2 · layers · heads · d_head · seq · batch · bytes"
           "   (7B, fp16: 0.5 MB per token, 2 GB at 4k tokens)", "small")
    _takeaway(c, 356, "Decode does little arithmetic per byte of cache read: long contexts are memory-bound.")
    return "l3-kv-cache.svg", c.render(
        "Left: prefill runs all five prompt tokens through the model at once and writes their keys "
        "and values into the cache. Right: each decode step computes q, k, v for one new token, "
        "appends k and v as a sixth cache entry, and attends over all six. The cache size formula "
        "is 2 times layers times heads times head dimension times sequence times batch times bytes."
    ), "KV cache: prefill writes the cache; each decode step appends one entry."


# ---------------------------------------------------------------------------
# 04 - pretrained transformers and tokenization
# ---------------------------------------------------------------------------


@diagram
def text_to_vector():
    c = Canvas(920, 330)
    c.title("From text to one sentence vector",
            "Every encoder pipeline, with the shapes at each step.")

    steps = [("text", "\"isn't magic\"", "default"),
             ("tokens", "isn, ', t, magic", "default"),
             ("ids", "101 … 102", "default"),
             ("embeddings", "token + position", "accent"),
             ("encoder", "6 layers, attention", "accent"),
             ("hidden states", "B × T × 768", "accent"),
             ("mean pool", "B × 768", "good")]
    x, w, gap = 24, 108, 18
    for i, (lab, sub, role) in enumerate(steps):
        c.box(x, 96, w, 76, lab, sub, role=role)
        if i < len(steps) - 1:
            c.arrow(x + w, 134, x + w + gap - 4, 134, accent=True)
        x += w + gap

    c.text(24, 206, "tokenizer (lossy, model-specific)", "small t-warn", anchor="start")
    _hline(c, 24, 360, 214, tone="warn")
    c.text(402, 206, "model (frozen or fine-tuned)", "small t-accent", anchor="start")
    _hline(c, 402, 738, 214, tone="accent")
    c.text(780, 206, "pooling", "small t-good", anchor="start")
    _hline(c, 780, 896, 214, tone="good")
    c.text(24, 244, "Padding positions still get hidden states: averaging them in changes the vector"
           " with whatever else is in the batch.", "small", anchor="start")

    _takeaway(c, 270, "Use the exact tokenizer the model was trained with, and pool with the attention mask.",
              width=920)
    return "l3-text-to-vector.svg", c.render(
        "A seven-step pipeline: text, subword tokens, ids with an attention mask, token plus position "
        "embeddings, a six-layer encoder, hidden states of shape batch by tokens by 768, and "
        "mask-aware mean pooling to one 768-dimensional vector per sentence."
    ), "Text to tokens to ids to hidden states to a pooled sentence vector."


@diagram
def padding_bucketing():
    c = Canvas(W, 400)
    c.title("Padding waste and length bucketing",
            "Every pad token costs as much compute as a real one. Same 6 texts, three batchings.")

    scale = 1.55  # px per token (max 128 tokens)
    panels = [("pad to max (128)", [[30, 120, 45], [60, 25, 90]], "fixed", "52%"),
              ("dynamic, data order", [[30, 120, 45], [60, 25, 90]], "dyn", "41%"),
              ("dynamic, sorted", [[25, 30, 45], [60, 90, 120]], "dyn", "25%")]
    for p, (name, batches, mode, waste) in enumerate(panels):
        x0 = 24 + p * 290
        c.text(x0, 92, name, "lbl", anchor="start")
        y = 106
        for b, rows in enumerate(batches):
            width = 128 if mode == "fixed" else max(rows)
            for n in rows:
                c.box(x0, y, width * scale, 14, role="ghost", radius=3)
                c.box(x0, y, n * scale, 14, role="accent", radius=3)
                y += 19
            c.text(x0 + width * scale + 6, y - 22, f"batch {b + 1}", "small", anchor="start")
            y += 14
        c.text(x0, y + 14, f"waste {waste}", "lbl t-bad" if p < 2 else "lbl t-good", anchor="start")

    c.box(24, 284, 18, 14, role="accent", radius=3)
    c.text(50, 296, "real tokens (attention mask = 1)", "small", anchor="start")
    c.box(290, 284, 18, 14, role="ghost", radius=3)
    c.text(316, 296, "padding (mask = 0, keys get −∞ in attention)", "small", anchor="start")
    c.text(24, 324, "Measured on AG News in 4.5: 59% → 50% → 17% waste, and 2.1-2.7× faster sorted.",
           "small", anchor="start")

    _takeaway(c, 344, "Inference: sort by length, batch, restore order. Training: sort within shuffled chunks.")
    return "l3-padding-bucketing.svg", c.render(
        "Three panels of the same six texts in two batches of three. Padding every row to 128 tokens "
        "wastes 52 percent of positions; dynamic padding to the longest row in each batch wastes 41 "
        "percent; sorting by length first so similar lengths share a batch wastes 25 percent."
    ), "Padding waste: pad to max vs dynamic padding vs length bucketing."


# ---------------------------------------------------------------------------
# 05 - calibration
# ---------------------------------------------------------------------------


@diagram
def calibration():
    c = Canvas(W, 420)
    c.title("Calibration and temperature scaling",
            "Among predictions made with confidence 0.8, about 80% should be right.")

    x0, y0, s = 90, 80, 250  # square plot, side s
    _hline(c, x0, x0 + s, y0 + s)
    _vline(c, x0, y0, y0 + s)
    c.curve(f"M {x0} {y0 + s} L {x0 + s} {y0}", tone="muted_fg", dashed=True, width=1.6)
    c.text(x0 + s / 2, y0 + s + 22, "confidence (max probability)", "small")
    c.text(x0 - 10, y0 + 8, "1", "small", anchor="end")
    c.text(x0 - 10, y0 + s, "0", "small", anchor="end")
    c.text(x0 + s, y0 + s + 40, "", "small")
    c.text(x0 - 30, y0 + s / 2, "accuracy", "small", anchor="middle")

    def curve(fn):
        pts = []
        for i in range(31):
            t = 0.25 + 0.75 * i / 30
            pts.append(f"{x0 + t * s:.1f} {y0 + s - fn(t) * s:.1f}")
        return "M " + " L ".join(pts)

    c.curve(curve(lambda t: min(1.0, t + 0.22 * math.sin(math.pi * (t - 0.25) / 0.75))), tone="warn")
    c.curve(curve(lambda t: max(0.0, t - 0.2 * math.sin(math.pi * (t - 0.25) / 0.75))), tone="bad")
    c.text(x0 + 70, y0 + 60, "under-confident", "small t-warn")
    c.text(x0 + 70, y0 + 74, "(above diagonal)", "small t-warn")
    c.text(x0 + 190, y0 + 200, "over-confident", "small t-bad")
    c.text(x0 + 190, y0 + 214, "(below)", "small t-bad")

    c.box(400, 80, 456, 56, "Temperature scaling: softmax(z / T)",
          "one scalar T fitted on VALIDATION by minimising NLL", role="accent")
    c.box(400, 148, 220, 56, "T < 1 sharpens", "fixes under-confidence", role="warn")
    c.box(636, 148, 220, 56, "T > 1 softens", "fixes over-confidence", role="bad")
    c.box(400, 216, 456, 56, "argmax unchanged", "so accuracy never changes, only the probabilities",
          role="default")
    c.box(400, 284, 456, 46, "this notebook: T = 0.59, ECE 0.124 → 0.034", role="good")

    _takeaway(c, 360, "ECE = Σ (|S_b| / n) · |acc(S_b) − conf(S_b)|: the gap to the diagonal, weighted by bin size.")
    return "l3-calibration.svg", c.render(
        "A reliability diagram: the diagonal is perfect calibration; an under-confident model's curve "
        "lies above it and an over-confident model's curve below it. Beside it: temperature scaling "
        "divides logits by one scalar T fitted on validation; T below 1 sharpens, T above 1 softens, "
        "and the argmax and accuracy never change. In this notebook T was 0.59 and ECE fell from "
        "0.124 to 0.034."
    ), "Reliability diagram and temperature scaling."


# ---------------------------------------------------------------------------
# 06 - parameter-efficient fine-tuning
# ---------------------------------------------------------------------------


@diagram
def finetune_ladder():
    c = Canvas(W, 380)
    c.title("Fine-tuning options, from cheapest to most flexible",
            "DistilBERT classifier, 67M parameters (numbers from notebooks 05 and 06).")

    steps = [(40, 220, "Linear probe", "frozen encoder + logistic head",
              "~3k trained", "default"),
             (320, 164, "LoRA (r = 8)", "low-rank adapters on q, v + new head",
              "741k trained (1.1%) · 263 MB state", "accent"),
             (600, 108, "Full fine-tuning", "every weight, small LR + warmup",
              "67M trained · 1,022 MB state", "warn")]
    for x, y, lab, sub, nums, role in steps:
        c.box(x, y, 240, 306 - y, "", role=role)
        c.text(x + 120, y + 26, lab, "lbl")
        c.text(x + 120, y + 46, sub, "small")
        c.text(x + 120, y + 68, nums, "small t-accent" if role == "accent" else "small")
    c.arrow(160, 210, 320, 150, dashed=True)
    c.arrow(440, 154, 600, 94, dashed=True)
    c.text(250, 166, "more trainable params,", "small", anchor="end")
    c.text(250, 180, "memory, time", "small", anchor="end")

    _takeaway(c, 326, "Start with the probe; climb only when validation shows a gap worth the extra cost.")
    return "l3-finetune-ladder.svg", c.render(
        "Three rising steps. Linear probe: frozen encoder, embeddings computed once, about 3,000 "
        "trained parameters. LoRA with rank 8: adapters on the query and value projections plus a "
        "new head, 741 thousand trained parameters and 263 MB of state. Full fine-tuning: all 67 "
        "million parameters and about 1 GB of state before activations."
    ), "Fine-tuning ladder: linear probe, LoRA, full fine-tuning."


@diagram
def lora():
    c = Canvas(W, 400)
    c.title("LoRA: a frozen weight plus a trainable low-rank update",
            "h = W x + (α / r) · B A x,   with r ≪ d.")

    c.box(24, 160, 90, 50, "x", "d_in = k", role="default")
    # frozen path (top)
    c.arrow(114, 175, 196, 124)
    c.box(200, 70, 130, 110, "W", "d × k, frozen", role="muted")
    # adapter path (bottom)
    c.arrow(114, 195, 196, 266, accent=True)
    c.box(200, 250, 130, 32, "A: r × k", role="accent", label_cls="small")
    c.arrow(330, 266, 384, 266, accent=True)
    c.box(388, 210, 30, 112, role="good", radius=4)
    c.text(428, 252, "B: d × r", "small", anchor="start")
    c.text(428, 268, "starts at 0", "small t-good", anchor="start")
    c.arrow(418, 230, 556, 196, accent=True)
    c.text(500, 236, "× α/r", "small t-accent")
    c.arrow(330, 125, 556, 176)
    c.circle(570, 186, 14, role="good")
    c.text(570, 191, "+", "lbl")
    c.arrow(584, 186, 626, 186, accent=True)
    c.box(630, 160, 90, 50, "h", "d_out = d", role="default")

    c.box(740, 70, 116, 110, "Trainable", "r(d + k) instead of d·k", role="accent")
    c.box(740, 196, 116, 126, "768×768, r=8", "12,288 vs 589,824: 2%", role="good")

    c.text(24, 340, "Merge after training: W′ = W + (α/r)·B·A has W's shape, so serving costs nothing extra.",
           "small", anchor="start")
    _takeaway(c, 352, "B = 0 at init, so training starts exactly at the pretrained model.")
    return "l3-lora.svg", c.render(
        "LoRA: the input x goes through the frozen weight W (d by k) and, in parallel, through a "
        "trainable r by k matrix A and a d by r matrix B that starts at zero; the adapter output is "
        "scaled by alpha over r and added. Trainable parameters per layer are r times (d plus k): "
        "12,288 instead of 589,824 for a 768 by 768 layer at rank 8."
    ), "LoRA: W x plus a scaled low-rank update B A x."


# ---------------------------------------------------------------------------
# 07-08 - retrieval
# ---------------------------------------------------------------------------


@diagram
def two_stage_retrieval():
    c = Canvas(W, 340)
    c.title("Two-stage retrieval",
            "A cheap, high-recall first stage; an expensive, precise re-ranker on a short list.")

    y, h = 110, 84
    c.box(24, y, 120, h, "Corpus", "5k-100M docs, embedded offline", role="default")
    c.arrow(144, y + h / 2, 170, y + h / 2, accent=True)
    c.box(174, y, 200, h, "Stage 1: BM25 + dense", "fused by RRF; ms per query; optimise Recall@100",
          role="accent")
    c.arrow(374, y + h / 2, 400, y + h / 2, accent=True)
    c.text(387, y + h / 2 - 8, "50", "small")
    c.box(404, y, 200, h, "Stage 2: cross-encoder", "one pass per pair; optimise nDCG@10",
          role="warn")
    c.arrow(604, y + h / 2, 630, y + h / 2, accent=True)
    c.box(634, y, 222, h, "Top 5 to the user", "or to an LLM's context window", role="good")

    c.text(274, 222, "RRF(d) = Σₛ 1 / (60 + rankₛ(d))", "mono")
    c.text(504, 222, "latency ∝ depth", "small")
    c.text(24, 254, "Stage 1 sets the ceiling: a document it misses can never be re-ranked up.",
           "small", anchor="start")

    _takeaway(c, 280, "Measure each stage on your own judged queries: off-the-shelf re-rankers can hurt (8.4).")
    return "l3-two-stage-retrieval.svg", c.render(
        "A pipeline: the corpus, embedded offline, feeds stage 1, BM25 plus a dense bi-encoder fused "
        "by reciprocal rank fusion, which returns the top 50; stage 2, a cross-encoder, scores each "
        "query-document pair and returns the top 5 to the user or an LLM."
    ), "Two-stage retrieval: hybrid first stage, cross-encoder re-ranker."


@diagram
def bi_vs_cross_encoder():
    c = Canvas(W, 380)
    c.title("Bi-encoder vs cross-encoder",
            "Independent vectors you can index, or a joint pass you cannot.")

    # Bi-encoder (left)
    c.text(24, 90, "Bi-encoder (stage 1)", "lbl", anchor="start")
    c.box(24, 104, 110, 40, "query", role="default", label_cls="small")
    c.box(24, 194, 110, 40, "document", role="default", label_cls="small")
    c.arrow(134, 124, 160, 124, accent=True)
    c.arrow(134, 214, 160, 214, accent=True)
    c.box(164, 104, 110, 40, "encoder", role="accent", label_cls="small")
    c.box(164, 194, 110, 40, "encoder", "offline, once", role="accent", label_cls="small")
    c.arrow(274, 124, 314, 156)
    c.arrow(274, 214, 314, 182)
    c.box(318, 146, 100, 46, "cosine", "q · d", role="good", label_cls="small")

    # Cross-encoder (right)
    c.text(470, 90, "Cross-encoder (stage 2)", "lbl", anchor="start")
    c.box(470, 146, 170, 46, "[CLS] q [SEP] d", role="default", label_cls="small")
    c.arrow(640, 169, 666, 169, accent=True)
    c.box(670, 136, 100, 66, "encoder", "tokens attend across", role="warn", label_cls="small")
    c.arrow(770, 169, 790, 169, accent=True)
    c.box(794, 146, 62, 46, "score", role="good", label_cls="small")

    c.box(24, 254, 394, 54, "N docs → N vectors, searched by ANN",
          "fast; misses fine-grained interactions", role="default")
    c.box(470, 254, 386, 54, "one forward per (q, d) pair",
          "usually more precise; only for a short list", role="default")

    _takeaway(c, 324, "Only the bi-encoder can precompute document vectors; the cross-encoder must see both.")
    return "l3-bi-vs-cross-encoder.svg", c.render(
        "Left: a bi-encoder encodes the query and the document separately (documents offline, once) "
        "and compares the vectors by cosine similarity. Right: a cross-encoder reads the query and "
        "document together in one transformer pass and outputs a relevance score, one pass per pair."
    ), "Bi-encoder (independent vectors) vs cross-encoder (joint pass per pair)."


@diagram
def infonce():
    c = Canvas(W, 420)
    c.title("Contrastive training with in-batch and hard negatives",
            "Each row is a classification: pick the query's own document out of all columns.")

    gx, gy, cell, B, H = 120, 120, 44, 4, 3
    for j in range(B):
        c.text(gx + j * cell + cell / 2, gy - 10, f"d{j + 1}", "small")
    for j in range(H):
        c.text(gx + (B + j) * cell + cell / 2 + 10, gy - 10, f"h{j + 1}", "small")
    c.text(gx + B * cell / 2, gy - 32, "batch documents", "small t-accent")
    c.text(gx + B * cell + 10 + H * cell / 2, gy - 32, "hard negatives", "small t-warn")
    for i in range(B):
        c.text(gx - 10, gy + i * cell + cell / 2 + 4, f"q{i + 1}", "small", anchor="end")
        for j in range(B + H):
            x = gx + j * cell + (10 if j >= B else 0)
            y = gy + i * cell
            if j < B and i == j:
                role, lab = "good", "+"
            elif j < B:
                role, lab = "muted", ""
            else:
                role, lab = "warn", ""
            c.box(x + 2, y + 2, cell - 4, cell - 4, lab, role=role, label_cls="small", radius=4)

    c.box(470, 100, 386, 60, "logits = q · d / τ", "unit vectors: cosine in [−1, 1]; τ = 0.05 (×20)",
          role="accent")
    c.box(470, 172, 386, 60, "loss = cross-entropy, target = diagonal",
          "B rows, B + H columns: every other row's positive is free", role="default")
    c.box(470, 244, 386, 60, "hard negatives: BM25 near-misses",
          "skip the top ranks: unlabelled positives (false negatives)", role="warn")

    c.box(24, 320, 18, 14, role="good", radius=3)
    c.text(50, 332, "positive", "small", anchor="start")
    c.box(130, 320, 18, 14, role="muted", radius=3)
    c.text(156, 332, "in-batch negative", "small", anchor="start")
    c.box(290, 320, 18, 14, role="warn", radius=3)
    c.text(316, 332, "hard negative", "small", anchor="start")

    _takeaway(c, 360, "Bigger batches = more negatives for free; harder negatives = more signal per row.")
    return "l3-infonce.svg", c.render(
        "A similarity matrix with four query rows and seven document columns: four batch documents "
        "whose diagonal entries are the positives and off-diagonal entries are in-batch negatives, "
        "plus three hard-negative columns mined with BM25. Logits are dot products of unit vectors "
        "divided by a temperature of 0.05, and the loss is cross-entropy with the diagonal as target."
    ), "InfoNCE: in-batch negatives plus hard negatives."


# ---------------------------------------------------------------------------
# 09 - distributed data parallel
# ---------------------------------------------------------------------------


@diagram
def ddp_step():
    c = Canvas(W, 400)
    c.title("One DDP training step",
            "Different data per rank, identical weights before and after.")

    ys = (84, 160, 236)
    for r, y in enumerate(ys):
        c.box(24, y, 120, 54, f"rank {r}", f"batch shard {r}", role="default")
        c.arrow(144, y + 27, 170, y + 27, accent=True)
        c.box(174, y, 170, 54, "forward + backward", "on the local shard", role="accent")
        c.arrow(344, y + 27, 370, y + 27, accent=True)
        c.box(374, y, 110, 54, f"local g{r}", "differs", role="warn")
        c.arrow(484, y + 27, 510, y + 27, accent=True)
        c.arrow(654, y + 27, 680, y + 27, accent=True)
        c.box(684, y, 172, 54, "optimizer step", "same g, same update", role="good")
    c.box(514, 84, 140, 206, "all-reduce", "g = (g0 + g1 + g2) / 3 on every rank", role="accent")

    c.text(24, 318, "Exact when shards are equal-sized and replicas start identical"
           " (DDP broadcasts rank 0's weights).", "small", anchor="start")
    _takeaway(c, 340, "Same maths as one GPU on an N× larger batch; DDP overlaps the all-reduce with backward.")
    return "l3-ddp-step.svg", c.render(
        "Three ranks each take a different shard of the batch, run forward and backward locally and "
        "get different local gradients; an all-reduce sums them and divides by N so every rank holds "
        "the mean gradient; each rank then takes the same optimizer step, so the replicas stay "
        "identical."
    ), "DDP: local forward/backward, all-reduce the gradients, identical update."


@diagram
def ring_allreduce():
    c = Canvas(W, 400)
    c.title("Ring all-reduce",
            "Each rank talks only to its neighbour; per-rank traffic barely grows with N.")

    pos = [(60, 96), (260, 96), (260, 236), (60, 236)]
    for r, (x, y) in enumerate(pos):
        c.box(x, y, 120, 56, f"rank {r}", "chunks a b c d", role="accent")
    c.arrow(180, 124, 256, 124, accent=True)
    c.arrow(320, 152, 320, 232, accent=True)
    c.arrow(260, 264, 184, 264, accent=True)
    c.arrow(120, 236, 120, 156, accent=True)
    c.text(220, 198, "send M/N", "small")
    c.text(220, 212, "per step", "small")

    c.box(440, 84, 416, 64, "1 · reduce-scatter: N − 1 steps",
          "add one chunk per step; each rank ends with one summed chunk",
          role="default")
    c.box(440, 160, 416, 64, "2 · all-gather: N − 1 steps",
          "pass the summed chunks on until every rank has all", role="default")
    c.box(440, 236, 416, 64, "bytes sent per rank = 2(N − 1)/N · M",
          "≈ 2M for large N · 7B fp16 grads, N = 8: 24.5 GB", role="warn")

    _takeaway(c, 334, "Bandwidth cost is ~2M whatever N is; latency grows with N and the slowest link sets the pace.")
    return "l3-ring-allreduce.svg", c.render(
        "Four ranks in a ring, each sending to its neighbour. A ring all-reduce is a reduce-scatter "
        "of N minus 1 steps followed by an all-gather of N minus 1 steps, each step sending one chunk "
        "of size M over N, so each rank sends 2(N-1)/N times M bytes."
    ), "Ring all-reduce: reduce-scatter then all-gather, 2(N−1)/N · M bytes per rank."


# ---------------------------------------------------------------------------
# 10 - scaling to large models
# ---------------------------------------------------------------------------


@diagram
def zero_stages():
    c = Canvas(W, 420)
    c.title("Model-state memory per GPU: DDP vs ZeRO stages",
            "Mixed-precision Adam, Ψ parameters, N = 4 GPUs. Activations not shown.")

    x0, scale = 220, 28  # px per byte/param
    n = 4
    rows = [("DDP (stage 0)", "everything replicated", 2, 2, 12, "16Ψ"),
            ("ZeRO-1", "shard optimizer state", 2, 2, 12 / n, "7Ψ"),
            ("ZeRO-2", "+ shard gradients", 2, 2 / n, 12 / n, "5.5Ψ"),
            ("ZeRO-3 / FSDP", "+ shard parameters", 2 / n, 2 / n, 12 / n, "4Ψ")]
    y = 112
    c.box(x0, 82, 18, 14, role="accent", radius=3)
    c.text(x0 + 24, 94, "fp16 params (2)", "small", anchor="start")
    c.box(x0 + 150, 82, 18, 14, role="warn", radius=3)
    c.text(x0 + 174, 94, "fp16 grads (2)", "small", anchor="start")
    c.box(x0 + 300, 82, 18, 14, role="good", radius=3)
    c.text(x0 + 324, 94, "fp32 master + m + v (12)", "small", anchor="start")
    for name, sub, p, g, o, total in rows:
        c.text(24, y + 18, name, "lbl", anchor="start")
        c.text(24, y + 34, sub, "small", anchor="start")
        x = x0
        for val, role in ((p, "accent"), (g, "warn"), (o, "good")):
            c.box(x, y + 4, val * scale, 32, role=role, radius=3)
            x += val * scale
        c.text(x + 10, y + 26, total, "lbl", anchor="start")
        y += 52

    c.text(24, 344, "7B on 8 GPUs: 112 → 38.5 → 26.2 → 14.0 GB. Stages 1-2 plateau at 4Ψ and 2Ψ;"
           " only stage 3 keeps shrinking with N.", "small", anchor="start")
    _takeaway(c, 362, "Stage 3 pays for it in communication: gather each layer's params just in time, twice.")
    return "l3-zero-stages.svg", c.render(
        "Horizontal stacked bars of model-state bytes per parameter on each of 4 GPUs. DDP replicates "
        "2 bytes of fp16 parameters, 2 of fp16 gradients and 12 of fp32 optimizer state: 16. ZeRO-1 "
        "shards the optimizer state: 7. ZeRO-2 also shards gradients: 5.5. ZeRO-3 or FSDP also "
        "shards parameters: 4."
    ), "ZeRO stages 1/2/3: what is sharded and the per-GPU memory."


@diagram
def tensor_parallel_mlp():
    c = Canvas(W, 400)
    c.title("Tensor parallelism: one MLP split across two GPUs",
            "Column-parallel first Linear, row-parallel second: one all-reduce per MLP.")

    c.box(24, 160, 100, 60, "x", "full, on both GPUs", role="default")
    for g, y in enumerate((90, 230)):
        c.text(150, y - 8, f"GPU {g}", "small t-accent", anchor="start")
        c.box(150, y, 170, 56, f"W1 columns half {g + 1}", "→ GELU (elementwise)", role="accent")
        c.arrow(320, y + 28, 356, y + 28, accent=True)
        c.box(360, y, 150, 56, f"W2 rows half {g + 1}", role="accent")
        c.arrow(510, y + 28, 546, y + 28, accent=True)
        c.box(550, y, 120, 56, "partial y", "full shape", role="warn")
        c.arrow(670, y + 28, 700, 190 + (-14 if g == 0 else 14))
    c.arrow(124, 180, 146, 124, accent=True)
    c.arrow(124, 200, 146, 252, accent=True)
    c.box(704, 150, 152, 80, "all-reduce (sum)", "→ y on both GPUs", role="good")

    c.text(24, 322, "Activations are communicated on the critical path, twice per layer in forward and in backward:"
           " keep TP inside an NVLink node.", "small", anchor="start")
    _takeaway(c, 340, "Tensor parallel splits the math of a layer; pipeline splits the layers; data splits the batch.")
    return "l3-tensor-parallel-mlp.svg", c.render(
        "The input x is replicated on two GPUs. Each GPU multiplies by half of the first weight's "
        "columns and applies GELU locally, then by the matching half of the second weight's rows, "
        "producing a full-shape partial output. One all-reduce sums the partial outputs so both GPUs "
        "hold y."
    ), "Megatron-style tensor parallelism of a transformer MLP."


@diagram
def pipeline_bubble():
    c = Canvas(W, 380)
    c.title("Pipeline parallelism and the bubble",
            "GPipe, p = 4 stages, m = 4 micro-batches; backward takes 2 units.")

    p, m, fwd, bwd = 4, 4, 1, 2
    x0, y0, u, rh = 110, 84, 32, 38
    end_f = (p - 1 + m) * fwd
    total = end_f + (p - 1 + m) * bwd
    for s in range(p):
        y = y0 + s * (rh + 8)
        c.text(x0 - 12, y + rh / 2 + 4, f"GPU {s}", "small", anchor="end")
        c.box(x0, y, total * u, rh, role="ghost", radius=4)
        for i in range(m):
            t = (s + i) * fwd
            c.box(x0 + t * u + 1, y + 2, fwd * u - 2, rh - 4, f"F{i + 1}", role="accent",
                  label_cls="small", radius=3)
        for j in range(m):
            t = end_f + ((p - 1 - s) + j) * bwd
            c.box(x0 + t * u + 1, y + 2, bwd * u - 2, rh - 4, f"B{m - j}", role="good",
                  label_cls="small", radius=3)
    c.text(x0 + 2 * u, y0 + 3 * (rh + 8) + rh / 2 + 4, "idle", "small t-bad")
    c.text(x0 + 9.5 * u, y0 + rh / 2 + 4, "idle", "small t-bad")
    c.text(x0, y0 + 4 * (rh + 8) + 16, "time →", "small", anchor="start")

    c.text(24, 312, "bubble = (p − 1) / (m + p − 1) = 3/7 ≈ 0.43 here; m = 16 → 0.16.  Rule of thumb: m ≥ 4p.",
           "lbl", anchor="start")
    _takeaway(c, 330, "Cheap point-to-point communication (works across nodes); the price is idle time.")
    return "l3-pipeline-bubble.svg", c.render(
        "A schedule grid for four pipeline stages and four micro-batches. Forward passes form a "
        "staircase down the stages, backward passes a reversed staircase; the empty triangles at the "
        "start and end are idle time, the bubble, a fraction (p-1)/(m+p-1) = 3/7 of the step."
    ), "Pipeline parallelism: GPipe schedule and the bubble."


@diagram
def scaling_frontier():
    c = Canvas(W, 420)
    c.title("Compute-optimal model size",
            "For each compute budget, loss vs model size is a U; the minima trace the frontier.")

    x0, x1, y0, y1 = 80, 500, 80, 320
    _hline(c, x0, x1, y1)
    _vline(c, x0, y0, y1)
    c.text((x0 + x1) / 2, y1 + 22, "model size N (log) — fixed C, so more N means fewer tokens D", "small")
    c.text(x0 - 10, y0 + 10, "loss", "small", anchor="end")

    mins = []
    for k, (tone, lab) in enumerate((("muted_fg", "C"), ("accent", "10C"), ("good", "100C"))):
        xm, ym, a = 170 + 120 * k, 200 + 45 * k, 0.012
        pts = [(xm - 90 + 180 * i / 40, ym - a * (-90 + 180 * i / 40) ** 2) for i in range(41)]
        c.curve("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts), tone=tone)
        mins.append((xm, ym))
        c.text(pts[-1][0] + 6, pts[-1][1] + 4, lab, "small", anchor="start")
    c.curve("M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in mins), tone="bad", dashed=True, width=2)
    for x, y in mins:
        c.circle(x, y, 5, role="bad")
    c.text(mins[-1][0] + 12, mins[-1][1] + 18, "frontier", "small t-bad", anchor="start")

    c.box(540, 80, 316, 56, "C ≈ 6 · N · D", "FLOPs: 2N forward + 4N backward per token", role="accent")
    c.box(540, 148, 316, 56, "Chinchilla: D ≈ 20 · N", "N and D both grow as √C", role="good")
    c.box(540, 216, 316, 56, "N = √(C / 120)", "5.9e23 FLOPs → 70B params, 1.4T tokens", role="default")
    c.box(540, 284, 316, 46, "Serving-heavy? Train past 20 D/N", role="warn", label_cls="small")

    _takeaway(c, 360, "Given a budget, size the model and data together; a bigger model on too few tokens wastes compute.")
    return "l3-scaling-frontier.svg", c.render(
        "Three U-shaped curves of loss against model size for compute budgets C, 10C and 100C; the "
        "minimum of each moves to larger models and lower loss, and a dashed line through the minima "
        "is the compute-optimal frontier. Beside it: C is about 6 N D, Chinchilla's D is about 20 N, "
        "so N equals the square root of C over 120."
    ), "Scaling laws: IsoFLOP curves and the compute-optimal frontier."


@diagram
def mixed_precision():
    c = Canvas(W, 400)
    c.title("fp16 range and loss scaling",
            "Tiny gradients underflow to zero in fp16; multiplying the loss by S lifts them into range.")

    x0, x1 = 80, 800
    e_lo, e_hi = -10, 6

    def px(e):
        return x0 + (e - e_lo) / (e_hi - e_lo) * (x1 - x0)

    ya = 250
    _hline(c, x0, x1, ya)
    for e in range(e_lo, e_hi + 1, 2):
        _vline(c, px(e), ya - 4, ya + 4)
        c.text(px(e), ya + 18, f"1e{e}", "small")
    c.text((x0 + x1) / 2, ya + 38, "magnitude of a gradient value (log scale)", "small")

    # fp16 zones
    c.box(x0, 268 + 34, px(math.log10(6e-8)) - x0, 22, "→ 0", role="bad", label_cls="small", radius=3)
    c.box(px(math.log10(6e-8)), 302, px(math.log10(6.1e-5)) - px(math.log10(6e-8)), 22,
          "subnormal", role="warn", label_cls="small", radius=3)
    c.box(px(math.log10(6.1e-5)), 302, px(math.log10(65504)) - px(math.log10(6.1e-5)), 22,
          "fp16 normal range", role="good", label_cls="small", radius=3)
    c.box(px(math.log10(65504)), 302, x1 - px(math.log10(65504)), 22, "inf", role="bad",
          label_cls="small", radius=3)
    c.text(x0 - 8, 318, "fp16", "small", anchor="end")

    def bump(center, height=110):
        pts = []
        for i in range(61):
            e = center - 3 + 6 * i / 60
            yv = ya - 4 - height * math.exp(-((e - center) ** 2) / 1.3)
            pts.append(f"{px(e):.1f} {yv:.1f}")
        return "M " + " L ".join(pts)

    c.curve(bump(-6.2), tone="bad", dashed=True)
    c.text(px(-6.2), 102, "raw gradients", "small t-bad")
    c.text(px(-6.2), 116, "median ~1e-6", "small t-bad")
    c.curve(bump(-1.4), tone="good")
    c.text(px(-1.4), 102, "× S = 2¹⁶", "small t-good")
    c.text(px(-1.4), 116, "all representable", "small t-good")
    c.arrow(px(-4.6), 170, px(-3.0), 170, accent=True)

    _takeaway(c, 340, "Scale up, unscale in fp32 before the step, halve S on inf. bf16 has fp32's range: no scaler.")
    return "l3-mixed-precision.svg", c.render(
        "A log-scale number line of gradient magnitudes. In fp16, values below about 6e-8 become "
        "zero, values up to 6e-5 are imprecise subnormals, and values above 65,504 overflow to "
        "infinity. Raw gradients centred around 1e-6 partly fall into the underflow zones; "
        "multiplying the loss by 2 to the 16 shifts them into the normal range."
    ), "fp16 underflow and loss scaling."
