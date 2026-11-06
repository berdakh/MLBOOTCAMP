#!/usr/bin/env python3
"""Generate the course diagrams as self-contained SVG.

Hand-written SVG rather than images from the web: nothing to license, nothing
to rot, reviewable in a diff, and readable in light or dark mode.

    python tools/make_diagrams.py          # write docs/assets/diagrams/*.svg
    python tools/make_diagrams.py --check  # fail if the files are stale (CI)
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "diagrams"

STYLE = """<style>
  .bg{fill:#ffffff} .box{fill:#f4f6fb;stroke:#3b4a6b;stroke-width:1.5}
  .accent{fill:#e3f0ff;stroke:#1f6feb;stroke-width:1.8} .warn{fill:#fff1e6;stroke:#c2410c;stroke-width:1.5}
  .ok{fill:#e8f7ee;stroke:#15803d;stroke-width:1.5} .muted{fill:#f8f8f8;stroke:#9aa3b2;stroke-width:1;stroke-dasharray:4 3}
  .t{font:600 14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#1b2333}
  .s{font:12px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#4a5568}
  .h{font:700 17px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#1b2333}
  .line{stroke:#3b4a6b;stroke-width:1.6;fill:none} .arrowhead{fill:#3b4a6b}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0f1420} .box{fill:#1a2233;stroke:#8da2c9} .accent{fill:#132a4a;stroke:#58a6ff}
    .warn{fill:#3a2414;stroke:#f0883e} .ok{fill:#12301f;stroke:#3fb950} .muted{fill:#161b26;stroke:#5c667a}
    .t,.h{fill:#e6edf3} .s{fill:#a9b4c4} .line{stroke:#8da2c9} .arrowhead{fill:#8da2c9}
  }
</style>"""


class Svg:
    def __init__(self, w: int, h: int, title: str):
        self.w, self.h, self.title = w, h, title
        self.parts: list[str] = [f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>']

    def box(self, x, y, w, h, title, sub="", cls="box"):
        self.parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>')
        lines = sub.split("\n") if sub else []
        ty = y + h / 2 - 7 * len(lines) + (5 if not lines else 0)
        self.parts.append(f'<text class="t" x="{x + w / 2}" y="{ty}" text-anchor="middle">{escape(title)}</text>')
        for i, line in enumerate(lines):
            self.parts.append(f'<text class="s" x="{x + w / 2}" y="{ty + 18 + 15 * i}" text-anchor="middle">{escape(line)}</text>')

    def text(self, x, y, s, cls="s", anchor="start"):
        self.parts.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{escape(s)}</text>')

    def arrow(self, x1, y1, x2, y2, label=""):
        self.parts.append(f'<path class="line" d="M{x1},{y1} L{x2},{y2}" marker-end="url(#ah)"/>')
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 - 6, label, anchor="middle")

    def path(self, d):
        self.parts.append(f'<path class="line" d="{d}" marker-end="url(#ah)"/>')

    def rect(self, x, y, w, h, cls):
        self.parts.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="3"/>')

    def render(self) -> str:
        defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse"><path class="arrowhead" d="M0,0 L10,5 L0,10 z"/>'
                "</marker></defs>")
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-label="{escape(self.title)}"><title>{escape(self.title)}</title>{STYLE}{defs}'
                + "".join(self.parts) + "</svg>\n")


def roadmap() -> str:
    s = Svg(1000, 300, "Five-day roadmap")
    s.text(500, 34, "Five days, one project: TaskFlow churn prediction", "h", "middle")
    days = [
        ("Day 1", "Reproducible\nprojects", "env · data contracts\nSQL · splits · baseline"),
        ("Day 2", "Evaluate &\nimprove", "pipelines · CV\nthresholds · artifacts"),
        ("Day 3", "Serve\npredictions", "FastAPI · contracts\nclients · tests"),
        ("Day 4", "Deploy &\noperate", "config · logs · latency\nmonitoring · rollback"),
        ("Day 5", "Demonstrate", "held-out test · docs\ndrills · mock interview"),
    ]
    for i, (d, title, sub) in enumerate(days):
        x = 20 + i * 196
        s.box(x, 70, 170, 110, d, title, "accent" if i in (0, 4) else "box")
        s.text(x + 85, 210, sub.split("\n")[0], anchor="middle")
        s.text(x + 85, 228, sub.split("\n")[1], anchor="middle")
        if i < 4:
            s.arrow(x + 170, 125, x + 194, 125)
    s.text(500, 275, "Then: five independent capstones, one at a time, on public data.", "s", "middle")
    return s.render()


def architecture() -> str:
    s = Svg(1000, 330, "Project architecture")
    s.text(500, 32, "The running project, end to end", "h", "middle")
    steps = [("Raw extract", "4 tables, CSV.gz\nSHA-256 verified"), ("Validate", "contracts\n+ cleaning"),
             ("SQL features", "point-in-time\nDuckDB"), ("Split", "time-based\n+ embargo"),
             ("Pipeline", "preprocess\n+ model"), ("Registry", "v1 v2 v3\nCURRENT")]
    for i, (t, sub) in enumerate(steps):
        x = 20 + i * 163
        s.box(x, 60, 140, 80, t, sub, "accent" if t in ("SQL features", "Registry") else "box")
        if i < len(steps) - 1:
            s.arrow(x + 140, 100, x + 161, 100)
    s.box(670, 210, 150, 80, "FastAPI service", "/predict /ready\n/metrics", "ok")
    s.box(860, 210, 120, 80, "Monitoring", "drift · latency\nlabels later", "warn")
    s.box(470, 210, 160, 80, "Client", "timeouts · retries\nrequest ids")
    s.arrow(910, 140, 760, 208)
    s.text(850, 170, "loads CURRENT", anchor="start")
    s.text(850, 186, "(rollback = move the pointer)", anchor="start")
    s.arrow(630, 250, 668, 250)
    s.arrow(820, 250, 858, 250)
    s.box(20, 210, 400, 80, "Days 1-2: data → model", "Days 3-4: model → service → operations", "muted")
    return s.render()


def point_in_time() -> str:
    s = Svg(1000, 320, "Point-in-time features and labels")
    s.text(500, 32, "One snapshot T: features look back, the label looks forward", "h", "middle")
    y = 185
    s.parts.append(f'<path class="line" d="M40,{y} L960,{y}" marker-end="url(#ah)"/>')
    s.text(960, y + 24, "time", anchor="end")
    s.rect(240, y - 50, 300, 34, "accent")
    s.text(390, y - 28, "features: rows with timestamp < T", "t", "middle")
    s.rect(130, y - 90, 410, 30, "muted")
    s.text(335, y - 70, "90-day ticket / payment windows", "s", "middle")
    s.rect(560, y + 16, 200, 34, "ok")
    s.text(660, y + 38, "label: churn in [T, T+30d)", "t", "middle")
    s.parts.append(f'<path class="line" d="M550,{y - 100} L550,{y + 70}" stroke-dasharray="5 4"/>')
    s.text(550, y - 106, "snapshot T", "t", "middle")
    s.rect(470, y + 16, 70, 34, "warn")
    s.text(460, y + 30, "a usage week that ends", anchor="end")
    s.text(460, y + 46, "after T is excluded", anchor="end")
    s.text(40, 300, "Leakage = any feature computed from rows at or after T (including 'status' columns exported today).", "s")
    return s.render()


def splits() -> str:
    s = Svg(1000, 330, "Three ways to split")
    s.text(500, 32, "Same rows, three splits, three different questions", "h", "middle")
    labels = [("Random rows", "optimistic: mixes past & future,\nsame customer on both sides", "warn"),
              ("Group (customer)", "answers: 'how well on customers\nwe have never seen?'", "box"),
              ("Time + embargo", "answers: 'how well next month?'\n← this project's use case", "ok")]
    for r, (name, sub, cls) in enumerate(labels):
        y = 70 + r * 85
        s.box(20, y, 210, 66, name, "", cls)
        for i in range(21):
            x = 250 + i * 25
            if r == 0:
                c = ["accent", "accent", "warn", "accent", "ok", "accent", "accent"][(i * 5) % 7]
            elif r == 1:
                c = "accent" if i % 5 not in (1, 3) else "ok"
            else:
                c = "accent" if i < 12 else ("muted" if i == 12 else ("ok" if i < 16 else "warn"))
            s.rect(x, y + 14, 20, 38, c)
        lines = sub.split("\n")
        s.text(790, y + 28, lines[0])
        s.text(790, y + 46, lines[1])
    for i, (cls, name) in enumerate([("accent", "train"), ("ok", "validation"), ("warn", "test"),
                                     ("muted", "embargo gap: label window crosses the boundary")]):
        x = 250 + i * 120
        s.rect(x, 306, 14, 14, cls)
        s.text(x + 20, 318, name)
    return s.render()


def serving() -> str:
    s = Svg(1000, 330, "Serving architecture")
    s.text(500, 32, "Inside the prediction service", "h", "middle")
    s.box(20, 120, 150, 80, "Caller", "JSON request\nx-request-id")
    s.rect(220, 70, 230, 180, "accent")
    s.text(335, 98, "FastAPI app", "t", "middle")
    s.text(335, 130, "middleware: id · timing · log", anchor="middle")
    s.text(335, 160, "Pydantic contract → 422", anchor="middle")
    s.text(335, 190, "/predict  /predict/batch", anchor="middle")
    s.text(335, 220, "/health  /ready  /metrics", anchor="middle")
    s.box(500, 120, 180, 80, "ModelHolder", "pipeline in memory\nswap under a lock")
    s.box(730, 60, 240, 80, "Registry", "immutable v1 v2 v3\nCURRENT pointer", "ok")
    s.box(730, 180, 240, 80, "Logs & metrics", "JSON lines · p50/p95\nrecent inputs → drift", "warn")
    s.arrow(170, 160, 218, 160)
    s.arrow(450, 160, 498, 160)
    s.arrow(728, 100, 682, 140)
    s.text(600, 98, "load at start / reload", anchor="middle")
    s.arrow(682, 175, 728, 210)
    s.text(500, 300, "The saved object includes preprocessing, so the service cannot transform inputs differently from training.", "s", "middle")
    return s.render()


def monitoring() -> str:
    s = Svg(1000, 320, "Monitoring and rollback")
    s.text(500, 32, "What you can know, and when", "h", "middle")
    cols = [("Service health", "now", "errors · latency · 5xx\nreadiness", "ok"),
            ("Data & score drift", "now", "PSI per feature · unseen\ncategories · score mean", "accent"),
            ("Model performance", "30+ days later", "PR-AUC on matured labels\nprecision of flagged", "warn")]
    for i, (t, when, sub, cls) in enumerate(cols):
        x = 30 + i * 320
        s.box(x, 60, 290, 130, t, f"available: {when}\n" + sub, cls)
    s.box(330, 230, 340, 70, "Decision", "page → rollback (pointer move)\nwarn → investigate / retrain", "box")
    s.arrow(175, 192, 380, 228)
    s.arrow(495, 192, 500, 228)
    s.arrow(815, 192, 620, 228)
    return s.render()


DIAGRAMS = {
    "roadmap.svg": roadmap,
    "architecture.svg": architecture,
    "point-in-time.svg": point_in_time,
    "splits.svg": splits,
    "serving.svg": serving,
    "monitoring.svg": monitoring,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, fn in DIAGRAMS.items():
        text = fn()
        path = OUT / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(OUT.parent.parent.parent)}")
    if args.check:
        if stale:
            print("stale diagrams (run python tools/make_diagrams.py): " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"all {len(DIAGRAMS)} diagrams up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
