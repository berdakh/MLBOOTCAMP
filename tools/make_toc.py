#!/usr/bin/env python3
"""Generate the tables of contents in README.md and each part's README.md.

The text between ``<!-- toc:start -->`` and ``<!-- toc:end -->`` is replaced
with a table built from the chapter folders (tools/book.py), so the contents
can never disagree with what is on disk.

    python tools/make_toc.py          # rewrite the tables
    python tools/make_toc.py --check  # fail if any table is stale (CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from book import CAPSTONES_DIR, PARTS, REPO_ROOT, chapters  # noqa: E402

COLAB = "https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
START, END = "<!-- toc:start -->", "<!-- toc:end -->"


def minutes(ch) -> str:
    text = ch.source.read_text(encoding="utf-8")
    m = re.search(r"about (\d+) minutes|~(\d+) min", text)
    return f"{m.group(1) or m.group(2)} min" if m else ""


def chapter_rows(part: str, base: Path) -> list[str]:
    rows = ["| Ch. | Chapter | Lab | Time |", "|---:|---|---|---|"]
    for ch in chapters(part):
        rel = Path(ch.folder).relative_to(base).as_posix() if base != REPO_ROOT else ch.rel
        text = f"[{ch.title}]({rel}/README.md)" if (ch.folder / "README.md").exists() else ch.title
        lab = f"[notebook]({rel}/notebook.ipynb) · [![Colab]({BADGE})]({COLAB}{ch.rel}/notebook.ipynb)"
        rows.append(f"| {ch.number} | {text} | {lab} | {minutes(ch)} |")
    return rows


def capstone_rows(base: Path) -> list[str]:
    rows = ["| | Capstone | Guided solution |", "|---|---|---|"]
    for folder in sorted(CAPSTONES_DIR.glob("cap[0-9]-*")):
        readme = folder / "README.md"
        title = readme.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() if readme.exists() else folder.name
        rel = folder.relative_to(base).as_posix() if base != REPO_ROOT else folder.relative_to(REPO_ROOT).as_posix()
        walk = (f"[walkthrough]({rel}/walkthrough.ipynb)"
                if (folder / "walkthrough.py").exists() else "in preparation")
        rows.append(f"| {folder.name[3]} | [{title}]({rel}/README.md) | {walk} |")
    return rows


def full_toc() -> list[str]:
    out = []
    for part_dir, title, _ in PARTS:
        out += [f"### [{title}]({part_dir}/README.md)", ""] + chapter_rows(part_dir, REPO_ROOT) + [""]
    out += ["### [Part IV · Capstone projects](part4-capstones/README.md)", ""] + capstone_rows(REPO_ROOT) + [""]
    return out


def targets() -> dict[Path, list[str]]:
    t = {REPO_ROOT / "README.md": full_toc()}
    for part_dir, _, _ in PARTS:
        base = REPO_ROOT / part_dir
        t[base / "README.md"] = chapter_rows(part_dir, base)
    t[CAPSTONES_DIR / "README.md"] = capstone_rows(CAPSTONES_DIR)
    return t


def render(path: Path, lines: list[str]) -> str | None:
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return None
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + START + "\n" + "\n".join(lines).rstrip() + "\n" + END + tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for path, lines in targets().items():
        new = render(path, lines)
        if new is None:
            print(f"note: no toc markers in {path.relative_to(REPO_ROOT)}", file=sys.stderr)
            continue
        if args.check:
            if path.read_text(encoding="utf-8") != new:
                stale.append(path.relative_to(REPO_ROOT).as_posix())
        else:
            path.write_text(new, encoding="utf-8")
    if stale:
        print("stale tables of contents (run python tools/make_toc.py): " + ", ".join(stale), file=sys.stderr)
        return 1
    print("tables of contents up to date" if args.check else "tables of contents written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
