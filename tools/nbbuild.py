#!/usr/bin/env python3
"""Build each chapter's notebook.ipynb from its notebook.py (percent format).

Why: notebook JSON is unreadable in a diff, so the source of truth for every lab
is a plain Python file next to it, and this script renders the .ipynb that
learners open. Also builds the capstone walkthroughs.

    python tools/nbbuild.py              # build everything
    python tools/nbbuild.py --check      # fail if any notebook is stale (CI)
    python tools/nbbuild.py ch06 cap1    # only paths containing these substrings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book import all_sources  # noqa: E402

KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO = {
    "codemirror_mode": {"name": "ipython", "version": 3},
    "file_extension": ".py",
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
    "pygments_lexer": "ipython3",
}


def _as_source_lines(text: str) -> list[str]:
    """Split a cell body into nbformat's list-of-lines-with-newlines shape."""
    text = text.rstrip("\n")
    if not text:
        return []
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _strip_markdown_comments(body: str) -> str:
    """Turn ``# some text`` comment lines back into raw markdown."""
    out = []
    for line in body.split("\n"):
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            # A markdown cell should be fully commented; keep anything odd
            # verbatim rather than silently dropping content.
            out.append(line)
    return "\n".join(out)


def parse_percent(text: str) -> list[dict]:
    """Parse percent-format text into a list of nbformat cell dicts."""
    cells: list[dict] = []
    current_kind: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_kind is None:
            return
        body = "\n".join(buffer).strip("\n")
        if not body.strip():
            return
        if current_kind == "markdown":
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _as_source_lines(_strip_markdown_comments(body)),
                }
            )
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": _as_source_lines(body),
                }
            )

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            buffer = []
            current_kind = "markdown" if "[markdown]" in stripped else "code"
            continue
        if current_kind is None:
            # Anything before the first cell marker (module docstring, etc.)
            continue
        buffer.append(line)

    flush()
    return cells


def build_notebook(src: Path) -> dict:
    cells = parse_percent(src.read_text(encoding="utf-8"))
    # Stable cell ids (nbformat 4.5 requires them): position-based, so a
    # rebuild of unchanged source is byte-identical.
    for number, cell in enumerate(cells):
        cell["id"] = f"{src.parent.name[:4]}-{number:03d}"
    if not cells:
        raise ValueError(f"{src.name} produced no cells - is it percent-formatted?")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": LANGUAGE_INFO,
            # Colab: open with the table of contents visible.
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*", help="build only sources matching these substrings")
    parser.add_argument("--check", action="store_true", help="verify notebooks are up to date")
    args = parser.parse_args()

    pairs = [(src, src.with_suffix(".ipynb")) for src in all_sources()]
    if args.only:
        pairs = [(s, o) for s, o in pairs
                 if any(token in s.relative_to(REPO_ROOT).as_posix() for token in args.only)]
    if not pairs:
        print("no notebook sources matched", file=sys.stderr)
        return 1

    stale: list[str] = []
    for src, out in pairs:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = render(build_notebook(src))
        if args.check:
            if not out.exists() or out.read_text(encoding="utf-8") != text:
                stale.append(str(out.relative_to(REPO_ROOT)))
            continue
        out.write_text(text, encoding="utf-8")
        n_cells = len(json.loads(text)["cells"])
        print(f"built {out.relative_to(REPO_ROOT)}  ({n_cells} cells)")
    sources = pairs

    if args.check:
        if stale:
            print("stale notebooks (run `python tools/nbbuild.py`):", file=sys.stderr)
            for name in stale:
                print(f"  - {name}", file=sys.stderr)
            return 1
        print(f"all {len(sources)} notebooks up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
