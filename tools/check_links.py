#!/usr/bin/env python3
"""Check that every relative link in the markdown files points at a real file.

Covers README.md, CONTRIBUTING.md and the markdown in every part, the
appendices, data/ and deploy/. The docs are read directly on
GitHub (there is no generated site), so a link must name the real file:
``guides/setup.md``, not ``guides/setup.html``.

External links (http/https) are not fetched - that would make the check slow
and flaky.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG = re.compile(r'<img[^>]+src="([^"]+)"')
FENCE = re.compile(r"^\s*(```|~~~)")


def strip_code(text: str) -> str:
    """Blank out fenced code blocks before looking for links.

    Python is full of ``thing[key](args)``, which is exactly the shape of a
    markdown link. Since code is full of such shapes, the
    checker reported `allowed[type(node.op)](evaluate(node.left)` as a broken
    link to a file called `evaluate(node.left`. Lines are replaced rather than
    removed so reported line numbers would still line up.
    """
    out, inside = [], False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return "\n".join(out)


def resolve(source: Path, target: str) -> bool:
    """Is `target`, written inside `source`, a file that exists?"""
    clean = target.split("#")[0]
    if not clean:
        return True  # a pure anchor
    base = source.parent
    candidates = [base / clean]
    return any(candidate.exists() for candidate in candidates)


def main() -> int:
    files = [REPO_ROOT / "README.md", REPO_ROOT / "CONTRIBUTING.md"]
    for folder in ("data", "deploy", "appendices", "part1-foundations", "part2-ml-at-scale",
                   "part3-deep-learning-systems", "part4-capstones"):
        files += sorted((REPO_ROOT / folder).rglob("*.md"))

    problems: list[str] = []
    checked = 0
    for path in files:
        if not path.exists():
            continue
        content = strip_code(path.read_text(encoding="utf-8"))
        for label, target in LINK.findall(content):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            checked += 1
            if not resolve(path, target):
                problems.append(f"{path.relative_to(REPO_ROOT)}: [{label}]({target})")
        for target in IMG.findall(content):
            if target.startswith(("http://", "https://", "data:")):
                continue
            checked += 1
            if not (path.parent / target).resolve().exists():
                problems.append(f"{path.relative_to(REPO_ROOT)}: <img src={target}>")

    print(f"checked {checked} relative links across {len(files)} files")
    if problems:
        print(f"\n{len(problems)} broken link(s):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("all relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
