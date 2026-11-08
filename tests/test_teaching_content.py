"""Guard the teaching scaffolding that every notebook promises.

* Every notebook has a "📖 Words to know" glossary table near the top.
* Display maths uses ``$$`` on its own line, balanced within each markdown
  cell - an unbalanced ``$$`` swallows the rest of the cell in Jupyter, Colab
  and GitHub alike, and nothing else would notice.
* Every figure in docs/assets/diagrams is used somewhere (a notebook, a doc,
  or a README), so the index never advertises a figure no lesson shows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from book import all_sources  # noqa: E402

ALL_SOURCES = all_sources()


def markdown_cells(src: Path) -> list[str]:
    """Markdown cells of a percent-format source, with the leading '# ' removed."""
    cells, current, inside = [], [], False
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            if inside:
                cells.append("\n".join(current))
            current, inside = [], "[markdown]" in line
            continue
        if inside:
            current.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
    if inside:
        cells.append("\n".join(current))
    return cells


def test_sources_found():
    assert len(ALL_SOURCES) >= 58


@pytest.mark.parametrize("src", ALL_SOURCES, ids=lambda p: p.parent.name[:5])
def test_every_notebook_has_a_glossary(src):
    text = src.read_text(encoding="utf-8")
    assert "## 📖 Words to know" in text, f"{src.name}: add a '## 📖 Words to know' table"
    section = text.split("## 📖 Words to know", 1)[1].split("# %%", 1)[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("# |") and "---" not in ln]
    assert len(rows) >= 6, f"{src.name}: the glossary should define at least 5 terms"


@pytest.mark.parametrize("src", ALL_SOURCES, ids=lambda p: p.parent.name[:5])
def test_display_maths_is_balanced(src):
    for number, cell in enumerate(markdown_cells(src), 1):
        fences = [ln for ln in cell.splitlines() if ln.strip() == "$$"]
        inline_pairs = sum(ln.count("$$") for ln in cell.splitlines() if ln.strip() != "$$")
        assert len(fences) % 2 == 0, f"{src.name}: markdown cell {number} has an unbalanced $$ block"
        assert inline_pairs % 2 == 0, f"{src.name}: markdown cell {number} has an unpaired $$"


def test_every_diagram_is_used():
    figures = {p.name for p in (ROOT / "figures").glob("*.svg")}
    texts = [p.read_text(encoding="utf-8") for p in ALL_SOURCES]
    for pattern in ("*.md", "part*/**/*.md", "appendices/*.md"):
        texts += [p.read_text(encoding="utf-8") for p in ROOT.glob(pattern)
                  if p.name != "figures.md"]
    corpus = "\n".join(texts)
    unused = sorted(f for f in figures if not re.search(re.escape(f), corpus))
    assert not unused, f"figures not embedded anywhere: {unused}"
