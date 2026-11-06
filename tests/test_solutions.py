"""Keep exercises, graders and solutions in sync.

* every ``> **Exercise K.**`` in notebook NN has a ``## NN-K`` solution heading;
* every notebook with a Checkpoint has a ``## Checkpoint · NN`` answer section;
* every grader ``exK`` in ``churnlab.grading.nbNN`` ACCEPTS the reference
  solution ``solutions.code.nbNN.exK`` - a grader that fails its own solution
  would teach learners the wrong thing;
* every grader reports TODO (not a crash) for an untouched stub.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebook_src"
SOLUTIONS = ROOT / "solutions"


def sources():
    return sorted(p for p in SRC.glob("*.py") if not p.name.startswith("_"))


def solution_file(src: Path) -> Path:
    return SOLUTIONS / f"{src.stem}.md"


def exercises():
    out = []
    for src in sources():
        for m in re.finditer(r"\*\*Exercise (\d+)\.\*\*", src.read_text(encoding="utf-8")):
            out.append((src, m.group(1)))
    return out


def grader_pairs():
    pairs = []
    for path in sorted((ROOT / "src" / "churnlab" / "grading").glob("nb*.py")):
        module = importlib.import_module(f"churnlab.grading.{path.stem}")
        for name in sorted(n for n in dir(module) if re.fullmatch(r"ex\d+", n)):
            pairs.append((path.stem, name))
    return pairs


def test_there_are_exercises():
    assert len(exercises()) >= 20


@pytest.mark.parametrize("src,number", exercises(), ids=lambda x: getattr(x, "stem", x))
def test_every_exercise_has_a_written_solution(src, number):
    path = solution_file(src)
    assert path.exists(), f"missing {path.relative_to(ROOT)}"
    nn = src.name[:2]
    assert re.search(rf"^## {nn}-{number} ", path.read_text(encoding="utf-8"), re.M), (
        f"{src.name} exercise {number} has no '## {nn}-{number} · ...' heading in {path.name}")


@pytest.mark.parametrize("src", [s for s in sources() if "## Checkpoint" in s.read_text(encoding="utf-8")],
                         ids=lambda p: p.stem)
def test_every_checkpoint_is_answered(src):
    text = solution_file(src).read_text(encoding="utf-8")
    assert f"## Checkpoint · {src.name[:2]}" in text


def test_solution_files_use_confidence_markers():
    text = "".join(p.read_text(encoding="utf-8") for p in SOLUTIONS.glob("[0-9]*.md"))
    for marker in ("✅", "🔬", "✍️"):
        assert marker in text


@pytest.mark.parametrize("module,name", grader_pairs())
def test_grader_accepts_reference_solution(module, name, isolated_artifacts):
    grader = getattr(importlib.import_module(f"churnlab.grading.{module}"), name)
    solution_module = importlib.import_module(f"solutions.code.{module}")
    assert hasattr(solution_module, name), f"solutions/code/{module}.py has no {name}"
    result = grader(getattr(solution_module, name))
    assert result.passed, f"{module}.{name} rejected its own solution: {result.message}"


@pytest.mark.parametrize("module,name", grader_pairs())
def test_grader_reports_todo_for_a_stub(module, name, isolated_artifacts):
    grader = getattr(importlib.import_module(f"churnlab.grading.{module}"), name)

    def stub(*args, **kwargs):
        raise NotImplementedError

    result = grader(stub)
    assert result.status == "todo"
