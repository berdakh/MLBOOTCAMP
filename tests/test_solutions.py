"""Keep each chapter's exercises, graders and solutions in sync.

For every chapter (see tools/book.py):

* every ``> **Exercise K.**`` in notebook.py has a ``## Exercise K ·`` heading
  in the chapter's solutions.md;
* a chapter with a Checkpoint has a ``## Checkpoint answers`` section;
* every grader ``exK`` in ``grading.chNN`` ACCEPTS the reference solution
  ``exK`` in the chapter's solutions.py - a grader that fails its own solution
  would teach learners the wrong thing;
* every grader reports TODO (not a crash) for an untouched stub.

Graders that set ``REQUIRES = {"torch"}`` are skipped when that package is not
installed, so the core suite still runs without PyTorch.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from book import chapters  # noqa: E402

CHAPTERS = chapters()


def _id(ch):
    return f"ch{ch.number:02d}"


def exercises():
    out = []
    for ch in CHAPTERS:
        for m in re.finditer(r"\*\*Exercise (\d+)\.\*\*", ch.source.read_text(encoding="utf-8")):
            out.append(pytest.param(ch, m.group(1), id=f"{_id(ch)}-ex{m.group(1)}"))
    return out


def grader_pairs():
    pairs = []
    for ch in CHAPTERS:
        path = ROOT / "src" / "grading" / f"ch{ch.number:02d}.py"
        if not path.exists():
            continue
        for name in sorted(set(re.findall(r"^def (ex\d+)\(", path.read_text(encoding="utf-8"), re.M))):
            pairs.append(pytest.param(ch, name, id=f"{_id(ch)}-{name}"))
    return pairs


def _load_solutions(ch):
    spec = importlib.util.spec_from_file_location(f"solutions_ch{ch.number:02d}", ch.solutions_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_grader(ch):
    mod = importlib.import_module(ch.grader_module)
    missing = [r for r in getattr(mod, "REQUIRES", ()) if importlib.util.find_spec(r) is None]
    if missing:
        pytest.skip(f"{ch.grader_module} needs {missing}")
    return mod


def test_book_has_58_chapters():
    assert [c.number for c in CHAPTERS] == list(range(1, 59))


@pytest.mark.parametrize("ch,number", exercises())
def test_every_exercise_has_a_written_solution(ch, number):
    text = ch.solutions_md.read_text(encoding="utf-8")
    assert re.search(rf"^## Exercise {number} ·", text, re.M), (
        f"{ch.rel}: exercise {number} has no '## Exercise {number} · ...' heading in solutions.md")


@pytest.mark.parametrize("ch", [c for c in CHAPTERS if "## Checkpoint" in c.source.read_text(encoding="utf-8")],
                         ids=_id)
def test_every_checkpoint_is_answered(ch):
    assert "## Checkpoint answers" in ch.solutions_md.read_text(encoding="utf-8")


def test_solution_files_use_confidence_markers():
    text = "".join(c.solutions_md.read_text(encoding="utf-8") for c in CHAPTERS)
    for marker in ("[verified]", "[expected]", "[open-ended]"):
        assert marker in text


@pytest.mark.parametrize("ch,name", grader_pairs())
def test_grader_accepts_reference_solution(ch, name, isolated_artifacts):
    grader = getattr(_import_grader(ch), name)
    solutions = _load_solutions(ch)
    assert hasattr(solutions, name), f"{ch.rel}/solutions.py has no {name}"
    result = grader(getattr(solutions, name))
    assert result.passed, f"{ch.grader_module}.{name} rejected its own solution: {result.message}"


@pytest.mark.parametrize("ch,name", grader_pairs())
def test_grader_reports_todo_for_a_stub(ch, name, isolated_artifacts):
    grader = getattr(_import_grader(ch), name)

    def stub(*args, **kwargs):
        raise NotImplementedError

    assert grader(stub).status == "todo"
