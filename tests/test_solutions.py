"""Keep exercises, graders and solutions in sync - for every level.

For each level (see tools/levels.py):

* every ``> **Exercise K.**`` in notebook NN has a ``## NN-K`` solution heading
  in ``<level>/solutions/NN_slug.md``;
* every notebook with a Checkpoint has a ``## Checkpoint · NN`` section;
* every grader ``exK`` in ``<package>.grading.nbNN`` ACCEPTS the reference
  solution ``<solutions_pkg>.code.nbNN.exK`` - a grader that fails its own
  solution would teach learners the wrong thing;
* every grader reports TODO (not a crash) for an untouched stub.

Graders marked ``REQUIRES = {"torch"}`` (module attribute) are skipped when
that package is not installed, so the core suite still runs without PyTorch.
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
from levels import LEVELS, sources  # noqa: E402


def exercises():
    out = []
    for level in LEVELS:
        for src in sources(level):
            for m in re.finditer(r"\*\*Exercise (\d+)\.\*\*", src.read_text(encoding="utf-8")):
                out.append(pytest.param(level, src, m.group(1), id=f"{level.name}-{src.stem[:2]}-{m.group(1)}"))
    return out


def checkpoints():
    return [pytest.param(level, src, id=f"{level.name}-{src.stem[:2]}")
            for level in LEVELS for src in sources(level)
            if "## Checkpoint" in src.read_text(encoding="utf-8")]


def grader_pairs():
    pairs = []
    for level in LEVELS:
        grading_dir = ROOT / "src" / level.package / "grading"
        for path in sorted(grading_dir.glob("nb*.py")):
            module = f"{level.package}.grading.{path.stem}"
            names = sorted(set(re.findall(r"^def (ex\d+)\(", path.read_text(encoding="utf-8"), re.M)))
            for name in names:
                pairs.append(pytest.param(level, module, path.stem, name, id=f"{level.name}-{path.stem}-{name}"))
    return pairs


def _import_grader(module: str):
    mod = importlib.import_module(module)
    missing = [r for r in getattr(mod, "REQUIRES", ()) if importlib.util.find_spec(r) is None]
    if missing:
        pytest.skip(f"{module} needs {missing}")
    return mod


def test_level1_has_exercises():
    assert sum(1 for p in exercises() if p.values[0].name == "level1") >= 20


@pytest.mark.parametrize("level,src,number", exercises())
def test_every_exercise_has_a_written_solution(level, src, number):
    path = level.solutions_dir / f"{src.stem}.md"
    assert path.exists(), f"missing {path.relative_to(ROOT)}"
    nn = src.name[:2]
    assert re.search(rf"^## {nn}-{number} ", path.read_text(encoding="utf-8"), re.M), (
        f"{src.name} exercise {number} has no '## {nn}-{number} · ...' heading in {path.name}")


@pytest.mark.parametrize("level,src", checkpoints())
def test_every_checkpoint_is_answered(level, src):
    text = (level.solutions_dir / f"{src.stem}.md").read_text(encoding="utf-8")
    assert f"## Checkpoint · {src.name[:2]}" in text


@pytest.mark.parametrize("level", [lv for lv in LEVELS if lv.solutions_dir.exists()], ids=lambda lv: lv.name)
def test_solution_files_use_confidence_markers(level):
    text = "".join(p.read_text(encoding="utf-8") for p in level.solutions_dir.glob("[0-9]*.md"))
    for marker in ("✅", "🔬", "✍️"):
        assert marker in text


@pytest.mark.parametrize("level,module,stem,name", grader_pairs())
def test_grader_accepts_reference_solution(level, module, stem, name, isolated_artifacts):
    grader = getattr(_import_grader(module), name)
    solution_module = importlib.import_module(f"{level.solutions_pkg}.code.{stem}")
    assert hasattr(solution_module, name), f"{level.solutions_pkg}/code/{stem}.py has no {name}"
    result = grader(getattr(solution_module, name))
    assert result.passed, f"{module}.{name} rejected its own solution: {result.message}"


@pytest.mark.parametrize("level,module,stem,name", grader_pairs())
def test_grader_reports_todo_for_a_stub(level, module, stem, name, isolated_artifacts):
    grader = getattr(_import_grader(module), name)

    def stub(*args, **kwargs):
        raise NotImplementedError

    assert grader(stub).status == "todo"
