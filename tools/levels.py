"""The course's levels and where each keeps its files. Shared by every tool.

Level 1 lives at the repository root (it came first); Levels 2 and 3 live in
their own folders with the same internal layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Level:
    name: str           # "level1", "level2", "level3"
    root: Path          # folder holding notebook_src/, notebooks/, solutions/
    package: str        # importable package with grading/ submodule
    solutions_pkg: str  # importable package holding code/nbNN.py

    @property
    def src_dir(self) -> Path:
        return self.root / "notebook_src"

    @property
    def nb_dir(self) -> Path:
        return self.root / "notebooks"

    @property
    def solutions_dir(self) -> Path:
        return self.root / "solutions"

    @property
    def rel(self) -> str:
        """Path prefix of this level relative to the repo root ('' for level 1)."""
        r = self.root.relative_to(REPO_ROOT).as_posix()
        return "" if r == "." else r + "/"


LEVELS = [
    Level("level1", REPO_ROOT, "churnlab", "solutions"),
    Level("level2", REPO_ROOT / "level2", "shoprank", "level2.solutions"),
    Level("level3", REPO_ROOT / "level3", "scaleml", "level3.solutions"),
]


def sources(level: Level) -> list[Path]:
    if not level.src_dir.exists():
        return []
    return sorted(p for p in level.src_dir.glob("*.py") if not p.name.startswith("_"))


def all_notebooks() -> list[Path]:
    out: list[Path] = []
    for level in LEVELS:
        if level.nb_dir.exists():
            out += sorted(level.nb_dir.glob("*.ipynb"))
    return out
