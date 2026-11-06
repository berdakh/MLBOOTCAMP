"""Colab helpers: where artifacts live, and how to keep them across sessions.

A Colab runtime is temporary. When it disconnects (idle ~90 minutes, or 12
hours at most on the free tier), everything under ``/content`` is deleted -
including models you trained. Three ways to keep work, best first:

1. **Google Drive** (``artifact_dir(use_drive=True)``): artifacts are written
   straight to your Drive and survive any disconnect. Requires one click to
   authorise.
2. **Zip and download** (``export_artifacts``): one file you keep locally and
   upload next session (``restore_artifacts``).
3. **Rebuild** (``churnlab.bootstrap``): every notebook can regenerate what
   earlier notebooks produced, deterministically, in under a minute on CPU.
   This is the safety net: no notebook depends on a previous session.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/berdakh/MLBOOTCAMP.git"
REPO_DIR = "MLBOOTCAMP"
DEFAULT_BRANCH = "main"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_SUBDIR = "MyDrive/mlbootcamp"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def find_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "churnlab").is_dir():
            return candidate
    raise FileNotFoundError("could not find the MLBOOTCAMP repository root (looked for src/churnlab)")


def artifact_dir(use_drive: bool = False, root: Path | None = None) -> Path:
    """Return (and create) the directory where this course writes artifacts.

    Precedence: ``BOOTCAMP_ARTIFACTS`` env var > Google Drive (if requested and
    in Colab) > ``<repo>/artifacts``.
    """
    env = os.environ.get("BOOTCAMP_ARTIFACTS")
    if env:
        path = Path(env)
    elif use_drive and in_colab():
        if not (DRIVE_MOUNT / "MyDrive").exists():
            from google.colab import drive  # type: ignore[import-not-found]

            drive.mount(str(DRIVE_MOUNT))
        path = DRIVE_MOUNT / DRIVE_SUBDIR / "artifacts"
    else:
        path = (root or find_root()) / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_artifacts(directory: Path, zip_path: Path | None = None, download: bool = True) -> Path:
    """Zip ``directory``; in Colab, also trigger a browser download."""
    directory = Path(directory)
    zip_path = Path(zip_path or directory.with_suffix(".zip"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(directory))
    if download and in_colab():
        from google.colab import files  # type: ignore[import-not-found]

        files.download(str(zip_path))
    return zip_path


def restore_artifacts(zip_path: Path, directory: Path, overwrite: bool = False) -> Path:
    """Unzip an exported archive into ``directory`` (refusing unsafe paths)."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"{directory} is not empty; pass overwrite=True to replace it")
    if overwrite and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (directory / member).resolve()
            if not str(target).startswith(str(directory.resolve())):
                raise ValueError(f"refusing to extract {member!r} outside {directory}")
        zf.extractall(directory)
    return directory


def notebook_url(notebook: str, branch: str = DEFAULT_BRANCH, owner_repo: str = "berdakh/MLBOOTCAMP") -> str:
    """Colab URL for a notebook path in this repo."""
    return f"https://colab.research.google.com/github/{owner_repo}/blob/{branch}/{notebook.lstrip('/')}"
