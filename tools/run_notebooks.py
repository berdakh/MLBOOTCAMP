#!/usr/bin/env python3
"""Execute notebooks top to bottom and record what happened.

This is the course's evidence that the notebooks run. By default every
notebook gets its OWN empty artifact directory (``--fresh``), which simulates
opening it in a brand-new Colab runtime where no earlier notebook was run: the
notebook must rebuild whatever it needs via ``churnlab.bootstrap``.

It is not Colab. It runs in whatever Python environment you launch it from.
The report says which Python and library versions that was, and nothing more.

Usage::

    python tools/run_notebooks.py                 # all notebooks, fresh dirs
    python tools/run_notebooks.py 09 10           # only matching names
    python tools/run_notebooks.py --part part1-foundations --report verification-log.md
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from book import PARTS, all_notebooks  # noqa: E402


def versions() -> str:
    mods = ["numpy", "pandas", "sklearn", "duckdb", "fastapi", "pydantic", "httpx", "uvicorn",
            "torch", "transformers"]
    out = [f"python {platform.python_version()}"]
    for name in mods:
        try:
            mod = __import__(name)
            out.append(f"{name} {mod.__version__}")
        except Exception:
            out.append(f"{name} (missing)")
    return ", ".join(out)


def run_one(path: Path, timeout: int, fresh: bool, save_dir: Path | None = None) -> tuple[str, float, str]:
    nb = nbformat.read(path, as_version=4)
    env_backup = os.environ.get("BOOTCAMP_ARTIFACTS")
    tmp = tempfile.mkdtemp(prefix=f"nb-{path.stem[:2]}-") if fresh else None
    if tmp:
        os.environ["BOOTCAMP_ARTIFACTS"] = tmp
    start = time.time()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        status, detail = "PASS", ""
    except CellExecutionError as exc:
        status, detail = "FAIL", str(exc).strip().splitlines()[-1][:300]
    except Exception as exc:  # timeouts, kernel death
        status, detail = "FAIL", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, save_dir / path.relative_to(REPO_ROOT).as_posix().replace("/", "__"))
        if env_backup is None:
            os.environ.pop("BOOTCAMP_ARTIFACTS", None)
        else:
            os.environ["BOOTCAMP_ARTIFACTS"] = env_backup
    return status, time.time() - start, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("only", nargs="*")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout, seconds")
    parser.add_argument("--shared", action="store_true", help="share one artifact dir (sequential run)")
    parser.add_argument("--report", type=Path, help="write a markdown report here")
    parser.add_argument("--part", choices=[p for p, _, _ in PARTS] + ["part4-capstones"],
                        help="run only this part's notebooks")
    parser.add_argument("--save-executed", type=Path, help="write executed copies (with outputs) here")
    args = parser.parse_args()

    notebooks = all_notebooks()
    if args.part:
        notebooks = [n for n in notebooks if n.relative_to(REPO_ROOT).parts[0] == args.part]
    if args.only:
        notebooks = [n for n in notebooks
                     if any(t in n.relative_to(REPO_ROOT).as_posix() for t in args.only)]
    if args.shared and "BOOTCAMP_ARTIFACTS" not in os.environ:
        os.environ["BOOTCAMP_ARTIFACTS"] = tempfile.mkdtemp(prefix="nb-shared-")

    # One OpenMP thread per kernel: on a shared or busy machine, gradient-boosted
    # trees otherwise oversubscribe the CPU and a 1-second fit can take minutes.
    # (Learners' runtimes are unaffected; this only applies to this tool.)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    env = versions()
    print(env)
    rows, failures = [], 0
    for path in notebooks:
        status, secs, detail = run_one(path, args.timeout, fresh=not args.shared, save_dir=args.save_executed)
        failures += status != "PASS"
        name = path.relative_to(REPO_ROOT).as_posix()
        print(f"{status}  {name:<62} {secs:6.1f}s  {detail}")
        rows.append((name, status, secs, detail))

    if args.report:
        lines = [
            "| Notebook | Result | Seconds | Detail |",
            "|---|---|---:|---|",
            *[f"| `{n}` | {s} | {t:.0f} | {d} |" for n, s, t, d in rows],
        ]
        header = (f"Executed {time.strftime('%Y-%m-%d')} with `tools/run_notebooks.py` "
                  f"({'shared' if args.shared else 'fresh'} artifact directory per notebook) on "
                  f"{platform.system()} - {env}.\n\n")
        args.report.write_text(header + "\n".join(lines) + "\n")
    print(f"\n{len(rows) - failures}/{len(rows)} notebooks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
