#!/usr/bin/env python3
"""Print the text outputs of an executed notebook, cell by cell (for authors).

    python tools/run_notebooks.py 03 --save-executed /tmp/executed
    python tools/show_outputs.py /tmp/executed/03_sql_for_ml.ipynb
"""

from __future__ import annotations

import sys

import nbformat


def main(path: str) -> None:
    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells, 1):
        if cell.cell_type != "code":
            continue
        print(f"--- cell {i}: {cell.source.splitlines()[0][:80] if cell.source else ''}")
        for out in cell.get("outputs", []):
            if out.output_type == "stream":
                print(out.text.rstrip()[:3000])
            elif out.output_type in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
                print(text.rstrip()[:3000])
            elif out.output_type == "error":
                print("ERROR:", out.ename, out.evalue)


if __name__ == "__main__":
    main(sys.argv[1])
