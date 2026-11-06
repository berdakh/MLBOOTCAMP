#!/usr/bin/env python3
"""Regenerate the committed raw extract in data/raw/ and print its checksums.

Run this only when you intend to change the dataset. Afterwards, paste the
printed digests into ``RAW_SHA256`` in ``src/churnlab/data.py`` and commit the
files together - the tests fail if they disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from churnlab.data import generate, write_raw  # noqa: E402


def main() -> int:
    digests = write_raw(generate(), ROOT / "data" / "raw")
    print("RAW_SHA256 = {")
    for name, digest in digests.items():
        print(f'    "{name}": "{digest}",')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
