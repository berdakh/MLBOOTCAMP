"""Where Level 2 keeps its data and artifacts, and cached access to ShopStream.

Everything is regenerated deterministically from a seed, so a fresh Colab
runtime rebuilds the data in well under a minute (``small`` scale) instead of
downloading it. Generation with the same library versions gives identical
data; across NumPy versions the random streams may differ slightly, so
notebooks quote ranges, not exact values.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churnlab.colab import artifact_dir

from .data import World, WorldConfig, load_impressions, make_world, write_dataset


def level2_dir() -> Path:
    path = artifact_dir() / "level2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(scale: str = "small") -> Path:
    return level2_dir() / f"shopstream_{scale}"


def ensure_dataset(scale: str = "small", partition_by_day: bool = False, verbose: bool = True) -> tuple[World, Path]:
    """Build (once) and return the world and the dataset directory."""
    world = make_world(WorldConfig.scale(scale))
    directory = dataset_dir(scale)
    marker = directory / "_SUCCESS"
    if not marker.exists():
        if verbose:
            print(f"generating ShopStream ({scale}) -> {directory} ...")
        write_dataset(world, directory, partition_by_day=partition_by_day)
        marker.write_text("ok\n")  # written last: a half-written dataset never looks complete
    elif verbose:
        print(f"using cached ShopStream ({scale}) at {directory}")
    return world, directory


def load(scale: str = "small", columns=None, include_hidden: bool = False):
    """Return (world, users, items, impressions). Hidden truth columns dropped by default."""
    world, directory = ensure_dataset(scale, verbose=False)
    imps = load_impressions(directory, columns)
    if not include_hidden:
        imps = imps.drop(columns=[c for c in ("examined",) if c in imps.columns])
    users = pd.read_parquet(directory / "users.parquet")
    items = pd.read_parquet(directory / "items.parquet")
    return world, users, items, imps


#: Fixed time split for Level 2 (days are 0..55): train on weeks 1-6,
#: validate on week 7, test on week 8 (test touched only in notebook 20).
TRAIN_DAYS = range(0, 42)
VALID_DAYS = range(42, 49)
TEST_DAYS = range(49, 56)
