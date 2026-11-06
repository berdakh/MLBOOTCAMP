"""Day 1 helpers: writing, measuring, hashing and partitioning event logs.

Notebooks 01 and 02 use these. Each function is small on purpose: the
notebooks build the idea by hand first, then point here.

Three ideas run through the module:

* **Write logs the way a warehouse stores them**: Parquet files, partitioned
  by day (``impressions/day=7/part-0003.parquet``), so a query for one week
  opens only that week's files.
* **Hash, don't draw**: a decision that must be the same tomorrow, in another
  process or on another machine (is this user in the sample? which reducer gets
  this key?) comes from a *stable* hash of the key, never from ``random`` and
  never from Python's built-in ``hash()``, which is salted per process for
  strings.
* **Measure**: sizes on disk and wall-clock times, so claims about formats and
  layouts are numbers from your own run.
"""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .data import HIDDEN_COLUMNS, World, simulate_logs

# --- measuring ----------------------------------------------------------------


@contextmanager
def timed(label: str | None = None, results: dict | None = None):
    """``with timed("parquet read", results=t): ...`` - store (and print) seconds."""
    start = time.perf_counter()
    yield
    seconds = time.perf_counter() - start
    if results is not None and label is not None:
        results[label] = seconds
    if label is not None:
        print(f"{label}: {seconds:.2f} s")


def dir_size_mb(path: Path) -> float:
    """Total size of a file, or of every file under a directory, in MB."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


# --- writing a partitioned log --------------------------------------------------


def write_days_partitioned(world: World, directory: Path, n_days: int | None = None) -> dict:
    """Simulate the world's logs and write ``directory/impressions/day=D/part-I.parquet``.

    Only the first ``n_days`` days are written (all days if None). Requests
    come out of :func:`shoprank.data.simulate_logs` in day order, so we can
    stop the generator as soon as it passes the last wanted day - the rows we
    keep are exactly the first days of the full world, not a different world.

    The hidden ``examined`` column is dropped: it is simulation truth, and a
    real log could never contain it. Partition folders are NOT zero-padded
    (``day=7``), so DuckDB infers ``day`` as an integer column.
    """
    directory = Path(directory)
    imp_dir = directory / "impressions"
    imp_dir.mkdir(parents=True, exist_ok=True)
    world.users.to_parquet(directory / "users.parquet", index=False)
    world.items.to_parquet(directory / "items.parquet", index=False)
    last = world.config.n_days if n_days is None else n_days
    rows = files = 0
    start = time.perf_counter()
    for i, chunk in enumerate(simulate_logs(world)):
        if chunk["day"].min() >= last:
            break  # the generator is in day order: nothing wanted remains
        chunk = chunk[chunk["day"] < last].drop(columns=HIDDEN_COLUMNS)
        for day, part in chunk.groupby("day"):
            pdir = imp_dir / f"day={int(day)}"
            pdir.mkdir(exist_ok=True)
            part.drop(columns="day").to_parquet(pdir / f"part-{i:04d}.parquet", index=False)
            files += 1
        rows += len(chunk)
    (directory / "_SUCCESS").write_text("ok\n")
    return {"rows": rows, "files": files, "days": last, "seconds": time.perf_counter() - start}


# --- stable hashing -------------------------------------------------------------


def stable_hash64(value, salt: str = "") -> int:
    """A 64-bit integer from SHA-256 of ``"<salt>:<value>"``: identical in every process."""
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return int(digest[:16], 16)


def stable_bucket(value, n_buckets: int = 100, salt: str = "") -> int:
    """Which of ``n_buckets`` buckets a key falls in (0..n_buckets-1), forever."""
    return stable_hash64(value, salt) % n_buckets


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised, stable 64-bit mixing of integer keys (the SplitMix64 finaliser).

    Hashing millions of integers through hashlib one by one is slow; this does
    the same job for integer keys in NumPy. Same input -> same output, in any
    process, on any machine.
    """
    z = np.asarray(x).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def partition_of(keys: np.ndarray, n_partitions: int) -> np.ndarray:
    """Hash-partition integer keys: the 'shuffle' step's routing decision."""
    return (splitmix64(keys) % np.uint64(n_partitions)).astype(np.int64)


# --- realistic dirt -------------------------------------------------------------


def inject_duplicates(df: pd.DataFrame, frac: float, seed: int = 0, ts_col: str = "ts") -> pd.DataFrame:
    """Re-deliver a fraction of rows, as an at-least-once logging pipeline does.

    Adds ``ingest_ts`` (when the row reached the warehouse): originals arrive
    1-60 s after the event, duplicates up to 10 minutes after. Returned in
    ingest order, like a raw table.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["ingest_ts"] = out[ts_col] + pd.to_timedelta(rng.integers(1, 61, len(out)), unit="s")
    dupes = out.sample(frac=frac, random_state=seed)
    dupes["ingest_ts"] = dupes["ingest_ts"] + pd.to_timedelta(rng.integers(1, 600, len(dupes)), unit="s")
    return pd.concat([out, dupes]).sort_values("ingest_ts", kind="stable").reset_index(drop=True)
