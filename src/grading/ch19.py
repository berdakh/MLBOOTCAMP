"""Self-checks for Level 2 notebook 01 · Event logs & columnar data."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from churnlab.checks import run_check


def _reference_keep(user_ids, pct, salt=""):
    return np.array([int(hashlib.sha256(f"{salt}:{u}".encode()).hexdigest()[:16], 16) % 100 < pct
                     for u in user_ids])


def ex1(user_hash_sample):
    def test():
        ids = np.arange(20_000)
        keep = np.asarray(user_hash_sample(ids, 10, salt="exp42"))
        assert keep.shape == ids.shape, f"expected {ids.shape} booleans, got shape {keep.shape}"
        assert keep.dtype == bool, f"expected a boolean array, got dtype {keep.dtype}"
        assert 0.08 < keep.mean() < 0.12, f"a 10% sample kept {keep.mean():.1%} of users"
        ref = _reference_keep(ids, 10, salt="exp42")
        wrong = int((keep != ref).sum())
        assert wrong == 0, (f"{wrong} decisions differ from sha256(f'{{salt}}:{{user_id}}') -> first 16 hex "
                            "digits -> int % 100 < pct")
        again = np.asarray(user_hash_sample(ids, 10, salt="exp42"))
        assert (again == keep).all(), "two calls gave different samples"
        dup = np.array([7, 7, 123, 7, 123])
        d = np.asarray(user_hash_sample(dup, 50, salt="exp42"))
        assert d[0] == d[1] == d[3] and d[2] == d[4], "the same user got different decisions"
        five = np.asarray(user_hash_sample(ids, 5, salt="exp42"))
        assert not (five & ~keep).any(), "the 5% sample must be a subset of the 10% sample"
        other = np.asarray(user_hash_sample(ids, 10, salt="exp43"))
        overlap = (other & keep).sum() / keep.sum()
        assert overlap < 0.2, f"a different salt should give a ~independent sample (overlap {overlap:.0%})"
        assert not np.asarray(user_hash_sample(ids[:100], 0)).any(), "pct=0 must keep nobody"
        assert np.asarray(user_hash_sample(ids[:100], 100)).all(), "pct=100 must keep everybody"
        strs = np.asarray(user_hash_sample(["u1", "u2", "u3"], 50, salt="s"))
        assert (strs == _reference_keep(["u1", "u2", "u3"], 50, "s")).all(), "string ids give wrong decisions"

    return run_check("Exercise 1 · user_hash_sample", test,
                     hint="np.array([int(hashlib.sha256(f'{salt}:{u}'.encode()).hexdigest()[:16], 16) % 100 < pct "
                          "for u in user_ids])")


def _make_partitioned(directory: Path) -> pd.DataFrame:
    """Days 0-19; day 0 and days 7-13 valid, every other file corrupt."""
    rng = np.random.default_rng(11)
    frames = []
    for day in range(20):
        pdir = directory / f"day={day}"
        pdir.mkdir(parents=True)
        if day == 0 or 7 <= day <= 13:
            n = int(rng.integers(40, 80))
            df = pd.DataFrame({"request_id": np.arange(n) // 4 + day * 1000,
                               "position": rng.integers(0, 4, n).astype("int8"),
                               "clicked": rng.random(n) < 0.3})
            df.to_parquet(pdir / "part-0000.parquet", index=False)
            frames.append(df.assign(day=day))
        else:
            (pdir / "part-0000.parquet").write_bytes(b"not a parquet file " * 64)
    return pd.concat(frames)


def ex2(daily_ctr_by_position_sql):
    def test():
        import duckdb

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "impressions"
            truth = _make_partitioned(root)
            sql = daily_ctr_by_position_sql(str(root))
            assert isinstance(sql, str), "return the SQL query as a string"
            con = duckdb.connect()
            try:
                got = con.execute(sql).df()
            except duckdb.Error as exc:
                msg = str(exc).splitlines()[0][:160]
                raise AssertionError("the query failed on the test data. If it complains about a corrupt "
                                     f"(non-Parquet) file, it read partitions outside days 7-13: {msg}") from None
            finally:
                con.close()
        want = (truth[truth.day.between(7, 13)]
                .groupby(["day", "position"]).clicked.agg(impressions="size", clicks="sum").reset_index())
        want["ctr"] = want.clicks / want.impressions
        for col in ("day", "position", "impressions", "clicks", "ctr"):
            assert col in got.columns, f"missing column {col!r}; got {list(got.columns)}"
        assert len(got) == len(want), f"expected {len(want)} rows (days 7-13 x positions), got {len(got)}"
        assert set(got.day.astype(int)) == set(range(7, 14)), f"days in result: {sorted(set(got.day))}"
        g = got[["day", "position", "impressions", "clicks", "ctr"]].astype(float).reset_index(drop=True)
        assert (g[["day", "position"]].values == want[["day", "position"]].astype(float).values).all(), \
            "rows must be ordered by day, position"
        assert np.allclose(g.impressions, want.impressions) and np.allclose(g.clicks, want.clicks), \
            "impression or click counts are wrong"
        assert np.allclose(g.ctr, want.ctr), "ctr values are wrong (integer division? cast clicked to INT/DOUBLE)"

    return run_check("Exercise 2 · daily CTR by position, days 7-13", test,
                     hint="FROM read_parquet(f'{impressions_dir}/*/*.parquet', hive_partitioning = true) "
                          "WHERE day BETWEEN 7 AND 13 GROUP BY day, position ORDER BY day, position")
