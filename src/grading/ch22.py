"""Self-checks for Level 2 notebook 04 · Streaming features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from churnlab.checks import run_check


class _RefCounter:
    def __init__(self, window, lateness):
        self.w, self.lateness = float(window), float(lateness)
        self.wm = -math.inf
        self.state: dict = {}
        self.late_events: list = []

    def _close(self, upto):
        done = sorted((ws, str(k), k) for (k, ws) in self.state if ws + self.w <= upto)
        return [(k, ws, self.state.pop((k, ws))) for ws, _, k in done]

    def add(self, key, t):
        ws = math.floor(t / self.w) * self.w
        if ws + self.w <= self.wm:
            self.late_events.append((key, t))
            return []
        self.state[(key, ws)] = self.state.get((key, ws), 0) + 1
        self.wm = max(self.wm, t - self.lateness)
        return self._close(self.wm)

    def flush(self):
        return self._close(math.inf)


def _norm(rows):
    return [(k, float(ws), int(c)) for k, ws, c in rows]


def _replay(cls, window, lateness, events):
    counter = cls(window, lateness)
    calls = [_norm(counter.add(k, t)) for k, t in events]
    return counter, calls, _norm(counter.flush())


def ex1(TumblingWindowCounter):
    def test():
        crafted = [("a", 1), ("b", 3), ("a", 12), ("a", 16), ("b", 8), ("a", 11), ("b", 9.99),
                   ("a", 30), ("a", 20), ("a", 19.9), ("c", 40), ("c", 45), ("b", 41)]
        ref, ref_calls, ref_flush = _replay(_RefCounter, 10, 5, crafted)
        got, calls, flushed = _replay(TumblingWindowCounter, 10, 5, crafted)
        for i, ((k, t), want, have) in enumerate(zip(crafted, ref_calls, calls)):
            assert have == want, (f"after add({k!r}, {t}) (event #{i + 1}) expected closed windows {want}, "
                                  f"got {have}")
        assert flushed == ref_flush, f"flush() expected {ref_flush}, got {flushed}"
        late = [(k, float(t)) for k, t in got.late_events]
        assert late == [(k, float(t)) for k, t in ref.late_events], \
            f"late_events expected {ref.late_events}, got {got.late_events}"
        # boundary: watermark exactly at the window end closes it; an event at the end belongs to the next window
        _, calls_b, flush_b = _replay(TumblingWindowCounter, 10, 5, [("x", 0), ("x", 10), ("x", 15), ("x", 9)])
        assert calls_b[2] == [("x", 0.0, 1)], "a window must close when watermark == window_start + window"
        assert flush_b == [("x", 10.0, 2)], "t = 10 belongs to window [10, 20), not [0, 10)"
        # zero lateness, random out-of-order stream vs reference
        rng = np.random.default_rng(4)
        t = np.sort(rng.uniform(0, 600, 300)) + rng.exponential(5, 300) * (rng.random(300) < 0.3)
        keys = rng.choice(["k1", "k2", "k3"], 300)
        events = list(zip(keys.tolist(), t.tolist()))
        for lateness in (0, 7):
            r, rc, rf = _replay(_RefCounter, 30, lateness, events)
            g, gc, gf = _replay(TumblingWindowCounter, 30, lateness, events)
            assert gc == rc and gf == rf, f"random stream (lateness={lateness}): emitted windows differ from reference"
            assert len(g.late_events) == len(r.late_events), (
                f"random stream (lateness={lateness}): {len(g.late_events)} late events, expected {len(r.late_events)}")
        total = sum(c for call in rc for _, _, c in call) + sum(c for _, _, c in rf)
        assert total + len(r.late_events) == len(events), "every event must be counted once or reported late"

    return run_check("Exercise 1 · TumblingWindowCounter", test,
                     hint="ws = math.floor(t / w) * w; late if ws + w <= watermark (before updating it); else "
                          "count, update watermark = max(wm, t - lateness), pop and return windows with "
                          "ws + w <= watermark sorted by (ws, key)")


def ex2(parity_report):
    def test():
        b = pd.DataFrame({"k": ["a", "b", "c", "d"], "w": [0, 0, 0, 900], "value": [1.0, 2.0, 3.0, np.nan]})
        s = pd.DataFrame({"k": ["a", "b", "d", "e"], "w": [0, 0, 900, 0], "value": [1.0, 2.5, np.nan, 7.0]})
        r = parity_report(b, s, ["k", "w"], 0.1)
        want = {"n_batch": 4, "n_stream": 4, "missing_in_stream": 1, "missing_in_batch": 1, "mismatched": 1,
                "max_abs_diff": 0.5, "passed": False}
        for k, v in want.items():
            assert k in r, f"missing key {k!r} in the report"
            assert (abs(r[k] - v) < 1e-9 if isinstance(v, float) else r[k] == v), f"{k}: expected {v}, got {r[k]}"
        assert abs(r["match_rate"] - 2 / 5) < 1e-9, f"match_rate: expected 0.4 (2 matches / 5 keys), got {r['match_rate']}"
        r2 = parity_report(b, s, ["k", "w"], 0.5)
        assert r2["mismatched"] == 0, "a difference equal to tol is a match"
        same = pd.DataFrame({"key": [1, 2, 3], "value": [0.1, 0.2, 0.3]})
        ok = parity_report(same, same.sample(frac=1, random_state=0), "key", 0.0)
        assert ok["passed"] is True or ok["passed"] == True, "identical frames (any row order) must pass"  # noqa: E712
        assert ok["match_rate"] == 1.0 and ok["max_abs_diff"] == 0.0
        one_nan = parity_report(pd.DataFrame({"key": [1], "value": [np.nan]}),
                                pd.DataFrame({"key": [1], "value": [0.0]}), "key", 1.0)
        assert one_nan["mismatched"] == 1, "NaN on one side only is a mismatch"
        empty = pd.DataFrame({"key": pd.Series([], dtype=int), "value": pd.Series([], dtype=float)})
        e = parity_report(empty, empty, "key", 0.0)
        assert e["passed"] and e["match_rate"] == 1.0 and e["max_abs_diff"] == 0.0, "two empty frames pass"
        dup = pd.DataFrame({"key": [1, 1], "value": [1.0, 1.0]})
        try:
            parity_report(dup, same, "key", 0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicated keys must raise ValueError")

    return run_check("Exercise 2 · parity_report", test,
                     hint="check df.duplicated(key).any(); merge(how='outer', on=key, indicator=True); "
                          "count left_only/right_only; on 'both' rows compare |a-b| <= tol with NaN handling")
