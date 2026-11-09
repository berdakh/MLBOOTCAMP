"""Self-checks for Level 3 notebook 19 · ML depth & research discussion."""

from __future__ import annotations

import hashlib
import math

from churnlab.checks import close, run_check

_SALT = "nb19-critique:"
# sha256(_SALT + letter)[:16] of the planted flaws. Stored hashed so the answer
# is not visible next to the exercise; the written solution explains each one.
_FLAW_HASHES = {"227fa91fee5ca1af", "0de7f07494946e2e", "3c071c50a11120c9", "8f3f82a5528c5557", "f4488c39f2b18c23"}
_OPTIONS = "ABCDEFGHI"


def _h(letter: str) -> str:
    return hashlib.sha256((_SALT + letter).encode()).hexdigest()[:16]


def _reference_lr(step, base_lr, warmup, total, min_lr=0.0):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def ex1(warmup_cosine_lr):
    def test():
        base, warm, total = 1e-3, 10, 110
        got = [warmup_cosine_lr(s, base, warm, total) for s in range(total + 5)]
        assert all(isinstance(v, float) for v in got), "return a float learning rate"
        assert close(got[0], base / warm), f"step 0 of a 10-step warmup: expected {base / warm:g}, got {got[0]:g}"
        if close(got[1], base / warm):
            raise AssertionError("step 1 equals step 0: use (step + 1) / warmup_steps")
        assert close(got[warm - 1], base), (
            f"the last warmup step ({warm - 1}) must reach base_lr={base:g}, got {got[warm - 1]:g}; use (step + 1)")
        assert close(got[warm], base), f"first decay step (progress 0) is base_lr, got {got[warm]:g}"
        mid = warm + (total - warm) // 2
        assert close(got[mid], base / 2, 1e-6), f"half-way through the decay the cosine gives base/2, got {got[mid]:g}"
        assert close(got[total], 0.0, 1e-9) and close(got[total + 4], 0.0, 1e-9), "at and after total_steps: min_lr (0)"
        decay = got[warm:total + 1]
        assert all(a >= b - 1e-15 for a, b in zip(decay, decay[1:])), "after warmup the LR must never increase"
        for s in (0, 5, 37, 80, 109):
            want = _reference_lr(s, base, warm, total)
            assert close(got[s], want, 1e-9), f"step {s}: expected {want:.6g}, got {got[s]:.6g}"
        lo = warmup_cosine_lr(200, 0.1, 0, 100, min_lr=0.01)
        assert close(lo, 0.01), f"min_lr is the floor after total_steps, got {lo}"
        assert close(warmup_cosine_lr(0, 0.1, 0, 100, min_lr=0.01), 0.1), "warmup_steps=0: start at base_lr"
        assert close(warmup_cosine_lr(50, 0.1, 0, 100, min_lr=0.01), 0.055), "half-way between base_lr and min_lr"
        for bad in [(-1, 0.1, 10, 100), (5, 0.1, 200, 100)]:
            try:
                warmup_cosine_lr(*bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"warmup_cosine_lr{bad} must raise ValueError")

    return run_check("Exercise 1 · warmup_cosine_lr", test,
                     hint="if step < warmup: return base*(step+1)/warmup; progress = min(1, (step-warmup)/max(1, "
                          "total-warmup)); return min_lr + (base-min_lr)*0.5*(1+cos(pi*progress))")


def ex2(critique_flags):
    def test():
        got = critique_flags()
        assert isinstance(got, (set, frozenset, list, tuple)), f"return a set of letters, got {type(got).__name__}"
        letters = {str(x).strip().upper() for x in got}
        unknown = letters - set(_OPTIONS)
        assert not unknown, f"options are A-I; unknown: {sorted(unknown)}"
        hashes = {_h(c): c for c in letters}
        wrong = sorted(c for h, c in hashes.items() if h not in _FLAW_HASHES)
        missed = len(_FLAW_HASHES - set(hashes))
        problems = []
        if wrong:
            problems.append(f"{len(wrong)} of your choices are not planted flaws")
        if missed:
            problems.append(f"you missed {missed} planted flaw(s)")
        assert not problems, "; ".join(problems) + " (the answer is checked as a whole)"

    return run_check("Exercise 2 · critique_flags", test,
                     hint="go through the template: claim vs split, selection on test, baseline tuning, uncertainty, "
                          "preprocessing before the split; a general opinion is not a flaw in this study")
