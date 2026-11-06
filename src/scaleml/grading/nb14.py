"""Self-checks for Level 3 notebook 14 · Multi-task ranking & exploration."""

from __future__ import annotations

from churnlab.checks import run_check


def _unpack(out):
    assert isinstance(out, tuple) and len(out) == 2, f"return a tuple (winner, price_per_click), got {out!r}"
    winner, price = out
    return (None if winner is None else int(winner)), float(price)


def ex1(second_price_auction):
    def test():
        # A: bid 2.0, pCTR 0.05 -> eCPM 100; B: bid 4.0, pCTR 0.02 -> 80; C: bid 1.0, pCTR 0.03 -> 30
        bids, pctrs = [2.0, 4.0, 1.0], [0.05, 0.02, 0.03]
        w, price = _unpack(second_price_auction(bids, pctrs, 0.0))
        assert w == 0, f"the winner is the highest bid x pCTR (index 0, eCPM 100), not the highest bid; got {w}"
        assert abs(price - 4.0 * 0.02 / 0.05) < 1e-9, (
            f"price per click should be the runner-up's bid x pCTR divided by the winner's pCTR = 1.6, got {price}")
        assert price <= bids[0] + 1e-12, "a winner never pays more than its own bid"
        w, price = _unpack(second_price_auction(bids, pctrs, 1.8))
        assert w == 0 and abs(price - 1.8) < 1e-9, (
            f"with a reserve of 1.8 per click, A still wins but pays max(reserve, 1.6) = 1.8; got {(w, price)}")
        w, price = _unpack(second_price_auction(bids, pctrs, 3.0))
        assert w == 1 and abs(price - 3.0) < 1e-9, (
            f"reserve 3.0 removes bids below it: only B (bid 4.0) is eligible and pays the reserve; got {(w, price)}")
        w, price = _unpack(second_price_auction(bids, pctrs, 5.0))
        assert w is None and price == 0.0, f"no bid reaches the reserve: return (None, 0.0); got {(w, price)}"
        w, price = _unpack(second_price_auction([1.0], [0.1], 0.0))
        assert w == 0 and price == 0.0, f"a single bidder with no reserve pays 0 (no competition); got {(w, price)}"
        w, price = _unpack(second_price_auction([1.0, 2.0], [0.1, 0.05], 0.0))
        assert w == 0 and abs(price - 1.0) < 1e-9, f"ties go to the lowest index, who pays its own bid here; got {(w, price)}"

    return run_check("Exercise 1 · second_price_auction", test,
                     hint="eligible = bid >= reserve; rank eligible by bid*pctr (stable, lowest index wins ties); "
                          "price = max(reserve, bid2*pctr2/pctr_winner), or reserve if there is no runner-up")


def ex2(thompson_sampling):
    def test():
        import numpy as np

        for true_ps, n in [([0.02, 0.04, 0.06, 0.1, 0.05], 4000), ([0.2, 0.5, 0.35], 2000)]:
            ps = np.asarray(true_ps)
            runs = []
            for seed in (0, 1, 2):
                reg = np.asarray(thompson_sampling(true_ps, n, seed), dtype=float)
                assert reg.shape == (n,), f"return one cumulative-regret value per round: shape ({n},), got {reg.shape}"
                assert np.all(np.diff(reg) >= -1e-12) and reg[0] >= 0, (
                    "cumulative regret must start >= 0 and never decrease")
                assert np.all(np.isin(np.round(np.diff(reg, prepend=0.0), 10), np.round(ps.max() - ps, 10))), (
                    "each round's regret must be max(true_ps) - true_ps[chosen arm] (expected, not realised, regret)")
                runs.append(reg)
            reg = np.mean(runs, axis=0)  # average over 3 seeds: the check should not depend on luck
            uniform = n * (ps.max() - ps.mean())
            assert reg[-1] < 0.45 * uniform, (
                f"mean final regret {reg[-1]:.1f} is not much better than choosing uniformly at random ({uniform:.1f})")
            third = n // 3
            first, last = reg[third - 1], reg[-1] - reg[-third - 1]
            assert last < 0.8 * first, (
                f"regret should grow sublinearly: last third added {last:.1f} vs first third {first:.1f}")
            again = np.asarray(thompson_sampling(true_ps, n, 0), dtype=float)
            assert np.array_equal(runs[0], again), "same seed must give the same result: use np.random.default_rng(seed)"

    return run_check("Exercise 2 · thompson_sampling", test,
                     hint="alpha = beta = ones(K); each round: theta = rng.beta(alpha, beta); a = argmax(theta); "
                          "reward = rng.random() < p[a]; update alpha/beta; regret += max(p) - p[a]")
