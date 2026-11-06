"""Self-checks for Level 2 notebook 17 · Serving a ranking cascade."""

from __future__ import annotations

import random
import time

from churnlab.checks import run_check


def ex1(LRUCache):
    def test():
        c = LRUCache(2)
        assert c.get("a") is None, "get on an empty cache should return None"
        c.put("a", 1)
        c.put("b", 2)
        assert c.get("a") == 1, "get('a') should return 1"
        c.put("c", 3)  # evicts b: a was used more recently
        assert c.get("b") is None, "'b' was least recently used and should have been evicted"
        assert c.get("a") == 1 and c.get("c") == 3, "'a' and 'c' should still be cached"
        c.put("a", 10)  # update moves to most-recent
        c.put("d", 4)  # evicts c
        assert c.get("a") == 10, "put on an existing key must update its value"
        assert c.get("c") is None, "updating 'a' should have made 'c' the least recently used"
        assert len(c) == 2, f"len(cache) should be 2, got {len(c)}"
        assert (c.hits, c.misses) == (4, 3), f"expected hits=4, misses=3, got hits={c.hits}, misses={c.misses}"
        assert abs(c.hit_rate - 4 / 7) < 1e-9, f"hit_rate should be hits/(hits+misses)=4/7, got {c.hit_rate}"
        assert LRUCache(3).hit_rate == 0.0, "hit_rate of an unused cache should be 0.0"
        c0 = LRUCache(1)
        c0.put(0, None)
        assert len(c0) == 1

        def per_op(capacity: int, n_ops: int = 20_000) -> float:
            best = float("inf")
            for _ in range(3):
                cache = LRUCache(capacity)
                for k in range(capacity):
                    cache.put(k, k)
                rng = random.Random(0)
                keys = [rng.randrange(capacity * 2) for _ in range(n_ops)]
                t = time.perf_counter()
                for k in keys:
                    if cache.get(k) is None:
                        cache.put(k, k)
                best = min(best, (time.perf_counter() - t) / n_ops)
            return best

        small, large = per_op(100), per_op(20_000)
        assert large < 10 * small, (
            f"operations got {large / small:.0f}x slower with 200x more entries - get/put are not O(1)")

    return run_check("Exercise 1 · LRUCache", test,
                     hint="collections.OrderedDict: move_to_end(key) on every hit/put, popitem(last=False) to evict")


def ex2(apply_business_rules):
    def test():
        meta = {i: {"category": i % 3, "launched": True} for i in range(12)}
        meta[5]["launched"] = False
        ranked = [0, 3, 3, 6, 5, 1, 9, 4, 2, 7, 99]
        out = apply_business_rules(ranked, meta, 2, 5)
        assert list(out) == [0, 3, 1, 4, 2], f"got {list(out)}; expected [0, 3, 1, 4, 2]"
        out = apply_business_rules(ranked, meta, 10, 4)
        assert list(out) == [0, 3, 6, 1], f"no cap: got {list(out)}, expected [0, 3, 6, 1]"
        out = apply_business_rules([0, 3, 6, 9, 1], meta, 1, 4)
        assert list(out) == [0, 1, 3, 6], (
            f"got {list(out)}; when the cap leaves fewer than k items, backfill with the "
            "highest-ranked items skipped only by the cap, in rank order, after the others: [0, 1, 3, 6]")
        assert list(apply_business_rules([5, 99], meta, 2, 3)) == [], "unlaunched and unknown items must be removed"
        assert list(apply_business_rules([], meta, 2, 3)) == []
        assert list(apply_business_rules([1, 1, 1], meta, 1, 3)) == [1], "duplicates must not be backfilled"

    return run_check("Exercise 2 · apply_business_rules", test,
                     hint="one pass: skip seen ids, skip unknown/unlaunched, keep a per-category counter; "
                          "remember cap-skipped items for the backfill")


def ex3(with_fallback):
    def test():
        def double(x):
            return 2 * x

        def boom(x):
            raise ValueError("ranker down")

        def slow(x):
            time.sleep(0.4)
            return 2 * x

        def fallback(x):
            return -x

        f = with_fallback(double, fallback, 0.2)
        assert f(3) == (6, "ok"), f"healthy primary: expected (6, 'ok'), got {f(3)!r}"
        g = with_fallback(boom, fallback, 0.2)
        assert g(3) == (-3, "error"), f"failing primary: expected (-3, 'error'), got {g(3)!r}"
        h = with_fallback(slow, fallback, 0.05)
        t = time.perf_counter()
        out = h(3)
        took = time.perf_counter() - t
        assert out == (-3, "timeout"), f"slow primary: expected (-3, 'timeout'), got {out!r}"
        assert took < 0.3, f"the call took {took:.2f}s; it must return soon after timeout_s=0.05"
        kw = with_fallback(lambda a, b=0: a + b, lambda a, b=0: a - b, 0.2)
        assert kw(5, b=2) == (7, "ok"), "pass *args and **kwargs through to primary_fn"

    return run_check("Exercise 3 · with_fallback", test,
                     hint="submit primary_fn to a ThreadPoolExecutor; future.result(timeout=...); "
                          "catch concurrent.futures.TimeoutError and Exception separately")
