"""Reference solutions for Level 2 notebook 17 (graded in CI)."""

from __future__ import annotations

import concurrent.futures as cf
from collections import OrderedDict


class ex1:  # noqa: N801 - graded as `LRUCache`
    """LRU cache with hit statistics; O(1) get and put."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return default

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def ex2(ranked_items, item_meta, max_per_category, k):
    seen, per_cat, out, capped = set(), {}, [], []
    for item in ranked_items:
        if item in seen:
            continue
        seen.add(item)
        meta = item_meta.get(item)
        if meta is None or not meta["launched"]:
            continue
        cat = meta["category"]
        if per_cat.get(cat, 0) >= max_per_category:
            capped.append(item)
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        out.append(item)
        if len(out) == k:
            return out
    return out + capped[: k - len(out)]


_POOL = cf.ThreadPoolExecutor(max_workers=8)


def ex3(primary_fn, fallback_fn, timeout_s):
    def wrapped(*args, **kwargs):
        future = _POOL.submit(primary_fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s), "ok"
        except cf.TimeoutError:
            return fallback_fn(*args, **kwargs), "timeout"
        except Exception:
            return fallback_fn(*args, **kwargs), "error"

    return wrapped
