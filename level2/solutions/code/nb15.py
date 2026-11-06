"""Reference solutions for notebook 15 · Coding patterns I. Graded by tests/test_solutions.py."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict


def ex1(nums, target):
    """All index pairs (i, j), i < j, with nums[i] + nums[j] == target, sorted. O(n + output)."""
    seen = defaultdict(list)                 # value -> indices seen so far
    pairs = []
    for j, x in enumerate(nums):
        for i in seen.get(target - x, ()):
            pairs.append((i, j))
        seen[x].append(j)
    pairs.sort()
    return pairs


def ex2(s):
    """Length of the longest substring without repeating characters. Sliding window, O(n)."""
    last = {}
    best = left = 0
    for right, ch in enumerate(s):
        if last.get(ch, -1) >= left:         # repeat inside the window: jump left past it
            left = last[ch] + 1
        last[ch] = right
        best = max(best, right - left + 1)
    return best


def ex3(items, k):
    """k most frequent items, most frequent first; ties by smaller item. O(n + u log k) with a heap."""
    if k <= 0:
        return []
    counts = Counter(items)
    # heap of size k keyed so that the "worst" kept item is on top
    heap = []
    for item, c in counts.items():
        key = (c, _Neg(item))
        if len(heap) < k:
            heapq.heappush(heap, (key, item))
        elif key > heap[0][0]:
            heapq.heapreplace(heap, (key, item))
    return [item for _, item in sorted(heap, reverse=True)]


class _Neg:
    """Reverses the order of an item, so a min-heap on (count, _Neg(item)) evicts larger items first on ties."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __lt__(self, other):
        return other.v < self.v

    def __eq__(self, other):
        return self.v == other.v


def ex4(intervals, new=None):
    """Merge closed intervals (touching ones merge). With ``new``: insert it into a sorted,
    non-overlapping list in O(n). Returns a new list of [start, end] lists."""
    if new is None:
        out = []
        for s, e in sorted(intervals):
            if out and s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out
    s, e = new
    out, i, n = [], 0, len(intervals)
    while i < n and intervals[i][1] < s:     # entirely before
        out.append(list(intervals[i]))
        i += 1
    while i < n and intervals[i][0] <= e:    # overlapping or touching: absorb
        s, e = min(s, intervals[i][0]), max(e, intervals[i][1])
        i += 1
    out.append([s, e])
    out.extend(list(iv) for iv in intervals[i:])
    return out


class ex5:  # noqa: N801  (graded as a class: MinStack)
    """Stack with O(1) push, pop, top and get_min. Empty pop/top/get_min raise IndexError."""

    def __init__(self):
        self._items = []
        self._mins = []                      # _mins[i] = min of _items[: i + 1]

    def push(self, x):
        self._items.append(x)
        self._mins.append(x if not self._mins or x < self._mins[-1] else self._mins[-1])

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        self._mins.pop()
        return self._items.pop()

    def top(self):
        if not self._items:
            raise IndexError("top of empty stack")
        return self._items[-1]

    def get_min(self):
        if not self._mins:
            raise IndexError("min of empty stack")
        return self._mins[-1]

    def __len__(self):
        return len(self._items)


def ex6(nums):
    """out[i] = product of all nums except nums[i], without division. O(n) time, O(1) extra."""
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):
        out[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out


def ex7(nums, k):
    """Number of contiguous subarrays summing to k (negatives allowed). Prefix sums + hash map, O(n)."""
    count = 0
    prefix = 0
    seen = {0: 1}                            # prefix-sum value -> how many times seen
    for x in nums:
        prefix += x
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
