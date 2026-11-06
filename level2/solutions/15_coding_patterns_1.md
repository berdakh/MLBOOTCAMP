# Solutions · 15 · Coding patterns I

Try each exercise against the clock before reading its answer. Markers:
✅ verified by running · 🔬 expected, depends on your run · ✍️ personal / open-ended.

Reference code: [`code/nb15.py`](code/nb15.py), graded in CI against
[`src/shoprank/grading/nb15.py`](../../src/shoprank/grading/nb15.py). Each grader
runs visible and hidden edge cases, ~200 random inputs against a brute-force
reference, and a timing check (2-second budget per size).

---

## 15-1 · `two_sum_pairs`

✅ Reference: `ex1` in [`code/nb15.py`](code/nb15.py).

```python
seen = defaultdict(list)            # value -> indices so far
pairs = []
for j, x in enumerate(nums):
    for i in seen.get(target - x, ()):
        pairs.append((i, j))
    seen[x].append(j)
pairs.sort()
```

**Complexity.** O(n + p log p) time where p = number of pairs (the pairs come
out grouped by j; the final sort orders them by i first — skip it if the
interviewer does not need sorted output, giving O(n + p)), O(n) space. Worst case p = n(n−1)/2 (all equal values with
target = 2x): output-bound, unavoidable.

**Edge cases.** Single element; the same index cannot pair with itself (we add
`j` to `seen` *after* looking up); duplicates; negatives; empty input.

✅ The all-pairs double loop was correct but took 4.4 s at n = 16,000 in the
grader.

---

## 15-2 · `longest_unique`

✅ Reference: `ex2` in [`code/nb15.py`](code/nb15.py).

```python
last, best, left = {}, 0, 0
for right, ch in enumerate(s):
    if last.get(ch, -1) >= left:
        left = last[ch] + 1
    last[ch] = right
    best = max(best, right - left + 1)
```

**Complexity.** O(n) time — `left` and `right` each move forward at most n
times; O(min(n, alphabet)) space.

✅ Without the `>= left` check (`if ch in last: left = last[ch] + 1`), `"abba"`
returns 3 instead of 2: at the final `a`, `left` jumps *back* from 2 to 1.

---

## 15-3 · `top_k_frequent`

✅ Reference: `ex3` in [`code/nb15.py`](code/nb15.py) (a size-k heap with a
reversed-order wrapper for tie-breaking). A shorter equivalent that is
perfectly acceptable in an interview:

```python
counts = Counter(items)
return heapq.nsmallest(k, counts, key=lambda x: (-counts[x], x)) if k > 0 else []
```

`heapq.nsmallest` keeps a heap of size k internally: O(u log k) after the
O(n) count, for u distinct items. Bucket sort by frequency gives O(n).

**Edge cases.** k = 0, k > u, ties (the specification decides; ask), items that
are not mutually comparable (then tie-break by first occurrence instead).

---

## 15-4 · `merge_intervals` (merge + insert)

✅ Reference: `ex4` in [`code/nb15.py`](code/nb15.py).

```python
if new is None:                                   # O(n log n)
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
s, e = new; out, i = [], 0                        # insert: O(n)
while i < len(intervals) and intervals[i][1] < s:  out.append(list(intervals[i])); i += 1
while i < len(intervals) and intervals[i][0] <= e: s, e = min(s, intervals[i][0]), max(e, intervals[i][1]); i += 1
out.append([s, e]); out.extend(list(iv) for iv in intervals[i:])
```

**Complexity.** Merge O(n log n) time, O(n) output; insert O(n) — you cannot
beat O(n) because the output is a new list (with in-place deletion you could
find the boundaries by binary search in O(log n), but deleting is still O(n)
in an array).

**Edge cases.** Touching closed intervals (`[1,4],[4,5]` merge); nesting
(`[1,10],[2,3]` — the `max`); `new` before all, after all, inside one, spanning
several; empty list; negative coordinates.

---

## 15-5 · `MinStack`

✅ Reference: `ex5` in [`code/nb15.py`](code/nb15.py) — a parallel stack
`mins[i] = min(items[:i+1])`.

**Complexity.** O(1) for every operation; O(n) extra space (the space-saving
variant pushes to `mins` only when `x <= mins[-1]`).

✅ A `min(self.items)` version passed all correctness tests and failed the
timing (2.2 s at 16,000 operations of each kind).

---

## 15-6 · `product_except_self`

✅ Reference: `ex6` in [`code/nb15.py`](code/nb15.py).

```python
out = [1] * n; prefix = 1
for i in range(n):           out[i] = prefix; prefix *= nums[i]
suffix = 1
for i in range(n - 1, -1, -1): out[i] *= suffix; suffix *= nums[i]
```

**Complexity.** O(n) time; O(1) extra space besides the output.

✅ The division shortcut (`total // x`) raised `ZeroDivisionError` on the
grader's first zero.

---

## 15-7 · `subarray_sum_count`

✅ Reference: `ex7` in [`code/nb15.py`](code/nb15.py).

```python
count, prefix, seen = 0, 0, {0: 1}
for x in nums:
    prefix += x
    count += seen.get(prefix - k, 0)
    seen[prefix] = seen.get(prefix, 0) + 1
```

**Why the order of the last two lines matters.** Looking up before inserting
the current prefix excludes the empty subarray (j = i) when k = 0.

**Complexity.** O(n) time, O(n) space.

✅ Two brute-force versions (with and without a running sum) were rejected by
the timing projection at n = 1,000 and n = 4,000.

---

## Checkpoint · 15

1. **First non-repeating character.** Clarify: character set, case
   sensitivity, return the character or index, what if none (−1 / None),
   input size. Examples: `"leetcode"` → `'l'`; `"aabb"` → none. Brute force:
   for each character, count it in the whole string, O(n²). Optimise: one pass
   with a `Counter`, a second pass returning the first character with count 1.
   Complexity: O(n) time, O(alphabet) space. Streaming follow-up: a dict of
   counts plus an ordered dict/queue of candidates. ✍️
2. **Sliding window cost.** Each index is added once (when `right` passes it)
   and removed at most once (when `left` passes it), so the inner `while` runs
   at most n times **in total**: amortised O(n). It fails when shrinking the
   window does not move the constraint monotonically — e.g. sums with negative
   numbers, or "exactly k" constraints — and is replaced by prefix sums with a
   hash map (15-7), or by "at most k" minus "at most k−1" tricks. ✅ (15-2, 15-7)
3. **Top 100 of 10 billion lines.** Exact: hash-partition the log by query
   into files/shards that fit in memory (a map-reduce shuffle, notebook 02),
   count per shard, keep a size-100 heap per shard, merge the heaps. Two
   passes over the data, exact. Approximate, one pass, small memory: a
   Count-Min sketch (or Space-Saving/Misra-Gries) plus a size-100 heap; the
   counts are overestimates with bounded error, and the top-100 set can be
   wrong near the cut-off — re-count the candidates exactly in a second pass if
   needed. ✍️
4. **Hash map worst case.** O(n) per operation when many keys collide in the
   same bucket (pathological or adversarial keys). Python randomises `str` and
   `bytes` hashing per process (`PYTHONHASHSEED`) to make crafted collisions
   hard; ints are not randomised (`hash(i) == i` for small ints), so keys that
   are multiples of the table size can collide. It matters for services
   hashing untrusted input (hash-flooding DoS) and for skewed join keys in
   distributed systems (the same "hot key" effect from notebook 02). ✍️
5. **Sort over hash.** When the output must be ordered anyway; when memory is
   tight (in-place sort, O(1) extra vs a hash table's overhead); when data is
   already sorted or nearly sorted (Timsort is O(n) there); for external data
   that does not fit in memory (external merge sort streams well; a hash table
   does not); when worst-case guarantees matter; and when the hash-based
   constants (hashing strings, cache misses) dominate at realistic n. ✍️
6. **O(1) extra space.** 15-6: write prefix products into the output, then
   multiply by a running suffix — O(1) extra (the output does not count). 15-1
   with a **sorted** input: two pointers from both ends, O(1) extra; on
   unsorted input you must sort first (O(n log n), and indices change unless you
   sort index pairs, which is O(n) space again). 15-7 with negatives cannot be
   done in O(1) extra space and O(n) time by any known method: say so and
   explain the trade-off (O(n²) time with O(1) space). ✍️
