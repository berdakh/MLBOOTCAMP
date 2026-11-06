# Solutions · 16 · Coding patterns II

Try each exercise against the clock before reading its answer. Markers:
✅ verified by running · 🔬 expected, depends on your run · ✍️ personal / open-ended.

Reference code: [`code/nb16.py`](code/nb16.py), graded in CI against
[`src/shoprank/grading/nb16.py`](../../src/shoprank/grading/nb16.py).

---

## 16-1 · `num_islands`

✅ Reference: `ex1` in [`code/nb16.py`](code/nb16.py) — scan; on unseen land,
count and BFS-flood with a `deque`, marking cells when they are enqueued.

**Complexity.** O(R·C) time; O(R·C) for `seen`.

✅ A recursive flood fill failed with `RecursionError` on the grader's
120 × 120 all-land grid.

**Edge cases.** `[]`, `[[]]`, single cell, diagonal neighbours (not
connected), one giant island, many single-cell islands, a snake-shaped island
(deep DFS paths).

---

## 16-2 · `course_order`

✅ Reference: `ex2` in [`code/nb16.py`](code/nb16.py) — Kahn's algorithm.

```python
graph = [[] for _ in range(n)]; indeg = [0] * n
for course, pre in prerequisites:
    graph[pre].append(course); indeg[course] += 1
queue = deque(i for i in range(n) if indeg[i] == 0); order = []
while queue:
    u = queue.popleft(); order.append(u)
    for v in graph[u]:
        indeg[v] -= 1
        if indeg[v] == 0: queue.append(v)
return order if len(order) == n else []
```

**Complexity.** O(V + E) time and space. Duplicate edges are harmless (each
adds one to the in-degree and is removed once). A self-loop gives an in-degree
that never reaches 0 → reported as a cycle.

**Parallelism follow-up.** Process Kahn's queue level by level: every node in
the same level has all prerequisites done and can run in parallel; the number
of levels is the critical path length.

---

## 16-3 · `dijkstra`

✅ Reference: `ex3` in [`code/nb16.py`](code/nb16.py) — adjacency lists,
heap of `(dist, node)`, skip stale entries.

**Complexity.** O((V + E) log V) time (each edge can push once), O(V + E) space.

✅ An O(V²) version (linear scan for the next node) passed every correctness
test and was stopped by the timing projection after 4,000 nodes (0.7-0.9 s,
projected ~12 s at 16,000).

**Edge cases.** Unreachable nodes stay `inf`; zero-weight edges are fine;
parallel edges (an adjacency *dict* would keep only one); self-loops never
improve a distance; the source need not be 0.

---

## 16-4 · `min_capacity`

✅ Reference: `ex4` in [`code/nb16.py`](code/nb16.py).

```python
lo, hi = max(weights), sum(weights)
while lo < hi:
    mid = (lo + hi) // 2
    if days_needed(mid) <= days: hi = mid
    else:                        lo = mid + 1
return lo
```

with `days_needed(cap)` a greedy scan: start a new day when the next package
does not fit.

**Why greedy feasibility is exact.** For a fixed capacity, loading as much as
possible each day leaves the remaining suffix as short as possible, so no
other loading can finish in fewer days (exchange argument).

**Complexity.** O(n · log(sum − max)) time, O(1) space.

✅ Scanning capacities upwards from `max(w)` was stopped by the timing check
(5.9 s at 4,000 packages).

---

## 16-5 · `lis_length`

✅ Reference: `ex5` in [`code/nb16.py`](code/nb16.py).

```python
tails = []
for x in nums:
    i = bisect.bisect_left(tails, x)
    if i == len(tails): tails.append(x)
    else:               tails[i] = x
return len(tails)
```

**Why `tails` stays sorted.** `tails[L]` is the smallest tail of any increasing
subsequence of length L + 1. A subsequence of length L + 2 contains one of
length L + 1 with a smaller tail, so `tails[L] < tails[L+1]`. Replacing the first
tail ≥ x by x keeps that property; appending happens only when x exceeds every
tail.

**Complexity.** O(n log n) time, O(n) space. `tails` is generally **not** a
valid subsequence (`[3, 4, 1]` ends with `tails = [1, 4]`); to reconstruct one,
store for each element the index of its predecessor.

---

## 16-6 · `edit_distance`

✅ Reference: `ex6` in [`code/nb16.py`](code/nb16.py) — bottom-up with two rows.

**Complexity.** O(m·n) time, O(min(m, n)) space (swap so the shorter string
indexes the row).

✅ Plain recursion without memoisation was stopped by the timing check at
length 10 (≈0.4 s there, projected far beyond the budget at length 300).
`functools.lru_cache` on a recursive `(i, j)` version is also O(m·n) and
accepted (recursion depth m + n = 600 at the largest test).

---

## 16-7 · `Trie`

✅ Reference: `ex7` in [`code/nb16.py`](code/nb16.py) — nested dicts with a
private sentinel object as the end-of-word marker (a sentinel cannot collide
with a real character, unlike a key such as `"$"` or `"#"`).

**Complexity.** O(L) per operation; space O(total characters inserted).

✅ A list-of-words implementation (`any(w.startswith(p) for w in words)`) was
stopped by the timing projection after 4,000 words.

---

## 16-8 · `UnionFind`

✅ Reference: `ex8` in [`code/nb16.py`](code/nb16.py) — iterative path
compression and union by size.

**Complexity.** O(α(n)) amortised per operation with both optimisations,
O(log n) with either alone; O(n) space.

✅ With neither optimisation, `union(0, i)` for every i built a chain and took
5.8 s at 16,000 nodes in the grader.

---

## Checkpoint · 16

1. **BFS vs Dijkstra vs Bellman-Ford.** BFS gives shortest paths when every edge
   has the same weight (it minimises hop count). Dijkstra handles non-negative
   weights. With negative edges Dijkstra can finalise a node too early and
   return wrong distances; use Bellman-Ford (O(V·E), also detects negative
   cycles) or, on a DAG, relax edges in topological order (O(V + E)). 0/1
   weights: 0-1 BFS with a deque. ✅ (16.3, 16-3)
2. **Pipeline dependencies.** Model tasks as a DAG (edge prereq → task); Kahn's
   algorithm yields an execution order; if fewer than all tasks are output,
   the remaining ones (in-degree > 0) contain the cycle — report them, or run
   a DFS with colours to print the actual cycle. Tasks in the same Kahn level
   can run in parallel; the longest path (by expected duration) is the critical
   path that bounds the total runtime. ✅ (16-2)
3. **Binary search on the answer.** A numeric answer in a known range and a
   monotone feasibility check that is cheaper than direct optimisation.
   ML-systems examples: the smallest number of replicas (or batch size) that
   keeps p99 latency under budget in a load test; the highest score threshold
   that keeps precision ≥ 95% (monotone if precision is monotone in the
   threshold — check, it is not always); the largest learning rate that does
   not diverge (roughly monotone, noisy). ✍️
4. **O(n log n) LIS.** See 16-5: `tails[L]` is the smallest tail of an increasing
   subsequence of length L + 1, sorted by an exchange argument; each element is
   placed by binary search. `tails` has the right **length** but mixes
   elements from different subsequences; reconstruction needs predecessor
   indices. ✅ (16-5)
5. **Autocomplete over 10M queries.** Hash set: O(1) exact lookup but no prefix
   queries (you would scan everything). Sorted array + binary search: compact
   (just the strings), prefix query = two binary searches O(log n · L) plus the
   matches, but inserts are O(n) (rebuild in batches). Trie: prefix query
   O(L + matches), inserts O(L), but memory-hungry (a node per character; Python
   dicts especially) — compressed tries / FSTs fix that. In production, top-k
   suggestions are usually precomputed per prefix offline and served from a
   key-value store or an FST, refreshed daily. ✍️
6. **Near-duplicate clustering.** Union-find over the candidate pairs (above a
   similarity threshold) gives connected components in near-linear time and
   streams well; shard by blocking keys to parallelise, or run connected
   components in a distributed framework. Risk: transitivity — A≈B and B≈C
   merges A with C even if they are dissimilar, and a few false-positive edges
   can chain into one giant cluster. Mitigate with a stricter threshold, a cap
   on component size or diameter, re-checking large components with a
   stricter pairwise rule (or correlation clustering), and auditing the
   largest clusters by hand. ✍️
