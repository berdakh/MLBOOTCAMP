"""Self-checks for Chapter 34 · Coding patterns II (graphs, binary search, DP, tries, union-find)."""

from __future__ import annotations

import math
import random

from churnlab.checks import run_check
from shoprank.drills import check_scaling


def _ref_islands(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen, count = set(), 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in seen:
                count += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    i, j = stack.pop()
                    for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= a < rows and 0 <= b < cols and grid[a][b] == 1 and (a, b) not in seen:
                            seen.add((a, b))
                            stack.append((a, b))
    return count


def ex1(num_islands):
    def test():
        g1 = [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
        cases = [(g1, 3), ([], 0), ([[]], 0), ([[0]], 0), ([[1]], 1), ([[1, 0], [0, 1]], 2),
                 ([[1, 1, 1], [0, 1, 0], [1, 1, 1]], 1), ([[1, 0, 1, 0, 1]], 3), ([[1], [1], [0], [1]], 2)]
        for grid, want in cases:
            snap = [row[:] for row in grid]
            got = num_islands(grid)
            assert grid == snap, "your function modified the grid - copy it or keep a separate 'seen' set"
            assert got == want, f"num_islands({grid}) = {got}, expected {want} (diagonals do not connect)"
        rng = random.Random(0)
        for _ in range(100):
            r, c = rng.randint(1, 8), rng.randint(1, 8)
            grid = [[int(rng.random() < 0.5) for _ in range(c)] for _ in range(r)]
            assert num_islands(grid) == _ref_islands(grid), f"wrong for {grid}"
        big = [[1] * 120 for _ in range(120)]           # one island, 14,400 cells deep for a recursive DFS
        assert num_islands(big) == 1, "a 120x120 grid of 1s is one island"
        snake = [[1 if (r % 4 != 1 or c == 79) and (r % 4 != 3 or c == 0) else 0 for c in range(80)] for r in range(80)]
        assert num_islands(snake) == _ref_islands(snake), "wrong on a long snake-shaped island"

        def make(n):
            side = math.isqrt(n)
            r = random.Random(n)
            return ([[int(r.random() < 0.55) for _ in range(side)] for _ in range(side)],)

        check_scaling(num_islands, make, [125 ** 2, 250 ** 2, 500 ** 2], label="num_islands",
                      validate=lambda n, args, res: _check_eq(res, _ref_islands(args[0]), n))

    return run_check("Exercise 1 · num_islands", test,
                     hint="scan every cell; on an unseen 1, count an island and flood it with an explicit "
                          "queue/stack (a recursive DFS hits Python's recursion limit on big islands)")


def _check_eq(got, want, n):
    assert got == want, f"wrong answer at n={n:,}: got {got}, expected {want}"


def _valid_order(n, prereqs, order):
    if sorted(order) != list(range(n)):
        return False
    pos = {c: i for i, c in enumerate(order)}
    return all(pos[pre] < pos[course] for course, pre in prereqs)


def _has_cycle(n, prereqs):
    indeg = [0] * n
    g = [[] for _ in range(n)]
    for c, p in prereqs:
        g[p].append(c)
        indeg[c] += 1
    q = [i for i in range(n) if indeg[i] == 0]
    seen = 0
    while q:
        u = q.pop()
        seen += 1
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return seen < n


def ex2(course_order):
    def test():
        cases = [(2, [[1, 0]]), (2, [[1, 0], [0, 1]]), (1, [[0, 0]]), (0, []), (3, []),
                 (4, [[1, 0], [2, 0], [3, 1], [3, 2]]), (3, [[1, 0], [1, 0]]), (3, [[0, 1], [1, 2], [2, 0]]),
                 (5, [[1, 0], [3, 2]])]
        for n, pre in cases:
            got = list(course_order(n, pre))
            if _has_cycle(n, pre):
                assert got == [], f"course_order({n}, {pre}) has a cycle: expected [], got {got}"
            else:
                assert _valid_order(n, pre, got), f"course_order({n}, {pre}) = {got} is not a valid order"
        rng = random.Random(1)
        for _ in range(200):
            n = rng.randint(1, 8)
            pre = [[rng.randrange(n), rng.randrange(n)] for _ in range(rng.randint(0, 10))]
            got = list(course_order(n, pre))
            if _has_cycle(n, pre):
                assert got == [], f"cycle in {pre} not detected"
            else:
                assert _valid_order(n, pre, got), f"invalid order {got} for n={n}, {pre}"

        def make(n):                                   # a long chain plus random extra forward edges
            r = random.Random(n)
            perm = list(range(n))
            r.shuffle(perm)
            pre = [[perm[i + 1], perm[i]] for i in range(n - 1)]
            for _ in range(n):
                i, j = sorted(r.sample(range(n), 2))
                pre.append([perm[j], perm[i]])
            return n, pre

        def validate(n, args, result):
            assert _valid_order(n, args[1], list(result)), f"invalid order at n={n:,}"

        check_scaling(course_order, make, [1000, 4000, 16000, 64000], label="course_order", validate=validate)

    return run_check("Exercise 2 · course_order", test,
                     hint="Kahn: indegree counts, queue of indegree-0 courses, pop/append/decrement; if fewer than n "
                          "courses come out there is a cycle. Recursive DFS overflows on the 64,000-long chain")


def _ref_dist(n, edges, s):
    dist = [math.inf] * n
    dist[s] = 0
    for _ in range(n):                                 # Bellman-Ford
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    return dist


def ex3(dijkstra):
    def test():
        edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1), (2, 3, 5)]
        assert list(dijkstra(4, edges, 0)) == [0, 3, 1, 4], f"dijkstra(4, {edges}, 0) = {dijkstra(4, edges, 0)}"
        assert list(dijkstra(3, [(0, 1, 2)], 0)) == [0, 2, math.inf], "unreachable nodes must be float('inf')"
        assert list(dijkstra(1, [], 0)) == [0]
        assert list(dijkstra(3, [(0, 1, 5), (0, 1, 2), (1, 1, 1), (1, 2, 0)], 0)) == [0, 2, 2], (
            "parallel edges, a self-loop and a zero-weight edge")
        assert list(dijkstra(3, [(1, 0, 1), (2, 1, 1)], 2)) == [2, 1, 0], "edges are directed; source need not be 0"
        rng = random.Random(2)
        for _ in range(100):
            n = rng.randint(1, 9)
            es = [(rng.randrange(n), rng.randrange(n), rng.randint(0, 9)) for _ in range(rng.randint(0, 20))]
            s = rng.randrange(n)
            assert list(dijkstra(n, es, s)) == _ref_dist(n, es, s), f"wrong for n={n}, edges={es}, source={s}"

        def make(n):
            r = random.Random(n)
            es = [(i, i + 1, r.randint(1, 100)) for i in range(n - 1)]
            es += [(r.randrange(n), r.randrange(n), r.randint(1, 100)) for _ in range(4 * n)]
            return n, es, 0

        def validate(n, args, result):
            assert len(result) == n and result[0] == 0 and all(d < math.inf for d in result), f"wrong at n={n:,}"

        check_scaling(dijkstra, make, [1000, 4000, 16000, 64000], label="dijkstra", validate=validate)

    return run_check("Exercise 3 · dijkstra", test,
                     hint="adjacency lists; heap of (dist, node); pop, skip if stale (d > dist[node]); relax "
                          "neighbours and push improved ones. Scanning all nodes for the minimum is O(V^2)")


def _ref_ship(weights, days):
    cap = max(weights)
    while True:
        used, load = 1, 0
        for w in weights:
            if load + w > cap:
                used, load = used + 1, 0
            load += w
        if used <= days:
            return cap
        cap += 1


def ex4(min_capacity):
    def test():
        cases = [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5, 15), ([3, 2, 2, 4, 1, 4], 3, 6), ([1, 2, 3, 1, 1], 4, 3),
                 ([7], 1, 7), ([7], 3, 7), ([5, 5, 5], 1, 15), ([5, 5, 5], 3, 5), ([1, 1, 10], 2, 10)]
        for w, d, want in cases:
            got = min_capacity(w, d)
            assert got == want, f"min_capacity({w}, {d}) = {got}, expected {want}"
        rng = random.Random(3)
        for _ in range(150):
            w = [rng.randint(1, 9) for _ in range(rng.randint(1, 10))]
            d = rng.randint(1, len(w) + 1)
            assert min_capacity(w, d) == _ref_ship(w, d), f"wrong for {w}, days={d}"

        def make(n):
            r = random.Random(n)
            return [r.randint(1, 1000) for _ in range(n)], 50

        check_scaling(min_capacity, make, [1000, 4000, 16000, 64000], label="min_capacity")

    return run_check("Exercise 4 · min_capacity", test,
                     hint="the answer lies in [max(w), sum(w)] and 'feasible(cap)' is monotone: binary search for the "
                          "smallest feasible cap; feasible = greedy day count <= days")


def _ref_lis(nums):
    best = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


def ex5(lis_length):
    def test():
        cases = [([10, 9, 2, 5, 3, 7, 101, 18], 4), ([0, 1, 0, 3, 2, 3], 4), ([7, 7, 7, 7], 1), ([], 0), ([5], 1),
                 ([5, 4, 3, 2, 1], 1), ([1, 2, 3, 4], 4), ([-2, -1, -3, 0], 3), ([1, 3, 6, 7, 9, 4, 10, 5, 6], 6)]
        for nums, want in cases:
            snap = list(nums)
            got = lis_length(nums)
            assert nums == snap, "your function modified its input"
            assert got == want, f"lis_length({nums}) = {got}, expected {want} (strictly increasing)"
        rng = random.Random(4)
        for _ in range(200):
            nums = [rng.randint(0, 8) for _ in range(rng.randint(0, 14))]
            assert lis_length(nums) == _ref_lis(nums), f"wrong for {nums}"

        def make(n):
            r = random.Random(n)
            return ([r.randint(0, 10 * n) for _ in range(n)],)

        check_scaling(lis_length, make, [1000, 4000, 16000, 64000, 200000], label="lis_length")

    return run_check("Exercise 5 · lis_length", test,
                     hint="tails[L] = smallest tail of an increasing subsequence of length L+1; for each x, "
                          "bisect_left(tails, x) and replace or append. len(tails) is the answer")


def _ref_edit(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def ex6(edit_distance):
    def test():
        cases = [("kitten", "sitting", 3), ("", "", 0), ("", "abc", 3), ("abc", "", 3), ("abc", "abc", 0),
                 ("a", "b", 1), ("ab", "ba", 2), ("flaw", "lawn", 2), ("über", "uber", 1)]
        for a, b, want in cases:
            got = edit_distance(a, b)
            assert got == want, f"edit_distance({a!r}, {b!r}) = {got}, expected {want}"
        rng = random.Random(5)
        for _ in range(150):
            a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 6)))
            b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 6)))
            assert edit_distance(a, b) == _ref_edit(a, b), f"wrong for {a!r}, {b!r}"

        def make(n):
            r = random.Random(n)
            return "".join(r.choice("abcdefgh") for _ in range(n)), "".join(r.choice("abcdefgh") for _ in range(n))

        check_scaling(edit_distance, make, [4, 6, 8, 10, 300], label="edit_distance",
                      validate=lambda n, args, res: _check_eq(res, _ref_edit(*args), n))
        assert edit_distance("intention", "execution") == 5

    return run_check("Exercise 6 · edit_distance", test,
                     hint="dp[i][j] = distance between a[:i] and b[:j]; dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, "
                          "dp[i-1][j-1] + (a[i-1] != b[j-1])); plain recursion without memo is exponential")


def ex7(Trie):  # noqa: N803
    def test():
        t = Trie()
        assert not t.search("") and not t.starts_with(""), "an empty trie contains nothing, not even the prefix ''"
        t.insert("apple")
        assert t.search("apple") and not t.search("app"), "search is for whole words"
        assert t.starts_with("app") and t.starts_with("apple") and t.starts_with("")
        assert not t.starts_with("apples") and not t.starts_with("b")
        t.insert("app")
        assert t.search("app") and t.search("apple")
        t.insert("app")
        assert t.search("app"), "inserting twice must be harmless"
        assert not t.search("ap") and not t.search("")
        t.insert("")
        assert t.search(""), "the empty word can be inserted"
        t2 = Trie()
        t2.insert("über")
        assert t2.starts_with("üb") and not t2.search("ub")
        rng = random.Random(6)
        words = {"".join(rng.choice("abc") for _ in range(rng.randint(1, 5))) for _ in range(40)}
        t3 = Trie()
        for w in words:
            t3.insert(w)
        for _ in range(300):
            q = "".join(rng.choice("abc") for _ in range(rng.randint(0, 5)))
            assert t3.search(q) == (q in words), f"search({q!r}) wrong"
            assert t3.starts_with(q) == any(w.startswith(q) for w in words), f"starts_with({q!r}) wrong"

        def run(words, queries):
            tr = Trie()
            for w in words:
                tr.insert(w)
            return sum(tr.search(q) + tr.starts_with(q[:4]) for q in queries)

        def make(n):
            r = random.Random(n)
            ws = ["".join(r.choice("abcdefgh") for _ in range(10)) for _ in range(n)]
            qs = ["".join(r.choice("abcdefgh") for _ in range(10)) for _ in range(n)]
            return ws, qs

        check_scaling(run, make, [1000, 4000, 16000, 64000], label="Trie insert + queries")

    return run_check("Exercise 7 · Trie", test,
                     hint="each node is a dict char -> child plus an end-of-word flag; walk one character at a time")


def ex8(UnionFind):  # noqa: N803
    def test():
        uf = UnionFind(5)
        assert uf.components() == 5
        assert uf.union(0, 1) is True and uf.union(1, 2) is True
        assert uf.union(0, 2) is False, "union of two nodes already connected must return False"
        assert uf.find(0) == uf.find(2) and uf.find(3) != uf.find(0)
        assert uf.components() == 3
        assert uf.union(3, 3) is False and uf.components() == 3, "a self-union changes nothing"
        rng = random.Random(7)
        for _ in range(100):
            n = rng.randint(1, 15)
            uf = UnionFind(n)
            adj = {i: set() for i in range(n)}
            for _ in range(rng.randint(0, 15)):
                a, b = rng.randrange(n), rng.randrange(n)
                uf.union(a, b)
                adj[a].add(b)
                adj[b].add(a)
            seen, comps = set(), 0
            for s in range(n):
                if s in seen:
                    continue
                comps += 1
                stack = [s]
                seen.add(s)
                while stack:
                    u = stack.pop()
                    for v in adj[u] - seen:
                        seen.add(v)
                        stack.append(v)
            assert uf.components() == comps, f"components() = {uf.components()}, expected {comps}"

        def run(n):
            uf = UnionFind(n)
            for i in range(1, n):                    # without path compression / union by size: a long chain
                uf.union(0, i)
            roots = {uf.find(i) for i in range(n)}
            return len(roots), uf.components()

        def validate(n, args, result):
            assert tuple(result) == (1, 1), f"after union(0, i) for every i there is one component (n={n:,})"

        check_scaling(run, lambda n: (n,), [1000, 4000, 16000, 64000, 200000], label="UnionFind", validate=validate)

    return run_check("Exercise 8 · UnionFind", test,
                     hint="parent array; find follows parents to the root and then points every visited node at it "
                          "(path compression); union links roots (smaller under larger) and decrements the count")
