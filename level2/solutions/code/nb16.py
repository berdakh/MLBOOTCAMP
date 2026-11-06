"""Reference solutions for notebook 16 · Coding patterns II. Graded by tests/test_solutions.py."""

from __future__ import annotations

import bisect
import heapq
from collections import deque


def ex1(grid):
    """Number of 4-connected islands of 1s. Iterative BFS, O(rows * cols); input not modified."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and not seen[r][c]:
                islands += 1
                seen[r][c] = True
                queue = deque([(r, c)])
                while queue:
                    i, j = queue.popleft()
                    for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == 1 and not seen[ni][nj]:
                            seen[ni][nj] = True
                            queue.append((ni, nj))
    return islands


def ex2(n, prerequisites):
    """Kahn's algorithm. prerequisites: pairs (course, prereq) = prereq before course.
    Returns an order of all n courses, or [] if there is a cycle. O(V + E)."""
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    for course, pre in prerequisites:
        graph[pre].append(course)
        indegree[course] += 1
    queue = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return order if len(order) == n else []


def ex3(n, edges, source):
    """Dijkstra with a binary heap and lazy deletion. edges: (u, v, w), directed, w >= 0.
    Returns a list of distances; float('inf') for unreachable nodes. O((V + E) log V)."""
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
    dist = [float("inf")] * n
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                       # stale entry
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def ex4(weights, days):
    """Least ship capacity to send packages in order within `days` days.
    Binary search on the answer in [max(w), sum(w)]; feasibility is a greedy O(n) scan."""
    def days_needed(cap):
        used, load = 1, 0
        for w in weights:
            if load + w > cap:
                used += 1
                load = 0
            load += w
        return used

    lo, hi = max(weights), sum(weights)
    while lo < hi:
        mid = (lo + hi) // 2
        if days_needed(mid) <= days:
            hi = mid
        else:
            lo = mid + 1
    return lo


def ex5(nums):
    """Length of the longest strictly increasing subsequence. Patience sorting, O(n log n).
    tails[i] = smallest possible tail of an increasing subsequence of length i + 1."""
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)     # bisect_left: strictly increasing
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def ex6(a, b):
    """Levenshtein distance (insert, delete, substitute, each cost 1). O(len(a) * len(b)) time,
    O(min) memory with two rows."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,            # delete ca
                         cur[j - 1] + 1,         # insert cb
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


class ex7:  # noqa: N801  (graded as a class: Trie)
    """Prefix tree. insert/search/starts_with are O(length of the word)."""

    _END = object()

    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node[self._END] = True

    def _walk(self, s):
        node = self.root
        for ch in s:
            node = node.get(ch)
            if node is None:
                return None
        return node

    def search(self, word):
        node = self._walk(word)
        return node is not None and self._END in node

    def starts_with(self, prefix):
        node = self._walk(prefix)
        return node is not None and bool(node)   # some inserted word continues (or ends) here


class ex8:  # noqa: N801  (graded as a class: UnionFind)
    """Disjoint sets with path compression and union by size. Near-O(1) amortised per operation."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self._components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:         # path compression (iterative: no recursion limit)
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self._components -= 1
        return True

    def components(self):
        return self._components
