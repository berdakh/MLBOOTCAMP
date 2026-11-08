# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 34 · Coding patterns II
#
# **Day 8 · ~90 minutes · CPU only**
#
# The second half of the pattern set: graphs (BFS, DFS, topological order,
# Dijkstra), binary search on the answer, dynamic programming, tries and
# union-find. These come up in interviews and in ML systems work — dependency
# graphs of pipelines, shortest paths in knowledge graphs, prefix search for
# query autocomplete, deduplicating near-identical items into clusters.
#
# By the end you will be able to:
#
# 1. write iterative BFS/DFS on grids and adjacency lists without hitting Python's recursion limit;
# 2. produce a topological order and detect cycles; run Dijkstra with a heap;
# 3. recognise "binary search on the answer" and write a bug-free binary search with a stated invariant;
# 4. set up a DP (state, transition, base case, order) and improve LIS from O(n²) to O(n log n);
# 5. implement a trie and union-find with path compression;
# 6. follow a 4-week daily practice plan with spaced repetition after the course.
#
# **You will produce:** eight graded solutions and your own practice schedule.
#
# Same rules as notebook 15: read the worked example, collapse it, solve the
# exercise against the clock while talking, then score yourself with the sheet
# from 15.9.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: nothing from earlier notebooks is needed here ----------------
from collections import deque

from shoprank.drills import growth_table
from grading import ch34 as grade

print("Python recursion limit:", sys.getrecursionlimit())

# %% [markdown]
# **Expected:** `Python recursion limit: 1000` (the CPython default, and what
# our verification run printed; some environments set it higher). Either way
# a recursive DFS over a 120 × 120 grid of land (14,400 cells deep) fails,
# which is why the graders include such inputs. Raising the limit with
# `sys.setrecursionlimit` is a patch, not a fix: deep enough recursion can
# still crash the process.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Adjacency list | for each node, the list of its neighbours | `graph[u] = [v, ...]` |
# | BFS | breadth-first search: visit nodes in layers of distance, using a queue | `shortest_steps` on a maze |
# | DFS | depth-first search: follow one path as deep as it goes, then back up, using a stack | `has_cycle` with three colours |
# | Topological order | an order in which every node comes after all its prerequisites | `course_order` (Exercise 2) |
# | In-degree | the number of edges pointing into a node | Kahn's algorithm starts from in-degree 0 |
# | Dijkstra | shortest paths with non-negative weights, always expanding the cheapest frontier node | Exercise 3 |
# | Lazy deletion | leave outdated heap entries in place and skip them when popped | `heapq` has no decrease-key |
# | Monotone predicate | a yes/no test that flips from False to True once as its input grows | "capacity c ships everything in D days" |
# | Dynamic programming | solve each subproblem once, store it, build bigger answers from stored ones | `min_coins`, edit distance |
# | Trie | a tree of characters in which words with a shared prefix share nodes | Exercise 7; autocomplete |
# | Union-find | sets that can be merged, with a fast "which set is x in?" | Exercise 8 |
# | Path compression | pointing every node a `find` passes straight at the root | the 16.6 timing demo |

# %% [markdown]
# ## 16.1 · Graphs: representations and BFS
#
# Interview graphs come as an **adjacency list** (`graph[u] = [v, ...]`), an
# **edge list** you convert into one (O(V + E)), or **implicitly** as a grid
# where neighbours are the 4 adjacent cells. BFS explores in layers, so the
# first time it reaches a node is along a path with the fewest edges.
# Template: a `deque`, a `seen` set marked **when a node is enqueued** (not when
# it is popped — otherwise a node can be enqueued many times).
#
# **Worked example:** fewest steps from the top-left to the bottom-right of a
# grid with walls (`1` = wall).

# %%
def shortest_steps(grid):
    rows, cols = len(grid), len(grid[0])
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    queue = deque([(0, 0, 0)])              # row, col, distance
    seen = {(0, 0)}
    while queue:
        r, c, d = queue.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return d
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))          # mark on enqueue
                queue.append((nr, nc, d + 1))
    return -1


maze = [[0, 0, 0, 0],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [0, 1, 1, 0]]
print(shortest_steps(maze), shortest_steps([[0, 1], [1, 0]]))
# O(rows * cols) time and space: every cell is enqueued at most once.

# %% [markdown]
# **Expected:** `6 -1`. DFS (a stack instead of a queue) also visits every
# reachable cell, but the first path it finds is not the shortest.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-bfs-vs-dfs.svg" alt="The same tree traversed by BFS in layers and by DFS deep-first, with visit order numbers" width="100%">
#
# *What to notice:* BFS's numbers grow layer by layer, so the first time it reaches a node is along a fewest-edges path; DFS finishes a whole branch before starting the next.
#
# > 💡 **Intuition.** BFS is a ripple in a pond: everything one step away is reached before
# > anything two steps away. DFS is a maze-walker with a hand on the wall: deep first, back
# > up only at dead ends. *Where it breaks down:* ripples measure *steps*, not distance; with
# > unequal edge lengths the first ripple to arrive may have taken the long way (16.3).

# %% [markdown]
# > **Exercise 1.** *(target 12 min)* **Number of islands.** Implement
# > `num_islands(grid)` for a list of lists of `0`/`1` ints → the number of
# > 4-connected groups of `1`s (diagonals do not connect). Do not modify `grid`.
# > O(rows × cols). Visible examples: the 4 × 5 grid
# > `[[1,1,0,0,0],[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]]` → 3; `[]` → 0; `[[1,0],[0,1]]` → 2.
# > Hidden: a 120 × 120 all-land grid, a long snake-shaped island, 100 random
# > grids, and timing up to 500 × 500.

# %%
def num_islands(grid):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(num_islands)

# %% [markdown]
# > 🗣️ **Talk it through.** *Clarify:* 4- or 8-connectivity? May I modify the grid
# > (sinking visited land is a common trick, but it destroys the caller's data —
# > ask)? Empty grid, ragged rows? *Plan:* scan cells; on unvisited land, count one
# > island and flood-fill it with BFS/iterative DFS. *Complexity:* O(R·C) time,
# > O(R·C) space for `seen` (O(min(R, C)) for the BFS frontier in practice).
# > *Python-specific point worth saying:* a recursive flood fill is fine in Java or
# > C++ for moderate grids but hits Python's recursion limit — use an explicit
# > stack. *Follow-ups:* union-find version (Exercise 8); islands arriving online
# > one cell at a time (union-find again).

# %% [markdown]
# ## 16.2 · DFS, cycles and topological order
#
# A **topological order** of a directed graph lists every node after all of its
# prerequisites; it exists if and only if the graph has no cycle. Two classic
# algorithms: **Kahn's** (repeatedly take a node with no remaining incoming
# edges; if some nodes never reach in-degree 0, there is a cycle) and **DFS
# post-order** (reverse of finishing order; a back edge to a node still "in
# progress" is a cycle). **Worked example:** cycle detection with three colours,
# written iteratively.

# %%
def has_cycle(n, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
    WHITE, GREY, BLACK = 0, 1, 2            # unvisited, on the current path, finished
    colour = [WHITE] * n
    for start in range(n):
        if colour[start] != WHITE:
            continue
        stack = [(start, iter(graph[start]))]
        colour[start] = GREY
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:               # all children done: finish the node
                colour[node] = BLACK
                stack.pop()
            elif colour[child] == GREY:     # back edge: a cycle
                return True
            elif colour[child] == WHITE:
                colour[child] = GREY
                stack.append((child, iter(graph[child])))
    return False


print(has_cycle(3, [(0, 1), (1, 2)]), has_cycle(3, [(0, 1), (1, 2), (2, 0)]), has_cycle(1, [(0, 0)]))
# O(V + E) time and space.

# %% [markdown]
# **Expected:** `False True True`. Keeping an iterator per stack frame is how
# you turn any recursive DFS into an iterative one without changing the visit
# order.

# %% [markdown]
# > **Exercise 2.** *(target 15 min)* **Course schedule.** Implement
# > `course_order(n, prerequisites)` → a list containing every course `0..n-1`
# > once, in an order where for each pair `[course, prereq]` the prerequisite
# > comes first; return `[]` if no such order exists (a cycle). Any valid order
# > is accepted. O(V + E). Visible examples: `2, [[1, 0]]` → `[0, 1]`;
# > `2, [[1, 0], [0, 1]]` → `[]`; `3, []` → any order of 0, 1, 2.
# > Hidden: self-loops, duplicate edges, `n = 0`, 200 random graphs, and a
# > 64,000-course chain.

# %%
def course_order(n, prerequisites):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(course_order)

# %% [markdown]
# > 🗣️ **Talk it through.** *Clarify the edge direction out loud* — `[a, b]` means
# > "b before a" here; getting it backwards gives a reversed order. Duplicates?
# > Self-loops (a cycle)? *Plan:* Kahn's algorithm: in-degrees, queue of zeros,
# > pop/append/decrement; `len(order) < n` ⇒ cycle. *Complexity:* O(V + E) time
# > and space. *ML link:* a pipeline orchestrator (Airflow-style DAG) runs tasks in
# > exactly this order; the cycle check is its "your DAG is not a DAG" error.

# %% [markdown]
# ## 16.3 · Weighted shortest paths: why BFS is not enough
#
# BFS minimises the number of **edges**. With weights, the path with fewer
# edges can cost more. Dijkstra generalises BFS by always expanding the
# cheapest frontier node (a heap instead of a queue); it is correct when every
# weight is ≥ 0. With negative edges, use Bellman-Ford (O(V·E)).

# %%
edges_demo = [(0, 1, 10), (0, 2, 1), (2, 3, 1), (3, 1, 1)]     # (u, v, weight)
adj = {u: [] for u in range(4)}
for u, v, w in edges_demo:
    adj[u].append(v)
hops = {0: 0}
q = deque([0])
while q:
    u = q.popleft()
    for v in adj[u]:
        if v not in hops:
            hops[v] = hops[u] + 1
            q.append(v)
print("BFS: node 1 reached in", hops[1], "hop(s), via the edge of weight 10")
print("cheapest path 0 -> 2 -> 3 -> 1 costs", 1 + 1 + 1)

# %% [markdown]
# **Expected:** BFS reaches node 1 in 1 hop (cost 10), while the 3-hop path
# costs 3. Heap entries are `(distance, node)` tuples; when a shorter distance
# to a node is found, push a new entry and skip stale ones on pop ("lazy
# deletion"), since `heapq` has no decrease-key.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-dijkstra.svg" alt="A four-node graph where BFS takes a one-hop edge of weight 10 while Dijkstra finds a three-hop path of cost 3, with the heap trace" width="100%">
#
# *What to notice:* the (10, 1) entry is pushed first but popped last, as a stale entry; correctness comes from popping in order of distance, which needs weights ≥ 0.

# %% [markdown]
# > **Exercise 3.** *(target 15 min)* **Dijkstra.** Implement
# > `dijkstra(n, edges, source)` for a **directed** graph with non-negative
# > weights given as `(u, v, w)` triples → a list of shortest distances from
# > `source` (`float('inf')` for unreachable nodes). O((V + E) log V).
# > Visible example: `dijkstra(4, [(0,1,4), (0,2,1), (2,1,2), (1,3,1), (2,3,5)], 0)`
# > → `[0, 3, 1, 4]`. Hidden: parallel edges, self-loops, zero weights, a source
# > other than 0, 100 random graphs against Bellman-Ford, timing up to
# > 64,000 nodes and ~320,000 edges.

# %%
def dijkstra(n, edges, source):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(dijkstra)

# %% [markdown]
# > 🗣️ **Talk it through.** *Clarify:* directed? negative weights (then not
# > Dijkstra)? one target or all nodes? return the path too? *Plan:* adjacency
# > lists; `dist = [inf] * n`; heap `[(0, source)]`; pop, skip stale, relax.
# > *Complexity:* O((V + E) log V) with a binary heap; O(V²) with an array scan,
# > which is better only for dense graphs (E ≈ V²). *Follow-ups:* the path itself
# > (store `parent[v]`); one target (stop when it is popped); A* with an
# > admissible heuristic; many queries on a road-size graph (precomputation such as
# > contraction hierarchies — just name it).

# %% [markdown]
# ## 16.4 · Binary search, and binary search on the answer
#
# Binary search needs a **monotone predicate**, not necessarily a sorted array:
# `False, False, …, False, True, …, True`; find the first `True`. Write it with an
# explicit invariant and the off-by-one bugs disappear:

# %%
def first_true(lo, hi, pred):
    """Smallest x in [lo, hi] with pred(x) True, assuming pred is monotone and pred(hi) is True."""
    while lo < hi:                    # invariant: the answer is in [lo, hi]
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid                  # mid could be the answer: keep it
        else:
            lo = mid + 1              # mid is not: discard it
    return lo


def isqrt(n):                         # largest r with r*r <= n  ==  first r with (r+1)^2 > n
    return first_true(0, n, lambda r: (r + 1) * (r + 1) > n)


print([isqrt(n) for n in [0, 1, 8, 9, 10, 10**12]])
# O(log n) predicate calls. The loop always terminates: [lo, hi] shrinks every step.

# %% [markdown]
# **Expected:** `[0, 1, 2, 3, 3, 1000000]`. "Binary search on the answer"
# means: the answer is a number in a known range, and *checking* a candidate is
# easier than *finding* the answer. Typical signals: "minimum capacity / speed /
# maximum distance such that …".
#
# > 💡 **Intuition.** Binary search on the answer is the "guess my number" game where the
# > only reply is "enough" or "not enough": each guess halves the range, so even a range of
# > a billion takes 30 guesses. *Where it breaks down:* the reply must be monotone; if
# > capacity 15 works but 16 does not, halving can jump past the answer.

# %% [markdown]
# > **Exercise 4.** *(target 15 min)* **Ship within D days.** Packages with
# > `weights` must be shipped **in the given order**; each day the ship loads
# > consecutive packages without exceeding its capacity. Implement
# > `min_capacity(weights, days)` → the smallest integer capacity that ships
# > everything within `days` days. Target O(n log(sum(w))). Visible examples:
# > `[1,2,3,4,5,6,7,8,9,10], 5` → 15; `[3,2,2,4,1,4], 3` → 6; `[1,2,3,1,1], 4` → 3.
# > Hidden: one package, `days` ≥ number of packages, `days = 1`, 150 random
# > cases, timing up to 64,000 packages.

# %%
def min_capacity(weights, days):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex4(min_capacity)

# %% [markdown]
# > 🗣️ **Talk it through.** *Clarify:* order fixed? integer weights? `days` ≥ 1?
# > *Observations to say aloud:* capacity below `max(w)` can never work;
# > `sum(w)` always works; if capacity c works, c + 1 works (monotone) → binary
# > search the first feasible c in `[max(w), sum(w)]`. *Feasibility:* greedy
# > day counting is optimal for a fixed capacity (filling each day as much as
# > possible never hurts). *Complexity:* O(n · log(sum w)) time, O(1) space.
# > *Naive:* trying every capacity upwards is O(n · (sum − max)).

# %% [markdown]
# ## 16.5 · Dynamic programming: state, transition, base, order
#
# DP applies when a problem has **optimal substructure** (the best answer is
# built from best answers to subproblems) and **overlapping subproblems** (the
# same subproblem recurs, so caching pays). Write four things before coding:
#
# 1. **State:** what `dp[i]` (or `dp[i][j]`) means, in one sentence;
# 2. **Transition:** how it is computed from smaller states;
# 3. **Base cases;**
# 4. **Order:** so that every state's dependencies are ready (or memoised recursion).
#
# **Worked example:** fewest coins to make an amount (unbounded coins).

# %%
def min_coins(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount              # dp[a] = fewest coins summing to a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1      # last coin is c
    return dp[amount] if dp[amount] < INF else -1


print(min_coins([1, 2, 5], 11), min_coins([2], 3), min_coins([1], 0))
# O(amount * len(coins)) time, O(amount) space. Greedy (largest coin first) fails
# for coins like [1, 3, 4] and amount 6: greedy gives 4+1+1, DP gives 3+3.
print("greedy trap:", min_coins([1, 3, 4], 6))

# %% [markdown]
# **Expected:** `3 -1 0` and `greedy trap: 2`. The same template solves
# knapsack, path counting and sequence alignment; the art is in step 1.
#
# > 💡 **Intuition.** DP is doing a problem with sticky notes: every time you solve a
# > subproblem, write the answer down, and never solve it twice. The state definition is
# > what you write on the note. *Where it breaks down:* it pays only if subproblems repeat;
# > when they do not (plain divide-and-conquer, e.g. merge sort), the notes are wasted memory.

# %% [markdown]
# > **Exercise 5.** *(target 15 min)* **Longest increasing subsequence.**
# > Implement `lis_length(nums)` → the length of the longest **strictly**
# > increasing subsequence (not necessarily contiguous) in **O(n log n)**. Visible
# > examples: `[10, 9, 2, 5, 3, 7, 101, 18]` → 4; `[0, 1, 0, 3, 2, 3]` → 4;
# > `[7, 7, 7, 7]` → 1; `[]` → 0. Hidden: 200 random lists against the O(n²) DP,
# > timing up to n = 200,000 (the O(n²) DP cannot fit).

# %%
def lis_length(nums):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex5(lis_length)

# %% [markdown]
# > 🗣️ **Talk it through.** *First the O(n²) DP* — state: `dp[i]` = length of the
# > longest increasing subsequence ending at i; transition: `1 + max(dp[j])` over
# > `j < i` with `nums[j] < nums[i]`. *Then improve:* keep `tails[L]` = the
# > smallest possible tail of an increasing subsequence of length L + 1; `tails` is
# > sorted, so each new x replaces the first tail ≥ x (`bisect_left`) or extends
# > the list. *Say clearly:* `tails` is **not** an actual subsequence; its length is
# > the answer. *Strict vs non-decreasing:* `bisect_left` vs `bisect_right`.
# > *Complexity:* O(n log n) time, O(n) space. Reconstructing the sequence needs
# > parent pointers.

# %% [markdown]
# > **Exercise 6.** *(target 18 min)* **Edit distance.** Implement
# > `edit_distance(a, b)` → the minimum number of single-character insertions,
# > deletions and substitutions turning `a` into `b` (Levenshtein). O(len(a) ·
# > len(b)) time. Visible examples: `"kitten", "sitting"` → 3; `"", "abc"` → 3;
# > `"intention", "execution"` → 5. Hidden: empty strings, non-ASCII, 150 random
# > pairs, and strings of length 300 (plain recursion without memoisation is
# > exponential and is stopped early by the timing check).

# %%
def edit_distance(a, b):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex6(edit_distance)

# %% [markdown]
# > 🗣️ **Talk it through.** *State:* `dp[i][j]` = distance between `a[:i]` and
# > `b[:j]`. *Transition:* `min(dp[i-1][j] + 1` (delete), `dp[i][j-1] + 1` (insert),
# > `dp[i-1][j-1] + (a[i-1] != b[j-1]))` (substitute or match). *Base:*
# > `dp[i][0] = i`, `dp[0][j] = j`. *Order:* row by row. *Complexity:* O(m·n) time;
# > O(min(m, n)) space keeping two rows. *ML links:* fuzzy matching of
# > queries/product titles, WER for speech recognition (edit distance over
# > words), sequence alignment.

# %% [markdown]
# ## 16.6 · Tries and union-find
#
# A **trie** stores strings by shared prefixes: each node is a dict from
# character to child plus an end-of-word marker. Lookup and prefix queries
# cost O(length of the query), independent of how many words are stored —
# the structure behind query autocomplete (with the top suggestions cached at
# each node) and prefix filters.
#
# **Union-find** (disjoint-set union) maintains a partition of `0..n-1` under
# `union(a, b)` and answers `find(x)` (the representative of x's set).
# **Path compression** (point every node visited by `find` at the root) plus
# **union by size/rank** makes each operation effectively O(1) amortised
# (O(α(n)), inverse Ackermann). Use it when edges arrive over time and you only
# need connectivity: deduplicating near-identical items into clusters, merging
# user identities across devices, Kruskal's minimum spanning tree.
#
# How much does path compression matter? Here is a find without it on the
# worst kind of input:

# %%
def chain_cost(n, compress):
    parent = list(range(n))

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        if compress:
            while parent[x] != root:
                parent[x], x = root, parent[x]
        return root

    for i in range(1, n):
        parent[find(0)] = i          # naive link: the old root goes under node i -> a growing chain
    return sum(find(i) for i in range(n))


print("without compression:")
_ = growth_table(lambda n: chain_cost(n, False), lambda n: (n,), [1000, 2000, 4000])
print("with compression:")
_ = growth_table(lambda n: chain_cost(n, True), lambda n: (n,), [1000, 2000, 4000, 100_000])

# %% [markdown]
# **Expected:** without compression the fitted exponent is about 2 (2.0-2.3 in
# our runs; each `find(0)` walks the whole chain) and 4,000 nodes take a few hundred
# milliseconds (~410 ms); with compression the exponent is about 1 and 100,000
# nodes take tens of milliseconds (~34 ms).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-union-find.svg" alt="A chain of parent pointers four steps long, flattened by path compression so every node points directly at the root" width="100%">
#
# *What to notice:* compression costs one extra pass during a `find` and pays for itself on every later `find` of any node on that path.

# %% [markdown]
# > **Exercise 7.** *(target 12 min)* **Trie.** Implement a class `Trie` with
# > `insert(word)`, `search(word)` → `True` if exactly `word` was inserted, and
# > `starts_with(prefix)` → `True` if some inserted word starts with `prefix`
# > (so `starts_with("")` is `True` only for a non-empty trie). The empty word may
# > be inserted. Each operation O(length of its argument). Hidden: duplicates,
# > non-ASCII, 300 random queries against a set, timing with up to 64,000 words.

# %%
class Trie:
    def __init__(self):
        pass  # YOUR CODE HERE

    def insert(self, word):
        raise NotImplementedError

    def search(self, word):
        raise NotImplementedError

    def starts_with(self, prefix):
        raise NotImplementedError


grade.ex7(Trie)

# %% [markdown]
# > 🗣️ **Talk it through.** *Clarify:* character set (26 lowercase letters → an
# > array of 26 children; Unicode → a dict), deletions needed? *Plan:* nested
# > dicts with an end marker; `search` and `starts_with` share a `_walk(prefix)`
# > helper. *Complexity:* O(L) per operation; space O(total characters) in the
# > worst case. *Follow-ups:* top-k completions per prefix (store counts, keep a
# > small heap at each node or precompute suggestions offline), compressed tries
# > (radix trees), memory vs a sorted list + `bisect` for prefix ranges.

# %% [markdown]
# > **Exercise 8.** *(target 12 min)* **Union-find.** Implement a class
# > `UnionFind(n)` with `find(x)` (with path compression), `union(a, b)` →
# > `True` if two different sets were merged (`False` if already connected), and
# > `components()` → the current number of sets. Hidden: self-unions, 100 random
# > graphs against BFS component counts, timing for 200,000 unions of the form
# > `union(0, i)` followed by a `find` of every node.

# %%
class UnionFind:
    def __init__(self, n):
        pass  # YOUR CODE HERE

    def find(self, x):
        raise NotImplementedError

    def union(self, a, b):
        raise NotImplementedError

    def components(self):
        raise NotImplementedError


grade.ex8(UnionFind)

# %% [markdown]
# > 🗣️ **Talk it through.** *Plan:* `parent` array, `size` array, a component
# > counter. `find`: walk to the root, then point every visited node at the root —
# > iteratively (a recursive `find` on a 200,000-long chain hits the recursion
# > limit). `union`: find both roots; if equal return False; attach the smaller
# > tree under the larger; decrement the count. *Complexity:* O(α(n)) amortised
# > per operation with both optimisations; O(log n) with either one alone.
# > *Compare with BFS:* BFS counts components of a fixed graph in O(V + E);
# > union-find wins when edges arrive incrementally or you need to answer
# > "connected?" between insertions.

# %% [markdown]
# ## 16.7 · After this notebook: a 4-week practice plan
#
# Two days teach the patterns; fluency comes from short, daily, *spaced*
# practice. The plan below assumes 45-60 minutes a day, 5-6 days a week, and
# reuses notebooks 13-16: rerun an exercise from a **blank cell** (delete your
# solution, keep the grader), and add new problems of the same pattern from any
# practice platform.
#
# | Week | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 | Day 6 (review) |
# |---|---|---|---|---|---|---|
# | 1 | hash maps, prefix sums (15: Ex1, Ex7) | sliding window, two pointers (15: Ex2) | stacks (15: Ex5 + 2 new) | heaps (15: Ex3 + 2 new) | intervals (15: Ex4 + 1 new) | redo the two lowest-scoring of the week, timed |
# | 2 | BFS/DFS grids (16: Ex1) | topological sort (16: Ex2) | Dijkstra (16: Ex3) | binary search on the answer (16: Ex4) | ML coding: logistic regression + AUC (13: Ex1, Ex5) | mock: two random problems from weeks 1-2, 45 min each, out loud |
# | 3 | DP 1-D (16: Ex5 + 2 new) | DP 2-D (16: Ex6 + 1 new) | tries, union-find (16: Ex7, Ex8) | ML coding: k-means, kNN (13: Ex3, Ex4) | ML coding: MLP backprop, attention (14: Ex3, Ex4) | redo week-1 problems scoring < 12 |
# | 4 | mixed: 2 unseen medium problems | ML coding: GBM + best split (14: Ex1, Ex2) | mixed: 1 hard problem, 60 min | mock loop: 1 coding + 1 ML coding, back to back | redo everything scoring < 12 | full mock loop with a friend (notebook 20's format) |
#
# **Spaced repetition.** Re-attempt every problem you solved on days +1, +3,
# +7 and +21 after first solving it, from a blank cell and against the target
# time. Drop it from the rotation after two consecutive scores of 12+ on the
# 15.9 sheet; reset the schedule after any score below 8. A plain spreadsheet
# with columns *problem, pattern, date, score, next review* is enough.
#
# **Using practice platforms honestly.** LeetCode-style platforms are useful
# for volume and for an automatic judge. Use them well:
#
# * pick problems **by pattern**, not by a list claiming to be "what company X
#   asks": nobody outside a company knows its current question bank, interview
#   questions rotate, and memorised answers are easy for an interviewer to
#   detect with one follow-up question;
# * set a timer and work 20-25 minutes before looking at hints or discussions;
#   after reading a solution, close it and re-implement it from memory the next
#   day — reading is not practice;
# * always say the complexity and edge cases out loud, even alone; the judge
#   only checks the output, the interviewer checks the process;
# * do not treat the platform's runtime percentile as a quality score: it is
#   noisy and rewards micro-optimisations interviewers do not care about;
# * mix easy and medium problems; hard problems are useful in week 4 for
#   stamina, but most interview problems are mediums.
#
# The [study plan](../../appendices/study-plan.md) has the longer 4-8 week
# schedule for the whole course.

# %%
from datetime import date, timedelta

start = date.today()
plan = [(pattern, [start + timedelta(days=d) for d in (0, 1, 3, 7, 21)])
        for pattern in ["sliding window", "topological sort", "LIS"]]
for pattern, days in plan:
    print(f"{pattern:<18}", "  ".join(d.isoformat() for d in days))

# %% [markdown]
# **Expected:** three lines, each with five dates — today, then +1, +3, +7 and
# +21 days. Replace the patterns with the problems you solved today and copy the
# dates into your calendar or spreadsheet.

# %% [markdown]
# ## 🧮 Key equations
#
# **1. Graph traversal cost**
#
# $$
# T_{\text{BFS}} = T_{\text{DFS}} = O(V + E)
# $$
#
# *In words:* each node is enqueued (or pushed) once and each edge is looked at once (twice if undirected). A grid of R × C cells has V = RC and E ≤ 4RC.
# *Example:* a 500 × 500 grid: 250,000 cells and about 1 million edge checks, a fraction of a second in Python.
#
# **2. Dijkstra's relaxation**
#
# $$
# d(v) \leftarrow \min\big(d(v),\ d(u) + w(u, v)\big), \qquad T = O\big((V + E) \log V\big) \text{ with a binary heap}
# $$
#
# *In words:* when you pop the closest unfinished node u, try every edge out of it; correct because, with $w \ge 0$, nothing popped later can be closer.
# *Example:* in 16.3, popping node 3 at distance 2 relaxes $d(1) = \min(10,\ 2 + 1) = 3$.
#
# **3. Binary search iterations**
#
# $$
# \text{iterations} = \lceil \log_2(\text{hi} - \text{lo} + 1) \rceil
# $$
#
# *Example:* Exercise 4's first visible case: capacities from max(w) = 10 to sum(w) = 55 are 46 candidates, so 6 feasibility checks of O(n) each.
#
# **4. The DP recurrences in this notebook**
#
# $$
# \text{coins}(a) = 1 + \min_{c \le a} \text{coins}(a - c), \qquad
# D(i, j) = \min\big(D(i-1, j) + 1,\ D(i, j-1) + 1,\ D(i-1, j-1) + \mathbb{1}[a_i \ne b_j]\big)
# $$
#
# *In words:* the best answer for a state is built from the best answers of smaller states; the base cases are $\text{coins}(0) = 0$, $D(i, 0) = i$, $D(0, j) = j$.
# *Example:* coins $\{1, 3, 4\}$, amount 6: $1 + \min(\text{coins}(5), \text{coins}(3), \text{coins}(2)) = 1 + \min(2, 1, 2) = 2$ (3 + 3), where greedy gives 3 (4 + 1 + 1). Edit distance "kitten" → "sitting" = 3.
#
# **5. Union-find**
#
# $$
# T(\text{m operations on n elements}) = O\big(m\, \alpha(n)\big), \qquad \alpha(n) \le 4 \text{ for any } n \text{ you will meet}
# $$
#
# *In words:* with path compression and union by size each operation is effectively constant time; with neither, a chain makes `find` O(n).

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `RecursionError: maximum recursion depth exceeded` | recursive DFS/flood fill/`find` on a deep input | explicit stack or queue; iterative path compression |
# | Exercise 1: "modified the grid" | sinking land in place | keep a `seen` set/array, or copy the grid first |
# | BFS is slow on big grids | `queue.pop(0)` on a list, or marking `seen` on pop | `collections.deque`; mark when enqueuing |
# | Exercise 2: valid-looking order rejected | edge direction reversed (`[course, prereq]`) | the prerequisite must come first: build `prereq → course` |
# | Exercise 2: misses a cycle | only checking for self-loops, or recursion without a "grey" state | Kahn's: cycle ⇔ fewer than n nodes output |
# | Exercise 3: wrong distances with parallel edges | adjacency dict `{v: w}` overwrites earlier edges | adjacency **lists** of `(v, w)` |
# | Exercise 3: slow at 16,000 nodes | linear scan for the minimum, or `heap.remove` | heap + skip stale entries |
# | Exercise 4: off by one / infinite loop | `hi = mid - 1` with a `lo < hi` loop, or `lo = mid` | keep the invariant: answer in `[lo, hi]`; `hi = mid` / `lo = mid + 1` |
# | Exercise 5: answer too large on duplicates | `bisect_right` (non-decreasing) | `bisect_left` for strictly increasing |
# | Exercise 6 hangs or is stopped early | recursion without memoisation | bottom-up table, or `functools.lru_cache` |
# | Exercise 8 slow | no path compression and no union by size | compress in `find`; attach smaller under larger |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. When does BFS give shortest paths, when do you need Dijkstra, and when does Dijkstra give wrong answers? What do you use then?
# 2. You are given the task dependencies of a training pipeline. How do you find an execution order, detect a circular dependency, and find which tasks can run in parallel?
# 3. What makes a problem suitable for "binary search on the answer"? Give an ML-systems example (hint: capacity or thresholds).
# 4. Explain the O(n log n) LIS algorithm and why `tails` stays sorted. What is the difference between the `tails` array and an actual longest subsequence?
# 5. Trie vs hash set vs sorted array with binary search for query autocomplete over 10 million queries: compare memory, prefix-query cost and update cost.
# 6. You must cluster 50 million product listings into groups of near-duplicates, given a stream of candidate pairs from a similarity model. Which data structure, and what are the risks of transitive merging?
#
# ➡️ **Next:** [`17_serving_a_ranking_cascade.ipynb`](../ch35-serving-a-ranking-cascade/notebook.ipynb) — Day 9: put retrieval, features and a ranker behind one service, with per-stage latency budgets.
