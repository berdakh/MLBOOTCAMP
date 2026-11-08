# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 33 · Coding patterns I
#
# **Part II · Chapter 33 · about 90 minutes · CPU only**
#
# Most interview processes for ML engineers at large technology companies
# include one or two classic coding rounds (data structures and algorithms),
# assessed much like a software engineer's. Hundreds of special techniques are
# not needed: most interview problems are variations on about a dozen patterns,
# and the round rewards a **visible, structured process** at least as much as
# the final code.
#
# After this lab you should be able to:
#
# 1. run a seven-step communication protocol from "clarify" to "complexity" without prompting;
# 2. state the cost of common Python operations, and measure it when unsure;
# 3. recognise and apply six patterns: hash maps, sliding windows, heaps, interval sweeps, stacks, prefix sums/products;
# 4. solve seven graded problems within 10-20 minutes each, with edge cases and complexity stated;
# 5. score your own attempts consistently, so that later practice is deliberate.
#
# **Outputs:** seven graded solutions and a filled-in self-scoring sheet.
#
# **How to use this lab.** For each pattern, read the worked example,
# then collapse the cell and do the exercise against the clock,
# talking aloud; recording yourself is useful. The graders check hidden
# edge cases and run a **timing check**: an O(n) or O(n log n) solution uses a
# few percent of a 2-second budget, an O(n²) one cannot fit.

# %% [markdown]
# Read first: [Chapter 33 text](README.md), which explains the concepts this lab measures. Worked answers are in [solutions.md](solutions.md).

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
import heapq
from collections import defaultdict, deque

from shoprank.drills import growth_table
from grading import ch33 as grade

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Brute force | the obvious correct solution, usually too slow, stated first as a baseline | checking every pair: O(n²) |
# | Time / space complexity | how running time and extra memory grow with the input size | O(n) time and O(n) extra space for a dictionary |
# | Amortised O(1) | cheap on average over many operations, even if one is occasionally slow | `list.append` when the list resizes |
# | Invariant | a statement that stays true at every step of a loop | "`s[left:right+1]` has no repeated character" |
# | Hash map | a dictionary: average O(1) lookup, insert and delete by key | complements in two-sum |
# | Two pointers / sliding window | two indices that only move forward over a sequence | `min_subarray_len` in 33.4 |
# | Heap | a tree kept partly ordered so the smallest item is always on top | `heapq`; top-k in O(n log k) |
# | Sweep | sort events by position, then process them in one pass | merging intervals; meeting rooms |
# | Monotonic stack | a stack kept in sorted order to answer "next greater / most recent smaller" questions | 33.7 |
# | Prefix sum | a running total, so any range sum is one subtraction | `sum(nums[i:j]) = P[j] − P[i]` |
# | Output-sensitive bound | a cost that depends on the size of the answer, not only the input | two-sum pairs: O(n + p) |

# %% [markdown]
# ## 33.1 · The communication protocol
#
# Interviewers usually assess four things: **problem solving** (was a good
# approach found, and how much help was needed), **coding** (correct, clean,
# idiomatic), **verification** (did the candidate test the code and find their
# own bugs), and **communication** (could the interviewer follow the reasoning).
# A silent correct solution can score lower than a narrated one with a small
# bug that the candidate then finds. Use the same seven steps every time, until
# they are automatic:
#
# | Step | ~Minutes (of 45) | What you do and say |
# |---|---|---|
# | 1. **Clarify** | 2-4 | Restate the problem. Ask: input size? types and ranges (negatives? duplicates? empty)? return format? in-place allowed? Write the answers as a comment. |
# | 2. **Examples** | 2-3 | Work one normal and one edge example by hand. This often reveals the pattern. |
# | 3. **Brute force** | 1-2 | State the obvious solution and its cost: "Checking every pair is O(n²)." It need not be coded. |
# | 4. **Optimise** | 3-8 | Name the bottleneck and the pattern that removes it: "The inner loop searches for a complement → a hash map makes that O(1)." Agree the plan before coding. |
# | 5. **Code** | 10-15 | Clear names, small helpers, narrate the invariants ("left is the first index of a window with no repeats"). |
# | 6. **Test** | 4-6 | Trace the code on the example **by hand**, line by line, then the edge cases from step 1. Fix bugs calmly, aloud. |
# | 7. **Complexity** | 1-2 | Time and extra space, and where they come from. Mention what would change for larger inputs (streaming, memory). |
#
# Phrases that help: *"Before I code, can I check the constraints?"*, *"Let me
# start with a brute force so we have a baseline"*, *"The invariant here is…"*,
# *"I'll trace through [2, 7, 11] to check"*, *"This is O(n) time and O(n)
# extra space for the dictionary."* When stuck, say what is being
# considered; an interviewer can only give hints to a candidate they can follow.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-interview-protocol.svg" alt="Seven steps from clarify to complexity with suggested minutes, and a loop from test back to code" width="100%">
#
# *What to notice:* steps 1–4 take up to a third of the time and involve no code; the dashed loop is where verification points are earned.

# %% [markdown]
# ## 33.2 · Know your costs (and measure when unsure)
#
# | Operation | Cost | Note |
# |---|---|---|
# | `x in list`, `list.index`, `list.remove`, `list.pop(0)`, `list.insert(0, x)` | O(n) | the usual hidden quadratic |
# | `x in set/dict`, `d[k] = v`, `set.add` | O(1) average | hashing |
# | `list.append`, `list.pop()` | O(1) amortised | |
# | `deque.appendleft/popleft` | O(1) | use for queues (BFS) |
# | `heapq.heappush/heappop` | O(log n) | min-heap; negate keys for a max-heap |
# | `sorted`, `list.sort` | O(n log n) | stable (Timsort) |
# | `s[a:b]`, `list[a:b]`, `s + t` | O(length) | slicing copies |
# | `bisect.bisect_left` | O(log n) | on a sorted list |
#
# The classic hidden quadratic: a queue built on a list.

# %%
def drain_list(n):
    q = list(range(n))
    while q:
        q.pop(0)                 # shifts every remaining element


def drain_deque(n):
    q = deque(range(n))
    while q:
        q.popleft()


print("list.pop(0):")
_ = growth_table(drain_list, lambda n: (n,), [12_500, 25_000, 50_000, 100_000])
print("deque.popleft():")
_ = growth_table(drain_deque, lambda n: (n,), [12_500, 25_000, 50_000, 100_000])

# %% [markdown]
# **Expected:** `list.pop(0)` is plainly quadratic: in our run ~0.8 s at
# 100,000 with a fitted exponent of 2.0-2.2 (doubling n quadruples the time), while
# `deque.popleft` took ~4 ms with an exponent near 1.1. A BFS over a
# 100,000-node graph with a list as its queue is therefore ~200× slower than it
# needs to be, and an interviewer who sees `queue.pop(0)` is likely to ask about it.
#
# > **Intuition.** `list.pop(0)` is a queue at a counter where, every time the first
# > person is served, everyone else must take one step forward: serving n people costs
# > about n²/2 steps. A `deque` moves the counter instead. *Where it breaks down:* the
# > shift is a fast C `memmove`, so for a few thousand items the cost is negligible; the
# > quadratic term dominates only at interview-sized inputs (10⁵ and up).

# %% [markdown]
# ## Which pattern?
#
# The sections below teach six patterns one at a time. In an interview the problem does not
# say which one it is; these questions, asked in order, usually do. Chapter 34 covers the
# graph and DP rows.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-pattern-chooser.svg" alt="Decision ladder: sorted input to two pointers or binary search, contiguous window to sliding window, seen-before to hash map, top-k to heap, overlapping ranges to sort and sweep, graphs to BFS, DFS or topological sort, overlapping subproblems to DP" width="100%">
#
# *What to notice:* each question is about the *shape* of the input or of the question, never about the topic of the story it is wrapped in.

# %% [markdown]
# ## 33.3 · Pattern: hash maps (complements and counts)
#
# **Signal:** "find a pair/element with property X", "have we seen this
# before?", "count occurrences". A dictionary turns an O(n) inner search into
# an O(1) lookup. **Worked example:** group words that are anagrams of each
# other.

# %%
def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        groups["".join(sorted(w))].append(w)      # canonical key: sorted letters
    return list(groups.values())


print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# Complexity: O(n · L log L) for n words of length L; O(n · L) extra space.
# Alternative key: a 26-count tuple -> O(n · L).

# %% [markdown]
# **Expected:** `[['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]` (dicts keep
# insertion order, so groups appear in order of first occurrence).

# %% [markdown]
# > **Exercise 1.** *(target 12 min)* **Two-sum, all pairs.** Implement
# > `two_sum_pairs(nums, target)` → a sorted list of **all** index pairs
# > `(i, j)` with `i < j` and `nums[i] + nums[j] == target`. An element cannot pair
# > with itself; duplicates are allowed (`[3, 3, 3], 6` → `[(0, 1), (0, 2), (1, 2)]`).
# > Target: O(n + p) time, where p is the number of pairs in the output.
# >
# > Visible examples: `[2, 7, 11, 15], 9` → `[(0, 1)]`; `[3], 6` → `[]`;
# > `[], 0` → `[]`. The grader also checks hidden edge cases, 200 random small
# > inputs against brute force, and timing up to n = 200,000.

# %%
def two_sum_pairs(nums, target):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(two_sum_pairs)

# %% [markdown]
# > **Talk it through.** *Clarify:* indices or values? Pairs ordered? Can the
# > same element be used twice? Duplicates? Negative numbers? *Brute force:* all
# > pairs, O(n²). *Optimise:* scan once; for each j, the partners are earlier
# > indices with value `target − nums[j]` — keep a dict value → list of indices.
# > *Complexity:* O(n + p) time, O(n) space. *Why the output term matters:* with
# > `[0, 0, …, 0]` and target 0 there are n(n−1)/2 pairs, so no algorithm can
# > beat O(n²) there. Stating this shows an understanding of output-sensitive bounds.
# > *Variants:* sorted input → two pointers, O(1) extra space; count pairs only →
# > keep counts, not lists, O(n).

# %% [markdown]
# ## 33.4 · Pattern: two pointers and sliding windows
#
# **Signal:** contiguous subarray/substring with a constraint ("longest",
# "shortest", "at most k distinct"), or pairs in a **sorted** array. Two
# indices move forward only, so the total work is O(n) even though there is a
# loop inside a loop. **Worked example:** shortest subarray with sum ≥ target
# (positive numbers only — the window shrinks safely because adding an element
# never decreases the sum).
#
# > **Intuition.** A sliding window is a caterpillar: the head moves forward to take in
# > more, the tail moves forward to let go, and neither ever moves back, so every element
# > is eaten once and released once: at most 2n steps. *Where it breaks down:* the tail may
# > only move when letting go cannot make the window valid again, which is false for
# > negative numbers; that is the cue for prefix sums.

# %%
def min_subarray_len(target, nums):
    best, left, total = float("inf"), 0, 0
    for right, x in enumerate(nums):
        total += x                                 # grow the window to include right
        while total >= target:                     # shrink while still valid
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best


assert min_subarray_len(7, [2, 3, 1, 2, 4, 3]) == 2       # [4, 3]
assert min_subarray_len(100, [1, 2, 3]) == 0
print("min_subarray_len OK")
# Each index enters and leaves the window at most once: O(n) time, O(1) space.

# %% [markdown]
# **Expected:** `min_subarray_len OK`. With negative numbers this breaks
# (shrinking could *increase* the sum); that is the cue for prefix sums
# (Exercise 7).

# %% [markdown]
# > **Exercise 2.** *(target 15 min)* **Longest substring without repeating
# > characters.** Implement `longest_unique(s)` → the length of the longest
# > substring of `s` in which no character repeats, in O(n).
# >
# > Visible examples: `"abcabcbb"` → 3 (`"abc"`); `"bbbbb"` → 1; `"pwwkew"` → 3;
# > `""` → 0. Hidden: a classic trap where the window's left edge must never move
# > backwards, non-ASCII characters, 300 random strings against brute force, and
# > a 51,200-character string with no repeats.

# %%
def longest_unique(s):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(longest_unique)

# %% [markdown]
# > **Talk it through.** *Clarify:* character set (ASCII or Unicode)? Case
# > sensitive? Return the length or the substring? *Brute force:* every start,
# > extend until a repeat: O(n·m) for answer length m, O(n²) worst case.
# > *Optimise:* window `[left, right]` with no repeats; dict `last[ch]` = last
# > index seen; on a repeat **inside** the window, jump `left` to `last[ch] + 1`.
# > *Invariant to say aloud:* "`s[left:right+1]` has no repeated character."
# > *Test by hand on `"abba"`*: at the final `a`, its last index (0) is *before*
# > `left` (2) — moving `left` back to 1 would be wrong. *Complexity:* O(n) time,
# > O(min(n, alphabet)) space.

# %% [markdown]
# ## 33.5 · Pattern: heaps (top-k, k-way merge, scheduling)
#
# **Signal:** "k largest/smallest/most frequent", "merge k sorted…", "next
# event by time". A heap of size k gives O(n log k) instead of sorting
# everything in O(n log n) — and it works on a **stream**, which is the
# argument that matters at scale (top items of a day's logs without holding them
# all). Python's `heapq` is a min-heap: to keep the k *largest*, keep a min-heap
# of size k and evict its smallest. **Worked example:** merge k sorted lists.
#
# > **Intuition.** A size-k min-heap is a bouncer at a club with k places who only
# > compares each newcomer with the *weakest* person inside: a stronger newcomer replaces them.
# > Nobody else needs comparing. *Where it breaks down:* the bouncer knows only who is
# > weakest; the people inside are not sorted, so sort the final k before returning them.

# %%
def merge_k_sorted(lists):
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]   # (value, list id, position)
    heapq.heapify(heap)
    out = []
    while heap:
        value, i, j = heapq.heappop(heap)
        out.append(value)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))
    return out


print(merge_k_sorted([[1, 4, 9], [2, 3, 10], [], [0, 11]]))
# N total items, k lists: O(N log k) time, O(k) extra space (plus the output).
# The list id in the tuple breaks ties, so values never need to be compared with lists.

# %% [markdown]
# **Expected:** `[0, 1, 2, 3, 4, 9, 10, 11]`. This is exactly how a ranking
# service merges candidate lists from several retrievers by score
# (Chapter 35).

# %% [markdown]
# > **Exercise 3.** *(target 12 min)* **Top-k frequent.** Implement
# > `top_k_frequent(items, k)` → the k most frequent items, most frequent first;
# > ties go to the **smaller** item; fewer than k distinct items → all of them;
# > `k = 0` → `[]`. Use a heap of size k: O(n + u log k) for u distinct items.
# >
# > Visible examples: `[1, 1, 1, 2, 2, 3], 2` → `[1, 2]`; `[4, 4, 5, 5, 6], 2` →
# > `[4, 5]`; `["b", "a", "b", "a", "c"], 1` → `["a"]`. Timing up to 256,000 items.

# %%
def top_k_frequent(items, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(top_k_frequent)

# %% [markdown]
# > **Talk it through.** *Clarify:* tie-breaking, k larger than the number of
# > distinct items, item types (comparable?). *Plan:* `Counter` in O(n), then a
# > size-k min-heap keyed so the *worst* kept item is on top: (count, item with
# > reversed order); or `heapq.nsmallest(k, counts, key=lambda x: (-counts[x], x))`,
# > which uses a heap internally. *Complexity:* O(n + u log k) time, O(u) space.
# > *Alternatives to mention:* sorting O(u log u) (fine if u is small); bucket
# > sort by count, O(n); for a stream too large for memory, a Count-Min sketch
# > plus a heap gives approximate heavy hitters (Chapter 20).

# %% [markdown]
# ## 33.6 · Pattern: intervals (sort, then sweep)
#
# **Signal:** ranges, bookings, sessions, "overlapping". Sort by start, then
# one pass that compares each interval with the last one kept — or sweep over
# start/end events. **Worked example:** the minimum number of meeting rooms
# (equivalently, peak concurrency of sessions — a capacity-planning question).

# %%
def min_rooms(meetings):
    ends = []                                            # min-heap of end times of rooms in use
    for start, end in sorted(meetings):
        if ends and ends[0] <= start:                    # a room is free (end <= start: half-open)
            heapq.heapreplace(ends, end)
        else:
            heapq.heappush(ends, end)
    return len(ends)


print(min_rooms([[0, 30], [5, 10], [15, 20]]), min_rooms([[7, 10], [2, 4]]), min_rooms([]))
# O(n log n) time (sort + heap), O(n) space.

# %% [markdown]
# **Expected:** `2 1 0`. Note the convention: here a meeting ending at 10 frees
# the room for one starting at 10 (half-open intervals). Exercise 4 uses
# **closed** intervals, where `[1, 4]` and `[4, 5]` *touch* and merge. Always
# ask which convention the interviewer intends.

# %% [markdown]
# > **Exercise 4.** *(target 18 min)* **Merge and insert intervals.** Implement
# > `merge_intervals(intervals, new=None)` for closed intervals `[start, end]`:
# >
# > * `new is None`: merge all overlapping or touching intervals (input in any
# >   order) and return them sorted, in O(n log n);
# > * `new` given: `intervals` is already **sorted and non-overlapping**; insert
# >   `new` and merge as needed in **O(n)** (no sorting).
# >
# > Return a new list of `[start, end]` lists; do not modify the input.
# > Visible examples: `[[1,3],[2,6],[8,10],[15,18]]` → `[[1,6],[8,10],[15,18]]`;
# > `[[1,4],[4,5]]` → `[[1,5]]`; `merge_intervals([[1,3],[6,9]], new=[2,5])` →
# > `[[1,5],[6,9]]`. Timing up to 200,000 intervals for both modes.

# %%
def merge_intervals(intervals, new=None):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex4(merge_intervals)

# %% [markdown]
# > **Talk it through.** *Clarify:* closed or half-open? Sorted input? May I
# > modify it? Empty input? *Merge:* sort by start; keep `out`; if
# > `start <= out[-1][1]` extend `out[-1][1] = max(...)` (the `max` handles nesting
# > like `[1,10],[2,3]`), else append. *Insert in O(n):* three phases — copy
# > intervals ending before `new` starts; absorb those starting at or before
# > `new`'s end (grow `new`); copy the rest. *Complexity:* O(n log n) / O(n) time,
# > O(n) output. *Bug to avoid:* `intervals.sort()` sorts the caller's list.

# %% [markdown]
# ## 33.7 · Pattern: stacks (matching, "most recent", monotonic)
#
# **Signal:** nesting and matching (brackets, tags, function calls), "the most
# recent unmatched…", "next greater element". **Worked example:** valid
# brackets.

# %%
def valid_brackets(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack                               # leftovers are unclosed


print([valid_brackets(s) for s in ["()[]{}", "([)]", "((", "", "a(b)c", "]"]])
# O(n) time, O(n) space in the worst case ("((((...").

# %% [markdown]
# **Expected:** `[True, False, False, True, True, False]`. The two easy-to-miss
# edge cases are a closing bracket on an **empty** stack (`"]"`) and leftovers
# at the end (`"(("`).

# %% [markdown]
# > **Exercise 5.** *(target 10 min)* **Min-stack.** Implement a class `MinStack`
# > with `push(x)`, `pop()` (returns the removed value), `top()` and `get_min()`,
# > **all O(1)**. `pop`, `top` and `get_min` on an empty stack raise `IndexError`.
# > The stack must stay correct when the minimum is pushed more than once.
# > The grader runs a scripted sequence, 2,000 random operations against a
# > reference, and timing for up to 200,000 pushes followed by as many `get_min` calls.

# %%
class MinStack:
    def __init__(self):
        pass  # YOUR CODE HERE

    def push(self, x):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def top(self):
        raise NotImplementedError

    def get_min(self):
        raise NotImplementedError


grade.ex5(MinStack)

# %% [markdown]
# > **Talk it through.** *Clarify:* what happens on an empty stack? Duplicate
# > values? *Naive:* `min(self.items)` in `get_min` is O(n). *Idea:* the minimum
# > of a stack only changes at push/pop, and each element "remembers" the minimum
# > below it — keep a parallel stack `mins[i] = min(items[:i+1])`. *Space
# > optimisation (mention):* push to `mins` only when `x <= mins[-1]`, pop from it
# > when the popped value equals `mins[-1]` — the `<=` is what makes duplicates
# > work. *Complexity:* O(1) per operation, O(n) extra space.

# %% [markdown]
# ## 33.8 · Pattern: prefix sums and prefix products
#
# **Signal:** many range-sum queries; "subarray with sum k" where values can be
# negative; "product of everything except…". Precompute running totals once;
# then a range is a difference: `sum(nums[i:j]) = P[j] − P[i]` with
# `P[0] = 0`. **Worked example:** O(1) range-sum queries after O(n) setup.

# %%
def build_prefix(nums):
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    return prefix


nums_demo = [3, -1, 4, 1, -5, 9]
P = build_prefix(nums_demo)
print("P =", P)
print("sum(nums[1:4]) =", P[4] - P[1], "| check:", sum(nums_demo[1:4]))

# %% [markdown]
# **Expected:** `P = [0, 3, 2, 6, 7, 2, 11]` and `sum(nums[1:4]) = 4 | check: 4`.
# The same idea gives two exercises: a prefix *product* from each side
# (Exercise 6), and "how many earlier prefixes equal `P − k`" with a hash map
# (Exercise 7). It was already used in ML code: `best_split` in Chapter 32.

# %% [markdown]
# > **Exercise 6.** *(target 10 min)* **Product of array except self.**
# > Implement `product_except_self(nums)` → a list `out` with `out[i]` = the
# > product of all elements except `nums[i]`, in O(n) time **without division**
# > (the empty product is 1). Visible examples: `[1, 2, 3, 4]` → `[24, 12, 8, 6]`;
# > `[-1, 1, 0, -3, 3]` → `[0, 0, 9, 0, 0]`. Hidden: one zero, two zeros, length 0
# > and 1, random inputs, timing up to n = 200,000.

# %%
def product_except_self(nums):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex6(product_except_self)

# %% [markdown]
# > **Talk it through.** *Clarify:* why no division? (zeros, and in some
# > settings division is expensive or unavailable, e.g. modular arithmetic.)
# > Overflow? (none in Python; in fixed-width languages, ask.) *Idea:* out[i] =
# > prefix product before i × suffix product after i; one pass left-to-right
# > writing prefixes into `out`, one pass right-to-left multiplying in a running
# > suffix. *Complexity:* O(n) time, O(1) extra space beyond the output.
# > *Test aloud:* a single zero, two zeros, length 1.

# %% [markdown]
# > **Exercise 7.** *(target 15 min)* **Subarray sum equals k.** Implement
# > `subarray_sum_count(nums, k)` → the number of contiguous, non-empty subarrays
# > whose sum is exactly `k`. Numbers may be **negative**, so a sliding window
# > does not work. Target O(n). Visible examples: `[1, 1, 1], 2` → 2;
# > `[1, 2, 3], 3` → 2; `[0, 0, 0], 0` → 6.

# %%
def subarray_sum_count(nums, k):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex7(subarray_sum_count)

# %% [markdown]
# > **Talk it through.** *Clarify:* negatives? zero-length subarrays (no)?
# > count or list? *Brute force:* all (i, j) with a running sum: O(n²).
# > *Optimise:* `sum(i..j) = P[j+1] − P[i] = k` ⇔ `P[i] = P[j+1] − k`, so walk once,
# > keep `count[prefix_value]`, and add `count[P − k]` at each step. *Initialise
# > `{0: 1}`* (the empty prefix) so subarrays starting at index 0 are counted —
# > the classic off-by-one. *Complexity:* O(n) time, O(n) space. *Variant:* "longest
# > subarray with sum k" → store the **first** index of each prefix value.

# %% [markdown]
# ## 33.9 · Self-scoring sheet
#
# Fill this in after each attempt. A score is only useful if it is
# applied consistently and comparable over time. Criteria (0 = no, 1 = partly, 2 = yes):
#
# | Criterion | 2 means |
# |---|---|
# | `clarified` | asked about size, types, edge inputs and output format before coding |
# | `brute_force_stated` | named a baseline and its complexity |
# | `optimal_found_alone` | reached the target complexity without looking anything up |
# | `code_correct_first_run` | the grader passed on the first or second run |
# | `tested_by_hand` | traced the code on an example and an edge case before running it |
# | `complexity_stated` | time and space, with where they come from |
# | `within_target_time` | finished within the target time (1 = within 1.5×) |
#
# 12-14: the pattern is secure. 8-11: redo it in 2-3 days. Below 8:
# redo it tomorrow, then again in a week (spaced repetition; the plan is at the
# end of Chapter 34).

# %%
my_scores = {   # edit: one row per exercise, seven criteria each scored 0-2 (None = not attempted)
    "ex1_two_sum_pairs": None, "ex2_longest_unique": None, "ex3_top_k_frequent": None,
    "ex4_merge_intervals": None, "ex5_min_stack": None, "ex6_product_except_self": None,
    "ex7_subarray_sum_count": None,
}
# Example of a filled row: "ex1_two_sum_pairs": dict(clarified=2, brute_force_stated=2,
#     optimal_found_alone=1, code_correct_first_run=2, tested_by_hand=1, complexity_stated=2, within_target_time=1)

for name, row in my_scores.items():
    if row is None:
        print(f"{name:<26} not scored yet")
        continue
    total = sum(row.values())
    verdict = "ready" if total >= 12 else "redo in 2-3 days" if total >= 8 else "redo tomorrow"
    weakest = min(row, key=row.get)
    print(f"{name:<26} {total:>2}/14  {verdict:<17} weakest: {weakest}")

# %% [markdown]
# **Expected:** seven lines reading `not scored yet` until the dict is filled in.

# %% [markdown]
# ## Key equations
#
# Coding rounds involve little algebra, but a few lines of arithmetic decide whether an approach can work before it is written.
#
# **1. Reading growth from timings**
#
# $$
# k \approx \frac{\log(t_2 / t_1)}{\log(n_2 / n_1)}
# $$
#
# *In words:* the local exponent that `growth_table` prints: if doubling n multiplies the time by 4, the algorithm is about O(n²); by 2, O(n).
# *Example:* 0.2 s at n = 50,000 and 0.8 s at 100,000: $k = \log 4 / \log 2 = 2$.
#
# **2. Input size → the complexity you need** (rough rule: pure Python does about $10^7$ simple steps per second)
#
# $$
# n \le 20 \Rightarrow O(2^n) \qquad n \le 10^3\text{–}10^4 \Rightarrow O(n^2) \qquad n \approx 10^5\text{–}10^6 \Rightarrow O(n \log n) \text{ or } O(n)
# $$
#
# *Example:* n = 200,000 in Exercise 1: $n^2 = 4 \times 10^{10}$ steps is over an hour; $n \log_2 n \approx 3.5 \times 10^6$ is a fraction of a second.
#
# **3. Why the sliding window is O(n)**
#
# $$
# \text{total steps} = \underbrace{n}_{\text{right moves}} + \underbrace{\le n}_{\text{left moves}} \le 2n
# $$
#
# *In words:* count pointer moves, not loop nesting. A `while` inside a `for` is not automatically quadratic.
#
# **4. Heap top-k versus sorting**
#
# $$
# O(n \log k) \ \text{vs}\ O(n \log n)
# $$
#
# *Example:* n = 10⁶, k = 10: $\log_2 10 \approx 3.3$ versus $\log_2 10^6 \approx 20$: about 6× fewer comparisons, and O(k) memory on a stream.
#
# **5. Prefix sums and "subarray sum equals k"**
#
# $$
# \sum_{t=i}^{j} x_t = P_{j+1} - P_i, \qquad P_0 = 0, \qquad \#\{\text{subarrays with sum } k\} = \sum_{j} \#\{\, i \le j : P_i = P_{j+1} - k \,\}
# $$
#
# *Example:* `nums = [1, 1, 1]`, k = 2: P = (0, 1, 2, 3); the pairs with difference 2 are (0, 2) and (1, 3), so the answer is 2. The `{0: 1}` initial count is $P_0$.
#
# **6. Output-sensitive bounds**
#
# $$
# O(n + p), \qquad p \le \binom{n}{2} = \frac{n(n-1)}{2}
# $$
#
# *Example:* two-sum on `[0] * n` with target 0 has $p = n(n-1)/2$ pairs: any algorithm that lists them is quadratic there, and stating this is part of a complete answer.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | A grader says "took 2.4 s at n=16,000" or "would take ~30 s" | an O(n²) step: nested loop, `x in list`, `list.pop(0)`, `min(list)` per call, slicing in a loop | find the inner O(n) operation; replace it with a dict/set, a deque, or a precomputed array |
# | Exercise 1: missing pairs with duplicates | storing only the **last** index per value | store a list of indices per value |
# | Exercise 2: `'abba'` returns 3 | moving `left` backwards to an index outside the window | `left = max(left, last[ch] + 1)`, or check `last[ch] >= left` first |
# | Exercise 3: right items, wrong order | tie-break not applied, or heap returns ascending order | sort the final k by (−count, item) |
# | Exercise 4: "modified its input" | `intervals.sort()` or mutating the inner lists | `sorted(intervals)` and append copies `[s, e]` |
# | Exercise 4 insert: wrong at the boundary | `<` vs `<=` for touching intervals | closed intervals: absorb when `start <= new_end` and copy-before when `end < new_start` |
# | Exercise 5: `get_min` wrong after popping a duplicate minimum | the min-stack pushes only on `<` | push on `<=`, or keep one min per element |
# | Exercise 6: `ZeroDivisionError` | the division shortcut | prefix and suffix products |
# | Exercise 7: off by the number of subarrays starting at 0 | missing the initial `{0: 1}` | count the empty prefix once before the loop |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Walk through the seven-step protocol on "find the first non-repeating character in a string". What do you ask in step 1, and what is the final complexity?
# 2. Why is a sliding window O(n) even though it has a `while` loop inside a `for` loop? When does the sliding window technique fail, and what replaces it?
# 3. The 100 most frequent queries are needed from a log of 10 billion lines that does not fit in memory. What do you do, and what is exact vs approximate in your answer?
# 4. A hash map gives O(1) lookups "on average". What is the worst case, why does it happen, and when could it matter in production?
# 5. When would you prefer sorting (O(n log n)) over a hash-based O(n) solution, even though it is asymptotically slower?
# 6. An interviewer accepts a solution as correct but asks to reduce extra space from O(n) to O(1). Pick one exercise from this lab and explain how (or why it is impossible).
#
# **Next chapter:** [Chapter 34 · Coding patterns II](../ch34-coding-patterns-2/notebook.ipynb) — graphs, binary search on the answer, dynamic programming, tries and union-find, and a 4-week practice plan.
