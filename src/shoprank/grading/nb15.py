"""Self-checks for notebook 15 · Coding patterns I.

Each grader runs the examples from the notebook, hidden edge cases, a randomised
comparison against a brute-force reference, and (where the pattern is about
complexity) a timing check that an O(n) / O(n log n) solution passes with a wide
margin and an O(n^2) one cannot.
"""

from __future__ import annotations

import random
from collections import Counter

from churnlab.checks import run_check

from ..drills import check_scaling


def _brute_pairs(nums, target):
    return [(i, j) for i in range(len(nums)) for j in range(i + 1, len(nums)) if nums[i] + nums[j] == target]


def ex1(two_sum_pairs):
    def test():
        cases = [([2, 7, 11, 15], 9), ([3, 3, 3], 6), ([3], 6), ([], 0), ([1, -1, 0, 2, -2], 0),
                 ([5, 5], 10), ([1, 2, 3], 100), ([0, 0, 0, 0], 0)]
        for nums, target in cases:
            snap = list(nums)
            got = two_sum_pairs(nums, target)
            assert nums == snap, "your function modified its input"
            want = _brute_pairs(nums, target)
            assert [tuple(p) for p in got] == want, f"two_sum_pairs({nums}, {target}) = {got}, expected {want}"
        rng = random.Random(0)
        for _ in range(200):
            nums = [rng.randint(-6, 6) for _ in range(rng.randint(0, 12))]
            t = rng.randint(-8, 8)
            assert [tuple(p) for p in two_sum_pairs(nums, t)] == _brute_pairs(nums, t), f"wrong for {nums}, {t}"

        def make(n):
            r = random.Random(n)
            nums = r.sample(range(10 * n), n)          # distinct values: output size is O(n)
            return nums, 10 * n

        check_scaling(two_sum_pairs, make, [1000, 4000, 16000, 64000, 200000], label="two_sum_pairs")

    return run_check("Exercise 1 · two_sum_pairs", test,
                     hint="one pass; a dict value -> list of indices seen so far; for each j, pair it with every "
                          "earlier i where nums[i] == target - nums[j]")


def _brute_longest(s):
    best = 0
    for i in range(len(s)):
        seen = set()
        for ch in s[i:]:
            if ch in seen:
                break
            seen.add(ch)
        best = max(best, len(seen))
    return best


def ex2(longest_unique):
    def test():
        for s, want in [("abcabcbb", 3), ("bbbbb", 1), ("pwwkew", 3), ("", 0), ("a", 1), ("abba", 2),
                        ("dvdf", 3), ("tmmzuxt", 5), (" ", 1), ("au", 2), ("aäaä€", 3)]:
            got = longest_unique(s)
            assert got == want, f"longest_unique({s!r}) = {got}, expected {want}"
        rng = random.Random(1)
        for _ in range(300):
            s = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 15)))
            assert longest_unique(s) == _brute_longest(s), f"wrong for {s!r}"

        def make(n):                                     # all characters distinct: the answer is n
            return ("".join(chr(0x100 + i) for i in range(n)),)

        def validate(n, args, result):
            assert result == n, f"string of {n} distinct characters: expected {n}, got {result}"

        check_scaling(longest_unique, make, [800, 3200, 12800, 51200], label="longest_unique", validate=validate)

    return run_check("Exercise 2 · longest_unique", test,
                     hint="window [left, right]; dict char -> last index; on a repeat INSIDE the window move "
                          "left to last[ch] + 1 (never backwards - see 'abba')")


def _ref_topk(items, k):
    c = Counter(items)
    return [x for x, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:max(k, 0)]]


def ex3(top_k_frequent):
    def test():
        cases = [([1, 1, 1, 2, 2, 3], 2), ([1], 1), ([], 3), ([4, 4, 5, 5, 6], 2), (["b", "a", "b", "a", "c"], 1),
                 ([3, 1, 2], 5), ([7, 7, 8], 0)]
        for items, k in cases:
            got = top_k_frequent(items, k)
            assert list(got) == _ref_topk(items, k), (f"top_k_frequent({items}, {k}) = {got}, expected "
                                                      f"{_ref_topk(items, k)} (ties: smaller item first)")
        rng = random.Random(2)
        for _ in range(200):
            items = [rng.randint(0, 9) for _ in range(rng.randint(0, 30))]
            k = rng.randint(0, 6)
            assert list(top_k_frequent(items, k)) == _ref_topk(items, k), f"wrong for {items}, k={k}"

        def make(n):
            r = random.Random(n)
            return [r.randint(0, n // 2) for _ in range(n)], 10

        check_scaling(top_k_frequent, make, [1000, 4000, 16000, 64000, 256000], label="top_k_frequent")

    return run_check("Exercise 3 · top_k_frequent", test,
                     hint="Counter, then a min-heap of size k on (count, item-with-reversed-order) - or "
                          "heapq.nsmallest(k, counts, key=lambda x: (-counts[x], x))")


def _ref_merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def ex4(merge_intervals):
    def test():
        cases = [([[1, 3], [2, 6], [8, 10], [15, 18]], [[1, 6], [8, 10], [15, 18]]),
                 ([[1, 4], [4, 5]], [[1, 5]]), ([], []), ([[5, 7]], [[5, 7]]),
                 ([[1, 10], [2, 3], [4, 5]], [[1, 10]]), ([[8, 10], [1, 3], [2, 6]], [[1, 6], [8, 10]]),
                 ([[1, 1], [1, 1]], [[1, 1]]), ([[-5, -1], [-2, 0]], [[-5, 0]])]
        for intervals, want in cases:
            snap = [list(iv) for iv in intervals]
            got = merge_intervals(intervals)
            assert intervals == snap, "your function modified its input (sorted it in place?)"
            assert [list(iv) for iv in got] == want, f"merge_intervals({snap}) = {got}, expected {want}"
        ins = [([[1, 3], [6, 9]], [2, 5], [[1, 5], [6, 9]]),
               ([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8], [[1, 2], [3, 10], [12, 16]]),
               ([], [5, 7], [[5, 7]]), ([[1, 5]], [6, 8], [[1, 5], [6, 8]]), ([[3, 5]], [0, 1], [[0, 1], [3, 5]]),
               ([[1, 5]], [2, 3], [[1, 5]]), ([[1, 2], [5, 6]], [2, 5], [[1, 6]])]
        for intervals, new, want in ins:
            got = merge_intervals(intervals, new)
            assert [list(iv) for iv in got] == want, f"merge_intervals({intervals}, new={new}) = {got}, expected {want}"
        rng = random.Random(3)
        for _ in range(200):
            ivs = []
            for _ in range(rng.randint(0, 8)):
                a = rng.randint(0, 30)
                ivs.append([a, a + rng.randint(0, 6)])
            assert [list(iv) for iv in merge_intervals(ivs)] == _ref_merge(ivs), f"wrong for {ivs}"
            base = _ref_merge(ivs)
            a = rng.randint(0, 30)
            new = [a, a + rng.randint(0, 6)]
            assert [list(iv) for iv in merge_intervals(base, new)] == _ref_merge(base + [new]), (
                f"insert {new} into {base}: wrong")

        def make(n):
            r = random.Random(n)
            out = []
            for _ in range(n):
                a = r.randint(0, 50 * n)
                out.append([a, a + r.randint(0, 40)])
            return (out,)

        check_scaling(merge_intervals, make, [1000, 4000, 16000, 64000, 200000], label="merge_intervals")

        def make_insert(n):
            base = [[10 * i, 10 * i + 5] for i in range(n)]
            return base, [10 * (n // 2) + 7, 10 * (n // 2) + 8]

        check_scaling(merge_intervals, make_insert, [1000, 4000, 16000, 64000, 200000],
                      label="merge_intervals(..., new)")

    return run_check("Exercise 4 · merge_intervals", test,
                     hint="sort by start; extend the last output interval while start <= its end. Insert: copy those "
                          "ending before new starts, absorb those starting <= new's end, copy the rest")


def ex5(MinStack):  # noqa: N803
    def test():
        st = MinStack()
        for x in (5, 3, 7, 3, 8):
            st.push(x)
        assert st.get_min() == 3 and st.top() == 8
        assert st.pop() == 8 and st.get_min() == 3
        assert st.pop() == 3 and st.get_min() == 3, "a duplicate minimum was pushed twice; popping one keeps min 3"
        assert st.pop() == 7 and st.pop() == 3 and st.get_min() == 5 and st.top() == 5
        assert st.pop() == 5
        for method in ("pop", "top", "get_min"):
            try:
                getattr(st, method)()
            except IndexError:
                pass
            else:
                raise AssertionError(f"{method}() on an empty stack should raise IndexError")
        st.push(-1)
        assert st.get_min() == -1, "stack must be reusable after being emptied"
        rng = random.Random(4)
        st, ref = MinStack(), []
        for _ in range(2000):
            if ref and rng.random() < 0.45:
                assert st.pop() == ref.pop()
            else:
                x = rng.randint(-50, 50)
                st.push(x)
                ref.append(x)
            if ref:
                assert st.get_min() == min(ref) and st.top() == ref[-1], "get_min/top wrong after random operations"

        def run(n):
            s = MinStack()
            for i in range(n, 0, -1):
                s.push(i)
            total = 0
            for _ in range(n):
                total += s.get_min()
            for _ in range(n):
                s.pop()
            return total

        check_scaling(run, lambda n: (n,), [1000, 4000, 16000, 64000, 200000], label="MinStack operations")

    return run_check("Exercise 5 · MinStack", test,
                     hint="keep a second stack: mins[i] = min of the first i+1 items; push/pop both together")


def _ref_product(nums):
    out = []
    for i in range(len(nums)):
        p = 1
        for j, x in enumerate(nums):
            if j != i:
                p *= x
        out.append(p)
    return out


def ex6(product_except_self):
    def test():
        cases = [[1, 2, 3, 4], [-1, 1, 0, -3, 3], [0, 0, 2], [0, 4, 5], [7], [], [2, 3], [-2, -3, 4]]
        for nums in cases:
            snap = list(nums)
            got = product_except_self(nums)
            assert nums == snap, "your function modified its input"
            assert list(got) == _ref_product(nums), f"product_except_self({nums}) = {got}, expected {_ref_product(nums)}"
        rng = random.Random(5)
        for _ in range(200):
            nums = [rng.randint(-3, 3) for _ in range(rng.randint(0, 9))]
            assert list(product_except_self(nums)) == _ref_product(nums), f"wrong for {nums}"

        def make(n):
            r = random.Random(n)
            return ([r.choice((-1, 1)) for _ in range(n)],)

        check_scaling(product_except_self, make, [1000, 4000, 16000, 64000, 200000], label="product_except_self")

    return run_check("Exercise 6 · product_except_self", test,
                     hint="out[i] = (product of everything left of i) * (product of everything right of i): "
                          "one pass left-to-right, one right-to-left. Zeros need no special case")


def _ref_subarrays(nums, k):
    return sum(1 for i in range(len(nums)) for j in range(i, len(nums)) if sum(nums[i:j + 1]) == k)


def ex7(subarray_sum_count):
    def test():
        cases = [([1, 1, 1], 2, 2), ([1, 2, 3], 3, 2), ([], 0, 0), ([0, 0, 0], 0, 6), ([1, -1, 0], 0, 3),
                 ([3, 4, 7, 2, -3, 1, 4, 2], 7, 4), ([-1, -1, 1], 0, 1), ([5], 5, 1), ([5], 0, 0)]
        for nums, k, want in cases:
            got = subarray_sum_count(nums, k)
            assert got == want, f"subarray_sum_count({nums}, {k}) = {got}, expected {want}"
        rng = random.Random(6)
        for _ in range(200):
            nums = [rng.randint(-3, 3) for _ in range(rng.randint(0, 12))]
            k = rng.randint(-3, 3)
            assert subarray_sum_count(nums, k) == _ref_subarrays(nums, k), (
                f"wrong for {nums}, k={k} (negatives: a sliding window does not work here)")

        def make(n):
            r = random.Random(n)
            return [r.randint(-5, 5) for _ in range(n)], 3

        check_scaling(subarray_sum_count, make, [1000, 4000, 16000, 64000, 200000], label="subarray_sum_count")

    return run_check("Exercise 7 · subarray_sum_count", test,
                     hint="sum(i..j) = P[j+1] - P[i]; walk once keeping a dict prefix_value -> count, starting "
                          "with {0: 1}; add count[P - k] at each step")
