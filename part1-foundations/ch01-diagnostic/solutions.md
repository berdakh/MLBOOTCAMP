# Solutions · Chapter 1 · Skills diagnostic

Do the diagnostic **before** reading this. The point is to measure where you
are, and reading first measures nothing. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `top_k_frequent`

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex1`), graded in CI.

```python
from collections import Counter

def top_k_frequent(items, k):
    counts = Counter(items)
    return sorted(counts, key=lambda item: (-counts[item], item))[:k]
```

**Why it works.** One sort with a composite key: negative count (most frequent
first), then the item (ascending tie-break). Slicing handles `k` larger than
the number of distinct items and `k = 0`.

**Common mistakes.** `Counter.most_common(k)` breaks ties by first appearance,
not by value, so it fails the tie tests. Sorting twice in the wrong order (a
stable sort must sort by the *secondary* key first).

**Push further.** For huge inputs with small `k`, `heapq.nsmallest(k, counts,
key=...)` is O(n log k). Say when that matters (rarely, below millions).

**If this took you more than 10 minutes:** notebooks 01, 05, 11 and 16.

---

## Exercise 2 · `weekly_active`

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex2`).

```python
def weekly_active(usage):
    weeks = pd.Index(sorted(usage["week_start"].unique()), name="week_start")
    active = usage[usage["sessions"] > 0]          # NaN > 0 and -1 > 0 are False
    counts = active.groupby("week_start")["customer_id"].nunique()
    return counts.reindex(weeks, fill_value=0)
```

**The traps, one per fixture row:** duplicates (`nunique`, not `size` or
`count`), the `-1` sentinel and NaN (a `> 0` filter drops both), and a week
with nobody active (filter-then-group loses it; `reindex(..., fill_value=0)`
puts it back).

**Common mistake.** `usage.groupby("week_start")["sessions"].apply(lambda s: (s > 0).sum())`
counts rows, so the duplicated row is counted twice.

**If this took you more than 10 minutes:** notebooks 02, 03 and 16.

---

## Exercise 3 · `customer_summary_sql`

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex3`).

```sql
WITH ranked AS (
    SELECT customer_id, amount_usd,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) AS rn
    FROM invoices
),
latest AS (SELECT customer_id, amount_usd AS latest_amount_usd FROM ranked WHERE rn = 1),
failed AS (
    SELECT customer_id, COUNT(*) FILTER (WHERE status = 'failed') AS failed_invoices
    FROM invoices GROUP BY customer_id
)
SELECT c.customer_id, c.plan, l.latest_amount_usd,
       COALESCE(f.failed_invoices, 0) AS failed_invoices
FROM customers AS c
LEFT JOIN latest AS l USING (customer_id)
LEFT JOIN failed AS f USING (customer_id)
ORDER BY c.customer_id
```

**Key decisions.** Aggregate invoices to one row per customer *before* joining
(no fan-out), `LEFT JOIN` from `customers` so customers without invoices
survive, and `COALESCE` the count (but not the amount: "no invoice" is not
"$0").

**Equally good.** `arg_max(amount_usd, invoice_date)` in DuckDB, or
`QUALIFY ROW_NUMBER() ... = 1`. An `ORDER BY invoice_date` without `DESC`
returns the *first* invoice; the fixture catches that.

**If this took you more than 10 minutes:** notebooks 02, 03 and 16.

---

## Exercise 4 · `test_rolling_mean`

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex4`).

```python
def test_rolling_mean(fn):
    values = [1, 2, 3, 4, 5]
    original = list(values)
    got = fn(values, 3)
    expected = [1.0, 1.5, 2.0, 3.0, 4.0]
    assert len(got) == len(expected)
    assert all(math.isclose(a, b) for a, b in zip(got, expected)), got
    assert values == original, "input was modified"
    got = fn([2, 4], 5)                       # shorter than the window
    assert len(got) == 2 and all(math.isclose(a, b) for a, b in zip(got, [2.0, 3.0]))
    assert [float(x) for x in fn([7, 8, 9], 1)] == [7.0, 8.0, 9.0]
    assert list(fn([], 3)) == []
```

**The hidden bugs** were an off-by-one window (n + 1 values), returning `[]`
when `len(values) < n`, and an implementation that consumes the caller's list.
A test catches them only if it (1) uses an example long enough for the window
to fill, computed **by hand**, (2) covers the short-input edge case the
docstring promises, and (3) checks a side effect: the input after the call.

**The skill being measured** is reading a specification and turning each
promise in it into an assertion. A test that recomputes the expected value with
the same algorithm as the implementation tests nothing.

**If this took you more than 10 minutes, or caught fewer than 3 bugs:** notebooks
01, 05, 11 (pytest suites that catch real bugs) and 16.

---

## Exercise 5 · Evaluation judgement

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex5`).

| Q | Answer | Why |
|---|---|---|
| 1 | **b** | A value computed at export time describes the future of every 2023 snapshot. A jump from 0.16 to 0.71 is a symptom, not a result. Notebook 02 §2.6, 04 §4.2. |
| 2 | **c** | The deployment question is "next month", so validate on later months than you train on, with a gap of at least the label window. Notebook 04. |
| 3 | **a** | Capacity is 200 calls: precision in the top 200 is the business metric; PR-AUC summarises ranking quality at the top. Accuracy is 95% for "nobody churns". Notebooks 04, 07. |
| 4 | **d** | Call when p × 150 > (1 − p) × 30, i.e. p > 1/6. F1 and precision = recall ignore the costs. Notebook 07. |
| 5 | **b** | A 0.004 AUC gap on ~360 positives is well inside bootstrap and seed noise. Repeated test-set peeking (d) is the worst option. Notebook 06. |
| 6 | **a** | Immediate change with no time for behaviour to change points at the pipeline, not the world: training/serving skew. Drift (b) is gradual. Notebooks 10, 14. |

**Scoring.** 6/6 strong; 4-5 review the notebooks for the missed numbers; ≤ 3
make notebooks 04-07 a priority.

---

## Exercise 6 · Pydantic `Order`

✅ Reference code: [`code/nb00.py`](solutions.py) (`ex6`).

```python
from pydantic import BaseModel, ConfigDict, Field

def make_model():
    class Order(BaseModel):
        model_config = ConfigDict(extra="forbid")
        order_id: str = Field(min_length=1)
        quantity: int = Field(ge=1, le=1000)
        unit_price: float = Field(gt=0)
        coupon: str | None = None
    return Order
```

**Details that matter.** `extra="forbid"` turns a misspelt field
(`unit_prise`) into a 422 instead of a silently ignored value. `gt=0` vs
`ge=0` is the difference between "free items are invalid" and "allowed". In
Pydantic's default (lax) mode `quantity=2.5` is rejected but `"5"` is coerced
to 5; use `strict=True` if you need to refuse strings.

**If this took you more than 10 minutes:** notebooks 09-11.

---

## Exercise 7 · The most rigorous evaluation you have done (written)

✍️ Personal. A strong three-sentence answer has: **what** was evaluated and on
what data, **how** the evaluation matched the claim (e.g. leave-one-subject-out
for a cross-subject claim, with the number of subjects and a spread), and
**one specific threat** with what was done about it. Example:

> I evaluated a cross-subject motor-imagery decoder with leave-one-subject-out
> CV over 20 subjects, reporting mean balanced accuracy with a bootstrap 95%
> interval over subjects. The decision of model and hyper-parameters was made
> on nested inner folds, so no held-out subject influenced it. The main threat
> is that all subjects were recorded on one amplifier in one lab, so the
> estimate says little about a new site - I would test that on an external
> dataset before claiming generality.

Weak answers name a metric without the split, or name "small sample size" as
the threat without saying what it does to the claim.

---

## Checkpoint answers

**1. Happy-path tests.** One example exercises one path; bugs live at
boundaries and in side effects. Always add: an edge case from the spec (empty,
shorter than a window, zero), a hand-computed expected value, and a check that
inputs are unchanged / no hidden state.

**2. Joins.** `LEFT JOIN` keeps unmatched left rows (with NULLs), `JOIN` drops
them. Joining two one-to-many tables on the same key produces every pair
(fan-out), multiplying sums. Aggregate per key first.

**3. Accuracy at 5%.** "Nobody churns" scores 95%. Report PR-AUC (ranking at
the top), precision/recall at the operating point, or precision@capacity.

**4. Validation at the API boundary.** Training data is validated once, by
you; requests arrive from clients you do not control, forever. Schemas reject
malformed or out-of-range inputs before they reach the model, where they would
otherwise produce a confident, meaningless score.

**5. Your weakest area.** ✍️ Personal. A good answer names the area, the
specific failure (e.g. "my test caught 1/3 bugs: I never checked the input"),
and a concrete plan (e.g. "do notebook 11's exercises without solutions").
