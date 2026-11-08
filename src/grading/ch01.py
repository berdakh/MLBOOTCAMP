"""Self-checks for Chapter 1 · Skills diagnostic.

Unlike the other graders, these exercises measure where you are *before* the
bootcamp, so ``summary`` turns the results into a study plan. The answer key
for the judgement questions (exercise 5) is stored as hashes so that reading
this file does not give the answers away.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from churnlab.checks import CheckResult, run_check

# ---------------------------------------------------------------------------
# Exercise 1 · Python
# ---------------------------------------------------------------------------


def ex1(top_k_frequent):
    def test():
        items = ["b", "a", "c", "a", "b", "d", "a", "c", "b"]
        got = top_k_frequent(items, 2)
        assert list(got) == ["a", "b"], f"top_k_frequent(items, 2) gave {got!r}; 'a' and 'b' both appear 3 times"
        got = top_k_frequent(items, 3)
        assert list(got) == ["a", "b", "c"], f"k=3 gave {got!r}"
        got = top_k_frequent(["z", "y", "x"], 2)
        assert list(got) == ["x", "y"], f"all counts tied: expected ascending order ['x', 'y'], got {got!r}"
        got = top_k_frequent([3, 1, 3, 2, 2, 1, 1], 2)
        assert list(got) == [1, 2], f"integers: expected [1, 2] (1 appears 3x; 2 and 3 tie, 2 < 3), got {got!r}"
        got = top_k_frequent(["a", "b"], 5)
        assert list(got) == ["a", "b"], f"k larger than the number of distinct items should return all, got {got!r}"
        assert list(top_k_frequent([], 3)) == [], "an empty input should return []"
        assert list(top_k_frequent(["a"], 0)) == [], "k=0 should return []"

    return run_check("Exercise 1 · Python · top_k_frequent", test,
                     hint="collections.Counter, then sorted(counts, key=lambda x: (-counts[x], x))[:k]")


# ---------------------------------------------------------------------------
# Exercise 2 · pandas
# ---------------------------------------------------------------------------


def _usage_fixture() -> pd.DataFrame:
    w1, w2, w3 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-15")
    rows = [
        ("C1", w1, 3), ("C1", w1, 3),           # duplicate row
        ("C2", w1, 0),                          # inactive
        ("C3", w1, -1),                         # sentinel: unknown, not active
        ("C1", w2, 1), ("C2", w2, 5), ("C3", w2, None),
        ("C4", w2, 2), ("C4", w2, 2),
        ("C2", w3, 0), ("C3", w3, -1),          # nobody active in w3
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "week_start", "sessions"])
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


def ex2(weekly_active):
    def test():
        usage = _usage_fixture()
        before = usage.copy()
        got = weekly_active(usage)
        pd.testing.assert_frame_equal(usage, before, obj="the input DataFrame (do not modify it)")
        assert isinstance(got, pd.Series), f"expected a pandas Series, got {type(got).__name__}"
        idx = pd.to_datetime(pd.Index(got.index))
        expected_idx = pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"])
        assert list(idx) == list(expected_idx), (
            f"index should be every week_start in the data, sorted; got {list(got.index)}")
        values = [int(v) for v in got.to_numpy()]
        assert values[0] == 1, f"week 2024-01-01: expected 1 active customer, got {values[0]} (duplicates? 0? -1?)"
        assert values[1] == 3, f"week 2024-01-08: expected 3 active customers, got {values[1]}"
        assert values[2] == 0, f"week 2024-01-15: expected 0 (a week with nobody active still appears), got {values[2]}"

    return run_check("Exercise 2 · pandas · weekly_active", test,
                     hint="filter sessions > 0, groupby('week_start')['customer_id'].nunique(), "
                          "then reindex to all weeks with fill_value=0")


# ---------------------------------------------------------------------------
# Exercise 3 · SQL
# ---------------------------------------------------------------------------


def _billing_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "plan": ["basic", "pro", "team", "pro"],
    })
    invoices = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "D"],
        "invoice_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-02-01",
                                        "2024-01-15", "2024-02-15", "2024-01-05"]),
        "amount_usd": [29.0, 35.0, 31.0, 79.0, 85.0, 199.0],
        "status": ["paid", "failed", "failed", "failed", "paid", "paid"],
    })
    return customers, invoices


def ex3(customer_summary_sql):
    def test():
        import duckdb

        sql = customer_summary_sql()
        assert isinstance(sql, str), f"the function must return the SQL text as a string, got {type(sql).__name__}"
        customers, invoices = _billing_fixture()
        con = duckdb.connect()
        con.register("customers", customers)
        con.register("invoices", invoices)
        got = con.execute(sql).df()
        con.close()
        need = {"customer_id", "plan", "latest_amount_usd", "failed_invoices"}
        assert need <= set(got.columns), f"missing columns {sorted(need - set(got.columns))}"
        got = got.sort_values("customer_id").reset_index(drop=True)
        assert list(got["customer_id"]) == ["A", "B", "C", "D"], (
            f"expected one row per customer A-D (C has no invoices), got {list(got['customer_id'])}")
        assert list(got["plan"]) == ["basic", "pro", "team", "pro"], "plan column is wrong"
        latest = got["latest_amount_usd"].tolist()
        assert latest[0] == 35.0, f"A's latest invoice (2024-03-01) is 35.0, got {latest[0]} - order by date, not insertion"
        assert latest[1] == 85.0 and latest[3] == 199.0, f"latest amounts for B, D wrong: {latest}"
        assert pd.isna(latest[2]), f"C has no invoices: latest_amount_usd should be NULL, got {latest[2]}"
        failed = [int(v) for v in got["failed_invoices"].fillna(-1)]
        assert failed == [2, 1, 0, 0], f"failed_invoices should be [2, 1, 0, 0] for A-D, got {failed}"

    return run_check("Exercise 3 · SQL · customer_summary_sql", test,
                     hint="aggregate invoices per customer first (ROW_NUMBER or arg_max for latest, "
                          "COUNT(*) FILTER for failed), then LEFT JOIN from customers and COALESCE the count")


# ---------------------------------------------------------------------------
# Exercise 4 · testing
# ---------------------------------------------------------------------------


def rolling_mean_last_n(values, n):
    """Correct reference: element i is the mean of the last (up to) n values ending at i."""
    out = []
    for i in range(len(values)):
        window = values[max(0, i - n + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def _bug_off_by_one(values, n):
    out = []
    for i in range(len(values)):
        window = values[max(0, i - n): i + 1]  # n + 1 values once i >= n
        out.append(sum(window) / len(window))
    return out


def _bug_short_input(values, n):
    if len(values) < n:
        return []  # "not enough data" - silently drops everything
    return rolling_mean_last_n(values, n)


def _bug_mutates_input(values, n):
    out, window = [], []
    while values:
        window.append(values.pop(0))  # consumes the caller's list
        window = window[-n:]
        out.append(sum(window) / len(window))
    return out


BUGGY = {
    "off-by-one window": _bug_off_by_one,
    "wrong result when len(values) < n": _bug_short_input,
    "mutates its input": _bug_mutates_input,
}


def ex4(test_rolling_mean):
    def test():
        try:
            test_rolling_mean(rolling_mean_last_n)
        except NotImplementedError:
            raise
        except AssertionError as exc:
            raise AssertionError(f"your test FAILS the correct implementation: {exc}") from None
        missed = []
        for name, buggy in BUGGY.items():
            try:
                test_rolling_mean(buggy)
            except NotImplementedError:
                raise
            except Exception:  # an assert (or a crash) inside the test = bug caught
                continue
            missed.append(name)
        caught = len(BUGGY) - len(missed)
        assert not missed, (f"your test caught {caught}/{len(BUGGY)} buggy implementations; "
                            f"it let through: {', '.join(missed)}")

    return run_check("Exercise 4 · testing · test_rolling_mean", test,
                     hint="test a hand-computed example, an input shorter than n, and that the "
                          "input list is unchanged after the call")


# ---------------------------------------------------------------------------
# Exercise 5 · evaluation judgement
# ---------------------------------------------------------------------------

# First 16 hex characters of sha256(f"q{number}:{letter}") for the correct
# letter, so reading this file does not reveal the answers. (Six questions with
# four options each are trivially brute-forced; please don't - it only cheats you.)
_KEY = {
    1: "de5bd8bd9e3b0d55",
    2: "751c8ea895fd8bcd",
    3: "6bfe40d993db6cd7",
    4: "fec3ebd5cbc8050f",
    5: "f49d1fd4e53bdf9e",
    6: "20be3f8962f206ba",
}


def _h(q: int, letter: str) -> str:
    return hashlib.sha256(f"q{q}:{letter.strip().lower()}".encode()).hexdigest()[:16]


def ex5(answers):
    def test():
        given = answers()
        assert isinstance(given, dict), f"answers() must return a dict like {{1: 'a', 2: 'c', ...}}, got {type(given).__name__}"
        given = {int(k): str(v) for k, v in given.items() if v is not None}
        missing = [q for q in _KEY if q not in given]
        wrong = [q for q in _KEY if q in given and _h(q, given[q]) != _KEY[q]]
        correct = len(_KEY) - len(missing) - len(wrong)
        result_holder["score"] = correct
        msg = f"{correct}/{len(_KEY)} correct"
        if wrong:
            msg += f"; revisit question(s) {', '.join(map(str, wrong))}"
        if missing:
            msg += f"; unanswered: {', '.join(map(str, missing))}"
        assert not wrong and not missing, msg

    result_holder: dict = {}
    result = run_check("Exercise 5 · evaluation · answers", test,
                       hint="the right answers are not shown here; Chapters 5-8 cover each scenario")
    result.score = result_holder.get("score")
    return result


# ---------------------------------------------------------------------------
# Exercise 6 · APIs (Pydantic)
# ---------------------------------------------------------------------------


def ex6(make_model):
    def test():
        from pydantic import BaseModel, ValidationError

        Order = make_model()
        assert isinstance(Order, type) and issubclass(Order, BaseModel), \
            "make_model() must return a Pydantic model CLASS (a subclass of BaseModel), not an instance"
        ok = Order(order_id="A-1", quantity=1, unit_price=9.99)
        assert ok.coupon is None, "coupon should default to None"
        Order(order_id="A-2", quantity=1000, unit_price=0.01, coupon="SAVE10")
        bad = {
            "empty order_id": dict(order_id="", quantity=1, unit_price=1.0),
            "quantity 0": dict(order_id="A", quantity=0, unit_price=1.0),
            "quantity 1001": dict(order_id="A", quantity=1001, unit_price=1.0),
            "quantity 2.5": dict(order_id="A", quantity=2.5, unit_price=1.0),
            "unit_price 0": dict(order_id="A", quantity=1, unit_price=0),
            "unit_price -5": dict(order_id="A", quantity=1, unit_price=-5.0),
            "missing unit_price": dict(order_id="A", quantity=1),
            "an extra field 'discount'": dict(order_id="A", quantity=1, unit_price=1.0, discount=0.5),
        }
        accepted = []
        for name, payload in bad.items():
            try:
                Order(**payload)
            except ValidationError:
                continue
            accepted.append(name)
        assert not accepted, f"these invalid payloads were accepted: {', '.join(accepted)}"

    return run_check("Exercise 6 · APIs · Order model", test,
                     hint="Field(min_length=1), Field(ge=1, le=1000), Field(gt=0), "
                          "model_config = ConfigDict(extra='forbid')")


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------

AREAS = {
    # area: (how it is measured, chapters to prioritise)
    "Python": ("exercise 1", ["2", "6", "12", "17"]),
    "pandas": ("exercise 2", ["3", "4", "17"]),
    "SQL": ("exercise 3", ["3", "4", "17"]),
    "Testing": ("exercise 4", ["2", "6", "12", "17"]),
    "Evaluation": ("exercise 5", ["5", "6", "7", "8"]),
    "APIs": ("exercise 6", ["10", "11", "12"]),
    "Deployment": ("self-rating", ["13", "14", "15"]),
    "Monitoring": ("self-rating", ["13", "14", "15"]),
    "Communication": ("self-rating", ["16", "18"]),
}

_EXERCISE_AREA = {"1": "Python", "2": "pandas", "3": "SQL", "4": "Testing", "5": "Evaluation", "6": "APIs"}

#: Self-rating statements: id -> (area, statement).
SELF_RATING_STATEMENTS = {
    "deploy_package": ("Deployment", "I have packaged a Python project (pyproject.toml) and installed it in a fresh environment"),
    "deploy_docker": ("Deployment", "I have written a Dockerfile for a Python service and explained each line"),
    "deploy_ci": ("Deployment", "I have set up CI that runs tests on every push"),
    "monitor_health": ("Monitoring", "I have added health checks and structured logs to a service"),
    "monitor_drift": ("Monitoring", "I have monitored a deployed model for drift or degraded performance"),
    "monitor_rollback": ("Monitoring", "I have planned or performed a rollback of a bad release"),
    "comm_explain": ("Communication", "I can explain a modelling decision to a non-specialist in two minutes"),
    "comm_defend": ("Communication", "I can defend an evaluation choice when an interviewer pushes back"),
    "comm_design": ("Communication", "I have done a timed ML system design interview (real or mock)"),
}


def _exercise_number(result: CheckResult) -> str | None:
    name = getattr(result, "name", "")
    if name.startswith("Exercise "):
        return name.split()[1]
    return None


def _graded_status(result: CheckResult | None) -> str:
    if result is None or result.status == "todo":
        return "not attempted"
    if result.status == "pass":
        return "strong"
    score = getattr(result, "score", None)
    if score is not None and score >= 4:
        return "review"
    return "priority" if result.status in ("fail", "error") else "review"


def _rated_status(ratings: list[int]) -> str:
    if not ratings:
        return "not rated"
    mean = sum(ratings) / len(ratings)
    if mean >= 3:
        return "strong"
    if mean >= 2:
        return "review"
    return "priority"


def summary(results, self_ratings: dict | None = None, path: Path | None = None) -> dict:
    """Print an area -> status -> chapters table and write ``diagnostic.json``.

    ``results`` is the list of CheckResults returned by ``ex1`` ... ``ex6``
    (re-running a check cell may append a newer result; the last one wins).
    ``self_ratings`` maps statement ids in ``SELF_RATING_STATEMENTS`` to 1-4.
    """
    latest: dict[str, CheckResult] = {}
    for r in results or []:
        number = _exercise_number(r)
        if number in _EXERCISE_AREA:
            latest[number] = r

    ratings_by_area: dict[str, list[int]] = {}
    for key, value in (self_ratings or {}).items():
        if key not in SELF_RATING_STATEMENTS or value is None:
            continue
        value = int(value)
        if not 1 <= value <= 4:
            print(f"ignoring self-rating {key}={value}: use 1-4")
            continue
        ratings_by_area.setdefault(SELF_RATING_STATEMENTS[key][0], []).append(value)

    rows = []
    for area, (measure, notebooks) in AREAS.items():
        if measure.startswith("exercise"):
            number = measure.split()[1]
            r = latest.get(number)
            status = _graded_status(r)
            detail = r.status if r is not None else "todo"
            if r is not None and getattr(r, "score", None) is not None:
                detail = f"{r.score}/6"
        else:
            vals = ratings_by_area.get(area, [])
            status = _rated_status(vals)
            detail = f"mean {sum(vals) / len(vals):.1f}" if vals else "-"
        rows.append({"area": area, "measured_by": measure, "result": detail, "status": status,
                     "notebooks": notebooks})

    order = {"priority": 0, "not attempted": 1, "not rated": 1, "review": 2, "strong": 3}
    print(f"{'area':<14}{'measured by':<13}{'result':<10}{'status':<15}prioritise chapters")
    print("-" * 72)
    for row in rows:
        print(f"{row['area']:<14}{row['measured_by']:<13}{row['result']:<10}{row['status']:<15}"
              f"{', '.join(row['notebooks'])}")

    focus: list[str] = []
    for row in sorted(rows, key=lambda r: order[r["status"]]):
        if row["status"] in ("priority", "review", "not attempted", "not rated"):
            for nb in row["notebooks"]:
                if nb not in focus:
                    focus.append(nb)
    print()
    if focus:
        print("Spend extra time on chapters (most urgent first):", ", ".join(focus))
    else:
        print("No weak area detected: work through Part I in order, then the optional extensions.")

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "areas": rows,
        "focus_notebooks": focus,
        "self_ratings": {k: v for k, v in (self_ratings or {}).items()},
    }
    if path is None:
        from churnlab.colab import artifact_dir

        path = artifact_dir() / "diagnostic.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    print(f"\nsaved {path}")
    return copy.deepcopy(record)
