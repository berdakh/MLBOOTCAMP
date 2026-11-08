"""Self-checks for exercises: immediate, specific feedback.

Each exercise in a notebook ends with a cell such as::

    grade.ex1(my_function)

The grader runs small tests against your work and prints one of:

    [PASS]  - every test passed
    [FAIL]  - the first failing test, with a hint
    [TODO]  - your function still raises NotImplementedError

Graders never raise, so a notebook always runs top to bottom even with
unfinished exercises. Passing the checks means the *mechanics* are right; it
does not replace being able to explain *why* - the checkpoint questions at the
end of each notebook test that.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable


class CheckResult:
    def __init__(self, name: str, status: str, message: str = ""):
        self.name, self.status, self.message = name, status, message

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def _ipython_display_(self) -> None:
        """Already printed by run_check; don't print it a second time in Jupyter."""

    def __repr__(self) -> str:
        icon = {"pass": "[PASS]", "fail": "[FAIL]", "todo": "[TODO]", "error": "[ERROR]"}[self.status]
        return f"{icon}  {self.name}" + (f"\n    {self.message}" if self.message else "")


def run_check(name: str, test: Callable[[], None], hint: str = "") -> CheckResult:
    """Run ``test``; translate its outcome into a printed, returned result."""
    try:
        test()
    except NotImplementedError:
        result = CheckResult(name, "todo", "not attempted yet - replace `raise NotImplementedError`")
    except AssertionError as exc:
        msg = str(exc) or "an assertion failed"
        result = CheckResult(name, "fail", msg + (f"\n    hint: {hint}" if hint else ""))
    except Exception as exc:  # the student's code crashed
        tb = traceback.format_exception_only(type(exc), exc)[-1].strip()
        result = CheckResult(name, "error", tb + (f"\n    hint: {hint}" if hint else ""))
    else:
        result = CheckResult(name, "pass")
    print(repr(result))
    return result


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
