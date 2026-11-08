"""Self-checks for Chapter 12 · API clients & tests."""

from __future__ import annotations

import time

from churnlab.checks import run_check
from churnlab.schemas import CustomerFeatures, Prediction

# ------------------------------------------------------------------ Exercise 1


class _FakeHTTP:
    """Replays a script of outcomes: an int is a status code, an exception is raised."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, path, **kwargs):
        import httpx

        outcome = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        request = httpx.Request("GET", f"http://fake.test{path}")
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("scripted failure", request=request)
        body = {"ok": True, "path": path} if outcome < 400 else {"detail": f"scripted {outcome}"}
        return httpx.Response(outcome, json=body, request=request)


def ex1(retrying_get):
    def test():
        import httpx

        def expect_success(script, retries, n_calls, what):
            fake = _FakeHTTP(script)
            out = retrying_get(fake, "/thing", retries, 0.001)
            assert out == {"ok": True, "path": "/thing"}, f"{what}: expected the JSON body, got {out!r}"
            assert fake.calls == n_calls, f"{what}: expected {n_calls} calls, made {fake.calls}"

        def expect_raise(script, retries, n_calls, what):
            fake = _FakeHTTP(script)
            try:
                retrying_get(fake, "/thing", retries, 0.001)
            except NotImplementedError:
                raise
            except Exception:
                pass
            else:
                raise AssertionError(f"{what}: expected an exception, got a return value")
            assert fake.calls == n_calls, f"{what}: expected {n_calls} call(s), made {fake.calls}"

        expect_success([200], 3, 1, "immediate success")
        expect_success([503, 503, 200], 3, 3, "two 503s then 200")
        expect_success([429, 502, 504, 200], 3, 4, "429, 502, 504 then 200")
        expect_success([httpx.ConnectError, 200], 3, 2, "a connection error then 200")
        expect_success([httpx.ReadTimeout, 200], 3, 2, "a read timeout then 200")
        expect_raise([422], 3, 1, "a 422 (must NOT be retried)")
        expect_raise([404], 3, 1, "a 404 (must NOT be retried)")
        expect_raise([500, 200], 3, 1, "a 500 (a bug: not in the retryable set)")
        expect_raise([503], 2, 3, "503 forever with retries=2 (1 call + 2 retries, then raise)")
        expect_raise([httpx.ConnectError], 1, 2, "connection errors forever with retries=1")
        start = time.perf_counter()
        expect_success([503, 503, 503, 200], 3, 4, "three 503s then 200")
        assert time.perf_counter() - start < 2.0, "with backoff=0.001 the retries should take milliseconds"

    return run_check("Exercise 1 · retrying_get", test,
                     hint="loop `for attempt in range(retries + 1)`; catch httpx.TransportError; "
                          "if status in {429, 502, 503, 504}: sleep and continue; else raise_for_status()")


# ------------------------------------------------------------------ Exercise 2


def _payload():
    from churnlab.api import example_payload
    from churnlab.bootstrap import ensure_split

    return example_payload(ensure_split().valid.iloc[0])


def _correct_client():
    from fastapi.testclient import TestClient

    from churnlab.api import create_app
    from churnlab.bootstrap import ensure_registry
    from churnlab.logs import ListHandler, get_logger
    from churnlab.settings import Settings

    registry = ensure_registry(verbose=False)
    return TestClient(create_app(Settings(registry_dir=registry.root),
                                 logger=get_logger("churn-api-grader", handler=ListHandler())))


class _UnboundedSessions(CustomerFeatures):
    """The real schema with one bug: the ge=0 bound on sessions_4w is gone."""

    sessions_4w: float


def _buggy_client(crash: bool):
    """A small app that differs from the real contract only in the sessions_4w bound."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    # Module-level class: FastAPI must be able to resolve the (string) annotation.
    @app.post("/predict", response_model=Prediction)
    def predict(customer: _UnboundedSessions):
        if crash and customer.sessions_4w < 0:
            raise ValueError("log1p of a negative number")  # becomes a 500
        return Prediction(customer_id=customer.customer_id, churn_probability=0.05, flagged=False,
                          threshold=0.19, model_version="buggy")

    return TestClient(app, raise_server_exceptions=False)


def ex2(test_rejects_negative_sessions):
    def test():
        try:
            test_rejects_negative_sessions(_correct_client())
        except (NotImplementedError, AssertionError):
            raise
        except Exception as exc:
            raise AssertionError(f"your test crashed against the CORRECT app: {type(exc).__name__}: {exc}") from exc

        for crash, what in ((False, "answers 200"), (True, "crashes with a 500")):
            try:
                test_rejects_negative_sessions(_buggy_client(crash))
            except NotImplementedError:
                raise
            except Exception:
                continue  # good: the test failed against the buggy app
            raise AssertionError(
                f"your test PASSED against a buggy app that {what} for sessions_4w = -1. "
                "Assert the exact status (422) and that the error names sessions_4w.")

    return run_check("Exercise 2 · test_rejects_negative_sessions", test,
                     hint="r = client.post('/predict', json={**valid_payload, 'sessions_4w': -1}); "
                          "assert r.status_code == 422; assert any('sessions_4w' in e['loc'] for e in r.json()['detail'])")
