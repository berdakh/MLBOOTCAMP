"""Reference solutions for notebook 11. Graded by tests/test_solutions.py."""

from __future__ import annotations

import time

import httpx

RETRYABLE = {429, 502, 503, 504}


def ex1(http, path: str, retries: int = 3, backoff: float = 0.1) -> dict:
    for attempt in range(retries + 1):
        last_try = attempt == retries
        try:
            r = http.get(path)
        except httpx.TransportError:  # connection errors and timeouts
            if last_try:
                raise
        else:
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRYABLE or last_try:
                r.raise_for_status()  # 4xx / 500: fail fast; retryable: out of attempts
        time.sleep(backoff * 2 ** attempt)
    raise AssertionError("unreachable")


def _valid_payload() -> dict:
    # In the notebook this is the global `valid_payload`.
    from churnlab.api import example_payload
    from churnlab.bootstrap import ensure_split

    return example_payload(ensure_split().valid.iloc[0])


def ex2(client) -> None:
    r = client.post("/predict", json={**_valid_payload(), "sessions_4w": -1})
    assert r.status_code == 422, f"expected 422, got {r.status_code}"
    errors = r.json()["detail"]
    assert any(e["loc"][-1] == "sessions_4w" and e["type"] == "greater_than_equal" for e in errors), errors
