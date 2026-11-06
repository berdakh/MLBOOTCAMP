"""A small, well-behaved client for the churn API.

What makes a client "well-behaved":

* a **timeout** on every call (the default in many libraries is "forever");
* **retries only for retryable failures** - connection errors, 429, 502/503/504
  - with exponential backoff, and never for 4xx validation errors, which will
  fail identically every time;
* a **request id** sent with each call so a failure can be found in the logs;
* errors surfaced as exceptions with the server's message, not swallowed.

It accepts any ``httpx.Client``, including FastAPI's ``TestClient``, so the
same code is exercised in tests (in-process) and against a real server.
"""

from __future__ import annotations

import time
import uuid

import httpx

RETRYABLE = {429, 502, 503, 504}


class ChurnAPIError(RuntimeError):
    def __init__(self, status: int, detail, request_id: str | None):
        super().__init__(f"HTTP {status}: {detail} (request_id={request_id})")
        self.status, self.detail, self.request_id = status, detail, request_id


class ChurnClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", http: httpx.Client | None = None,
                 timeout: float = 5.0, retries: int = 3, backoff: float = 0.2):
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self.retries = retries
        self.backoff = backoff

    def _call(self, method: str, path: str, **kwargs) -> dict:
        request_id = uuid.uuid4().hex[:12]
        headers = {"x-request-id": request_id}
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.http.request(method, path, headers=headers, **kwargs)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
            else:
                if r.status_code < 400:
                    return r.json()
                if r.status_code not in RETRYABLE:
                    detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else r.text
                    raise ChurnAPIError(r.status_code, detail, request_id)
                last_exc = ChurnAPIError(r.status_code, r.text, request_id)
            if attempt < self.retries:
                time.sleep(self.backoff * (2 ** attempt))
        raise last_exc  # type: ignore[misc]

    def health(self) -> dict:
        return self._call("GET", "/health")

    def ready(self) -> dict:
        return self._call("GET", "/ready")

    def model(self) -> dict:
        return self._call("GET", "/model")

    def predict(self, customer: dict) -> dict:
        return self._call("POST", "/predict", json=customer)

    def predict_batch(self, customers: list[dict], chunk_size: int = 500) -> list[dict]:
        """Split large inputs into chunks below the server's batch limit."""
        out: list[dict] = []
        for i in range(0, len(customers), chunk_size):
            out.extend(self._call("POST", "/predict/batch",
                                  json={"customers": customers[i:i + chunk_size]})["predictions"])
        return out
