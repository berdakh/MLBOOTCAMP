"""Structured (JSON) logging.

One JSON object per line is what log platforms (CloudWatch, Stackdriver, Loki,
Datadog) parse best: you can filter on ``model_version`` or compute p95 of
``latency_ms`` without regular expressions. Never log raw personal data;
log identifiers and aggregates.
"""

from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ListHandler(logging.Handler):
    """Keeps formatted records in memory - lets a notebook or test read the logs."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))

    def parsed(self) -> list[dict]:
        return [json.loads(r) for r in self.records]


def get_logger(name: str = "churn-api", level: str = "INFO", handler: logging.Handler | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    h = handler or logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    logger.addHandler(h)
    return logger


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})
