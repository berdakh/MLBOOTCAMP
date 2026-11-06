"""Production entry point for the TaskFlow churn API.

    uvicorn serve:app --host 0.0.0.0 --port 8000     # how the container runs it
    python serve.py                                   # the same, port from $PORT (default 8000)

All configuration comes from CHURN_* environment variables (see churnlab/settings.py).
An invalid value stops the process at start-up with a clear message, before it
accepts any traffic: failing fast is safer than serving with a wrong setting.
"""

from __future__ import annotations

import logging
import os
import sys

from churnlab.api import create_app
from churnlab.logs import get_logger
from churnlab.settings import Settings


def build_app():
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from None
    # One JSON object per line on stdout: container platforms collect stdout.
    logger = get_logger(settings.service_name, settings.log_level,
                        handler=logging.StreamHandler(sys.stdout))
    return create_app(settings, logger=logger)


app = build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="warning")
