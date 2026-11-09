"""churnlab - the running project of the ML Engineering Bootcamp.

A small, readable, end-to-end ML system for a fictional subscription business:

    data.py        the dataset (synthetic, committed, checksummed)
    validate.py    data contracts and cleaning decisions
    features.py    point-in-time features and labels in SQL (DuckDB)
    splits.py      leakage-safe time / group splits and rolling-origin CV
    modeling.py    sklearn pipelines: preprocessing + model in one object
    evaluation.py  metrics, bootstrap CIs, thresholds, calibration, slices
    artifacts.py   model bundles with metadata and integrity checks
    registry.py    versions, a CURRENT pointer, promotion and rollback
    schemas.py     the API contract (Pydantic)
    settings.py    configuration from environment variables
    logs.py        structured JSON logging
    api.py         the FastAPI service (app factory)
    client.py      a retrying, timeout-aware API client
    monitoring.py  drift (PSI), alert rules, latency summaries
    colab.py       artifact persistence across Colab sessions
    bootstrap.py   rebuild any earlier chapter's outputs in seconds
    checks.py      exercise self-checks; graders live in src/grading/chNN.py

Read the modules: they are short and commented for a first reading.
"""

__version__ = "1.0.0"
