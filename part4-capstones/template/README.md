# TODO: Project title — one line that names the question, not the algorithm

> **Starter template.** Copy this folder into a new repository for each
> capstone, rename the package `capstone`, and replace every `TODO`. The
> capstone briefs are in [docs/capstones/](../README.md).
> Delete this note when you copy it.

**Honesty statement (keep this in your README):** this is a portfolio project
on public data, *inspired by* problems in TODO-domain. It is not work for, or
endorsed by, any company, and no business impact is claimed. Every number in
this README comes from a run of the code in this repository, on the split
named next to it.

## 1. Question

TODO: the business question in one or two sentences, and **who would act on
the answer** (a grid operator scheduling generation; a maintenance planner; a
replenishment buyer; a home-feed ranker; a moderation team lead). Then: what
decision changes if the model is good, and what happens if it is wrong.

## 2. Data and licence

| | |
|---|---|
| Source | TODO: official URL (not a mirror) |
| Files used | TODO: exact file names and the SHA-256 pinned in `configs/default.toml` |
| Licence / terms | TODO: the licence name and a one-line quote of the key condition. *Read the official page yourself* |
| Citation | TODO |
| Size | TODO: rows, entities, date range — measured by `make data`, not copied |

Cleaning decisions, known defects and what the data can and cannot say are in
[DATA_CARD.md](DATA_CARD.md). Raw data is **not** committed; `make data`
downloads and verifies it.

## 3. Approach

TODO: unit of prediction, horizon, label definition. The baseline(s), then the
model(s), and *why* each improvement was tried. Keep it to what you actually ran.

## 4. Evaluation protocol

TODO: the split (time cutoff / rolling origins / entity-level / leave-last-out),
why it matches how the model would be used, which split was used for model
selection and which was touched **once** for the final numbers. The metrics,
how each is defined, and how uncertainty is estimated (e.g. bootstrap over
entities or days).

## 5. Results

TBD — fill from your run. Do not paste numbers from papers or other
repositories into this table.

| Model | Split | Metric 1 | Metric 2 | 95% CI (Metric 1) | Notes |
|---|---|---|---|---|---|
| Baseline: TODO | test | TBD | TBD | TBD | |
| Model: TODO | test | TBD | TBD | TBD | |

Error analysis (slices where the model is worst, and a hypothesis for each):
TBD — fill from your run.

## 6. Limitations

TODO: what the data cannot tell you, where the evaluation is optimistic, what
would have to be true for this to work in production, and what you would do
next with more time.

## 7. How to run

```bash
python -m venv .venv && source .venv/bin/activate
make setup       # pip install -e ".[dev]"
make data        # download + checksum + validate
make train       # baseline + model -> artifacts/
make evaluate    # held-out metrics -> reports/metrics.json
make test        # unit + smoke tests, no network
make serve       # API on http://127.0.0.1:8000  (MODEL_DIR=artifacts/current)
```

Configuration lives in [configs/default.toml](configs/default.toml)
(read with the standard-library `tomllib`, Python 3.11+).

## 8. API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness — the process answers; never loads the model |
| GET | `/ready` | readiness — 200 with `model_version` if a model is loaded, else 503 |
| POST | `/predict` | 1–1000 instances; unknown fields → 422; no model → 503 |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"instances": [{"id": "a", "x1": 0.1, "x2": -0.2}]}'
# {"model_version": "current", "predictions": [{"id": "a", "prediction": ...}]}
```

TODO: replace with your real request/response schema and list each
validation rule. Say what is precomputed (batch) and what is computed per
request.

## 9. Reproducibility

- Python version and `pip freeze > requirements-lock.txt` from the run that
  produced the results table.
- Seeds: `configs/default.toml` → `project.seed`.
- Data: SHA-256 of every raw file, checked by `make data`.
- Every saved model has `metadata.json` with config, features, data hash,
  library versions and (TODO) git commit.
- CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `ruff` and
  `pytest` on toy data on every push.

Model details: [MODEL_CARD.md](MODEL_CARD.md).
