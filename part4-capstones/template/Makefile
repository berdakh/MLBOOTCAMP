# Every target is one command a reviewer can run. Keep them working.
PY ?= python
CONFIG ?= configs/default.toml

.PHONY: setup data train evaluate test lint serve clean

setup:            ## install the package and dev tools into the active environment
	$(PY) -m pip install -e ".[dev]"

data:             ## download (checksummed) and validate the raw data -> data/
	$(PY) -m capstone.cli data --config $(CONFIG)

train:            ## fit baseline + model on the training split -> artifacts/
	$(PY) -m capstone.cli train --config $(CONFIG)

evaluate:         ## score on the held-out split -> reports/metrics.json
	$(PY) -m capstone.cli evaluate --config $(CONFIG)

test:             ## unit + smoke tests (no network, seconds)
	$(PY) -m pytest -q

lint:
	ruff check .

serve:            ## run the API locally on :8000 (needs `make train` first)
	$(PY) -m uvicorn capstone.api:create_app --factory --host 127.0.0.1 --port 8000

clean:
	rm -rf artifacts/* reports/* .pytest_cache .ruff_cache
