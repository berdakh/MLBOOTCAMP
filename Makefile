# Course maintenance tasks. Learners do not need these; maintainers do.

.PHONY: help install test notebooks check lint run-notebooks diagrams data clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime + maintainer requirements
	pip install -r requirements.txt -r requirements-dev.txt

test:  ## Unit tests for churnlab, graders and solutions (no network)
	python -m pytest

notebooks:  ## Rebuild notebooks/*.ipynb from notebook_src/*.py
	python tools/nbbuild.py

diagrams:  ## Regenerate docs/assets/diagrams/*.svg
	python tools/make_diagrams.py

data:  ## Regenerate data/raw (then update RAW_SHA256 in src/churnlab/data.py)
	python tools/make_dataset.py

check:  ## Everything CI checks: sync, structure, links, tests
	python tools/nbbuild.py --check
	python tools/make_diagrams.py --check
	python tools/check_notebooks.py
	python tools/check_links.py
	python -m pytest

run-notebooks:  ## Execute every notebook, each in a fresh artifact dir (slow: ~10 min)
	python tools/run_notebooks.py --report docs/guides/verification-log.md

lint:  ## Lint the package, tests and tools
	ruff check src tests tools solutions

clean:  ## Remove caches and generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache artifacts generated
