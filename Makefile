# Course maintenance tasks. Readers do not need these; maintainers do.

.PHONY: help install test notebooks check lint run-notebooks diagrams data clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install Part I + maintainer requirements (Parts II-III: requirements-part2/3.txt)
	pip install -r requirements.txt -r requirements-dev.txt

test:  ## Unit tests, graders vs reference solutions, teaching-content checks (no network)
	python -m pytest

notebooks:  ## Rebuild every chapter's notebook.ipynb from its notebook.py
	python tools/nbbuild.py

diagrams:  ## Regenerate figures/*.svg and appendices/figures.md
	python tools/make_diagrams.py

data:  ## Regenerate data/raw (then update RAW_SHA256 in src/churnlab/data.py)
	python tools/make_dataset.py

check:  ## Everything CI checks: sync, structure, links, tests
	python tools/nbbuild.py --check
	python tools/make_diagrams.py --check
	python tools/check_notebooks.py
	python tools/check_links.py
	python -m pytest

run-notebooks:  ## Execute every Part I lab in a fresh artifact dir (use --part for others)
	python tools/run_notebooks.py --part part1-foundations --report verification-log.md

lint:  ## Lint the packages, tests, tools and reference solutions
	ruff check src tests tools part1-foundations part2-ml-at-scale part3-deep-learning-systems part4-capstones

clean:  ## Remove caches and generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache artifacts generated
