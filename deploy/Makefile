# Reference targets for the deployment files. Run from the repository root:
#   make -f deploy/Makefile <target>
# Targets marked [docker] are optional external extensions: they need Docker.

IMAGE ?= churn-api
TAG ?= 1.0.0
PORT ?= 8000
REGISTRY_DIR ?= $(CURDIR)/artifacts/registry

.PHONY: test serve smoke image run

test:  ## unit tests, as CI runs them
	python -m pytest -q

serve:  ## run the API locally, no Docker
	cd deploy && PYTHONPATH=../src CHURN_REGISTRY_DIR=$(REGISTRY_DIR) python -m uvicorn serve:app --port $(PORT)

smoke:  ## probe a running service
	python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:$(PORT)/ready', timeout=5).read().decode())"

image:  ## [docker] build the image
	cp deploy/.dockerignore .dockerignore
	docker build -f deploy/Dockerfile -t $(IMAGE):$(TAG) .

run:  ## [docker] run the image with the registry mounted read-only
	docker run --rm -p $(PORT):8000 -v $(REGISTRY_DIR):/models/registry:ro $(IMAGE):$(TAG)
