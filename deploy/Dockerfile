# syntax=docker/dockerfile:1
# Reference image for the TaskFlow churn API.
#
# OPTIONAL EXTERNAL EXTENSION: needs Docker on your own machine or in CI.
# Not runnable in Colab, and not executed in the course's verification.
#
# Build from the REPOSITORY ROOT (the build context must contain src/):
#   cp deploy/.dockerignore .dockerignore
#   docker build -f deploy/Dockerfile -t churn-api:1.0.0 .

# A small official base image, pinned by tag. For byte-for-byte rebuilds pin
# the digest as well:  FROM python:3.12-slim@sha256:<digest>
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, from the lock file: this slow layer is cached and only
# rebuilt when requirements-lock.txt changes, not on every code edit.
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Then the package itself (dependencies already satisfied, so --no-deps).
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .
COPY deploy/serve.py .

# The model is NOT baked in: the registry is mounted at run time
# (docker run -v <registry>:/models/registry:ro ...). To bake a model instead,
# copy ONE immutable bundle and pin it:
#   COPY artifacts/registry/v1/ /models/registry/v1/
#   ENV CHURN_MODEL_VERSION=v1

# Never run as root inside the container.
RUN useradd --create-home --uid 10001 app
USER app

# Defaults only; every deployment overrides what it needs with -e / its platform.
ENV CHURN_REGISTRY_DIR=/models/registry \
    CHURN_LOG_LEVEL=INFO \
    CHURN_MAX_BATCH_SIZE=1000

EXPOSE 8000

# Liveness probe for Docker / Compose (Kubernetes ignores this and uses its own probes).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Two worker processes, each with its own copy of the model. See deploy/README.md.
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
