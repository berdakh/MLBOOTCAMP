# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 13 · Packaging & configuration
#
# **Part I · Chapter 13 · about 45 minutes · CPU only**
#
# In Chapters 10–12 the service ran inside a notebook, configured by whoever
# wrote the cell. In production it runs as a process that someone else starts,
# on a machine its author may never log into, configured only through its
# environment. This lab turns the project into something that can be
# installed, configured without editing code, put in a container image and
# tested by CI, and then starts it exactly as a container would.
#
# After this chapter you should be able to:
#
# 1. read a `pyproject.toml` and explain dependency *ranges* versus a *lock file*;
# 2. configure a service from environment variables, and make a bad value fail at start-up with a clear message;
# 3. write a Dockerfile and explain every line, including what it deliberately does *not* do;
# 4. choose between baking the model into the image and loading it from a registry at start-up;
# 5. start the service as a separate process, probe `/ready`, call `/predict`, read its logs, and stop it.
#
# **Lab output:** `generated/deploy/` (Dockerfile, `.dockerignore`, `serve.py`, `Makefile`,
# `ci-example.yml`, `README.md`), the same files that are committed in the repository's `deploy/`
# folder, and `artifacts/serve_logs/` with the captured output of the service you start.
#
# > **Optional external extension — not runnable in Colab, not executed in our verification.**
# > Colab cannot run Docker, and nothing in this course builds an image or deploys to a
# > host. This lab *writes and explains* the Dockerfile and CI workflow. Building and
# > running the image on your own machine is described in `deploy/README.md`.

# %% [markdown]
# **Read first:** [Chapter 13 text](README.md) explains the concepts, equations and caveats behind this lab.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: a registry with a promoted model (notebook 08) ---------------
from churnlab import bootstrap
from grading import ch13 as grade

registry = bootstrap.ensure_registry()

# %% [markdown]
# **Expected:** `versions ['v1', 'v2'], CURRENT=v1` (built in ~10-20 seconds the
# first time, loaded instantly after that).

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **`pyproject.toml`** | the file that makes a folder an installable Python package | name, version, dependencies of `churnlab` |
# | **Dependency range** | the versions a package claims to work with | `pandas>=2.1` |
# | **Lock file** | the exact versions one environment was built and verified with | `requirements-lock.txt` |
# | **Environment variable** | a named setting the operating system passes to a process | `CHURN_LOG_LEVEL=DEBUG` |
# | **Twelve-factor config** | anything that differs between deployments comes from the environment, not the code | same image in dev and production |
# | **Fail fast** | refuse to start on bad configuration instead of running wrongly | `ValueError` naming the variable |
# | **Secret** | a credential injected at run time, never stored in code or images | a database password |
# | **Container image** | a packaged file system plus a start command, built from a Dockerfile | `python:3.11-slim` + `churnlab` |
# | **Layer cache** | Docker reuses unchanged build steps, so their order matters | install dependencies before copying code |
# | **CI** | automated checks on every push | `ci-example.yml` runs the tests |

# %% [markdown]
# ## 13.1 · `pyproject.toml`: what makes `churnlab` installable
#
# Until now the notebooks put `src/` on `sys.path` by hand. A service cannot rely
# on that: it needs the package *installed*, with its dependencies declared in
# one standard file. Here is the project's:

# %%
print((ROOT / "pyproject.toml").read_text())

# %% [markdown]
# Read it in four blocks:
#
# | Block | What it says | Why it matters |
# |---|---|---|
# | `[build-system]` | *how* to build the package (setuptools ≥ 68) | `pip install .` reads this first; without it pip guesses |
# | `[project]` | name, version, `requires-python`, **`dependencies`** | the package's promise: "I work with numpy ≥ 1.26, pandas ≥ 2.1, ..." |
# | `[project.optional-dependencies]` | extras, installed with `pip install -e ".[dev]"` | test and lint tools stay out of the production image |
# | `[tool.setuptools.packages.find] where = ["src"]` | the **src layout** | the package is importable only once installed, so tests cannot pass by accidentally importing the working directory |
#
# The everyday command is `pip install -e .` (*editable*): pip installs the
# dependencies and links the package to your checkout, so edits in `src/`
# are visible without reinstalling. A container uses a normal (non-editable)
# `pip install .`, because the code in an image never changes.
#
# Is `churnlab` installed in *this* runtime, or only on `sys.path`?

# %%
from importlib import metadata

import churnlab

try:
    print("installed distribution: churnlab", metadata.version("churnlab"))
except metadata.PackageNotFoundError:
    print("churnlab is NOT installed as a distribution; it is imported from", Path(churnlab.__file__).parent)

# %% [markdown]
# **Expected:** in Colab and in the course's test runs, *not installed*: the
# setup cell puts `src/` on `sys.path` instead, which is quick but only works
# inside this process. Remember that: later in this notebook you start the
# service as a **separate** process, and it will need `PYTHONPATH` set
# explicitly (or an installed package).

# %% [markdown]
# ## 13.2 · Ranges versus a lock file
#
# `pyproject.toml` and `requirements.txt` declare **ranges** (`pandas>=2.1`):
# they say what the code is *compatible* with, so the package installs next to
# other software. A **lock file** (`requirements-lock.txt`) records the
# **exact** versions one environment was built and verified with. Libraries
# publish ranges; deployable services and reproducible results need a lock.
#
# > **Intuition.** A range is "any 2-litre pot will do" in a recipe; a lock
# > file is the exact pot, model number and all, that the recipe was tested
# > in. Cookbooks print ranges; a restaurant that must serve the same dish
# > every night uses the lock. *Where it breaks down:* a lock freezes security
# > fixes out too; it has to be refreshed deliberately, with the tests re-run.
#
# How far is this runtime from the lock?

# %%
import pandas as pd

lock = {}
for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
    if line.strip() and not line.startswith("#"):
        name, version = line.split("==")
        lock[name.strip()] = version.strip()

rows = []
for name, locked in lock.items():
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed = "(missing)"
    rows.append({"package": name, "locked": locked, "installed": installed, "matches": installed == locked})
lock_table = pd.DataFrame(rows)
print(lock_table.to_string(index=False))
print(f"\n{lock_table['matches'].sum()} of {len(lock_table)} packages match the lock exactly")

# %% [markdown]
# **Expected:** some mismatches are normal in Colab, which ships its own
# versions. Nothing breaks, because the code only needs the *ranges*. But a
# number you report (a PR-AUC, a latency) belongs to the environment you
# measured it in: say which. The Docker image below installs **from the lock**,
# so every container gets the same versions.
#
# Two honest caveats about this lock: it pins only the top-level packages (a
# full lock from `pip-compile`, `uv pip compile` or `poetry lock` also pins
# every transitive dependency, with hashes), and it includes notebook tools
# (`matplotlib`, `pytest`) that a lean production image would leave out.

# %% [markdown]
# ## 13.3 · Configuration comes from the environment
#
# The *twelve-factor* rule: anything that differs between deployments (paths,
# a pinned model version, limits, log level) is read from **environment
# variables** at start-up. Code and image stay identical across dev, staging and
# production; only the environment changes. `churnlab.settings.Settings` reads
# `CHURN_*` variables once into a frozen object:
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-config-env.svg" alt="One container image runs in dev, staging and production; each sets CHURN_ variables, Settings.from_env validates them once, and the service either starts or refuses to start naming the bad variable" width="100%">
#
# *What to notice:* the three environments share everything except a few variables, and the only two outcomes are "start with typed, frozen settings" or "do not start at all".
#
# > **Intuition.** The image is a rental car; environment variables are the
# > seat and mirror settings each driver adjusts. Same car, same engine, no
# > welding. *Where it breaks down:* a wrong mirror setting is visible at
# > once; a wrong threshold is not, which is why the settings must be
# > validated before the first request.

# %%
import inspect

from churnlab.settings import Settings

print(inspect.getsource(Settings.from_env))

# %% [markdown]
# `from_env` accepts an `environ` dictionary, so we can try environments without
# touching this process's real one:

# %%
good = {"CHURN_REGISTRY_DIR": "/models/registry", "CHURN_MODEL_VERSION": "v1",
        "CHURN_LOG_LEVEL": "warning", "CHURN_MAX_BATCH_SIZE": "500"}
print("good   :", Settings.from_env(environ=good))
print("missing:", Settings.from_env(environ={}))

for label, env in [("threshold", {"CHURN_THRESHOLD_OVERRIDE": "1.5"}),
                   ("log level", {"CHURN_LOG_LEVEL": "LOUD"}),
                   ("batch size", {"CHURN_MAX_BATCH_SIZE": "lots"})]:
    try:
        Settings.from_env(environ=env)
    except ValueError as exc:
        print(f"invalid {label:<10} -> ValueError: {exc}")

# %% [markdown]
# **Expected:** the good environment is parsed into typed values (`Path`,
# `int`, upper-cased log level); the empty environment falls back to every
# default; each invalid value raises `ValueError`.
#
# Look closely at the three messages. Two name the variable
# (`CHURN_THRESHOLD_OVERRIDE`, `CHURN_LOG_LEVEL`). The third,
# `invalid literal for int() with base 10: 'lots'`, does not: whoever reads it
# in a deployment log has to guess which of a dozen variables was wrong. A good
# configuration error names the **variable**, the **bad value** and the
# **allowed values**. Exercise 2, after the Dockerfile, asks for one that does.
#
# **Why fail at start-up?** A service that starts with a wrong threshold
# serves wrong decisions quietly for weeks. A service that refuses to start
# fails its deployment in seconds, and the old version keeps serving.
#
# **Secrets** (database passwords, API tokens) arrive the same way, injected
# by the platform's secret store at run time. They are never written into the
# image, the repository, a notebook or a log. The repository's `.gitignore`
# already excludes `.env` files and `*.key`; the `.dockerignore` below does the
# same for the image.
#
# > **Research bridge.** This is the experiment-config habit from chapter 2
# > applied to a running service: one object, built once, recorded. The
# > difference is the source: a config file you edit becomes environment
# > variables someone else sets.

# %% [markdown]
# ## 13.4 · Writing the deployment files
#
# Six files turn the package into a deployable service. The notebook writes
# them from strings into `generated/deploy/` (ignored by git), so you can read
# and change them; the repository's `deploy/` folder holds the same files,
# committed, and section 13.6 checks that the two agree.
#
# **1. The Dockerfile.** Read it once, then read the line-by-line table under it.

# %%
DOCKERFILE = r"""# syntax=docker/dockerfile:1
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
"""
print(f"{len(DOCKERFILE.splitlines())} lines")

# %% [markdown]
# | Line | What it does | Why |
# |---|---|---|
# | `FROM python:3.12-slim` | base image: Debian slim + Python 3.12 | a *pinned tag*, never `latest`: `latest` changes under you. A digest (`@sha256:...`) pins it completely |
# | `ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1` | no `.pyc` files; unbuffered stdout | logs reach the platform immediately instead of sitting in a buffer when the process dies |
# | `WORKDIR /app` | working directory for everything after | relative paths are predictable |
# | `COPY requirements-lock.txt .` then `RUN pip install --no-cache-dir -r ...` | install exact, verified versions | **layer caching**: this slow step re-runs only when the lock changes. `--no-cache-dir` keeps pip's download cache (often 100+ MB) out of the image |
# | `COPY pyproject.toml README.md ./`, `COPY src/ src/`, `RUN pip install --no-deps .` | install `churnlab` itself | code changes often, so it comes *after* the dependency layer. `README.md` is copied because `pyproject.toml` names it as the package readme |
# | `COPY deploy/serve.py .` | the entry point | it builds the app from environment variables |
# | *(commented)* `COPY artifacts/registry/v1/ ...` | the "bake the model in" alternative | discussed below and in Exercise 3 |
# | `RUN useradd ... app` / `USER app` | run as an unprivileged user | if the service is compromised, the attacker is not root inside the container |
# | `ENV CHURN_REGISTRY_DIR=... CHURN_LOG_LEVEL=...` | **defaults** for configuration | documented in one place; overridden per deployment (`docker run -e ...`). No secrets here: `ENV` values are visible to anyone with the image |
# | `EXPOSE 8000` | documents the port | documentation only; publishing it is `docker run -p 8000:8000` |
# | `HEALTHCHECK ... urllib ... /health` | Docker marks the container unhealthy if `/health` stops answering | uses Python's standard library because the slim image has no `curl`. It probes **liveness**, not readiness (chapter 14) |
# | `CMD ["uvicorn", "serve:app", ..., "--workers", "2"]` | start the server | exec form (a JSON list): uvicorn is PID 1 and receives the platform's stop signal directly. `--workers 2` = two processes |
#
# **What `--workers` really means.** Each worker is a separate process that
# imports `serve.py` and loads its own copy of the model, so memory grows with
# the worker count, and CPU-bound prediction gets real parallelism (one
# process's GIL does not block another). Two consequences recur in later chapters:
# `/metrics` counts only what *one* worker saw, and `POST /admin/reload` reloads
# only the worker that received it. On Kubernetes the usual pattern is one
# worker per container and more replicas; on a single VM, roughly one worker
# per CPU core.
#
# > **Intuition.** Baking the model in is printing the map into the car's
# > manual; loading from the registry is a satnav that downloads the current
# > map at start-up. The manual never fails to load but is out of date the
# > day a road changes. *Where it breaks down:* a satnav that cannot reach its
# > server must still have *something* to show, so a registry-loading service
# > needs a clear "not ready" state (chapter 14).
#
# **Bake the model in, or load it from the registry?** The Dockerfile above
# mounts the registry at run time. The alternative bakes one bundle into the
# image:
#
# | | Bake into the image | Load from a registry at start-up |
# |---|---|---|
# | What is deployed | one immutable image = code + model | code image + a pointer (`CURRENT`) |
# | Rollback | redeploy the previous image (minutes) | rewrite the pointer and reload (seconds) |
# | Reproducibility | perfect: the image digest identifies everything | needs the registry to keep old bundles immutable |
# | Start-up dependency | none | registry must be reachable, or the pod is not ready |
# | Image size / build time | grows with the model; rebuild per model | small; build only when code changes |
# | Good fit | small models, rare retrains, strict audit | frequent retrains, many models, fast rollback |
#
# **2. `.dockerignore`.** Everything in the build context is sent to the Docker
# daemon and can end up in an image. This file keeps secrets, data and bulk
# out. Docker reads it from the **root of the build context**, which is why the
# build instructions copy it there first.

# %%
DOCKERIGNORE = """# Copy to the repository root before building (Docker reads .dockerignore from the
# root of the build context):  cp deploy/.dockerignore .dockerignore

# Secrets must never enter an image.
.env
.env.*
*.key
secrets.json

# Version control, caches and environments.
.git
.github
**/__pycache__
**/*.py[cod]
.venv
venv
.pytest_cache
.ruff_cache
**/.ipynb_checkpoints

# Course material the service does not need.
notebooks
notebook_src
docs
capstones
solutions
tests
tools
data
generated

# Models travel through the registry, mounted at run time. To bake one bundle
# into the image instead, keep this line and add an exception such as:
#   !artifacts/registry/v1
artifacts
*.joblib
"""

# %% [markdown]
# **3. `serve.py`.** The production entry point is tiny on purpose: build
# `Settings` from the environment, build the app, done. Two details matter:
# a configuration error stops the process with exit code 2 and a one-line
# message, and the JSON logs go to **stdout**, where container platforms
# collect them.

# %%
SERVE_PY = r'''"""Production entry point for the TaskFlow churn API.

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
'''

# %% [markdown]
# **4. A `Makefile`** with the commands people actually type, so nobody has to
# remember the flags. Make requires a TAB before each command; the string below
# uses four spaces for readability and converts them.

# %%
MAKEFILE = """# Reference targets for the deployment files. Run from the repository root:
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
""".replace("\n    ", "\n\t")

# %% [markdown]
# **5. A CI workflow.** Continuous integration runs the checks on every push and
# pull request, on a clean machine, so "works on my laptop" stops being the
# test. Three jobs: lint and unit tests on two Python versions; a smoke test
# that starts the real service and probes it; and (optional) building the image.

# %%
CI_YAML = r"""# Reference CI workflow (GitHub Actions). It is NOT active in this repository:
# GitHub only runs workflows in .github/workflows/. To use it in your own
# repository:  cp deploy/ci-example.yml .github/workflows/ci.yml
name: ci

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Lint
        run: ruff check src tests tools solutions
      - name: Notebooks match their sources
        run: python tools/nbbuild.py --check
      - name: Unit tests
        run: python -m pytest -q

  smoke:
    needs: test
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - name: Install the locked environment and the package
        run: |
          pip install -r requirements-lock.txt
          pip install --no-deps -e .
      - name: Build a registry
        run: python -c "from churnlab.bootstrap import ensure_registry; ensure_registry()"
      - name: Start the service and probe it
        working-directory: deploy
        env:
          CHURN_REGISTRY_DIR: ${{ github.workspace }}/artifacts/registry
        run: |
          python -m uvicorn serve:app --port 8000 > server.log 2>&1 &
          for i in $(seq 1 30); do
            curl -sf http://127.0.0.1:8000/ready && break
            sleep 1
          done
          curl -sf http://127.0.0.1:8000/ready
          curl -sf http://127.0.0.1:8000/health
      - name: Server log
        if: always()
        run: cat deploy/server.log

  image:
    # Optional: builds the image to prove the Dockerfile works. Pushing it to a
    # registry needs credentials (repository secrets) and is left out on purpose.
    needs: test
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          cp deploy/.dockerignore .dockerignore
          docker build -f deploy/Dockerfile -t churn-api:${{ github.sha }} .
"""

# %% [markdown]
# **6. `README.md`** for the folder: how to run, build and deploy, with the
# optional parts labelled as such.

# %%
README_MD = """# deploy/ · reference deployment files

These files are **generated by notebook 12** (`notebooks/12_packaging_and_configuration.ipynb`)
and committed here as a reference. The notebook checks that what it generates
matches this folder. Edit the notebook source, not these files.

| File | Purpose |
|---|---|
| `serve.py` | production entry point: `Settings.from_env()` -> `create_app()`; exposes `app` for `uvicorn serve:app` |
| `Dockerfile` | the container image: python:3.12-slim, locked dependencies, non-root user, health check |
| `.dockerignore` | keeps secrets, data, notebooks and models out of the build context |
| `Makefile` | `make -f deploy/Makefile test / serve / smoke / image / run` |
| `ci-example.yml` | a GitHub Actions workflow: lint, unit tests, a smoke test of the real service, an image build |

## 1 · Run locally, without Docker (runs anywhere, including Colab)

```bash
python -c "import sys; sys.path.insert(0, 'src'); from churnlab.bootstrap import ensure_registry; ensure_registry()"
cd deploy
PYTHONPATH=../src CHURN_REGISTRY_DIR=../artifacts/registry python -m uvicorn serve:app --port 8000
# in another terminal
curl http://127.0.0.1:8000/ready
```

Notebook 12 does exactly this in a subprocess and stops it again.

## 2 · Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CHURN_REGISTRY_DIR` | `artifacts/registry` (`/models/registry` in the image) | where the model registry is |
| `CHURN_MODEL_VERSION` | unset = follow `CURRENT` | pin one version, e.g. `v1` |
| `CHURN_MAX_BATCH_SIZE` | `1000` | larger `/predict/batch` requests get HTTP 413 |
| `CHURN_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `CHURN_THRESHOLD_OVERRIDE` | unset = the model's own threshold | a number in (0, 1) |
| `CHURN_SERVICE_NAME` | `churn-api` | logger name |

An invalid value stops the process at start-up with exit code 2 and a
`configuration error: ...` line on stderr. Secrets never go in this table, the
image or the repository: inject them from your platform's secret store.

## 3 · Optional external extension: build and run the image

Not runnable in Colab, and **not executed in the course's verification**. You
need Docker on your own machine.

```bash
# from the repository root
cp deploy/.dockerignore .dockerignore
docker build -f deploy/Dockerfile -t churn-api:1.0.0 .
docker run --rm -p 8000:8000 -v "$PWD/artifacts/registry:/models/registry:ro" churn-api:1.0.0
curl http://127.0.0.1:8000/ready
```

The registry is mounted read-only: the container reads `CURRENT` and the
bundles; promotions and rollbacks are done outside it, followed by
`POST /admin/reload`. With `--workers 2` each worker holds its own model and
its own `/metrics`, and a reload reaches only the worker that receives it.
Restart the container (or run one worker per container) to switch every
worker together.

## 4 · Optional external extension: deploy to a host

Not part of the course and not verified. Any platform that runs a container
image works (a VM with Docker, Cloud Run, Azure Container Apps, Fly.io,
Render, Kubernetes). The steps are always the same:

1. push the image to a container registry your platform can pull from;
2. give the service the model: mount or sync the model registry, or bake one
   bundle into the image (see notebook 12 for the trade-off);
3. set the `CHURN_*` variables; inject any secrets from the platform's secret store;
4. point the platform's **liveness** probe at `/health` and its **readiness** probe at `/ready`;
5. put authentication in front of `POST /admin/reload` (the reference app has none).

## 5 · CI

`ci-example.yml` is not active here. Copy it to `.github/workflows/ci.yml` in
your own repository to run it on every push and pull request.
"""

# %% [markdown]
# Now write all six files:

# %%
DEPLOY_FILES = {
    "Dockerfile": DOCKERFILE,
    ".dockerignore": DOCKERIGNORE,
    "serve.py": SERVE_PY,
    "Makefile": MAKEFILE,
    "ci-example.yml": CI_YAML,
    "README.md": README_MD,
}
GEN = ROOT / "generated" / "deploy"
GEN.mkdir(parents=True, exist_ok=True)
for name, text in DEPLOY_FILES.items():
    (GEN / name).write_text(text, encoding="utf-8")
    print(f"wrote {(GEN / name).relative_to(ROOT)}  ({len(text.splitlines())} lines)")

try:  # a syntax check of the workflow, if PyYAML is available (it is in Colab)
    import yaml

    workflow = yaml.safe_load(CI_YAML)
    print("ci-example.yml parses; jobs:", list(workflow["jobs"]))
except ImportError:
    print("PyYAML not installed; skipped the workflow syntax check")
compile(SERVE_PY, "serve.py", "exec")
print("serve.py compiles")

# %% [markdown]
# **Expected:** six files written under `generated/deploy/`, the workflow's
# three jobs `['test', 'smoke', 'image']` (or a note that PyYAML is missing),
# and `serve.py compiles`.
#
# A Dockerfile reviewer looks for a handful of well-known problems. Exercise 1
# automates that review.

# %% [markdown]
# > **Exercise 1.** Write `dockerfile_problems(text)` that returns a **sorted list
# > without duplicates** of the problem codes found in a Dockerfile's text,
# > from this fixed set:
# >
# > | Code | Report it when |
# > |---|---|
# > | `"latest_tag"` | a `FROM` image has no tag, or the tag `latest` (a digest `@sha256:...` counts as pinned; `FROM scratch` is fine) |
# > | `"root_user"` | there is no `USER` instruction, or the **last** one is `root` or `0` |
# > | `"copies_env_file"` | a `COPY` or `ADD` instruction names a `.env` file (`.env`, `.env.prod`, `config/.env`, ...) |
# > | `"no_healthcheck"` | there is no `HEALTHCHECK` instruction (`HEALTHCHECK NONE` also counts as none) |
# > | `"pip_cache"` | a `RUN` instruction contains `pip install` without `--no-cache-dir` |
# >
# > Instruction names are case-insensitive. Ignore comment lines (starting
# > with `#`). Join lines that end with a backslash `\` to the next line before
# > checking, because one instruction can span several lines. `FROM` may carry
# > `--platform=...` and `AS name`. The Dockerfile you just generated should
# > give `[]`.

# %%
def dockerfile_problems(text: str) -> list[str]:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(dockerfile_problems)

# %% [markdown]
# > **Exercise 2.** Back to configuration. Write `settings_from_env(env)` for a launcher script that
# > needs three settings. It takes a dictionary of environment variables and
# > returns `{"port": int, "workers": int, "log_level": str}`.
# >
# > | Variable | Default | Valid values |
# > |---|---|---|
# > | `CHURN_PORT` | `8000` | an integer from 1 to 65535 |
# > | `CHURN_WORKERS` | `1` | an integer from 1 to 64 |
# > | `CHURN_LOG_LEVEL` | `"INFO"` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, any letter case; return it upper-cased |
# >
# > A missing variable *or an empty string* means "use the default". Other keys
# > in `env` are ignored. Any invalid value (not an integer, out of range,
# > unknown level) raises `ValueError` whose message contains the **variable
# > name**: section 13.3 showed why that matters.

# %%
def settings_from_env(env: dict) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(settings_from_env)

# %% [markdown]
# ## 13.5 · Start the service the way a container would
#
# Docker is not available here, but we can do what the container's `CMD` does:
# start `uvicorn serve:app` as a **separate process**, configured only by
# environment variables. That is the real test of `serve.py`: it cannot see any
# variable from this notebook unless it is in the process environment.
#
# The child process needs three things from its environment: where the
# registry is, a port, and `PYTHONPATH` (because `churnlab` is not installed,
# section 13.1). Its stdout and stderr go to files, so a chatty server can
# never block on a full pipe.

# %%
import socket
import time

import httpx


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_service(env_overrides: dict, log_dir: Path):
    """Start `uvicorn serve:app` from generated/deploy in a child process."""
    port = free_port()
    log_dir.mkdir(parents=True, exist_ok=True)
    out, err = open(log_dir / "stdout.log", "w"), open(log_dir / "stderr.log", "w")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **env_overrides}
    cmd = [sys.executable, "-m", "uvicorn", "serve:app", "--host", "127.0.0.1",
           "--port", str(port), "--log-level", "warning"]
    proc = subprocess.Popen(cmd, cwd=GEN, env=env, stdout=out, stderr=err)
    return proc, f"http://127.0.0.1:{port}", (out, err)


def wait_until_ready(proc, base_url: str, timeout: float = 60.0) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            if httpx.get(base_url + "/ready", timeout=1.0).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise TimeoutError("server did not become ready")


LOGS = ART / "serve_logs"
proc, base_url, files = start_service({"CHURN_REGISTRY_DIR": str(registry.root),
                                       "CHURN_LOG_LEVEL": "INFO"}, LOGS / "good")
try:
    seconds = wait_until_ready(proc, base_url)
    print(f"pid {proc.pid} ready at {base_url} after {seconds:.1f}s")
    print("GET /health ->", httpx.get(base_url + "/health").json())
    print("GET /ready  ->", httpx.get(base_url + "/ready").json())

    split = bootstrap.ensure_split()
    from churnlab.api import example_payload

    payload = example_payload(split.valid.iloc[0])
    r = httpx.post(base_url + "/predict", json=payload, headers={"x-request-id": "nb12-demo-1"}, timeout=5)
    print("POST /predict ->", r.status_code, r.json())
    print("response headers: x-request-id =", r.headers["x-request-id"],
          "| x-model-version =", r.headers["x-model-version"])
finally:
    proc.terminate()
    proc.wait(timeout=10)
    for fh in files:
        fh.close()
print("exit code after terminate:", proc.returncode)

# %% [markdown]
# **Expected:** ready after a few seconds (most of it is importing pandas and
# scikit-learn and loading the model; this is the service's **cold start**);
# `/health` → `{"status": "ok", ...}`, `/ready` → `{"status": "ready",
# "model_version": "v1", ...}`, one prediction with `model_version` `v1`, and
# the request id you sent echoed back. The exit code after `terminate()` is
# negative (`-15`, killed by SIGTERM) on Linux and macOS.
#
# The service's stdout is its log. Each line is one JSON object:

# %%
import json

stdout_lines = (LOGS / "good" / "stdout.log").read_text().splitlines()
for line in stdout_lines:
    print(line)
events = [json.loads(line) for line in stdout_lines if line.startswith("{")]
print("\nthe /predict request, as the platform would see it:")
print([e for e in events if e.get("request_id") == "nb12-demo-1"])

# %% [markdown]
# **Expected:** a `model_loaded` event with `model_version: v1`, then one
# `request` event per call with `path`, `status`, `latency_ms` and
# `model_version`. The request id you chose is in the log line: that is how a
# caller's complaint ("request nb12-demo-1 was slow") is found among millions
# of lines. Chapter 14 analyses these logs properly.
#
# Now a deployment with a mistake in it: someone sets the threshold override to
# `1.5` (meaning 15%?) instead of `0.15`.

# %%
proc, base_url, files = start_service({"CHURN_REGISTRY_DIR": str(registry.root),
                                       "CHURN_THRESHOLD_OVERRIDE": "1.5"}, LOGS / "bad")
try:
    code = proc.wait(timeout=60)
finally:
    for fh in files:
        fh.close()
print("exit code:", code)
print("stderr   :", (LOGS / "bad" / "stderr.log").read_text().strip().splitlines()[-1])

# %% [markdown]
# **Expected:** `exit code: 2` and `configuration error: CHURN_THRESHOLD_OVERRIDE
# must be in (0, 1), got 1.5`. The process never listened on its port. On a
# real platform the new version fails its rollout and the old one keeps
# serving, which is the desired outcome for a configuration typo.

# %% [markdown]
# ## 13.6 · Are the committed files the generated ones?
#
# The repository's `deploy/` folder is meant to be exactly what this notebook
# generates. A check that compares them catches the day someone edits one
# copy and forgets the other (CI could run the same comparison).

# %%
import difflib

COMMITTED = ROOT / "deploy"
all_match = True
for name, text in DEPLOY_FILES.items():
    committed = COMMITTED / name
    if not committed.exists():
        print(f"{name:<15} MISSING from deploy/")
        all_match = False
        continue
    same = committed.read_text(encoding="utf-8") == text
    all_match &= same
    print(f"{name:<15} {'matches' if same else 'DIFFERS'}")
    if not same:
        diff = difflib.unified_diff(committed.read_text(encoding="utf-8").splitlines(), text.splitlines(),
                                    "deploy/" + name, "generated/deploy/" + name, lineterm="", n=1)
        print("\n".join(list(diff)[:20]))
print("\ngenerated files match deploy/:", all_match)

# %% [markdown]
# **Expected:** every file `matches` and the last line is `True`. If you edited
# a string above, you will see a short diff instead. That is the point of the
# check.

# %% [markdown]
# > **Exercise 3 (written).** For *this* project (a churn model retrained
# > roughly monthly, scored by a customer-success tool, rollback must be fast),
# > would you bake the model into the image or load it from the registry at
# > start-up? Choose one and justify it in four to six sentences: mention
# > rollback time, reproducibility/audit, what happens when the registry is
# > unreachable, and what you would do to cover the weakness of your choice.
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: No module named 'churnlab'` in the server's `stderr.log` | the child process does not inherit this notebook's `sys.path` | pass `PYTHONPATH=<repo>/src` in its environment (as `start_service` does) or `pip install -e .` |
# | `server exited early with code 2` | a `CHURN_*` value is invalid | read `stderr.log`: the line starting `configuration error:` names the variable |
# | `/ready` stays 503 | `CHURN_REGISTRY_DIR` points at an empty or wrong folder | check the path is absolute (the child runs in `generated/deploy/`, so relative paths resolve there) |
# | `address already in use` | a previous server is still running | re-run the cell (it picks a free port); stop old ones with `proc.terminate()` or restart the runtime |
# | The cell hangs | the child is waiting on a full output pipe | send its output to files, as here, never to `subprocess.PIPE` without reading it |
# | `docker: command not found` | Colab has no Docker | expected: building the image is an optional external extension on your own machine |
# | Generated files `DIFFERS` from `deploy/` | a string in section 13.4 was edited | fine while experimenting; to update the reference, copy `generated/deploy/*` into `deploy/` |

# %% [markdown]
# ## Checkpoint
#
# Answer each question aloud, without notes, in under a minute:
#
# 1. What is the difference between the dependency list in `pyproject.toml` and `requirements-lock.txt`, and which does the Docker image install from?
# 2. Why are configuration values read from environment variables, and why should an invalid value stop the process at start-up?
# 3. Walk through the Dockerfile: why is the dependency layer copied before the source code, and why is there a `USER` line?
# 4. What changes when you run uvicorn with `--workers 4`, and what does it break about `/metrics` and `/admin/reload`?
# 5. Bake the model into the image or load it from a registry: give one strong argument for each.
# 6. Where do secrets come from in this design, and name two places they must never appear.
#
# **Next chapter:** [Chapter 14 · Health, logs & latency](../ch14-health-logs-and-latency/notebook.ipynb) — with the service running, we ask whether it is healthy, what it logs, and how fast it is.

