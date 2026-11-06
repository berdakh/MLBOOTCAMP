# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/01_reproducible_project.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 01 · A reproducible ML project
#
# **Day 1 · ~40 minutes · CPU only**
#
# In research, a result is usually reproduced by the person who produced it, on
# the machine that produced it, soon after. In industry it has to be reproduced
# by someone else, on another machine, months later, often while something is
# broken. This notebook sets up the habits that make that possible, using the
# project you will carry through all five days.
#
# By the end you will be able to:
#
# 1. explain the four things that must be pinned for a result to reproduce (code, data, environment, randomness);
# 2. explain why the project lives in a package (`src/churnlab/`) and not in notebooks;
# 3. make a configuration hashable and give every run an identity;
# 4. write a run record that a colleague could use to reproduce your number;
# 5. keep your work across temporary Colab sessions.
#
# **You will produce:** `artifacts/runs/<run_id>.json`, a run record for a first small experiment.

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
# --- Catch up: nothing from earlier notebooks is needed here ----------------
from churnlab.grading import nb01 as grade

# %% [markdown]
# ## 1.1 · What has to be pinned
#
# A number reproduces only if **all four** of these are the same as when it was
# produced. Miss one and you will eventually get a different number with no
# idea why.
#
# | Pin | How this course pins it | What goes wrong without it |
# |---|---|---|
# | **Code** | git commit of the repository | "it worked last week" |
# | **Data** | SHA-256 of every raw file, fingerprint of every derived table | silently different rows after a re-export |
# | **Environment** | `requirements.txt` (ranges) + `requirements-lock.txt` (exact) | a library default changes between versions |
# | **Randomness** | explicit `random_state` / `default_rng(seed)` everywhere | a "better" model that was a lucky seed |
#
# There is a fifth, which no tool fixes: **the decisions**. Why this split? Why
# this threshold? Those go in writing, next to the result - you will do that on
# Day 2 (model card) and Day 5 (README).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/architecture.svg" alt="The project: raw extract, validation, SQL features, split, pipeline, registry, service, monitoring" width="100%">

# %% [markdown]
# ## 1.2 · Why a package, not a notebook
#
# Notebooks are excellent for *explaining* and *exploring*. They are poor
# places to *keep* code: cells run out of order, functions cannot be imported
# by a service or a test, and diffs are JSON. So the course keeps the real code
# in `src/churnlab/` and the notebooks import it. You will read each module the
# day it becomes relevant.

# %%
for path in sorted((ROOT / "src" / "churnlab").glob("*.py")):
    lines = path.read_text().splitlines()
    first_doc = next((ln.strip(' "') for ln in lines if ln.strip().strip('"')), "")
    print(f"{path.name:<16}{len(lines):>5} lines   {first_doc[:70]}")

# %% [markdown]
# **Expected:** about 16 modules, each a few dozen to a few hundred lines, each
# starting with a one-line summary. Nothing is hidden: `api.py` *is* the
# service; `features.py` *is* the SQL.
#
# The rest of the layout follows a convention most Python teams will recognise:
#
# ```
# src/churnlab/    importable code (the "src layout" stops tests passing by accident
#                  because the working directory happens to be on sys.path)
# tests/           pytest tests for that code - run in CI on every push
# notebook_src/    notebook sources as plain .py; notebooks/ is generated from them
# data/raw/        immutable inputs, checksummed
# artifacts/       everything the code produces (ignored by git - see .gitignore)
# ```

# %%
print((ROOT / ".gitignore").read_text().split("# --- Secrets")[0].split("# --- Course artifacts")[1])

# %% [markdown]
# Why are models **not** committed? They are large, they are reproducible from
# code + data + environment, and a pickle executes code when loaded. Ship them
# through a registry (Day 2), not through git.

# %% [markdown]
# ## 1.3 · The environment is part of the result
#
# Two files, two jobs:
#
# * `requirements.txt` - **lower bounds**. Easy to install anywhere, including
#   Colab, which already has most of it.
# * `requirements-lock.txt` - **exact versions** the notebooks were executed
#   with. Use it when you need the same numbers, not just working code.
#
# Record what you actually ran with. It costs one cell:

# %%
import platform

import duckdb
import fastapi
import numpy as np
import pandas as pd
import sklearn


def environment_record() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(terse=True),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "duckdb": duckdb.__version__,
        "fastapi": fastapi.__version__,
        "colab": "google.colab" in sys.modules,
    }


environment_record()

# %% [markdown]
# **Expected:** a dictionary of versions. Yours may differ from a classmate's;
# that is exactly why it is recorded.
#
# And the code version: the git commit. In Colab the setup cell made a shallow
# clone, so this works there too.

# %%
def git_commit(root: Path = ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


git_commit()

# %% [markdown]
# A `-dirty` suffix means uncommitted changes: the commit alone no longer
# describes the code that ran. That is fine while exploring and not fine for a
# number you report.

# %% [markdown]
# ## 1.4 · Randomness: pin it, then check it is pinned
#
# Two runs of the "same" experiment, with and without a seed:

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 8))
y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=1.0, size=600) > 0).astype(int)


def experiment(seed):
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    return cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()


print("no seed  :", [round(float(experiment(None)), 4) for _ in range(3)])
print("seed=1   :", [round(float(experiment(1)), 4) for _ in range(3)])
print("seeds 0-4:", [round(float(experiment(s)), 4) for s in range(5)])

# %% [markdown]
# **Expected:** the unseeded runs differ in the third or fourth decimal; the
# seeded runs are identical; and the spread *across* seeds is often as large
# as the "improvements" people report.
#
# So pinning a seed makes a run **repeatable**; it does not make a result
# **reliable**. When two models differ by less than the across-seed spread, you
# have not shown that one is better. Day 2 does this properly with confidence
# intervals.
#
# > 🧠 **Research bridge.** This is the same discipline as reporting
# > cross-subject EEG results as mean ± spread over subjects rather than one
# > lucky split. The habit transfers directly; the vocabulary changes.

# %% [markdown]
# ## 1.5 · Configuration as data, and a run identity
#
# Every knob that changes a result belongs in **one** configuration object,
# not scattered through cells. A config that is plain data can be saved,
# compared and hashed, and the hash becomes the run's name.

# %%
import hashlib
import json

config = {
    "model": "logreg",
    "C": 1.0,
    "features": ["sessions_4w", "tenure_days", "failed_payments_90d"],
    "train_end": "2024-04-01",
    "seed": 0,
}

print(json.dumps(config, sort_keys=True))
print(hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12])

# %% [markdown]
# `sort_keys=True` matters: `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are the
# same configuration and must get the same identity.

# %% [markdown]
# > **Exercise 1.** Write `config_hash(config)` that returns the first 12
# > hexadecimal characters of the SHA-256 of the configuration. It must be the
# > same for dictionaries that differ only in key order (including nested
# > dictionaries), and different when any value changes.

# %%
def config_hash(config: dict) -> str:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(config_hash)

# %% [markdown]
# ## 1.6 · A first experiment, with a run record
#
# Now a real (small) experiment on the project data. The feature table is
# built by SQL you will write yourself in notebook 03; here,
# `churnlab.bootstrap` builds it for you in a few seconds so this notebook can
# stand alone.

# %%
from churnlab import bootstrap
from churnlab.artifacts import dataframe_fingerprint
from churnlab.splits import time_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

features = bootstrap.ensure_features()
split = time_split(features, train_end=pd.Timestamp(config["train_end"]))

cols = config["features"]
model = LogisticRegression(C=config["C"], max_iter=1000)
model.fit(split.train[cols], split.train["churned_30d"])
valid_scores = model.predict_proba(split.valid[cols])[:, 1]
metrics = {
    "valid_pr_auc": round(float(average_precision_score(split.valid["churned_30d"], valid_scores)), 4),
    "valid_positive_rate": round(float(split.valid["churned_30d"].mean()), 4),
}
metrics

# %% [markdown]
# **Expected:** PR-AUC around 0.1 on validation, against a positive rate near
# 0.05. A random ranking would score about the positive rate, so three raw
# features already carry signal. (PR-AUC and why it, not accuracy, is the
# headline metric are covered on Day 2.)
#
# A number like this is only useful with its context. Here is the record that
# makes it reproducible:

# %%
from datetime import datetime, timezone

run_record = {
    "run_id": None,  # filled in below from the config hash
    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_commit": git_commit(),
    "config": config,
    "data": {
        "train_fingerprint": dataframe_fingerprint(split.train[cols + ["churned_30d"]]),
        "train_rows": len(split.train),
        "valid_rows": len(split.valid),
    },
    "environment": environment_record(),
    "metrics": metrics,
}
run_record["run_id"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

runs_dir = ART / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)
(runs_dir / f"{run_record['run_id']}.json").write_text(json.dumps(run_record, indent=2))
print(json.dumps(run_record, indent=2)[:900])

# %% [markdown]
# This is all an experiment tracker (MLflow, Weights & Biases) does at its
# core: config, code version, data version, environment, metrics, under an ID.
# Writing one by hand once makes the tools obvious when you meet them.

# %% [markdown]
# > **Exercise 2.** Write `record_run(directory, config, metrics)` that writes
# > `<directory>/<config_hash>.json` and returns its path. The JSON must contain
# > the keys `run_id` (= your `config_hash(config)`), `config`, `metrics`,
# > `created_at` (ISO-8601 string) and `environment` (a dict containing at least
# > `python` and `sklearn` versions). Re-running with the same config must
# > overwrite the same file, not create a new one.

# %%
def record_run(directory: Path, config: dict, metrics: dict) -> Path:
    # YOUR CODE HERE (you may call config_hash and environment_record)
    raise NotImplementedError


grade.ex2(record_run)

# %% [markdown]
# ## 1.7 · Keeping work across Colab sessions
#
# A Colab runtime is temporary: after a disconnect, everything under
# `/content` is gone, including `artifacts/`. You have three defences, used
# throughout the course:
#
# 1. **Google Drive.** Set `USE_DRIVE = True` in the setup cell; artifacts are
#    then written to `MyDrive/mlbootcamp/artifacts` and survive any disconnect.
# 2. **Zip and download.** `export_artifacts` zips the directory (and, in
#    Colab, downloads it); `restore_artifacts` puts it back next session.
# 3. **Rebuild.** Every notebook calls `churnlab.bootstrap` to regenerate what
#    earlier notebooks made. It is deterministic and takes seconds, so you are
#    never stuck because a previous session vanished.
#
# Try the zip round trip (it writes to a temporary folder, not your artifacts,
# and does not trigger a download here):

# %%
import tempfile

from churnlab.colab import export_artifacts, restore_artifacts

with tempfile.TemporaryDirectory() as tmp:
    archive = export_artifacts(runs_dir, Path(tmp) / "runs.zip", download=False)
    restored = restore_artifacts(archive, Path(tmp) / "restored")
    print("archive:", archive.name, f"{archive.stat().st_size} bytes")
    print("restored files:", sorted(p.name for p in restored.iterdir()))

# %% [markdown]
# **Expected:** the restored folder contains the same run-record file(s).
#
# `restore_artifacts` refuses archive members that would land outside the
# target folder (a "zip slip"). Treat archives like any other untrusted input.

# %% [markdown]
# > **Exercise 3 (written).** Think of your own cross-subject EEG/BCI work (or
# > any research project). Name three ways a colleague re-running it a year
# > later could get a different number, and for each, the specific pin from
# > section 1.1 that would have prevented it. Write your answer in the cell
# > below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: No module named 'churnlab'` | setup cell not run, or run from a different folder | run the first code cell; locally, start Jupyter inside the repo |
# | `git_commit()` returns `unknown` | the folder is not a git clone (e.g. a downloaded zip) | clone with git, or accept that the code version is unrecorded and say so |
# | Drive mount pop-up never appears | browser blocked the pop-up | allow pop-ups for colab.research.google.com, re-run the setup cell |
# | Artifacts vanished after lunch | the Colab runtime was recycled | re-run: `bootstrap` rebuilds them; set `USE_DRIVE = True` to avoid it |
# | You edited a file in `src/` but nothing changed | Python cached the old module | *Runtime → Restart session*, then run from the top |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. What four things must be pinned for a number to reproduce, and how is each pinned here?
# 2. A colleague's run gives PR-AUC 0.168, yours 0.163, same code. Name three things you would compare first.
# 3. Why does pinning a seed not make a model comparison trustworthy?
# 4. Why keep code in `src/` rather than in notebooks, and why the *src layout* specifically?
# 5. Why are trained models excluded from git, and how do they travel instead?
#
# ➡️ **Next:** [`02_data_acquisition_and_validation.ipynb`](02_data_acquisition_and_validation.ipynb) — get the raw data, prove it is the right data, and find what is wrong with it.
