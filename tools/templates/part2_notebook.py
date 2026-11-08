# Template for a chapter lab in part2-ml-at-scale. Not built (it lives in tools/templates).
# Copy to part2-ml-at-scale/chNN-slug/notebook.py, then replace chNN-slug, NN and the placeholders.

# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/chNN-slug/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter NN · Title
#
# **Part II · Chapter NN · about 90 minutes · CPU (a GPU only speeds up the optional parts)**
#
# Motivation in two sentences.
#
# By the end you will be able to:
#
# 1. ...
# 2. ...
#
# **You will produce:** ...

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: rebuild anything earlier notebooks would have produced -------
from grading import chNN as grade

# %% [markdown]
# ## N.1 · First section

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` | setup cell not run | run the first code cell |

# %% [markdown]
# ## Checkpoint
#
# 1. ...
#
# **Next chapter:** [Chapter NN+1 · Title](../chNN-next/notebook.ipynb) — why.
