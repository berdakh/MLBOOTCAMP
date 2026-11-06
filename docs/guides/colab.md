---
title: Working in Colab
parent: Guides
nav_order: 3
---

# Working in Colab
{: .no_toc }

Every notebook runs in a free Colab CPU runtime. The one thing to understand
is that a Colab runtime is **temporary**, and to decide in advance how you will
keep your work.

1. TOC
{:toc}

---

## Runtimes

- **CPU is enough.** Every core cell is designed for a free CPU runtime. GPU
  sections are labelled optional and guarded by `RUN_GPU_EXTENSION = False`;
  nothing later depends on them. Do not request a GPU for the core material:
  it does not make scikit-learn faster.
- **Each notebook gets its own runtime** by default. Two notebooks open in two
  tabs do not share variables or files under `/content` unless you arrange it.
- **Package versions are Colab's.** Colab preinstalls a set of libraries and
  updates them from time to time. The setup cell installs only what is
  missing or too old (`requirements.txt` has lower bounds). Versions you
  trained with are recorded in each model's metadata.

## Disconnects

Colab recycles runtimes. On the free tier, an idle runtime is typically
disconnected after somewhere around 90 minutes and any runtime after at most
about 12 hours; Google changes these limits and does not guarantee them.
When a runtime is recycled, **everything under `/content` is deleted**: the
cloned repository, your feature table, your trained models, the registry.

The course gives you three safety nets. Use at least one.

| Option | How | Survives a disconnect? | Cost |
|---|---|---|---|
| **1. Google Drive** | `USE_DRIVE = True` in the setup cell | Yes, automatically | One authorisation click per session; Drive I/O is slower than local disk |
| **2. Zip and download** | `export_artifacts` at the end; `restore_artifacts` next time | Yes, if you remember to export | Two manual steps per session |
| **3. Rebuild** | Do nothing: the catch-up cell rebuilds | Not needed | Seconds to about a minute per notebook; your *own* changes are not rebuilt |

Option 3 is always there. Every notebook's catch-up cell calls a
`churnlab.bootstrap.ensure_*` function that rebuilds what earlier notebooks
produced, deterministically. So you **never lose the ability to continue**.
What you can lose is your own work: a model you tuned differently, an exercise
answer you wrote in a cell, notes. Options 1 and 2 protect that.

{: .tip }
> Notebook cells you edit are part of the notebook, not the artifacts. To keep
> them, *File → Save a copy in Drive* before you start working. Otherwise your
> edits live only in the browser tab.

## The `USE_DRIVE` switch

In the setup cell:

```python
USE_DRIVE = True
```

With `True`, in Colab, `artifact_dir` mounts Google Drive (it asks for
authorisation once per session) and uses `MyDrive/mlbootcamp/artifacts/`.
The repository itself is still cloned into `/content` and re-cloned after a
disconnect; only the artifacts persist.

Things to know:

- A model saved to Drive in one session may be loaded in a later session where
  Colab has a newer scikit-learn. The loader then warns about a version
  mismatch. See [Troubleshooting](troubleshooting.html#scikit-learn-version-mismatch-when-loading-a-bundle).
- Two notebooks running at once with `USE_DRIVE = True` write to the same
  folder. That is fine for reading; avoid training into the same registry
  from two notebooks at the same moment.

## The `BOOTCAMP_ARTIFACTS` environment variable

The artifact directory is chosen in this order:

1. the `BOOTCAMP_ARTIFACTS` environment variable, if set;
2. Google Drive, if `USE_DRIVE = True` and you are in Colab;
3. `<repo>/artifacts/`.

Set the variable to put artifacts anywhere you like, for example a separate
Drive folder per capstone or a scratch directory to test a fresh start. Set
it **before** the setup cell runs, in a cell above it:

```python
import os
os.environ["BOOTCAMP_ARTIFACTS"] = "/content/drive/MyDrive/mlbootcamp/fresh_try"
```

If you point it into Drive, mount Drive first
(`from google.colab import drive; drive.mount("/content/drive")`), because
`artifact_dir` only mounts Drive itself on the `USE_DRIVE` path.

## Zip and download

At the end of a session:

```python
from churnlab.colab import export_artifacts
export_artifacts(ART)          # writes artifacts.zip next to the folder; in Colab, also downloads it
```

At the start of the next session, upload the zip with the file browser on the
left (it lands in `/content`), then, after the setup cell:

```python
from pathlib import Path
from churnlab.colab import restore_artifacts
restore_artifacts(Path("/content/artifacts.zip"), ART, overwrite=True)
```

`restore_artifacts` refuses to overwrite a non-empty directory unless you pass
`overwrite=True`, and refuses archive entries that would extract outside the
target directory.

## Rebuild with `churnlab.bootstrap`

What each catch-up function provides:

| Function | Rebuilds | Originally from |
|---|---|---|
| `ensure_raw()` | The raw extract, checksum-verified (downloads only if a file is missing or corrupt) | Notebook 02 |
| `load_clean()` | The four tables, cleaned | Notebook 02 |
| `ensure_features()` | `features.parquet`: the (snapshot, customer) training table | Notebook 03 |
| `ensure_split()` | The fixed train / validation / test time split | Notebook 04 |
| `ensure_registry()` | A registry with v1 (logistic regression) and v2 (boosted trees), the higher validation PR-AUC promoted to `CURRENT` | Notebook 08 |

Each function reuses what is already on disk and builds only what is missing.
To force a clean rebuild, delete the relevant file or folder in the artifact
directory and rerun the catch-up cell.

## Running a server in Colab

Day 3 and Day 4 start a local `uvicorn` server inside the runtime and call it
from the notebook. It is reachable only from that runtime, not from the
internet. That is enough for everything in the course. Exposing it publicly
needs a tunnelling service or a real host, which is outside the course.

If you rerun a cell that starts the server, the port may still be in use; see
[Troubleshooting](troubleshooting.html#port-already-in-use).

## Working on capstones in Colab

The same setup works. A pattern that keeps capstones apart:

```python
import os
os.environ["BOOTCAMP_ARTIFACTS"] = "/content/drive/MyDrive/capstones/ercot/artifacts"
```

For anything you will show in a portfolio, keep the code in a git repository
(a capstone repository from the template), not only in Colab notebooks.
