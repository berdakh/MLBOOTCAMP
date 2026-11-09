# Working in Colab

Every lab in the book runs in a free Google Colab runtime: Parts I and II on
a CPU, Part III on the free T4 GPU (or, more slowly, on a CPU). The most
important property of a Colab runtime is that it is **temporary**. This
appendix explains what that implies and how to decide, in advance, how your
work will be kept.


---

## Runtimes

- **Parts I and II need only a CPU.** Every core cell of these parts is
  designed for a free CPU runtime. The one GPU section in Part I (a
  small PyTorch network in Chapter 7) is labelled optional and guarded by
  `RUN_GPU_EXTENSION = False`, and no later chapter depends on it. A GPU does not make scikit-learn faster, so there
  is no reason to request one for this material.
- **Part III is written for a GPU.** Choose *Runtime → Change runtime type →
  T4 GPU* before running a Part III lab. The setup cell prints the device it
  found and the compute budget it chose. On a CPU the labs still run, with a
  reduced budget (fewer examples, shorter sequences, smaller models), and the
  results are correspondingly weaker. The free GPU allowance is limited and
  varies; when it is exhausted, Colab offers only CPU runtimes for a while.
- **Each lab has its own runtime** by default. Two labs open in two browser
  tabs do not share variables or files under `/content` unless you arrange
  it.
- **Package versions are Colab's.** Colab preinstalls a set of libraries and
  updates them from time to time. The setup cell installs only what is
  missing or older than the lower bounds in the part's requirements file.
  The versions a model was trained with are recorded in its metadata.

## Disconnects

Colab recycles runtimes. On the free tier, an idle runtime is typically
disconnected after somewhere around 90 minutes and any runtime after at most
about 12 hours; Google changes these limits and does not guarantee them.
When a runtime is recycled, **everything under `/content` is deleted**: the
cloned repository, your feature table, your trained models, the registry.

The book provides three ways to protect against this loss. Use at least one.

| Option | How | Survives a disconnect? | Cost |
|---|---|---|---|
| **1. Google Drive** | `USE_DRIVE = True` in the setup cell | Yes, automatically | One authorisation click per session; Drive I/O is slower than local disk |
| **2. Zip and download** | `export_artifacts` at the end; `restore_artifacts` next time | Yes, if you remember to export | Two manual steps per session |
| **3. Rebuild** | Do nothing: the catch-up cell rebuilds | Not needed | Seconds to a few minutes per lab; your *own* changes are not rebuilt |

Option 3 is always available. Every lab's catch-up cell calls functions
(in Part I, `churnlab.bootstrap.ensure_*`) that rebuild deterministically what
earlier chapters produced. The ability to continue is therefore **never
lost**. What can be lost is your own work: a model tuned differently, an
exercise answer written in a cell, notes. Options 1 and 2 protect that.

> **Note.** Cells you edit belong to the notebook, not to the artifacts. To
> keep them, choose *File → Save a copy in Drive* before starting work;
> otherwise the edits exist only in the browser tab.

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
  mismatch. See [Troubleshooting](troubleshooting.md#scikit-learn-version-mismatch-when-loading-a-bundle).
- Two labs running at once with `USE_DRIVE = True` write to the same
  folder. Reading from it concurrently is safe; training into the same
  registry from two labs at the same moment is not.
- Part II generates synthetic data sets under `artifacts/level2/` inside the
  artifact directory. With `USE_DRIVE = True` these are kept as well, which
  saves regeneration time but uses Drive space. Models and data sets that
  Part III downloads from the Hugging Face Hub are cached separately, in the
  directory named by the `HF_HOME` environment variable.

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

## Rebuilding with `churnlab.bootstrap`

The Part I catch-up functions, and what each provides:

| Function | Rebuilds | Originally from |
|---|---|---|
| `ensure_raw()` | The raw extract, checksum-verified (downloads only if a file is missing or corrupt) | Chapter 3 |
| `load_clean()` | The four tables, cleaned | Chapter 3 |
| `ensure_features()` | `features.parquet`: the (snapshot, customer) training table | Chapter 4 |
| `ensure_split()` | The fixed train / validation / test time split | Chapter 5 |
| `ensure_registry()` | A registry with v1 (logistic regression) and v2 (boosted trees), the higher validation PR-AUC promoted to `CURRENT` | Chapter 9 |

Each function reuses what is already on disk and builds only what is
missing. To force a clean rebuild, delete the relevant file or folder in the
artifact directory and rerun the catch-up cell. The labs of Parts II and III
follow the same principle with their own functions (for example, regenerating
the ShopStream data set when its folder is missing); each lab's catch-up cell
names what it rebuilds.

## Running a server in Colab

Chapters 10–15 start a local `uvicorn` server inside the runtime and call it
from the lab. The server is reachable only from that runtime, not from the
internet, which is sufficient for everything in the book. Making it publicly
reachable requires a tunnelling service or a real host, which is outside the
scope of the book.

If you rerun a cell that starts the server, the port may still be in use; see
[Troubleshooting](troubleshooting.md#port-already-in-use).

## Working on capstones in Colab

The same setup works for the capstone projects of Part IV. The following
pattern keeps the artifacts of different capstones apart:

```python
import os
os.environ["BOOTCAMP_ARTIFACTS"] = "/content/drive/MyDrive/capstones/ercot/artifacts"
```

For work that others will read, such as a portfolio project, keep the code
in a git repository created from the
[capstone template](../part4-capstones/template/README.md), not only in Colab
notebooks.
