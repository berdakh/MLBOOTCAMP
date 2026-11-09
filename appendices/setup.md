# Setup

The labs can be run in two ways. Google Colab requires no installation and is
the simplest choice. A local installation is closer to the way the work is
done in practice, and it is required for the optional Docker extension of
[Chapter 13](../part1-foundations/ch13-packaging-and-configuration/README.md).

---

## Option A: Google Colab (no installation)

1. Open any lab from its Colab badge. Every chapter's `README.md` and the
   table of contents in the [root README](../README.md) carry one; for
   example, [the Chapter 1 lab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).
2. Choose *Runtime → Run all*, or run the first code cell on its own.
3. For Part III, first choose a GPU runtime: *Runtime → Change runtime type →
   T4 GPU*.

The first code cell of every lab, the **setup cell**, does the following:

- it clones the repository into `/content/MLBOOTCAMP` if it is not already
  there;
- it installs the requirements file of the chapter's part
  (`requirements.txt` for Part I, `requirements-part2.txt` for Part II,
  `requirements-part3.txt` for Part III). Colab already provides most of the
  packages, so this usually takes seconds;
- it finds the repository root and adds `src/` to the import path, so that
  `import churnlab` (and, in later parts, `shoprank` and `scaleml`) works;
- it chooses the **artifact directory**, where models, feature tables, the
  model registry and generated datasets are written, and prints its path;
- in Part III, it also selects the device (GPU if present) and a compute
  budget, and prints both.

Near the top of the setup cell is a single switch:

```python
USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect
```

Read [Working in Colab](colab.md) before deciding. In brief, `True` asks you
to authorise Google Drive once per session and then keeps your artifacts
across disconnects.

The second code cell, the **catch-up cell**, rebuilds anything that earlier
chapters would have produced. Any lab can therefore be opened in a fresh
runtime, in any order.

## Option B: local installation

Requirements: Python 3.10 or newer, git, and about 1 GB of disk space for the
Part I environment (Parts II and III need more, mainly for PyTorch and
Hugging Face models).

```bash
git clone https://github.com/berdakh/MLBOOTCAMP.git
cd MLBOOTCAMP
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # Part I
python -m pytest                   # about a minute, no network needed
jupyter lab part1-foundations/ch01-diagnostic/notebook.ipynb
```

The later parts add to the Part I requirements; each file includes the one
before it:

```bash
pip install -r requirements-part2.txt   # Part II: adds PyTorch, LightGBM and others
pip install -r requirements-part3.txt   # Part III: adds Transformers, Datasets and others
```

If `jupyter` is not found, install it into the same environment and register
the kernel, so that the labs use this environment's Python:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name mlbootcamp
```

Then choose the `mlbootcamp` kernel in Jupyter.

### Installing the package in editable mode (recommended)

```bash
pip install -e .
```

This makes `import churnlab` work from any directory, and changes to
`src/churnlab/*.py` take effect after a kernel restart (or after
`importlib.reload`; see
[Troubleshooting](troubleshooting.md#code-changes-not-picked-up)). The labs'
setup cell works without it.

### Exact versions

`requirements.txt` gives **lower bounds** on versions, which keeps the Colab
setup fast. `requirements-lock.txt` pins the exact versions with which the
Part I labs were executed (see [Verification](verification.md)). Use it when
a result must be reproduced exactly, or when a version difference is the
suspected cause of a problem:

```bash
pip install -r requirements-lock.txt
```

### Where artifacts are written locally

By default, artifacts are written to `<repo>/artifacts/`. An environment
variable overrides this:

```bash
export BOOTCAMP_ARTIFACTS=/path/to/somewhere   # Windows: set BOOTCAMP_ARTIFACTS=...
```

To start again from nothing, delete the artifact directory; the catch-up cells
rebuild what each lab needs.

### Choosing the device in Parts II and III

The `scaleml` toolkit chooses CUDA if it is available, then Apple MPS, then
the CPU. Two environment variables override the choice:

```bash
export BOOTCAMP_DEVICE=cpu       # force a device, for example to reproduce a CPU-only run
export BOOTCAMP_BUDGET=cpu       # force a compute budget: gpu, cpu or smoke
```

The `smoke` budget is the smallest; it is used by automated tests to show that
code runs, and its numerical results are not meaningful.

## Checking that the installation works

In a notebook or Python session started from the repository root:

```python
import sys; sys.path.insert(0, "src")
from churnlab import bootstrap
split = bootstrap.ensure_split(verbose=True)
print(split.summary())
```

The output is a table with three rows (train, validation, test), their
snapshot date ranges and their positive rates. The first call builds the
feature table, which takes from a few seconds to a minute on a laptop CPU.

## Maintainer tools (optional)

These are not needed for reading the book or running the labs:

```bash
pip install -r requirements-dev.txt
make check           # what CI runs: notebook sync, figures, structure, links, tests
```

[CONTRIBUTING.md](../CONTRIBUTING.md) explains how the chapters, labs,
figures and solutions are built.
