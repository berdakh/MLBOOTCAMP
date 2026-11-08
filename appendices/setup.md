# Setup

Two ways to run the course. Colab needs nothing installed. A local install is
closer to how you will work on the job, and is needed for the optional
Docker extension.


---

## Option A: Google Colab (nothing to install)

1. Open any notebook from its Colab link (each [day page](../part1-foundations/_dayplans/index.md) has them),
   for example [notebook 00](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).
2. *Runtime → Run all*, or run the first code cell on its own.

The first code cell (the **setup cell**) does everything:

- clones the repository into `/content/MLBOOTCAMP` if it is not already there;
- installs `requirements.txt` (Colab already has most packages, so this
  usually takes seconds);
- finds the repository root and adds `src/` to the import path, so
  `import churnlab` works;
- chooses the **artifact directory** where models, feature tables and the
  registry are written, and prints it.

Near the top of the setup cell is one switch:

```python
USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect
```

Read [Working in Colab](colab.md) before deciding. In short: `True` asks you
to authorise Google Drive once per session and then keeps your work across
disconnects.

The second code cell (the **catch-up cell**) rebuilds anything earlier
notebooks would have produced. You can open any notebook in a fresh runtime.

## Option B: local install

Requirements: Python 3.10 or newer, git, and about 1 GB of disk for the
environment.

```bash
git clone https://github.com/berdakh/MLBOOTCAMP.git
cd MLBOOTCAMP
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                   # about a minute, no network needed
jupyter lab notebooks/00_diagnostic.ipynb
```

If `jupyter` is not found, install it into the same environment
(`pip install jupyterlab`) and register the kernel so notebooks use this
environment's Python:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name mlbootcamp
```

Then choose the `mlbootcamp` kernel in Jupyter.

### Installing the package in editable mode (recommended)

```bash
pip install -e .
```

This makes `import churnlab` work from any directory, and changes you make to
`src/churnlab/*.py` are picked up after a kernel restart (or an
`importlib.reload`, see [Troubleshooting](troubleshooting.md#code-changes-not-picked-up)).
The notebooks' setup cell works without it.

### Exact versions

`requirements.txt` gives **lower bounds**, which keeps Colab setup fast.
`requirements-lock.txt` pins the exact versions the notebooks were executed
with. Use it when you need to reproduce a result exactly, or when a version
difference is the suspected cause of a problem:

```bash
pip install -r requirements-lock.txt
```

### Where artifacts go locally

By default, `<repo>/artifacts/`. Override with an environment variable:

```bash
export BOOTCAMP_ARTIFACTS=/path/to/somewhere   # Windows: set BOOTCAMP_ARTIFACTS=...
```

To start completely fresh, delete the artifacts directory. The catch-up cells
rebuild it.

## Checking it works

In a notebook or Python session from the repository root:

```python
import sys; sys.path.insert(0, "src")
from churnlab import bootstrap
split = bootstrap.ensure_split(verbose=True)
print(split.summary())
```

You should see a table with three rows (train, valid, test), their snapshot
date ranges and positive rates. The first run builds the feature table, which
takes seconds to a minute on a laptop CPU.

## Maintainer tools (optional)

Not needed to take the course:

```bash
pip install -r requirements-dev.txt
make check           # what CI runs: notebook sync, structure, links, diagrams, tests
```

See the repository's `CONTRIBUTING.md` for how the notebooks are built.
