# Troubleshooting

Start here when something breaks. Organised by what you see. Each notebook
also ends with a short troubleshooting table for its own failures.


---

## First, three questions

1. **Did the setup cell run in this runtime?** After a disconnect or
   *Restart session*, every variable and import is gone. Run from the top.
2. **Which Python and which versions?**
   ```python
   import sys, sklearn, pandas; print(sys.executable, sklearn.__version__, pandas.__version__)
   ```
3. **Where are your artifacts?** The setup cell prints `artifacts:`. If that
   folder is empty, the catch-up cell will rebuild it.

---

## Colab and environment

<a id="colab-disconnected-and-my-artifacts-are-gone"></a>

### Colab disconnected and my artifacts are gone

**Cause.** The runtime was recycled (idle too long, or the session limit).
Everything under `/content` is deleted, including the default artifact
directory.

**Fix.** Rerun the notebook from the top. The catch-up cell rebuilds features,
the split and the registry deterministically, usually in under a minute.

**Prevent.** Set `USE_DRIVE = True` in the setup cell, or export a zip at the
end of each session. See [Working in Colab](colab.md#disconnects). Your own
cell edits are not artifacts: *File → Save a copy in Drive* keeps them.

### `ModuleNotFoundError: No module named 'churnlab'`

The setup cell has not run in this runtime, or you started Python from a
directory outside the repository. Run the setup cell. Locally, either run from
the repository root or `pip install -e .`.

<a id="pip-conflicts-in-colab"></a>

### pip warnings or conflicts during the Colab setup cell

You may see:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...
```

**What it means.** Colab preinstalls many packages with their own pins.
Installing or upgrading one package can leave another preinstalled package
with a version it did not ask for. pip reports it but does not fail.

**Usually harmless** for this course: the notebooks import only what is in
`requirements.txt`. Continue unless something actually breaks.

**When it is not harmless.** If the install upgraded a compiled package that
was already imported (NumPy is the usual one), you will see strange errors
such as `numpy.dtype size changed` or an `ImportError` from a compiled module.
Colab may also show a *Restart session* button. Then: *Runtime → Restart
session*, and run from the top **without** re-running a manual `pip install`
of a different version.

**Do not** `pip install` packages one by one with `--upgrade` in Colab unless
you need a specific version. If you do need one (for example to match a saved
model's scikit-learn), install it, restart the session, then run from the top.

<a id="code-changes-not-picked-up"></a>

### Code changes not picked up

You edited `src/churnlab/api.py` (or a capstone module), reran the cell, and
nothing changed.

**Cause.** Python imports a module once per process. Re-running `import` does
not re-read the file.

**Fix, simplest first:**

1. *Runtime → Restart session* (Colab) or *Kernel → Restart* (Jupyter), then
   run from the top. Always works.
2. Reload the module:
   ```python
   import importlib, churnlab.api
   importlib.reload(churnlab.api)
   from churnlab.api import create_app   # re-bind names imported with `from`
   ```
   `reload` does not update objects that already exist (an app you created
   earlier still uses the old code) or names other modules imported from it.
   When in doubt, restart.
3. For a long session of editing, autoreload:
   ```python
   %load_ext autoreload
   %autoreload 2
   ```

A uvicorn server running in the background is a **separate process**: it does
not see your changes until you stop and restart it.

<a id="port-already-in-use"></a>

### Port already in use

```
ERROR:    [Errno 98] Address already in use
```

**Cause.** A uvicorn server started by an earlier cell (or an earlier run of
this cell) is still listening on the port.

**Fix.**

- If you kept the process handle, stop it: `proc.terminate(); proc.wait()`.
- Otherwise, find and stop whatever holds the port (Linux, macOS, Colab):
  ```bash
  lsof -i :8000          # shows the PID
  kill <PID>
  ```
  In Colab, prefix with `!` in a cell. If `lsof` is missing,
  `fuser -k 8000/tcp` does the same.
- Or start on another port (`--port 8001`) and point the client at it.
- *Restart session* also stops every child process.

Prevent it by starting servers in a `try/finally` that terminates the process,
or by checking the port before starting.

---

## Data

<a id="the-na-region-became-missing"></a>

### The `NA` region became missing

**Symptom.** After reading a raw CSV yourself, the region `NA` (North America)
has disappeared, region has far more missing values than the roughly 2%
planted, and after cleaning the `unknown` category is huge.

**Cause.** By default, `pandas.read_csv` treats the strings `NA`, `N/A`,
`NULL`, `nan` and several others as missing. A region code of `NA` is
silently converted to `NaN`.

**Fix.** Tell pandas that only empty fields are missing:

```python
pd.read_csv(path, keep_default_na=False, na_values=[""])
```

That is what `churnlab.data.load_raw` does. Use it, or copy its arguments.
The same trap catches country codes (Namibia is `NA`), gene names, and
abbreviations in free text.

<a id="duckdb-type-errors"></a>

### DuckDB type errors

Typical messages:

```
Binder Error: Cannot compare values of type TIMESTAMP and type VARCHAR - an explicit cast is required
Binder Error: No function matches the given name and argument types 'date_diff(STRING_LITERAL, VARCHAR, DATE)'
Conversion Error: Could not convert string '...' to INT32
```

**Causes and fixes:**

| Cause | Fix |
|---|---|
| A date column is still a string in pandas (read from CSV without parsing) | `df[col] = pd.to_datetime(df[col])` before registering; or `CAST(col AS DATE)` in SQL |
| Comparing a date column with a string *column* or string-typed value (a bare literal like `'2024-04-01'` is usually cast for you) | Parse the string column, or write a typed literal: `DATE '2024-04-01'` |
| Interval arithmetic on a string | Same as above: parse first |
| An integer column holds a sentinel string or empty values read as strings | Fix at read time (`na_values`) or `TRY_CAST(col AS INTEGER)` and inspect the NULLs it produces |
| A pandas `object` column of mixed types | Make the dtype explicit before `con.register` |

To see what DuckDB thinks the types are:

```python
con.execute("DESCRIBE SELECT * FROM usage_weekly").df()
```

### My SQL feature differs from the reference by a little
Usually a boundary: `<` versus `<=`, or using `week_start` where the week's
*end* (`week_start + INTERVAL 7 DAY`) is meant. Compare on one customer and one
snapshot, printing the rows each version includes. See
[Handbook 3](../part1-foundations/_handbook/03-data-sql-and-leakage.md#point-in-time-correctness).

---

## Models and artifacts

<a id="scikit-learn-version-mismatch-when-loading-a-bundle"></a>

### scikit-learn version mismatch when loading a bundle

Warnings such as:

```
UserWarning: model trained with sklearn 1.5.2, running 1.6.1
InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.5.2 when using version 1.6.1
```

**Cause.** The bundle was saved with a different scikit-learn minor version,
typically because it was saved to Drive in an earlier Colab session and Colab
has since updated scikit-learn. Pickled estimators are not guaranteed to work
across versions, and failures can be silent (wrong outputs rather than
errors).

**Fixes, safest first:**

1. **Retrain.** Delete the registry folder in your artifact directory and
   rerun the catch-up cell; `ensure_registry()` retrains in seconds.
2. **Match the version.** `pip install scikit-learn==<version in metadata.json>`,
   restart the session, rerun.
3. **Accept the risk knowingly**, for a notebook exercise only: check that the
   loaded model's scores on a few rows match the scores recorded at training
   time.

In a service, use `strict_versions=True` so a mismatch fails at start-up
instead of producing quietly different predictions.

### `ArtifactError: model.joblib does not match the hash in metadata.json`

The model file changed after it was saved: a partial copy, a sync conflict in
Drive, or a manual edit. Do not load it. Delete that version folder and
retrain, or restore from your zip.

### `ArtifactError: ... already exists; bundles are immutable`

You tried to save a model into an existing version folder. That is refused on
purpose. Register a new version instead (`registry.register(...)`).

### `nothing to roll back to`

Rollback needs at least two promotions in `history.jsonl`. Promote a second
version first.

---

## API

<a id="422-unprocessable-entity"></a>

### 422 Unprocessable Entity

**This is the contract working.** The request did not match the schema. Read
the body:

```json
{"detail": [{"type": "extra_forbidden", "loc": ["body", "sesions_4w"], "msg": "Extra inputs are not permitted", ...}]}
```

`loc` is the path to the offending field; `msg` is the rule it broke. Common
causes:

| `type` / message | Cause |
|---|---|
| `extra_forbidden` | A misspelled or unexpected field (`extra="forbid"`) |
| `missing` | A required field was not sent |
| `greater_than_equal` / `less_than_equal` | A value outside its bounds (negative sessions, utilisation above 1.5) |
| `float_parsing` / `string_type` | Wrong type: a string where a number is expected, or vice versa |
| `too_short` on `customers` | An empty batch |

Fix the request, not the server. Do not retry a 422: it will fail the same way
every time. The course's client raises immediately on 4xx.

### 503 from `/ready` or `/predict`

No model is loaded. The `detail` field says why, for example "no CURRENT
model in …". Check `CHURN_REGISTRY_DIR` points at a registry with a `CURRENT`
file, or run `bootstrap.ensure_registry()`.

### 413 from `/predict/batch`

The batch is larger than `max_batch_size` (default 1000). Use the client's
`predict_batch`, which chunks inputs.

### The client hangs

Something has no timeout. The course's `ChurnClient` sets one (5 s by
default). If you wrote your own `httpx` or `requests` call, add
`timeout=...`.

---

## Tests

### Tests pass locally but a notebook check fails (or the other way round)

Usually different working directories or different artifact directories.
Tests use temporary directories; notebooks use your artifact directory. Print
both paths. Stale artifacts from an earlier version of your code are a common
cause: delete and rebuild.

### `pytest` collects nothing

Run it from the repository root (`python -m pytest`), where `pyproject.toml`
configures `testpaths` and the import path.

---

## Still stuck

- Reduce the problem: one customer, one snapshot, one request.
- Read the source. The `churnlab` modules are short and commented.
- Check [Verification](verification.md) for what was tested in which
  environment.
- Open an issue on the [repository](https://github.com/berdakh/MLBOOTCAMP/issues)
  with the full traceback, the versions from the three questions above, and
  whether you were in Colab or local.
