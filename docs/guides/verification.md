---
title: Verification
parent: Guides
nav_order: 11
---

# What was verified, and what was not

This page records exactly what was executed when the course was built. It
does **not** claim Colab testing: none of these runs happened in Google Colab.

## Summary

| Check | Result | Where |
|---|---|---|
| All 18 notebooks executed top to bottom, exercises unattempted | **18/18 pass** | main env and Colab-like env (below) |
| Each notebook started from an *empty* artifact directory | yes | simulates opening it in a new runtime with no earlier notebook run |
| Unit tests: `churnlab`, every grader vs its reference solution, stubs report TODO | **209 pass** | both environments |
| Capstone template tests | 7 pass | main env |
| Notebooks in sync with `notebook_src/`, diagrams in sync, structure, Colab badges, links | pass | `make check` |
| Lint (`ruff check src tests tools solutions`) | pass | main env |

## The two environments

1. **Main environment:** Linux build machine, Python 3.13 with current
   libraries (pandas 3.0, scikit-learn 1.9, DuckDB 2.0, FastAPI 0.16x).
2. **Colab-like environment:** a separate virtual environment, Python 3.12,
   pinned to `requirements-lock.txt` (numpy 2.0.2, pandas 2.2.2,
   scikit-learn 1.6.1, SciPy 1.15.3, DuckDB 1.2.2, FastAPI 0.115). These are
   versions of the kind a Colab runtime has had, but **it is not Colab**:
   Colab's preinstalled package set changes over time, its CPU is different,
   and its setup cell (`git clone` + `pip install`) was not exercised here.

Runs used `OMP_NUM_THREADS=1` (set by `tools/run_notebooks.py`) because the
build machine was shared. Timings are from that machine and will differ on
yours.

## Notebook runs: main environment

| Notebook | Result | Seconds | Detail |
|---|---|---:|---|
| `00_diagnostic.ipynb` | PASS | 1 |  |
| `01_reproducible_project.ipynb` | PASS | 7 |  |
| `02_data_acquisition_and_validation.ipynb` | PASS | 3 |  |
| `03_sql_for_ml.ipynb` | PASS | 4 |  |
| `04_leakage_safe_splits_and_baseline.ipynb` | PASS | 4 |  |
| `05_pipelines_and_cross_validation.ipynb` | PASS | 15 |  |
| `06_model_selection.ipynb` | PASS | 41 |  |
| `07_thresholds_and_calibration.ipynb` | PASS | 6 |  |
| `08_error_analysis_and_artifacts.ipynb` | PASS | 7 |  |
| `09_fastapi_and_contracts.ipynb` | PASS | 5 |  |
| `10_consistent_serving.ipynb` | PASS | 6 |  |
| `11_api_clients_and_tests.ipynb` | PASS | 23 |  |
| `12_packaging_and_configuration.ipynb` | PASS | 8 |  |
| `13_health_logs_and_latency.ipynb` | PASS | 40 |  |
| `14_monitoring_and_rollback.ipynb` | PASS | 11 |  |
| `15_final_evaluation_and_docs.ipynb` | PASS | 17 |  |
| `16_python_and_sql_drills.ipynb` | PASS | 3 |  |
| `17_system_design_and_mock_interview.ipynb` | PASS | 4 |  |

## Notebook runs: Colab-like environment

| Notebook | Result | Seconds | Detail |
|---|---|---:|---|
| `00_diagnostic.ipynb` | PASS | 1 |  |
| `01_reproducible_project.ipynb` | PASS | 7 |  |
| `02_data_acquisition_and_validation.ipynb` | PASS | 2 |  |
| `03_sql_for_ml.ipynb` | PASS | 4 |  |
| `04_leakage_safe_splits_and_baseline.ipynb` | PASS | 4 |  |
| `05_pipelines_and_cross_validation.ipynb` | PASS | 14 |  |
| `06_model_selection.ipynb` | PASS | 37 |  |
| `07_thresholds_and_calibration.ipynb` | PASS | 5 |  |
| `08_error_analysis_and_artifacts.ipynb` | PASS | 5 |  |
| `09_fastapi_and_contracts.ipynb` | PASS | 4 |  |
| `10_consistent_serving.ipynb` | PASS | 6 |  |
| `11_api_clients_and_tests.ipynb` | PASS | 20 |  |
| `12_packaging_and_configuration.ipynb` | PASS | 7 |  |
| `13_health_logs_and_latency.ipynb` | PASS | 30 |  |
| `14_monitoring_and_rollback.ipynb` | PASS | 9 |  |
| `15_final_evaluation_and_docs.ipynb` | PASS | 14 |  |
| `16_python_and_sql_drills.ipynb` | PASS | 3 |  |
| `17_system_design_and_mock_interview.ipynb` | PASS | 3 |  |

## Not verified

* **Google Colab itself.** First run in Colab: the setup cell's `git clone`
  and `pip install -r requirements.txt` against Colab's preinstalled versions.
  If something breaks there, see [troubleshooting](troubleshooting.html) and
  please open an issue with the notebook number and the error.
* **Optional GPU extension** (notebook 06, a small PyTorch MLP). Guarded by
  `RUN_GPU_EXTENSION = False` and never executed. PyTorch was not installed
  in either test environment.
* **Docker builds, container runs, CI on GitHub, and hosting** (notebook 12,
  `deploy/`). Taught through generated files and explanations. Not executed.
  What *was* executed: `serve.py` started as a local process with environment
  variables, probed on `/health`, `/ready` and `/predict`, then stopped.
* **Notebooks with exercises completed.** Each grader is tested against its
  reference solution in `solutions/code/` by `pytest`; the notebooks themselves
  were executed only with exercises unattempted.
* **Capstones.** The briefs' data sources were researched through search-engine
  extracts of the official pages: the official hosts (ercot.com, nasa.gov,
  archive.ics.uci.edu, grouplens.org, huggingface.co) could not be reached from
  the build machine. Each fact in a brief is labelled *verified from an
  official page snippet* or *to verify*. Confirm licences on the official pages
  before you use the data.

## Known limitations found during verification (by design, taught in the notebooks)

* `metadata.json` in a model bundle is not hashed, so editing it (for example
  the threshold) is not detected. Only `model.joblib` is. Notebook 10 shows this.
* `monitoring.psi` uses reference quantiles, so it reports ~0 for features
  that are mostly zero even when their rate changes. Notebook 14 shows this,
  together with the categorical alternative.
* `weeks_since_last_active` shows large PSI between any two periods because its
  fractional part depends on the snapshot's weekday. It is a calendar effect,
  not real drift, and notebook 14 uses it as an example of a false alarm.

## Re-running the verification yourself

```bash
pip install -r requirements.txt -r requirements-dev.txt
make check                 # sync, structure, links, unit tests
make run-notebooks         # executes all 18; writes docs/guides/verification-log.md
```
