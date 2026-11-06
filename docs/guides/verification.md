---
title: Verification
parent: Guides
nav_order: 11
---

# What was verified, and what was not

This page records exactly what was executed when the course was built (Level 1 first; Levels 2–3 at the end). It
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


## Levels 2 and 3

All 58 notebooks (Level 1: 18, Level 2: 20, Level 3: 20) were executed top to
bottom, each from an empty artifact directory, with exercises unattempted:
**58/58 pass**. Environment: Python 3.12, PyTorch 2.8 (CPU), Transformers 5.x,
numpy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, with OMP_NUM_THREADS=2 on a
4-core machine. **CPU only: no GPU was available**, so Level 3 ran with its
reduced  budget and every T4 statement in it is marked "not measured in
our verification runs". The full test suite (599 tests, including every Level 2
and Level 3 grader against its reference solution) passes in this environment.

Hugging Face Hub downloads used  on the build machine
(its network blocked the Xet transfer host). Colab does not need this.

### Level 2

| Notebook | Result | Seconds |
|---|---|---:|
| `level2/notebooks/01_event_logs_and_columnar_data.ipynb` | PASS | 50 |
| `level2/notebooks/02_aggregation_and_joins_at_scale.ipynb` | PASS | 20 |
| `level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb` | PASS | 36 |
| `level2/notebooks/04_streaming_features.ipynb` | PASS | 15 |
| `level2/notebooks/05_implicit_feedback_and_position_bias.ipynb` | PASS | 22 |
| `level2/notebooks/06_learning_to_rank.ipynb` | PASS | 68 |
| `level2/notebooks/07_candidate_generation.ipynb` | PASS | 26 |
| `level2/notebooks/08_two_tower_retrieval_and_ann.ipynb` | PASS | 60 |
| `level2/notebooks/09_pytorch_training_done_right.ipynb` | PASS | 48 |
| `level2/notebooks/10_sequence_models_for_user_behavior.ipynb` | PASS | 74 |
| `level2/notebooks/11_ab_testing_fundamentals.ipynb` | PASS | 18 |
| `level2/notebooks/12_advanced_experimentation.ipynb` | PASS | 35 |
| `level2/notebooks/13_ml_from_scratch_1.ipynb` | PASS | 4 |
| `level2/notebooks/14_ml_from_scratch_2.ipynb` | PASS | 2 |
| `level2/notebooks/15_coding_patterns_1.ipynb` | PASS | 2 |
| `level2/notebooks/16_coding_patterns_2.ipynb` | PASS | 1 |
| `level2/notebooks/17_serving_a_ranking_cascade.ipynb` | PASS | 33 |
| `level2/notebooks/18_model_optimization_for_inference.ipynb` | PASS | 41 |
| `level2/notebooks/19_recsys_and_search_system_design.ipynb` | PASS | 5 |
| `level2/notebooks/20_final_evaluation_and_mock_loop.ipynb` | PASS | 35 |

### Level 3 (CPU budget)

| Notebook | Result | Seconds |
|---|---|---:|
| `level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb` | PASS | 10 |
| `level3/notebooks/02_training_efficiency_and_memory.ipynb` | PASS | 40 |
| `level3/notebooks/03_transformer_from_scratch.ipynb` | PASS | 34 |
| `level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb` | PASS | 65 |
| `level3/notebooks/05_fine_tuning_a_classifier.ipynb` | PASS | 271 |
| `level3/notebooks/06_parameter_efficient_fine_tuning.ipynb` | PASS | 250 |
| `level3/notebooks/07_embeddings_and_semantic_search.ipynb` | PASS | 119 |
| `level3/notebooks/08_contrastive_training_and_reranking.ipynb` | PASS | 202 |
| `level3/notebooks/09_distributed_data_parallel.ipynb` | PASS | 40 |
| `level3/notebooks/10_scaling_to_large_models.ipynb` | PASS | 9 |
| `level3/notebooks/11_quantization_and_distillation.ipynb` | PASS | 226 |
| `level3/notebooks/12_high_throughput_serving.ipynb` | PASS | 74 |
| `level3/notebooks/13_ctr_prediction_at_scale.ipynb` | PASS | 70 |
| `level3/notebooks/14_multitask_ranking_and_exploration.ipynb` | PASS | 47 |
| `level3/notebooks/15_fairness_robustness_and_privacy.ipynb` | PASS | 59 |
| `level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb` | PASS | 215 |
| `level3/notebooks/17_ml_platform_design.ipynb` | PASS | 3 |
| `level3/notebooks/18_large_scale_design_cases.ipynb` | PASS | 3 |
| `level3/notebooks/19_ml_depth_and_research_discussion.ipynb` | PASS | 96 |
| `level3/notebooks/20_full_interview_loop_and_portfolio.ipynb` | PASS | 3 |

### Not verified for Levels 2–3

* Any run on a GPU, including Colab's T4: the `gpu` budget paths, fp16
  autocast speed-ups, CUDA memory statistics, and the optional bitsandbytes cells.
* The Colab setup cells (`pip install -r level2/requirements.txt`,
  `level3/requirements.txt`) against Colab's preinstalled versions.
* Multi-GPU and multi-node behaviour: taught with real multi-process gloo code
  plus arithmetic, never measured on more than one machine.
* Timings: measured on a shared 4-core machine and only indicative.
