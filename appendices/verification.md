# Verification: what was executed, and what was not

This appendix records exactly what was executed when the book's material was
built: Part I first, then Parts II and III (at the end of this appendix). It
does **not** claim any testing in Google Colab: none of these runs took place
in Colab.

**A note on names.** The runs recorded here took place before the material was
restructured as a book. At that time Part I was called "Level 1" and its labs
were numbered 00–17 (`notebooks/00_diagnostic.ipynb` and so on), and Parts II
and III were "Level 2" and "Level 3" with labs numbered 01–20 in
`level2/notebooks/` and `level3/notebooks/`. In the book the chapters are
numbered 1–58 and each lab is the `notebook.ipynb` in its chapter folder. The
tables below give, for every recorded run, the chapter number, the lab's
current path and the file name under which it was recorded. The
restructuring renamed and moved files; no run has been repeated under the new
names, and the results below are those of the recorded runs.

## Summary

| Check | Result | Where |
|---|---|---|
| All 18 Part I labs (Chapters 1–18) executed top to bottom, exercises unattempted | **18/18 pass** | main env and Colab-like env (below) |
| Each lab started from an *empty* artifact directory | yes | simulates opening it in a new runtime with no earlier lab run |
| Unit tests: `churnlab`, every grader vs its reference solution, stubs report TODO | **209 pass** | both environments |
| Capstone template tests | 7 pass | main env |
| Notebooks in sync with their sources (then in `notebook_src/`; now each chapter's `notebook.py`), diagrams in sync, structure, Colab badges, links | pass | `make check` |
| Lint (`ruff check src tests tools solutions`; the reference solutions, then in `solutions/`, are now each chapter's `solutions.py`) | pass | main env |

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

## Part I lab runs: main environment

| Ch. | Lab (current path) | Recorded as | Result | Seconds |
|---:|---|---|---|---:|
| 1 | `part1-foundations/ch01-diagnostic/notebook.ipynb` | `00_diagnostic.ipynb` | PASS | 1 |
| 2 | `part1-foundations/ch02-reproducible-project/notebook.ipynb` | `01_reproducible_project.ipynb` | PASS | 7 |
| 3 | `part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb` | `02_data_acquisition_and_validation.ipynb` | PASS | 3 |
| 4 | `part1-foundations/ch04-sql-for-ml/notebook.ipynb` | `03_sql_for_ml.ipynb` | PASS | 4 |
| 5 | `part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb` | `04_leakage_safe_splits_and_baseline.ipynb` | PASS | 4 |
| 6 | `part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb` | `05_pipelines_and_cross_validation.ipynb` | PASS | 15 |
| 7 | `part1-foundations/ch07-model-selection/notebook.ipynb` | `06_model_selection.ipynb` | PASS | 41 |
| 8 | `part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb` | `07_thresholds_and_calibration.ipynb` | PASS | 6 |
| 9 | `part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb` | `08_error_analysis_and_artifacts.ipynb` | PASS | 7 |
| 10 | `part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb` | `09_fastapi_and_contracts.ipynb` | PASS | 5 |
| 11 | `part1-foundations/ch11-consistent-serving/notebook.ipynb` | `10_consistent_serving.ipynb` | PASS | 6 |
| 12 | `part1-foundations/ch12-api-clients-and-tests/notebook.ipynb` | `11_api_clients_and_tests.ipynb` | PASS | 23 |
| 13 | `part1-foundations/ch13-packaging-and-configuration/notebook.ipynb` | `12_packaging_and_configuration.ipynb` | PASS | 8 |
| 14 | `part1-foundations/ch14-health-logs-and-latency/notebook.ipynb` | `13_health_logs_and_latency.ipynb` | PASS | 40 |
| 15 | `part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb` | `14_monitoring_and_rollback.ipynb` | PASS | 11 |
| 16 | `part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb` | `15_final_evaluation_and_docs.ipynb` | PASS | 17 |
| 17 | `part1-foundations/ch17-python-and-sql-drills/notebook.ipynb` | `16_python_and_sql_drills.ipynb` | PASS | 3 |
| 18 | `part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb` | `17_system_design_and_mock_interview.ipynb` | PASS | 4 |

## Part I lab runs: Colab-like environment

| Ch. | Lab (current path) | Recorded as | Result | Seconds |
|---:|---|---|---|---:|
| 1 | `part1-foundations/ch01-diagnostic/notebook.ipynb` | `00_diagnostic.ipynb` | PASS | 1 |
| 2 | `part1-foundations/ch02-reproducible-project/notebook.ipynb` | `01_reproducible_project.ipynb` | PASS | 7 |
| 3 | `part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb` | `02_data_acquisition_and_validation.ipynb` | PASS | 2 |
| 4 | `part1-foundations/ch04-sql-for-ml/notebook.ipynb` | `03_sql_for_ml.ipynb` | PASS | 4 |
| 5 | `part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb` | `04_leakage_safe_splits_and_baseline.ipynb` | PASS | 4 |
| 6 | `part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb` | `05_pipelines_and_cross_validation.ipynb` | PASS | 14 |
| 7 | `part1-foundations/ch07-model-selection/notebook.ipynb` | `06_model_selection.ipynb` | PASS | 37 |
| 8 | `part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb` | `07_thresholds_and_calibration.ipynb` | PASS | 5 |
| 9 | `part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb` | `08_error_analysis_and_artifacts.ipynb` | PASS | 5 |
| 10 | `part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb` | `09_fastapi_and_contracts.ipynb` | PASS | 4 |
| 11 | `part1-foundations/ch11-consistent-serving/notebook.ipynb` | `10_consistent_serving.ipynb` | PASS | 6 |
| 12 | `part1-foundations/ch12-api-clients-and-tests/notebook.ipynb` | `11_api_clients_and_tests.ipynb` | PASS | 20 |
| 13 | `part1-foundations/ch13-packaging-and-configuration/notebook.ipynb` | `12_packaging_and_configuration.ipynb` | PASS | 7 |
| 14 | `part1-foundations/ch14-health-logs-and-latency/notebook.ipynb` | `13_health_logs_and_latency.ipynb` | PASS | 30 |
| 15 | `part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb` | `14_monitoring_and_rollback.ipynb` | PASS | 9 |
| 16 | `part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb` | `15_final_evaluation_and_docs.ipynb` | PASS | 14 |
| 17 | `part1-foundations/ch17-python-and-sql-drills/notebook.ipynb` | `16_python_and_sql_drills.ipynb` | PASS | 3 |
| 18 | `part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb` | `17_system_design_and_mock_interview.ipynb` | PASS | 3 |

## Not verified

* **Google Colab itself.** First run in Colab: the setup cell's `git clone`
  and `pip install -r requirements.txt` against Colab's preinstalled versions.
  If something breaks there, see [Troubleshooting](troubleshooting.md) and
  open an issue with the chapter number and the error.
* **Optional GPU extension** (Chapter 7, recorded as notebook 06: a small PyTorch MLP). Guarded by
  `RUN_GPU_EXTENSION = False` and never executed. PyTorch was not installed
  in either test environment.
* **Docker builds, container runs, CI on GitHub, and hosting** (Chapter 13,
  recorded as notebook 12; `deploy/`). Taught through generated files and explanations. Not executed.
  What *was* executed: `serve.py` started as a local process with environment
  variables, probed on `/health`, `/ready` and `/predict`, then stopped.
* **Labs with exercises completed.** Each grader is tested against its
  reference solution by `pytest` (the reference code was then in
  `solutions/code/` and is now each chapter's `solutions.py`); the labs
  themselves were executed only with exercises unattempted.
* **Capstones.** The briefs' data sources were researched through search-engine
  extracts of the official pages: the official hosts (ercot.com, nasa.gov,
  archive.ics.uci.edu, grouplens.org, huggingface.co) could not be reached from
  the build machine. Each fact in a brief is labelled *verified from an
  official page snippet* or *to verify*. Confirm licences on the official pages
  before you use the data.

## Known limitations found during verification (by design, taught in the labs)

* `metadata.json` in a model bundle is not hashed, so editing it (for example
  the threshold) is not detected. Only `model.joblib` is. Chapter 11 (recorded
  as notebook 10) shows this.
* `monitoring.psi` uses reference quantiles, so it reports ~0 for features
  that are mostly zero even when their rate changes. Chapter 15 (recorded as
  notebook 14) shows this, together with the categorical alternative.
* `weeks_since_last_active` shows large PSI between any two periods because its
  fractional part depends on the snapshot's weekday. It is a calendar effect,
  not real drift, and Chapter 15 uses it as an example of a false alarm.

## Repeating the verification

```bash
pip install -r requirements.txt -r requirements-dev.txt
make check                 # sync, structure, links, unit tests
make run-notebooks         # executes the 18 Part I labs; writes verification-log.md
```

`tools/run_notebooks.py --part part2-ml-at-scale` (or
`part3-deep-learning-systems`) runs the labs of another part in the same way.


## Parts II and III

All 58 labs (Part I: 18, Part II: 20, Part III: 20) were executed top to
bottom, each from an empty artifact directory, with exercises unattempted:
**58/58 pass**. Environment: Python 3.12, PyTorch 2.8 (CPU), Transformers 5.x,
numpy 2.0.2, pandas 2.2.2, scikit-learn 1.6.1, with OMP_NUM_THREADS=2 on a
4-core machine. **CPU only: no GPU was available**, so Part III ran with its
reduced `cpu` budget and every T4 statement in it is marked "not measured in
our verification runs". The full test suite (599 tests, including every
Part II and Part III grader against its reference solution) passed in this
environment.

Hugging Face Hub downloads on the build machine had to work around its
network, which blocked the Xet transfer host. Colab does not need this.

### Part II (Chapters 19–38)

| Ch. | Lab (current path) | Recorded as | Result | Seconds |
|---:|---|---|---|---:|
| 19 | `part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb` | `level2/notebooks/01_event_logs_and_columnar_data.ipynb` | PASS | 50 |
| 20 | `part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb` | `level2/notebooks/02_aggregation_and_joins_at_scale.ipynb` | PASS | 20 |
| 21 | `part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb` | `level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb` | PASS | 36 |
| 22 | `part2-ml-at-scale/ch22-streaming-features/notebook.ipynb` | `level2/notebooks/04_streaming_features.ipynb` | PASS | 15 |
| 23 | `part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb` | `level2/notebooks/05_implicit_feedback_and_position_bias.ipynb` | PASS | 22 |
| 24 | `part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb` | `level2/notebooks/06_learning_to_rank.ipynb` | PASS | 68 |
| 25 | `part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb` | `level2/notebooks/07_candidate_generation.ipynb` | PASS | 26 |
| 26 | `part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb` | `level2/notebooks/08_two_tower_retrieval_and_ann.ipynb` | PASS | 60 |
| 27 | `part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb` | `level2/notebooks/09_pytorch_training_done_right.ipynb` | PASS | 48 |
| 28 | `part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb` | `level2/notebooks/10_sequence_models_for_user_behavior.ipynb` | PASS | 74 |
| 29 | `part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb` | `level2/notebooks/11_ab_testing_fundamentals.ipynb` | PASS | 18 |
| 30 | `part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb` | `level2/notebooks/12_advanced_experimentation.ipynb` | PASS | 35 |
| 31 | `part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb` | `level2/notebooks/13_ml_from_scratch_1.ipynb` | PASS | 4 |
| 32 | `part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb` | `level2/notebooks/14_ml_from_scratch_2.ipynb` | PASS | 2 |
| 33 | `part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb` | `level2/notebooks/15_coding_patterns_1.ipynb` | PASS | 2 |
| 34 | `part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb` | `level2/notebooks/16_coding_patterns_2.ipynb` | PASS | 1 |
| 35 | `part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb` | `level2/notebooks/17_serving_a_ranking_cascade.ipynb` | PASS | 33 |
| 36 | `part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb` | `level2/notebooks/18_model_optimization_for_inference.ipynb` | PASS | 41 |
| 37 | `part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb` | `level2/notebooks/19_recsys_and_search_system_design.ipynb` | PASS | 5 |
| 38 | `part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb` | `level2/notebooks/20_final_evaluation_and_mock_loop.ipynb` | PASS | 35 |

### Part III (Chapters 39–58, CPU budget)

| Ch. | Lab (current path) | Recorded as | Result | Seconds |
|---:|---|---|---|---:|
| 39 | `part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb` | `level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb` | PASS | 10 |
| 40 | `part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb` | `level3/notebooks/02_training_efficiency_and_memory.ipynb` | PASS | 40 |
| 41 | `part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb` | `level3/notebooks/03_transformer_from_scratch.ipynb` | PASS | 34 |
| 42 | `part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb` | `level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb` | PASS | 65 |
| 43 | `part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb` | `level3/notebooks/05_fine_tuning_a_classifier.ipynb` | PASS | 271 |
| 44 | `part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb` | `level3/notebooks/06_parameter_efficient_fine_tuning.ipynb` | PASS | 250 |
| 45 | `part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb` | `level3/notebooks/07_embeddings_and_semantic_search.ipynb` | PASS | 119 |
| 46 | `part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb` | `level3/notebooks/08_contrastive_training_and_reranking.ipynb` | PASS | 202 |
| 47 | `part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb` | `level3/notebooks/09_distributed_data_parallel.ipynb` | PASS | 40 |
| 48 | `part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb` | `level3/notebooks/10_scaling_to_large_models.ipynb` | PASS | 9 |
| 49 | `part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb` | `level3/notebooks/11_quantization_and_distillation.ipynb` | PASS | 226 |
| 50 | `part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb` | `level3/notebooks/12_high_throughput_serving.ipynb` | PASS | 74 |
| 51 | `part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb` | `level3/notebooks/13_ctr_prediction_at_scale.ipynb` | PASS | 70 |
| 52 | `part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb` | `level3/notebooks/14_multitask_ranking_and_exploration.ipynb` | PASS | 47 |
| 53 | `part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb` | `level3/notebooks/15_fairness_robustness_and_privacy.ipynb` | PASS | 59 |
| 54 | `part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb` | `level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb` | PASS | 215 |
| 55 | `part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb` | `level3/notebooks/17_ml_platform_design.ipynb` | PASS | 3 |
| 56 | `part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb` | `level3/notebooks/18_large_scale_design_cases.ipynb` | PASS | 3 |
| 57 | `part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb` | `level3/notebooks/19_ml_depth_and_research_discussion.ipynb` | PASS | 96 |
| 58 | `part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb` | `level3/notebooks/20_full_interview_loop_and_portfolio.ipynb` | PASS | 3 |

### Not verified for Parts II and III

* Any run on a GPU, including Colab's T4: the `gpu` budget paths, fp16
  autocast speed-ups, CUDA memory statistics, and the optional bitsandbytes cells.
* The Colab setup cells (recorded as `pip install -r level2/requirements.txt`
  and `level3/requirements.txt`; now `requirements-part2.txt` and
  `requirements-part3.txt`) against Colab's preinstalled versions.
* Multi-GPU and multi-node behaviour: taught with real multi-process gloo code
  plus arithmetic, never measured on more than one machine.
* Timings: measured on a shared 4-core machine and only indicative.

## The book edition (chapters 1–58 and the capstone walkthroughs)

After the course was restructured as a book, every lab and every capstone
walkthrough was executed again, top to bottom, each from an empty artifact
directory, with exercises unattempted: **63/63 pass**. Environment: Python
3.12, PyTorch 2.8 (CPU), Transformers 5.x, numpy 2.0.2, pandas 2.2.2,
scikit-learn 1.6.1, OMP_NUM_THREADS=2, on a shared 4-core machine with no
GPU (Part III therefore ran on its reduced `cpu` budget). The full test
suite (784 tests, including every exercise check against its reference
solution) passes in the same environment.

### Part I

| Lab | Result | Seconds |
|---|---|---:|
| `part1-foundations/ch01-diagnostic/notebook.ipynb` | PASS | 1 |
| `part1-foundations/ch02-reproducible-project/notebook.ipynb` | PASS | 6 |
| `part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb` | PASS | 2 |
| `part1-foundations/ch04-sql-for-ml/notebook.ipynb` | PASS | 3 |
| `part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb` | PASS | 4 |
| `part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb` | PASS | 13 |
| `part1-foundations/ch07-model-selection/notebook.ipynb` | PASS | 26 |
| `part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb` | PASS | 5 |
| `part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb` | PASS | 4 |
| `part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb` | PASS | 4 |
| `part1-foundations/ch11-consistent-serving/notebook.ipynb` | PASS | 4 |
| `part1-foundations/ch12-api-clients-and-tests/notebook.ipynb` | PASS | 17 |
| `part1-foundations/ch13-packaging-and-configuration/notebook.ipynb` | PASS | 6 |
| `part1-foundations/ch14-health-logs-and-latency/notebook.ipynb` | PASS | 23 |
| `part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb` | PASS | 7 |
| `part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb` | PASS | 11 |
| `part1-foundations/ch17-python-and-sql-drills/notebook.ipynb` | PASS | 2 |
| `part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb` | PASS | 3 |

### Part II

| Lab | Result | Seconds |
|---|---|---:|
| `part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb` | PASS | 29 |
| `part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb` | PASS | 13 |
| `part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb` | PASS | 22 |
| `part2-ml-at-scale/ch22-streaming-features/notebook.ipynb` | PASS | 9 |
| `part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb` | PASS | 12 |
| `part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb` | PASS | 48 |
| `part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb` | PASS | 18 |
| `part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb` | PASS | 43 |
| `part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb` | PASS | 36 |
| `part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb` | PASS | 65 |
| `part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb` | PASS | 17 |
| `part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb` | PASS | 25 |
| `part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb` | PASS | 3 |
| `part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb` | PASS | 3 |
| `part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb` | PASS | 2 |
| `part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb` | PASS | 1 |
| `part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb` | PASS | 26 |
| `part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb` | PASS | 27 |
| `part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb` | PASS | 4 |
| `part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb` | PASS | 24 |

### Part III (CPU budget)

| Lab | Result | Seconds |
|---|---|---:|
| `part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb` | PASS | 9 |
| `part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb` | PASS | 29 |
| `part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb` | PASS | 35 |
| `part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb` | PASS | 56 |
| `part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb` | PASS | 165 |
| `part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb` | PASS | 192 |
| `part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb` | PASS | 92 |
| `part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb` | PASS | 161 |
| `part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb` | PASS | 34 |
| `part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb` | PASS | 8 |
| `part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb` | PASS | 187 |
| `part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb` | PASS | 62 |
| `part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb` | PASS | 48 |
| `part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb` | PASS | 35 |
| `part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb` | PASS | 44 |
| `part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb` | PASS | 190 |
| `part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb` | PASS | 2 |
| `part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb` | PASS | 2 |
| `part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb` | PASS | 71 |
| `part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb` | PASS | 2 |

### Part IV · Capstone walkthroughs

| Walkthrough | Data | Result | Seconds |
|---|---|---|---:|
| `part4-capstones/cap1-predictive-maintenance/walkthrough.ipynb` | synthetic sample in the official format (official host unreachable here) | PASS | 63 |
| `part4-capstones/cap2-electricity-load-forecasting/walkthrough.ipynb` | synthetic sample in the official format (official host unreachable here) | PASS | 83 |
| `part4-capstones/cap3-retail-demand-forecasting/walkthrough.ipynb` | real public data | PASS | 60 |
| `part4-capstones/cap4-movie-recommendations/walkthrough.ipynb` | synthetic sample in the official format (official host unreachable here) | PASS | 8 |
| `part4-capstones/cap5-content-moderation/walkthrough.ipynb` | real public data | PASS | 361 |

Capstones 3 (UCI Online Retail II) and 5 (Civil Comments, WILDS release)
ran on the real public data; their README results tables were generated by
those runs. Capstones 1 (NASA C-MAPSS), 2 (ERCOT) and 4 (MovieLens) could not
download their data from this machine: the walkthroughs download from the
official sources in Colab, and here they were verified on synthetic samples
with the same files and formats (set `MLBOOTCAMP_CAPSTONE_SAMPLE=1` to
reproduce). Their results tables remain *TBD* until a run on the real data.
