# Machine Learning Engineering: From Models to Systems

*A practical textbook with runnable labs, in four parts and 58 chapters.*

This book teaches the engineering that turns a trained model into a system
that other people can rely on: reproducible experiments, data that is correct
at the moment of prediction, evaluation that matches how a model will be used,
services with explicit contracts and tests, deployment, monitoring, and the
ranking, retrieval and deep-learning systems that large technology companies
build. Each chapter has a written text and a hands-on lab that runs in a free
Google Colab runtime.

<img src="figures/roadmap.svg" alt="Overview of the book's progression from a reproducible project to evaluation, serving, operations and demonstration" width="100%">

## Preface

**Who the book is for.** Readers who can already train models, for example
researchers in signal processing, neuroscience or applied statistics, and who
want to work as machine learning engineers, applied scientists or algorithm
engineers. We assume working knowledge of Python, NumPy, pandas, scikit-learn
and the basics of deep learning. We do not assume prior experience with SQL,
software testing, web services, deployment or production monitoring; these
are taught from the beginning.

**What the book covers.** Research training develops modelling skill. It
rarely develops the surrounding practice that industry depends on: pinning the
code, data and environment so a result can be reproduced; computing features
only from information available at prediction time; choosing a data split
that answers the question being asked; turning a score into a decision under
stated costs; serving the same transformation that was trained; and noticing
when a deployed model stops working. Part I teaches this practice on one small
project. Parts II and III apply it to the systems that dominate industrial
machine learning: ranking and recommendation over large logs, and deep
learning on GPUs. Part IV contains five projects on public data with complete
guided solutions.

**How each chapter is organised.** Every chapter is a folder containing:

| File | Purpose |
|---|---|
| `README.md` | The chapter text: learning objectives, explanations, figures, equations, key terms, summary, further reading |
| `notebook.ipynb` | The lab: runnable code, exercises with automatic checks, and checkpoint questions |
| `solutions.md` | Worked answers to every exercise and checkpoint question |

Read the chapter text, then work through the lab. Attempt each exercise before
opening the solutions; the automatic checks in the lab tell you whether your
code is correct without revealing the answer. Exercise checks print `[PASS]`,
`[FAIL]` (with a hint) or `[TODO]`.

**How long it takes.** At about three hours a day, Part I takes about one
week, and Parts II and III about two weeks each. A capstone takes four to six
days. [Appendix: study plan](appendices/study-plan.md) gives a schedule, and
[Appendix: readiness rubric](appendices/readiness-rubric.md) describes the
level of skill each part aims at.

**What a short course cannot do.** Working through the book gives you
understanding, a vocabulary for the common failure modes, and finished
projects you can explain. It does not by itself give fluency: writing these
things quickly from a blank file, or answering design questions under time
pressure, requires repeated practice over weeks. The book's drills and mock
interviews are a starting point for that practice.

## How to begin

**In Google Colab (no installation).** Open any lab from the table of contents
below and choose *Runtime → Run all*. The first cell downloads this repository
and installs what is missing. Every lab rebuilds what it needs from earlier
chapters, so chapters can be opened in any order and in a fresh runtime.
Part III is written for Colab's free T4 GPU (*Runtime → Change runtime type*);
it also runs on a CPU with smaller data, more slowly.

**On your own computer.**

```bash
git clone https://github.com/berdakh/MLBOOTCAMP.git
cd MLBOOTCAMP
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # Part I; Parts II-III: requirements-part2.txt / -part3.txt
python -m pytest                                     # about a minute, no network needed
jupyter lab part1-foundations/ch01-diagnostic/notebook.ipynb
```

Begin with [Chapter 1, the skills diagnostic](part1-foundations/ch01-diagnostic/README.md).
It takes about an hour and shows which chapters deserve the most attention.

## Contents

<!-- toc:start -->
### [Part I · Foundations of ML Engineering](part1-foundations/README.md)

| Ch. | Chapter | Lab | Time |
|---:|---|---|---|
| 1 | [Skills diagnostic](part1-foundations/ch01-diagnostic/README.md) | [notebook](part1-foundations/ch01-diagnostic/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb) | 60 min |
| 2 | [A reproducible ML project](part1-foundations/ch02-reproducible-project/README.md) | [notebook](part1-foundations/ch02-reproducible-project/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb) | 40 min |
| 3 | [Data acquisition & validation](part1-foundations/ch03-data-acquisition-and-validation/README.md) | [notebook](part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb) | 45 min |
| 4 | [SQL for ML](part1-foundations/ch04-sql-for-ml/README.md) | [notebook](part1-foundations/ch04-sql-for-ml/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb) | 55 min |
| 5 | [Leakage-safe splits & a baseline](part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md) | [notebook](part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb) | 45 min |
| 6 | [Pipelines & cross-validation](part1-foundations/ch06-pipelines-and-cross-validation/README.md) | [notebook](part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb) | 50 min |
| 7 | [Model selection](part1-foundations/ch07-model-selection/README.md) | [notebook](part1-foundations/ch07-model-selection/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb) | 45 min |
| 8 | [Thresholds & calibration](part1-foundations/ch08-thresholds-and-calibration/README.md) | [notebook](part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb) | 40 min |
| 9 | [Error analysis & artifacts](part1-foundations/ch09-error-analysis-and-artifacts/README.md) | [notebook](part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb) | 45 min |
| 10 | [FastAPI & contracts](part1-foundations/ch10-fastapi-and-contracts/README.md) | [notebook](part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb) | 55 min |
| 11 | [Consistent serving](part1-foundations/ch11-consistent-serving/README.md) | [notebook](part1-foundations/ch11-consistent-serving/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb) | 45 min |
| 12 | [API clients & tests](part1-foundations/ch12-api-clients-and-tests/README.md) | [notebook](part1-foundations/ch12-api-clients-and-tests/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb) | 60 min |
| 13 | [Packaging & configuration](part1-foundations/ch13-packaging-and-configuration/README.md) | [notebook](part1-foundations/ch13-packaging-and-configuration/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb) | 45 min |
| 14 | [Health, logs & latency](part1-foundations/ch14-health-logs-and-latency/README.md) | [notebook](part1-foundations/ch14-health-logs-and-latency/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb) | 55 min |
| 15 | [Monitoring & rollback](part1-foundations/ch15-monitoring-and-rollback/README.md) | [notebook](part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb) | 60 min |
| 16 | [Final evaluation & documentation](part1-foundations/ch16-final-evaluation-and-docs/README.md) | [notebook](part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb) | 50 min |
| 17 | [Python & SQL drills](part1-foundations/ch17-python-and-sql-drills/README.md) | [notebook](part1-foundations/ch17-python-and-sql-drills/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb) | 60 min |
| 18 | [System design & mock interview](part1-foundations/ch18-system-design-and-mock-interview/README.md) | [notebook](part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb) | 70 min |

### [Part II · Machine Learning at Scale](part2-ml-at-scale/README.md)

| Ch. | Chapter | Lab | Time |
|---:|---|---|---|
| 19 | [Event logs & columnar data](part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md) | [notebook](part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb) | 90 min |
| 20 | [Aggregation & joins at scale](part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md) | [notebook](part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb) | 90 min |
| 21 | [Feature pipelines & feature stores](part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md) | [notebook](part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb) | 90 min |
| 22 | [Streaming features](part2-ml-at-scale/ch22-streaming-features/README.md) | [notebook](part2-ml-at-scale/ch22-streaming-features/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb) | 90 min |
| 23 | [Implicit feedback & position bias](part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md) | [notebook](part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb) | 90 min |
| 24 | [Learning to rank](part2-ml-at-scale/ch24-learning-to-rank/README.md) | [notebook](part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb) | 90 min |
| 25 | [Candidate generation](part2-ml-at-scale/ch25-candidate-generation/README.md) | [notebook](part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb) | 90 min |
| 26 | [Two-tower retrieval & ANN](part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md) | [notebook](part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb) | 90 min |
| 27 | [PyTorch training done right](part2-ml-at-scale/ch27-pytorch-training-done-right/README.md) | [notebook](part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb) | 90 min |
| 28 | [Sequence models for user behaviour](part2-ml-at-scale/ch28-sequence-models-for-user-behavior/README.md) | [notebook](part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb) | 90 min |
| 29 | [A/B testing fundamentals](part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md) | [notebook](part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb) | 90 min |
| 30 | [Advanced experimentation](part2-ml-at-scale/ch30-advanced-experimentation/README.md) | [notebook](part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb) | 90 min |
| 31 | [ML from scratch I](part2-ml-at-scale/ch31-ml-from-scratch-1/README.md) | [notebook](part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb) | 90 min |
| 32 | [ML from scratch II](part2-ml-at-scale/ch32-ml-from-scratch-2/README.md) | [notebook](part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb) | 90 min |
| 33 | [Coding patterns I](part2-ml-at-scale/ch33-coding-patterns-1/README.md) | [notebook](part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb) | 90 min |
| 34 | [Coding patterns II](part2-ml-at-scale/ch34-coding-patterns-2/README.md) | [notebook](part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb) | 90 min |
| 35 | [Serving a ranking cascade](part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md) | [notebook](part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb) | 90 min |
| 36 | [Model optimisation for inference](part2-ml-at-scale/ch36-model-optimization-for-inference/README.md) | [notebook](part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb) | 90 min |
| 37 | [Recommendation & search system design](part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md) | [notebook](part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb) | 90 min |
| 38 | [Final evaluation & mock loop](part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/README.md) | [notebook](part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb) | 90 min |

### [Part III · Deep Learning and Large-Scale Systems](part3-deep-learning-systems/README.md)

| Ch. | Chapter | Lab | Time |
|---:|---|---|---|
| 39 | [GPU fundamentals & profiling](part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md) | [notebook](part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb) | 90 min |
| 40 | [Training efficiency & memory](part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md) | [notebook](part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb) | 90 min |
| 41 | [A transformer from scratch](part3-deep-learning-systems/ch41-transformer-from-scratch/README.md) | [notebook](part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb) | 90 min |
| 42 | [Pretrained transformers & tokenisation](part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/README.md) | [notebook](part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb) | 90 min |
| 43 | [Fine-tuning a classifier](part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md) | [notebook](part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb) | 90 min |
| 44 | [Parameter-efficient fine-tuning](part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md) | [notebook](part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb) | 90 min |
| 45 | [Embeddings & semantic search](part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md) | [notebook](part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb) | 90 min |
| 46 | [Contrastive training & re-ranking](part3-deep-learning-systems/ch46-contrastive-training-and-reranking/README.md) | [notebook](part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb) | 90 min |
| 47 | [Distributed data parallel](part3-deep-learning-systems/ch47-distributed-data-parallel/README.md) | [notebook](part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb) | 90 min |
| 48 | [Scaling to large models](part3-deep-learning-systems/ch48-scaling-to-large-models/README.md) | [notebook](part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb) | 90 min |
| 49 | [Quantisation & distillation](part3-deep-learning-systems/ch49-quantization-and-distillation/README.md) | [notebook](part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb) | 90 min |
| 50 | [High-throughput serving](part3-deep-learning-systems/ch50-high-throughput-serving/README.md) | [notebook](part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb) | 90 min |
| 51 | [CTR prediction at scale](part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md) | [notebook](part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb) | 90 min |
| 52 | [Multi-task ranking & exploration](part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md) | [notebook](part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb) | 90 min |
| 53 | [Fairness, robustness & privacy](part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md) | [notebook](part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb) | 90 min |
| 54 | [Evaluating LLM systems & guardrails](part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/README.md) | [notebook](part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb) | 90 min |
| 55 | [ML platform design](part3-deep-learning-systems/ch55-ml-platform-design/README.md) | [notebook](part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb) | 90 min |
| 56 | [Large-scale design cases](part3-deep-learning-systems/ch56-large-scale-design-cases/README.md) | [notebook](part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb) | 90 min |
| 57 | [ML depth & research discussion](part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md) | [notebook](part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb) | 90 min |
| 58 | [Full interview loop & portfolio](part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/README.md) | [notebook](part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb) | 90 min |

### [Part IV · Capstone projects](part4-capstones/README.md)

| | Capstone | Guided solution |
|---|---|---|
| 1 | [Capstone 1 · Predictive maintenance (NASA C-MAPSS)](part4-capstones/cap1-predictive-maintenance/README.md) | [walkthrough](part4-capstones/cap1-predictive-maintenance/walkthrough.ipynb) |
| 2 | [Capstone 2 · Electricity load forecasting (ERCOT)](part4-capstones/cap2-electricity-load-forecasting/README.md) | [walkthrough](part4-capstones/cap2-electricity-load-forecasting/walkthrough.ipynb) |
| 3 | [Capstone 3 · Retail demand forecasting (UCI Online Retail)](part4-capstones/cap3-retail-demand-forecasting/README.md) | [walkthrough](part4-capstones/cap3-retail-demand-forecasting/walkthrough.ipynb) |
| 4 | [Capstone 4 · Movie recommendations (MovieLens)](part4-capstones/cap4-movie-recommendations/README.md) | [walkthrough](part4-capstones/cap4-movie-recommendations/walkthrough.ipynb) |
| 5 | [Capstone 5 · Content moderation triage (Civil Comments)](part4-capstones/cap5-content-moderation/README.md) | [walkthrough](part4-capstones/cap5-content-moderation/walkthrough.ipynb) |
<!-- toc:end -->

## Appendices

| Appendix | Contents |
|---|---|
| [Glossary](appendices/glossary.md) | Every term used in the book, defined, with the chapters where it is taught |
| [Equations](appendices/equations.md) | The key equations of all chapters, grouped by topic, each stated in words |
| [List of figures](appendices/figures.md) | Every figure, with a caption |
| [Setup](appendices/setup.md) · [Colab](appendices/colab.md) · [Before you start](appendices/before-you-start.md) | Installing, running in Colab, keeping work across sessions |
| [Troubleshooting](appendices/troubleshooting.md) | Common failures and their causes |
| [Interview questions](appendices/interview-questions.md) · [Cheat sheet](appendices/cheatsheet.md) | Questions by topic with what a complete answer contains; reference snippets |
| [Study plan](appendices/study-plan.md) · [Readiness rubric](appendices/readiness-rubric.md) · [Diagnostic guide](appendices/diagnostic.md) | Planning and self-assessment |
| [Verification](appendices/verification.md) | Exactly what was executed when the book was built, where, and what was not |
| [About the solutions](appendices/solutions.md) | How solutions are organised and how their confidence is marked |

## The data used in the book

* **Part I** uses *TaskFlow*, a synthetic subscription-software company whose
  customers may cancel ("churn"). The data is generated once, committed to
  [`data/raw/`](data/README.md) and checksummed. Synthetic data was chosen
  deliberately: it is available in Colab without an account, small enough to
  train on a CPU in seconds, relational (so SQL is needed), and it contains
  planted defects with known ground truth. Chapter 2 explains the churn
  problem for readers who have not met it.
* **Part II** uses *ShopStream*, a synthetic marketplace in which the true
  relevance of every (user, item) pair is known. This makes it possible to
  check whether a correction for position bias actually recovers relevance,
  which real click logs never allow.
* **Part III** uses public datasets and pretrained models from the Hugging
  Face Hub (each chapter names them and their licences) and a synthetic
  advertising dataset.
* **Part IV** uses public data from NASA, ERCOT, the UCI repository,
  GroupLens and Jigsaw/Google, each under its own terms, stated in the
  project's text.

No synthetic dataset in this book describes real people or companies.

## Repository layout

```
part1-foundations/            Chapters 1-18   (each chNN-*/ holds README.md, notebook, solutions)
part2-ml-at-scale/            Chapters 19-38
part3-deep-learning-systems/  Chapters 39-58
part4-capstones/              Five projects with guided solutions, and a project template
appendices/                   Glossary, equations, figures, setup, study aids
figures/                      All figures (SVG), generated by tools/make_diagrams.py
src/                          The book's Python packages:
    churnlab/                   Part I project (data, features, models, API, monitoring)
    shoprank/                   Part II project (ShopStream, ranking, retrieval, experiments)
    scaleml/                    Part III toolkit (devices, training loop, models, serving)
    grading/                    Exercise checks, one module per chapter (chNN.py)
data/                         The committed Part I dataset
deploy/                       Reference deployment files (Chapter 13)
tests/  tools/                Tests and build tools (see CONTRIBUTING.md)
```

## For maintainers

```bash
make test            # unit tests, exercise checks against reference solutions
make notebooks       # rebuild every notebook.ipynb from its notebook.py
make check           # everything CI checks
```

Labs are written as plain Python files (`notebook.py`, percent format) and
the `.ipynb` files are generated from them, so changes are readable in a
diff. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Code is released under the [MIT licence](LICENSE); text and figures under
CC BY 4.0. Public datasets used in Parts III and IV keep their own licences,
stated where they are used.
