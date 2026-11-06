# Level 3 · Deep Learning & Large-Scale ML Systems

**Ten days, about three hours a day · 20 notebooks · free Colab T4 GPU (every notebook also runs on a CPU with a reduced budget)**

Level 3 prepares you for ML engineer roles at large technology companies,
where the models are deep, the data does not fit on one machine and the
interview loop expects depth. You will profile and speed up GPU training,
build a transformer and then fine-tune pretrained ones, train retrieval
models, distribute training across processes, make inference fast and cheap,
build ads-style CTR and multi-task rankers, evaluate LLM-based systems and
their guardrails, and finish with staff-level system design and a full mock
loop.

**Prerequisites:** Levels 1 and 2, or equivalent: comfortable PyTorch, a
working understanding of ranking, retrieval and experimentation.

## Hardware: what "T4 core" means

Each notebook targets the free Colab T4 GPU (16 GB memory, fast fp16, no
bf16). In Colab, choose **Runtime → Change runtime type → T4 GPU** before
running. Sizes are not hard-coded: `scaleml.device.budget()` picks them for
the hardware it finds, and every notebook prints the budget it used.

| Budget | When | Typical size |
|---|---|---|
| `gpu` | a CUDA GPU is present | 20k training examples, 128-token sequences, 2 epochs |
| `cpu` | no GPU | 1.5k examples, 64 tokens, 1 epoch: shows the method, not the accuracy |
| `smoke` | `BOOTCAMP_BUDGET=smoke`, used by automated checks | proves the code runs |

Distributed training (Day 5) runs *real* multi-process `torch.distributed`
code on one machine (the `gloo` backend on CPU) and explains what changes on
a multi-GPU cluster. Free Colab has one GPU, so multi-GPU speed-ups are
explained with arithmetic, not measured.

## Data and models

All public, downloaded from the Hugging Face Hub on first use and cached:
text classification datasets (AG News, IMDB, SST-2), a scientific-claims
retrieval set, and small pretrained models (for example DistilBERT, MiniLM
sentence encoders, GPT-2 small, a 0.5B instruction model). Each notebook names
the exact dataset and model, with a link to its card. **Read each card's
licence before reusing anything outside this course.** CTR and multi-task
ranking (Day 7) use a synthetic Criteo-like generator, so no licence question
arises.

## The notebooks

| # | Notebook | You learn to | Time |
|---|---|---|---|
| **Day 1 — GPU performance** ||||
| 01 | [GPU fundamentals & profiling](notebooks/01_gpu_fundamentals_and_profiling.ipynb) | The GPU execution and memory model, synchronisation and timing, `torch.profiler`, input-pipeline bottlenecks, fp16 on a T4 | 90 min |
| 02 | [Training efficiency & memory](notebooks/02_training_efficiency_and_memory.ipynb) | Memory arithmetic (weights, gradients, optimiser, activations), gradient accumulation, activation checkpointing, `torch.compile`, debugging OOM | 90 min |
| **Day 2 — Transformers** ||||
| 03 | [A transformer from scratch](notebooks/03_transformer_from_scratch.ipynb) | Build and train a small decoder-only transformer; causal masking; sampling; a KV cache and its speed-up | 90 min |
| 04 | [Pretrained transformers & tokenisation](notebooks/04_pretrained_transformers_and_tokenization.ipynb) | Tokenisers, model heads, pooling, padding and truncation, batched inference throughput | 90 min |
| **Day 3 — Fine-tuning** ||||
| 05 | [Fine-tuning a classifier](notebooks/05_fine_tuning_a_classifier.ipynb) | Full fine-tuning against a linear probe and a TF-IDF baseline; proper evaluation, calibration, error analysis | 90 min |
| 06 | [Parameter-efficient fine-tuning](notebooks/06_parameter_efficient_fine_tuning.ipynb) | LoRA from scratch and with `peft`; memory, time and accuracy against full fine-tuning; merging adapters | 90 min |
| **Day 4 — Embeddings & retrieval** ||||
| 07 | [Embeddings & semantic search](notebooks/07_embeddings_and_semantic_search.ipynb) | Sentence embeddings, FAISS, lexical vs dense vs hybrid retrieval, retrieval evaluation | 90 min |
| 08 | [Contrastive training & re-ranking](notebooks/08_contrastive_training_and_reranking.ipynb) | Fine-tune a bi-encoder with in-batch and hard negatives; a cross-encoder re-ranker; the two-stage trade-off | 90 min |
| **Day 5 — Distributed training** ||||
| 09 | [Distributed data parallel](notebooks/09_distributed_data_parallel.ipynb) | `torch.distributed` by hand: processes, ranks, all-reduce, DDP, samplers, rank-0 logging, checkpoints | 90 min |
| 10 | [Scaling to large models](notebooks/10_scaling_to_large_models.ipynb) | ZeRO/FSDP, tensor and pipeline parallelism, compute budgets, scaling laws, fault tolerance, all by arithmetic and simulation | 90 min |
| **Day 6 — Efficient inference** ||||
| 11 | [Quantisation & distillation](notebooks/11_quantization_and_distillation.ipynb) | int8 quantisation, distilling a fine-tuned model into a small student; accuracy, latency and size | 90 min |
| 12 | [High-throughput serving](notebooks/12_high_throughput_serving.ipynb) | Dynamic batching, throughput vs latency curves, KV cache and continuous batching for generation, cost per 1k requests | 90 min |
| **Day 7 — Ads & multi-objective ranking** ||||
| 13 | [CTR prediction at scale](notebooks/13_ctr_prediction_at_scale.ipynb) | Sparse features, the hashing trick, embedding tables, a deep-and-cross model, normalised entropy, calibration, delayed feedback | 90 min |
| 14 | [Multi-task ranking & exploration](notebooks/14_multitask_ranking_and_exploration.ipynb) | Multi-task models, objective blending, auctions and eCPM, bandits (ε-greedy, Thompson sampling) | 90 min |
| **Day 8 — Responsible and LLM systems** ||||
| 15 | [Fairness, robustness & privacy](notebooks/15_fairness_robustness_and_privacy.ipynb) | Subgroup error analysis, threshold effects, shift and perturbation tests, differential privacy in practice | 90 min |
| 16 | [Evaluating LLM systems & guardrails](notebooks/16_evaluating_llm_systems_and_guardrails.ipynb) | Evaluation sets for generation, LLM-as-judge pitfalls, prompt injection, a guardrail classifier, cost and latency | 90 min |
| **Day 9 — Staff-level system design** ||||
| 17 | [ML platform design](notebooks/17_ml_platform_design.ipynb) | Feature platform, training platform, registry, continuous training and deployment: the trade-offs that come up at staff level | 90 min |
| 18 | [Large-scale design cases](notebooks/18_large_scale_design_cases.ipynb) | Ads ranking, video recommendations, an LLM assistant, real-time fraud: numbers, deep dives, failure modes | 90 min |
| **Day 10 — The loop** ||||
| 19 | [ML depth & research discussion](notebooks/19_ml_depth_and_research_discussion.ipynb) | Depth questions on optimisation, generalisation, transformers and recsys; critiquing a paper; research to production | 90 min |
| 20 | [Full interview loop & portfolio](notebooks/20_full_interview_loop_and_portfolio.ipynb) | A timed full loop with leveling rubrics; packaging your portfolio and capstones | 90 min |

## Honest limits

* A T4 is a 2018-era data-centre GPU. It shows every technique here at small
  scale; it does not reproduce the absolute numbers of production clusters,
  and the notebooks never pretend it does.
* Multi-GPU and multi-node behaviour is taught through real distributed code
  on processes, plus arithmetic. Measuring it needs hardware free Colab does
  not have.
* LLM content in Level 3 is about *engineering and evaluation* of systems
  that use language models: fine-tuning small models, retrieval, serving and
  guardrails. Pre-training large models is out of scope.

## Layout

```
level3/
  notebook_src/   percent-format sources (edit these)
  notebooks/      generated .ipynb (outputs cleared)
  solutions/      written answers + code/ reference solutions graded in CI
  requirements.txt
src/scaleml/      the Level 3 package (device, train, textdata, topic modules, grading/)
```
