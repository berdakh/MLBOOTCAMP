# Part III · Deep Learning and Large-Scale Systems

*Chapters 39–58 · about two weeks at three hours a day · Google Colab T4 GPU (every lab also runs on a CPU with a reduced budget)*

Part III covers the deep-learning side of industrial machine learning: GPU
performance and memory, transformers built from scratch and then fine-tuned,
retrieval models, distributed training, efficient inference, advertising and
multi-task ranking, fairness and privacy, the evaluation of systems built on
language models, and system design at the scale of large organisations.

**Prerequisites.** Parts I and II, or equivalent: working PyTorch, and an
understanding of ranking, retrieval and controlled experiments.

## Hardware and budgets

The labs are written for the free Colab T4 GPU (16 GB of memory, fast fp16,
no bf16). In Colab, choose *Runtime → Change runtime type → T4 GPU* before
running. Data and model sizes are not fixed in the code: the function
`scaleml.device.budget()` chooses them for the hardware it finds, and each
lab prints the budget it used.

| Budget | When it is used | Typical size |
|---|---|---|
| `gpu` | a CUDA GPU is present | 20,000 training examples, 128-token sequences, 2 epochs |
| `cpu` | no GPU | 1,500 examples, 64 tokens, 1 epoch: demonstrates the method, not the attainable accuracy |
| `smoke` | `BOOTCAMP_BUDGET=smoke`, used by automated checks | verifies that the code runs |

Distributed training (Chapters 47–48) uses real multi-process
`torch.distributed` code on one machine (the `gloo` backend) and explains
what changes on a multi-GPU cluster. Free Colab provides one GPU, so
multi-GPU speed-ups are derived by calculation rather than measured.

## Data and models

All datasets and models are public and are downloaded from the Hugging Face
Hub on first use: text classification datasets (AG News, IMDB, SST-2), a
scientific-claim retrieval set, a toxicity dataset with identity annotations,
and small pretrained models (DistilBERT, MiniLM sentence encoders, GPT-2
small, a 0.5-billion-parameter instruction model). Each chapter names the
exact dataset and model with a link to its card; read the licence on the card
before reusing them elsewhere. Advertising and multi-task ranking
(Chapters 51–52) use a synthetic dataset.

## Scope and limitations

* A T4 is a 2018 data-centre GPU. It demonstrates every technique at small
  scale; it does not reproduce the absolute numbers of production clusters.
* Multi-GPU and multi-node behaviour is taught with real distributed code
  on processes and with arithmetic, not measured.
* Material on language models concerns the engineering and evaluation of
  systems that use them: fine-tuning small models, retrieval, serving and
  guardrails. Pre-training large models is outside the scope of the book.

## Organisation

| Chapters | Theme |
|---|---|
| 39–40 | GPU performance: execution and memory model, profiling, training memory |
| 41–42 | Transformers: from scratch, and pretrained models with tokenisation |
| 43–44 | Fine-tuning: full fine-tuning against baselines; parameter-efficient methods (LoRA) |
| 45–46 | Embeddings and retrieval: semantic search, contrastive training, re-ranking |
| 47–48 | Distributed training: data parallelism; scaling to large models |
| 49–50 | Efficient inference: quantisation and distillation; high-throughput serving |
| 51–52 | Advertising and multi-objective ranking: CTR prediction, multi-task models, auctions, bandits |
| 53–54 | Responsible systems: fairness, robustness and privacy; evaluating LLM systems and guardrails |
| 55–56 | System design: ML platforms; large-scale design cases |
| 57–58 | Depth questions and research discussion; the full interview loop and portfolio |

## Chapters

<!-- toc:start -->
| Ch. | Chapter | Lab | Time |
|---:|---|---|---|
| 39 | [GPU fundamentals & profiling](ch39-gpu-fundamentals-and-profiling/README.md) | [notebook](ch39-gpu-fundamentals-and-profiling/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb) | 90 min |
| 40 | Training efficiency & memory | [notebook](ch40-training-efficiency-and-memory/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb) | 90 min |
| 41 | A transformer from scratch | [notebook](ch41-transformer-from-scratch/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb) | 90 min |
| 42 | Pretrained transformers & tokenisation | [notebook](ch42-pretrained-transformers-and-tokenization/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb) | 90 min |
| 43 | Fine-tuning a classifier | [notebook](ch43-fine-tuning-a-classifier/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb) | 90 min |
| 44 | Parameter-efficient fine-tuning | [notebook](ch44-parameter-efficient-fine-tuning/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb) | 90 min |
| 45 | Embeddings & semantic search | [notebook](ch45-embeddings-and-semantic-search/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb) | 90 min |
| 46 | Contrastive training & re-ranking | [notebook](ch46-contrastive-training-and-reranking/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb) | 90 min |
| 47 | Distributed data parallel | [notebook](ch47-distributed-data-parallel/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb) | 90 min |
| 48 | Scaling to large models | [notebook](ch48-scaling-to-large-models/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb) | 90 min |
| 49 | [Quantisation & distillation](ch49-quantization-and-distillation/README.md) | [notebook](ch49-quantization-and-distillation/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb) | 90 min |
| 50 | High-throughput serving | [notebook](ch50-high-throughput-serving/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb) | 90 min |
| 51 | CTR prediction at scale | [notebook](ch51-ctr-prediction-at-scale/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb) | 90 min |
| 52 | Multi-task ranking & exploration | [notebook](ch52-multitask-ranking-and-exploration/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb) | 90 min |
| 53 | Fairness, robustness & privacy | [notebook](ch53-fairness-robustness-and-privacy/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb) | 90 min |
| 54 | Evaluating LLM systems & guardrails | [notebook](ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb) | 90 min |
| 55 | ML platform design | [notebook](ch55-ml-platform-design/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb) | 90 min |
| 56 | Large-scale design cases | [notebook](ch56-large-scale-design-cases/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb) | 90 min |
| 57 | ML depth & research discussion | [notebook](ch57-ml-depth-and-research-discussion/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb) | 90 min |
| 58 | Full interview loop & portfolio | [notebook](ch58-full-interview-loop-and-portfolio/notebook.ipynb) · [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb) | 90 min |
<!-- toc:end -->

**Next:** [Part IV · Capstone projects](../part4-capstones/README.md)
