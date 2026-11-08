# Glossary

The course dictionary for Levels 1–3: every term from every notebook's "📖 Words to know" table, merged into one A–Z, with links to where each is taught. Where a term has a research counterpart, it is given in brackets. For the maths, see the [equation sheet](equations.md); for pictures, the [diagram index](figures.md).

**Lost?** Start with the plain-language tables below, then look the word up in the A–Z. Every analogy is a crutch, not a definition; where one can mislead, the table says so.

*Taught in* links open the notebook on GitHub. L1 = Level 1 (churn, CPU), L2 = Level 2 (recommendation and search at scale), L3 = Level 3 (deep learning, GPUs and large-scale design).

**Jump to:** [In plain terms](#in-plain-terms) · [0–9](#0-9-and-symbols) · [A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [F](#f) · [G](#g) · [H](#h) · [I](#i) · [J](#j) · [K](#k) · [L](#l) · [M](#m) · [N](#n) · [O](#o) · [P](#p) · [Q](#q) · [R](#r) · [S](#s) · [T](#t) · [U](#u) · [V](#v) · [W](#w) · [Z](#z)

---

## In plain terms

### Data and leakage

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Leakage** | Sitting an exam with the answer sheet visible through the envelope. Scores are great; nothing was learned. | Rarely that blatant: usually one innocent-looking column, or a split that let the future into training. |
| **Point-in-time correctness** | Reading only the newspapers printed *before* the day you are predicting. | "Printed before" must mean *available* before: a record back-dated later is still from the future. |
| **Snapshot date (T)** | The moment you freeze the world and ask "what happens next?". | One customer has many snapshots, so rows are not independent (see *Cluster bootstrap*). |
| **Embargo (gap)** | A no-man's-land between training and test, so no label straddles the border. | Its width is the label horizon, not a round number you pick. |
| **Training/serving skew** | A recipe written twice, from memory, by two cooks. The dish tastes different and nobody knows why. | The fix is not "be careful": it is one shared function used in both places. |
| **Fan-out** | A join that photocopies rows. Sums and counts come out inflated, silently. | The output still looks plausible; only a row-count check catches it. |
| **Feature store** | A shared pantry: the same labelled ingredients for the training kitchen and the serving kitchen. | It stores values, not correctness: a leaky feature in the store is leaky everywhere. |

### Evaluation and decisions

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Baseline** | The "do nothing clever" bar. A model that cannot clear it has not earned its complexity. | A weak baseline flatters everything; report the strongest simple one. |
| **Precision / recall** | Precision: how many of the people you called were really leaving. Recall: how many of the leavers you called. | Neither knows what a call costs or what a saved customer is worth; that is the threshold's job. |
| **PR-AUC** | How well the top of your list is packed with positives, across every cut-off. | Its floor is the positive rate, so 0.15 can be good (5% base) or bad (40% base). |
| **ROC-AUC** | The chance a random positive is scored above a random negative. | With rare positives it stays high while the top of the list is mostly wrong. |
| **Threshold** | The line on a ruler above which you act. | The model gives scores; the business picks the line, from costs and capacity. |
| **Calibration** | A weather forecaster whose "70%" days rain 70% of the time. | A perfectly calibrated model can still be useless at ranking (always say the base rate). |
| **Bootstrap** | Re-running the exam on reshuffled copies of the question paper to see how much the score wobbles. | Resample the right unit: customers, not rows, when rows come in groups. |
| **Held-out test set** | The sealed envelope you open once, at the end. | Open it twice and it becomes a second validation set. |
| **Slice** | Checking the class average subject by subject, not just overall. | Small slices have wide intervals; a "bad" slice may be noise. |

### Experimentation

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **A/B test** | A fair coin decides who gets the new version, so any difference is caused by it. | Only if the coin was fair: check the split (sample-ratio mismatch) before the result. |
| **MDE** | The smallest change your test is big enough to notice. | Smaller real effects exist; your test just cannot see them. |
| **Peeking** | Checking the scoreboard every minute and stopping when your team is ahead. | Every look is another chance for a false win, unless the test is designed for it. |
| **CUPED** | Grading improvement rather than final height: subtract what you could predict from last month. | Only uses data from *before* randomisation; anything later can absorb the effect. |
| **Interleaving** | A blind taste test: both rankers' items mixed in one list, and the user's clicks vote. | Says which ranker users prefer, not how much revenue changes. |

### Ranking and recommendation

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Cascade** | A hiring funnel: a cheap screen for millions, a careful interview for a few hundred. | Whatever the first stage drops, no later stage can bring back. |
| **Two-tower model** | Users and items each get a postcode; recommend the items that live nearby. | User and item never meet inside the model, so fine interactions are lost; that is the ranker's job. |
| **ANN** | Looking for a book by walking to the right shelf instead of reading every spine. | Fast because it sometimes skips the right shelf: measure its recall. |
| **NDCG@k** | Marks for putting the best answers first, with less credit for each step down the page. | Needs relevance labels; clicks are a biased stand-in. |
| **Position bias** | People click the first result partly because it is first. | Not all of it: some of the top item's clicks are deserved. |
| **IPS** | Giving extra weight to the votes from people who rarely got a ballot. | Rare positions get huge weights and noisy estimates, hence clipping. |
| **Cold start** | A new shop with no reviews yet. | Content features help, but only exposure creates real feedback. |

### Deep learning, transformers and GPUs

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Embedding** | A position on a map where similar things sit close together. | "Close" means used in similar ways, not "means the same": opposites often sit together. |
| **Attention** | Each word asks a question, every other word holds up a label, and the best matches lend their meaning. | Cost grows with the square of the sequence length. |
| **KV cache** | Keeping your notes from earlier pages so you do not reread the whole book for every new word. | The notes grow with every token and become the memory limit when serving. |
| **LoRA** | Sticky notes on a manual instead of reprinting the manual. | The notes are low-rank: a task that needs a big rewrite will not fit on them. |
| **Quantisation** | Rounding every price to the nearest coin to make the receipt smaller. | Some weights and activations matter far more than others; outliers break naive rounding. |
| **Distillation** | A student copying a teacher's reasoning (how sure, about what), not just the final answers. | The student can only be as good as what the teacher shows it on the transfer set. |
| **Mixed precision (autocast)** | Doing rough work on scrap paper and the final sums in ink. | fp16 underflows without loss scaling; bf16 avoids that at the cost of precision. |
| **Gradient accumulation** | Paying for a large order in instalments: the same total, in smaller pieces. | Same gradient, not the same speed: it trades memory for time. |
| **DDP** | Several students each mark a pile of papers, then agree on one average grade before the next pile. | The agreeing (all-reduce) costs time that grows with model size. |
| **Arithmetic intensity** | How much cooking you do per trip to the fridge. | Low intensity means the GPU waits on memory however fast its maths is. |

### Serving and operations

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **Idempotent** | A lift button: pressing it five times calls one lift. | "Create an order" is not idempotent unless you design it to be (an idempotency key). |
| **Exponential backoff** | Knocking again, waiting longer each time, so you do not hammer a door nobody can open. | Retries help with blips, not outages; cap them. |
| **Tail latency** | The slowest checkout queue in the supermarket, which is the one you remember. | A page waiting on 20 calls almost always hits someone's tail. |
| **Little's law** | Queue length = arrival rate × time each person spends inside. | Holds for averages over a stable period, not during a spike. |
| **SLO** | The promise on the menu: "served within 10 minutes, 99 times in 100". | An SLO with no error budget and no consequence is a wish. |
| **Canary release** | A miner's canary: send a little traffic first and watch it closely. | A small canary cannot detect a small regression; size it like an experiment. |
| **Rollback** | An undo button you have actually tested. | Undo fixes the model, not the decisions it already made. |
| **Data drift** | The customers walking in look different from last year's. | Drift is not damage: inputs can move with no harm, and harm can come with no input drift (concept drift). |
| **Concept drift** | The same customers now behave differently: the rules changed. | You only see it once labels arrive, which can take weeks. |

### Ads, fairness and privacy

| Term | Think of it as | Where it breaks down |
|---|---|---|
| **eCPM** | Expected money per thousand showings: bid × chance of a click. | Only as good as pCTR's calibration: a 10% calibration error is a 10% pricing error. |
| **Second-price auction** | Win by bidding highest, pay what it took to beat the runner-up. | Real ad auctions add reserves, quality scores and pacing, so "bid your value" is a simplification. |
| **Multi-armed bandit** | Trying restaurants: mostly go to your favourite, sometimes try a new one. | Assumes the restaurants do not change; real CTRs drift. |
| **Differential privacy (ε, δ)** | A crowd photo where no single face changes the picture much. | Protects individuals, not groups; and every query spends privacy budget. |
| **Subgroup AUC** | Grading the model's ranking only among comments that mention one group. | A small subgroup gives a noisy number; report its interval. |

---

## A to Z

## 0–9 and symbols

**3D parallelism** — Tensor × pipeline × data parallelism combined. *Example:* TP inside a node, PP across nodes, DP across replicas. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**413 Content Too Large** — The request is bigger than the service accepts. *Example:* a batch of 2,500 rows. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**422 Unprocessable Content** — The body parsed but broke the schema. *Example:* `tenure_days = -5`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**503 Service Unavailable** — The service cannot answer right now. *Example:* no model loaded yet. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

[Back to top](#glossary)

## A

**A/A test** — An experiment where both arms get the identical treatment, so every "win" is a false positive. *Example:* 400 simulated A/A runs measure each test's false-positive rate. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**A/B test** — Splitting live traffic between two versions for long enough to measure the effect on an outcome. The causal test of whether a model helps. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb), [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb). *See also:* [CUPED](#c), [Power (1 − β)](#p), [MDE](#m).

**Accidental hit** — The user's own positive appearing again as a negative. *Example:* masked in `rt.in_batch_softmax_loss`. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

**Activation checkpointing** — Keeping only some activations and recomputing the rest during backward. *Example:* 82 MB → 14 MB in 2.4, for ~+30% time. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Activations** — Intermediate tensors from the forward pass that backward needs, kept until it uses them. *Example:* 82 MB for one batch here, 5.8× the model state. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb). *See also:* [Model state](#m).

**Adapter** — The small set of trained weights that turns a base model into a task model. *Example:* the 2.83 MB safetensors file in 6.6. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb). *See also:* [Safetensors](#s).

**Adjacency list** — For each node, the list of its neighbours. *Example:* `graph[u] = [v, ...]`. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**ADR** — Architecture decision record: one decision, its options, consequences and revisit triggers. *Example:* `adr-001-online-feature-store.md`. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Agreement** — Share of examples where two models give the same prediction. *Example:* int8 vs fp32 agree on ~96%. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Alert severity** — *page* wakes someone now; *warn* opens a ticket. *Example:* error rate → page; drift → warn. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Algebraic aggregate** — Partial results merge exactly into a fixed-size state. *Example:* count, sum, avg as (sum, count). *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Allowed lateness** — How long to wait for stragglers before closing a window. *Example:* 0, 10 min, 1 hour in 4.4. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**All-reduce** — Every rank contributes a tensor and every rank receives the sum (or max, …). *Example:* sum = 10 for 1 + 2 + 3 + 4. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**α (alpha)** — A scale for the update, applied as α / r so changing r does not change its size. *Example:* α = 16, so α / r = 2. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**ALS** — Alternating least squares: fix items, solve users; fix users, solve items. *Example:* a few seconds here. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Amortised cost** — Average cost per step over a whole run, even if some steps are expensive. *Example:* each item pushed and popped at most once. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Amortised O(1)** — Cheap on average over many operations, even if one is occasionally slow. *Example:* `list.append` when the list resizes. *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**ANN search** — Approximate nearest-neighbour search: fast, slightly incomplete. *Example:* FAISS. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb), [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb), [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**App factory** — A function (`create_app(settings)`) that builds a new web app from configuration, instead of creating one at import time. Makes testing and configuration easy. *Example:* `create_app(settings)`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Arithmetic intensity** — FLOPs done per byte moved to or from memory. *Example:* `x + y`: 1/12; a 1024 × 1024 matmul: ~170. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Artifact** — Any file a run produces and a later step uses: a feature table, a model bundle, a report. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb), [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb). *See also:* [Bundle](#b).

**Artifact directory** — Where this course writes artifacts. Chosen by `BOOTCAMP_ARTIFACTS`, then Drive (`USE_DRIVE`), then `<repo>/artifacts`. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb), [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb). *See also:* [`USE_DRIVE`](#u).

**As-of join** — Attach the latest right row with timestamp ≤ the left row's. *Example:* the price when the item was shown. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Asynchronous launch** — The CPU puts a kernel in a queue and carries on without waiting for the result. *Example:* the "without sync" time in 1.2 is only the launch. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**At-least-once** — Every event arrives, some more than once. *Example:* 2% re-delivered impressions. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Atomic download** — Write to a temporary name and rename only when complete and verified. *Example:* `fetch` writes `<name>.part`, then renames. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Attack success rate (ASR)** — Share of injected inputs where the attacker got what they wanted. *Example:* 0.67 (terse), 0.50 (schema). *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Attention** — The operation that lets each position build its representation as a weighted average of other positions' values, with weights from query–key similarity. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb), [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Attention head** — One independent attention computation on a slice of the channels. *Example:* 4 heads of 16 channels each. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb). *See also:* [Attention](#a).

**Attention mask** — 1 for real tokens, 0 for padding; masked keys get −∞ before the softmax. *Example:* `batch["attention_mask"]`. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Audit log** — An append-only record of who changed what and why. *Example:* `history.jsonl`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

[Back to top](#glossary)

## B

**Backfill** — Computing a new feature for past dates. *Example:* recomputing day 30 alone must equal the full run. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Background** — All test comments that do *not* mention that identity. *Example:* the comparison population for BPSN / BNSP. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Backoff (exponential)** — Waiting longer after each failed retry (0.2 s, 0.4 s, 0.8 s, …) so a struggling server is not overwhelmed. Usually with random jitter. *Example:* 0.2 s, 0.4 s, 0.8 s. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb). *See also:* [Retry](#r), [Jitter](#j).

**Backprop** — The chain rule applied from the loss back to every parameter, reusing stored forward values. *Example:* `mlp_grads` in Exercise 3. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb). *See also:* [Parameter (Ψ)](#p).

**Baseline** — The simplest reasonable model or rule, used to judge whether complexity earns its place. A dummy (base-rate) baseline and a simple model are both worth reporting. *Example:* prior, hand rule, small logistic regression. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [Hand rule](#h).

**Batching** — Sending many rows in one request so fixed costs are paid once. *Example:* 1,000 rows per call. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Batch scoring** — Scoring many rows on a schedule and writing results somewhere, rather than answering requests. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**Batch vs online scoring** — Precompute scores on a schedule vs compute per request. *Example:* daily list vs live badge. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**`BCEWithLogitsLoss`** — Binary cross-entropy that applies the sigmoid itself, stably. *Example:* the click model's loss. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb). *See also:* [Cross-entropy (log loss)](#c).

**bf16** — A 16-bit format with fp32's exponent range and less precision. *Example:* A100/H100 training; no scaler needed. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**BFS** — Breadth-first search: visit nodes in layers of distance, using a queue. *Example:* `shortest_steps` on a maze. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Biased vs unbiased variance** — Dividing by n vs n − 1. *Example:* the forward pass uses n, the running variance n − 1. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Bitwise reproducible** — Two runs give identical floats, not just close ones. *Example:* `==` on the loss lists in 9.5. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Blending with quotas** — Merging generators with a fixed number of slots each. *Example:* 70 ALS + 30 category popularity. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**BM25** — A classic lexical relevance score built from term frequency, rarity and document length. *Example:* lexical retrieval in design B. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb), [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb). *See also:* [Lexical retrieval](#l).

**BNSP AUC** — Non-toxic background vs toxic subgroup comments: low means toxic posts about the group are missed. *Example:* 0.91–0.95 (high). *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb). *See also:* [Group (query)](#g).

**Bootstrap** — Estimating uncertainty by resampling the data with replacement and recomputing a statistic many times. *Example:* "is 0.745 really above 0.741?". *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb), [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb). *See also:* [Confidence interval (CI)](#c), [Cluster (group) bootstrap](#c), [Paired bootstrap](#p).

**Bootstrap CI** — A confidence interval from re-sampling the test queries many times. *Example:* the 95% intervals in 8.3. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb). *See also:* [Confidence interval (CI)](#c).

**BPSN AUC** — Toxic background vs non-toxic subgroup comments: low means benign mentions score like toxicity. *Example:* 0.67–0.75 for *black*, *muslim*. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Break-even probability $p^{*}$** — The probability above which a call pays for itself. *Example:* $30 / (150 + 30) = 1/6$. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Brier score** — Mean squared difference between predicted probability and outcome. Lower is better; measures probability quality. *Example:* 0.0467 vs 0.0494 for the base rate. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Broadcasting** — NumPy stretching size-1 axes so two arrays line up without copying. *Example:* `A[:, None, :] - B[None, :, :]` makes an (m, n, d) array. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Broadcast join** — Copy the small table to every worker; no shuffle of the big one. *Example:* impressions ⋈ sellers. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Brute force** — The obvious correct solution, usually too slow, stated first as a baseline. *Example:* checking every pair: O(n²). *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Budget pacing** — Spreading an advertiser's daily budget over the day. *Example:* participation rate 0.40. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Bundle** — In this course, an immutable directory with `model.joblib` and `metadata.json` (features, versions, threshold, data fingerprint, metrics, hash). *Example:* `model.joblib` + `metadata.json`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb). *See also:* [Fingerprint](#f).

[Back to top](#glossary)

## C

**Caching allocator** — PyTorch keeps freed GPU memory reserved for reuse instead of returning it. *Example:* `memory_reserved` > `memory_allocated`. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Calibration** — Agreement between predicted probabilities and observed frequencies. Needed when decisions use probability values. *Example:* needed before a cost-based threshold. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb), [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb), [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb). *See also:* [Temperature scaling](#t), [Brier score](#b), [Reliability diagram](#r).

**Canary release** — Sending a small share of traffic to a new version and increasing it if metrics hold. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Candidate** — One model configuration being compared. *Example:* `dummy`, `logreg`, `hgb`, `hgb_deeper`. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb).

**Candidate generation** — Picking a few hundred items out of the catalogue, cheaply. *Example:* retrieve 100 of 4,000. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb), [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Candidate source** — One retrieval method that proposes items to rank. *Example:* two-tower ANN, co-visitation, trending. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb). *See also:* [Co-visitation](#c).

**Capacity / top-k** — A fixed number of calls per period, filled with the highest scores. *Example:* 40 calls a month. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Cardinality** — How many distinct values a categorical field can take. *Example:* 4 to 1,000,000 per field. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Cascade** — A chain of stages, each cheaper per item than the next, that shrinks the candidate set. *Example:* 4,000 items → ~300 candidates → top 10. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Catalogue NDCG@10** — The shown top 10 scored against the user's true best 10 in the whole catalogue. *Example:* cascade 0.875 without the category cap. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb). *See also:* [Cascade](#c).

**Catastrophic forgetting** — A too-large learning rate overwrites what pre-training learned. *Example:* why 5e-5, not 1e-3. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Catch-up cell** — The second code cell of every notebook; calls `churnlab.bootstrap.ensure_*` to rebuild earlier outputs. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb). *See also:* [Bootstrap](#b).

**Causal mask** — Blocks attention from position i to any later position j > i. *Example:* `causal_mask(n)`, Exercise 2. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb), [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb). *See also:* [Attention](#a).

**Checkpoint** — Everything needed to continue a run exactly. *Example:* `click_mlp.pt`: model, optimizer, scheduler, RNG. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Checksum** — A hash (here SHA-256) of a file's bytes, used to confirm a file is exactly the expected one. *Example:* `verify_raw` compares each raw file's hash. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb), [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Chunking** — Splitting a big job into requests below the service's limit. *Example:* 2,500 rows as 1,000 + 1,000 + 500. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Churn** — A customer ending their subscription. *Example:* a TaskFlow account cancels in March. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**Churn rate** — Share of customers at the start of a period who leave during it. *Example:* ≈ 5.5% of active accounts per month in this data. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**CI** — Automated checks on every push. *Example:* `ci-example.yml` runs the tests. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**CI / CD / CT** — Continuous integration / delivery / training: automated tests, rollout and retraining. *Example:* 17.3. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Client** — The code that calls a service and handles its answers and failures. *Example:* `ChurnClient`. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Clipping** — Capping IPS weights at M: less variance, more bias. *Example:* cap 4, 6 or none in 5.5. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Closed-loop load test** — Each simulated client sends its next request only after the previous answer. *Example:* the 1 / 4 / 16 / 32 client runs. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Cluster (group) bootstrap** — A bootstrap that resamples whole groups (customers) rather than rows, because rows within a group are correlated (across-subject vs across-trial variability). *Example:* `groups=customer_id`. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb). *See also:* [Bootstrap](#b), [Group (query)](#g).

**`COALESCE`** — The first non-NULL argument. *Example:* "no tickets" becomes 0 after a left join. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb). *See also:* [`LEFT JOIN`](#l).

**Cohort** — The customers who share a start period. *Example:* everyone who signed up in March 2023. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Cold start** — A user or item with no interaction history. *Example:* about 270 cold users, 170 new items. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb), [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**Collate function** — Stacks individual samples into one batch. *Example:* the default `collate_fn` for per-row datasets. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Collective** — An operation every rank must call, in the same order. *Example:* `all_reduce`, `broadcast`, `all_gather`, `barrier`. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**Collision** — Two different keys landing in the same hash bucket and sharing a weight. *Example:* 1,024 buckets: every key collides. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Columnar format** — A file that stores each column contiguously, typed and compressed. *Example:* Parquet, vs row-by-row CSV. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Column order** — The position of each feature in the array the model receives. *Example:* swapping two columns silently swaps two coefficients. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb).

**`ColumnTransformer`** — Applies a different small pipeline to each named group of columns. *Example:* `log`, `num` and `cat` branches. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb). *See also:* [Group (query)](#g).

**Combiner** — A map-side pre-aggregation that shrinks the shuffle. *Example:* 2.4M rows → ~15k partial rows. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Compute-optimal (Chinchilla)** — The model size and token count that minimise loss for a fixed FLOP budget. *Example:* ≈ 20 tokens per parameter. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb). *See also:* [Parameter (Ψ)](#p).

**Concept drift** — A change in the relationship between inputs and the outcome. Can hurt performance with no visible input drift. *Example:* the same usage now means a different risk. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb). *See also:* [Label delay](#l), [Data drift](#d).

**Confidence interval (CI)** — A range that would contain the true value in about 95% of repeated experiments. *Example:* PR-AUC 0.163, CI ≈ [0.137, 0.197]. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb). *See also:* [PR-AUC (average precision)](#p).

**Confidence weight** — How much a cell of the matrix counts in implicit ALS. *Example:* c_ui = 1 + α·r_ui. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Config hash / run id** — A short name derived from the configuration itself. *Example:* the first 12 hex characters of the config's SHA-256. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**Confusion matrix** — A table of true class vs predicted class counts. *Example:* Business → Sci/Tech is the largest error. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Container image** — A packaged file system plus a start command, built from a Dockerfile. *Example:* `python:3.11-slim` + `churnlab`. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Context manager** — An object used with `with` that runs setup and cleanup code. *Example:* the drill timer. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Continuous batching** — Re-forming the generation batch at every decode step. *Example:* vLLM, TGI, the simulation in 12.6. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Contract (API)** — The schema of requests and responses and the meaning of errors: what the service promises callers. *Example:* 422 = your request is wrong. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Contract (data)** — Written rules a table must satisfy: types, nulls, ranges, allowed values, uniqueness. *Example:* `plan` must be one of `basic`, `pro`, `team`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Contract test** — Checks the API accepts and rejects exactly what it promised. *Example:* unknown field → 422. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Contrastive learning** — Training by pulling matching pairs together and pushing other pairs apart. *Example:* (claim, abstract) pairs in 8.2. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Corpus / qrels** — The documents to search / the judged (query, relevant document) pairs. *Example:* 5,183 abstracts; ~1.1 relevant per test query. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Correlation ρ** — How tightly two quantities move together, from −1 to 1. *Example:* ρ ≈ 0.71 between pre-period and experiment clicks. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Cosine decay** — Lowering the learning rate along half a cosine to a minimum. *Example:* `warmup_cosine_lr` (Exercise 1). *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Cosine similarity** — The angle-based similarity of two vectors, from −1 to 1. *Example:* 0.89 vs 0.69 for related vs unrelated pairs. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Counterfactual swap** — Replacing one identity term with another and checking the score does not move. *Example:* "gay" → "straight". *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Covariate** — A quantity measured before assignment that predicts the outcome. *Example:* each user's clicks in the two-week pre-period. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Coverage** — Share of the catalogue that appears in anyone's candidates. *Example:* 0.025 for global popularity. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Co-visitation** — How often two items are clicked in the same session. *Example:* "people who clicked X also clicked Y". *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Critical batch size** — The batch beyond which larger batches stop buying proportionally faster progress. *Example:* linear scaling breaks at 2,048. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Cross-encoder** — A model that reads query and item text together: accurate but expensive. *Example:* the L2 ranker on the top 50. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb), [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Cross-entropy (log loss)** — Minus the log of the probability the model gave the true class. *Example:* `logsumexp(z) - z[y]`. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Cross-entropy / perplexity** — Average −log probability of the correct next token / its exponential. *Example:* ln 65 ≈ 4.17 at init → perplexity 65. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Cross feature** — A feature that fires for a *combination* of values (this ad on this site). *Example:* what DCN cross layers learn. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**CTR / pCTR** — Click-through rate: clicks ÷ impressions; pCTR is a model's predicted click probability. *Example:* base CTR ≈ 3%. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**CUPED** — Subtract θ × (covariate − its mean) from the metric to remove the predictable part of its variance. *Example:* variance factor about 0.5 for clicks per user. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**`CURRENT` pointer** — A one-line file in the registry naming the live model version. Changing it deploys or rolls back. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**CVR / pCVR** — Conversion rate *among clicks*; pCVR is its prediction. *Example:* ≈ 15% of clicks convert. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

[Back to top](#glossary)

## D

**Data drift** — A change in the distribution of inputs compared with a reference period. *Example:* sessions fall by a fifth. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb). *See also:* [PSI (Population Stability Index)](#p), [Concept drift](#c).

**DAU** — Daily active users. *Example:* 300M for the ads case. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**DDP** — PyTorch's data parallelism: replicas on every rank, gradients averaged during backward. *Example:* `DistributedDataParallel` in 9.3. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb). *See also:* [ZeRO stage 1/2/3](#z), [FSDP](#f), [All-reduce](#a).

**Decision flip** — A row whose call/no-call decision changes between two implementations. *Example:* several hundred flips out of 6,956. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb).

**Default NA strings** — Strings pandas turns into missing values unless told not to. *Example:* `"NA"`, `"N/A"`, `"null"`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Deflection** — Share of conversations resolved without a human. *Example:* 30–70% in the assistant case. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Delayed feedback** — Labels (conversions) that arrive long after the event. *Example:* exponential delays, mean 2 days. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Delta method** — A straight-line (first-order Taylor) approximation that gives the variance of a function of means. *Example:* SE 0.00168 vs a true spread of 0.00169. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Dense retrieval / bi-encoder** — Embed queries and documents separately into vectors, search by similarity. *Example:* `all-MiniLM-L6-v2`, 384 dimensions. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Dependency range** — The versions a package claims to work with. *Example:* `pandas>=2.1`. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Depth round** — An interview round that drills one topic with "why?" until the edge of your knowledge. *Example:* 19.0. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Deque** — A list you can add to or remove from at both ends in $O(1)$. *Example:* `collections.deque`. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**DFS** — Depth-first search: follow one path as deep as it goes, then back up, using a stack. *Example:* `has_cycle` with three colours. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Differential privacy (ε, δ)** — A guarantee that one record barely changes what training outputs; smaller ε = stronger. *Example:* ε ≤ 1.0 at σ = 8. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Dijkstra** — Shortest paths with non-negative weights, always expanding the cheapest frontier node. *Example:* Exercise 3. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**DistributedSampler** — Gives each rank a disjoint, equal-sized slice of the (shuffled) dataset. *Example:* `set_epoch(epoch)` reshuffles. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**Distribution shift** — Test data drawn from a different time or source than training. *Example:* 2016 → late 2017 comments. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Double descent** — Test error that falls, peaks at the interpolation threshold, then falls again. *Example:* 0.51 → ~190 → 0.30. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb). *See also:* [Interpolation threshold](#i).

**DP-SGD** — SGD with per-example gradient clipping and Gaussian noise. *Example:* `clip_and_noise` in Exercise 2. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb). *See also:* [Clipping](#c), [Gradient clipping](#g).

**DuckDB** — An in-process SQL engine that queries DataFrames and files directly, without a server. *Example:* `con.execute(sql).df()`. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**Dynamic batching** — Grouping requests that arrive close together into one model call. *Example:* `max_batch_size=32, max_wait_ms=2`. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Dynamic int8 quantisation** — Weights stored as 8-bit integers ahead of time, activations quantised per batch. *Example:* `DynamicQuantizedLinear` layers. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb), [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Dynamic programming** — Solve each subproblem once, store it, build bigger answers from stored ones. *Example:* `min_coins`, edit distance. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

[Back to top](#glossary)

## E

**Eager-mode overhead** — The fixed Python cost PyTorch pays per operation, dominant for small inputs. *Example:* PyTorch is 4–6× slower than ORT at batch 1. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**Early stopping** — Keeping the weights from the best validation epoch and stopping when it stops improving. *Example:* `EarlyStopping(patience=...)`. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb), [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**ECE** — Expected calibration error: average gap between confidence and accuracy over bins. *Example:* 0.124 → 0.034 after temperature scaling. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb). *See also:* [Calibration](#c), [Temperature scaling](#t).

**eCPM** — Expected revenue per 1,000 impressions: pCTR × bid × 1000. *Example:* the auction's ranking score. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Elevator pitch** — The project in 60 seconds: problem, approach, result with uncertainty, limit. *Example:* section 15.10. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Embargo (gap)** — Rows dropped near a split boundary because their label window overlaps the next period. Here, 30 days (the label horizon). *Example:* a 30-day gap for a 30-day label. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb).

**Embedding** — A learned vector for an id or category. *Example:* 32 numbers per user id. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb), [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Embedding table** — A learned vector per id (row), looked up by index. *Example:* 2^18 rows × 8 dims. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Endpoint** — An HTTP method plus a path the service answers. *Example:* `POST /predict`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Entity** — The thing a feature describes, identified by a key. *Example:* a user (`user_id`), an item, a user × category. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**Entity (group) leakage** — The same entity (customer, subject, device) in both training and evaluation, so the score partly measures memorisation (within-subject vs cross-subject evaluation). *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [Memorisation](#m).

**Environment variable** — A named setting the operating system passes to a process. *Example:* `CHURN_LOG_LEVEL=DEBUG`. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Error budget** — The amount of unreliability an SLO allows (100% minus the target) over its window. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Evaluation set** — A fixed, versioned set of inputs with known correct outputs. *Example:* 12 template tickets with gold fields. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Event time** — When the event happened, on the device. *Example:* `event_ts`; features are defined on it. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Exact match** — Every field of the reply is right. *Example:* 3/12 for the schema prompt. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Examination** — The user actually looked at the item; never logged. *Example:* the hidden `examined` column. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Examination hypothesis** — A click needs the item to be examined *and* found relevant. *Example:* $P(\text{click}) = P(\text{exam}) \cdot P(\text{rel})$. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Expected value (decision)** — For one action: probability-weighted gain minus cost. Used to derive thresholds from stated costs. *Example:* ≈ \$3,400 over three months. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Expert / gate (MMoE)** — Parallel sub-networks, mixed per task by a learned softmax weighting. *Example:* 3 experts, one gate per task. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Exploration traffic** — Requests shown in random order, on purpose. *Example:* `is_random = True`, 5% of requests. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb), [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb).

**Exploratory result** — Anything decided after looking; reported, but labelled as such. *Example:* test-period slices in 15.7. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Export-time column** — A value describing the account on the day of the export, not on an earlier date. *Example:* `status`, `last_active_week`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Exposed (triggered) user** — An assigned user who actually reached the tested surface. *Example:* about 5% of assigned users never make a request. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Exposure bias** — Clicks exist only for items the old system chose to show. *Example:* next-week clicks vs true top-20. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**`extra="forbid"`** — Reject fields the schema does not declare. *Example:* a typo such as `sesions_4w`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

[Back to top](#glossary)

## F

**Fail fast** — Refuse to start on bad configuration instead of running wrongly. *Example:* `ValueError` naming the variable. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Fail-open / fail-closed** — When a dependency fails: allow by default / block by default. *Example:* chosen per fraud risk tier. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**False negative** — A 'negative' that is actually relevant but was never judged. *Example:* 15% of mined negatives are another claim's positive. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Fan-out** — A join that repeats rows, inflating sums and counts silently. *Example:* 212 sessions reported as 1,484. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb), [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**FastAPI** — A Python web framework that builds request handling and docs from type hints. *Example:* `@app.post("/predict")`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Feature store** — A system that stores feature definitions and values for both training (historical, point-in-time) and serving (latest). Not used in this course; the idea is shared feature code. *Example:* point-in-time joins offline, low-latency reads online. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Feature view** — A named group of features for one entity, with its definition and TTL. *Example:* `user_snap`: impressions and clicks, last 7 days. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb). *See also:* [Group (query)](#g).

**Feedback loop** — The model's actions change the data it is later trained on. *Example:* saved customers stop looking like churners. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**Field-level F1** — Partial credit: harmonic mean of field precision and field recall. *Example:* 0.69 (terse) vs 0.80 (schema). *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Fine-tuning** — Continuing to train a pretrained model, usually all of it, on your task. *Example:* 40 AdamW steps at lr 5e-5 in 5.3. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Fingerprint** — A short hash of a DataFrame's contents, stored with a model to record exactly which training data it saw. *Example:* 16 hex characters. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**Fitted statistics** — Numbers learned from training data that the transform depends on. *Example:* medians, means, standard deviations. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb).

**Fixture** — Reusable setup that pytest passes into tests. *Example:* a temporary registry. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**FLOP/s** — Floating-point operations (multiplies, adds) per second: the speed of arithmetic. *Example:* a T4 peaks near 8 TFLOP/s in fp32, 65 in fp16. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**FPR / FNR** — Share of negatives flagged / share of positives missed, at a threshold. *Example:* FPR 0.083 vs 0.010 at 0.5. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Frame (`ROWS` / `RANGE`)** — Which rows a window includes: a count of rows, or a span of values such as dates. *Example:* `RANGE BETWEEN INTERVAL 21 DAY PRECEDING`. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**Freshness** — The time span a table actually covers. *Example:* usage runs to the end of December 2024. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb), [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**Frozen parameter** — A weight that is used in the forward pass but never updated. *Example:* all 66M DistilBERT weights under LoRA. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**FSDP** — PyTorch's fully sharded data parallel: ZeRO-3, gathering each layer just in time. *Example:* `fully_shard` in the 10.1 demo. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**FWER / FDR** — The chance of at least one false positive / the expected share of false ones among discoveries. *Example:* Bonferroni controls FWER, Benjamini–Hochberg controls FDR. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

[Back to top](#glossary)

## G

**Gain importance** — Total loss reduction from splits on a feature. *Example:* `u_cat_clicks` leads LambdaMART. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [LambdaMART](#l).

**Gang scheduling** — Starting all workers of a multi-GPU job together, or none. *Example:* distributed training jobs. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Garden of forking paths** — Many small, reasonable choices made after looking, which add up to optimism. *Example:* changing a threshold after seeing test. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**GIL** — CPython runs Python bytecode on one thread at a time per process. *Example:* why threads do not add CPU throughput. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Graceful degradation** — Answering worse but sensibly when a dependency fails. *Example:* ranker timeout → popularity order. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb). *See also:* [Timeout](#t).

**Gradient** — The slope of the loss for one parameter, computed by backward; same size as the weights. *Example:* `p.grad`, 4 bytes per parameter in fp32. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb). *See also:* [Parameter (Ψ)](#p).

**Gradient accumulation** — Adding the gradients of several micro-batches before one optimizer step. *Example:* `TrainConfig(grad_accum_steps=k)`. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Gradient bucket** — A group of gradients all-reduced together as soon as they are ready. *Example:* ~25 MB, `bucket_cap_mb`. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb). *See also:* [Group (query)](#g).

**Gradient check** — Comparing a hand-derived gradient with a finite-difference estimate. *Example:* relative error 1e-11 when right, 0.33 with a missing factor 2. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Gradient clipping** — Rescale the gradient when its norm exceeds a threshold. *Example:* `clip_grad_norm_(…, 1.0)`. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Group (query)** — The set of items ranked together; scores are compared only inside it. *Example:* one request of 10 items. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).

**Group sizes** — How LightGBM learns which rows form a group: counts of contiguous rows. *Example:* `[10, 10, 10, …]`. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [Group (query)](#g).

**Group split** — Every row of one entity (customer) goes to the same side. *Example:* "customers we have never seen". *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb).

**Group-wise scales** — One scale per small block of weights (e.g. 128), needed for int4. *Example:* GPTQ / AWQ 4-bit LLM loaders. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Guardrail** — A fast check before or after the model that blocks, flags or reroutes. *Example:* rules + TF-IDF classifier. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb). *See also:* [TF-IDF](#t).

**Guardrail metric** — Something that must not get worse while you optimise the main metric. *Example:* latency, complaint rate. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb), [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

[Back to top](#glossary)

## H

**Half-open interval [a, b)** — Includes a, excludes b. *Example:* the label window [T, T + 30 days). *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**`handle_unknown="ignore"`** — An unseen category becomes all zeros instead of an error. *Example:* a new region in production. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb). *See also:* [Unseen category](#u).

**Hand rule** — An if-then rule written by a person, no fitting. *Example:* zero sessions or a failed payment. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb).

**Hang** — Ranks blocked forever in a collective one rank never reached. *Example:* an `all_reduce` inside `if rank == 0:`. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb). *See also:* [Collective](#c).

**Hard negative** — A document a retriever ranks high that is not labelled relevant. *Example:* BM25's top unlabelled hit, after `SKIP_TOP`. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Hash** — A short fingerprint of a file; changing one byte changes it. *Example:* the 16-character SHA-256 prefix stored with the results. *Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb). *See also:* [Fingerprint](#f).

**Hashing trick** — Mapping each (field, value) to a bucket with a hash, instead of a vocabulary. *Example:* `hash(field, value) mod n`. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Hash map** — A dictionary: average O(1) lookup, insert and delete by key. *Example:* complements in two-sum. *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Headroom** — Spare capacity kept for spikes, a lost replica or a deploy. *Example:* 30% in the ads fleet. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Heap** — A tree kept partly ordered so the smallest item is always on top. *Example:* `heapq`; top-k in O(n log k). *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Held-out test set** — Data used exactly once, after every decision is fixed. *Example:* Jul-Nov 2024. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Heterogeneous effect** — A treatment effect that differs between segments. *Example:* 15 country × device slices whose true lift is identical. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Hidden state** — The contextual vector the encoder outputs for each token. *Example:* `last_hidden_state`, `(batch, seq, 768)`. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Hit rate** — The share of lookups the cache answers without a round trip. *Example:* at most 0.29 for one day of requests. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Hive partitioning** — One folder per value of a column, value in the path. *Example:* `day=0/` … `day=55/`. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**HNSW** — A layered neighbour graph searched greedily with a beam of `efSearch`. *Example:* `IndexHNSWFlat`. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb), [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Holdout** — A random group deliberately not acted on, so the model stays measurable. *Example:* customers who are never called. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb). *See also:* [Group (query)](#g).

**Holistic aggregate** — State grows with the data; cannot combine to a fixed size. *Example:* count(DISTINCT), median. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**HR@10** — Share of users whose target is in the top 10. *Example:* about 0.07 for SASRec. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb). *See also:* [SASRec](#s).

**Hybrid retrieval** — Combining lexical and dense results. *Example:* best on both metrics here. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Hyperparameter** — A setting chosen before fitting, not learned from data. *Example:* `C`, `max_leaf_nodes`. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb).

[Back to top](#glossary)

## I

**Idempotent** — An operation that has the same effect whether performed once or several times. Scoring is; "create a task" is not unless designed to be. *Example:* a prediction; not a payment. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Idempotent update** — Applying it twice has the same effect as once. *Example:* dedup by `event_id`. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**idf** — Inverse document frequency: rare terms count more than common ones. *Example:* "prion" vs "protein". *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Implicit feedback** — Behaviour (clicks, purchases) used as a label instead of explicit ratings. *Example:* `clicked`. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Impression** — One item shown to one user at one position. *Example:* one row of `impressions`: 10 per request. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb), [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Imputer** — Fills missing values with a statistic learned on training data. *Example:* median of `features_used_4w`. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb).

**In-batch negatives** — The other items in the batch serve as negatives for each user. *Example:* a 1024 × 1024 logit matrix. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb), [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**In-degree** — The number of edges pointing into a node. *Example:* Kahn's algorithm starts from in-degree 0. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Inertia** — The k-means objective: total squared distance from each point to its centre. *Example:* Exercise 3 must be within 2% of sklearn's. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**InfoNCE** — The contrastive loss: cross-entropy over similarity scores, target = the true pair. *Example:* Exercise 1. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb). *See also:* [Cross-entropy (log loss)](#c).

**Inner vs left join** — Inner keeps matching pairs only; left keeps every left row, NULL where no match. *Example:* customers with no tickets survive only a `LEFT JOIN`. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb). *See also:* [`LEFT JOIN`](#l).

**Interleaving** — Merge two rankers' lists into one and credit each click to the ranker that contributed the item. *Example:* about 20× fewer users than an A/B test for 80% power. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb). *See also:* [A/B test](#a), [Power (1 − β)](#p).

**Interpolation threshold** — The model size at which it can just fit every training point (p ≈ n). *Example:* p = 100 for 100 points. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Invariant** — A statement that stays true at every step of a loop. *Example:* "`s[left:right+1]` has no repeated character". *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Inverse propensity scoring (IPS)** — Weight each click by 1/θ_p to undo examination. *Example:* a click at position 9 counts about 8. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb).

**Inverted dropout** — Zero units with probability p in training and scale the rest by 1/(1−p). *Example:* Exercise 6; identity at inference. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Irreducible error** — Mistakes no model can fix with the available features. *Example:* healthy-looking customers who leave anyway. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**IterableDataset** — Only `__iter__`; it decides the order itself. *Example:* `tk.ParquetDayStream`. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**IVF** — K-means cells; search only the `nprobe` closest cells. *Example:* `IndexIVFFlat`. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

[Back to top](#glossary)

## J

**Jitter** — A random fraction added to the wait so clients do not retry in lockstep. *Example:* `delay × uniform(0.5, 1.5)`. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb). *See also:* [Retry](#r).

**JSON Lines log** — A text file with one JSON record per line, appended and never rewritten. *Example:* `progress.jsonl`. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

[Back to top](#glossary)

## K

**Kernel** — One function the GPU runs on thousands of threads; every PyTorch op launches at least one. *Example:* `a @ b` launches one matrix-multiply (GEMM) kernel. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Kernel fusion** — Merging several elementwise ops into one kernel so the data is read once. *Example:* what `torch.compile` does (2.5). *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Key skew** — A few keys hold a large share of the rows. *Example:* the top seller's (seller, user) pairs. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Killed / surviving mutant** — A test caught the change / no test noticed it. *Example:* a survivor points at a missing test. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**k-means++** — An initialisation that picks each new centre with probability ∝ squared distance. *Example:* spreads the first centres across the blobs. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Knowledge distillation** — Training a small student to match a large teacher's probabilities. *Example:* a 193-parameter student taught by LightGBM. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb), [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb), [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**KV cache** — Stored keys and values of past tokens, so each new token is computed once. *Example:* 2–5× faster generation in 3.7. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb). *See also:* [Paged attention](#p), [Prefill vs decode](#p).

[Back to top](#glossary)

## L

**L2 normalisation** — Scaling each vector to length 1, so dot product = cosine. *Example:* all norms 1.0000 in 7.3. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**L2 regularisation** — Adding (λ/2)‖w‖² to the loss so the weights stay small. *Example:* `l2` in Exercise 1; the intercept is not penalised. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Label delay** — Outcomes arrive only after the label window closes. *Example:* 30 days. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Label leakage** — A feature that encodes the outcome, directly or by proxy (`status = 'cancelled'`). *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).

**Label maturity** — Fraction of eventual labels already observed. *Example:* 30% after 7 days (20-day mean delay). *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Label window** — The period after T in which a cancellation counts as "churned". *Example:* [T, T + 30 days). *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**LambdaMART** — RankNet gradients weighted by. *Example:* ΔNDCG. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [RankNet](#r).

**Late event** — Arrives for a window that is already closed. *Example:* about 2.5% with no lateness. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Latency** — Time from sending a request to receiving the answer. *Example:* milliseconds per `/predict`. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Latency budget** — An end-to-end time target split among the steps of a request. *Example:* 200 ms: feature lookup, model, network. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb), [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Launch criteria** — The decision rule, written before the experiment starts. *Example:* `launch_criteria.md` in 12.6. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Layer cache** — Docker reuses unchanged build steps, so their order matters. *Example:* install dependencies before copying code. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**LayerNorm** — Rescales each token's vector to mean 0 and variance 1, then applies a learned scale and shift. *Example:* pre-LN: inside each branch. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Lazy deletion** — Leave outdated heap entries in place and skip them when popped. *Example:* `heapq` has no decrease-key. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Leakage** — Information in training or evaluation that would not exist at prediction time. *Example:* a column exported after the snapshot date. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb), [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb), [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb). *See also:* [Embargo (gap)](#e), [Time split](#t), [Point-in-time correctness](#p), [Snapshot date (T)](#s).

**Leave-last-out** — Hold out each user's last event as the test target. *Example:* here inside the validation week only. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**`LEFT JOIN`** — Keep every row of the left table, with NULLs where the right table has no match. *Example:* customers with no invoices still appear. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).

**Left padding** — Fill short sequences with 0s at the *start*. *Example:* `[0, 0, 18, 5, 10]`. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**Length bucketing** — Grouping texts of similar length into the same batch to cut padding. *Example:* 59% → 17% waste in 4.5. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Lexical retrieval** — Matching the exact words of the query. *Example:* BM25. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Lift@k%** — Precision in the top k% of scores divided by the base rate. *Example:* lift@10% ≈ 3.2. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**Linear probe** — A linear classifier trained on frozen embeddings; tests what the encoder already knows. *Example:* 0.858 test accuracy in 5.2. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Linear scaling rule** — Multiplying the learning rate by k when the batch grows k times. *Example:* batch 32 → 512, LR 0.01 → 0.16. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Listwise loss** — Optimises a list metric (smoothly). *Example:* LambdaRank / LambdaMART. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [LambdaMART](#l).

**Little's law** — Average requests in a system = arrival rate × average time each spends there ($L = \lambda W$). Turns a QPS and a latency into the concurrency you must support. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb), [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Liveness probe** — A check that the process is alive. If it fails, the process is restarted. Should not depend on the model. *Example:* `GET /health`. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**LLM-as-judge** — Using a language model to decide which of two outputs is better. *Example:* the 0.5B model reading P("A") vs P("B"). *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Local rank** — A process's index on its own machine; picks which GPU it uses. *Example:* `torch.cuda.set_device(local_rank)`. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**Lock file** — A list of the exact package versions a result was produced with. *Example:* `requirements-lock.txt`. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb), [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Logging policy** — The production ranker that chose what was shown. *Example:* ShopStream's popularity + weak-affinity mix. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Logit** — The raw score before the sigmoid or softmax. *Example:* the click model's single output. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb), [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb), [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Log loss** — Average surprise of the true labels under the predicted probabilities. *Example:* what logistic regression minimises. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb), [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**logQ correction** — Subtract log q_i from item i's logit during training only. *Example:* recall 0.24 → 0.41. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

**Log-sum-exp trick** — Computing log Σ eᶻ after subtracting the largest z, so nothing overflows. *Example:* softmax of `[1000, 1001, 999]` without NaN. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Loop** — The full set of interview rounds for one candidate. *Example:* phone screen + six onsite rounds. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb). *See also:* [Phone screen](#p).

**LoRA** — Learn a low-rank correction B·A next to a frozen weight matrix. *Example:* `LoRALinear` in 6.2. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb). *See also:* [Adapter](#a), [QLoRA](#q), [PEFT](#p).

**Loss scaling** — Multiplying the loss so small fp16 gradients do not round to zero. *Example:* `GradScaler`, S = 2¹⁶ to start. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**LRU cache** — Keeps the most recently used entries and evicts the least recently used when full. *Example:* the user-feature cache in 17.4. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

[Back to top](#glossary)

## M

**Map-style dataset** — Has `__len__` and `__getitem__(i)`; the sampler picks indices. *Example:* `RowDataset`, `tk.TensorBatchDataset`. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Map task** — Reads one input split and emits (key, value) pairs. *Example:* 4 mappers over the impression files. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Mask** — A boolean saying which keys each query may use. *Example:* causal and padding masks; True = may attend. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Materialisation** — Copying the newest offline values into the online store. *Example:* the daily loop in 3.6. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb). *See also:* [Online store](#o).

**Matrix factorisation** — Approximating the user × item matrix by user vectors times item vectors. *Example:* `rt.implicit_als`. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Matured cohort** — Snapshots whose labels are all known. *Example:* the latest month older than 30 days. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**MDE** — Minimum detectable effect: the smallest true effect the test is sized to catch. *Example:* +1 percentage point of conversion. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Membership inference** — Guessing whether a record was in the training set from the model's behaviour on it. *Example:* loss-threshold attack, AUC 0.64. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Memorisation** — Fitting training labels that carry no signal (e.g. flipped ones) exactly. *Example:* 100% train accuracy at 40% noise. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Memory bandwidth** — How many bytes per second can move between GPU memory and its cores. *Example:* ~320 GB/s on a T4. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Memory-bound / compute-bound** — Limited by moving bytes / limited by doing arithmetic. *Example:* elementwise add / large matmul. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Merge** — Folding B·A into W after training so serving is as fast as the base model. *Example:* `merge_and_unload()`. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**MFU** — Model FLOPs utilisation: useful FLOP/s divided by the hardware peak. *Example:* ~40% for well-tuned large runs. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb). *See also:* [FLOP/s](#f).

**Micro-batch** — One of the small pieces a large batch is split into. *Example:* 4 micro-batches of 8 in 2.3. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Micro-batch / bubble** — A slice of the batch fed through the pipeline / the idle time while it fills and drains. *Example:* 3/7 idle at p = m = 4. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**Mid / senior / staff** — Generic levels of scope and independence, not any company's ladder. *Example:* 20.3. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Minimum-norm solution** — Among all weight vectors that fit the data exactly, the one with the smallest length. *Example:* `pinv(X) @ y`. *Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb).

**Mixed precision (autocast)** — Running matmuls in fp16/bf16 while keeping sensitive values in fp32. *Example:* `torch.autocast`; halves activations, not model state. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb). *See also:* [Model state](#m).

**ML platform** — Shared infrastructure many teams use to build, ship and watch models. *Example:* "fifty teams" in the prompt. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**ML system design interview** — A 35-45 minute spoken design of an ML product, end to end. *Example:* "churn prediction for 10M accounts". *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**Mock loop** — A timed rehearsal of the interview rounds, scored against a rubric. *Example:* four rounds, 165 minutes. *Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb).

**Model card** — A short document describing a model: intended use, data, metrics with intervals and split, limitations, versions. *Example:* `artifacts/model_card.md`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb), [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Model state** — Weights + gradients + optimizer state (not activations). *Example:* 16Ψ bytes for mixed-precision Adam. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb). *See also:* [Optimizer state](#o).

**Model version header** — The response names which model answered. *Example:* `v1`. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb).

**modules_to_save** — Layers `peft` trains fully alongside the adapters. *Example:* the new classification head. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**Monitoring** — Measuring a live system continuously so someone can act. *Example:* service health, drift, performance. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Monotone predicate** — A yes/no test that flips from False to True once as its input grows. *Example:* "capacity c ships everything in D days". *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Monotonic stack** — A stack kept in sorted order to answer "next greater / most recent smaller" questions. *Example:* 15.7. *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**MRR** — Mean of 1 / rank of the first relevant item. *Example:* 0.73 for the logged order (clicks). *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).

**MTBF** — Mean time between failures. *Example:* 128 nodes at 50 days each → 9.4 h for the job. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**Multi-adapter serving** — One base model in memory, a different adapter per request or tenant. *Example:* S-LoRA, vLLM multi-LoRA. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb). *See also:* [Adapter](#a).

**Multi-armed bandit** — Repeatedly choosing among options with unknown payoffs while learning them. *Example:* 10 ads with CTRs 1–5%. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Multi-task learning** — One network predicting several targets with shared layers. *Example:* click and conversion heads. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Multi-task ranker** — One network with several heads, each predicting a different outcome. *Example:* click, add-to-cart and purchase heads. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**Mutation testing** — Deliberately breaking the code to check that some test fails. *Example:* a mutant that ignores the threshold. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

[Back to top](#glossary)

## N

**NDCG@k** — Discounted gain of the top k, divided by the best possible. *Example:* 0.81 for the logged order (truth). *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb), [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb), [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Negative downsampling** — Keeping all clicks but only a fraction `w` of non-clicks for training. *Example:* `w = 0.1` in 13.7. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb).

**Negative transfer** — One task's training hurting another task in a shared model. *Example:* clickbait helps clicks, hurts conversions. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Next-item prediction** — Given a user's past events in order, predict the next one. *Example:* the last validation click. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**NLL** — Negative log-likelihood: the cross-entropy of the true class. *Example:* 0.487 → 0.427 on test. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb). *See also:* [Cross-entropy (log loss)](#c).

**`no_grad()`** — Stop recording the graph for backprop. *Example:* (1) around validation. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb). *See also:* [Backprop](#b).

**Normalised entropy (NE)** — Log loss divided by the log loss of always predicting the average CTR; lower is better. *Example:* constant model NE = 1.00. *Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch51-ctr-prediction-at-scale/notebook.ipynb). *See also:* [Log loss](#l).

**North-star metric** — The business outcome the system ultimately serves. *Example:* purchases per user per week. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**`num_workers`** — Separate processes that load batches in parallel. *Example:* 0 vs 2 on a 2-core runtime. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

[Back to top](#glossary)

## O

**Offline–online gap** — Offline and online evaluations disagreeing about which model is better. *Example:* logging policy first offline, third online. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb). *See also:* [Logging policy](#l).

**Offline / online metric** — Measured on historical data / measured on live users, usually by experiment. *Example:* PR-AUC / retained revenue. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb). *See also:* [PR-AUC (average precision)](#p).

**Offline store** — The full history of feature values, for building training sets. *Example:* DuckDB tables of daily snapshots. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb). *See also:* [DuckDB](#d).

**Off-policy evaluation** — Estimating a new policy's value from another policy's logs. *Example:* IPS value of a re-ranking. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**One-hot encoding** — One 0/1 column per category level. *Example:* `region_EU`, `region_NA`, . *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb).

**One-to-many** — One row on one side matches many on the other. *Example:* one customer, 41 usage weeks. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**Online feature store** — A low-latency key–value lookup of features at request time. *Example:* `OnlineStore`, 1 ms simulated per call. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Online store** — Only the latest value per key, readable in about a millisecond. *Example:* `OnlineStore`, a dict with a TTL. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**ONNX** — A portable file format for a model's computation graph. *Example:* `mlp.onnx`, with a dynamic batch dimension. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**ONNX Runtime (ORT)** — A C++ engine that executes ONNX graphs without Python in the loop. *Example:* about 0.006 ms per call at batch 1. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**OOM** — Out of memory: an allocation larger than what is free on the device. *Example:* `torch.OutOfMemoryError`. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**OpenAPI** — The machine-readable description of the contract. *Example:* `/openapi.json`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Operating point** — The threshold or top-k rule actually used to act. *Example:* the validation threshold, applied to test. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Optimizer state** — Extra numbers the optimizer keeps per parameter between steps. *Example:* AdamW's two moments m and v: 8 bytes per parameter. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb). *See also:* [Parameter (Ψ)](#p).

**Oracle** — The simulation's ground truth, used only to grade. *Example:* `true_p`. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Out-of-core** — Processing data larger than memory by streaming and spilling. *Example:* DuckDB with `memory_limit = '64MB'`. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb). *See also:* [DuckDB](#d).

**Out-of-order event** — Arrives after an event that happened later. *Example:* about 65% of events here. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Output-sensitive bound** — A cost that depends on the size of the answer, not only the input. *Example:* two-sum pairs: O(n + p). *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

[Back to top](#glossary)

## P

**p95 / p99 latency** — The response time below which 95% / 99% of requests complete. *Example:* p95 = 95% of requests are faster. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb), [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Padding** — Filler tokens that make every row of a batch the same length. *Example:* `padding=True` pads to the longest row. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Padding mask** — Blocks attention to padding positions. *Example:* `padding_mask(lengths, n)`, Exercise 3. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb). *See also:* [Attention](#a).

**Padding token** — The id that means "no item here". *Example:* 0, so item ids are shifted by +1. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**Paged attention** — Storing the KV cache in fixed-size blocks found through a block table. *Example:* 16-token blocks in 12.7. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb). *See also:* [KV cache](#k).

**Paired bootstrap** — Bootstrapping the *difference* between two models on the same resamples, to test whether one is better. *Example:* `paired_bootstrap_diff`. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb), [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb), [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Pairwise accuracy** — Share of differently-labelled pairs ordered correctly. *Example:* 0.53 logged vs 0.85 learned. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).

**Pairwise loss** — Penalises each mis-ordered pair inside a group. *Example:* RankNet. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [Group (query)](#g), [RankNet](#r).

**Parameter (Ψ)** — One trainable number in the model; Ψ is the usual symbol for how many there are. *Example:* 929,536 on the CPU budget. *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Pareto frontier** — Settings that cannot improve one metric without hurting another. *Example:* clicks vs value when blending. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Parity check** — Comparing two implementations' outputs element by element within a tolerance. *Example:* max difference 1.2e-6 after export. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**Parity test** — Scores the same rows offline and through the running API and requires them to match. *Example:* max difference ≈ 5e-7. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb), [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Partition pruning** — Skipping whole folders from the path alone. *Example:* `Scanning Files: 7/56`. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Path compression** — Pointing every node a `find` passes straight at the root. *Example:* the 16.6 timing demo. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**pCTCVR** — Probability an impression is clicked *and* converts: pCTR × pCVR. *Example:* expected conversions per impression. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Peak factor** — How much busier the busiest hour is than the average. *Example:* ×3. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**Peak QPS** — Queries per second at the busiest time of day. *Example:* about 28,900 for the home feed. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb), [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Peak vs leak** — A single step that needs too much memory vs memory that grows every step. *Example:* accumulating `loss` instead of `loss.item()` (2.6). *Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch40-training-efficiency-and-memory/notebook.ipynb).

**Peeking** — Checking repeatedly and stopping at the first significant result. *Example:* 14 daily looks give a false-positive rate of about 0.2. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**PEFT** — Parameter-efficient fine-tuning: train a small number of new or selected parameters. *Example:* LoRA, adapters, prefix tuning. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb). *See also:* [`train()` / `eval()`](#t), [Fine-tuning](#f).

**Percentile interval** — The 2.5th to 97.5th percentiles of the bootstrap replicates. *Example:* cascade 0.715 [0.709, 0.721]. *Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb). *See also:* [Bootstrap](#b), [Cascade](#c).

**Per-tensor vs per-channel** — One scale for a whole matrix vs one scale per output row. *Example:* per-channel cuts the weight error 2.4× in 11.3. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Phone screen** — A first, shorter round that gates the onsite. *Example:* 45 minutes of coding. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Pickle / joblib** — Python's way to save objects; loading one can execute code. *Example:* why the hash is checked first. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**PII** — Personally identifiable information: names, emails, phone numbers. *Example:* the regex redactor in 15.6. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Pin** — Record the exact version of something a result depends on. *Example:* a git commit, a file checksum, a lock file, a seed. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb). *See also:* [Checksum](#c), [Lock file](#l).

**Pinned memory** — Page-locked host RAM the GPU can copy from directly, in the background. *Example:* `pin_memory=True` + `non_blocking=True`. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Pipeline** — One object that chains preprocessing steps and a model; `fit` and `predict` run them all. *Example:* `make_model("logreg")`. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb).

**Pipeline parallelism** — Putting consecutive groups of layers on different GPUs. *Example:* 4 stages in 10.3. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**Plateau** — A region where many settings score the same within noise. *Example:* logreg for `C` from 0.003 to 10. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb).

**Point-in-time correctness** — Features computed only from data available before the prediction time; labels only from after. *Example:* state from days 0–41 serves days 42–48. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Point-in-time feature** — A feature built only from rows that existed before T. *Example:* usage from weeks that ended by T. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**Point-in-time join** — For each event, the feature values as they were just before it. *Example:* `ASOF JOIN … e.ts > s.snapshot_ts`. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**Pointwise loss** — Scores each item on its own, like a classifier. *Example:* log loss on `clicked`. *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb). *See also:* [Log loss](#l).

**Pooled vs per-user CTR** — Total clicks ÷ total impressions, versus the average of each user's own click rate. *Example:* both about 0.145 per arm in 11.2. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Pooling** — Turning per-token vectors into one vector per text. *Example:* CLS (position 0) vs mask-aware mean. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Popularity bias** — A generator's tendency to return already-popular items. *Example:* the mean log-popularity column. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb).

**Position-based model (PBM)** — Examination depends only on the position. *Example:* θ_p = (1 + p)^−0.9 in ShopStream. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Position bias** — A judge preferring an answer because of its slot, not its content. *Example:* answered "A" every time. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Positive pair** — A query and a document known to be relevant to it. *Example:* every judged pair in the train qrels. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Positive rate** — Share of rows whose label is 1. *Example:* 5% of customers churn. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).

**Postmortem** — A blameless write-up of an incident and how to prevent it. *Example:* Exercise 3. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Power (1 − β)** — The chance the test detects an effect that is really there. *Example:* 80% power for 10% → 11% at 14,749 users per arm. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**PR-AUC (average precision)** — Area under the precision-recall curve. Its no-skill baseline equals the positive rate. *Example:* the headline metric from Day 1 on. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [Positive rate](#p), [ROC-AUC](#r).

**Precision** — Share of flagged accounts that were going to churn. *Example:* ≈ 0.27 at the cost threshold. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb). *See also:* [Precision@k](#p), [Threshold](#t), [Recall](#r).

**Precision@k** — Share of true positives among the k highest-scored rows. Fits capacity-limited actions. *Example:* per month, at the team's capacity. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb), [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Predicate pushdown** — Applying a filter inside the scan, using file statistics. *Example:* row groups skipped by `request_id` range. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Prediction drift** — The distribution of scores changes. *Example:* mean score ratio above 1.5. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Prefill / decode tokens** — Prompt tokens processed in parallel vs output tokens generated one at a time. *Example:* ~130 in, ~32 out per request. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Prefill vs decode** — Processing the whole prompt in one pass vs generating one token per step. *Example:* 12.5. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Prefix sum** — A running total; the sum of any prefix is one lookup. *Example:* `np.cumsum(y)` and `np.cumsum(y**2)`. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb), [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb). *See also:* [Running total](#r).

**Pre-ranker** — A cheap model that trims thousands of candidates to hundreds. *Example:* 2,000 → 500 in design A. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**Pre-registration** — Writing the analysis plan, with its decision rule, before seeing the data. *Example:* `test_plan.json` and its hash. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb).

**Prior (dummy) model** — Gives every row the same score. *Example:* PR-AUC = positive rate = 0.052. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [PR-AUC (average precision)](#p).

**Process group / backend** — The set of processes that talk to each other / the library that moves the bytes. *Example:* gloo on CPU, NCCL on GPUs. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**Processing time** — When the event reached the stream processor. *Example:* `arrival_ts`, median 2 s later. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Product quantisation** — Compressing vectors into a few bytes of codebook ids. *Example:* IVF-PQ, 32 floats → 16 bytes. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

**Profiler** — A tool that records which operations took the time. *Example:* the `torch.profiler` table in 1.6. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Project brief** — A one-page summary of a project with every number tied to its split. *Example:* `brief_*.md`. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Promote / roll back** — Point `CURRENT` at a new version / at the previous one. *Example:* `registry.promote("v1")`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**Prompt injection** — Instructions hidden in user-supplied data that the model then follows. *Example:* "ignore all previous instructions…". *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

**Propensity θ_p** — The probability that position p is examined. *Example:* θ_0 = 1, θ_9 ≈ 0.13. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb).

**Proxy metric** — A fast, measurable stand-in for the north star, with known blind spots. *Example:* feed CTR rewards clickbait. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**Ψ (psi)** — The number of model parameters, in memory formulas. *Example:* 7B: Ψ = 7 × 10⁹. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**PSI (Population Stability Index)** — A drift measure comparing binned distributions of a reference and a current sample. Rule-of-thumb bands 0.1 and 0.25. *Example:* 0.1-0.25 = moderate. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Pydantic model** — A class that declares field types and limits and validates data against them. *Example:* `CustomerFeatures`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**`pyproject.toml`** — The file that makes a folder an installable Python package. *Example:* name, version, dependencies of `churnlab`. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

[Back to top](#glossary)

## Q

**QLoRA** — LoRA on a base model stored in 4 bits. *Example:* a 7B base in ~3.5 GB. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**QPS** — Queries (requests) per second the system must handle. *Example:* ≈ 520 at peak for an in-app badge. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb).

**`QUALIFY`** — Filters on a window function, like `HAVING` filters on an aggregate. *Example:* keep `ROW_NUMBER() = 1` per customer. *Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb). *See also:* [Window function](#w).

**Quantisation** — Storing (and computing with) numbers in fewer bits, e.g. 8-bit integers instead of 32-bit floats. *Example:* DistilBERT's Linear layers converted to int8. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Query, key, value** — Three vectors per token: what it is looking for, what it offers, what it hands over. *Example:* `x @ w_qkv` split in three (Exercise 1). *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

[Back to top](#glossary)

## R

**Randomisation unit** — The thing one coin flip assigns to an arm. *Example:* the user: all of a user's requests see the same ranker. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**RankNet** — Logistic loss on the score difference of a pair. *Example:* log(1 + e^−(s_i − s_j)). *Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).

**Rank r** — The inner size of B and A; how many independent directions the update can use. *Example:* r = 8. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**Rank statistic (AUC)** — AUC as the chance a random positive outscores a random negative, ties counting ½. *Example:* Exercise 5 in O(n log n). *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Rank / world size** — A process's unique number / how many processes the job has. *Example:* ranks 0–3, world size 4. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**Ratio metric** — A sum divided by a sum, where both vary per user. *Example:* pooled CTR = Σ clicks ÷ Σ impressions. *Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch30-advanced-experimentation/notebook.ipynb).

**Raw extract** — The files exactly as exported from the source system, never edited by hand. *Example:* four `.csv.gz` files in `data/raw/`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Readiness** — Whether you can do a skill unaided, from a blank file, in interview time. *Example:* `ready: 0/20` on a fresh run. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Readiness probe** — A check that the service can do its job (model loaded). If it fails, traffic is routed elsewhere. *Example:* `GET /ready` returns 503 without a model. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Recalibration** — A monotone map from score to probability, learned on held-out later rows. *Example:* sigmoid (Platt) or isotonic. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Recall** — Share of all churners that were flagged. *Example:* ≈ 0.14 at the cost threshold. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb). *See also:* [Recall@K](#r), [Precision](#p).

**Recall@K** — Share of a user's relevant items found among K candidates. *Example:* 0.34 for ALS at K = 100. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb), [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Recall vs exact** — Share of the exact top-k that the approximate search returns. *Example:* 0.92 at `nprobe = 4`. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

**Reduce-scatter / all-gather** — The two halves of a ring all-reduce: sum one chunk each / share every chunk with all. *Example:* 9.2, 9.6. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb). *See also:* [All-reduce](#a).

**Reduce task** — Merges all values of the keys it owns. *Example:* sums clicks per `item_id`. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Referential integrity** — Every foreign key points at a row that exists. *Example:* each `customer_id` in `tickets` is in `customers`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Registry** — A store of model versions with a pointer to the live one, and a history of promotions and rollbacks. *Example:* `v1/`, `v2/`, `CURRENT`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Regression stump** — A depth-1 tree: one threshold, two leaf values. *Example:* `best_split` in Exercise 1. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Regret** — Expected reward lost compared with always choosing the best option. *Example:* clicks lost over 50,000 rounds. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Re-implementation** — Writing the same transform a second time, from a description. *Example:* the backend's version of the notebook function. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb).

**Reliability diagram** — Plot of mean predicted vs observed rate, per score bin. *Example:* the diagonal is perfect calibration. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb). *See also:* [Calibration](#c).

**Replica** — One copy of the service behind a load balancer. *Example:* 18 replicas at 50 QPS each, 60% utilised. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb), [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Reproducible** — Someone else gets the same number from the same inputs. *Example:* a colleague re-runs your run record a year later. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb). *See also:* [Run record](#r).

**Request** — One feed load: the 10 items shown together. *Example:* `request_id`; ~245,000 in `small`. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Request id** — An identifier carried through logs so one call can be traced. *Example:* the `x-request-id` header. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb), [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Re-ranking depth** — How many first-stage candidates the re-ranker scores. *Example:* top 50. *Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Resampling unit** — What the bootstrap draws: rows, or whole groups (customers). *Example:* customers are the default here. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb). *See also:* [Bootstrap](#b).

**Reserve price** — A floor: bids below it are not eligible, the winner pays at least it. *Example:* 0.5 for uniform values. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Residual (pseudo-residual)** — Target minus current prediction; for squared loss, the negative gradient. *Example:* `y − F` in Exercise 2. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Residual stream** — The running vector each block reads from and adds its output to. *Example:* `x = x + Attention(LayerNorm(x))`. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb). *See also:* [LayerNorm](#l), [Attention](#a).

**`response_model`** — The schema the response is filtered and validated against. *Example:* internal fields never leak out. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Retention at month +k** — Share of a cohort still active k months after its start. *Example:* `ret_m1`, `ret_m2`, `ret_m3`. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Retention team** — The people who act on the predictions, with a fixed capacity. *Example:* customer success calls the highest-risk accounts. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**Retrieval recall** — The share of the truly best items that made it into the candidates. *Example:* about 0.85 of the true top 10. *Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Retry** — Sending the same request again after a failure that might not repeat. *Example:* a 503 while the model loads. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Revision pinning** — Loading a model at a fixed commit so files cannot change under you. *Example:* `revision=BERT_REV`. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**ROC-AUC** — Probability a random positive is scored above a random negative; random = 0.5. *Example:* ≈ 0.72 here. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [PR-AUC (average precision)](#p).

**Rollback** — Returning to the previous model version. Here: rewriting `CURRENT` and reloading. *Example:* seconds, no retraining. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb).

**Rolling-origin CV** — Time-series cross-validation: train on everything before a cut-off, validate on the next block, move the cut-off forward. *Example:* 4 folds inside the train period. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Roofline** — The plot of attainable FLOP/s against arithmetic intensity. *Example:* ridge point ≈ 25 FLOP/byte for a T4 in fp32. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb). *See also:* [FLOP/s](#f), [Arithmetic intensity](#a).

**Row group** — A horizontal slice of a Parquet file with min/max per column. *Example:* ~20 groups of 122,880 rows. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**RRF** — Reciprocal rank fusion: combine rankings by summing 1 / (k + rank). *Example:* the hybrid in 7.6. *Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch45-embeddings-and-semantic-search/notebook.ipynb).

**Rubric anchors** — Short descriptions of what each score means, so scores are comparable over time. *Example:* 1 = missing or wrong … 4 = strong. *Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb), [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Running statistics** — Moving averages of mean and variance that batch norm uses at inference. *Example:* `momentum = 0.1` in Exercise 5. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Running total** — A cumulative sum ordered by time. *Example:* `SUM(...) OVER (ORDER BY month)`. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Run record** — A small file saved with each experiment: code version, data fingerprint, config, seed, metrics with split. *Example:* `artifacts/runs/<run_id>.json`. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb). *See also:* [Fingerprint](#f).

[Back to top](#glossary)

## S

**Safetensors** — A weight format that stores only tensors, so loading cannot run code. *Example:* `use_safetensors=True`. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Salting** — Splitting a hot key into S sub-keys, then re-combining. *Example:* `(seller, hash(user) % S)`. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Sampled metric** — Rank the target against a sample of negatives, not the whole catalogue. *Example:* 100 uniform negatives in 10.6. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**Sample-ratio mismatch (SRM)** — Arm sizes too unequal to be chance, which means broken assignment or logging. *Example:* 0.505 of users in control, SRM p ≈ 3e-6. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Sample selection bias (SSB)** — Training on one population (clicked rows) and serving on another (all rows). *Example:* naive pCVR −14–19% on non-clicked impressions. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Sampling probability q_i** — How likely item i is to appear in a batch. *Example:* proportional to its clicks. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

**SASRec** — A causal transformer over item ids for next-item recommendation. *Example:* Kang & McAuley, 2018. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**Scale and zero-point** — The two numbers that map floats to integers and back. *Example:* see the figure in 18.3. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**Scaled dot-product attention** — Softmax(QKᵀ/√d)V: a weighted average of values, weighted by query–key match. *Example:* Exercise 4. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Scale `s`** — The real-valued size of one integer step; `x ≈ q × s`. *Example:* `max(abs(W[row])) / 127` for one weight row. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Schema validation** — Checking a request against declared field types and limits before using it. *Example:* Pydantic rejects an unknown field. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).

**Search budget** — The number of configurations you allow yourself, fixed in advance. *Example:* 8 per model family. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb).

**Second-price auction** — The winner pays the smallest bid that would still have won, not its own bid. *Example:* `bid₂ · pCTR₂ / pCTR₁` per click. *Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/notebook.ipynb).

**Secret** — A credential injected at run time, never stored in code or images. *Example:* a database password. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Seed** — The starting value of a random number generator. *Example:* `random_state=0` makes a run repeatable. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**Selection memo** — The written decision, with the evidence that supports it. *Example:* `artifacts/selection.json`. *Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch07-model-selection/notebook.ipynb).

**Selective labels** — Only the cases the current system let through get labels. *Example:* declined payments never get chargebacks. *Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Self-attention** — Each position builds its output as a weighted mix of the others. *Example:* 2 blocks in `sr.SASRec`. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb). *See also:* [SASRec](#s).

**Self-normalised IPS** — Divide the IPS sum by the sum of the weights. *Example:* a standard variance reducer. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Sentinel value** — A special value meaning "unknown" or "not applicable" (here `sessions = -1`). Must be converted to missing, not treated as data. *Example:* `sessions = -1` in the usage log. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Session** — A group of one user's events close in time. *Example:* one user's clicks on one day. *Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb). *See also:* [Group (query)](#g).

**Session window** — Per key, events closer than a gap belong together. *Example:* a user's burst of activity. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Shadow / canary / blue-green** — Ways to expose a new model gradually or switch back fast. *Example:* canary = 1-10% of traffic. *Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch15-monitoring-and-rollback/notebook.ipynb), [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Shadow deployment** — Running a new model on live inputs without using its outputs, to compare behaviour safely. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Sharding** — Splitting an index or table across machines and querying the parts in parallel. *Example:* 77 GB of HNSW over 4–8 shards. *Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb).

**Showback / chargeback** — Showing each team its cost / actually billing it. *Example:* per-team GPU spend. *Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch55-ml-platform-design/notebook.ipynb).

**Shrinkage (learning rate)** — The fraction of each new tree's prediction that is added. *Example:* `lr = 0.1`. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**Shuffle** — Sends every pair over the network to the reducer that owns its key. *Example:* reducer = `hash(key) % n_reducers`. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Shuffled k-fold** — Split rows at random into k folds, each used once for validation. *Example:* assumes rows are exchangeable. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb).

**Side output** — Late events kept aside for a batch job to reconcile. *Example:* `sideOutputLateData` in Flink. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Signal** — One quality interviewers score, which recurs across rounds. *Example:* `verification`, `communication`. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Significance level α** — The false-positive rate you accept when there is no effect. *Example:* α = 0.05. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Sketch** — A small fixed-size summary with a known error. *Example:* HyperLogLog, t-digest, a reservoir. *Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/notebook.ipynb).

**Skew (training/serving)** — Any difference between feature computation in training and in serving. Silent and harmful. *Example:* a forgotten `log1p` in the API. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb), [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb), [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**Slice analysis** — Computing metrics separately for subgroups to find where the model fails. *Example:* `plan = basic`, `tenure 0-30d`. *Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch09-error-analysis-and-artifacts/notebook.ipynb).

**Sliding window** — Fixed size, advancing by a smaller hop; windows overlap. *Example:* 1 hour, every 15 min. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**SLO (service level objective)** — A target for a service metric over a window, such as a success rate or latency percentile. *Example:* 99.9% of requests under 200 ms over 30 days. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb), [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Snapshot** — A (date, entity) row in the training table, with features from before the date and a label from after. *Example:* the row for day `d`, `snapshot_ts = d 00:00`. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb).

**Snapshot date (T)** — The day a prediction is made; only data from before T may be used. *Example:* the first day of each month. *Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb), [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**Soft labels** — The teacher's probabilities used as training targets instead of 0/1 clicks. *Example:* `teacher_probs`. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb), [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Spearman correlation** — The correlation of ranks: do two models order items the same way? *Example:* 0.997 between ORT int8 and fp32. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**Split** — Which data a number was measured on (held-out users, last 2 weeks, 5 seeds…). *Example:* "name the split". *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**SSE** — Sum of squared differences from the mean: how spread out a group is. *Example:* `S₂ − S₁²/m` in 14.1. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb). *See also:* [Group (query)](#g).

**Stable hash sample** — Keep a key if hash(salt + key) falls in a bucket; same answer everywhere. *Example:* `sha256(f"{salt}:{user_id}")`. *Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch19-event-logs-and-columnar-data/notebook.ipynb).

**Stable tie-break** — A rule that decides equal keys the same way every time. *Example:* "the later record wins". *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb).

**Standard error (SE)** — The typical size of an estimate's random error across repeated experiments. *Example:* naive SE 0.0006 vs a true spread of 0.0017. *Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb).

**Standardise** — Subtract the training mean and divide by the training standard deviation. *Example:* $z = (x - \mu)/\sigma$. *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb).

**STAR** — Situation, Task, Action, Result: a structure for behavioural interview answers. *Example:* most of the time on Action. *Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch18-system-design-and-mock-interview/notebook.ipynb), [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Straggler** — One slow rank that every other rank has to wait for at each collective. *Example:* a ring is as fast as its slowest link. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb). *See also:* [Collective](#c).

**Stratified split** — A split that keeps each class's share in both parts. *Example:* 5% churners in train and test. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Structured logging** — Logs as one JSON object per line with named fields, so they can be filtered and aggregated. *Example:* `{"request_id": ..., "status": 200, "ms": 4.1}`. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Subgroup** — The comments that mention one identity. *Example:* comments mentioning *muslim*. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Subgroup AUC** — Ranking quality measured only inside one subgroup. *Example:* 0.78–0.80 for four groups. *Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb).

**Subword tokenisation** — Splitting text into frequent pieces: common words whole, rare words in parts. *Example:* `un ##bel ##ie ##va ##bly`. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Support problem** — Items the old policy never showed cannot be evaluated. *Example:* no clicks to re-weight. *Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb).

**Sweep** — Sort events by position, then process them in one pass. *Example:* merging intervals; meeting rooms. *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Synchronize** — Make the CPU wait until every queued GPU kernel has finished. *Example:* `torch.cuda.synchronize()`, inside `timer` and `prof.benchmark`. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

[Back to top](#glossary)

## T

**Tail latency** — The slow end of the distribution (p95 and above). *Example:* what users complain about. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb), [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch56-large-scale-design-cases/notebook.ipynb).

**Target encoding** — Replacing a category with the mean label for that category. *Example:* "this customer's churn rate". *Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch06-pipelines-and-cross-validation/notebook.ipynb). *See also:* [Churn rate](#c).

**Target modules** — Which Linear layers receive adapters. *Example:* `q_lin`, `v_lin`. *Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/notebook.ipynb).

**Temperature τ** — Divides the scores before the softmax; small τ sharpens it. *Example:* τ = 0.05. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb), [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch46-contrastive-training-and-reranking/notebook.ipynb).

**Temperature scaling** — Dividing all logits by one fitted number T before the softmax. *Example:* T = 0.59 on the CPU budget. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Temperature T** — A divisor on logits that softens probabilities so near-misses carry signal. *Example:* `T = 2.0`. *Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb), [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Temperature, top-k** — Divide logits by τ before softmax / keep only the k most likely tokens. *Example:* τ = 0.8, k = 10 in 3.6. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Temporal leakage** — Training on data from after the evaluation period, typically via a random split of time-ordered data. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb).

**Tensor Core** — A unit that multiplies small fp16 matrices many times faster than ordinary cores. *Example:* why fp16 autocast speeds up a T4 (1.5). *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb).

**Tensor parallelism** — Splitting the matrices of one layer across GPUs. *Example:* column- then row-parallel MLP in 10.2. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb).

**`TestClient`** — Calls the app in-process with real HTTP semantics, no server. *Example:* `client.post("/predict", json=body)`. *Taught in:* [L1 · 09 FastAPI & contracts](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb).

**Test that fails on a bug** — An assertion that a correct function passes and a wrong one does not. *Example:* the grader runs your test against hidden buggy versions. *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb).

**TF-IDF** — Word counts re-weighted so that words frequent in one text but rare overall count most. *Example:* the 8k uni- and bigram baseline in 5.1. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Threshold** — The score above which the model's output triggers an action. *Example:* 0.19. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**Throughput** — Requests completed per second. *Example:* `throughput_rps` in the load test. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb), [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

**Tied embeddings** — The same item table embeds inputs and scores outputs. *Example:* hidden state · item embedding. *Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch28-sequence-models-for-user-behavior/notebook.ipynb).

**Time complexity (Big-O)** — How the work grows with input size $n$. *Example:* a loop inside a loop over $n$ is $O(n^2)$. *Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch17-python-and-sql-drills/notebook.ipynb), [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Timeout** — The longest a client waits before giving up on a call. *Example:* 5 s by default. *Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch12-api-clients-and-tests/notebook.ipynb).

**Time split** — Training on earlier periods, validating and testing on later ones. *Example:* train to 2024-03, validate 2024-04..06. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb). *See also:* [`train()` / `eval()`](#t).

**Token id** — The integer index of a piece in the vocabulary; what the model actually receives. *Example:* `[CLS]` = 101, `[SEP]` = 102, `[PAD]` = 0. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Token / vocabulary** — The unit the model reads (here one character) / the set of all of them. *Example:* 65 distinct characters. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Tolerance** — The largest difference a test accepts as "equal". *Example:* 1e-6, because the API rounds to 6 decimals. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb), [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch36-model-optimization-for-inference/notebook.ipynb).

**Topological order** — An order in which every node comes after all its prerequisites. *Example:* `course_order` (Exercise 2). *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**torchrun** — The launcher that starts one process per GPU and sets RANK, WORLD_SIZE, …. *Example:* `torchrun --standalone --nproc_per_node=4`. *Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch47-distributed-data-parallel/notebook.ipynb).

**TP / FP / FN / TN** — Flagged churner / flagged non-churner / missed churner / correctly ignored. *Example:* 64 calls a month, ≈ 27% of them TPs. *Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch08-thresholds-and-calibration/notebook.ipynb).

**`train()` / `eval()`** — Switch layer behaviour (dropout, batch norm). *Example:* (3) and (2) in the loop. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Transfer set** — Unlabelled text the teacher labels so the student sees more examples. *Example:* the 1,000 extra AG News texts. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Trie** — A tree of characters in which words with a shared prefix share nodes. *Example:* Exercise 7; autocomplete. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Truncation** — Cutting sequences longer than the model's limit. *Example:* `max_length=16`; DistilBERT's limit is 512. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**TTL** — The maximum age before a stored value counts as missing. *Example:* 1 day for daily snapshots. *Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/notebook.ipynb), [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch35-serving-a-ranking-cascade/notebook.ipynb).

**Tumbling window** — Fixed-size, non-overlapping time buckets. *Example:* clicks per category per 15 min. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb).

**Twelve-factor config** — Anything that differs between deployments comes from the environment, not the code. *Example:* same image in dev and production. *Taught in:* [L1 · 12 Packaging & configuration](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch13-packaging-and-configuration/notebook.ipynb).

**Two pointers / sliding window** — Two indices that only move forward over a sequence. *Example:* `min_subarray_len` in 15.4. *Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch33-coding-patterns-1/notebook.ipynb).

**Two-tower model** — Two networks embed users and items separately; the score is a dot product. *Example:* `TwoTower` in 8.1. *Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/notebook.ipynb).

[Back to top](#glossary)

## U

**Union-find** — Sets that can be merged, with a fast "which set is x in?". *Example:* Exercise 8. *Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch34-coding-patterns-2/notebook.ipynb).

**Unique key** — The columns that should identify exactly one row. *Example:* (`customer_id`, `week_start`) in `usage_weekly`. *Taught in:* [L1 · 02 Data acquisition & validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb).

**Unseen category** — A categorical value not present in training. Handled here by `handle_unknown="ignore"` and counted by monitoring. *Example:* a new region; one-hot gives all zeros. *Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch11-consistent-serving/notebook.ipynb). *See also:* [`handle_unknown="ignore"`](#h).

**Upstream gradient** — The gradient of the loss with respect to a layer's output. *Example:* `dout` in the 14.2 table. *Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch32-ml-from-scratch-2/notebook.ipynb).

**`USE_DRIVE`** — The setup-cell switch that stores artifacts in Google Drive so they survive Colab disconnects. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb), [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch02-reproducible-project/notebook.ipynb).

**Utilisation `u`** — Fraction of paid hours the server is actually busy. *Example:* 40% in the cost table. *Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch50-high-throughput-serving/notebook.ipynb).

[Back to top](#glossary)

## V

**Validation-test gap** — How much the score changes on the untouched period. *Example:* 0.163 on validation vs ≈ 0.197 on test. *Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch16-final-evaluation-and-docs/notebook.ipynb).

**Validation vs test** — Validation is for choosing and checking; test is touched once at the end. *Example:* test = Jul-Nov 2024, notebook 15. *Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb), [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/notebook.ipynb), [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb).

**Vectorisation** — Replacing Python loops with whole-array NumPy operations. *Example:* pairwise distances: hundreds of ms with loops, under 1 ms as one matrix product. *Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch31-ml-from-scratch-1/notebook.ipynb).

**Verbosity bias** — A judge preferring longer answers that add nothing. *Example:* the padded "notes" field. *Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb).

[Back to top](#glossary)

## W

**Warmup** — Starting the learning rate near 0 and raising it over the first steps. *Example:* 10% of steps, then cosine decay. *Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch43-fine-tuning-a-classifier/notebook.ipynb), [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/notebook.ipynb). *See also:* [Cosine decay](#c).

**Warmup + cosine** — Ramp the learning rate up linearly, then decay it along a cosine. *Example:* `tk.warmup_cosine`. *Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb).

**Watermark** — The processor's claim: events up to time W have (almost) all arrived. *Example:* max event time − allowed lateness. *Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch22-streaming-features/notebook.ipynb). *See also:* [Allowed lateness](#a).

**Weakest signal** — The signal with the lowest mean across the rounds that score it. *Example:* `verification` (2.0). *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Weighted overall** — A weighted mean of per-round means. *Example:* 2.77 for the EXAMPLE attempt. *Taught in:* [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb).

**Weight-only quantisation** — Only the weights are low-bit; the arithmetic still runs in floating point. *Example:* the int8/int4 simulation in 11.3. *Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch49-quantization-and-distillation/notebook.ipynb).

**Weight tying** — The output layer reuses the input embedding matrix. *Example:* `weights tied: True`. *Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb).

**Window function** — A SQL value computed from related rows without collapsing them. *Example:* "the latest invoice per customer". *Taught in:* [L1 · 00 Skills diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb), [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch04-sql-for-ml/notebook.ipynb).

**WordPiece / byte-level BPE** — BERT's and GPT-2's ways of learning those pieces from a corpus. *Example:* 30,522 vs 50,257 tokens. *Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb).

**Worker** — A separate server process with its own interpreter and model copy. *Example:* `uvicorn --workers 2`. *Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch14-health-logs-and-latency/notebook.ipynb).

**Worker / prefetch** — A background process that prepares the next batches while the model trains. *Example:* `num_workers=2` in 1.7. *Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/notebook.ipynb). *See also:* [`num_workers`](#n).

[Back to top](#glossary)

## Z

**ZeRO stage 1/2/3** — Shard optimizer state / + gradients / + parameters across data-parallel GPUs. *Example:* 7B on 8 GPUs: 38.5 / 26.2 / 14.0 GB. *Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch48-scaling-to-large-models/notebook.ipynb). *See also:* [Optimizer state](#o).

[Back to top](#glossary)
