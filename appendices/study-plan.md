# Study plan

This appendix gives a schedule for the whole book in **sessions of about
three hours**. At one session a day, Part I takes about a week, the three
capstones that consolidate it about three weeks, and Parts II and III about
two weeks each. The plan is organised by chapter ranges rather than by
calendar dates, so it can be followed at any pace; a section at the end
adapts it to less time.

The plan alternates two kinds of work. The chapters introduce ideas and
techniques with step-by-step labs. The capstones, drills and mock interviews
between the parts turn those techniques into skills that can be exercised
without the labs' guidance.

---

## What Part I provides, and what it does not

It is worth being clear about this, both privately and when describing the
work to others.

| Part I provides | What requires continued practice |
|---|---|
| One working, tested and documented end-to-end project that you can explain | Building the same kind of project again, on less tidy data, without step-by-step labs |
| The vocabulary: leakage, embargo, skew, readiness, drift, rollback, calibration | Using that vocabulary precisely under follow-up questions |
| Point-in-time SQL that you have read and modified | Writing window functions and date joins from an empty editor, quickly and correctly |
| A FastAPI service and tests that you have run | Writing a service and its tests from an empty file in under an hour |
| One worked system design and one scored mock interview | Ten or more designs, spoken aloud, timed, with feedback |
| A first measurement of your gaps | Closing them |

Fluency comes from repetition with feedback, and no text can compress that
process.

## Stage 1: Part I (Chapters 1–18), seven sessions

| Session | Chapters | Content |
|---|---|---|
| 1 | [Before you start](before-you-start.md); [Chapter 1](../part1-foundations/ch01-diagnostic/README.md); [Chapters 2–3](../part1-foundations/ch02-reproducible-project/README.md) | The skills diagnostic; a reproducible project; data acquisition and validation |
| 2 | [Chapters 4–5](../part1-foundations/ch04-sql-for-ml/README.md) | SQL for point-in-time features; leakage-safe splits and baselines |
| 3 | [Chapters 6–9](../part1-foundations/ch06-pipelines-and-cross-validation/README.md) | Pipelines and cross-validation, model selection, thresholds and calibration, error analysis and artifacts |
| 4 | [Chapters 10–12](../part1-foundations/ch10-fastapi-and-contracts/README.md) | FastAPI and contracts, consistent serving, API clients and tests |
| 5 | [Chapters 13–15](../part1-foundations/ch13-packaging-and-configuration/README.md) | Packaging and configuration, health checks and latency, monitoring and rollback |
| 6 | [Chapters 16–18](../part1-foundations/ch16-final-evaluation-and-docs/README.md) | Final evaluation and documentation, Python and SQL drills, system design and a mock interview |
| 7 | Review | Catch up on unfinished exercises; rate yourself on the [readiness rubric](readiness-rubric.md); choose the first capstone |

**Outcome of Stage 1:** the TaskFlow project in your own git repository, a
recorded five-minute explanation of it, and a first set of rubric ratings.

## Stage 2: three capstones, drills and mock interviews (about 21 sessions)

### The shape of a session

| Block | Minutes | Content |
|---|---|---|
| Drills | 30 | SQL or Python, alternately, timed, from an empty editor. The drills of [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md) are the model |
| Capstone | 120 | The current capstone, following its brief |
| Explanation | 30 | Explain the session's capstone work aloud or in writing, as to an interviewer, and answer two questions from the [interview questions](interview-questions.md) |

In a session devoted to a mock interview, the mock and its review replace the
capstone block.

### Capstone A, with SQL fluency (seven sessions)

| Sessions | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 1–4 | Minimum viable project: data download with checksums, validation, a leakage-safe split, a baseline, evaluation with confidence intervals | SQL: joins and aggregation, then window functions (`ROW_NUMBER`, `LAG`, running sums) |
| 5 | Package, tests, README, model card | Python: a small module with tests, from an empty file |
| 6 | **Mock interview 1**: a detailed walkthrough of the TaskFlow project (30 min) and a system design (40 min), recorded | — |
| 7 | Review of the recording; rubric ratings | — |

### Capstone B, with serving and testing fluency (seven sessions)

| Sessions | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 1–4 | Minimum viable project, plus a FastAPI service or a batch scoring job, with tests | SQL: date logic and point-in-time joins, with pandas equivalents |
| 5 | Monitoring outline (PSI on a later period), README, model card | Python: a retrying client and its tests, from an empty file |
| 6 | **Mock interview 2**: coding (45 min, pandas or SQL) and behavioural questions (20 min, in the STAR format) | — |
| 7 | Review; rubric ratings | — |

### Capstone C, with interview readiness (seven sessions)

| Sessions | Capstone (120 min) | Drills (30 min) |
|---|---|---|
| 1–4 | Minimum viable project, concentrating on the skill that the rubric rates lowest | Mixed and timed: one SQL, one Python and one evaluation question per session |
| 5 | Final revision of all three capstones: each has a README with results, intervals, limitations and instructions for running it | Repeat the diagnostic of Chapter 1 |
| 6 | **Mock interview 3**: a full interview loop, preferably with another person: coding, depth in machine learning, system design, behavioural questions | — |
| 7 | Final rubric ratings; decide what to continue practising | — |

### Which capstones, in which order

The five capstone briefs are in [Part IV](../part4-capstones/README.md),
which recommends an order and gives its reasons. At the pace above, **three
minimum viable projects** fit into Stage 2 alongside drills and mock
interviews: a minimum viable project takes four to five sessions, and
extensions take longer. Part IV recommends C-MAPSS first and ERCOT second, as
capstones A and B, and as capstone C the project closest to the roles you are
applying for. The remaining two can be done later. To choose the third:

| Target field | Capstone to prioritise |
|---|---|
| Energy, utilities, forecasting | [ERCOT load forecasting](../part4-capstones/cap2-electricity-load-forecasting/README.md) |
| Industrial, IoT, sensor and signal-processing roles (closest to EEG work) | [C-MAPSS predictive maintenance](../part4-capstones/cap1-predictive-maintenance/README.md) |
| Retail, marketplaces, supply chain | [Online Retail demand forecasting](../part4-capstones/cap3-retail-demand-forecasting/README.md) |
| Recommendation, ranking, personalisation | [MovieLens recommendations](../part4-capstones/cap4-movie-recommendations/README.md) |
| Trust and safety, content, natural language processing | [Civil Comments moderation](../part4-capstones/cap5-content-moderation/README.md) |

Each capstone has a guided solution (`walkthrough.ipynb` in its folder).
Consult it after building your own version, not before; see
[About the solutions](solutions.md).

## Mock interviews

- **Record every mock interview.** Watching the recording is uncomfortable,
  and it is where most of the learning happens.
- **Practise with another person when possible**: a colleague, an
  acquaintance working in industry, or a peer-practice group. A self-scored
  mock is better than none, but self-assessment tends to be lenient.
- **Score with the [readiness rubric](readiness-rubric.md)**, and write down
  the two weakest moments together with what a better answer would have
  contained.
- **Vary the format**: a detailed project walkthrough, live coding, system
  design, breadth questions on machine learning, and behavioural questions.

## Stage 3: Part II (Chapters 19–38), about ten sessions

Stages 1 and 2 prepare for roles centred on modelling at most companies. For
machine learning engineering roles at large technology companies, continue
with Parts II and III. Both are dense; some sessions will take longer than
three hours, and it is more important to identify honestly what remains
uncertain, using each part's final chapter, than to finish on schedule.

| Session | Chapters | Content |
|---|---|---|
| 1 | [19–20](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md) | Event logs and columnar data; aggregation and joins at scale |
| 2 | [21–22](../part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md) | Feature pipelines and feature stores; streaming features |
| 3 | [23–24](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md) | Implicit feedback and position bias; learning to rank |
| 4 | [25–26](../part2-ml-at-scale/ch25-candidate-generation/README.md) | Candidate generation; two-tower retrieval and approximate nearest neighbours |
| 5 | [27–28](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md) | PyTorch training; sequence models for user behaviour |
| 6 | [29–30](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md) | A/B testing; advanced experimentation |
| 7 | [31–32](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md) | Machine learning algorithms implemented from scratch |
| 8 | [33–34](../part2-ml-at-scale/ch33-coding-patterns-1/README.md) | Coding patterns for interviews |
| 9 | [35–36](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md) | Serving a ranking cascade; model optimisation for inference |
| 10 | [37–38](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md) | Recommendation and search system design; final evaluation and a mock interview loop |

In parallel, continue with 30 minutes of drills per session, using the
coding patterns of Chapters 33–34 once they have been studied.

## Stage 4: Part III (Chapters 39–58), about ten sessions

Run the labs on a Colab T4 GPU (see [Working in Colab](colab.md)).

| Session | Chapters | Content |
|---|---|---|
| 1 | [39–40](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md) | GPU fundamentals and profiling; training efficiency and memory |
| 2 | [41–42](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md) | A transformer from scratch; pretrained transformers and tokenisation |
| 3 | [43–44](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md) | Fine-tuning a classifier; parameter-efficient fine-tuning |
| 4 | [45–46](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md) | Embeddings and semantic search; contrastive training and re-ranking |
| 5 | [47–48](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md) | Distributed data parallelism; scaling to large models |
| 6 | [49–50](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md) | Quantisation and distillation; high-throughput serving |
| 7 | [51–52](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md) | Click-through-rate prediction at scale; multi-task ranking and exploration |
| 8 | [53–54](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md) | Fairness, robustness and privacy; evaluating LLM systems and guardrails |
| 9 | [55–56](../part3-deep-learning-systems/ch55-ml-platform-design/README.md) | ML platform design; large-scale design cases |
| 10 | [57–58](../part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md) | Depth in machine learning and research discussion; a full interview loop and portfolio |

In parallel, continue the drills and do one system design, aloud and timed,
every few sessions.

## Stage 5: integration (about ten sessions)

Extend one capstone with a technique from Parts II–III, for example MovieLens
with a two-tower retriever and LambdaMART re-ranking, or Civil Comments with
a fine-tuned transformer and a subgroup analysis. Finish with two full mock
interview loops in the format of
[Chapter 58](../part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/README.md),
with a real interviewer if possible.

## With less time

| Time available | Keep | Drop or postpone |
|---|---|---|
| Two hours a day | Part I over eight to nine sessions; drills in every session; one mock interview after each capstone | The third capstone |
| One hour a day | Part I over about two weeks, reading each chapter's text and doing the core sections and graded exercises of its lab; drills | Capstones, until Part I is complete |
| Weekends only | Part I at two to three chapters per weekend day; 20 minutes of drills on weekday evenings | Expect Stages 1 and 2 to take about three months |

## After the plan

Continue the drills at 20–30 minutes a day, since fluency fades quickly
without them, and keep extending the set of interview questions you have
practised aloud. Real interviews are themselves practice, and the feedback
they give is the most accurate measure of readiness available.
