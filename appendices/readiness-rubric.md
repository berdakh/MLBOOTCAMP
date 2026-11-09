# Readiness rubric

This rubric is a way of judging, from evidence, whether you can carry out the
core tasks of the work **independently**. It rates three abilities in each of
the areas that the book teaches:

- **Explain**: state what to do and why, clearly, under follow-up questions.
- **Code**: produce it, from an empty file, in a reasonable time.
- **Troubleshoot**: find and fix it when it is broken and you did not break it.

Recognising a correct answer when it is shown to you is not on the scale.
Interviews and the work itself require production, not recognition.

Areas 1–7 are taught in Part I and practised in the capstones. Areas 8 and 9
are taught in Parts II and III.


---

## The ratings

To avoid confusion with the parts of the book, the four steps of the scale
are called *ratings*.

| Rating | Name | Description |
|---|---|---|
| **1** | Aware | You recognise the idea and can follow worked code, but cannot produce it alone |
| **2** | Assisted | You can do it slowly, with references, examples or the lab open |
| **3** | Independent | You can do it alone, from an empty file, in the time an interview allows, and explain your choices |
| **4** | Fluent | You do it quickly, handle variations and follow-up questions, and can teach it or find mistakes in other people's work |

**Target for interviews:** rating 3 for Explain and Code in every area
relevant to the role, and rating 3 for Troubleshoot in the areas central to
it: data and SQL and evaluation for applied scientist roles; in addition APIs,
deployment and operations, and monitoring for machine learning engineer roles.
A rating of 4 in some area distinguishes a candidate.

Rate yourself only from **evidence produced in a self-test** (the protocol is
below), not from how familiar something feels.

## What each part aims at

| Part | Areas | Target after the part and its practice |
|---|---|---|
| Part I (Chapters 1–18) and the capstones | 1–7 | Rating 3 for Explain and Code in areas 1–7, and rating 3 for Troubleshoot in areas 1 and 2; Part I alone typically reaches rating 2, and the capstones, drills and mock interviews of the [study plan](study-plan.md) supply the rest |
| Part II (Chapters 19–38) | 8, and 1, 2 and 4 at larger scale | Rating 3 for Explain in area 8; rating 3 for Code on the algorithmic and coding-pattern exercises of Chapters 31–34 |
| Part III (Chapters 39–58) | 9 | Rating 3 for Explain in area 9 and in system design; rating 2–3 for Code, since fluency with GPU training requires more practice on real hardware than the book can provide |

---

## 1. Data and SQL

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Can define leakage and point-in-time correctness | Can explain why a random split leaks on TaskFlow, with notes | Explains label, temporal and group leakage with an example of each, and the embargo, without notes | Spots a leak in someone else's query or design and explains the impact before measuring it |
| **Code** | Can modify the book's feature SQL | Writes joins and `GROUP BY` from scratch; window functions with a reference | Writes a point-in-time feature query with correct date boundaries and a window function, unaided, in about 20 minutes | Writes it in a new schema, in pandas and SQL, and adds a test that would catch a boundary bug |
| **Troubleshoot** | Can read a DuckDB error | Fixes type errors and duplicates with hints | Finds why a feature differs from a reference (boundary, duplicate, NULL handling) by isolating one entity | Diagnoses silent problems (`NA` read as missing, late-arriving rows) from symptoms in the model's behaviour |

## 2. Evaluation

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows accuracy is wrong for imbalance | Explains PR-AUC vs ROC-AUC with notes | Explains metrics, cluster bootstrap, paired comparison, threshold from costs (derives 1/6) and calibration, unaided | Adapts the evaluation to a new decision with different costs and capacity, live, and states what would change their mind |
| **Code** | Calls `classification_report` as in the labs | Writes a bootstrap CI with a reference | Writes a group bootstrap and a cost-based threshold search from an empty file | Writes the evaluation for a new problem type (ranking, forecasting) with the right resampling unit |
| **Troubleshoot** | Notices a suspiciously high score | Checks for leakage when prompted | Finds why test disagrees with validation (drift, leakage, selection on validation) and says which | Detects miscalibration, slice failures and metric bugs before they are reported |

## 3. Modular Python and testing

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows why notebooks alone are not sufficient | Explains the package layout of `churnlab` | Explains why each module has its boundary, what a good test asserts, fixtures, and testing with temporary directories | Reviews someone's structure and tests and says what is missing and why it matters |
| **Code** | Edits an existing module | Writes a function and a test with a reference | Writes a small module with type hints, docstrings and pytest tests (including a failure case) from an empty file in about 45 minutes | Refactors a notebook into a tested package quickly; uses parametrisation and fixtures naturally |
| **Troubleshoot** | Reads a traceback | Fixes import and path problems with hints | Fixes stale-module, working-directory and environment problems unaided | Makes flaky tests deterministic; finds why tests pass locally and fail in CI |

## 4. APIs and serving

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows what an endpoint is | Explains the book's schema and application factory with notes | Explains contract decisions, status codes, skew and its prevention, idempotence, retries and batching unaided | Chooses batch or online prediction for a new problem and justifies the trade-offs of latency, cost and failure |
| **Code** | Runs the book's API | Writes an endpoint with a Pydantic model using a reference | Writes a FastAPI app with a schema, `/predict`, `/health`, `/ready` and `TestClient` tests from an empty file in about an hour | Adds a batch endpoint, limits, request ids and a retrying client, with tests, quickly |
| **Troubleshoot** | Reads a 422 body | Fixes 422s and port conflicts with hints | Finds training/serving skew by comparing scores; fixes model-loading and version problems | Diagnoses latency problems by separating model time from overhead |

## 5. Deployment and operations

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows what a container is | Explains the generated Dockerfile with notes | Explains each Dockerfile line, environment configuration, liveness vs readiness, CI steps and rollback unaided | Explains release strategies (shadow, canary, A/B) and chooses one for a given risk |
| **Code** | Edits a settings value | Writes a Dockerfile or CI workflow from a template | Writes a `Settings` class from environment variables, a Dockerfile and a CI workflow that runs tests, from memory | Builds and runs the container and CI for a capstone (the optional external extension) |
| **Troubleshoot** | Recognises a failed health check | Fixes configuration errors with hints | Diagnoses a restart loop, a missing environment variable, or a failed reload from logs | Writes the runbook others use |

## 6. Monitoring

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows models degrade | Explains PSI with notes | Explains the three monitoring layers, drift vs performance, delayed labels and feedback effects unaided | Designs monitoring and alert severities for a new system, including a holdout for unbiased labels |
| **Code** | Calls `drift_report` | Implements PSI with a reference | Implements PSI and a set of alert rules from an empty file | Builds a matured-label performance check joined back to predictions |
| **Troubleshoot** | Reads a drift report | Identifies the drifted feature | Decides whether a drift alert is an upstream data bug, a real shift that matters, or noise, and acts | Leads a rollback-then-investigate response and writes up the cause |

## 7. Communication

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Can describe the project with slides or notes | Walks through the project in 10+ minutes with notes | Walks through TaskFlow or a capstone in 5 minutes, unaided: problem, data, split, metric with interval, decision, limits | Adjusts the explanation to the listener (engineer, manager, scientist) and handles challenging follow-up questions calmly |
| **Code** (writing) | Writes notes for yourself | Writes a README with help | Writes a README and model card a stranger can use: how to run, results with intervals and split, limitations | Writes design documents others review and approve |
| **Troubleshoot** (recovery) | Freezes when stuck | Recovers with a hint | Says "I don't know, here is how I'd find out", and reasons aloud to a sensible answer | Turns a mistake mid-interview into a demonstration of good judgment |

## 8. Ranking, retrieval and experimentation (Part II)

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Can define NDCG and a click-through rate | Explains position bias and the candidate-generation and ranking cascade with notes | Explains position bias and its correction by inverse propensity weighting, the difference between pointwise and listwise ranking, two-tower retrieval with approximate nearest neighbours, and the sample size of an A/B test, unaided | Designs the offline and online evaluation of a new ranking system, including the experiment, its guardrail metrics and its variance reduction, and states what would invalidate it |
| **Code** | Runs the labs' ranking code | Implements NDCG@k and recall@k with a reference | Implements NDCG@k, an inverse-propensity-weighted estimate and a two-sample test with a confidence interval, from an empty file | Implements a small retrieval and ranking pipeline end to end, with tests, on a new data set |
| **Troubleshoot** | Notices that an offline metric and an online result disagree | Checks for position bias or a mismatched candidate set when prompted | Finds why an offline gain did not appear online (bias in the logs, a changed candidate set, an underpowered experiment, a sample ratio mismatch) | Diagnoses interference between experiments or feedback loops from the logged data alone |

## 9. Deep learning systems (Part III)

| | Rating 1 | Rating 2 | Rating 3 | Rating 4 |
|---|---|---|---|---|
| **Explain** | Knows that a GPU executes work asynchronously and has limited memory | Explains mixed precision and data parallelism with notes | Explains where training memory goes, mixed precision, gradient checkpointing, data parallelism, parameter-efficient fine-tuning, quantisation and batched serving, with the trade-offs of each, unaided | Plans the training and serving of a model of a given size on given hardware, with estimates of memory, time and cost |
| **Code** | Runs the labs on a GPU runtime | Writes a PyTorch training loop with a reference | Writes a correct training loop with mixed precision, checkpointing of state and evaluation, and fine-tunes a pretrained model, from an empty file | Adapts a training job to several processes or a smaller memory budget, and measures the result |
| **Troubleshoot** | Reads a CUDA out-of-memory error | Reduces the batch size or sequence length to fix it, with hints | Finds why a loss diverges or a run is slow (wrong timing without synchronisation, input pipeline stalls, numerical overflow in half precision) | Diagnoses hangs and divergence between processes in distributed training |

---

## The self-test protocol

The protocol consists of three drills. Do each area's drills at the end of
Part I, after each capstone, and at the end of Parts II and III. Record the
results in one table, so that change over time is visible.

### Drill 1: rebuilding from an empty file (Code)

1. Choose one piece: a point-in-time SQL feature, a group bootstrap
   confidence interval, a FastAPI application with tests, a `Settings` class,
   PSI with alert rules; for Parts II and III, NDCG@k, a two-sample test, or
   a training loop.
2. Open an **empty file**. Close the labs, the book's source code and AI
   assistants. Documentation for syntax is allowed, and its use should be
   noted.
3. Start a timer. Build it until it works and has at least one test.
4. Rating: 3 if it works within the times given in the tables with at
   most look-ups of syntax; 2 if it needed examples or took much longer; 1
   if you could not finish.

### Drill 2: explaining to a camera (Explain)

1. Choose a question from the [interview questions](interview-questions.md)
   or a checkpoint question from a lab.
2. Record yourself answering it aloud, without notes, in at most 3 minutes.
3. Watch the recording. Compare it with the points listed for the question
   ("A complete answer covers…") or with the chapter's checkpoint answers.
   Count how many points you covered, and note any incorrect statements,
   filler and unexplained jargon.
4. Rating: 3 if you covered most points correctly and clearly; 2 if you
   covered some or were unclear; 1 if you could not structure an answer.

Also record a five-minute explanation of a project after each capstone; it
is the most common opening of a technical interview.

### Drill 3: breaking and fixing (Troubleshoot)

Ask someone to introduce one defect into a copy of a working project, or
write a script that does so and leave it unread for a day. Examples:

| Defect | How it should be found |
|---|---|
| Change a feature's date boundary from `<` to `<=` or use `week_start` instead of week end | A suspicious validation improvement; comparing one customer's feature by hand |
| Read the raw CSV without `keep_default_na=False` | Validation report; category counts |
| Fit the scaler before the cross-validation split | Comparing CV scores with and without the pipeline |
| Resample rows instead of customers in the bootstrap | Interval width |
| Remove `extra="forbid"` from the schema | A test sending a misspelled field |
| Make `/health` load the model | Reading the code; a slow-load test |
| Retry on 422 in the client | A test counting requests |
| Point `CURRENT` at a version saved with a different feature list | Readiness failure message |

Rating: 3 if you find and fix it within 30 minutes and can say how it would
have shown itself in production; 2 if you needed hints; 1 if it was not
found.

---

## Using the ratings

- **Lowest rating first.** The drills and the emphasis of the next capstone
  go to the area and ability with the lowest rating, not to the one you
  enjoy most.
- **Troubleshooting lags behind.** It usually trails Explain and Code by one
  rating. This is normal: troubleshooting skill comes from encountering
  failures, which the capstones provide.
- **Be strict.** If you are unsure whether a result was a 2 or a 3, it was
  a 2.
