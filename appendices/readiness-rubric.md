# Readiness rubric

A way to judge, honestly, whether you can do the job's core tasks **on your
own**. It scores three abilities in each of seven areas:

- **Explain**: say what to do and why, clearly, under follow-up questions.
- **Code**: produce it, from a blank file, in reasonable time.
- **Troubleshoot**: find and fix it when it is broken and you did not break it.

Recognising the right answer when you see it is not on the scale. Interviews
and jobs need production, not recognition.


---

## The levels

| Level | Name | In one line |
|---|---|---|
| **1** | Aware | You recognise the idea and can follow worked code, but cannot produce it alone |
| **2** | Assisted | You can do it with references, examples or the notebook open, slowly |
| **3** | Independent | You can do it alone, from a blank file, in interview time, and explain choices |
| **4** | Fluent | You do it quickly, handle variations and follow-ups, and can teach it or spot others' mistakes |

**Target for interviewing:** level 3 in every area for Explain and Code, level 3
in Troubleshoot for the areas central to the role (data/SQL and evaluation for
Applied Scientist; APIs, deployment/ops and monitoring as well for ML
Engineer). Level 4 somewhere is what makes you memorable.

Score only from **evidence produced in a self-test** (protocol below), not
from how familiar something feels.

---

## 1. Data and SQL

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Can define leakage and point-in-time | Can explain why a random split leaks on TaskFlow, with notes | Explains label, temporal and group leakage with an example of each, and the embargo, without notes | Spots a leak in someone else's query or design and explains the impact before measuring it |
| **Code** | Can modify the course's feature SQL | Writes joins and `GROUP BY` from scratch; window functions with a reference | Writes a point-in-time feature query with correct date boundaries and a window function, unaided, in about 20 minutes | Writes it in a new schema, in pandas and SQL, and adds a test that would catch a boundary bug |
| **Troubleshoot** | Can read a DuckDB error | Fixes type errors and duplicates with hints | Finds why a feature differs from a reference (boundary, duplicate, NULL handling) by isolating one entity | Diagnoses silent problems (`NA` read as missing, late-arriving rows) from symptoms in the model's behaviour |

## 2. Evaluation

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Knows accuracy is wrong for imbalance | Explains PR-AUC vs ROC-AUC with notes | Explains metrics, cluster bootstrap, paired comparison, threshold from costs (derives 1/6) and calibration, unaided | Adapts the evaluation to a new decision with different costs and capacity, live, and states what would change their mind |
| **Code** | Calls `classification_report` from the course | Writes a bootstrap CI with a reference | Writes a group bootstrap and a cost-based threshold search from a blank file | Writes the evaluation for a new problem type (ranking, forecasting) with the right resampling unit |
| **Troubleshoot** | Notices a suspiciously high score | Checks for leakage when prompted | Finds why test disagrees with validation (drift, leakage, selection on validation) and says which | Detects miscalibration, slice failures and metric bugs before they are reported |

## 3. Modular Python and testing

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Knows why notebooks are not enough | Explains the package layout of `churnlab` | Explains why each module has its boundary, what a good test asserts, fixtures, and testing with temporary directories | Reviews someone's structure and tests and says what is missing and why it matters |
| **Code** | Edits an existing module | Writes a function and a test with a reference | Writes a small module with type hints, docstrings and pytest tests (including a failure case) from a blank file in about 45 minutes | Refactors a notebook into a tested package quickly; uses parametrisation and fixtures naturally |
| **Troubleshoot** | Reads a traceback | Fixes import and path problems with hints | Fixes stale-module, working-directory and environment problems unaided | Makes flaky tests deterministic; finds why tests pass locally and fail in CI |

## 4. APIs and serving

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Knows what an endpoint is | Explains the course's schema and app factory with notes | Explains contract decisions, status codes, skew and its prevention, idempotence, retries and batching unaided | Chooses batch vs online for a new problem and defends latency, cost and failure trade-offs |
| **Code** | Runs the course's API | Writes an endpoint with a Pydantic model using a reference | Writes a FastAPI app with a schema, `/predict`, `/health`, `/ready` and `TestClient` tests from a blank file in about an hour | Adds a batch endpoint, limits, request ids and a retrying client, with tests, quickly |
| **Troubleshoot** | Reads a 422 body | Fixes 422s and port conflicts with hints | Finds training/serving skew by comparing scores; fixes model-loading and version problems | Diagnoses latency problems by separating model time from overhead |

## 5. Deployment and operations

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Knows what a container is | Explains the generated Dockerfile with notes | Explains each Dockerfile line, environment configuration, liveness vs readiness, CI steps and rollback unaided | Explains release strategies (shadow, canary, A/B) and chooses one for a given risk |
| **Code** | Edits a settings value | Writes a Dockerfile or CI workflow from a template | Writes a `Settings` class from environment variables, a Dockerfile and a CI workflow that runs tests, from memory | Builds and runs the container and CI for a capstone (the optional external extension) |
| **Troubleshoot** | Recognises a failed health check | Fixes configuration errors with hints | Diagnoses a restart loop, a missing environment variable, or a failed reload from logs | Writes the runbook others use |

## 6. Monitoring

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Knows models degrade | Explains PSI with notes | Explains the three monitoring layers, drift vs performance, delayed labels and feedback effects unaided | Designs monitoring and alert severities for a new system, including a holdout for unbiased labels |
| **Code** | Calls `drift_report` | Implements PSI with a reference | Implements PSI and a set of alert rules from a blank file | Builds a matured-label performance check joined back to predictions |
| **Troubleshoot** | Reads a drift report | Identifies the drifted feature | Decides whether a drift alert is an upstream data bug, a real shift that matters, or noise, and acts | Leads a rollback-then-investigate response and writes up the cause |

## 7. Communication

| | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---|---|---|---|
| **Explain** | Can describe the project with slides or notes | Walks through the project in 10+ minutes with notes | Walks through TaskFlow or a capstone in 5 minutes, unaided: problem, data, split, metric with interval, decision, limits | Adjusts the explanation to the listener (engineer, manager, scientist) and handles hostile follow-ups calmly |
| **Code** (writing) | Writes notes for yourself | Writes a README with help | Writes a README and model card a stranger can use: how to run, results with intervals and split, limitations | Writes design documents others review and approve |
| **Troubleshoot** (recovery) | Freezes when stuck | Recovers with a hint | Says "I don't know, here is how I'd find out", and reasons aloud to a sensible answer | Turns a mistake mid-interview into a demonstration of good judgment |

---

## The self-test protocol

Three drills. Do each area's drills at the end of week 1 and of each later
week. Record the results in one table so you can see movement.

### Drill 1: blank-file rebuild (Code)

1. Pick one piece: a point-in-time SQL feature, a group bootstrap CI, a
   FastAPI app with tests, a `Settings` class, PSI with alert rules.
2. Open an **empty file**. Close the notebooks, the course source and AI
   assistants. Documentation for syntax is allowed and should be noted.
3. Start a timer. Build it until it works and has at least one test.
4. Score: level 3 if it works within the times given in the tables with at
   most syntax lookups; level 2 if it needed examples or much longer; level 1
   if you could not finish.

### Drill 2: explain to camera (Explain)

1. Pick a question from the [question bank](interview-questions.md) or a
   checkpoint question.
2. Record yourself answering it aloud, without notes, in at most 3 minutes.
3. Watch it back. Compare with "a strong answer covers…". Count how many
   points you hit, and note any wrong statements, filler and unexplained
   jargon.
4. Score: level 3 if you covered most points correctly and clearly; level 2 if
   you covered some or were unclear; level 1 if you could not structure an
   answer.

Also record a five-minute project walkthrough once a week; it is the most
common opening of a technical interview.

### Drill 3: break and fix (Troubleshoot)

Have someone (or a script you write and then forget for a day) introduce one
bug into a copy of a working project. Some to try:

| Break | Should be found by |
|---|---|
| Change a feature's date boundary from `<` to `<=` or use `week_start` instead of week end | A suspicious validation improvement; comparing one customer's feature by hand |
| Read the raw CSV without `keep_default_na=False` | Validation report; category counts |
| Fit the scaler before the cross-validation split | Comparing CV scores with and without the pipeline |
| Resample rows instead of customers in the bootstrap | Interval width |
| Remove `extra="forbid"` from the schema | A test sending a misspelled field |
| Make `/health` load the model | Reading the code; a slow-load test |
| Retry on 422 in the client | A test counting requests |
| Point `CURRENT` at a version saved with a different feature list | Readiness failure message |

Score: level 3 if you find and fix it within 30 minutes and can say how it
would have shown up in production; level 2 with hints; level 1 if not found.

---

## Using the scores

- **Lowest score first.** Next week's drills and capstone emphasis go to the
  area and ability with the lowest score, not the one you enjoy.
- **Troubleshoot lags.** It usually trails Explain and Code by a level. That is
  normal; it comes from encountering failures, which capstones provide.
- **Be strict.** If you are not sure whether it was a 2 or a 3, it was a 2.
