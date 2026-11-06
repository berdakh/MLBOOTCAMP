---
title: "1 · The ML engineering job"
parent: Handbook
nav_order: 1
---

# The ML engineering job
{: .no_toc }

What the three job titles this course targets usually mean, what a researcher
already brings, and which research habits need to change.

1. TOC
{:toc}

---

## Three titles, overlapping jobs

Job titles are not standardised. The same title means different work at
different companies, and sometimes in different teams of one company. Read
the job description, not the title. With that warning, the usual centre of
gravity of each role is:

| | ML Engineer | Applied Scientist | Algorithm Engineer |
|---|---|---|---|
| **Core question** | Does the model work reliably in production, and keep working? | Which model or method solves this business problem, and how do we know? | Is this algorithm correct, fast and robust on the target system? |
| **Typical output** | Pipelines, services, training and evaluation infrastructure, monitoring | Experiments, analyses, model prototypes, design documents, sometimes papers | Optimised implementations, signal-processing or estimation code, often for constrained hardware or real-time use |
| **Code expectation** | Production code, reviewed, tested, owned after launch | Clean, reproducible research code; often production-adjacent | Production code, frequently performance-critical (C++, CUDA or optimised Python) |
| **Interviews usually test** | Coding, SQL, ML fundamentals, ML system design, debugging and operations | ML depth, experimental design, statistics, coding, a research or modelling deep dive | Algorithms and data structures, maths, numerical methods, domain depth, coding |
| **Where researchers struggle** | Testing, APIs, deployment, on-call thinking | Translating results into business decisions; SQL | Systems-level performance; production C++ (if required) |

The overlap is large. All three expect you to write code other people can run,
to evaluate a model without fooling yourself, and to explain a decision to a
non-specialist. This course concentrates on exactly that overlap, plus the
production skills that the ML Engineer role adds.

## What a researcher already brings

These transfer directly, and you should say so in interviews, with examples:

| Research habit | Engineering name for it |
|---|---|
| Cross-subject or leave-one-subject-out evaluation | Group-aware splitting; preventing entity leakage |
| Not touching the held-out session until the end | A locked test set; one-time final evaluation |
| Reporting means with confidence intervals or across seeds | Uncertainty in offline metrics; not shipping noise |
| Checking that preprocessing (filtering, ICA, normalisation) is fitted on training data only | Preventing preprocessing leakage; `Pipeline` discipline |
| Artifact rejection, outlier handling, channel quality checks | Data validation and data contracts |
| Building a baseline before a deep model | Honest baselines |
| Reading the literature before building | Knowing prior art; not reinventing a solved problem |
| Explaining results to reviewers who want to reject the paper | Defending decisions under questioning |

If you did EEG/BCI work, your cross-subject evaluation story is one of the
strongest leakage stories you can tell. It is concrete, it has a measurable
before and after, and it maps exactly onto entity leakage in customer, patient
or device data. See the [STAR examples](../guides/interview-questions.html#behavioural-questions).

## Research habits that hurt

These are not character flaws; they are rational in research and costly in
production.

| Habit | Why it is rational in research | Why it hurts in a product team |
|---|---|---|
| Optimise the metric to the third decimal | Papers are compared on benchmark numbers | A 0.003 gain inside the noise is not a result; the threshold and the data matter more |
| One long notebook | Fast iteration, one author | Nobody else can review, test, rerun or deploy it |
| "It works on my machine" environments | You are the only user | The model must run on a server you have never seen, months later |
| Results without decisions | The paper's job is to inform | The team needs to know what to *do*: whom to call, what to flag, when to retrain |
| Novelty first | Novelty gets published | A boring model that is monitored beats a clever one nobody can debug |
| Quiet failure is fine | A failed run just gets rerun | A silent failure in production costs money until someone notices |
| Evaluation once, at the end | The paper is finished when it is published | The model is *starting* when it ships; evaluation continues with monitoring |

## What the job looks like day to day

A rough, typical shape for a modelling-focused engineer in a product team,
across a quarter rather than a day:

1. Understand the decision the model supports and how it is measured.
2. Find and validate the data. Usually the longest step, usually in SQL.
3. Build a baseline and an honest evaluation. Often the baseline ships first.
4. Improve the model while keeping the evaluation honest.
5. Package it, serve it or schedule it, test it.
6. Monitor it, respond when it degrades, retrain, roll back.
7. Write it up so the next person can own it.

Research training concentrates on step 4. The bootcamp spends most of its time
on steps 2, 3, 5 and 6, and Day 5 on 7.

## What interviewers are listening for

Across all three roles, interviewers repeatedly probe the same things:

- **Do you know where numbers come from?** Which split, which period, which
  unit of resampling, how many rows.
- **Do you think about failure?** What happens if the input is wrong, the
  service is down, the data shifts or the label is delayed?
- **Can you trade off?** Latency against accuracy, recall against reviewer
  workload, a simple model you can monitor against a complex one you cannot.
- **Can you communicate?** A clear two-minute explanation beats a complete
  twenty-minute one.

The [interview question bank](../guides/interview-questions.html) is organised
around those four.

## Honest limits of this course

- It teaches with scikit-learn and tabular data because that runs on a free
  CPU and is what most applied interviews probe. It does not teach large-scale
  distributed training or GPU serving.
- It teaches deployment by writing and explaining files. It cannot give you
  the experience of being paged at night for a service you own.
- It cannot give you domain knowledge for the company you apply to. The
  capstones use public data from energy, retail, maintenance, recommendations
  and moderation so you can at least speak to one of those domains.
