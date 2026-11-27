# Interview questions

This appendix lists questions by topic, each followed by the points that **a
complete answer covers**. The points are not a script to memorise. They are
intended for the *explaining to a camera* drill of the
[readiness rubric](readiness-rubric.md): answer the question aloud first, then
compare your answer with the points.

The points are those that interviewers commonly listen for. A good answer
need not contain all of them, and it should be expressed in your own words
with your own examples. Each section names the chapters in which its topic is
taught; the checkpoint questions at the end of every lab, with their answers
in the chapter's `solutions.md`, provide further practice.


---

<a id="data-sql-and-leakage"></a>

## Data, SQL and leakage

*Taught in:* [Chapter 3](../part1-foundations/ch03-data-acquisition-and-validation/README.md), [Chapter 4](../part1-foundations/ch04-sql-for-ml/README.md), [Chapter 5](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md), [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md); at scale, [Chapters 19–21](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md).

### Write a query giving, for each customer and snapshot date, their sessions in the 28 days before the snapshot.

A complete answer covers: asking how time is recorded (event time vs week start,
time zones, late arrivals); a join on customer with a date range condition
strictly before the snapshot; handling customers with no activity (a `LEFT
JOIN` and `COALESCE` to 0, or explicitly leaving NULL and saying why); checking
the result on one customer by hand.

### What is the difference between `ROW_NUMBER`, `RANK` and `DENSE_RANK`? When would you use each?

A complete answer covers: ties (unique numbers vs gaps vs no gaps); a concrete use
such as "latest invoice per customer" with `ROW_NUMBER() OVER (PARTITION BY
customer_id ORDER BY invoice_date DESC)` and filtering `= 1`; that tie-breaking
must be deterministic (add a second `ORDER BY` key).

### Explain point-in-time correctness.

A complete answer covers: features from data strictly before the prediction
time, labels from the window after; why "strictly" (events on the boundary may
be recorded after the decision); period boundaries (a weekly bucket is known at
its end); columns updated in place and current-state dimension tables; how you
would test it.

### What is data leakage? Give three kinds.

A complete answer covers: label (target) leakage, temporal leakage, group/entity
leakage, preprocessing leakage; one concrete example of each; how each inflates
offline metrics; how each is prevented (point-in-time features, time split
with embargo, group split, pipelines).

### Your model's validation score jumped after adding one feature. What do you check?

A complete answer covers: suspicion first; whether the feature would exist with
that value at prediction time; its timestamp and how it is populated; its
importance and its relationship with the label; retraining without it; comparing
on a later, untouched period.

### How do you validate a new data source?

A complete answer covers: a written contract (types, nulls, ranges, allowed
values, uniqueness, keys); row counts and freshness; distribution comparisons
with what you expect; defaults and sentinels (`-1`, `9999`, `NA`); documenting
each cleaning decision; running checks automatically on every load.

<a id="evaluation-and-metrics"></a>

## Evaluation and metrics

*Taught in:* [Chapter 5](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md), [Chapter 7](../part1-foundations/ch07-model-selection/README.md), [Chapter 8](../part1-foundations/ch08-thresholds-and-calibration/README.md), [Chapter 9](../part1-foundations/ch09-error-analysis-and-artifacts/README.md), [Chapter 16](../part1-foundations/ch16-final-evaluation-and-docs/README.md).

### Why is accuracy a poor metric for churn prediction? What would you use?

A complete answer covers: about 5% positives in the book's synthetic data, so
"nobody churns" is about 95% accurate; PR-AUC for ranking with its base-rate
baseline; precision, recall and alert volume at the operating threshold;
precision@k for capacity-limited teams; Brier score if probabilities drive
decisions.

### How do you know model B is really better than model A?

A complete answer covers: the same evaluation data; a paired bootstrap of the
difference; resampling the correct unit (customers, not rows); whether the
difference matters at the operating threshold, not only in AUC; a rolling-origin
or multi-period check; the cost of the more complex model.

### How would you choose the classification threshold?

A complete answer covers: start from the decision and its costs; the expected-value
rule (act when `p × value > (1 − p) × cost`); the TaskFlow example (assumed $150
vs $30 gives 1/6); that this requires calibrated probabilities, or an empirical
search on validation; capacity constraints turning it into top-k; stating the
assumptions with the threshold.

### What is calibration and when does it matter?

A complete answer covers: predicted probabilities matching observed frequencies;
reliability tables/curves; Brier score; when decisions use the probability
value (cost thresholds, expected revenue) it matters, when only ranking is used
it does not; Platt or isotonic calibration on held-out data; class weighting
and resampling distort calibration.

### What is a cluster (group) bootstrap and why use it?

A complete answer covers: resampling whole groups (customers, subjects, sites)
instead of rows; rows within a group are correlated, so the row bootstrap's
intervals are too narrow; the research analogy of across-subject variability.

### Test performance is clearly lower than validation. What happened and what do you do?

A complete answer covers: candidates (time drift between periods, leakage in
validation, overfitting to validation through many choices, different base
rates, small sample noise); checking with intervals; *not* re-tuning on test;
reporting both and explaining.

### How do you evaluate across subgroups?

A complete answer covers: slice metrics by meaningful groups; small-slice
uncertainty (flag, don't hide; add intervals); differences in base rate vs
differences in model quality; what action a slice result would trigger.

## Modelling

*Taught in:* [Chapter 6](../part1-foundations/ch06-pipelines-and-cross-validation/README.md), [Chapter 7](../part1-foundations/ch07-model-selection/README.md).

### When would you choose logistic regression over gradient-boosted trees?

A complete answer covers: small or mostly linear (after transforms) problems;
calibration out of the box; interpretability and stability; ease of serving
and monitoring; that the comparison should be measured, not assumed; on the
book's data, Chapter 7 measures whether trees justify their extra complexity.

### Why put preprocessing inside the scikit-learn `Pipeline`?

A complete answer covers: no leakage across CV folds (fitted on training folds
only); one object to save and serve, preventing training/serving skew;
`ColumnTransformer` for per-column treatment; `handle_unknown="ignore"` for
unseen categories.

### How do you handle class imbalance?

A complete answer covers: often you need nothing beyond the right metrics and
threshold; class weights or resampling change calibration; evaluate at the
operating point; more positive data if available; avoid oversampling before
splitting.

### How do you cross-validate time-ordered data?

A complete answer covers: rolling-origin (expanding window) folds; an embargo
when labels look ahead; never shuffling across time; keeping the final test
period out of all of it.

## Reproducibility and code quality

*Taught in:* [Chapter 2](../part1-foundations/ch02-reproducible-project/README.md), [Chapter 12](../part1-foundations/ch12-api-clients-and-tests/README.md), [Chapter 13](../part1-foundations/ch13-packaging-and-configuration/README.md).

### How do you make your experiments reproducible?

A complete answer covers: code version (commit), data version (checksums,
fingerprints), configuration recorded with results, pinned environments,
seeds, run records or an experiment tracker; an example of a result that did
not reproduce and why.

### What do you test in an ML codebase?

A complete answer covers: pure functions (feature logic on tiny hand-made
inputs, including boundary dates); data contracts; that the pipeline trains and
predicts on a small sample; API contract tests; consistency between training
and serving outputs; not asserting exact model metrics, but sanity ranges or
invariants.

### How would you turn a research notebook into production code?

A complete answer covers: extract functions into modules with clear inputs and
outputs; configuration out of the code; tests for each function; a pipeline
object for preprocessing and model; versioned artifacts with metadata; keep the
notebook as a thin demonstration that imports the modules.

## APIs and serving

*Taught in:* [Chapter 10](../part1-foundations/ch10-fastapi-and-contracts/README.md), [Chapter 11](../part1-foundations/ch11-consistent-serving/README.md), [Chapter 12](../part1-foundations/ch12-api-clients-and-tests/README.md), [Chapter 14](../part1-foundations/ch14-health-logs-and-latency/README.md).

### Design the API for a churn model.

A complete answer covers: batch or online first; request and response schemas
with types and bounds; rejecting unknown fields; handling unknown categories;
model version in responses; status codes (422, 413, 503); a batch endpoint with
a limit; health and readiness endpoints; documentation generated from the schema.

### What is training/serving skew and how do you prevent it?

A complete answer covers: any difference in feature computation between training
and serving; examples (re-implemented preprocessing, different windows, missing
value encoding, column order, library versions); prevention (save and serve one
pipeline, shared feature code, metadata checks at load); a test that compares
scores for the same rows.

### Which errors should a client retry?

A complete answer covers: connection errors, timeouts, 429, 502/503/504; with
exponential backoff and jitter; never 4xx validation errors; a timeout on every
call; idempotence (safe for scoring; needs an idempotency key for writes);
request ids for tracing.

### How would you reduce prediction latency?

A complete answer covers: measure percentiles first and separate model time from
overhead; batching; loading the model once; avoiding per-request DataFrame
work; simpler features or model; caching; hardware and concurrency; trading off
accuracy explicitly.

<a id="deployment-and-operations"></a>

## Deployment and operations

*Taught in:* [Chapter 13](../part1-foundations/ch13-packaging-and-configuration/README.md), [Chapter 14](../part1-foundations/ch14-health-logs-and-latency/README.md), [Chapter 15](../part1-foundations/ch15-monitoring-and-rollback/README.md).

### What is the difference between a liveness probe and a readiness probe?

A complete answer covers: liveness is "is the process alive" (restart if not);
readiness is "can it serve" (route traffic away if not); checking the model in
liveness causes restart loops during slow loads; the book's `/health` vs
`/ready`.

### Walk me through this Dockerfile.

A complete answer covers: base image choice and version pinning; installing
dependencies before copying code (layer caching); not running as root; copying
only what is needed; configuration from environment variables, not baked in;
the command that starts the server; where the model comes from (baked in vs
mounted vs downloaded at start-up) and the trade-off.

### How do you roll back a bad model?

A complete answer covers: immutable versioned artifacts; a pointer or alias for
the live version; changing the pointer and reloading; an audit log of
promotions; rollback first, investigate second; keeping the previous version
warm or quickly loadable.

### What would your CI pipeline run?

A complete answer covers: linting, unit tests, data contract tests on a sample, a
small training run, API contract tests, building the image; failing fast;
keeping it quick; separate, slower model-quality checks before promotion.

<a id="monitoring"></a>

## Monitoring

*Taught in:* [Chapter 15](../part1-foundations/ch15-monitoring-and-rollback/README.md).

### How would you monitor this model in production?

A complete answer covers: three layers (service, data/predictions, performance);
which are available immediately and which wait for labels; concrete metrics
and alert severities; delayed labels; feedback effects from acting on
predictions; rollback.

### What is PSI? What are its limitations?

A complete answer covers: binned comparison of current vs reference
distributions using reference quantiles; the conventional 0.1 / 0.25 bands as
rules of thumb; sensitivity to bins and sample size; drift does not imply
worse performance; look at the distributions.

### Labels arrive 30 days late. How do you know if the model is degrading?

A complete answer covers: leading indicators (input drift, score distribution,
share flagged); matured-cohort evaluation when labels arrive; a holdout group
not acted on, so labels stay unbiased; alerting on both.

### Your drift alert fired. What do you do?

A complete answer covers: check for upstream data bugs first (schema change,
broken join, unit change, new category); check whether the drifted feature
matters to the model; check the score distribution and any matured
performance; decide between ignore, fix the pipeline, retrain, or roll back;
write it up.

<a id="ranking-retrieval-and-experimentation"></a>

## Ranking, retrieval and experimentation

*Taught in:* [Chapter 23](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md), [Chapter 24](../part2-ml-at-scale/ch24-learning-to-rank/README.md), [Chapter 25](../part2-ml-at-scale/ch25-candidate-generation/README.md), [Chapter 26](../part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md), [Chapter 29](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md), [Chapter 30](../part2-ml-at-scale/ch30-advanced-experimentation/README.md).

### Why can clicks not be used directly as relevance labels?

A complete answer covers: position bias (items shown higher are examined more
often, whatever their relevance); the examination hypothesis (a click
requires both examination and relevance); items never shown have no labels at
all; presentation effects such as thumbnails; corrections by randomised
experiments (swapping positions) to estimate examination propensities, then
inverse propensity weighting; the variance cost of large weights and
clipping.

### What does NDCG measure, and when is it the wrong metric?

A complete answer covers: graded relevance, a logarithmic discount by
position, normalisation by the ideal ordering so that values lie in [0, 1];
the cut-off k matching how many items a user sees; that it is computed per
query and averaged; that it ignores diversity, novelty and business
constraints; that offline NDCG on logged data inherits the logging policy's
biases.

### Why do recommendation systems use a cascade of candidate generation and ranking?

A complete answer covers: the cost of scoring every item with an expensive
model; cheap, high-recall retrieval (two-tower embeddings with approximate
nearest-neighbour search, co-occurrence, popularity) followed by a precise
ranker on a few hundred candidates; measuring recall@k of the retrieval stage
separately, since the ranker cannot recover an item that was not retrieved;
latency budgets per stage.

### How do you decide how long an A/B test must run?

A complete answer covers: the minimum detectable effect that matters; the
metric's variance; significance level and power; the resulting sample size
per arm; at least one full weekly cycle; not stopping early when the result
first looks significant (or using a sequential method designed for it);
variance reduction with pre-experiment covariates (CUPED) to shorten the test.

### The treatment is significantly better on the primary metric. What else do you check before acting?

A complete answer covers: a sample ratio mismatch check; guardrail metrics
(latency, errors, revenue, retention); novelty and primacy effects over time;
heterogeneous effects in important segments; multiple comparisons if many
metrics were inspected; whether the randomisation unit matches the unit of
analysis.

<a id="deep-learning-systems"></a>

## Deep learning systems

*Taught in:* [Chapter 39](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md), [Chapter 40](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md), [Chapter 44](../part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md), [Chapter 47](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md), [Chapter 49](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md), [Chapter 50](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md).

### Where does GPU memory go during training, and how would you reduce it?

A complete answer covers: parameters, gradients, optimiser state (two extra
values per parameter for Adam) and activations; activations growing with
batch size and sequence length; mixed precision; gradient checkpointing
(recomputing activations in the backward pass); gradient accumulation to keep
the effective batch size; sharding optimiser state across devices; measuring
with the framework's memory statistics rather than guessing.

### How do you time GPU code correctly?

A complete answer covers: GPU kernels run asynchronously, so a timer must
synchronise before reading; warm-up iterations (context creation, autotuning,
allocator growth); reporting the median of repeated runs; separating data
loading time from compute time; using a profiler to find the dominant
operations.

### Explain data-parallel training.

A complete answer covers: each process holds a full copy of the model and
processes a different shard of each batch; gradients are averaged with an
all-reduce before the optimiser step, so all copies stay identical; a
distributed sampler so that shards do not overlap; the effective batch size
and learning-rate scaling; communication overlapping with computation; why
every process must call every collective operation in the same order.

### When would you fine-tune all the weights of a pretrained model, and when only adapters such as LoRA?

A complete answer covers: the memory and storage cost of full fine-tuning;
LoRA adds trainable low-rank matrices to frozen weights, so only a small
fraction of parameters is trained and stored per task; quality is often close
to full fine-tuning but should be measured; many tasks can share one base
model; full fine-tuning when the domain shift is large and resources allow.

### How would you reduce the cost of serving a transformer model?

A complete answer covers: measuring latency percentiles and throughput first;
dynamic batching; shorter or bucketed sequences; quantisation (for example
int8) and its measured effect on accuracy; distillation into a smaller model;
caching; choosing hardware by cost per request at the required latency.

<a id="ml-system-design"></a>

## ML system design

Practise each aloud, in 35–40 minutes, with the framework of
[Chapter 18](../part1-foundations/ch18-system-design-and-mock-interview/README.md).
Larger designs are treated in
[Chapter 37](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md)
and [Chapter 56](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md).

### Design a system to predict which customers will churn next month.

A complete answer covers: the action and capacity; batch scoring; point-in-time
snapshot features; a time split with embargo; precision@capacity and
cost-based thresholds; the 30-day label delay; the treatment effect (contacted
customers' outcomes change; keep a holdout); monitoring and retraining.

### Design a system to flag toxic comments for human review.

A complete answer covers: review capacity as the binding constraint; precision
at the review budget; thresholds for auto-action vs review; labelling quality
and annotator disagreement; subgroup errors (identity terms); adversarial
adaptation; latency needs; feedback from reviewer decisions.

### Design a demand forecasting system for a retailer.

A complete answer covers: forecast unit and horizon; sales are not demand
(stock-outs censor demand); seasonal naive baselines; backtesting with rolling
origins; error metrics that suit intermittent demand; how forecasts drive
ordering; monitoring forecast error by segment.

### Design a recommendation system for a streaming service.

A complete answer covers: candidate generation and ranking stages; implicit
feedback; time-aware evaluation (train on the past, test on later
interactions); Recall@k and NDCG@k; cold start for new users and items;
popularity bias and feedback loops; online A/B testing.

### Design a predictive-maintenance alerting system for industrial equipment.

A complete answer covers: the decision (inspect, replace, ignore) and cost of a
missed failure vs a needless inspection; splitting by unit (machine) not by
time step; remaining-useful-life vs failure-within-horizon framing; sensor
data quality; alert fatigue; edge vs cloud inference.

<a id="behavioural-questions"></a>

## Behavioural questions

Use **STAR**: **S**ituation (one or two sentences of context), **T**ask (what
you were responsible for), **A**ction (what *you* did, specifically), **R**esult
(what changed, with a measured outcome where you have one, and what you
learned). Keep it to about two minutes. Use real numbers from your own work
only; if you do not remember a number, say so rather than inventing one.

Research experience maps onto these questions well. Prepare four or five
stories you can adapt.

### Tell me about a time you found a flaw in an evaluation.

**Mapping from a project: cross-subject evaluation as a leakage story.** This
outline fits any project with repeated measurements per subject, such as
patients, study participants, sensors or devices.

| | Example outline (fill in your own details and numbers) |
|---|---|
| **S** | A classifier reported high accuracy using a random split of measurements pooled across sessions or subjects |
| **T** | You were responsible for the evaluation, or for reproducing the result before a submission |
| **A** | You noticed that measurements from the same subject and session were in both train and test, so the model could exploit subject- and session-specific signal. You re-ran with leave-one-subject-out (or train-on-session-1, test-on-session-2) and moved preprocessing fitting inside each fold |
| **R** | Accuracy fell from your within-subject figure to your cross-subject figure (state both, from your records). The reported number now matched how the system would be used on new users. You adopted the protocol for later work |

A complete answer covers: the specific mechanism of the leak, the measured size
of the difference, the change in protocol, and the engineering translation
("this is entity leakage; in a customer model I would split by customer or by
time").

### Tell me about a time you made a technical decision with incomplete information.

Project mapping: choosing a preprocessing pipeline, a data-cleaning
threshold, or a model family before all data was collected. A strong answer
covers: what you knew and did not, how you limited the risk (a baseline, a
pilot, a reversible choice), and what you would decide differently now.

### Tell me about a time your results did not reproduce.

Research mapping: an analysis that gave a different number on a new machine,
a new library version, or after a colleague re-ran it. A complete answer covers:
how you found the cause (an unrecorded parameter, random seed, data version,
library change), the fix, and the habit you adopted (run records, pinned
environments, checksums).

### Tell me about a time you had to explain a complex result to a non-expert.

Research mapping: explaining a decoding result to clinicians, a funding body,
or students. A complete answer covers: knowing what the listener needed to
decide; leading with the conclusion and its uncertainty; one concrete example
instead of a method description; checking understanding.

### Tell me about a disagreement with a collaborator or advisor.

A complete answer covers: the substance of the disagreement (technical, not
personal); how you sought evidence rather than authority; what was decided;
what you learned, including if you were wrong.

### Tell me about a project that failed.

Research mapping: an approach that did not beat the baseline, or a dataset that
turned out unusable. A complete answer covers: an accurate account; when and how
you recognised it; what you salvaged (a negative result, a tool, a cleaned
dataset); what you would do differently, such as an earlier baseline or an
earlier check of data quality.

### Why are you moving from research to industry?

A complete answer covers: a positive reason (seeing models used, working on
systems end to end, faster feedback); what you bring (evaluation rigour,
experimental design, domain depth); what you have done to build the
engineering side (this book's project, capstones, and what you learned); no
criticism of academia.

### Tell me about a time you had to learn something quickly.

A complete answer covers: the deadline and the gap; how you learned (building a
small working version first, reading source, asking); the outcome; working
through this book is a fair example if described accurately, with what you
built.
