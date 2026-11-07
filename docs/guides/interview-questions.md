# Interview question bank

Questions by topic, each with what **a strong answer covers**. These are not
scripts to memorise. Use them for the *explain to camera* drill in the
[readiness rubric](readiness-rubric.md): answer aloud first, then compare.

The points listed are what interviewers commonly listen for. A strong answer
does not need all of them, and it should be in your own words with your own
examples.


---

<a id="data-sql-and-leakage"></a>

## Data, SQL and leakage

### Write a query giving, for each customer and snapshot date, their sessions in the 28 days before the snapshot.

A strong answer covers: asking how time is recorded (event time vs week start,
time zones, late arrivals); a join on customer with a date range condition
strictly before the snapshot; handling customers with no activity (a `LEFT
JOIN` and `COALESCE` to 0, or explicitly leaving NULL and saying why); checking
the result on one customer by hand.

### What is the difference between `ROW_NUMBER`, `RANK` and `DENSE_RANK`? When would you use each?

A strong answer covers: ties (unique numbers vs gaps vs no gaps); a concrete use
such as "latest invoice per customer" with `ROW_NUMBER() OVER (PARTITION BY
customer_id ORDER BY invoice_date DESC)` and filtering `= 1`; that tie-breaking
must be deterministic (add a second `ORDER BY` key).

### Explain point-in-time correctness.

A strong answer covers: features from data strictly before the prediction
time, labels from the window after; why "strictly" (events on the boundary may
be recorded after the decision); period boundaries (a weekly bucket is known at
its end); columns updated in place and current-state dimension tables; how you
would test it.

### What is data leakage? Give three kinds.

A strong answer covers: label (target) leakage, temporal leakage, group/entity
leakage, preprocessing leakage; one concrete example of each; how each inflates
offline metrics; how each is prevented (point-in-time features, time split
with embargo, group split, pipelines).

### Your model's validation score jumped after adding one feature. What do you check?

A strong answer covers: suspicion first; whether the feature would exist with
that value at prediction time; its timestamp and how it is populated; its
importance and its relationship with the label; retraining without it; comparing
on a later, untouched period.

### How do you validate a new data source?

A strong answer covers: a written contract (types, nulls, ranges, allowed
values, uniqueness, keys); row counts and freshness; distribution comparisons
with what you expect; defaults and sentinels (`-1`, `9999`, `NA`); documenting
each cleaning decision; running checks automatically on every load.

<a id="evaluation-and-metrics"></a>

## Evaluation and metrics

### Why is accuracy a poor metric for churn prediction? What would you use?

A strong answer covers: about 5% positives in the course's synthetic data, so
"nobody churns" is about 95% accurate; PR-AUC for ranking with its base-rate
baseline; precision, recall and alert volume at the operating threshold;
precision@k for capacity-limited teams; Brier score if probabilities drive
decisions.

### How do you know model B is really better than model A?

A strong answer covers: the same evaluation data; a paired bootstrap of the
difference; resampling the correct unit (customers, not rows); whether the
difference matters at the operating threshold, not only in AUC; a rolling-origin
or multi-period check; the cost of the more complex model.

### How would you choose the classification threshold?

A strong answer covers: start from the decision and its costs; the expected-value
rule (act when `p × value > (1 − p) × cost`); the TaskFlow example (assumed $150
vs $30 gives 1/6); that this requires calibrated probabilities, or an empirical
search on validation; capacity constraints turning it into top-k; stating the
assumptions with the threshold.

### What is calibration and when does it matter?

A strong answer covers: predicted probabilities matching observed frequencies;
reliability tables/curves; Brier score; when decisions use the probability
value (cost thresholds, expected revenue) it matters, when only ranking is used
it does not; Platt or isotonic calibration on held-out data; class weighting
and resampling distort calibration.

### What is a cluster (group) bootstrap and why use it?

A strong answer covers: resampling whole groups (customers, subjects, sites)
instead of rows; rows within a group are correlated, so the row bootstrap's
intervals are too narrow; the research analogy of across-subject variability.

### Test performance is clearly lower than validation. What happened and what do you do?

A strong answer covers: candidates (time drift between periods, leakage in
validation, overfitting to validation through many choices, different base
rates, small sample noise); checking with intervals; *not* re-tuning on test;
reporting both and explaining.

### How do you evaluate across subgroups?

A strong answer covers: slice metrics by meaningful groups; small-slice
uncertainty (flag, don't hide; add intervals); differences in base rate vs
differences in model quality; what action a slice result would trigger.

## Modelling

### When would you choose logistic regression over gradient-boosted trees?

A strong answer covers: small or mostly linear (after transforms) problems;
calibration out of the box; interpretability and stability; ease of serving
and monitoring; that the comparison should be measured, not assumed; on the
course data, notebook 06 measures whether trees earn their extra complexity.

### Why put preprocessing inside the scikit-learn `Pipeline`?

A strong answer covers: no leakage across CV folds (fitted on training folds
only); one object to save and serve, preventing training/serving skew;
`ColumnTransformer` for per-column treatment; `handle_unknown="ignore"` for
unseen categories.

### How do you handle class imbalance?

A strong answer covers: often you need nothing beyond the right metrics and
threshold; class weights or resampling change calibration; evaluate at the
operating point; more positive data if available; avoid oversampling before
splitting.

### How do you cross-validate time-ordered data?

A strong answer covers: rolling-origin (expanding window) folds; an embargo
when labels look ahead; never shuffling across time; keeping the final test
period out of all of it.

## Reproducibility and code quality

### How do you make your experiments reproducible?

A strong answer covers: code version (commit), data version (checksums,
fingerprints), configuration recorded with results, pinned environments,
seeds, run records or an experiment tracker; an example of a result that did
not reproduce and why.

### What do you test in an ML codebase?

A strong answer covers: pure functions (feature logic on tiny hand-made
inputs, including boundary dates); data contracts; that the pipeline trains and
predicts on a small sample; API contract tests; consistency between training
and serving outputs; not asserting exact model metrics, but sanity ranges or
invariants.

### How would you turn a research notebook into production code?

A strong answer covers: extract functions into modules with clear inputs and
outputs; configuration out of the code; tests for each function; a pipeline
object for preprocessing and model; versioned artifacts with metadata; keep the
notebook as a thin demonstration that imports the modules.

## APIs and serving

### Design the API for a churn model.

A strong answer covers: batch or online first; request and response schemas
with types and bounds; rejecting unknown fields; handling unknown categories;
model version in responses; status codes (422, 413, 503); a batch endpoint with
a limit; health and readiness endpoints; documentation generated from the schema.

### What is training/serving skew and how do you prevent it?

A strong answer covers: any difference in feature computation between training
and serving; examples (re-implemented preprocessing, different windows, missing
value encoding, column order, library versions); prevention (save and serve one
pipeline, shared feature code, metadata checks at load); a test that compares
scores for the same rows.

### Which errors should a client retry?

A strong answer covers: connection errors, timeouts, 429, 502/503/504; with
exponential backoff and jitter; never 4xx validation errors; a timeout on every
call; idempotence (safe for scoring; needs an idempotency key for writes);
request ids for tracing.

### How would you reduce prediction latency?

A strong answer covers: measure percentiles first and separate model time from
overhead; batching; loading the model once; avoiding per-request DataFrame
work; simpler features or model; caching; hardware and concurrency; trading off
accuracy explicitly.

<a id="deployment-and-operations"></a>

## Deployment and operations

### What is the difference between a liveness probe and a readiness probe?

A strong answer covers: liveness is "is the process alive" (restart if not);
readiness is "can it serve" (route traffic away if not); checking the model in
liveness causes restart loops during slow loads; the course's `/health` vs
`/ready`.

### Walk me through this Dockerfile.

A strong answer covers: base image choice and version pinning; installing
dependencies before copying code (layer caching); not running as root; copying
only what is needed; configuration from environment variables, not baked in;
the command that starts the server; where the model comes from (baked in vs
mounted vs downloaded at start-up) and the trade-off.

### How do you roll back a bad model?

A strong answer covers: immutable versioned artifacts; a pointer or alias for
the live version; changing the pointer and reloading; an audit log of
promotions; rollback first, investigate second; keeping the previous version
warm or quickly loadable.

### What would your CI pipeline run?

A strong answer covers: linting, unit tests, data contract tests on a sample, a
small training run, API contract tests, building the image; failing fast;
keeping it quick; separate, slower model-quality checks before promotion.

<a id="monitoring"></a>

## Monitoring

### How would you monitor this model in production?

A strong answer covers: three layers (service, data/predictions, performance);
which are available immediately and which wait for labels; concrete metrics
and alert severities; delayed labels; feedback effects from acting on
predictions; rollback.

### What is PSI? What are its limitations?

A strong answer covers: binned comparison of current vs reference
distributions using reference quantiles; the conventional 0.1 / 0.25 bands as
rules of thumb; sensitivity to bins and sample size; drift does not imply
worse performance; look at the distributions.

### Labels arrive 30 days late. How do you know if the model is degrading?

A strong answer covers: leading indicators (input drift, score distribution,
share flagged); matured-cohort evaluation when labels arrive; a holdout group
not acted on, so labels stay unbiased; alerting on both.

### Your drift alert fired. What do you do?

A strong answer covers: check for upstream data bugs first (schema change,
broken join, unit change, new category); check whether the drifted feature
matters to the model; check the score distribution and any matured
performance; decide between ignore, fix the pipeline, retrain, or roll back;
write it up.

<a id="ml-system-design"></a>

## ML system design

Practise each aloud, in 35–40 minutes, with the framework from
[Handbook 7](../handbook/07-ml-system-design.md).

### Design a system to predict which customers will churn next month.

A strong answer covers: the action and capacity; batch scoring; point-in-time
snapshot features; a time split with embargo; precision@capacity and
cost-based thresholds; the 30-day label delay; the treatment effect (contacted
customers' outcomes change; keep a holdout); monitoring and retraining.

### Design a system to flag toxic comments for human review.

A strong answer covers: review capacity as the binding constraint; precision
at the review budget; thresholds for auto-action vs review; labelling quality
and annotator disagreement; subgroup errors (identity terms); adversarial
adaptation; latency needs; feedback from reviewer decisions.

### Design a demand forecasting system for a retailer.

A strong answer covers: forecast unit and horizon; sales are not demand
(stock-outs censor demand); seasonal naive baselines; backtesting with rolling
origins; error metrics that suit intermittent demand; how forecasts drive
ordering; monitoring forecast error by segment.

### Design a recommendation system for a streaming service.

A strong answer covers: candidate generation and ranking stages; implicit
feedback; time-aware evaluation (train on the past, test on later
interactions); Recall@k and NDCG@k; cold start for new users and items;
popularity bias and feedback loops; online A/B testing.

### Design a predictive-maintenance alerting system for industrial equipment.

A strong answer covers: the decision (inspect, replace, ignore) and cost of a
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

**Mapping from research: cross-subject EEG evaluation as a leakage story.**

| | Example outline (fill in your own details and numbers) |
|---|---|
| **S** | A BCI decoder reported high accuracy using a random split of trials pooled across sessions or subjects |
| **T** | You were responsible for the evaluation, or for reproducing the result before a submission |
| **A** | You noticed that trials from the same subject and session were in both train and test, so the model could exploit subject- and session-specific signal. You re-ran with leave-one-subject-out (or train-on-session-1, test-on-session-2) and moved preprocessing fitting inside each fold |
| **R** | Accuracy fell from your within-subject figure to your cross-subject figure (state both, from your records). The reported number now matched how the system would be used on new users. You adopted the protocol for later work |

A strong answer covers: the specific mechanism of the leak, the measured size
of the difference, the change in protocol, and the engineering translation
("this is entity leakage; in a customer model I would split by customer or by
time").

### Tell me about a time you made a technical decision with incomplete information.

Research mapping: choosing a preprocessing pipeline, artifact-rejection
threshold, or model family before all data was collected. A strong answer
covers: what you knew and did not, how you limited the risk (a baseline, a
pilot, a reversible choice), and what you would decide differently now.

### Tell me about a time your results did not reproduce.

Research mapping: an analysis that gave a different number on a new machine,
a new library version, or after a colleague re-ran it. A strong answer covers:
how you found the cause (an unrecorded parameter, random seed, data version,
library change), the fix, and the habit you adopted (run records, pinned
environments, checksums).

### Tell me about a time you had to explain a complex result to a non-expert.

Research mapping: explaining a decoding result to clinicians, a funding body,
or students. A strong answer covers: knowing what the listener needed to
decide; leading with the conclusion and its uncertainty; one concrete example
instead of a method description; checking understanding.

### Tell me about a disagreement with a collaborator or advisor.

A strong answer covers: the substance of the disagreement (technical, not
personal); how you sought evidence rather than authority; what was decided;
what you learned, including if you were wrong.

### Tell me about a project that failed.

Research mapping: an approach that did not beat the baseline, or a dataset that
turned out unusable. A strong answer covers: an honest account; when and how
you recognised it; what you salvaged (a negative result, a tool, a cleaned
dataset); what you would do differently, such as an earlier baseline or an
earlier check of data quality.

### Why are you moving from research to industry?

A strong answer covers: a positive reason (seeing models used, working on
systems end to end, faster feedback); what you bring (evaluation rigour,
experimental design, domain depth); what you have done to build the
engineering side (this course's project, capstones, and what you learned); no
criticism of academia.

### Tell me about a time you had to learn something quickly.

A strong answer covers: the deadline and the gap; how you learned (building a
small working version first, reading source, asking); the outcome; this
bootcamp is a fair example if described honestly, with what you built.
