# Glossary

Terms used in the course, with the meaning they have here. Where a term has a
research counterpart, it is given in brackets.

| Term | Meaning |
|---|---|
| **A/B test** | Splitting live traffic between two versions for long enough to measure the effect on an outcome. The causal test of whether a model helps |
| **App factory** | A function (`create_app(settings)`) that builds a new web app from configuration, instead of creating one at import time. Makes testing and configuration easy |
| **Artifact** | Any file a run produces and a later step uses: a feature table, a model bundle, a report |
| **Artifact directory** | Where this course writes artifacts. Chosen by `BOOTCAMP_ARTIFACTS`, then Drive (`USE_DRIVE`), then `<repo>/artifacts` |
| **Backoff (exponential)** | Waiting longer after each failed retry (0.2 s, 0.4 s, 0.8 s, …) so a struggling server is not overwhelmed. Usually with random jitter |
| **Baseline** | The simplest reasonable model or rule, used to judge whether complexity earns its place. A dummy (base-rate) baseline and a simple model are both worth reporting |
| **Batch scoring** | Scoring many rows on a schedule and writing results somewhere, rather than answering requests |
| **Bootstrap** | Estimating uncertainty by resampling the data with replacement and recomputing a statistic many times |
| **Brier score** | Mean squared difference between predicted probability and outcome. Lower is better; measures probability quality |
| **Bundle** | In this course, an immutable directory with `model.joblib` and `metadata.json` (features, versions, threshold, data fingerprint, metrics, hash) |
| **Calibration** | Agreement between predicted probabilities and observed frequencies. Needed when decisions use probability values |
| **Canary release** | Sending a small share of traffic to a new version and increasing it if metrics hold |
| **Catch-up cell** | The second code cell of every notebook; calls `churnlab.bootstrap.ensure_*` to rebuild earlier outputs |
| **Checksum** | A hash (here SHA-256) of a file's bytes, used to confirm a file is exactly the expected one |
| **Cluster (group) bootstrap** | A bootstrap that resamples whole groups (customers) rather than rows, because rows within a group are correlated (across-subject vs across-trial variability) |
| **Concept drift** | A change in the relationship between inputs and the outcome. Can hurt performance with no visible input drift |
| **Contract (API)** | The schema of requests and responses and the meaning of errors: what the service promises callers |
| **Contract (data)** | Written rules a table must satisfy: types, nulls, ranges, allowed values, uniqueness |
| **`CURRENT` pointer** | A one-line file in the registry naming the live model version. Changing it deploys or rolls back |
| **Data drift** | A change in the distribution of inputs compared with a reference period |
| **DuckDB** | An in-process SQL engine that queries DataFrames and files directly, without a server |
| **Embargo (gap)** | Rows dropped near a split boundary because their label window overlaps the next period. Here, 30 days (the label horizon) |
| **Entity (group) leakage** | The same entity (customer, subject, device) in both training and evaluation, so the score partly measures memorisation (within-subject vs cross-subject evaluation) |
| **Error budget** | The amount of unreliability an SLO allows (100% minus the target) over its window |
| **Expected value (decision)** | For one action: probability-weighted gain minus cost. Used to derive thresholds from stated costs |
| **Feature store** | A system that stores feature definitions and values for both training (historical, point-in-time) and serving (latest). Not used in this course; the idea is shared feature code |
| **Fingerprint** | A short hash of a DataFrame's contents, stored with a model to record exactly which training data it saw |
| **Idempotent** | An operation that has the same effect whether performed once or several times. Scoring is; "create a task" is not unless designed to be |
| **Label leakage** | A feature that encodes the outcome, directly or by proxy (`status = 'cancelled'`) |
| **Lift@k%** | Precision in the top k% of scores divided by the base rate |
| **Liveness probe** | A check that the process is alive. If it fails, the process is restarted. Should not depend on the model |
| **Model card** | A short document describing a model: intended use, data, metrics with intervals and split, limitations, versions |
| **Paired bootstrap** | Bootstrapping the *difference* between two models on the same resamples, to test whether one is better |
| **Point-in-time correctness** | Features computed only from data available before the prediction time; labels only from after |
| **PR-AUC (average precision)** | Area under the precision-recall curve. Its no-skill baseline equals the positive rate |
| **Precision@k** | Share of true positives among the k highest-scored rows. Fits capacity-limited actions |
| **PSI (Population Stability Index)** | A drift measure comparing binned distributions of a reference and a current sample. Rule-of-thumb bands 0.1 and 0.25 |
| **p95 / p99 latency** | The response time below which 95% / 99% of requests complete |
| **Readiness probe** | A check that the service can do its job (model loaded). If it fails, traffic is routed elsewhere |
| **Registry** | A store of model versions with a pointer to the live one, and a history of promotions and rollbacks |
| **Rollback** | Returning to the previous model version. Here: rewriting `CURRENT` and reloading |
| **Rolling-origin CV** | Time-series cross-validation: train on everything before a cut-off, validate on the next block, move the cut-off forward |
| **Run record** | A small file saved with each experiment: code version, data fingerprint, config, seed, metrics with split |
| **Sentinel value** | A special value meaning "unknown" or "not applicable" (here `sessions = -1`). Must be converted to missing, not treated as data |
| **Shadow deployment** | Running a new model on live inputs without using its outputs, to compare behaviour safely |
| **Skew (training/serving)** | Any difference between feature computation in training and in serving. Silent and harmful |
| **SLO (service level objective)** | A target for a service metric over a window, such as a success rate or latency percentile |
| **Slice analysis** | Computing metrics separately for subgroups to find where the model fails |
| **Snapshot** | A (date, entity) row in the training table, with features from before the date and a label from after |
| **STAR** | Situation, Task, Action, Result: a structure for behavioural interview answers |
| **Structured logging** | Logs as one JSON object per line with named fields, so they can be filtered and aggregated |
| **Temporal leakage** | Training on data from after the evaluation period, typically via a random split of time-ordered data |
| **Threshold** | The score above which the model's output triggers an action |
| **Time split** | Training on earlier periods, validating and testing on later ones |
| **Unseen category** | A categorical value not present in training. Handled here by `handle_unknown="ignore"` and counted by monitoring |
| **`USE_DRIVE`** | The setup-cell switch that stores artifacts in Google Drive so they survive Colab disconnects |
