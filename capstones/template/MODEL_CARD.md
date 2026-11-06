# Model card — TODO model name, version

Keep this short and true. Every number comes from `reports/metrics.json` of a
named run. "TBD" is an acceptable entry; an invented number is not.

## Model details

- **Version / date / commit:** TODO
- **Type:** TODO (e.g. gradient-boosted trees on lag features; 1D-CNN on 30-cycle windows)
- **Inputs:** TODO (the feature list, which is also the API contract)
- **Output:** TODO (a number in which units / a score in [0, 1] / a ranked list of k items)
- **Training data:** TODO (dataset, date range or entities, see DATA_CARD.md)
- **Author / contact:** TODO

## Intended use

- **Intended:** a portfolio demonstration on public data; TODO describe the
  decision it would support in a real setting.
- **Out of scope:** TODO (e.g. real-time grid dispatch; safety-critical
  maintenance decisions; automatic removal of content without human review;
  any commercial use the data licence forbids).

## Evaluation

| Split | Metric | Value | 95% CI | Baseline value |
|---|---|---|---|---|
| test | TODO | TBD — fill from your run | TBD | TBD |

- **Protocol:** TODO (split, why, how often the test split was touched)
- **Slices / subgroups:** TODO (peak hours, engine fault mode, intermittent
  items, cold-start users, identity subgroups) — TBD — fill from your run.

## Decision threshold / operating point (if any)

TODO: chosen threshold, how it was chosen (validation split, capacity, cost),
and what precision/recall/volume it implies.

## Limitations and risks

TODO: known failure modes, data the model never saw, how it could cause harm
if misused, and what monitoring would catch it.

## Monitoring plan

TODO: input drift (e.g. PSI per feature), prediction drift, delayed-label
performance, alert thresholds, and the rollback procedure.
