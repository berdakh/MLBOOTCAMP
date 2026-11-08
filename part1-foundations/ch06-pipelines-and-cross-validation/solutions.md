# Solutions · Chapter 6 · Pipelines & cross-validation

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `build_preprocessor`

[verified] Reference code: [`code/nb05.py`](solutions.py) (`ex1`), graded in CI.

```python
def build_preprocessor(numeric, categorical):
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])
```

**Why it works.** Each branch is itself a pipeline, so the imputer's medians
and the scaler's means are learned in `fit` (on whatever the caller passes,
here the train slice) and only *applied* in `transform`. `handle_unknown="ignore"`
maps unseen levels to an all-zero block, so the width never changes.
`add_indicator=True` keeps the information "this value was missing", which is
often predictive (here: no tickets → no resolution time).

**Common mistakes.**

* Returning a fitted transformer (`.fit(...)` inside the function). The
  function then decides *which* data it is fitted on, which is exactly the
  decision that must stay with the caller.
* Imputing with `SimpleImputer()` on the categorical columns *after*
  one-hot encoding (or not at all): `OneHotEncoder` treats NaN as its own
  category in recent sklearn, so that is fine, but a numeric NaN reaching
  `StandardScaler` → `LogisticRegression` is not.
* `OneHotEncoder(sparse_output=False)` is fine; forgetting that the default
  output can be sparse when you inspect it is the usual stumble (`.toarray()`).

---

## Exercise 2 · `rolling_cv_scores`

[verified] Reference code: [`code/nb05.py`](solutions.py) (`ex2`).

```python
def rolling_cv_scores(make_model_fn, df, n_folds):
    scores = []
    for train_idx, valid_idx in rolling_origin_folds(df, n_folds=n_folds):
        train, valid = df.iloc[train_idx], df.iloc[valid_idx]
        model = make_model_fn().fit(train[FEATURES], train[LABEL])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        scores.append(float(average_precision_score(valid[LABEL], p)))
    return scores
```

**Key points.** The folds are **positions**, so `.iloc` (not `.loc`) is
required; the grader passes a shuffled frame with a non-default index to
catch `.loc`. A *fresh* model per fold (`make_model_fn()`) avoids a classic
bug: re-fitting one object is fine in sklearn, but keeping a reference to the
"fold 1 model" and finding it silently became the fold 4 model is not.
`cross_validate(model, X, y, cv=list(rolling_origin_folds(df)))` is the
one-line equivalent.

---

## Exercise 3 · Find the leak (written)

[open-ended] Open-ended, but a complete answer finds all of these:

| # | Line | Problem | Expected size |
|---|---|---|---|
| 1 | `cust_churn_rate = groupby("customer_id")[label].mean()` | target encoding over the **whole** table including the row's own label and the validation period. For a churner with k rows, the feature is 1/k; for everyone else 0 | **huge**: in notebook 05 the same feature pushed validation PR-AUC from 0.163 to ≈ 0.58 |
| 2 | `plan_churn_rate` over the whole table | target encoding with future labels; only 5 plans, so each rate is averaged over thousands of rows and the leak is small, but it is still the future | small |
| 3 | `sessions_z` with full-table mean/std | preprocessing fitted on validation (and test) rows | tiny (notebook 05 measured ≈ 0.0001 for a similar leak) |
| 4 | `train = snapshot < 2024-04-01` | **no embargo**: the 2024-03-01 rows have labels that look into April, the validation period | small, but a systematic optimism |
| 5 | `valid = snapshot >= 2024-04-01` | the "validation" set includes the **test** period too; anything tuned on it has spent the test set | process failure, not a score change |

The largest is #1: it contains the answer. Corrected version, in words: drop
`cust_churn_rate` entirely (its honest version, labels whose window closed
before the snapshot, is almost always 0 because churn is terminal); use
`time_split` (embargo, three periods); put scaling inside a `Pipeline`
fitted on train; if a plan-level rate is wanted, compute it from training
labels only (or, better, let one-hot encoding of `plan` do the same job).
Then distrust 0.41 until you can explain it, because notebook 04's honest
baseline was far lower.

---

## Checkpoint answers

**1. Pipeline when the leak was tiny.** Because (a) you cannot know it is
tiny without measuring it each time, and other fitted steps (feature
selection, target encoding, PCA) leak much more; (b) the pipeline is also
the serving artifact: the saved object applies exactly the training-time
medians, scales and encodings, which removes a whole class of
training/serving skew.

**2. Per-customer churn rate doubles PR-AUC.** First check whether the
feature uses labels from the row itself or from after the snapshot (it
does: a full-table mean includes the current label). Then compute the
honest version: only labels whose 30-day window closed before the snapshot.
Here that version is almost always zero, because a customer who churned has
left the table; the whole apparent signal was the current label.

**3. Rolling-origin CV.** Like k-fold, but each fold trains on all data
before a cut-off and validates on the period right after it, and the cut-off
moves forward. It reproduces how the model is used (train on the past,
score the future) and keeps one customer's later rows out of its own
training fold. Shuffled k-fold is acceptable when rows really are
exchangeable: no time order in deployment, no repeated entities (or a
group k-fold for entities).

**4. 0.186 CV vs 0.163 next period.** Not necessarily broken. Check the size
of the gap against the fold-to-fold spread (≈ 0.02) and the bootstrap CI
on validation (≈ ±0.03); compare the positive rate (0.055 vs 0.052); look
for drift in key features; and remember CV is used to choose, not to
forecast. A gap much larger than the noise, or one that grows month by
month, is the signal to investigate.

**5. `handle_unknown="ignore"`.** An unseen category becomes all zeros in its
one-hot block, so the service keeps working and the model treats the row as
"none of the known levels". Monitor the rate of unseen categories per
column; a jump means an upstream change (a renamed plan, a new region) that
the model is silently ignoring.
