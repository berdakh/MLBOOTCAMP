# Solutions · Chapter 9 · Error analysis & artifacts

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `tenure_bucket`

[verified] Reference code: [`code/nb08.py`](solutions.py) (`ex1`), graded in CI.

```python
TENURE_EDGES = [0, 30, 90, 180, 365, 730, np.inf]
TENURE_LABELS = ["0-30d", "30-90d", "90-180d", "180-365d", "1-2y", "2y+"]

def tenure_bucket(tenure_days):
    return pd.cut(tenure_days, bins=TENURE_EDGES, labels=TENURE_LABELS, right=False, ordered=True)
```

**Key points.** `right=False` makes intervals `[left, right)`, so 30 days is
"30-90d" (the default `right=True` puts it in "0-30d"). `np.inf` as the last
edge catches every long tenure. `pd.cut` keeps the index and leaves NaN as
NaN, and returns an ordered categorical, so slices sort young → old in
reports and plots.

---

## Exercise 2 · `worst_slices`

[verified] Reference code: [`code/nb08.py`](solutions.py) (`ex2`).

```python
def worst_slices(report, metric, min_rows, n):
    eligible = report[report["rows"] >= min_rows].dropna(subset=[metric])
    return eligible.sort_values(metric, kind="stable").head(n)
```

Filtering by size **before** ranking is the point: the lowest metric in an
unfiltered slice report is almost always a tiny slice, i.e. noise. Boolean
indexing and `sort_values` return new frames, so the input is untouched.

---

## Exercise 3 · `write_model_card`

[verified] Reference code: [`code/nb08.py`](solutions.py) (`ex3`). The essentials:

```python
lines = [f"# Model card · {meta['model_name']} {meta['version']}", "",
         "## Intended use", "...",
         "## Training data", f"- rows: {meta['training_data']['rows']}", ...,
         "## Evaluation", "| metric | value |", "|---|---|",
         *[f"| {k} | {v:.3f} |" for k, v in meta["metrics"]["validation"].items()],
         "## Decision threshold", f"Flag when the score is at least {meta['threshold']:.3f}.",
         "## Limitations", "...", "## Ethical considerations", "..."]
Path(path).write_text("\n".join(lines), encoding="utf-8")
return path
```

**Why generate it.** A card written by hand drifts from the artifact the
first time someone retrains. Generated from `metadata.json`, the numbers are
always the deployed model's. The prose sections (intended use, limitations,
ethics) still need a human, and they are the parts reviewers read most.

---

## Exercise 4 · The discount request (written)

[open-ended] Personal. A strong reply covers:

> "The model card lists pricing and discount decisions under *not intended
> for*, for three reasons. First, the score was validated for choosing whom
> to **call**, where a wrong flag costs one $30 call; a wrong discount costs
> 20% of revenue on every renewal, and the threshold, the costs and the
> evidence would all have to be redone. Second, a discount for "risky"
> customers rewards exactly the behaviour the model reads as risk (lower
> usage, complaints, payment trouble); once customers learn that, the
> features stop meaning what they did and the model degrades. Third, the
> discount changes the label itself: customers who would have left now stay,
> so future training data says the risky ones don't churn. I'd propose a
> small randomised test instead: among high-risk accounts, offer the
> discount to a random half and measure retention and revenue against the
> other half, and decide on the measured uplift, ideally with a model
> trained for *uplift*, not for churn risk."

Also worth mentioning: fairness (some plans or regions get discounts more
often than others for reasons unrelated to need) and that the scores are
not visible to the customer.

---

## Checkpoint answers

**1. Region at 0.08 vs 0.16.** The slice's size and number of positives; a
customer-bootstrap interval for the slice (and whether it overlaps the
overall one); the slice's positive rate (PR-AUC's baseline is the positive
rate, so compare lift or ROC-AUC); whether the slice has data problems (the
"NA" region parsed as missing). Only then talk about a problem, and say
"at this sample size".

**2. False negatives.** Most look like true negatives: normal usage, no
cancellation tickets, no payment failures, low scores. Around half of them
were "healthy" on every available signal [expected], so no model on these features
can rank them above the many healthy customers who stay; that error needs
new information (renewal dates, champion activity, ticket text). The rest
have weak signals and might move with better features or more data. We know
by comparing feature profiles of FN vs TP vs TN, not by guessing.

**3. `metadata.json`.** `features` / `categorical`: what input, in which order
(serving contract). `training_data` fingerprint, rows, period: trained on
what? `libraries`: can I load it safely here? `threshold`: what decision rule
ships with it? `metrics.validation`: how good, on which period?
`trained_at`, `version`, `params`, `notes`: when, which, how, why.
`model_sha256`: is this the same file that was evaluated?

**4. The hash.** Protects against corruption, truncation, accidental
replacement, and loading a model that differs from the evaluated one, and it
is checked before unpickling. It does not protect against someone who can
edit both files (they can recompute the hash); that needs signing or access
control, and it does not make a pickle from an untrusted source safe.

**5. Deploy and roll back.** `registry.promote("v2", reason=...)` writes "v2"
to a temp file and atomically replaces `CURRENT`; the service loads
`CURRENT` on (re)start or reload. `registry.rollback(reason=...)` reads the
history, finds the version live before v2 (v1) and writes it the same way.
Bundles are immutable and the pointer change is one atomic file replace, so
there is no state in which half of the model is v1 and half v2; both
events are appended to `history.jsonl` for audit.

**6. Not for.** Pricing or discount decisions (it was not validated for that,
and acting on it changes behaviour and labels); fully automated actions
without a person (errors are frequent at any useful threshold: precision
≈ 0.27); also scoring brand-new accounts with no history, or periods far from
the training data without re-validation.
