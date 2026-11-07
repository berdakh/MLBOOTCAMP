# Capstone 5 · Toxic-comment triage for human review on Civil Comments

**Recommended order: 5th** · MVP ≈ 5 days at 3 h/day · CPU core, GPU optional

A public-data demonstration inspired by trust-and-safety problems on social
platforms. It uses the public Civil Comments dataset; it is not work for, or
endorsed by, Meta, Google, Jigsaw or any platform, and it describes no real
moderation system. The model **ranks comments for human review**; it does not
remove anything on its own.


---

## Business question

*Given a fixed number of human moderator reviews per day, which comments
should go to the review queue so that reviewers see as many genuinely toxic
comments as possible — without the model systematically over-flagging
comments that merely mention an identity group?*

**Who acts on it:** a moderation team lead who sets queue capacity and
policy, and the reviewers who make the final decision. The model's job is
**prioritisation**. Its errors have different costs: a missed toxic comment
(false negative) harms readers; a wrongly flagged comment (false positive)
costs reviewer time and, if it falls disproportionately on comments about
particular identities, silences the people being discussed.

## Why this project

* **PR-AUC and threshold selection** under an operational constraint (queue
  capacity), building directly on the bootcamp's thresholds and calibration
  day.
* **Subgroup error analysis** with identity labels: FPR/FNR per group and the
  bias AUCs from the dataset's own paper — the core of responsible-ML
  interview questions.
* **Noisy, fractional labels** from multiple annotators, and what that means
  for "ground truth".
* **A compact pretrained transformer vs a TF-IDF baseline**, with an honest
  cost/benefit comparison (latency, CPU vs GPU, accuracy).
* **Human-in-the-loop design**: bands, capacity, escalation — product sense
  that Applied Scientist loops test.

## Data

| Fact | Status |
|---|---|
| Hugging Face: <https://huggingface.co/datasets/google/civil_comments> | **verified from an official page snippet** (dataset card) |
| Licence on the HF dataset card: **CC0** (both the dataset and the comment text) | **verified from an official page snippet** |
| Each comment has 7 **fractional** labels: `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack`, `sexual_explicit` | **verified from an official page snippet** (TFDS catalogue) |
| HF default config splits of roughly 1.80M train / 97K validation / 97K test | **to verify** — count them |
| **The HF default config does not include the identity columns** | **verified from an official page snippet** |
| TFDS catalogue: <https://www.tensorflow.org/datasets/catalog/civil_comments>; config **`CivilCommentsIdentities`** includes identity labels for about 405,130 train and 21,577 test comments | **verified from an official page snippet** |
| Kaggle competition "Jigsaw Unintended Bias in Toxicity Classification" provides `train.csv` / `all_data.csv` with identity columns for an annotated subset; downloading requires accepting the competition rules | **to verify** — read the rules page; they may restrict use |
| The **WILDS** benchmark has a CivilComments dataset with identity metadata | **verified** that it exists; its splits and licence are **to verify** |
| Labels are the fraction of annotators who applied the label; the common convention is **toxic if `toxicity ≥ 0.5`** | **to verify** in the dataset documentation |

**Which source for identity labels?** The MVP needs identity labels for the
subgroup analysis, and the HF default config lacks them. Pick one, record it
in the data card, and do not mix sources (split definitions differ):

1. **TFDS `civil_comments/CivilCommentsIdentities`** — official, versioned,
   identity labels for the annotated subset (recommended).
2. **WILDS CivilComments** — convenient if you want its predefined splits and
   group metadata; verify its licence and split construction.
3. **Kaggle `all_data.csv`** — richest metadata, but you must accept the
   competition rules; check whether they permit your use.

**Licence gist.** The HF card states CC0 — dedication to the public domain,
no attribution legally required. Check the card yourself and check the terms
of whichever mirror you use (TFDS, WILDS, Kaggle); **cite the paper anyway**.
The comments are real user text that can be offensive; do not republish
examples gratuitously in your README.

**Citation** (**verified from an official page snippet**):

> D. Borkan, L. Dixon, J. Sorensen, N. Thain, L. Vasserman. "Nuanced Metrics
> for Measuring Unintended Bias with Real Data for Text Classification".
> 2019. arXiv:1903.04561

**Known gotchas** (**to verify**):

* **Heavy class imbalance**: toxic comments (≥ 0.5) are a small minority —
  measure the rate in each split. Accuracy and ROC-AUC look good for useless
  models; use PR-AUC.
* **Fractional labels**: 0.5 is a convention; comments near 0.5 are ones
  annotators disagreed on. Keep the raw fraction.
* **Identity labels are themselves annotator fractions** and only exist on a
  subset; many subgroups are small.
* **Duplicate and near-duplicate comments** (reposts, templates) can cross
  splits.
* The full training set is large; TF-IDF on all of it needs several GB of
  RAM. Start with a documented, stratified subsample.

## Problem framing

* **Unit of prediction:** one comment (text only).
* **Output:** a score in [0, 1] interpreted as "priority for review", and a
  band: `auto_approve` / `human_review` (and, as an optional policy,
  `priority_review` for the highest scores — never automatic removal in this
  project).
* **Label:** `toxic = toxicity ≥ 0.5` (binary), and keep `toxicity` as a
  fractional target for an extension (regression or soft-label training).
* **Subgroup membership:** identity column ≥ 0.5 (same convention), evaluated
  only on comments that have identity annotations.
* **Horizon:** none (instant classification), but the **operating point** is
  set by capacity: if reviewers can handle *C* comments per day out of *V*
  arriving, the queue takes the top *C / V* fraction of scores. *C* and *V*
  are **assumptions you choose and state** (e.g. review 1%, 2%, 5% of
  traffic); do not present them as any platform's real numbers.

## Baselines

1. **Keyword list:** flag comments containing terms from a small public
   profanity/slur list — shows the identity-term over-flagging problem very
   concretely. (Choose a list with a clear licence, or write a tiny one.)
2. **TF-IDF (word 1–2-grams + char 3–5-grams) + logistic regression**
   (`class_weight="balanced"` or tuned), trained on a stratified subsample
   (start ~200k comments; measure RAM and time; scale up if it fits). **This is
   the main baseline**, CPU-only, minutes to train.

## Improvements

* **Compact pretrained classifier:** a distilled transformer (e.g. a
  DistilBERT-, MiniLM- or DistilRoBERTa-sized model from the HF Hub — verify
  each checkpoint's licence) fine-tuned on a **subsample** (e.g. 20k–50k
  comments, stratified, documented seed), max length ~128–256 tokens.
  * **CPU feasibility:** inference on the validation/test sets and
    fine-tuning on a *small* subsample are possible on CPU but slow; measure
    and report examples/second.
  * **Full fine-tuning on the full training set is GPU-optional** (Colab GPU
    if available) and is an extension, not part of the MVP.
  * Alternatively, use a frozen encoder to produce embeddings once, then a
    logistic regression head — cheap on CPU after the one-off embedding pass.
* **Calibration** (Platt or isotonic on validation) so that scores mean
  something to reviewers and thresholds are stable across retrains.
* **Bias mitigation (extension):** reweighting or augmenting non-toxic
  identity-mention comments; evaluate whether subgroup FPR drops without
  hurting overall PR-AUC.

## Evaluation protocol

**Splits.** Use the **standard splits of your chosen source** (TFDS / WILDS /
Kaggle). Use train for fitting, **validation for model selection, calibration
and thresholds**, and the test split **once**. Before anything, deduplicate on
normalised text across splits and record how many duplicates crossed (keep
the official splits, but remove cross-split duplicates from train).

**Overall metrics.**

* **PR-AUC (average precision)** — primary for an imbalanced problem.
* **ROC-AUC** — secondary (comparable with the literature).
* **Precision and recall at fixed thresholds**, and at **capacity-defined
  thresholds**: for review budgets of e.g. 1%, 2% and 5% of comments, report
  the threshold, precision (share of the queue that is toxic — reviewer
  efficiency) and recall (share of toxic comments that reach a reviewer).
* **Calibration:** reliability curve and Brier score on validation/test.
* Uncertainty: bootstrap 95% CIs over comments.

**Subgroup metrics** (on the identity-annotated subset; report subgroup size
next to every number and skip or flag groups below a minimum count, e.g. 100
comments or 20 toxic):

* **FPR and FNR per subgroup at the chosen operating point**, compared with the
  overall FPR/FNR, with bootstrap CIs.
* The paper's bias AUCs: **Subgroup AUC** (within the subgroup), **BPSN AUC**
  (background-positive, subgroup-negative: are non-toxic comments mentioning
  the group scored higher than toxic comments not mentioning it?) and **BNSP
  AUC** (the reverse). A low BPSN AUC is the "identity term over-flagging"
  signature.
* A table of the most over-flagged non-toxic comments per subgroup, *described*
  rather than quoted where text is offensive.

**Human-review queue analysis.** For each budget, a short table: comments per
day assumed, queue size, expected toxic in queue (precision × size), expected
missed toxic (from recall), and the subgroup share of the queue vs the
subgroup share of truly toxic comments.

## Leakage traps

* **Threshold or calibration fitted on the test split.**
* **Duplicate comments across splits** (reposts) inflating scores.
* **Identity labels as model features.** They are human annotations not
  available at inference time; they are for evaluation only.
* **Other annotation columns as features** (e.g. `severe_toxicity`,
  `insult`, annotator counts, reaction counts in Kaggle metadata) — they are
  labels or post-hoc signals, not inputs.
* **TF-IDF vocabulary or IDF fitted on validation/test text.** Fit the
  vectoriser inside the training pipeline.
* **Subsample chosen using test statistics**, or a subsample that is not
  stratified, then compared to a model trained on a different one.

## MVP (≈ 5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repo from template; read the HF card and the terms of your identity-label source; `make data` (TFDS or WILDS) with recorded versions; counts per split, toxic rate, identity-annotated subset size; dedup across splits; documented stratified subsample |
| 2 | Keyword and TF-IDF + LR baselines in a `Pipeline`; PR-AUC, ROC-AUC, precision/recall at thresholds and at capacity budgets, all unit-tested on tiny hand-made cases |
| 3 | Subgroup metrics (FPR/FNR, Subgroup/BPSN/BNSP AUC) with minimum-count rules and bootstrap CIs; calibration on validation; choose the operating point on validation |
| 4 | Compact pretrained model: frozen-embedding + LR on CPU, *or* fine-tuning on a small subsample (GPU if available); same evaluation; latency per comment measured; final test run once |
| 5 | Queue analysis; error analysis; API with bands and model version; README, model card (intended use: triage for human review only), data card |

**Extensions** (optional):

* Full fine-tuning on the full training set — **GPU only**.
* Soft-label training on the `toxicity` fraction; compare with binary.
* Bias mitigation and a before/after subgroup table.
* Per-subgroup calibration analysis.
* Active-learning loop: which comments should reviewers label next?

## API design

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model and tokenizer/vectoriser loaded; returns `model_version`, `threshold`, `review_budget` |
| POST | `/score` | 1–64 comments → scores and bands |

Request:

```json
{"comments": [{"id": "c-001", "text": "Example comment text goes here."}]}
```

Response:

```json
{
  "model_version": "v2",
  "threshold": 0.37,
  "review_budget": 0.02,
  "results": [
    {"id": "c-001", "score": 0.12, "band": "auto_approve", "truncated": false}
  ]
}
```

(Scores and thresholds are illustrative placeholders, not results.)

**Validation rules:** 1–64 comments per request; `id` required and unique
within the request; `text` a non-empty string after stripping whitespace,
at most e.g. 5,000 characters (longer → 422, or truncated with
`"truncated": true` — choose one and test it); valid UTF-8; unknown fields
rejected. The API never returns a "remove" decision. It must not log raw
comment text by default (log ids, scores and lengths).

**Precomputed vs online:** the model, tokenizer/vectoriser, calibration map
and threshold are loaded once at startup from the bundle. Scoring is
online per request (batch for efficiency). The threshold is derived offline
from the validation set and the chosen review budget, stored in the model
metadata, and returned in each response so reviewers' dashboards know which
operating point produced a band.

## Limitations

* **Label noise and annotator disagreement:** labels are fractions of a small
  number of crowd raters; comments near 0.5 are genuinely contested, and the
  0.5 cut-off is a convention, not truth.
* **Identity-term over-flagging:** models learn that identity words co-occur
  with toxicity and flag benign mentions; your subgroup table will show where.
* **Identity labels are partial** (an annotated subset) and also noisy;
  small subgroups have wide CIs and some cannot be evaluated at all.
* **Domain and time:** comments come from one commenting platform in a
  specific period (read the dataset card for provenance); language, slang and norms drift.
* **Toxicity ≠ policy violation:** a platform's policy is more specific and
  context-dependent than a single toxicity score.
* **Offensive content:** the dataset contains hateful text; handle examples
  with care in your README and presentations.
* Capacity numbers are **assumptions** for illustration.

## Interview questions

1. **Why PR-AUC rather than accuracy or ROC-AUC?** Strong answer: low
   prevalence; precision is what reviewers feel; ROC-AUC can be high while the
   top of the ranking is poor; numbers from your run.
2. **How did you pick the threshold?** Covers: review capacity → top-q
   fraction on validation; precision/recall at that point; recalibrate when
   traffic or prevalence changes.
3. **What is BPSN AUC and what did it show?** Covers: definition; low BPSN =
   benign identity mentions scored above toxic non-mentions; your subgroups and
   counts.
4. **Labels are fractions. How did you use them, and what are the risks?**
   Covers: 0.5 convention, disagreement near the boundary, soft labels,
   evaluation on confident examples as a sensitivity check.
5. **Your transformer beats TF-IDF by a small PR-AUC margin. Ship it?**
   Covers: CI overlap, latency and cost per comment, subgroup errors,
   maintenance; decision depends on capacity economics.
6. **How would you reduce false positives on identity mentions?** Covers:
   data augmentation/reweighting, counterfactual tests (swap identity terms),
   evaluation before/after, trade-offs.
7. **Design the moderation pipeline end to end.** Covers: score → bands →
   human review → decisions logged as new labels; appeals; priority queue;
   latency; never auto-remove from a single model without policy approval.
8. **How do you monitor this after launch?** Covers: score distribution drift,
   queue size, reviewer agreement rate with the model, delayed labels from
   reviews (biased: only reviewed items get labels), subgroup FPR over time.
9. **Why must identity labels not be features?** Covers: unavailable at
   inference; would encode the protected attribute directly; evaluation-only.
10. **How did you run this on a CPU?** Covers: TF-IDF baseline, subsampling,
    frozen embeddings, measured throughput; full fine-tuning on GPU as an
    optional extension.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `modtriage`)
- [ ] `DATA_CARD.md`: HF card licence (CC0) quoted, identity-label source and its terms, citation, split counts, subsample seed, dedup counts
- [ ] Tests: vectoriser fitted only on train; no cross-split duplicates in train; identity columns absent from model inputs
- [ ] Keyword + TF-IDF/LR baselines and a compact pretrained model on the same splits
- [ ] Results table: PR-AUC, ROC-AUC, precision/recall at 1%/2%/5% review budgets, with CIs — **TBD — fill from your run**
- [ ] Subgroup table: n, FPR, FNR, Subgroup/BPSN/BNSP AUC — **TBD — fill from your run**
- [ ] Queue analysis table under stated capacity assumptions
- [ ] API with bands, threshold and model version in responses; no raw-text logging; validation tests
- [ ] `MODEL_CARD.md`: intended use = triage for human review; out of scope = automatic removal
- [ ] CI green on a few synthetic, non-offensive example comments

## Stretch: connect to the bootcamp's churnlab patterns

* **Registry:** store the threshold and review budget **inside** each model
  version's metadata, so promoting or rolling back a model also restores the
  operating point that was validated with it —
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py).
* **Monitoring:** track the daily share of comments routed to review (a jump
  means drift or an attack), PSI on the score distribution, and — as reviewer
  decisions arrive — precision of the queue and subgroup FPR, with alert
  rules as in
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py).
  Note the selection bias: only queued comments get human labels.
