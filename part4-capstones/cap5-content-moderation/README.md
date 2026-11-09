# Capstone 5 · Content moderation triage (Civil Comments)

**Recommended order: 5th** · MVP ≈ 5 days at 3 h/day · CPU core, GPU optional

**Guided solution:** [walkthrough.ipynb](walkthrough.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap5-content-moderation/walkthrough.ipynb)
— a complete worked solution of the MVP, step by step: attempt each step yourself, then compare with
ours. Its CPU core runs on the real data in about ten minutes; DistilBERT fine-tuning is an optional,
guarded GPU extension.

A public-data demonstration inspired by trust-and-safety problems on social platforms. It uses the
public Civil Comments dataset; it is not work for, or endorsed by, Meta, Google, Jigsaw or any
platform, and it describes no real moderation system. The model **ranks comments for human review**;
it does not remove anything on its own.

---

## Business question

*Given a fixed number of human moderator reviews per day, which comments should go to the review
queue so that reviewers see as many genuinely toxic comments as possible, without the model
systematically over-flagging comments that merely mention an identity group?*

**Who acts on it:** a moderation team lead who sets queue capacity and policy, and the reviewers who
make the final decision. The model's job is **prioritisation**. Its errors have different costs: a
missed toxic comment (false negative) harms readers; a wrongly flagged comment (false positive) costs
reviewer time and, if such errors fall disproportionately on comments about particular identities,
silences the people being discussed.

## Why this project

* **PR-AUC and threshold selection** under an operational constraint (queue capacity), building
  directly on [Chapter 8](../../part1-foundations/ch08-thresholds-and-calibration/README.md)
  (thresholds and calibration).
* **Subgroup error analysis** with identity labels: FPR/FNR per group and the bias AUCs from the
  dataset's own paper, as in [Chapter 53](../../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md).
* **Noisy, fractional labels** from several annotators, and what they mean for "ground truth".
* **A compact pretrained transformer vs a TF-IDF baseline**, with an honest comparison of cost and
  benefit (latency, CPU vs GPU, accuracy).
* **Human-in-the-loop design:** bands, capacity and escalation.

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
| The **WILDS** benchmark has a CivilComments dataset with identity metadata | **verified** that it exists; its licence is **to verify** |
| WILDS CivilComments on the HF Hub (`pietrolesci/civilcomments-wilds`, revision `c227534`): 448,000 comments with toxicity and identity annotations; splits 269,038 train / 45,180 val / 133,782 test, toxic rate ≈ 0.11 in each | **verified by the walkthrough run** |
| Labels are the fraction of annotators who applied the label; the common convention is **toxic if `toxicity ≥ 0.5`** | **to verify** in the dataset documentation (the walkthrough uses this convention, as do WILDS and the Jigsaw competition) |

**Which source for identity labels?** The MVP needs identity labels for the subgroup analysis, and the
HF default config lacks them. Pick one, record it in the data card, and do not mix sources (split
definitions differ):

1. **TFDS `civil_comments/CivilCommentsIdentities`**: official, versioned, identity labels for the
   annotated subset.
2. **WILDS CivilComments**: convenient if you want its predefined splits and group metadata; verify
   its licence and split construction. **The walkthrough uses this option** through a pinned Hugging
   Face mirror (the same source as Chapter 53), because it downloads in one file without TensorFlow.
3. **Kaggle `all_data.csv`**: the richest metadata, but you must accept the competition rules; check
   whether they permit your use.

**Licence gist.** The HF card states CC0: dedication to the public domain, no attribution legally
required. Check the card yourself and check the terms of whichever mirror you use (TFDS, WILDS,
Kaggle); **cite the paper anyway**. The comments are real user text that can be offensive; do not
republish examples in your README (the walkthrough prints ids, lengths and scores only).

**Citation** (**verified from an official page snippet**):

> D. Borkan, L. Dixon, J. Sorensen, N. Thain, L. Vasserman. "Nuanced Metrics for Measuring Unintended
> Bias with Real Data for Text Classification". 2019. arXiv:1903.04561

**Known gotchas** (counts from the walkthrough's run on the WILDS release; **to verify** for other
sources):

* **Heavy class imbalance:** about 11% of comments are toxic (≥ 0.5) in every split. Accuracy and
  ROC-AUC look good for weak models; use PR-AUC.
* **Fractional labels:** 0.5 is a convention; comments near 0.5 are ones annotators disagreed on. Keep
  the raw fraction.
* **Identity labels are themselves annotator fractions**, and many subgroups are small (a few hundred
  test comments for some groups).
* **Duplicate and near-duplicate comments** (reposts, templates) can cross splits: about 1,400 training
  comments repeat a validation or test text after normalisation.
* The full training set is large; TF-IDF with character n-grams on all of it needs several GB of RAM.
  Start with a documented, stratified subsample (the walkthrough uses 100,000 comments).

## Problem framing

* **Unit of prediction:** one comment (text only).
* **Output:** a score in [0, 1] interpreted as "priority for review", and a band: `auto_approve` /
  `human_review` (and, as an optional policy, `priority_review` for the highest scores; never
  automatic removal in this project).
* **Label:** `toxic = toxicity ≥ 0.5` (binary); keep `toxicity` as a fractional target for an
  extension (regression or soft-label training).
* **Subgroup membership:** identity column ≥ 0.5 (the same convention), evaluated only on comments
  that have identity annotations.
* **Horizon:** none (instant classification), but the **operating point** is set by capacity: if
  reviewers can handle *C* comments per day out of *V* arriving, the queue takes the top *C / V*
  fraction of scores. *C* and *V* are **assumptions you choose and state** (for example, review 1%, 2%
  or 5% of traffic); do not present them as any platform's real numbers.

## Baselines

1. **Keyword list:** flag comments containing terms from a small public profanity list, or a tiny list
   you write yourself (the walkthrough uses 20 mild insults). It shows concretely how far simple
   matching is from the task.
2. **TF-IDF (word 1–2-grams + character 3–5-grams) + logistic regression** (`class_weight="balanced"`
   or tuned), trained on a stratified subsample (start with about 200,000 comments; measure RAM and
   time; scale up if it fits). **This is the main baseline**: CPU only, minutes to train.

## Improvements

* **Compact pretrained classifier:** a distilled transformer (for example a DistilBERT-, MiniLM- or
  DistilRoBERTa-sized model from the HF Hub; verify each checkpoint's licence) fine-tuned on a
  **subsample** (for example 20,000–50,000 comments, stratified, with a documented seed), maximum
  length about 128–256 tokens.
  * **CPU feasibility:** inference on the validation and test sets and fine-tuning on a *small*
    subsample are possible on a CPU but slow; measure and report examples per second.
  * **Full fine-tuning on the full training set is GPU-optional** (a Colab GPU if available) and is an
    extension, not part of the MVP.
  * Alternatively, use a frozen encoder to produce embeddings once, then a logistic regression head:
    cheap on a CPU after the one-off embedding pass (the walkthrough's CPU core does this with
    `all-MiniLM-L6-v2`).
* **Calibration** (Platt or isotonic, on validation) so that scores mean something to reviewers and
  thresholds are stable across retrains.
* **Bias mitigation (extension):** reweighting or augmenting non-toxic identity-mention comments;
  check whether subgroup FPR falls without hurting overall PR-AUC.

## Evaluation protocol

**Splits.** Use the **standard splits of your chosen source** (TFDS, WILDS or Kaggle). Use train for
fitting, **validation for model selection, calibration and thresholds**, and the test split **once**.
Before anything else, de-duplicate on normalised text across splits and record how many duplicates
crossed (keep the official splits, but remove cross-split duplicates from train).

**Overall metrics.**

* **PR-AUC (average precision):** primary for an imbalanced problem.
* **ROC-AUC:** secondary (comparable with the literature).
* **Precision and recall at fixed thresholds**, and at **capacity-defined thresholds**: for review
  budgets of, for example, 1%, 2% and 5% of comments, report the threshold, precision (the share of
  the queue that is toxic: reviewer efficiency) and recall (the share of toxic comments that reach a
  reviewer).
* **Calibration:** reliability curve and Brier score on validation and test.
* Uncertainty: bootstrap 95% CIs over comments.

**Subgroup metrics** (on the identity-annotated subset; report the subgroup size next to every number
and skip or flag groups below a minimum count, for example 100 comments or 20 toxic comments):

* **FPR and FNR per subgroup at the chosen operating point**, compared with the overall FPR and FNR,
  with bootstrap CIs.
* The paper's bias AUCs: **Subgroup AUC** (within the subgroup), **BPSN AUC** (background-positive,
  subgroup-negative: are non-toxic comments mentioning the group scored higher than toxic comments not
  mentioning it?) and **BNSP AUC** (the reverse). A low BPSN AUC is the signature of identity-term
  over-flagging.
* A table of the most over-flagged non-toxic comments per subgroup, *described* (id, length, score,
  raw toxicity fraction) rather than quoted.

**Human-review queue analysis.** For each budget, a short table: comments per day assumed, queue size,
expected toxic comments in the queue (precision × size), expected missed toxic comments (from recall),
and the subgroup's share of the queue against its share of truly toxic comments.

## Results

The tables below are generated by Step 10 of the [walkthrough](walkthrough.ipynb) from its run on the
real data; they were not typed by hand. Your own run should reproduce them up to small differences
from library versions and thread counts.

Data: CivilComments-WILDS (pietrolesci/civilcomments-wilds@c227534); official test split, 133,782 comments, toxic rate 0.114. Model: TF-IDF (word 1–2 + char 3–5) + logistic regression on a stratified 100,000-comment training subsample, Platt-calibrated and thresholded on validation. 95% CIs: bootstrap over comments (200).

| Metric | Value (95% CI) |
|---|---|
| PR-AUC | 0.644 (0.636–0.652) |
| ROC-AUC | 0.911 (0.909–0.913) |
| Brier (calibrated) | 0.0630 (0.0622–0.0642) |

| Review budget | Threshold | Queue share (test) | Precision (95% CI) | Recall (95% CI) |
|---|---|---|---|---|
| 1% | 0.968 | 0.010 | 0.965 (0.956–0.975) | 0.084 (0.079–0.089) |
| 2% | 0.893 | 0.019 | 0.918 (0.907–0.930) | 0.154 (0.148–0.159) |
| 5% | 0.618 | 0.050 | 0.779 (0.767–0.788) | 0.341 (0.334–0.348) |
| fixed 0.5 | 0.500 | 0.067 | 0.715 | 0.420 |

Head-to-head on a random 10,000-comment test subsample (same comments for every model):

| Model | Training comments | PR-AUC (95% CI) | ROC-AUC | Δ PR-AUC vs TF-IDF full (95% CI) |
|---|---|---|---|---|
| tfidf_full | 100,000 | 0.657 (0.631–0.684) | 0.913 | +0.000 (+0.000 to +0.000) |
| tfidf_small | 8,000 | 0.494 (0.465–0.526) | 0.850 | -0.163 (-0.182 to -0.142) |
| minilm_probe | 8,000 | 0.407 (0.381–0.440) | 0.813 | -0.250 (-0.279 to -0.216) |
| keyword | 0 | 0.268 (0.244–0.292) | 0.630 | -0.389 (-0.413 to -0.362) |

Subgroups at the 5% budget (threshold 0.618); overall FPR 0.012, FNR 0.659. Groups with fewer than 100 comments or 20 toxic comments are not interpreted.

| Subgroup | n | Toxic | FPR (95% CI) | FNR (95% CI) | Subgroup AUC | BPSN AUC | BNSP AUC |
|---|---|---|---|---|---|---|---|
| black | 4,872 | 1,537 | 0.102 (0.093–0.111) | 0.536 (0.514–0.562) | 0.795 | 0.685 | 0.957 |
| white | 7,969 | 2,246 | 0.085 (0.078–0.093) | 0.533 (0.517–0.558) | 0.814 | 0.693 | 0.963 |
| homosexual_gay_or_lesbian | 3,810 | 1,095 | 0.067 (0.059–0.077) | 0.650 (0.619–0.681) | 0.794 | 0.764 | 0.930 |
| muslim | 6,982 | 1,627 | 0.047 (0.041–0.053) | 0.635 (0.612–0.656) | 0.818 | 0.773 | 0.939 |
| transgender | 747 | 154 | 0.040 (0.024–0.056) | 0.714 (0.649–0.792) | 0.832 | 0.820 | 0.920 |
| psychiatric_or_mental_illness | 1,676 | 348 | 0.023 (0.016–0.032) | 0.624 (0.572–0.667) | 0.869 | 0.836 | 0.934 |
| latino | 693 | 129 | 0.039 (0.027–0.053) | 0.550 (0.465–0.651) | 0.877 | 0.841 | 0.934 |
| jewish | 2,674 | 437 | 0.030 (0.022–0.036) | 0.682 (0.645–0.726) | 0.860 | 0.846 | 0.923 |
| male | 14,295 | 2,203 | 0.024 (0.021–0.027) | 0.644 (0.622–0.661) | 0.881 | 0.867 | 0.925 |
| female | 16,449 | 2,270 | 0.018 (0.015–0.020) | 0.673 (0.654–0.690) | 0.884 | 0.879 | 0.919 |
| asian | 1,518 | 178 | 0.015 (0.008–0.022) | 0.674 (0.601–0.736) | 0.907 | 0.903 | 0.915 |
| christian | 13,361 | 1,260 | 0.010 (0.008–0.012) | 0.731 (0.706–0.756) | 0.897 | 0.923 | 0.883 |

Queue under the assumption of 100,000 comments a day:

| Budget | Queue/day | Expected toxic in queue | Expected toxic missed | Precision | Recall |
|---|---|---|---|---|---|
| 1% | 1,000 | 965 | 10,418 | 0.965 | 0.084 |
| 2% | 2,000 | 1,836 | 9,628 | 0.918 | 0.154 |
| 5% | 5,000 | 3,893 | 7,500 | 0.779 | 0.341 |

CPU budget used: device cpu, 2 threads; TF-IDF fit 54 s, scoring 0.42 ms/comment; MiniLM embeddings 83 comments/s (278 s); whole notebook 7.6 min.

**Reading the tables.** At a 5% review budget the queue is about 78% toxic but reaches only about a
third of all toxic comments; recall at budget *q* cannot exceed *q* divided by the toxic rate (≈ 0.11).
At this CPU budget the frozen MiniLM probe (8,000 training comments) is clearly *worse* than TF-IDF
trained on the same 8,000 comments, and far below TF-IDF trained on 100,000: more data for a cheap
model beat a frozen general-purpose sentence encoder. Benign comments mentioning *black* or *white*
are queued about 7–8 times as often as benign comments overall (FPR 0.10 and 0.085 vs 0.012), with
BPSN AUCs near 0.69 and BNSP AUCs above 0.95: the identity-term over-flagging signature.

Your own repository's tables must come from your own run; until you have run it, each cell says
*TBD — fill from your run*.

## Leakage traps

* **Threshold or calibration fitted on the test split.**
* **Duplicate comments across splits** (reposts) inflating scores.
* **Identity labels as model features.** They are human annotations not available at inference time;
  they are for evaluation only.
* **Other annotation columns as features** (for example `severe_toxicity`, `insult`, annotator counts,
  reaction counts in the Kaggle metadata): they are labels or post-hoc signals, not inputs.
* **TF-IDF vocabulary or IDF fitted on validation or test text.** Fit the vectoriser inside the
  training pipeline.
* **A subsample chosen using test statistics**, or a subsample that is not stratified and is then
  compared with a model trained on a different one.

## MVP (≈ 5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repository from the template; read the HF card and the terms of your identity-label source; `make data` (TFDS or WILDS) with recorded versions; counts per split, toxic rate, identity-annotated subset size; de-duplication across splits; documented stratified subsample |
| 2 | Keyword and TF-IDF + LR baselines in a `Pipeline`; PR-AUC, ROC-AUC, precision/recall at thresholds and at capacity budgets, all unit-tested on tiny hand-made cases |
| 3 | Subgroup metrics (FPR/FNR, Subgroup/BPSN/BNSP AUC) with minimum-count rules and bootstrap CIs; calibration on validation; choose the operating point on validation |
| 4 | Compact pretrained model: frozen embeddings + LR on CPU, *or* fine-tuning on a small subsample (GPU if available); the same evaluation; latency per comment measured; final test run once |
| 5 | Queue analysis; error analysis; API with bands and model version; README, model card (intended use: triage for human review only), data card |

**Extensions** (optional):

* Full fine-tuning on the full training set (**GPU only**).
* Soft-label training on the `toxicity` fraction; compare with binary training.
* Bias mitigation and a before/after subgroup table.
* Per-subgroup calibration analysis.
* An active-learning loop: which comments should reviewers label next?

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

**Validation rules:** 1–64 comments per request; `id` required and unique within the request; `text` a
non-empty string after stripping whitespace, at most for example 5,000 characters (longer → 422, or
truncated with `"truncated": true`; choose one and test it; the walkthrough rejects); valid UTF-8;
unknown fields rejected. The API never returns a "remove" decision. It must not log raw comment text
by default (log ids, scores and lengths).

**Precomputed vs online:** the model, tokenizer or vectoriser, calibration map and threshold are loaded
once at start-up from the bundle. Scoring is online per request (batched for efficiency). The threshold
is derived offline from the validation set and the chosen review budget, stored in the model metadata
and returned in each response, so that reviewers' dashboards know which operating point produced a
band.

## Limitations

* **Label noise and annotator disagreement:** labels are fractions of a small number of crowd raters;
  comments near 0.5 are genuinely contested, and the 0.5 cut-off is a convention, not truth.
* **Identity-term over-flagging:** models learn that identity words co-occur with toxicity and flag
  benign mentions; your subgroup table will show where.
* **Identity labels are partial** (an annotated subset) and also noisy; small subgroups have wide CIs
  and some cannot be evaluated at all.
* **Domain and time:** the comments come from one commenting platform in a specific period (read the
  dataset card for provenance); language, slang and norms drift.
* **Toxicity ≠ policy violation:** a platform's policy is more specific and context-dependent than a
  single toxicity score.
* **Offensive content:** the dataset contains hateful text; handle examples with care in your README
  and presentations.
* Capacity numbers are **assumptions** for illustration.

## Interview questions

1. **Why PR-AUC rather than accuracy or ROC-AUC?** Strong answer: low prevalence; precision is what
   reviewers experience; ROC-AUC can be high while the top of the ranking is poor; numbers from your
   run.
2. **How did you pick the threshold?** Covers: review capacity → top-*q* fraction on validation;
   precision and recall at that point; recalibrate when traffic or prevalence changes.
3. **What is BPSN AUC and what did it show?** Covers: the definition; a low BPSN means benign identity
   mentions are scored above toxic non-mentions; your subgroups and counts.
4. **Labels are fractions. How did you use them, and what are the risks?** Covers: the 0.5 convention,
   disagreement near the boundary, soft labels, evaluation on confidently labelled examples as a
   sensitivity check.
5. **Your transformer beats TF-IDF by a small PR-AUC margin. Ship it?** Covers: CI overlap, latency and
   cost per comment, subgroup errors, maintenance; the decision depends on capacity economics.
6. **How would you reduce false positives on identity mentions?** Covers: data augmentation or
   reweighting, counterfactual tests (swap identity terms), evaluation before and after, trade-offs.
7. **Design the moderation pipeline end to end.** Covers: score → bands → human review → decisions
   logged as new labels; appeals; a priority queue; latency; never automatic removal from a single
   model without policy approval.
8. **How do you monitor this after launch?** Covers: score distribution drift, queue size, reviewer
   agreement with the model, delayed labels from reviews (biased, because only reviewed items get
   labels), subgroup FPR over time.
9. **Why must identity labels not be features?** Covers: unavailable at inference; they would encode
   the protected attribute directly; evaluation only.
10. **How did you run this on a CPU?** Covers: the TF-IDF baseline, subsampling, frozen embeddings,
    measured throughput; full fine-tuning on a GPU as an optional extension.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `modtriage`)
- [ ] `DATA_CARD.md`: HF card licence (CC0) quoted, identity-label source and its terms, citation, split counts, subsample seed, de-duplication counts
- [ ] Tests: vectoriser fitted only on train; no cross-split duplicates in train; identity columns absent from model inputs
- [ ] Keyword and TF-IDF/LR baselines and a compact pretrained model on the same splits
- [ ] Results table: PR-AUC, ROC-AUC, precision/recall at 1%/2%/5% review budgets, with CIs, filled from your own run (the walkthrough's final step generates it; until you run it, **TBD — fill from your run**)
- [ ] Subgroup table: n, FPR, FNR, Subgroup/BPSN/BNSP AUC, filled from your own run (**TBD — fill from your run**)
- [ ] Queue analysis table under stated capacity assumptions
- [ ] API with bands, threshold and model version in responses; no raw-text logging; validation tests
- [ ] `MODEL_CARD.md`: intended use = triage for human review; out of scope = automatic removal
- [ ] CI green on a few synthetic, non-offensive example comments

## Stretch: connect to the course's churnlab patterns

* **Registry:** store the threshold and review budget **inside** each model version's metadata, so
  that promoting or rolling back a model also restores the operating point that was validated with it
  — [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py)
  ([Chapter 15](../../part1-foundations/ch15-monitoring-and-rollback/README.md)).
* **Monitoring:** track the daily share of comments routed to review (a jump means drift or an
  attack), PSI on the score distribution and, as reviewer decisions arrive, the precision of the queue
  and subgroup FPR, with alert rules as in
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py). Note
  the selection bias: only queued comments receive human labels.
