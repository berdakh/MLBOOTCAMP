# Solutions · Chapter 53 · Fairness, robustness & privacy

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## Exercise 1 · `bpsn_bnsp_auc`

✅ Reference code: [`code/nb15.py`](solutions.py) (`ex1`), graded in CI.

```python
def bpsn_bnsp_auc(y, s, subgroup):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); g = np.asarray(subgroup, bool)
    bpsn = (g & (y == 0)) | (~g & (y == 1))   # background positives + subgroup negatives
    bnsp = (g & (y == 1)) | (~g & (y == 0))   # background negatives + subgroup positives
    return roc_auc_score(y[bpsn], s[bpsn]), roc_auc_score(y[bnsp], s[bnsp])
```

**Reading them.** A model that adds a constant to every score of comments mentioning the
identity lowers BPSN (its benign subgroup comments climb above toxic background ones)
and raises BNSP. That is the classic "identity term = toxic" bias. Low subgroup AUC with
normal BPSN/BNSP means the model is simply worse at ranking within that conversation.

---

## Exercise 2 · `clip_and_noise`

✅ Reference code: [`code/nb15.py`](solutions.py) (`ex2`).

```python
def clip_and_noise(per_example_grads, clip, sigma, rng):
    g = np.asarray(per_example_grads, float)
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    clipped = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))
    return (clipped.sum(0) + rng.normal(0.0, sigma * clip, size=g.shape[1])) / len(g)
```

**Notes.** Clipping bounds each example's contribution to the sum (sensitivity `C`), so
the noise std `σC` is calibrated to that bound; the noise on the *mean* is `σC/B`, which is
why DP-SGD prefers large batches. Clip per example, not the averaged batch gradient
(that bounds nothing about an individual record). In PyTorch, per-example gradients come
from `torch.func.vmap(grad(...))` or Opacus's hooks.

---

## Exercise 3 · Explaining and mitigating a subgroup gap

✍️ Model answer:

(a) Benign comments that mention Muslims are scored as more toxic than many genuinely
toxic comments about other topics, so at the production threshold people discussing their
own religion are three times as likely to have harmless comments hidden. The harm falls
on the group the policy should protect, and it erodes trust.

(b) Causes: in the training data the identity term co-occurs mostly with toxic comments
(attacks mention it more often than everyday conversation does), so the model learns the
word itself as a toxicity signal; label noise / annotator bias on comments mentioning the
identity; few benign examples mentioning it; a bag-of-words model cannot use context
("I am Muslim" vs an insult).

(c) Data: add or up-weight benign comments mentioning the identity (sampling, templates,
counterfactual augmentation swapping identity terms); trade-off: synthetic text is less
natural, may shift other metrics. Model: a contextual model (fine-tuned transformer),
fairness-aware training (e.g. penalising subgroup score gaps, adversarial debiasing),
or masking identity terms; trade-off: overall AUC can drop slightly, more compute,
masking loses information needed to detect identity attacks. Policy: per-context thresholds
or route low-confidence identity-mentioning comments to human review instead of automatic
removal; trade-off: cost, legal questions about group-dependent treatment, slower moderation.

(d) Monitor BPSN/BNSP and FPR/FNR per identity on fresh labelled samples (with CIs), appeal
and reversal rates per group, the share of removals that mention identity terms, template
counterfactual tests in CI. Roll back if subgroup FPR gaps exceed an agreed limit beyond
their confidence interval, or appeals concentrate on one group.

---

## Checkpoint answers

1. ✅ Subgroup AUC: AUC within comments mentioning the group. BPSN: subgroup negatives
   vs background positives; low = the group's benign comments outrank real toxicity
   elsewhere (false positives caused by the mention). BNSP: subgroup positives vs
   background negatives; low = toxic comments about the group are missed. BPSN detects
   "mentions are flagged as toxic", because it pits benign mentions against toxic
   non-mentions: an identity-term penalty pushes exactly those benign mentions up.

2. ✅ The score distributions of non-toxic comments differ by group, so one threshold
   cuts them at different quantiles: different FPRs (15.3). Per-group thresholds can
   equalise FPR but then change FNR, and with different base rates no non-trivial
   classifier can equalise FPR, FNR and calibration (positive predictive value)
   simultaneously (Kleinberg et al.; Chouldechova). You choose which error matters most
   for the harm in question, and document it.

3. 🔬 Shift tests: train on one period/source and evaluate on a later period and other
   sources (15.4: catches temporal drift, new topics/vocabulary, source-specific labelling);
   slice metrics by segment. Perturbation tests: typos and leetspeak (catches brittle
   tokens, evasion), counterfactual identity swaps (catches spurious identity features),
   paraphrases, negation, length changes. Also check calibration under shift, not just AUC.

4. ✅ Inferring whether a specific record was in the training set from the model's
   behaviour (loss, confidence). Vulnerability comes from memorisation: over-fitting,
   small datasets, rare/outlier records, large models. Measure with a shadow-model or
   loss-threshold attack: AUC of distinguishing members from non-members (15.6: weak
   regularisation gave a higher attack AUC than strong regularisation). DP training
   bounds it provably.

5. ✍️ Each step: sample a batch; compute per-example gradients; clip each to norm ≤ C;
   sum; add Gaussian noise N(0, σ²C²); divide by the batch size; take the optimiser step.
   C bounds any record's influence (too small biases gradients, too large needs more noise
   for the same privacy); σ (noise multiplier) sets the privacy per step. An accountant
   composes the steps (with the sampling rate) into (ε, δ): for any neighbouring
   datasets, the probability of any outcome changes by at most a factor e^ε (plus δ).
   Smaller ε, stronger privacy, lower accuracy (15.7: AUC fell as σ grew).

6. ✍️ Fairness: subgroup AUC/BPSN/BNSP and FPR/FNR per relevant identity and per language
   variety, with CIs, on locally labelled data from that country; threshold choice
   justified per harm; template and counterfactual tests in local languages. Robustness:
   evaluation on recent local data (not translated training data), slang/typo/transliteration
   perturbations, adversarial red-teaming, calibration. Privacy: PII redaction in training
   and logs, data minimisation and retention per local law (e.g. GDPR), membership-inference
   audit or DP training if user content is trained on, access controls. Plus: human appeal
   path, monitoring dashboards per group, rollback criteria.
