# Solutions · Chapter 54 · Evaluating LLM systems & guardrails

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run or hardware · ✍️ personal / open-ended.

---

## Exercise 1 · `field_f1`

✅ Reference code: [`code/nb16.py`](solutions.py) (`ex1`), graded in CI.

```python
def field_f1(pred_dict, gold_dict):
    norm = lambda v: re.sub(r"\s+", " ", str(v).strip().lower())
    predicted = {k: norm(v) for k, v in pred_dict.items() if v is not None and norm(v) != ""}
    gold = {k: norm(v) for k, v in gold_dict.items()}
    if not predicted and not gold:
        return 1.0
    tp = sum(1 for k, v in predicted.items() if k in gold and gold[k] == v)
    if tp == 0:
        return 0.0
    p, r = tp / len(predicted), tp / len(gold)
    return 2 * p * r / (p + r)
```

**Notes.** Decide the normalisation rules (case, whitespace, `#` in order ids, synonyms
such as "refund request" vs "refund") before you look at results, write them down, and
apply them to every system equally. Exact match answers "is the whole record usable?";
field-F1 shows *which* fields fail. Report both.

---

## Exercise 2 · `position_bias_rate`

✅ Reference code: [`code/nb16.py`](solutions.py) (`ex2`).

```python
def position_bias_rate(judgments_ab, judgments_ba):
    same = [a.strip().upper() == b.strip().upper() for a, b in zip(judgments_ab, judgments_ba)]
    return sum(same) / len(same)
```

**Notes.** The same candidate is labelled A in the first order and B in the second, so a
content-driven judge flips its letter. The same letter twice means the slot decided.
Report it with the share of "A" answers (which slot is favoured) and judge accuracy per
order; count only order-consistent verdicts as wins, or average the two probabilities.

---

## Exercise 3 · `guardrail`

✅ Reference code: [`code/nb16.py`](solutions.py) (`ex3`): rules only, which pass the hidden
set; in the notebook, combining them with the classifier (`rules OR classifier`) is the
natural answer:

```python
RULES = [r"\b(ignore|disregard|forget)\b[^.]{0,40}\b(instructions?|rules|prompt|above|told|everything)\b",
         r"\b(you are now|pretend (you are|to be)|act as|from now on)\b",
         r"\[/?system\]|<\|im_start\|>|#{3}|\bsystem override\b|\bnew (policy|instructions)\b",
         r"\b(system prompt|original instructions|hidden prompt|repeat them)\b",
         r"\b(idiots?|morons?|clowns?|garbage|pathetic|losers?|dumbest|incompetent|shut up)\b"]

def guardrail(text):
    if any(re.search(p, text, flags=re.I) for p in RULES):
        return True
    return gclf.predict_proba(gvec.transform([text]))[0, 1] >= THRESH
```

**Notes.** Rules give precision and explainability on known patterns and catch new
*families* only if you anticipated their wording; the classifier generalises to
paraphrases but needs labelled data and a threshold chosen on training/validation data
(never on the test set). Note the traps in the benign data ("please disregard my duplicate
ticket", "ignore my previous email"): a bare `ignore|disregard` rule fails the FPR budget.
Real deployments add a second model-based check on the *output* (schema validation,
policy checks) and log every block for review. A synthetic test set this regular is easy;
real attacks are not, so keep red-teaming.

---

## Exercise 4 · Evaluating a cheaper model

✍️ Model answer:

(a) A few hundred tickets sampled from recent production traffic (stratified by
category, language, length, difficulty; including rare but costly cases: angry customers,
legal threats, PII-heavy tickets), PII-redacted, with reference replies or rubric labels
written by experienced agents; a frozen held-out portion that prompt tuning never sees,
refreshed periodically.

(b) Automatic: format/schema validity, factual consistency checks against the ticket and
order data (right order id, policy-compliant refund amounts), length, latency and cost.
Human evaluation on a sample with a rubric (correctness, helpfulness, tone, policy). An
LLM judge for scale, but: a stronger model than either candidate and not from the same
family, both orders per pair, a rubric plus reference answer, calibrated against human
labels (agreement/κ) before use, and blind to which model produced which reply.

(c) Paired comparison on the same tickets, bootstrap or paired tests for the difference
with a pre-registered non-inferiority margin ("the cheap model is at most 2 points
worse"), sample size from a power calculation (16.2 showed how wide intervals are with
12 tickets); slice results by category so an average does not hide a broken segment.

(d) Prompt-injection suite with attack success rate at or below the current model, toxicity
and harassment probes, PII leakage (does it repeat other customers' data, emails, card
numbers?), refusal behaviour, jailbreak suites; all as regression tests in CI.

(e) After launch: A/B test or staged rollout with agent edit rate (how much humans change
drafts), resolution time, CSAT, escalation and reopen rates, guardrail block rates, latency
p50/p95, cost per ticket, sampled human audits; alert and roll back on regressions.

---

## Checkpoint answers

1. ✍️ Sample real traffic, stratify, label carefully (exact fields or rubric), freeze a
   held-out split, version it. A small set with exact labels gives cheap, deterministic,
   unbiased metrics you can re-run on every change; an LLM-judged large set adds noise and
   the judge's own biases (16.3) and may reward style over correctness. Use LLM judges
   for scale only after validating them against human labels.

2. 🔬 A confidence interval for the *paired* difference (bootstrap over examples, 16.2:
   with 12 tickets intervals were very wide); a pre-registered metric and normalisation;
   evidence it is not overfitted to the set (fresh held-out examples); per-slice results;
   checking that the gain is not just formatting the parser happens to prefer; and ideally
   an online test. 3 points on 50 examples is usually within noise.

3. ✅ Position bias (swap orders; 16.3 measured it), verbosity bias (compare a padded
   and an unpadded correct answer; 16.3), self-preference (judge outputs from several model
   families and compare with human preferences). Mitigate: both orders + consistency
   filter, length-controlled comparisons or instructions, a different and stronger judge,
   rubric and references, human calibration.

4. ✅ Untrusted input containing instructions that the model follows because instructions
   and data share one channel (16.4: the schema prompt told the model not to follow such
   instructions, and attacks still succeeded on part of the tickets). Instructions in the
   prompt are only a request, not an enforcement mechanism. Layers: least privilege for
   tools, output validation and business rules, human approval for risky actions, input
   and output guardrails, isolation/marking of untrusted content, monitoring and red-teaming.

5. ✍️ Before: rules + a classifier on the input (injection, abuse, PII), cheap and fast;
   after: schema validation, policy checks, toxicity/PII on the output. Threshold set on
   validation data for a target FPR on benign traffic (blocking real customers is costly),
   then measure recall per attack family, including families held out from training
   (16.5), plus latency. Review blocked samples; feed new attacks back into training.

6. 🔬 Needs: tokens in/out per request (16.6 measured them per prompt variant), throughput
   per accelerator at the target latency (batching!), peak vs average load, price per
   accelerator-hour or per token. 10M/day ≈ 116 requests/s average, more at peak; cost =
   requests × (input tokens × input price + output tokens × output price) for an API, or
   accelerators needed × hourly price for self-hosting. Levers: shorter prompts and outputs,
   prefix caching, smaller or distilled models, quantisation, batching, early rejection by
   cheap guardrails, caching repeated answers.
