# Equations

This appendix collects the key equations of the book in one place, grouped by topic. Each one is derived or motivated in the chapter linked under it, usually with a worked example and a lab cell that checks it numerically. The [glossary](glossary.md) defines the words; this appendix holds the mathematics.

## How to use this appendix

An equation recited from memory demonstrates little. What matters, in study and in technical interviews alike, is knowing what the equation *means* and *when it fails to apply*. For any equation here:

1. **State it in words first.** "NDCG is the discounted gain of the list, divided by the gain of the best possible order." Then write the symbols.
2. **State the assumptions.** Are the rows independent? Are the probabilities calibrated? Are the delays exponential? Follow-up questions usually concern an assumption rather than the algebra.
3. **Give a numerical example.** "With a 5% base rate, a constant score has PR-AUC 0.05, so 0.15 is three times the floor." A concrete number shows that the equation has been used, not only read.
4. **State the decision it supports.** Choosing a threshold, sizing a fleet of servers, or deciding whether an experiment has enough users.

Each entry below has the same structure: the name, the equation, one sentence *in words*, and a link to the chapter where it is taught.

## Contents

- [Evaluation and metrics](#evaluation-and-metrics) (18)
- [Probability, statistics and experimentation](#probability-statistics-and-experimentation) (21)
- [Data, SQL and features](#data-sql-and-features) (18)
- [Linear models and optimisation](#linear-models-and-optimisation) (11)
- [Trees and boosting](#trees-and-boosting) (2)
- [Ranking, retrieval and recommendation](#ranking-retrieval-and-recommendation) (25)
- [Deep learning and transformers](#deep-learning-and-transformers) (17)
- [Efficient training and scaling](#efficient-training-and-scaling) (22)
- [Serving and systems](#serving-and-systems) (34)
- [Monitoring](#monitoring) (5)
- [Ads, CTR and bandits](#ads-ctr-and-bandits) (7)
- [Privacy and fairness](#privacy-and-fairness) (5)
- [Coding interviews: complexity](#coding-interviews-complexity) (11)

## Evaluation and metrics

### Positive (base) rate

```math
\hat{\pi} = \frac{1}{N}\sum_{i=1}^{N} y_i
```

*In words:* The share of rows whose label is 1; every ranking metric's floor depends on it.

*Taught in:* [Ch. 4 · SQL for ML](../part1-foundations/ch04-sql-for-ml/README.md)

### Precision, recall, F1

```math
P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}
```

*In words:* How pure the flagged set is, how much of the target it covers, and their harmonic mean.

*Taught in:* [Ch. 8 · Thresholds & calibration](../part1-foundations/ch08-thresholds-and-calibration/README.md) · [Ch. 54 · Evaluating LLM systems & guardrails](../part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/README.md)

### F1 and macro-F1

```math
F_1 = \frac{2PR}{P + R}, \qquad \text{macro-}F_1 = \frac{1}{C} \sum_{c=1}^{C} F_{1,c}
```

*In words:* The harmonic mean of precision $P$ and recall $R$, averaged over classes with equal weight.

*Taught in:* [Ch. 43 · Fine-tuning a classifier](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md)

### Precision at k

```math
\text{precision@}k = \frac{1}{k} \sum_{i \in \text{top-}k} y_i
```

*In words:* Of the $k$ highest-scored rows, the share that are positive; fits an action with fixed capacity.

*Taught in:* [Ch. 17 · Python & SQL drills](../part1-foundations/ch17-python-and-sql-drills/README.md) · [Ch. 16 · Final evaluation & documentation](../part1-foundations/ch16-final-evaluation-and-docs/README.md)

### Lift at k

```math
\mathrm{lift@}k = \frac{\mathrm{precision@}k}{\pi}
```

*In words:* How many times better than random the top of the list is.

*Taught in:* [Ch. 8 · Thresholds & calibration](../part1-foundations/ch08-thresholds-and-calibration/README.md) · [Ch. 9 · Error analysis & artifacts](../part1-foundations/ch09-error-analysis-and-artifacts/README.md)

### ROC-AUC as a probability

```math
\text{ROC-AUC} = P\left(s^{+} > s^{-}\right) + \tfrac{1}{2} P\left(s^{+} = s^{-}\right)
```

*In words:* The chance that a random positive outscores a random negative.

*Taught in:* [Ch. 5 · Leakage-safe splits & a baseline](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md)

### ROC-AUC from ranks

```math
\text{AUC} = \frac{\sum_{i \in \text{pos}} r_i - n_+(n_+ + 1)/2}{n_+\, n_-}
```

*In words:* The fraction of (positive, negative) pairs ordered correctly, read off the positives' ranks (average ranks for ties).

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### Average precision (PR-AUC)

```math
\text{AP} = \sum_{n} (R_n - R_{n-1})\, P_n = \frac{1}{N_{+}} \sum_{k:\ y_{(k)} = 1} \text{precision@}k
```

*In words:* Average the precision at the rank of every positive.

*Taught in:* [Ch. 5 · Leakage-safe splits & a baseline](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md)

### No-skill floors

```math
\text{AP}_{\text{constant}} = \pi, \qquad \text{ROC-AUC}_{\text{constant}} = 0.5
```

*In words:* A constant score has PR-AUC equal to the positive rate and ROC-AUC 0.5, so always quote PR-AUC next to $\pi$.

*Taught in:* [Ch. 5 · Leakage-safe splits & a baseline](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md)

### Lift-normalised slice PR-AUC

```math
\text{normalised PR-AUC}_s = \frac{\text{PR-AUC}_s}{\pi_s}
```

*In words:* Divide each slice's PR-AUC by its own positive rate before comparing slices.

*Taught in:* [Ch. 9 · Error analysis & artifacts](../part1-foundations/ch09-error-analysis-and-artifacts/README.md)

### Log loss (binary cross-entropy)

```math
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]
```

*In words:* The average surprise of the true labels; a confident wrong answer is punished hard. It is what logistic regression minimises, which is why it tends to come out calibrated.

*Taught in:* [Ch. 8 · Thresholds & calibration](../part1-foundations/ch08-thresholds-and-calibration/README.md) · [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md)

### Normalised entropy

```math
\mathrm{NE} = \frac{\mathrm{LL}}{H(\bar{p})}, \qquad H(\bar{p}) = -\big[\, \bar{p} \log \bar{p} + (1 - \bar{p}) \log (1 - \bar{p}) \,\big]
```

*In words:* Log loss divided by the log loss of always predicting the base rate; below 1 means the model beats the prior.

*Taught in:* [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md)

### Brier score

```math
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2
```

*In words:* Mean squared error of the probabilities; the base-rate constant scores $\pi(1-\pi)$.

*Taught in:* [Ch. 8 · Thresholds & calibration](../part1-foundations/ch08-thresholds-and-calibration/README.md)

### Expected calibration error

```math
\mathrm{ECE} = \sum_{b=1}^{B} \frac{\lvert S_b \rvert}{n} \, \big\lvert \mathrm{acc}(S_b) - \mathrm{conf}(S_b) \big\rvert
```

*In words:* Bin predictions by confidence, and average the gap between accuracy and confidence, weighted by how many predictions fall in each bin.

*Taught in:* [Ch. 43 · Fine-tuning a classifier](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md)

### Temperature scaling

```math
\mathbf p = \mathrm{softmax}(\mathbf z / T), \qquad T^{\star} = \arg\min_{T > 0} \mathrm{NLL}_{\text{val}}(T)
```

*In words:* Divide every logit by one scalar $T$ chosen on validation data. $T > 1$ softens, $T < 1$ sharpens, the argmax never moves.

*Taught in:* [Ch. 43 · Fine-tuning a classifier](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md)

### Expected value of a threshold, and the break-even probability

```math
\text{EV}(t) = TP(t)\cdot V - FP(t)\cdot C,
\qquad
p^{*} = \frac{C}{V + C}
```

*In words:* Calling an account with probability $p$ is worth $pV - (1-p)C$, positive exactly when $p > p^{*}$.

*Taught in:* [Ch. 8 · Thresholds & calibration](../part1-foundations/ch08-thresholds-and-calibration/README.md)

### Spearman rank correlation

```math
\rho_s = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}
```

*In words:* The Pearson correlation of the ranks; 1 means both models order every item identically.

*Taught in:* [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md)

### LLM-judge position bias and attack success rate

```math
\text{bias rate} = \frac{1}{n} \sum_{i} \mathbb{1}\big[\, v_i^{(X,Y)} = v_i^{(Y,X)} \,\big], \qquad \mathrm{ASR} = \frac{\#\text{injections that got the attacker's value}}{\#\text{injections}}
```

*In words:* A judge that picks the same slot in both orders is following position, not quality; ASR is the share of prompt injections that worked.

*Taught in:* [Ch. 54 · Evaluating LLM systems & guardrails](../part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/README.md)

## Probability, statistics and experimentation

### Churn compounds

```math
\text{retained}(n) = (1 - c)^n
```

*In words:* Each month keeps a fraction $1 - c$ of whoever is left, so the fractions multiply.

*Taught in:* [Ch. 2 · A reproducible ML project](../part1-foundations/ch02-reproducible-project/README.md)

### Across-seed spread

```math
\bar{x} = \frac{1}{k}\sum_{i=1}^{k} x_i, \qquad
s = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k} (x_i - \bar{x})^2}
```

*In words:* $s$ is the typical distance of one seed's score from the average.

*Taught in:* [Ch. 2 · A reproducible ML project](../part1-foundations/ch02-reproducible-project/README.md)

### Standard error of a proportion

```math
\mathrm{SE}(\hat p) = \sqrt{\frac{\hat p\,(1 - \hat p)}{n}},
\qquad
n \approx \frac{1.96^2\, p\,(1-p)}{m^2}
```

*In words:* The noise in a rate shrinks with $\sqrt{n}$; the second formula is the sample size for a 95% margin of $\pm m$.

*Taught in:* [Ch. 7 · Model selection](../part1-foundations/ch07-model-selection/README.md) · [Ch. 19 · Event logs & columnar data](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md) · [Ch. 43 · Fine-tuning a classifier](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md)

### Interval width from sample size

```math
\text{CI width} \propto \frac{1}{\sqrt{n}}
```

*In words:* Four times the data halves the interval; for a slice, $n$ is its number of positives (or customers).

*Taught in:* [Ch. 9 · Error analysis & artifacts](../part1-foundations/ch09-error-analysis-and-artifacts/README.md) · [Ch. 16 · Final evaluation & documentation](../part1-foundations/ch16-final-evaluation-and-docs/README.md)

### Bootstrap percentile interval

```math
\text{CI}_{95\%} = \left[\, Q_{0.025}\big(\hat\theta^{*}_{1..B}\big),\; Q_{0.975}\big(\hat\theta^{*}_{1..B}\big) \,\right]
```

*In words:* The middle 95% of the metric across resamples.

*Taught in:* [Ch. 7 · Model selection](../part1-foundations/ch07-model-selection/README.md)

### Paired bootstrap for a difference

```math
d_i = s_i^{B} - s_i^{A}, \qquad \bar{d}^{\,*}_b = \frac{1}{n} \sum_{i \in \text{resample}_b} d_i, \qquad \mathrm{CI}_{95\%} = \big[\, q_{2.5\%}(\bar{d}^{\,*}),\ q_{97.5\%}(\bar{d}^{\,*}) \,\big]
```

*In words:* Score both models on the same resampled items, take the difference each time, and read off the interval of the differences.

*Taught in:* [Ch. 7 · Model selection](../part1-foundations/ch07-model-selection/README.md) · [Ch. 46 · Contrastive training & re-ranking](../part3-deep-learning-systems/ch46-contrastive-training-and-reranking/README.md) · [Ch. 54 · Evaluating LLM systems & guardrails](../part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/README.md)

### Pre-registered decision rule

```math
\text{report "better"} \iff Q_{0.025}\big(\Delta^{*}_{1..B}\big) > 0
```

*In words:* The whole 95% interval of the difference must sit above zero.

*Taught in:* [Ch. 16 · Final evaluation & documentation](../part1-foundations/ch16-final-evaluation-and-docs/README.md)

### Why copied rows do not help

```math
\mathrm{SE}_{\text{row}}(k \text{ copies}) = \frac{\mathrm{SE}_{\text{row}}}{\sqrt{k}}, \qquad \mathrm{SE}_{\text{customer}}(k \text{ copies}) = \mathrm{SE}_{\text{customer}}
```

*In words:* Duplicating rows shrinks a row bootstrap's SE by $\sqrt{k}$ without adding information; a customer-level bootstrap is unchanged.

*Taught in:* [Ch. 7 · Model selection](../part1-foundations/ch07-model-selection/README.md)

### Grouped bootstrap and the design effect

```math
\mathrm{Var}_{\text{grouped}} \approx \mathrm{Var}_{\text{iid}} \cdot \big(1 + (m - 1)\,\rho_{\text{ICC}}\big)
```

*In words:* With $m$ correlated rows per user, a row-level bootstrap underestimates the variance by this design effect; resampling users gets it right automatically.

*Taught in:* [Ch. 38 · Final evaluation & mock loop](../part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/README.md)

### Paired differences

```math
\mathrm{Var}(A - B) = \mathrm{Var}(A) + \mathrm{Var}(B) - 2\mathrm{Cov}(A, B)
```

*In words:* Scoring two systems on the same requests makes their errors correlated, and the covariance term removes the shared part.

*Taught in:* [Ch. 38 · Final evaluation & mock loop](../part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/README.md)

### Two-proportion z-test

```math
z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p\,(1-\hat p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}, \qquad \hat p = \frac{x_A + x_B}{n_A + n_B}
```

*In words:* The observed difference in rates, measured in standard errors computed as if both arms shared one rate (the null hypothesis).

*Taught in:* [Ch. 29 · A/B testing fundamentals](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md)

### Standard error of a difference, and its confidence interval

```math
\mathrm{SE}(\bar y_B - \bar y_A) = \sqrt{\frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B}}, \qquad \text{95\% CI} = (\bar y_B - \bar y_A) \pm 1.96\,\mathrm{SE}
```

*In words:* Variances of independent estimates add, so the SE of a difference is the root of the summed squared SEs.

*Taught in:* [Ch. 29 · A/B testing fundamentals](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md)

### Sample size per arm

```math
n \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\, p(1-p)}{\delta^2}
```

*In words:* (how many standard errors of separation you need)² × (noise per unit) ÷ (effect)². The 2 is there because both arms are noisy.

*Taught in:* [Ch. 29 · A/B testing fundamentals](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md) · [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

### Sample-ratio mismatch: chi-square goodness of fit

```math
\chi^2 = \sum_{\text{arms}} \frac{(O - E)^2}{E}, \qquad \text{1 degree of freedom for two arms}
```

*In words:* How far the observed arm counts are from the planned split, scaled by the expected counts.

*Taught in:* [Ch. 29 · A/B testing fundamentals](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md)

### Multiple testing: Bonferroni and Benjamini–Hochberg

```math
\text{Bonferroni: reject } H_i \text{ if } p_i \le \frac{\alpha}{m}; \qquad \text{BH: } k = \max\Big\{ i : p_{(i)} \le \frac{i}{m}\,q \Big\}, \text{ reject } p_{(1)}, \dots, p_{(k)}
```

*In words:* Bonferroni splits α evenly over m tests (controls the chance of **any** false positive); BH sorts the p-values and compares each with a rising threshold (controls the **share** of false discoveries). With m independent null tests, $P(\text{any false positive}) = 1 - (1-\alpha)^m$.

*Taught in:* [Ch. 29 · A/B testing fundamentals](../part2-ml-at-scale/ch29-ab-testing-fundamentals/README.md)

### CUPED adjustment and its variance

```math
y^{cv}_i = y_i - \theta\,(x_i - \bar x), \qquad \theta = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(x)}, \qquad \mathrm{Var}(y^{cv}) = \mathrm{Var}(y)\,(1 - \rho^2)
```

*In words:* Regress the metric on its pre-period value and keep the residual (plus the mean). Because $x$ is fixed before randomisation, the difference between arms is unchanged; only the noise shrinks, by the factor $1 - \rho^2$, and so does the sample size you need.

*Taught in:* [Ch. 30 · Advanced experimentation](../part2-ml-at-scale/ch30-advanced-experimentation/README.md)

### Delta method for a ratio of means

```math
\mathrm{Var}\!\left(\frac{\bar c}{\bar n}\right) \approx \frac{1}{N}\left( \frac{\sigma_c^2}{\mu_n^2} - \frac{2\,\mu_c\,\sigma_{cn}}{\mu_n^3} + \frac{\mu_c^2\,\sigma_n^2}{\mu_n^4} \right)
```

*In words:* Linearise $R = \mu_c / \mu_n$ around the means; the variance of the linear approximation is $`\nabla R^\top \Sigma\, \nabla R / N`$ with gradient $\nabla R = (1/\mu_n,\ -\mu_c/\mu_n^2)$.

*Taught in:* [Ch. 30 · Advanced experimentation](../part2-ml-at-scale/ch30-advanced-experimentation/README.md)

### The interleaving statistic

```math
\Delta = \frac{1}{U} \sum_{u=1}^{U} \left( \text{clicks}_{A,u} - \text{clicks}_{B,u} \right), \qquad t = \frac{\Delta}{s_\Delta / \sqrt{U}}
```

*In words:* Per user, credited clicks for A minus credited clicks for B; a one-sample t-test asks whether the mean difference is zero.

*Taught in:* [Ch. 30 · Advanced experimentation](../part2-ml-at-scale/ch30-advanced-experimentation/README.md)

### What row sampling does to a per-request metric

```math
P(\text{click kept} \mid \text{request kept}) = \frac{\mathbb{E}\left[1-(1-f)^{c}\right]}{1-(1-f)^{10}}
```

*In words:* A request with $c$ clicks keeps at least one of them with probability $1-(1-f)^c$, and survives at all with probability $1-(1-f)^{10}$.

*Taught in:* [Ch. 19 · Event logs & columnar data](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md)

### A hash sample's size

```math
\mathbb{E}[\text{kept}] = N q, \qquad \mathrm{SD}[\text{kept}] = \sqrt{N q (1-q)}
```

*In words:* Each key is kept with probability $q$, independently of its id, so the sample size is binomial.

*Taught in:* [Ch. 19 · Event logs & columnar data](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md)

### Reservoir sampling (Algorithm R)

```math
P(\text{item } i \text{ in sample after } n \text{ items}) = \frac{k}{n}
```

*In words:* After reading $n$ items, every one of them is in the size-$k$ reservoir with the same probability.

*Taught in:* [Ch. 20 · Aggregation & joins at scale](../part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md)

## Data, SQL and features

### Label

```math
y_i(T) = \mathbb{1}\left[\, T \le \text{churn\_date}_i < T + 30\ \text{days} \,\right]
```

*In words:* 1 if the account cancels in the 30 days starting at T, else 0.

*Taught in:* [Ch. 4 · SQL for ML](../part1-foundations/ch04-sql-for-ml/README.md)

### Completed-week rule

```math
w + 7\ \text{days} \le T
```

*In words:* The week must have ended by T.

*Taught in:* [Ch. 4 · SQL for ML](../part1-foundations/ch04-sql-for-ml/README.md)

### Embargo condition

```math
T + h \le t_v
```

*In words:* The label must be settled before the evaluation period starts.

*Taught in:* [Ch. 5 · Leakage-safe splits & a baseline](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md)

### Fan-out row count

```math
\text{rows after join} = \sum_{c} n_U(c)\, n_K(c)
\quad\Rightarrow\quad
\text{SUM}_{\text{joined}}(x) = \sum_{c} n_K(c) \sum_{u \in U_c} x_u
```

*In words:* Each key contributes the product of its row counts, so every value from $U$ is counted once per matching row of $K$.

*Taught in:* [Ch. 4 · SQL for ML](../part1-foundations/ch04-sql-for-ml/README.md) · [Ch. 20 · Aggregation & joins at scale](../part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md)

### Cohort retention

```math
\text{ret}_k(c) = \frac{\lvert\{\text{customers of cohort } c \text{ active in month } c + k\}\rvert}{\lvert \text{cohort } c \rvert}
```

*In words:* The share of a sign-up cohort still active $k$ months later.

*Taught in:* [Ch. 17 · Python & SQL drills](../part1-foundations/ch17-python-and-sql-drills/README.md)

### A window feature, cut off strictly before the snapshot

```math
x_u(d) = \sum_{t=d-7}^{d-1} \text{clicks}_u(t)
```

*In words:* The value stamped at day $d$ sums the 7 *complete* days before $d$.

*Taught in:* [Ch. 21 · Feature pipelines & feature stores](../part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md)

### The point-in-time rule, with a TTL

```math
s^{*}(e) = \underset{s:\; k_s = k_e,\; \tau_s < t_e}{\arg\max}\; \tau_s, \qquad \text{use } s^{*} \text{ only if } t_e - \tau_{s^{*}} \le \text{TTL}
```

*In words:* Take the latest snapshot strictly before the event, and use it only if it is young enough.

*Taught in:* [Ch. 21 · Feature pipelines & feature stores](../part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md)

### Feature age at serving

```math
\text{age} = t_\text{request} - t_\text{newest event in the value}
```

*In words:* What the model sees online is as old as the newest data behind it.

*Taught in:* [Ch. 21 · Feature pipelines & feature stores](../part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md)

### A smoothed click rate

```math
\widetilde{\text{CTR}} = \frac{\text{clicks} + \alpha\, p_0}{\text{impressions} + \alpha}
```

*In words:* Add $\alpha$ pseudo-impressions at the prior rate, so tiny counts do not produce extreme rates.

*Taught in:* [Ch. 21 · Feature pipelines & feature stores](../part2-ml-at-scale/ch21-feature-pipelines-and-feature-stores/README.md)

### Target encoding

```math
\mathrm{TE}(c) = \frac{1}{n_c} \sum_{i \in c} y_i
```

*In words:* The mean label of the rows in that level. It is honest only if the rows $i$ are strictly earlier, with closed label windows, and never include the row being scored.

*Taught in:* [Ch. 6 · Pipelines & cross-validation](../part1-foundations/ch06-pipelines-and-cross-validation/README.md)

### Standardisation

```math
z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
```

*In words:* How many training standard deviations a value sits from the training mean.

*Taught in:* [Ch. 6 · Pipelines & cross-validation](../part1-foundations/ch06-pipelines-and-cross-validation/README.md)

### log1p transform

```math
\tilde{x} = \ln(1 + x)
```

*In words:* Turns "per unit" into "roughly per doubling", and maps 0 to 0.

*Taught in:* [Ch. 6 · Pipelines & cross-validation](../part1-foundations/ch06-pipelines-and-cross-validation/README.md)

### Partitioning by hash

```math
\text{reducer}(k) = h(k) \bmod R
```

*In words:* A key always goes to the same reducer, and keys spread evenly *if* no key is hot.

*Taught in:* [Ch. 20 · Aggregation & joins at scale](../part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md)

### HyperLogLog relative error

```math
\frac{\mathrm{SE}(\hat n)}{n} \approx \frac{1.04}{\sqrt{m}}, \qquad m = 2^{p}
```

*In words:* The standard error of a distinct count shrinks with the square root of the number of registers.

*Taught in:* [Ch. 20 · Aggregation & joins at scale](../part2-ml-at-scale/ch20-aggregation-and-joins-at-scale/README.md)

### Which tumbling window an event belongs to

```math
\text{start}(t) = \left\lfloor \frac{t}{w} \right\rfloor w
```

*In words:* Round the event time down to a multiple of the window size.

*Taught in:* [Ch. 22 · Streaming features](../part2-ml-at-scale/ch22-streaming-features/README.md)

### Sliding windows: how many contain one event

```math
\{\, s = \lfloor t/h \rfloor h - j h \;:\; j = 0, \dots, S/h - 1 \,\}
```

*In words:* With size $S$ and hop $h$, each event falls in $S/h$ windows, so the state does $S/h$ times the updates.

*Taught in:* [Ch. 22 · Streaming features](../part2-ml-at-scale/ch22-streaming-features/README.md)

### Watermark, closing and lateness

```math
W = \max_{\text{seen}} t - L, \qquad \text{close } [s, s+w) \text{ when } W \ge s + w, \qquad \text{late if } s + w \le W_\text{before}
```

*In words:* The watermark trails the largest event time seen; a window is emitted once the watermark passes its end; an event whose window has already closed is late.

*Taught in:* [Ch. 22 · Streaming features](../part2-ml-at-scale/ch22-streaming-features/README.md)

### Session windows

```math
t \text{ joins } [a, b] \iff a - g \le t \le b + g
```

*In words:* An event joins (and can bridge) every session within the gap $g$.

*Taught in:* [Ch. 22 · Streaming features](../part2-ml-at-scale/ch22-streaming-features/README.md)

## Linear models and optimisation

### Logistic regression

```math
\log\frac{p}{1-p} = \beta_0 + \sum_j \beta_j z_j,
\qquad
p = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}
```

*In words:* Each standardised feature adds $\beta_j z_j$ to the log-odds of churn.

*Taught in:* [Ch. 6 · Pipelines & cross-validation](../part1-foundations/ch06-pipelines-and-cross-validation/README.md)

### Logistic regression: loss and gradient

```math
L(w, b) = \frac{1}{n}\sum_{i=1}^n \big[\log(1 + e^{z_i}) - y_i z_i\big] + \frac{\lambda}{2}\lVert w\rVert^2, \qquad \nabla_w L = \frac{1}{n} X^\top(p - y) + \lambda w, \quad \frac{\partial L}{\partial b} = \overline{p - y}
```

*In words:* The log loss written stably in terms of logits; its gradient is "prediction minus target", pushed back through the features.

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### Softmax cross-entropy gradient

```math
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\log p_y, \qquad \frac{\partial L}{\partial z_k} = p_k - \mathbb{1}[k = y]
```

*In words:* The gradient for each class is its predicted probability, minus 1 for the true class. Confidently wrong → large gradient; confidently right → almost zero. For a batch, $dW = X^\top (P - Y) / n$.

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### Log-sum-exp

```math
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad m = \max_j z_j
```

*In words:* Shift by the maximum before exponentiating; the answer is unchanged and nothing overflows. Cross-entropy from logits is $\mathrm{logsumexp}(z) - z_y$.

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### Minimum-norm least squares and ridge

```math
\hat{w}_{\text{min-norm}} = X^{+} y = \arg\min_w \lVert w \rVert_2 \ \text{ s.t. } Xw = y \ \ (p > n), \qquad \hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
```

*In words:* When there are more features than points, infinitely many $w$ fit exactly; the pseudo-inverse picks the shortest. Near $p = n$ the smallest singular value $\sigma_{\min}$ of $X$ is tiny and $\lVert \hat{w} \rVert$ scales like $1/\sigma_{\min}$: the peak. Ridge's $\lambda$ keeps $1/(\sigma^2 + \lambda)$ bounded.

*Taught in:* [Ch. 57 · ML depth & research discussion](../part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md)

### k-means objective (inertia)

```math
J = \sum_{i=1}^n \lVert x_i - \mu_{c(i)} \rVert^2, \qquad c(i) = \arg\min_k \lVert x_i - \mu_k \rVert, \qquad \mu_k = \mathrm{mean}\{x_i : c(i) = k\}
```

*In words:* Assign each point to its nearest centre, move each centre to the mean of its points; each step can only lower $J$.

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### Two tools you use to verify everything else

```math
\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2\,a^\top b + \lVert b \rVert^2, \qquad \frac{\partial f}{\partial w_i} \approx \frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}
```

*In words:* The distance expansion turns pairwise distances into one matrix product; the central difference has error O(ε²) and is the reference for every gradient check.

*Taught in:* [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md)

### The AdamW update

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \lambda\, \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
```

*In words:* Keep running averages of the gradient and of its square, correct their start-up bias, step by their ratio, and shrink the weights separately (decoupled weight decay).

*Taught in:* [Ch. 27 · PyTorch training done right](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md) · [Ch. 40 · Training efficiency & memory](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md)

### Linear warmup, then cosine decay

```math
\eta_t = \eta_{\max} \cdot \begin{cases} \dfrac{t+1}{T_w} & t < T_w \\[6pt] r + (1-r)\,\dfrac{1}{2}\Big(1 + \cos\big(\pi\, \tfrac{t - T_w}{T - T_w}\big)\Big) & t \ge T_w \end{cases}
```

*In words:* Ramp up over $T_w$ steps, then follow half a cosine down to a floor.

*Taught in:* [Ch. 27 · PyTorch training done right](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md) · [Ch. 57 · ML depth & research discussion](../part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md)

### Learning rate vs batch size

```math
\eta(kB) = k\, \eta(B) \ \ \text{(linear, SGD)}, \qquad \eta(kB) = \sqrt{k}\, \eta(B) \ \ \text{(square-root, more conservative)}
```

*In words:* A bigger batch has less gradient noise, so it tolerates a larger step: linear scaling for SGD, square-root as the safer rule.

*Taught in:* [Ch. 57 · ML depth & research discussion](../part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md)

### Gradient-norm clipping

```math
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{c}{\lVert \mathbf{g} \rVert_2}\right)
```

*In words:* If the whole gradient is longer than $c$, shrink it to length $c$; its direction does not change.

*Taught in:* [Ch. 27 · PyTorch training done right](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md)

## Trees and boosting

### SSE from sums, and the split gain

```math
\text{SSE} = \sum (y - \bar y)^2 = S_2 - \frac{S_1^2}{m}, \qquad \text{gain} = \text{SSE}_{\text{parent}} - \text{SSE}_{\text{left}} - \text{SSE}_{\text{right}}
```

*In words:* A group's spread needs only its count, sum and sum of squares; the best split is the one that removes the most spread (variance reduction).

*Taught in:* [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

### Gradient boosting update

```math
r_i^{(m)} = -\frac{\partial L(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)}, \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x), \qquad F_0 = \arg\min_c \sum_i L(y_i, c)
```

*In words:* Fit a small tree $h_m$ to the negative gradient (for $L = \tfrac12 (y-F)^2$, the residual $y - F$) and add a shrunken copy.

*Taught in:* [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

## Ranking, retrieval and recommendation

### DCG and NDCG

```math
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{\,\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \mathrm{NDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}
```

*In words:* Add up each item's gain, discounted by the log of its rank, and divide by the best possible order's value.

*Taught in:* [Ch. 24 · Learning to rank](../part2-ml-at-scale/ch24-learning-to-rank/README.md) · [Ch. 31 · ML from scratch I](../part2-ml-at-scale/ch31-ml-from-scratch-1/README.md) · [Ch. 45 · Embeddings & semantic search](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md)

### MRR

```math
\mathrm{MRR} = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\text{rank}_q}
```

*In words:* The average, over requests, of one over the rank of the first relevant item.

*Taught in:* [Ch. 24 · Learning to rank](../part2-ml-at-scale/ch24-learning-to-rank/README.md)

### HR@10 and NDCG@10 with one target

```math
\mathrm{HR@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \mathbf{1}[r_u < 10], \qquad \mathrm{NDCG@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \frac{\mathbf{1}[r_u < 10]}{\log_2(r_u + 2)}
```

*In words:* A hit if the target ranks in the top 10; NDCG gives a hit at rank $r$ (0 = top) the discount $1/\log_2(r + 2)$.

*Taught in:* [Ch. 28 · Sequence models for user behaviour](../part2-ml-at-scale/ch28-sequence-models-for-user-behavior/README.md)

### Recall@K

```math
\mathrm{Recall@}K(u) = \frac{\lvert \text{top}_K(u) \cap T_u \rvert}{\lvert T_u \rvert}
```

*In words:* Of the items that were relevant for a user, what share did the K candidates contain?

*Taught in:* [Ch. 25 · Candidate generation](../part2-ml-at-scale/ch25-candidate-generation/README.md) · [Ch. 45 · Embeddings & semantic search](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md) · [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md)

### Catalogue coverage

```math
\mathrm{Coverage} = \frac{\left\lvert \bigcup_{u} \text{top}_K(u) \right\rvert}{\lvert \mathcal{I} \rvert}
```

*In words:* The share of the catalogue that appears in at least one user's list.

*Taught in:* [Ch. 25 · Candidate generation](../part2-ml-at-scale/ch25-candidate-generation/README.md)

### RankNet's pairwise loss and its gradient

```math
L_{ij} = \log\!\left(1 + e^{-\sigma (s_i - s_j)}\right), \qquad \lambda_{ij} = \frac{\partial L_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma (s_i - s_j)}}
```

*In words:* A logistic loss on the score difference of a pair where $i$ should beat $j$.

*Taught in:* [Ch. 24 · Learning to rank](../part2-ml-at-scale/ch24-learning-to-rank/README.md)

### LambdaRank

```math
\lambda^{\text{LR}}_{ij} = \lambda_{ij} \cdot \left\lvert \Delta \mathrm{NDCG}_{ij} \right\rvert
```

*In words:* Use RankNet's gradient, scaled by how much NDCG would change if $i$ and $j$ swapped places.

*Taught in:* [Ch. 24 · Learning to rank](../part2-ml-at-scale/ch24-learning-to-rank/README.md)

### Co-visitation counts

```math
C(a, b) = \sum_{s} \mathbf{1}[a \in s]\, \mathbf{1}[b \in s], \quad a \neq b
```

*In words:* Count the sessions in which both items appear (once per session).

*Taught in:* [Ch. 25 · Candidate generation](../part2-ml-at-scale/ch25-candidate-generation/README.md)

### Implicit ALS objective and its closed-form step

```math
\min_{X, Y} \sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2\Big), \qquad x_u = (Y^\top C_u Y + \lambda I)^{-1} Y^\top C_u\, p_u
```

*In words:* Weighted least squares on every cell, with weight $c_{ui}$; with item vectors fixed, each user vector is a small linear solve.

*Taught in:* [Ch. 25 · Candidate generation](../part2-ml-at-scale/ch25-candidate-generation/README.md)

### BPR, the pairwise alternative

```math
L_\text{BPR} = -\sum_{(u, i, j)} \log \sigma\!\left(x_u^\top y_i - x_u^\top y_j\right) + \lambda \lVert \Theta \rVert^2
```

*In words:* For a clicked item $i$ and a sampled unclicked item $j$, make the user score $i$ above $j$.

*Taught in:* [Ch. 25 · Candidate generation](../part2-ml-at-scale/ch25-candidate-generation/README.md)

### The two-tower score

```math
s(u, i) = \frac{\mathbf{u}^\top \mathbf{v}_i}{\tau}, \qquad \lVert \mathbf{u} \rVert = \lVert \mathbf{v}_i \rVert = 1
```

*In words:* The cosine similarity of the two normalised vectors, divided by a temperature.

*Taught in:* [Ch. 26 · Two-tower retrieval & ANN](../part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md)

### Cosine similarity

```math
\cos(\mathbf a, \mathbf b) = \frac{\mathbf a \cdot \mathbf b}{\lVert \mathbf a \rVert \, \lVert \mathbf b \rVert}
```

*In words:* The angle between two vectors, ignoring their lengths; for unit vectors it is just the dot product.

*Taught in:* [Ch. 42 · Pretrained transformers & tokenisation](../part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/README.md) · [Ch. 45 · Embeddings & semantic search](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md)

### In-batch softmax (InfoNCE) with the logQ correction

```math
L = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\!\big(s(u_b, i_b) - \log q_{i_b}\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_b, i_j) - \log q_{i_j}\big)}
```

*In words:* For each user in the batch, cross-entropy of picking their own item out of the batch's items, with every item's logit lowered by the log of how often it is sampled.

*Taught in:* [Ch. 26 · Two-tower retrieval & ANN](../part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md)

### InfoNCE with in-batch and hard negatives

```math
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathbf q_i \cdot \mathbf d_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf q_i \cdot \mathbf d_j / \tau)}
```

*In words:* For every query, a softmax over all $N$ documents in the batch (its positive, the other $B - 1$ positives and $N - B$ hard negatives) and the cross-entropy of picking its own.

*Taught in:* [Ch. 46 · Contrastive training & re-ranking](../part3-deep-learning-systems/ch46-contrastive-training-and-reranking/README.md)

### A random-guess baseline for the loss

```math
\mathcal{L}_{\text{uniform}} = \ln N
```

*In words:* A model that cannot tell the positive from $N - 1$ negatives scores $\ln N$; useful loss starts below that.

*Taught in:* [Ch. 46 · Contrastive training & re-ranking](../part3-deep-learning-systems/ch46-contrastive-training-and-reranking/README.md)

### BM25

```math
\mathrm{BM25}(q, d) = \sum_{t \in q} \mathrm{idf}(t)\, \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1 \big(1 - b + b\, \lvert d \rvert / \text{avgdl}\big)}, \qquad
\mathrm{idf}(t) = \ln\!\Big(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\Big)
```

*In words:* Rare query terms (high idf) that appear in the document count most; repeated occurrences saturate ($k_1$); long documents are penalised ($b$).

*Taught in:* [Ch. 45 · Embeddings & semantic search](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md) · [Ch. 37 · Recommendation & search system design](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md)

### Reciprocal rank fusion

```math
\mathrm{RRF}(d) = \sum_{s} \frac{1}{k + \mathrm{rank}_s(d)}, \qquad k = 60
```

*In words:* Add one over (constant plus rank) across rankers; consistent agreement beats one first place.

*Taught in:* [Ch. 45 · Embeddings & semantic search](../part3-deep-learning-systems/ch45-embeddings-and-semantic-search/README.md)

### Recall of an approximate index

```math
\mathrm{Recall@}k = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{\lvert \text{ANN}_k(q) \cap \text{Exact}_k(q) \rvert}{k}
```

*In words:* The share of the exact top-$k$ that the approximate search also returned.

*Taught in:* [Ch. 26 · Two-tower retrieval & ANN](../part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md)

### Cost of exact search

```math
\text{cost} = N \cdot d \ \text{multiply-adds per query}
```

*In words:* One dot product per item per query.

*Taught in:* [Ch. 26 · Two-tower retrieval & ANN](../part2-ml-at-scale/ch26-two-tower-retrieval-and-ann/README.md)

### How ANN search cost scales

```math
\text{exact: } O(n\,d), \qquad \text{IVF: } O\!\left(\frac{n \cdot \text{nprobe}}{\text{nlist}}\, d\right), \qquad \text{HNSW: roughly } O(\log n)\ \text{hops}
```

*In words:* IVF scans only the probed lists; HNSW walks a graph in roughly logarithmic hops.

*Taught in:* [Ch. 37 · Recommendation & search system design](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md)

### Re-ranking cost

```math
\text{latency} \approx \text{depth} \times t_{\text{pair}} / \text{parallelism}, \qquad \text{pair inferences/s} = \text{QPS} \times \text{depth}
```

*In words:* Every query pays one cross-encoder pass per re-ranked candidate.

*Taught in:* [Ch. 46 · Contrastive training & re-ranking](../part3-deep-learning-systems/ch46-contrastive-training-and-reranking/README.md)

### The position-based click model

```math
P(c = 1 \mid u, i, p) = \underbrace{\theta_p}_{P(\text{exam} \mid p)} \cdot \underbrace{r_{u,i}}_{P(\text{rel} \mid u, i)}
```

*In words:* A click happens only if the position is examined *and* the item is relevant, and the two are independent.

*Taught in:* [Ch. 23 · Implicit feedback & position bias](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md) · [Ch. 38 · Final evaluation & mock loop](../part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/README.md)

### Propensities from random traffic

```math
\frac{\theta_p}{\theta_0} = \frac{\text{CTR}_\text{rand}(p)}{\text{CTR}_\text{rand}(0)}
```

*In words:* With random order, mean relevance is the same at every position, so CTR ratios are examination ratios.

*Taught in:* [Ch. 23 · Implicit feedback & position bias](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md)

### The IPS click-rate estimator, and clipping

```math
\hat r_\text{IPS} = \frac{1}{n} \sum_{j=1}^{n} \frac{c_j}{\theta_{p_j}}, \qquad \hat r_\text{clip} = \frac{1}{n} \sum_{j=1}^{n} c_j \min\!\left(\frac{1}{\theta_{p_j}}, M\right)
```

*In words:* Divide each click by the chance that its position was examined; optionally cap the weight at $M$.

*Taught in:* [Ch. 23 · Implicit feedback & position bias](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md)

### Off-policy value of a re-ranking

```math
\hat V(\text{new}) = \sum_{i \in \text{request}} \frac{\theta_{q_i}}{\theta_{p_i}}\, c_i
```

*In words:* A click observed at logged position $p$ that the new ranking moves to position $q$ counts $\theta_q / \theta_p$ clicks.

*Taught in:* [Ch. 23 · Implicit feedback & position bias](../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/README.md) · [Ch. 38 · Final evaluation & mock loop](../part2-ml-at-scale/ch38-final-evaluation-and-mock-loop/README.md)

## Deep learning and transformers

### Binary cross-entropy on a logit, and its value at initialisation

```math
\ell(z, y) = -\big[y \log \sigma(z) + (1-y) \log(1-\sigma(z))\big], \qquad \ell^{*}_\text{const} = -\big[p \log p + (1-p) \log(1-p)\big]
```

*In words:* The negative log of the probability the model gave to the true label; the best constant predictor scores the label entropy.

*Taught in:* [Ch. 27 · PyTorch training done right](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md)

### Softmax and cross-entropy

```math
\mathrm{softmax}(\mathbf{z})_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}, \qquad \ell = -\log \mathrm{softmax}(\mathbf{z})_y
```

*In words:* Turn $C$ logits into probabilities, and pay $-\log$ of the probability of the right class.

*Taught in:* [Ch. 27 · PyTorch training done right](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md) · [Ch. 43 · Fine-tuning a classifier](../part3-deep-learning-systems/ch43-fine-tuning-a-classifier/README.md)

### Softmax with temperature

```math
p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
```

*In words:* Divide the logits by $\tau$ before the softmax: lower $\tau$ sharpens the distribution, higher flattens it.

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md)

### Backprop through a two-layer MLP

```math
\begin{aligned}
dZ_2 &= (P - Y)/n, & dW_2 &= H^\top dZ_2, & db_2 &= \textstyle\sum_i dZ_2 \\
dH &= dZ_2 W_2^\top, & dZ_1 &= dH \odot \mathbb{1}[Z_1 > 0], & dW_1 &= X^\top dZ_1
\end{aligned}
```

*In words:* Start from "prediction minus target" and walk the graph backwards; every gradient has the shape of its variable.

*Taught in:* [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

### Batch norm

```math
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat x + \beta, \qquad \text{running} \leftarrow (1 - m)\cdot\text{running} + m \cdot \text{batch statistic}
```

*In words:* Standardise each feature with the batch's mean and (biased) variance during training, with the running averages at inference.

*Taught in:* [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

### LayerNorm

```math
\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \frac{1}{C}\sum_c x_c, \quad \sigma^2 = \frac{1}{C}\sum_c (x_c - \mu)^2
```

*In words:* Per token, subtract the mean and divide by the standard deviation over channels, then apply a learned scale $\gamma$ and shift $\beta$.

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md)

### Inverted dropout

```math
y = \frac{x \odot m}{1 - p}, \quad m_j \sim \mathrm{Bernoulli}(1 - p), \qquad \mathbb{E}[y] = x
```

*In words:* Zero each unit with probability $p$ and scale the survivors by $1/(1-p)$, so the expected output is unchanged; do nothing at inference.

*Taught in:* [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

### Scaled dot-product attention

```math
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}} + M\Big)\, V
```

*In words:* Score every query against every key, scale, hide the future, normalise each row, average the values.

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md) · [Ch. 28 · Sequence models for user behaviour](../part2-ml-at-scale/ch28-sequence-models-for-user-behavior/README.md) · [Ch. 32 · ML from scratch II](../part2-ml-at-scale/ch32-ml-from-scratch-2/README.md)

### Multi-head attention

```math
\mathrm{MHA}(X) = \mathrm{Concat}(\text{head}_1, \dots, \text{head}_H)\, W^O, \qquad
\text{head}_i = \mathrm{Attention}(X W_i^Q, X W_i^K, X W_i^V)
```

*In words:* $H$ attentions on $d_k = C/H$ channels each, side by side, mixed by $W^O$.

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md)

### The attention mask inside attention

```math
\text{score}_{ij} \;\leftarrow\; \text{score}_{ij} + (1 - m_j)\cdot(-\infty)
```

*In words:* A padded key $j$ gets $-\infty$, so after the softmax its weight is exactly 0 for every query.

*Taught in:* [Ch. 42 · Pretrained transformers & tokenisation](../part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/README.md)

### Mask-aware mean pooling

```math
\mathbf{s} = \frac{\sum_{t=1}^{T} m_t\, \mathbf{h}_t}{\sum_{t=1}^{T} m_t}
```

*In words:* Average the hidden vectors of the real tokens only.

*Taught in:* [Ch. 42 · Pretrained transformers & tokenisation](../part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/README.md)

### Cross-entropy and perplexity

```math
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta\big(x_{t+1} \mid x_{\le t}\big), \qquad
\mathrm{PPL} = e^{\mathcal{L}}
```

*In words:* The average surprise at the true next token; perplexity is "as unsure as a uniform choice among PPL options".

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md)

### Next-item loss over the catalogue

```math
L = -\frac{1}{\lvert \mathcal{T} \rvert} \sum_{t \in \mathcal{T}} \log \frac{\exp(\mathbf{h}_t^\top \mathbf{e}_{y_{t+1}})}{\sum_{j=1}^{N} \exp(\mathbf{h}_t^\top \mathbf{e}_j)}
```

*In words:* At every non-padding position $t$, softmax cross-entropy of the next item over all $N$ items, scored by the dot product with the (tied) item embeddings.

*Taught in:* [Ch. 28 · Sequence models for user behaviour](../part2-ml-at-scale/ch28-sequence-models-for-user-behavior/README.md)

### Self-attention cost

```math
\text{compute} = O(T^2 d) \text{ per layer}, \qquad M_{\text{scores}} = L \cdot H \cdot T^2 \cdot b \text{ bytes}
```

*In words:* Attention compute and score memory grow with the square of sequence length.

*Taught in:* [Ch. 57 · ML depth & research discussion](../part3-deep-learning-systems/ch57-ml-depth-and-research-discussion/README.md)

### The LoRA layer

```math
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r}
```

*In words:* The frozen weight's output plus a low-rank correction, scaled by $\alpha / r$.

*Taught in:* [Ch. 44 · Parameter-efficient fine-tuning](../part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md)

### LoRA trainable parameters

```math
\underbrace{r\,(d + k)}_{\text{LoRA}} \quad \text{vs} \quad \underbrace{d\,k}_{\text{full}}
```

*In words:* LoRA trains $r(d + k)$ numbers per matrix instead of $dk$.

*Taught in:* [Ch. 44 · Parameter-efficient fine-tuning](../part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md)

### Merging

```math
W' = W_0 + \frac{\alpha}{r} B A
```

*In words:* After training, fold the update into a weight of the original shape: zero extra latency, but one task per merged copy.

*Taught in:* [Ch. 44 · Parameter-efficient fine-tuning](../part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md)

## Efficient training and scaling

### FLOPs of a matrix multiply

```math
\text{FLOPs}\big(A_{m \times k}\,B_{k \times n}\big) = 2\,m\,n\,k
```

*In words:* Each of the $m n$ outputs is a dot product of length $k$, i.e. $k$ multiplies and $k$ adds.

*Taught in:* [Ch. 39 · GPU fundamentals & profiling](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md)

### Arithmetic intensity

```math
I = \frac{\text{FLOPs performed}}{\text{bytes moved to and from memory}}
```

*In words:* How much arithmetic you get for every byte you read or write.

*Taught in:* [Ch. 39 · GPU fundamentals & profiling](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md)

### The roofline

```math
P_{\text{attainable}} = \min\big(P_{\text{peak}},\; I \cdot B_{\text{mem}}\big), \qquad I^{\star} = \frac{P_{\text{peak}}}{B_{\text{mem}}}
```

*In words:* You cannot beat the chip's peak, and you cannot compute faster than memory can feed you.

*Taught in:* [Ch. 39 · GPU fundamentals & profiling](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md)

### Epoch time of an input pipeline

```math
T_{\text{epoch}} \approx n_{\text{batches}} \big(t_{\text{load}} + t_{\text{step}}\big) \;\;(\text{no workers}), \qquad
T_{\text{epoch}} \approx n_{\text{batches}} \max\!\Big(\frac{t_{\text{load}}}{k},\; t_{\text{step}}\Big) \;\;(k \text{ workers})
```

*In words:* Without workers, loading and compute add up; with $k$ workers they overlap and the slower of the two sets the pace.

*Taught in:* [Ch. 39 · GPU fundamentals & profiling](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md)

### Padding waste

```math
\text{waste} = 1 - \frac{\sum_i \ell_i}{\sum_{\text{batches } b} \lvert b \rvert \cdot \max_{i \in b} \ell_i}
```

*In words:* The share of computed positions that are padding.

*Taught in:* [Ch. 42 · Pretrained transformers & tokenisation](../part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/README.md)

### Model-state memory

```math
M_{\text{state}} = \Psi \,\big(b_{w} + b_{g} + b_{\text{opt}}\big)
```

*In words:* Every parameter pays for its weight, its gradient and its optimizer state.

*Taught in:* [Ch. 40 · Training efficiency & memory](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md)

### Training memory with a frozen base

```math
M \approx b_w\, \Psi_{\text{total}} + 16\, \Psi_{\text{trained}} + M_{\text{act}}
```

*In words:* Every weight must be stored ($b_w$ bytes each), but only trained weights need gradients and AdamW state (16 bytes each in fp32).

*Taught in:* [Ch. 44 · Parameter-efficient fine-tuning](../part3-deep-learning-systems/ch44-parameter-efficient-fine-tuning/README.md)

### Activation memory of a transformer

```math
M_{\text{act}} \approx L \cdot B \cdot T \cdot \big(34\,h + 5\,a\,T\big) \;\text{bytes}
```

*In words:* Linear in layers, batch and hidden size, and **quadratic in sequence length** through the attention probabilities.

*Taught in:* [Ch. 40 · Training efficiency & memory](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md)

### Gradient accumulation

```math
g = \sum_{i=1}^{k} \frac{n_i}{N}\, \nabla_\theta \bar{\ell}_i, \qquad N = \sum_{i=1}^{k} n_i
```

*In words:* Weight each micro-batch's mean-loss gradient by its share of the examples.

*Taught in:* [Ch. 40 · Training efficiency & memory](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md)

### Global batch size

```math
B_{\text{global}} = N \cdot b_{\text{per GPU}} \cdot k_{\text{accum}}
```

*In words:* The batch the optimiser sees is GPUs times per-GPU batch times accumulation steps.

*Taught in:* [Ch. 47 · Distributed data parallel](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md)

### Activation checkpointing cost

```math
\text{step time} \approx F + 2F \;\longrightarrow\; F + 2F + F = \tfrac{4}{3} \times \text{(no checkpointing)}
```

*In words:* Backward costs about two forwards; recomputing adds one more.

*Taught in:* [Ch. 40 · Training efficiency & memory](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md)

### Loss scaling

```math
\nabla_\theta (S \cdot \mathcal L) = S \cdot \nabla_\theta \mathcal L, \qquad \text{fp16 normal range} \approx [6.1 \times 10^{-5},\; 65{,}504]
```

*In words:* Scale the loss up so gradients are representable in fp16, then divide by $S$ in fp32 before the update.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

### Data parallelism averages gradients

```math
g = \frac{1}{N} \sum_{r=1}^{N} g_r, \qquad g_r = \nabla_\theta \frac{1}{b} \sum_{j \in \text{shard } r} \ell_j
```

*In words:* With equal shards of size $b$, the mean of the ranks' mean-loss gradients is the gradient of the mean loss over the global batch $N b$.

*Taught in:* [Ch. 47 · Distributed data parallel](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md)

### Ring all-reduce traffic

```math
\text{bytes sent per rank} = 2\,\frac{N - 1}{N}\, M
```

*In words:* $N - 1$ reduce-scatter steps plus $N - 1$ all-gather steps, each sending $M / N$.

*Taught in:* [Ch. 47 · Distributed data parallel](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md)

### All-reduce time and bus bandwidth

```math
T_{\text{comm}} \approx 2\,\frac{N-1}{N}\,\frac{M}{\text{BW}} + 2(N-1)\,\alpha, \qquad \text{busbw} = \text{algbw} \cdot 2\,\frac{N-1}{N}
```

*In words:* A bandwidth term that barely depends on $N$, plus a latency term ($\alpha$ per hop) that grows with it.

*Taught in:* [Ch. 47 · Distributed data parallel](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md)

### Scaling efficiency

```math
E = \frac{T_{\text{compute}}}{T_{\text{step}}}, \qquad T_{\text{step}} \approx T_{\text{compute}} + (1 - o)\, T_{\text{comm}}
```

*In words:* The share of the step spent on useful compute; $o$ is the fraction of communication hidden behind backward.

*Taught in:* [Ch. 47 · Distributed data parallel](../part3-deep-learning-systems/ch47-distributed-data-parallel/README.md)

### Model-state memory per GPU (mixed-precision Adam, $N$ data-parallel GPUs)

```math
M_{\text{DDP}} = 16\Psi, \quad M_{\text{Z1}} = 4\Psi + \frac{12\Psi}{N}, \quad M_{\text{Z2}} = 2\Psi + \frac{14\Psi}{N}, \quad M_{\text{Z3}} = \frac{16\Psi}{N}
```

*In words:* Each stage shards one more of optimizer state (12 bytes), gradients (2) and parameters (2).

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

### Pipeline bubble

```math
\text{bubble} = \frac{p - 1}{m + p - 1}
```

*In words:* $p - 1$ slots of fill and drain out of $m + p - 1$.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

### Tensor parallelism

```math
\text{column: } y = \big[\, x W_1^\top,\; x W_2^\top \,\big] \;(\text{all-gather}), \qquad
\text{row: } y = x_1 W_1^\top + x_2 W_2^\top \;(\text{all-reduce})
```

*In words:* Split the weight by output columns and all-gather the halves, or by input rows and all-reduce the partial sums.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

### Training compute and time

```math
C \approx 6\,N D, \qquad t \approx \frac{C}{n_{\text{GPU}} \cdot P_{\text{peak}} \cdot \text{MFU}}
```

*In words:* 6 FLOPs per parameter per token; time is that divided by the FLOP/s you actually get.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md) · [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

### Chinchilla compute-optimal sizing

```math
D^{\star} \approx 20\, N^{\star} \;\;\Rightarrow\;\; C = 6 N^{\star} (20 N^{\star}) \;\;\Rightarrow\;\; N^{\star} = \sqrt{C / 120}
```

*In words:* Compute-optimal training uses about 20 tokens per parameter, so the best model size grows with the square root of compute.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

### Optimal checkpoint interval (Young 1974; Daly 2006)

```math
\text{waste}(\tau) \approx \frac{C}{\tau} + \frac{\tau/2 + R}{M}, \qquad \tau^{\star} = \sqrt{2\,C\,M} \;\;(\text{Daly: } \approx \sqrt{2CM} - C)
```

*In words:* Checkpoint cost per interval plus, per failure, half an interval of lost work and a restart.

*Taught in:* [Ch. 48 · Scaling to large models](../part3-deep-learning-systems/ch48-scaling-to-large-models/README.md)

## Serving and systems

### Symmetric int8 quantisation

```math
q = \mathrm{clamp}\!\big(\mathrm{round}(x / s),\,-127,\,127\big), \qquad s = \frac{\max_i \lvert x_i \rvert}{127}, \qquad \hat{x} = q\,s
```

*In words:* Divide by the step size, round to the nearest integer, and multiply back.

*Taught in:* [Ch. 49 · Quantisation & distillation](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md)

### Affine (asymmetric) int8 quantisation

```math
s = \frac{x_{\max} - x_{\min}}{255}, \qquad z = \mathrm{round}\!\left(-128 - \frac{x_{\min}}{s}\right), \qquad q = \mathrm{clip}\!\left(\mathrm{round}(x/s) + z,\ -128,\ 127\right), \qquad \hat x = s\,(q - z)
```

*In words:* A scale turns float units into integer steps, a zero-point shifts so that 0.0 lands exactly on an integer; the rounding error per value is at most $s/2$.

*Taught in:* [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md)

### Rounding error bound

```math
\lvert x - \hat{x} \rvert \le \frac{s}{2}
```

*In words:* Rounding moves a value by at most half a step.

*Taught in:* [Ch. 49 · Quantisation & distillation](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md)

### Model size in memory

```math
\text{bytes} \approx \text{parameters} \times \frac{\text{bits per parameter}}{8}
```

*In words:* Memory is parameters times bytes per parameter; int8 is a quarter of fp32.

*Taught in:* [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md) · [Ch. 49 · Quantisation & distillation](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md)

### Distillation loss

```math
\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(p_t^{(T)} \,\Vert\, p_s^{(T)}\right) + (1 - \alpha)\, \mathrm{CE}(z_s, y), \qquad p^{(T)} = \mathrm{softmax}(z / T)
```

*In words:* Match the teacher's softened distribution (weight $\alpha$) and the true label (weight $1 - \alpha$).

*Taught in:* [Ch. 49 · Quantisation & distillation](../part3-deep-learning-systems/ch49-quantization-and-distillation/README.md) · [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md)

### Parity criterion

```math
\max_i \left\lvert p^{\text{offline}}_i - p^{\text{API}}_i \right\rvert \le \varepsilon,
\qquad \varepsilon = 10^{-6}
```

*In words:* The largest disagreement must be below what rounding to six decimals can explain ($0.5 \times 10^{-6}$), with a little slack.

*Taught in:* [Ch. 11 · Consistent serving](../part1-foundations/ch11-consistent-serving/README.md) · [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md)

### How a skewed feature moves a score

```math
\Delta \eta = \beta_j\, \delta_j,
\qquad
p_{\text{served}} = \sigma\big(\eta + \beta_j \delta_j\big)
```

*In words:* The error in the feature, times its coefficient, is added straight to the log-odds.

*Taught in:* [Ch. 11 · Consistent serving](../part1-foundations/ch11-consistent-serving/README.md)

### Batching throughput

```math
\text{throughput}(B) = \frac{B}{t(B)}, \qquad \text{latency} \approx t_{\text{wait}} + t(B)
```

*In words:* A batch of $B$ requests finishes together, so throughput rises as long as the call time $t(B)$ grows more slowly than $B$.

*Taught in:* [Ch. 50 · High-throughput serving](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md)

### Arithmetic intensity of a Linear layer

```math
I(B) \approx \frac{2\,B\,d_{\text{in}}\,d_{\text{out}}}{b\,d_{\text{in}}\,d_{\text{out}}} = \frac{2B}{b}\ \ \text{FLOPs per byte}
```

*In words:* Each weight read ($b$ bytes) is used for one multiply-add per request in the batch. It becomes compute-bound when $I(B)$ exceeds the hardware's peak FLOP/s ÷ memory bandwidth.

*Taught in:* [Ch. 50 · High-throughput serving](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md)

### KV-cache size

```math
\text{bytes} = 2 \cdot L \cdot H \cdot d_{\text{head}} \cdot T \cdot B \cdot b
```

*In words:* A key and a value (the 2) per layer, head, channel, position and sequence.

*Taught in:* [Ch. 41 · A transformer from scratch](../part3-deep-learning-systems/ch41-transformer-from-scratch/README.md) · [Ch. 50 · High-throughput serving](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md)

### Percentile

```math
p_q = x_{(\lceil q N / 100 \rceil)}
```

*In words:* The value with $`q\%`$ of observations at or below it. NumPy's default interpolates between neighbours, so the two differ slightly on small samples.

*Taught in:* [Ch. 14 · Health, logs & latency](../part1-foundations/ch14-health-logs-and-latency/README.md)

### Little's law

```math
L = \lambda\, W
```

*In words:* Average requests in flight = arrival rate × average time in the system.

*Taught in:* [Ch. 14 · Health, logs & latency](../part1-foundations/ch14-health-logs-and-latency/README.md) · [Ch. 18 · System design & mock interview](../part1-foundations/ch18-system-design-and-mock-interview/README.md) · [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md) · [Ch. 50 · High-throughput serving](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md)

### Why latency explodes near saturation

```math
W = \frac{S}{1 - \rho}, \qquad \rho = \lambda S
```

*In words:* As utilisation ρ approaches 1, waiting time grows without bound, which is why capacity plans target 50–70% utilisation.

*Taught in:* [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md)

### Single-process ceiling

```math
\lambda_{\max} \approx \frac{1}{t_{\text{CPU per request}}}
```

*In words:* CPU-bound Python serves at most one request per CPU-time per process, however many threads it has.

*Taught in:* [Ch. 14 · Health, logs & latency](../part1-foundations/ch14-health-logs-and-latency/README.md)

### Fan-out and tail latency

```math
P(\text{request waits for a slow shard}) = 1 - (1 - q)^N
```

*In words:* With N parallel calls, each slow with probability q, the request is slow if any one is.

*Taught in:* [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md) · [Ch. 14 · Health, logs & latency](../part1-foundations/ch14-health-logs-and-latency/README.md)

### The latency budget

```math
\sum_{\text{stages}} b_s + b_{\text{outside}} \le \text{SLO}, \qquad p_{99}\Big(\sum_s t_s\Big) \le \sum_s p_{99}(t_s)
```

*In words:* Stage budgets plus slack must fit the SLO; because tail latencies rarely coincide, the p99 of the sum is below the sum of p99s, which makes per-stage budgets conservative.

*Taught in:* [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md) · [Ch. 18 · System design & mock interview](../part1-foundations/ch18-system-design-and-mock-interview/README.md) · [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md)

### Cache hit rate and its effect on latency

```math
h \le 1 - \frac{\text{distinct keys}}{\text{requests}}, \qquad \mathbb{E}[t] = h\, t_{\text{hit}} + (1 - h)\, t_{\text{miss}}
```

*In words:* A cache can serve at most the repeated keys; expected latency mixes hit and miss times.

*Taught in:* [Ch. 35 · Serving a ranking cascade](../part2-ml-at-scale/ch35-serving-a-ranking-cascade/README.md)

### Amdahl's law: what a speed-up is worth end to end

```math
\text{overall speed-up} = \frac{1}{(1 - f) + f / k}
```

*In words:* Speeding up a stage that is a fraction $f$ of the time can never beat $1/(1-f)$ overall.

*Taught in:* [Ch. 36 · Model optimisation for inference](../part2-ml-at-scale/ch36-model-optimization-for-inference/README.md)

### Exponential backoff

```math
d_n = b \cdot 2^{n} \quad (\times\ \text{a random factor, with jitter})
```

*In words:* Double the wait after each failed attempt, with random jitter so clients do not retry in lockstep.

*Taught in:* [Ch. 12 · API clients & tests](../part1-foundations/ch12-api-clients-and-tests/README.md)

### Worst-case time of one logical call

```math
T_{\max} = (R + 1)\,\tau + \sum_{n=0}^{R-1} b\, 2^{n} = (R + 1)\,\tau + b\,(2^{R} - 1)
```

*In words:* Every attempt may hang until the timeout, plus all the waits.

*Taught in:* [Ch. 12 · API clients & tests](../part1-foundations/ch12-api-clients-and-tests/README.md)

### Why retries help with transient failures

```math
P(\text{give up}) = q^{\,R + 1}
```

*In words:* If attempts fail independently with probability $q$, all $R + 1$ fail with probability $q^{R+1}$; during an outage they are not independent.

*Taught in:* [Ch. 12 · API clients & tests](../part1-foundations/ch12-api-clients-and-tests/README.md)

### Capacity: QPS and replicas

```math
\text{QPS}_{\text{avg}} = \frac{\text{DAU} \times r}{86{,}400},
\qquad
\text{QPS}_{\text{peak}} = f \cdot \text{QPS}_{\text{avg}},
\qquad
\text{replicas} = \left\lceil \frac{\text{QPS}_{\text{peak}}}{u \cdot \text{QPS}_{\text{replica}}} \right\rceil
```

*In words:* Average QPS from daily users, times a peak factor, divided by what one replica handles at target utilisation.

*Taught in:* [Ch. 18 · System design & mock interview](../part1-foundations/ch18-system-design-and-mock-interview/README.md) · [Ch. 37 · Recommendation & search system design](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md) · [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md)

### Fleet size with headroom

```math
\text{replicas} = \max\!\left(2,\ \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS}_{\text{replica}} \cdot (1 - h)} \right\rceil \right)
```

*In words:* At least two replicas, and enough to carry peak traffic while keeping a fraction $h$ of each spare.

*Taught in:* [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md) · [Ch. 58 · Full interview loop & portfolio](../part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/README.md)

### Scoring load and servers

```math
\text{rows/s} = \text{QPS} \times \text{candidates scored}, \qquad \text{servers} = \left\lceil \frac{\text{rows/s}}{\text{rows/s per core} \times \text{cores} \times \text{utilisation}} \right\rceil
```

*In words:* Rows scored per second is QPS times candidates; divide by what the fleet scores per second at target utilisation.

*Taught in:* [Ch. 37 · Recommendation & search system design](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md)

### Batch-scoring time

```math
t = \frac{N}{\text{rows per second}}
```

*In words:* If the whole population scores in minutes, a nightly batch job beats an online service.

*Taught in:* [Ch. 18 · System design & mock interview](../part1-foundations/ch18-system-design-and-mock-interview/README.md)

### Embedding and index memory

```math
\text{bytes} = n \times d \times b \quad (\text{flat}), \qquad n\,(d\,b + 2M \cdot 4) \quad (\text{HNSW}), \qquad n\,(\text{code bytes} + 8) \quad (\text{IVF-PQ})
```

*In words:* Flat storage is $n \times d \times$ bytes; HNSW adds graph links; product quantisation compresses each vector to a few bytes.

*Taught in:* [Ch. 37 · Recommendation & search system design](../part2-ml-at-scale/ch37-recsys-and-search-system-design/README.md)

### Embedding-table memory

```math
\text{bytes} = \text{rows} \times \text{dim} \times \text{bytes per value} \times (1 + \text{optimiser slots})
```

*In words:* Embedding tables dominate recommender memory, and Adam's two slots triple them during training.

*Taught in:* [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md)

### Online feature-store size

```math
\text{GB} = \frac{E \cdot (k + F \cdot b) \cdot o \cdot r}{10^9}
```

*In words:* Entities times (key plus features times bytes), times store overhead and replicas.

*Taught in:* [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

### Cost per 1,000 requests

```math
\text{cost}_{1k} = \frac{1000 \cdot p}{3600 \cdot r \cdot u}
```

*In words:* The hourly price divided by the number of requests actually served in that hour.

*Taught in:* [Ch. 50 · High-throughput serving](../part3-deep-learning-systems/ch50-high-throughput-serving/README.md)

### The price of idle capacity

```math
\text{cost per useful GPU-hour} = \frac{\text{price per hour}}{u}
```

*In words:* An accelerator used a fraction $u$ of the time costs $1/u$ times its price per useful hour.

*Taught in:* [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

### Cost per request

```math
\text{API-style: } c = n_{\text{in}}\, p_{\text{in}} + n_{\text{out}}\, p_{\text{out}}, \qquad \text{self-hosted: } c = \frac{p_{\text{hour}}}{3600} \cdot t_{\text{request}}
```

*In words:* API: tokens in and out times their prices; self-hosted: the hourly price times the seconds each request occupies.

*Taught in:* [Ch. 54 · Evaluating LLM systems & guardrails](../part3-deep-learning-systems/ch54-evaluating-llm-systems-and-guardrails/README.md)

### LLM cost per conversation

```math
P_t = S + R + U + (t - 1)(U + A), \qquad \text{cost} = \sum_{t=1}^{n} \big( P_t\, p_{\text{in}} + A\, p_{\text{out}} \big)
```

*In words:* Every turn re-sends the system prompt, retrieved text and the whole history, so input tokens grow with each turn.

*Taught in:* [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md)

### Blended cost with a human fallback

```math
c = c_{\text{LLM}} + (1 - d)\, c_{\text{human}}
```

*In words:* The LLM cost plus the human cost of every conversation it fails to deflect.

*Taught in:* [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md)

### Risk score

```math
\text{risk} = \text{likelihood} \times \text{impact} \qquad (\text{each } 1\text{–}5)
```

*In words:* Rate each risk 1–5 on both axes and deal with the largest products first.

*Taught in:* [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

## Monitoring

### Population Stability Index

```math
\mathrm{PSI} = \sum_{b=1}^{B} (a_b - e_b)\, \ln\frac{a_b}{e_b}
```

*In words:* For each bin, the change in share times the log of the ratio of shares; both factors have the same sign, so every term is $\ge 0$.

*Taught in:* [Ch. 15 · Monitoring & rollback](../part1-foundations/ch15-monitoring-and-rollback/README.md)

### Score-mean ratio (prediction drift)

```math
r = \frac{\bar{s}_{\text{current}}}{\bar{s}_{\text{reference}}}
```

*In words:* A cheap prediction-drift alarm: the current mean score over the reference mean.

*Taught in:* [Ch. 15 · Monitoring & rollback](../part1-foundations/ch15-monitoring-and-rollback/README.md)

### Calibration ratio on a matured cohort

```math
\text{calibration ratio} = \frac{\bar{s}}{\bar{y}}
```

*In words:* Mean predicted over observed rate, once labels have matured; 1 is perfect.

*Taught in:* [Ch. 15 · Monitoring & rollback](../part1-foundations/ch15-monitoring-and-rollback/README.md)

### Rollback rule

```math
\text{roll back if } \text{PR-AUC}_{\text{cand}} < (1 - \delta)\,\text{PR-AUC}_{\text{base}}
```

*In words:* Agree in advance how much worse a candidate may be before it is rolled back automatically.

*Taught in:* [Ch. 15 · Monitoring & rollback](../part1-foundations/ch15-monitoring-and-rollback/README.md)

### Delayed feedback and label maturity

```math
P(y_{\text{obs}} = 1 \mid x, e) = \mathrm{pCVR}(x) \cdot F(e), \qquad F(e) = 1 - e^{-e/\tau}\ \ \text{(exponential delays)}
```

*In words:* A conversion is visible only once it has happened; with exponential delays, a share $1 - e^{-e/\tau}$ is visible after time $e$.

*Taught in:* [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md) · [Ch. 55 · ML platform design](../part3-deep-learning-systems/ch55-ml-platform-design/README.md)

## Ads, CTR and bandits

### Re-calibration after negative downsampling

```math
\frac{q}{1 - q} = \frac{1}{w} \cdot \frac{p}{1 - p} \quad\Longrightarrow\quad p = \frac{q}{q + (1 - q)/w}
```

*In words:* Keeping a random fraction $w$ of negatives divides their count by $1/w$, so the odds the model learns are $1/w$ times the true odds; multiply them back by $w$.

*Taught in:* [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md)

### Hash collisions

```math
P(\text{a key shares its bucket}) = 1 - \left(1 - \tfrac{1}{n}\right)^{k - 1} \approx 1 - e^{-k/n}
```

*In words:* With $k$ keys hashed into $n$ buckets, a key shares its bucket with probability about $1 - e^{-k/n}$.

*Taught in:* [Ch. 51 · CTR prediction at scale](../part3-deep-learning-systems/ch51-ctr-prediction-at-scale/README.md)

### ESMM

```math
\mathrm{pCTCVR}(x) = \mathrm{pCTR}(x) \times \mathrm{pCVR}(x), \qquad \mathcal{L} = \mathrm{BCE}(\mathrm{pCTR}, \text{click}) + \mathrm{BCE}(\mathrm{pCTCVR}, \text{click} \wedge \text{conv})
```

*In words:* Both losses are defined on **every** impression; pCVR is never fitted on clicked rows alone.

*Taught in:* [Ch. 52 · Multi-task ranking & exploration](../part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md)

### Blended ranking score

```math
\text{score} = \mathrm{pCTR}^{a} \cdot \mathrm{pCVR}^{b} \cdot \text{value}
```

*In words:* With calibrated predictions and $a = b = 1$, the score is expected value per impression.

*Taught in:* [Ch. 52 · Multi-task ranking & exploration](../part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md)

### eCPM and the second price

```math
\mathrm{eCPM}_i = \mathrm{pCTR}_i \cdot \text{bid}_i \cdot 1000, \qquad \text{price per click} = \max\!\left(\text{reserve},\ \frac{\text{bid}_2 \cdot \mathrm{pCTR}_2}{\mathrm{pCTR}_1}\right) \le \text{bid}_1
```

*In words:* Rank by expected revenue per 1,000 impressions; the winner pays per click the smallest bid that would still beat the runner-up.

*Taught in:* [Ch. 52 · Multi-task ranking & exploration](../part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md) · [Ch. 56 · Large-scale design cases](../part3-deep-learning-systems/ch56-large-scale-design-cases/README.md)

### Revenue equivalence

```math
\mathbb{E}[\text{revenue}] = \frac{n-1}{n+1} \qquad (n \text{ bidders, values i.i.d. } U[0, 1])
```

*In words:* With $n$ bidders and uniform values, first-price and second-price auctions earn the same expected revenue.

*Taught in:* [Ch. 52 · Multi-task ranking & exploration](../part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md)

### Bandits: regret, UCB1 and Thompson sampling

```math
\mathrm{Regret}(T) = \sum_{t=1}^{T} \big( p^{*} - p_{a_t} \big), \qquad a_t^{\mathrm{UCB1}} = \arg\max_k \left( \hat{p}_k + \sqrt{\frac{2 \ln t}{n_k}} \right)
```

```math
\theta_k \sim \mathrm{Beta}(\alpha_k, \beta_k), \quad a_t = \arg\max_k \theta_k, \qquad \alpha_{a_t} \mathrel{+}= r_t, \quad \beta_{a_t} \mathrel{+}= 1 - r_t
```

*In words:* Regret counts the clicks lost by not always playing the best arm; UCB1 adds an optimism bonus that shrinks with plays; Thompson sampling draws from each arm's Beta posterior and plays the best draw.

*Taught in:* [Ch. 52 · Multi-task ranking & exploration](../part3-deep-learning-systems/ch52-multitask-ranking-and-exploration/README.md)

## Privacy and fairness

### Bias AUCs (subgroup, BPSN, BNSP)

```math
\mathrm{AUC}(P, N) = \Pr\big(s(x^{+}) > s(x^{-})\big), \quad x^{+} \sim P,\ x^{-} \sim N \quad (\text{ties count } \tfrac12)
```

```math
\begin{aligned}
\text{Subgroup AUC} &= \mathrm{AUC}(\text{SP},\ \text{SN}) \\
\text{BPSN AUC} &= \mathrm{AUC}(\text{BP},\ \text{SN}) \\
\text{BNSP AUC} &= \mathrm{AUC}(\text{SP},\ \text{BN})
\end{aligned}
```

*In words:* AUC is the chance a positive outscores a negative; the bias AUCs restrict the pairs to mixes of subgroup and background examples.

*Taught in:* [Ch. 53 · Fairness, robustness & privacy](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md)

### Error rates per group at threshold $t$

```math
\mathrm{FPR}_g(t) = \Pr(s \ge t \mid y = 0, g), \qquad \mathrm{FNR}_g(t) = \Pr(s < t \mid y = 1, g)
```

*In words:* False-positive and false-negative rates computed separately per group, at the same threshold.

*Taught in:* [Ch. 53 · Fairness, robustness & privacy](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md)

### Power mean over groups

```math
M_p(m_1, \dots, m_G) = \Big(\frac{1}{G} \sum_{g=1}^{G} m_g^{\,p}\Big)^{1/p}, \qquad p = -5
```

*In words:* A mean that, for negative $p$, is pulled towards the worst group.

*Taught in:* [Ch. 53 · Fairness, robustness & privacy](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md)

### Differential privacy

```math
\Pr[A(D) \in S] \le e^{\varepsilon}\, \Pr[A(D') \in S] + \delta \quad \text{for all neighbouring } D, D' \text{ and all } S
```

*In words:* Adding or removing one record changes the probability of any outcome by at most a factor $e^{\varepsilon}$ (plus a small slack $\delta$).

*Taught in:* [Ch. 53 · Fairness, robustness & privacy](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md)

### One DP-SGD step

```math
\bar{g}_i = g_i \cdot \min\!\left(1, \frac{C}{\lVert g_i \rVert_2}\right), \qquad \tilde{g} = \frac{1}{n}\left( \sum_{i=1}^{n} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) \right), \qquad \theta \leftarrow \theta - \eta\, \tilde{g}
```

*In words:* Clip each example's gradient to norm $C$, add Gaussian noise scaled to $C$, then average and step.

*Taught in:* [Ch. 53 · Fairness, robustness & privacy](../part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/README.md)

## Coding interviews: complexity

### Reading growth from timings

```math
k \approx \frac{\log(t_2 / t_1)}{\log(n_2 / n_1)}
```

*In words:* The local exponent that `growth_table` prints: if doubling n multiplies the time by 4, the algorithm is about O(n²); by 2, O(n).

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md)

### Input size → the complexity you need

```math
n \le 20 \Rightarrow O(2^n) \qquad n \le 10^3\text{–}10^4 \Rightarrow O(n^2) \qquad n \approx 10^5\text{–}10^6 \Rightarrow O(n \log n) \text{ or } O(n)
```

*In words:* Pure Python does about $10^7$ simple steps per second, so $n$ tells you which complexity you can afford.

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md) · [Ch. 17 · Python & SQL drills](../part1-foundations/ch17-python-and-sql-drills/README.md)

### Why the sliding window is O(n)

```math
\text{total steps} = \underbrace{n}_{\text{right moves}} + \underbrace{\le n}_{\text{left moves}} \le 2n
```

*In words:* Count pointer moves, not loop nesting. A `while` inside a `for` is not automatically quadratic.

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md)

### Heap top-k versus sorting

```math
O(n \log k) \ \text{vs}\ O(n \log n)
```

*In words:* A size-$k$ heap keeps the top $k$ of a stream in $O(n \log k)$ time and $O(k)$ memory.

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md)

### Prefix sums and "subarray sum equals k"

```math
\sum_{t=i}^{j} x_t = P_{j+1} - P_i, \qquad P_0 = 0, \qquad \#\{\text{subarrays with sum } k\} = \sum_{j} \#\{\, i \le j : P_i = P_{j+1} - k \,\}
```

*In words:* Any subarray sum is a difference of two prefix sums, so count earlier prefixes equal to the current one minus $k$.

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md)

### Output-sensitive bounds

```math
O(n + p), \qquad p \le \binom{n}{2} = \frac{n(n-1)}{2}
```

*In words:* If the answer itself can have $p$ items, no algorithm beats $O(n + p)$.

*Taught in:* [Ch. 33 · Coding patterns I](../part2-ml-at-scale/ch33-coding-patterns-1/README.md)

### Graph traversal cost

```math
T_{\text{BFS}} = T_{\text{DFS}} = O(V + E)
```

*In words:* Each node is enqueued (or pushed) once and each edge is looked at once (twice if undirected). A grid of R × C cells has V = RC and E ≤ 4RC.

*Taught in:* [Ch. 34 · Coding patterns II](../part2-ml-at-scale/ch34-coding-patterns-2/README.md)

### Dijkstra's relaxation

```math
d(v) \leftarrow \min\big(d(v),\ d(u) + w(u, v)\big), \qquad T = O\big((V + E) \log V\big) \text{ with a binary heap}
```

*In words:* When you pop the closest unfinished node u, try every edge out of it; correct because, with $w \ge 0$, nothing popped later can be closer.

*Taught in:* [Ch. 34 · Coding patterns II](../part2-ml-at-scale/ch34-coding-patterns-2/README.md)

### Binary search iterations

```math
\text{iterations} = \lceil \log_2(\text{hi} - \text{lo} + 1) \rceil
```

*In words:* Each check halves the range of candidates.

*Taught in:* [Ch. 34 · Coding patterns II](../part2-ml-at-scale/ch34-coding-patterns-2/README.md)

### The DP recurrences in this notebook

```math
\text{coins}(a) = 1 + \min_{c \le a} \text{coins}(a - c), \qquad
D(i, j) = \min\big(D(i-1, j) + 1,\ D(i, j-1) + 1,\ D(i-1, j-1) + \mathbb{1}[a_i \ne b_j]\big)
```

*In words:* The best answer for a state is built from the best answers of smaller states; the base cases are $\text{coins}(0) = 0$, $D(i, 0) = i$, $D(0, j) = j$.

*Taught in:* [Ch. 34 · Coding patterns II](../part2-ml-at-scale/ch34-coding-patterns-2/README.md)

### Union-find

```math
T(\text{m operations on n elements}) = O\big(m\, \alpha(n)\big), \qquad \alpha(n) \le 4 \text{ for any } n \text{ you will meet}
```

*In words:* With path compression and union by size each operation is effectively constant time; with neither, a chain makes `find` O(n).

*Taught in:* [Ch. 34 · Coding patterns II](../part2-ml-at-scale/ch34-coding-patterns-2/README.md)

---

See also: the [glossary](glossary.md) for definitions, the [interview question bank](interview-questions.md) for practice, and the [diagram index](figures.md) for every figure.
