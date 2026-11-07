# The equation sheet

Every key equation in the course, in one place, grouped by topic. Each one is taught (with a worked example and usually a code cell that checks it) in the notebook linked under it. The [glossary](glossary.md) defines the words; this page holds the maths.

## How to use this sheet in an interview

An equation recited from memory proves little. Interviewers want to see that you know what it *means* and *when it is wrong*. For any equation here:

1. **Say it in words first.** "NDCG is the discounted gain of the list, divided by the gain of the best possible order." Then write the symbols.
2. **State the assumptions.** Independent rows? Calibrated probabilities? Exponential delays? Most interview follow-ups attack an assumption, not the algebra.
3. **Give a one-number example.** "With a 5% base rate, a constant score has PR-AUC 0.05, so 0.15 is three times the floor." A number shows you have used it.
4. **Say what you would do with it.** Pick a threshold, size a fleet, decide whether a test has enough users.

Each entry below has the same shape: name, the equation, one line *in words*, and where it is taught (L1 = Level 1, L2 = Level 2, L3 = Level 3, then the notebook number).

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

$$
\hat{\pi} = \frac{1}{N}\sum_{i=1}^{N} y_i
$$

*In words:* The share of rows whose label is 1; every ranking metric's floor depends on it.

*Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/03_sql_for_ml.ipynb)

### Precision, recall, F1

$$
P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}
$$

*In words:* How pure the flagged set is, how much of the target it covers, and their harmonic mean.

*Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb) · [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb)

### F1 and macro-F1

$$
F_1 = \frac{2PR}{P + R}, \qquad \text{macro-}F_1 = \frac{1}{C} \sum_{c=1}^{C} F_{1,c}
$$

*In words:* The harmonic mean of precision $P$ and recall $R$, averaged over classes with equal weight.

*Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)

### Precision at k

$$
\text{precision@}k = \frac{1}{k} \sum_{i \in \text{top-}k} y_i
$$

*In words:* Of the $k$ highest-scored rows, the share that are positive; fits an action with fixed capacity.

*Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb) · [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/15_final_evaluation_and_docs.ipynb)

### Lift at k

$$
\operatorname{lift@}k = \frac{\operatorname{precision@}k}{\pi}
$$

*In words:* How many times better than random the top of the list is.

*Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb) · [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/08_error_analysis_and_artifacts.ipynb)

### ROC-AUC as a probability

$$
\text{ROC-AUC} = P\left(s^{+} > s^{-}\right) + \tfrac{1}{2} P\left(s^{+} = s^{-}\right)
$$

*In words:* The chance that a random positive outscores a random negative.

*Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/04_leakage_safe_splits_and_baseline.ipynb)

### ROC-AUC from ranks

$$
\text{AUC} = \frac{\sum_{i \in \text{pos}} r_i - n_+(n_+ + 1)/2}{n_+\, n_-}
$$

*In words:* The fraction of (positive, negative) pairs ordered correctly, read off the positives' ranks (average ranks for ties).

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### Average precision (PR-AUC)

$$
\text{AP} = \sum_{n} (R_n - R_{n-1})\, P_n = \frac{1}{N_{+}} \sum_{k:\ y_{(k)} = 1} \text{precision@}k
$$

*In words:* Average the precision at the rank of every positive.

*Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/04_leakage_safe_splits_and_baseline.ipynb)

### No-skill floors

$$
\text{AP}_{\text{constant}} = \pi, \qquad \text{ROC-AUC}_{\text{constant}} = 0.5
$$

*In words:* A constant score has PR-AUC equal to the positive rate and ROC-AUC 0.5, so always quote PR-AUC next to $\pi$.

*Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/04_leakage_safe_splits_and_baseline.ipynb)

### Lift-normalised slice PR-AUC

$$
\text{normalised PR-AUC}_s = \frac{\text{PR-AUC}_s}{\pi_s}
$$

*In words:* Divide each slice's PR-AUC by its own positive rate before comparing slices.

*Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/08_error_analysis_and_artifacts.ipynb)

### Log loss (binary cross-entropy)

$$
\text{LogLoss} = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]
$$

*In words:* The average surprise of the true labels; a confident wrong answer is punished hard. It is what logistic regression minimises, which is why it tends to come out calibrated.

*Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb) · [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb)

### Normalised entropy

$$
\mathrm{NE} = \frac{\mathrm{LL}}{H(\bar{p})}, \qquad H(\bar{p}) = -\big[\, \bar{p} \log \bar{p} + (1 - \bar{p}) \log (1 - \bar{p}) \,\big]
$$

*In words:* Log loss divided by the log loss of always predicting the base rate; below 1 means the model beats the prior.

*Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb)

### Brier score

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2
$$

*In words:* Mean squared error of the probabilities; the base-rate constant scores $\pi(1-\pi)$.

*Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb)

### Expected calibration error

$$
\operatorname{ECE} = \sum_{b=1}^{B} \frac{\lvert S_b \rvert}{n} \, \big\lvert \operatorname{acc}(S_b) - \operatorname{conf}(S_b) \big\rvert
$$

*In words:* Bin predictions by confidence, and average the gap between accuracy and confidence, weighted by how many predictions fall in each bin.

*Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)

### Temperature scaling

$$
\mathbf p = \operatorname{softmax}(\mathbf z / T), \qquad T^{\star} = \arg\min_{T > 0} \operatorname{NLL}_{\text{val}}(T)
$$

*In words:* Divide every logit by one scalar $T$ chosen on validation data. $T > 1$ softens, $T < 1$ sharpens, the argmax never moves.

*Taught in:* [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)

### Expected value of a threshold, and the break-even probability

$$
\text{EV}(t) = TP(t)\cdot V - FP(t)\cdot C,
\qquad
p^{*} = \frac{C}{V + C}
$$

*In words:* Calling an account with probability $p$ is worth $pV - (1-p)C$, positive exactly when $p > p^{*}$.

*Taught in:* [L1 · 07 Thresholds & calibration](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/07_thresholds_and_calibration.ipynb)

### Spearman rank correlation

$$
\rho_s = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)}
$$

*In words:* The Pearson correlation of the ranks; 1 means both models order every item identically.

*Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb)

### LLM-judge position bias and attack success rate

$$
\text{bias rate} = \frac{1}{n} \sum_{i} \mathbb{1}\big[\, v_i^{(X,Y)} = v_i^{(Y,X)} \,\big], \qquad \mathrm{ASR} = \frac{\#\text{injections that got the attacker's value}}{\#\text{injections}}
$$

*In words:* A judge that picks the same slot in both orders is following position, not quality; ASR is the share of prompt injections that worked.

*Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb)

## Probability, statistics and experimentation

### Churn compounds

$$
\text{retained}(n) = (1 - c)^n
$$

*In words:* Each month keeps a fraction $1 - c$ of whoever is left, so the fractions multiply.

*Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/01_reproducible_project.ipynb)

### Across-seed spread

$$
\bar{x} = \frac{1}{k}\sum_{i=1}^{k} x_i, \qquad
s = \sqrt{\frac{1}{k-1}\sum_{i=1}^{k} (x_i - \bar{x})^2}
$$

*In words:* $s$ is the typical distance of one seed's score from the average.

*Taught in:* [L1 · 01 A reproducible ML project](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/01_reproducible_project.ipynb)

### Standard error of a proportion

$$
\operatorname{SE}(\hat p) = \sqrt{\frac{\hat p\,(1 - \hat p)}{n}},
\qquad
n \approx \frac{1.96^2\, p\,(1-p)}{m^2}
$$

*In words:* The noise in a rate shrinks with $\sqrt{n}$; the second formula is the sample size for a 95% margin of $\pm m$.

*Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb) · [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/01_event_logs_and_columnar_data.ipynb) · [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)

### Interval width from sample size

$$
\text{CI width} \propto \frac{1}{\sqrt{n}}
$$

*In words:* Four times the data halves the interval; for a slice, $n$ is its number of positives (or customers).

*Taught in:* [L1 · 08 Error analysis & artifacts](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/08_error_analysis_and_artifacts.ipynb) · [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/15_final_evaluation_and_docs.ipynb)

### Bootstrap percentile interval

$$
\text{CI}_{95\%} = \left[\, Q_{0.025}\big(\hat\theta^{*}_{1..B}\big),\; Q_{0.975}\big(\hat\theta^{*}_{1..B}\big) \,\right]
$$

*In words:* The middle 95% of the metric across resamples.

*Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb)

### Paired bootstrap for a difference

$$
d_i = s_i^{B} - s_i^{A}, \qquad \bar{d}^{\,*}_b = \frac{1}{n} \sum_{i \in \text{resample}_b} d_i, \qquad \mathrm{CI}_{95\%} = \big[\, q_{2.5\%}(\bar{d}^{\,*}),\ q_{97.5\%}(\bar{d}^{\,*}) \,\big]
$$

*In words:* Score both models on the same resampled items, take the difference each time, and read off the interval of the differences.

*Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb) · [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/08_contrastive_training_and_reranking.ipynb) · [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb)

### Pre-registered decision rule

$$
\text{report "better"} \iff Q_{0.025}\big(\Delta^{*}_{1..B}\big) > 0
$$

*In words:* The whole 95% interval of the difference must sit above zero.

*Taught in:* [L1 · 15 Final evaluation & documentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/15_final_evaluation_and_docs.ipynb)

### Why copied rows do not help

$$
\operatorname{SE}_{\text{row}}(k \text{ copies}) = \frac{\operatorname{SE}_{\text{row}}}{\sqrt{k}}, \qquad \operatorname{SE}_{\text{customer}}(k \text{ copies}) = \operatorname{SE}_{\text{customer}}
$$

*In words:* Duplicating rows shrinks a row bootstrap's SE by $\sqrt{k}$ without adding information; a customer-level bootstrap is unchanged.

*Taught in:* [L1 · 06 Model selection](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/06_model_selection.ipynb)

### Grouped bootstrap and the design effect

$$
\operatorname{Var}_{\text{grouped}} \approx \operatorname{Var}_{\text{iid}} \cdot \big(1 + (m - 1)\,\rho_{\text{ICC}}\big)
$$

*In words:* With $m$ correlated rows per user, a row-level bootstrap underestimates the variance by this design effect; resampling users gets it right automatically.

*Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/20_final_evaluation_and_mock_loop.ipynb)

### Paired differences

$$
\operatorname{Var}(A - B) = \operatorname{Var}(A) + \operatorname{Var}(B) - 2\operatorname{Cov}(A, B)
$$

*In words:* Scoring two systems on the same requests makes their errors correlated, and the covariance term removes the shared part.

*Taught in:* [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/20_final_evaluation_and_mock_loop.ipynb)

### Two-proportion z-test

$$
z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p\,(1-\hat p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}, \qquad \hat p = \frac{x_A + x_B}{n_A + n_B}
$$

*In words:* The observed difference in rates, measured in standard errors computed as if both arms shared one rate (the null hypothesis).

*Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/11_ab_testing_fundamentals.ipynb)

### Standard error of a difference, and its confidence interval

$$
\operatorname{SE}(\bar y_B - \bar y_A) = \sqrt{\frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B}}, \qquad \text{95\% CI} = (\bar y_B - \bar y_A) \pm 1.96\,\operatorname{SE}
$$

*In words:* Variances of independent estimates add, so the SE of a difference is the root of the summed squared SEs.

*Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/11_ab_testing_fundamentals.ipynb)

### Sample size per arm

$$
n \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\, p(1-p)}{\delta^2}
$$

*In words:* (how many standard errors of separation you need)² × (noise per unit) ÷ (effect)². The 2 is there because both arms are noisy.

*Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/11_ab_testing_fundamentals.ipynb) · [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

### Sample-ratio mismatch: chi-square goodness of fit

$$
\chi^2 = \sum_{\text{arms}} \frac{(O - E)^2}{E}, \qquad \text{1 degree of freedom for two arms}
$$

*In words:* How far the observed arm counts are from the planned split, scaled by the expected counts.

*Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/11_ab_testing_fundamentals.ipynb)

### Multiple testing: Bonferroni and Benjamini–Hochberg

$$
\text{Bonferroni: reject } H_i \text{ if } p_i \le \frac{\alpha}{m}; \qquad \text{BH: } k = \max\Big\{ i : p_{(i)} \le \frac{i}{m}\,q \Big\}, \text{ reject } p_{(1)}, \dots, p_{(k)}
$$

*In words:* Bonferroni splits α evenly over m tests (controls the chance of **any** false positive); BH sorts the p-values and compares each with a rising threshold (controls the **share** of false discoveries). With m independent null tests, $P(\text{any false positive}) = 1 - (1-\alpha)^m$.

*Taught in:* [L2 · 11 A/B testing fundamentals](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/11_ab_testing_fundamentals.ipynb)

### CUPED adjustment and its variance

$$
y^{cv}_i = y_i - \theta\,(x_i - \bar x), \qquad \theta = \frac{\operatorname{cov}(x, y)}{\operatorname{var}(x)}, \qquad \operatorname{Var}(y^{cv}) = \operatorname{Var}(y)\,(1 - \rho^2)
$$

*In words:* Regress the metric on its pre-period value and keep the residual (plus the mean). Because $x$ is fixed before randomisation, the difference between arms is unchanged; only the noise shrinks, by the factor $1 - \rho^2$, and so does the sample size you need.

*Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/12_advanced_experimentation.ipynb)

### Delta method for a ratio of means

$$
\operatorname{Var}\!\left(\frac{\bar c}{\bar n}\right) \approx \frac{1}{N}\left( \frac{\sigma_c^2}{\mu_n^2} - \frac{2\,\mu_c\,\sigma_{cn}}{\mu_n^3} + \frac{\mu_c^2\,\sigma_n^2}{\mu_n^4} \right)
$$

*In words:* Linearise $R = \mu_c / \mu_n$ around the means; the variance of the linear approximation is $\nabla R^\top \Sigma\, \nabla R / N$ with gradient $\nabla R = (1/\mu_n,\ -\mu_c/\mu_n^2)$.

*Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/12_advanced_experimentation.ipynb)

### The interleaving statistic

$$
\Delta = \frac{1}{U} \sum_{u=1}^{U} \left( \text{clicks}_{A,u} - \text{clicks}_{B,u} \right), \qquad t = \frac{\Delta}{s_\Delta / \sqrt{U}}
$$

*In words:* Per user, credited clicks for A minus credited clicks for B; a one-sample t-test asks whether the mean difference is zero.

*Taught in:* [L2 · 12 Advanced experimentation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/12_advanced_experimentation.ipynb)

### What row sampling does to a per-request metric

$$
P(\text{click kept} \mid \text{request kept}) = \frac{\mathbb{E}\left[1-(1-f)^{c}\right]}{1-(1-f)^{10}}
$$

*In words:* A request with $c$ clicks keeps at least one of them with probability $1-(1-f)^c$, and survives at all with probability $1-(1-f)^{10}$.

*Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/01_event_logs_and_columnar_data.ipynb)

### A hash sample's size

$$
\mathbb{E}[\text{kept}] = N q, \qquad \operatorname{SD}[\text{kept}] = \sqrt{N q (1-q)}
$$

*In words:* Each key is kept with probability $q$, independently of its id, so the sample size is binomial.

*Taught in:* [L2 · 01 Event logs & columnar data](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/01_event_logs_and_columnar_data.ipynb)

### Reservoir sampling (Algorithm R)

$$
P(\text{item } i \text{ in sample after } n \text{ items}) = \frac{k}{n}
$$

*In words:* After reading $n$ items, every one of them is in the size-$k$ reservoir with the same probability.

*Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/02_aggregation_and_joins_at_scale.ipynb)

## Data, SQL and features

### Label

$$
y_i(T) = \mathbb{1}\left[\, T \le \text{churn\_date}_i < T + 30\ \text{days} \,\right]
$$

*In words:* 1 if the account cancels in the 30 days starting at T, else 0.

*Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/03_sql_for_ml.ipynb)

### Completed-week rule

$$
w + 7\ \text{days} \le T
$$

*In words:* The week must have ended by T.

*Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/03_sql_for_ml.ipynb)

### Embargo condition

$$
T + h \le t_v
$$

*In words:* The label must be settled before the evaluation period starts.

*Taught in:* [L1 · 04 Leakage-safe splits & a baseline](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/04_leakage_safe_splits_and_baseline.ipynb)

### Fan-out row count

$$
\text{rows after join} = \sum_{c} n_U(c)\, n_K(c)
\quad\Rightarrow\quad
\text{SUM}_{\text{joined}}(x) = \sum_{c} n_K(c) \sum_{u \in U_c} x_u
$$

*In words:* Each key contributes the product of its row counts, so every value from $U$ is counted once per matching row of $K$.

*Taught in:* [L1 · 03 SQL for ML](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/03_sql_for_ml.ipynb) · [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/02_aggregation_and_joins_at_scale.ipynb)

### Cohort retention

$$
\text{ret}_k(c) = \frac{\lvert\{\text{customers of cohort } c \text{ active in month } c + k\}\rvert}{\lvert \text{cohort } c \rvert}
$$

*In words:* The share of a sign-up cohort still active $k$ months later.

*Taught in:* [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb)

### A window feature, cut off strictly before the snapshot

$$
x_u(d) = \sum_{t=d-7}^{d-1} \text{clicks}_u(t)
$$

*In words:* The value stamped at day $d$ sums the 7 *complete* days before $d$.

*Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb)

### The point-in-time rule, with a TTL

$$
s^{*}(e) = \underset{s:\; k_s = k_e,\; \tau_s < t_e}{\arg\max}\; \tau_s, \qquad \text{use } s^{*} \text{ only if } t_e - \tau_{s^{*}} \le \text{TTL}
$$

*In words:* Take the latest snapshot strictly before the event, and use it only if it is young enough.

*Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb)

### Feature age at serving

$$
\text{age} = t_\text{request} - t_\text{newest event in the value}
$$

*In words:* What the model sees online is as old as the newest data behind it.

*Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb)

### A smoothed click rate

$$
\widetilde{\text{CTR}} = \frac{\text{clicks} + \alpha\, p_0}{\text{impressions} + \alpha}
$$

*In words:* Add $\alpha$ pseudo-impressions at the prior rate, so tiny counts do not produce extreme rates.

*Taught in:* [L2 · 03 Feature pipelines & feature stores](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/03_feature_pipelines_and_feature_stores.ipynb)

### Target encoding

$$
\operatorname{TE}(c) = \frac{1}{n_c} \sum_{i \in c} y_i
$$

*In words:* The mean label of the rows in that level. It is honest only if the rows $i$ are strictly earlier, with closed label windows, and never include the row being scored.

*Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb)

### Standardisation

$$
z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
$$

*In words:* How many training standard deviations a value sits from the training mean.

*Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb)

### log1p transform

$$
\tilde{x} = \ln(1 + x)
$$

*In words:* Turns "per unit" into "roughly per doubling", and maps 0 to 0.

*Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb)

### Partitioning by hash

$$
\text{reducer}(k) = h(k) \bmod R
$$

*In words:* A key always goes to the same reducer, and keys spread evenly *if* no key is hot.

*Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/02_aggregation_and_joins_at_scale.ipynb)

### HyperLogLog relative error

$$
\frac{\operatorname{SE}(\hat n)}{n} \approx \frac{1.04}{\sqrt{m}}, \qquad m = 2^{p}
$$

*In words:* The standard error of a distinct count shrinks with the square root of the number of registers.

*Taught in:* [L2 · 02 Aggregation & joins at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/02_aggregation_and_joins_at_scale.ipynb)

### Which tumbling window an event belongs to

$$
\text{start}(t) = \left\lfloor \frac{t}{w} \right\rfloor w
$$

*In words:* Round the event time down to a multiple of the window size.

*Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/04_streaming_features.ipynb)

### Sliding windows: how many contain one event

$$
\{\, s = \lfloor t/h \rfloor h - j h \;:\; j = 0, \dots, S/h - 1 \,\}
$$

*In words:* With size $S$ and hop $h$, each event falls in $S/h$ windows, so the state does $S/h$ times the updates.

*Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/04_streaming_features.ipynb)

### Watermark, closing and lateness

$$
W = \max_{\text{seen}} t - L, \qquad \text{close } [s, s+w) \text{ when } W \ge s + w, \qquad \text{late if } s + w \le W_\text{before}
$$

*In words:* The watermark trails the largest event time seen; a window is emitted once the watermark passes its end; an event whose window has already closed is late.

*Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/04_streaming_features.ipynb)

### Session windows

$$
t \text{ joins } [a, b] \iff a - g \le t \le b + g
$$

*In words:* An event joins (and can bridge) every session within the gap $g$.

*Taught in:* [L2 · 04 Streaming features](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/04_streaming_features.ipynb)

## Linear models and optimisation

### Logistic regression

$$
\log\frac{p}{1-p} = \beta_0 + \sum_j \beta_j z_j,
\qquad
p = \sigma(\eta) = \frac{1}{1 + e^{-\eta}}
$$

*In words:* Each standardised feature adds $\beta_j z_j$ to the log-odds of churn.

*Taught in:* [L1 · 05 Pipelines & cross-validation](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/05_pipelines_and_cross_validation.ipynb)

### Logistic regression: loss and gradient

$$
L(w, b) = \frac{1}{n}\sum_{i=1}^n \big[\log(1 + e^{z_i}) - y_i z_i\big] + \frac{\lambda}{2}\lVert w\rVert^2, \qquad \nabla_w L = \frac{1}{n} X^\top(p - y) + \lambda w, \quad \frac{\partial L}{\partial b} = \overline{p - y}
$$

*In words:* The log loss written stably in terms of logits; its gradient is "prediction minus target", pushed back through the features.

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### Softmax cross-entropy gradient

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\log p_y, \qquad \frac{\partial L}{\partial z_k} = p_k - \mathbb{1}[k = y]
$$

*In words:* The gradient for each class is its predicted probability, minus 1 for the true class. Confidently wrong → large gradient; confidently right → almost zero. For a batch, $dW = X^\top (P - Y) / n$.

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### Log-sum-exp

$$
\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}, \qquad m = \max_j z_j
$$

*In words:* Shift by the maximum before exponentiating; the answer is unchanged and nothing overflows. Cross-entropy from logits is $\operatorname{logsumexp}(z) - z_y$.

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### Minimum-norm least squares and ridge

$$
\hat{w}_{\text{min-norm}} = X^{+} y = \arg\min_w \lVert w \rVert_2 \ \text{ s.t. } Xw = y \ \ (p > n), \qquad \hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

*In words:* When there are more features than points, infinitely many $w$ fit exactly; the pseudo-inverse picks the shortest. Near $p = n$ the smallest singular value $\sigma_{\min}$ of $X$ is tiny and $\lVert \hat{w} \rVert$ scales like $1/\sigma_{\min}$: the peak. Ridge's $\lambda$ keeps $1/(\sigma^2 + \lambda)$ bounded.

*Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/19_ml_depth_and_research_discussion.ipynb)

### k-means objective (inertia)

$$
J = \sum_{i=1}^n \lVert x_i - \mu_{c(i)} \rVert^2, \qquad c(i) = \arg\min_k \lVert x_i - \mu_k \rVert, \qquad \mu_k = \operatorname{mean}\{x_i : c(i) = k\}
$$

*In words:* Assign each point to its nearest centre, move each centre to the mean of its points; each step can only lower $J$.

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### Two tools you use to verify everything else

$$
\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2\,a^\top b + \lVert b \rVert^2, \qquad \frac{\partial f}{\partial w_i} \approx \frac{f(w + \varepsilon e_i) - f(w - \varepsilon e_i)}{2\varepsilon}
$$

*In words:* The distance expansion turns pairwise distances into one matrix product; the central difference has error O(ε²) and is the reference for every gradient check.

*Taught in:* [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb)

### The AdamW update

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \lambda\, \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
\end{aligned}
$$

*In words:* Keep running averages of the gradient and of its square, correct their start-up bias, step by their ratio, and shrink the weights separately (decoupled weight decay).

*Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/09_pytorch_training_done_right.ipynb) · [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/02_training_efficiency_and_memory.ipynb)

### Linear warmup, then cosine decay

$$
\eta_t = \eta_{\max} \cdot \begin{cases} \dfrac{t+1}{T_w} & t < T_w \\[6pt] r + (1-r)\,\dfrac{1}{2}\Big(1 + \cos\big(\pi\, \tfrac{t - T_w}{T - T_w}\big)\Big) & t \ge T_w \end{cases}
$$

*In words:* Ramp up over $T_w$ steps, then follow half a cosine down to a floor.

*Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/09_pytorch_training_done_right.ipynb) · [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/19_ml_depth_and_research_discussion.ipynb)

### Learning rate vs batch size

$$
\eta(kB) = k\, \eta(B) \ \ \text{(linear, SGD)}, \qquad \eta(kB) = \sqrt{k}\, \eta(B) \ \ \text{(square-root, more conservative)}
$$

*In words:* A bigger batch has less gradient noise, so it tolerates a larger step: linear scaling for SGD, square-root as the safer rule.

*Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/19_ml_depth_and_research_discussion.ipynb)

### Gradient-norm clipping

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{c}{\lVert \mathbf{g} \rVert_2}\right)
$$

*In words:* If the whole gradient is longer than $c$, shrink it to length $c$; its direction does not change.

*Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/09_pytorch_training_done_right.ipynb)

## Trees and boosting

### SSE from sums, and the split gain

$$
\text{SSE} = \sum (y - \bar y)^2 = S_2 - \frac{S_1^2}{m}, \qquad \text{gain} = \text{SSE}_{\text{parent}} - \text{SSE}_{\text{left}} - \text{SSE}_{\text{right}}
$$

*In words:* A group's spread needs only its count, sum and sum of squares; the best split is the one that removes the most spread (variance reduction).

*Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

### Gradient boosting update

$$
r_i^{(m)} = -\frac{\partial L(y_i, F)}{\partial F}\Big|_{F = F_{m-1}(x_i)}, \qquad F_m(x) = F_{m-1}(x) + \nu\, h_m(x), \qquad F_0 = \arg\min_c \sum_i L(y_i, c)
$$

*In words:* Fit a small tree $h_m$ to the negative gradient (for $L = \tfrac12 (y-F)^2$, the residual $y - F$) and add a shrunken copy.

*Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

## Ranking, retrieval and recommendation

### DCG and NDCG

$$
\operatorname{DCG@}k = \sum_{i=1}^{k} \frac{2^{\,\text{rel}_i} - 1}{\log_2(i + 1)}, \qquad \operatorname{NDCG@}k = \frac{\operatorname{DCG@}k}{\operatorname{IDCG@}k}
$$

*In words:* Add up each item's gain, discounted by the log of its rank, and divide by the best possible order's value.

*Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/06_learning_to_rank.ipynb) · [L2 · 13 ML from scratch I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/13_ml_from_scratch_1.ipynb) · [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/07_embeddings_and_semantic_search.ipynb)

### MRR

$$
\operatorname{MRR} = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

*In words:* The average, over requests, of one over the rank of the first relevant item.

*Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/06_learning_to_rank.ipynb)

### HR@10 and NDCG@10 with one target

$$
\operatorname{HR@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \mathbf{1}[r_u < 10], \qquad \operatorname{NDCG@}10 = \frac{1}{\lvert U \rvert} \sum_{u} \frac{\mathbf{1}[r_u < 10]}{\log_2(r_u + 2)}
$$

*In words:* A hit if the target ranks in the top 10; NDCG gives a hit at rank $r$ (0 = top) the discount $1/\log_2(r + 2)$.

*Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/10_sequence_models_for_user_behavior.ipynb)

### Recall@K

$$
\operatorname{Recall@}K(u) = \frac{\lvert \text{top}_K(u) \cap T_u \rvert}{\lvert T_u \rvert}
$$

*In words:* Of the items that were relevant for a user, what share did the K candidates contain?

*Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb) · [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/07_embeddings_and_semantic_search.ipynb) · [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb)

### Catalogue coverage

$$
\operatorname{Coverage} = \frac{\left\lvert \bigcup_{u} \text{top}_K(u) \right\rvert}{\lvert \mathcal{I} \rvert}
$$

*In words:* The share of the catalogue that appears in at least one user's list.

*Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb)

### RankNet's pairwise loss and its gradient

$$
L_{ij} = \log\!\left(1 + e^{-\sigma (s_i - s_j)}\right), \qquad \lambda_{ij} = \frac{\partial L_{ij}}{\partial s_i} = \frac{-\sigma}{1 + e^{\sigma (s_i - s_j)}}
$$

*In words:* A logistic loss on the score difference of a pair where $i$ should beat $j$.

*Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/06_learning_to_rank.ipynb)

### LambdaRank

$$
\lambda^{\text{LR}}_{ij} = \lambda_{ij} \cdot \left\lvert \Delta \operatorname{NDCG}_{ij} \right\rvert
$$

*In words:* Use RankNet's gradient, scaled by how much NDCG would change if $i$ and $j$ swapped places.

*Taught in:* [L2 · 06 Learning to rank](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/06_learning_to_rank.ipynb)

### Co-visitation counts

$$
C(a, b) = \sum_{s} \mathbf{1}[a \in s]\, \mathbf{1}[b \in s], \quad a \neq b
$$

*In words:* Count the sessions in which both items appear (once per session).

*Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb)

### Implicit ALS objective and its closed-form step

$$
\min_{X, Y} \sum_{u,i} c_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2\Big), \qquad x_u = (Y^\top C_u Y + \lambda I)^{-1} Y^\top C_u\, p_u
$$

*In words:* Weighted least squares on every cell, with weight $c_{ui}$; with item vectors fixed, each user vector is a small linear solve.

*Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb)

### BPR, the pairwise alternative

$$
L_\text{BPR} = -\sum_{(u, i, j)} \log \sigma\!\left(x_u^\top y_i - x_u^\top y_j\right) + \lambda \lVert \Theta \rVert^2
$$

*In words:* For a clicked item $i$ and a sampled unclicked item $j$, make the user score $i$ above $j$.

*Taught in:* [L2 · 07 Candidate generation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/07_candidate_generation.ipynb)

### The two-tower score

$$
s(u, i) = \frac{\mathbf{u}^\top \mathbf{v}_i}{\tau}, \qquad \lVert \mathbf{u} \rVert = \lVert \mathbf{v}_i \rVert = 1
$$

*In words:* The cosine similarity of the two normalised vectors, divided by a temperature.

*Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/08_two_tower_retrieval_and_ann.ipynb)

### Cosine similarity

$$
\cos(\mathbf a, \mathbf b) = \frac{\mathbf a \cdot \mathbf b}{\lVert \mathbf a \rVert \, \lVert \mathbf b \rVert}
$$

*In words:* The angle between two vectors, ignoring their lengths; for unit vectors it is just the dot product.

*Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb) · [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/07_embeddings_and_semantic_search.ipynb)

### In-batch softmax (InfoNCE) with the logQ correction

$$
L = -\frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\!\big(s(u_b, i_b) - \log q_{i_b}\big)}{\sum_{j=1}^{B} \exp\!\big(s(u_b, i_j) - \log q_{i_j}\big)}
$$

*In words:* For each user in the batch, cross-entropy of picking their own item out of the batch's items, with every item's logit lowered by the log of how often it is sampled.

*Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/08_two_tower_retrieval_and_ann.ipynb)

### InfoNCE with in-batch and hard negatives

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\mathbf q_i \cdot \mathbf d_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf q_i \cdot \mathbf d_j / \tau)}
$$

*In words:* For every query, a softmax over all $N$ documents in the batch (its positive, the other $B - 1$ positives and $N - B$ hard negatives) and the cross-entropy of picking its own.

*Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/08_contrastive_training_and_reranking.ipynb)

### A random-guess baseline for the loss

$$
\mathcal{L}_{\text{uniform}} = \ln N
$$

*In words:* A model that cannot tell the positive from $N - 1$ negatives scores $\ln N$; useful loss starts below that.

*Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/08_contrastive_training_and_reranking.ipynb)

### BM25

$$
\operatorname{BM25}(q, d) = \sum_{t \in q} \operatorname{idf}(t)\, \frac{f(t,d)\,(k_1 + 1)}{f(t,d) + k_1 \big(1 - b + b\, \lvert d \rvert / \text{avgdl}\big)}, \qquad
\operatorname{idf}(t) = \ln\!\Big(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\Big)
$$

*In words:* Rare query terms (high idf) that appear in the document count most; repeated occurrences saturate ($k_1$); long documents are penalised ($b$).

*Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/07_embeddings_and_semantic_search.ipynb) · [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb)

### Reciprocal rank fusion

$$
\operatorname{RRF}(d) = \sum_{s} \frac{1}{k + \operatorname{rank}_s(d)}, \qquad k = 60
$$

*In words:* Add one over (constant plus rank) across rankers; consistent agreement beats one first place.

*Taught in:* [L3 · 07 Embeddings & semantic search](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/07_embeddings_and_semantic_search.ipynb)

### Recall of an approximate index

$$
\operatorname{Recall@}k = \frac{1}{\lvert Q \rvert} \sum_{q \in Q} \frac{\lvert \text{ANN}_k(q) \cap \text{Exact}_k(q) \rvert}{k}
$$

*In words:* The share of the exact top-$k$ that the approximate search also returned.

*Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/08_two_tower_retrieval_and_ann.ipynb)

### Cost of exact search

$$
\text{cost} = N \cdot d \ \text{multiply-adds per query}
$$

*In words:* One dot product per item per query.

*Taught in:* [L2 · 08 Two-tower retrieval & ANN](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/08_two_tower_retrieval_and_ann.ipynb)

### How ANN search cost scales

$$
\text{exact: } O(n\,d), \qquad \text{IVF: } O\!\left(\frac{n \cdot \text{nprobe}}{\text{nlist}}\, d\right), \qquad \text{HNSW: roughly } O(\log n)\ \text{hops}
$$

*In words:* IVF scans only the probed lists; HNSW walks a graph in roughly logarithmic hops.

*Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb)

### Re-ranking cost

$$
\text{latency} \approx \text{depth} \times t_{\text{pair}} / \text{parallelism}, \qquad \text{pair inferences/s} = \text{QPS} \times \text{depth}
$$

*In words:* Every query pays one cross-encoder pass per re-ranked candidate.

*Taught in:* [L3 · 08 Contrastive training & re-ranking](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/08_contrastive_training_and_reranking.ipynb)

### The position-based click model

$$
P(c = 1 \mid u, i, p) = \underbrace{\theta_p}_{P(\text{exam} \mid p)} \cdot \underbrace{r_{u,i}}_{P(\text{rel} \mid u, i)}
$$

*In words:* A click happens only if the position is examined *and* the item is relevant, and the two are independent.

*Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/05_implicit_feedback_and_position_bias.ipynb) · [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/20_final_evaluation_and_mock_loop.ipynb)

### Propensities from random traffic

$$
\frac{\theta_p}{\theta_0} = \frac{\text{CTR}_\text{rand}(p)}{\text{CTR}_\text{rand}(0)}
$$

*In words:* With random order, mean relevance is the same at every position, so CTR ratios are examination ratios.

*Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/05_implicit_feedback_and_position_bias.ipynb)

### The IPS click-rate estimator, and clipping

$$
\hat r_\text{IPS} = \frac{1}{n} \sum_{j=1}^{n} \frac{c_j}{\theta_{p_j}}, \qquad \hat r_\text{clip} = \frac{1}{n} \sum_{j=1}^{n} c_j \min\!\left(\frac{1}{\theta_{p_j}}, M\right)
$$

*In words:* Divide each click by the chance that its position was examined; optionally cap the weight at $M$.

*Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/05_implicit_feedback_and_position_bias.ipynb)

### Off-policy value of a re-ranking

$$
\hat V(\text{new}) = \sum_{i \in \text{request}} \frac{\theta_{q_i}}{\theta_{p_i}}\, c_i
$$

*In words:* A click observed at logged position $p$ that the new ranking moves to position $q$ counts $\theta_q / \theta_p$ clicks.

*Taught in:* [L2 · 05 Implicit feedback & position bias](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/05_implicit_feedback_and_position_bias.ipynb) · [L2 · 20 Final evaluation & mock loop](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/20_final_evaluation_and_mock_loop.ipynb)

## Deep learning and transformers

### Binary cross-entropy on a logit, and its value at initialisation

$$
\ell(z, y) = -\big[y \log \sigma(z) + (1-y) \log(1-\sigma(z))\big], \qquad \ell^{*}_\text{const} = -\big[p \log p + (1-p) \log(1-p)\big]
$$

*In words:* The negative log of the probability the model gave to the true label; the best constant predictor scores the label entropy.

*Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/09_pytorch_training_done_right.ipynb)

### Softmax and cross-entropy

$$
\operatorname{softmax}(\mathbf{z})_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}, \qquad \ell = -\log \operatorname{softmax}(\mathbf{z})_y
$$

*In words:* Turn $C$ logits into probabilities, and pay $-\log$ of the probability of the right class.

*Taught in:* [L2 · 09 PyTorch training done right](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/09_pytorch_training_done_right.ipynb) · [L3 · 05 Fine-tuning a classifier](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/05_fine_tuning_a_classifier.ipynb)

### Softmax with temperature

$$
p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
$$

*In words:* Divide the logits by $\tau$ before the softmax: lower $\tau$ sharpens the distribution, higher flattens it.

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb)

### Backprop through a two-layer MLP

$$
\begin{aligned}
dZ_2 &= (P - Y)/n, & dW_2 &= H^\top dZ_2, & db_2 &= \textstyle\sum_i dZ_2 \\
dH &= dZ_2 W_2^\top, & dZ_1 &= dH \odot \mathbb{1}[Z_1 > 0], & dW_1 &= X^\top dZ_1
\end{aligned}
$$

*In words:* Start from "prediction minus target" and walk the graph backwards; every gradient has the shape of its variable.

*Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

### Batch norm

$$
\hat x = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat x + \beta, \qquad \text{running} \leftarrow (1 - m)\cdot\text{running} + m \cdot \text{batch statistic}
$$

*In words:* Standardise each feature with the batch's mean and (biased) variance during training, with the running averages at inference.

*Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

### LayerNorm

$$
\operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
\mu = \frac{1}{C}\sum_c x_c, \quad \sigma^2 = \frac{1}{C}\sum_c (x_c - \mu)^2
$$

*In words:* Per token, subtract the mean and divide by the standard deviation over channels, then apply a learned scale $\gamma$ and shift $\beta$.

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb)

### Inverted dropout

$$
y = \frac{x \odot m}{1 - p}, \quad m_j \sim \operatorname{Bernoulli}(1 - p), \qquad \mathbb{E}[y] = x
$$

*In words:* Zero each unit with probability $p$ and scale the survivors by $1/(1-p)$, so the expected output is unchanged; do nothing at inference.

*Taught in:* [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

### Scaled dot-product attention

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}} + M\Big)\, V
$$

*In words:* Score every query against every key, scale, hide the future, normalise each row, average the values.

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb) · [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/10_sequence_models_for_user_behavior.ipynb) · [L2 · 14 ML from scratch II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/14_ml_from_scratch_2.ipynb)

### Multi-head attention

$$
\operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H)\, W^O, \qquad
\text{head}_i = \operatorname{Attention}(X W_i^Q, X W_i^K, X W_i^V)
$$

*In words:* $H$ attentions on $d_k = C/H$ channels each, side by side, mixed by $W^O$.

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb)

### The attention mask inside attention

$$
\text{score}_{ij} \;\leftarrow\; \text{score}_{ij} + (1 - m_j)\cdot(-\infty)
$$

*In words:* A padded key $j$ gets $-\infty$, so after the softmax its weight is exactly 0 for every query.

*Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb)

### Mask-aware mean pooling

$$
\mathbf{s} = \frac{\sum_{t=1}^{T} m_t\, \mathbf{h}_t}{\sum_{t=1}^{T} m_t}
$$

*In words:* Average the hidden vectors of the real tokens only.

*Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb)

### Cross-entropy and perplexity

$$
\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta\big(x_{t+1} \mid x_{\le t}\big), \qquad
\operatorname{PPL} = e^{\mathcal{L}}
$$

*In words:* The average surprise at the true next token; perplexity is "as unsure as a uniform choice among PPL options".

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb)

### Next-item loss over the catalogue

$$
L = -\frac{1}{\lvert \mathcal{T} \rvert} \sum_{t \in \mathcal{T}} \log \frac{\exp(\mathbf{h}_t^\top \mathbf{e}_{y_{t+1}})}{\sum_{j=1}^{N} \exp(\mathbf{h}_t^\top \mathbf{e}_j)}
$$

*In words:* At every non-padding position $t$, softmax cross-entropy of the next item over all $N$ items, scored by the dot product with the (tied) item embeddings.

*Taught in:* [L2 · 10 Sequence models for user behaviour](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/10_sequence_models_for_user_behavior.ipynb)

### Self-attention cost

$$
\text{compute} = O(T^2 d) \text{ per layer}, \qquad M_{\text{scores}} = L \cdot H \cdot T^2 \cdot b \text{ bytes}
$$

*In words:* Attention compute and score memory grow with the square of sequence length.

*Taught in:* [L3 · 19 ML depth & research discussion](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/19_ml_depth_and_research_discussion.ipynb)

### The LoRA layer

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r}
$$

*In words:* The frozen weight's output plus a low-rank correction, scaled by $\alpha / r$.

*Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/06_parameter_efficient_fine_tuning.ipynb)

### LoRA trainable parameters

$$
\underbrace{r\,(d + k)}_{\text{LoRA}} \quad \text{vs} \quad \underbrace{d\,k}_{\text{full}}
$$

*In words:* LoRA trains $r(d + k)$ numbers per matrix instead of $dk$.

*Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/06_parameter_efficient_fine_tuning.ipynb)

### Merging

$$
W' = W_0 + \frac{\alpha}{r} B A
$$

*In words:* After training, fold the update into a weight of the original shape: zero extra latency, but one task per merged copy.

*Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/06_parameter_efficient_fine_tuning.ipynb)

## Efficient training and scaling

### FLOPs of a matrix multiply

$$
\text{FLOPs}\big(A_{m \times k}\,B_{k \times n}\big) = 2\,m\,n\,k
$$

*In words:* Each of the $m n$ outputs is a dot product of length $k$, i.e. $k$ multiplies and $k$ adds.

*Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb)

### Arithmetic intensity

$$
I = \frac{\text{FLOPs performed}}{\text{bytes moved to and from memory}}
$$

*In words:* How much arithmetic you get for every byte you read or write.

*Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb)

### The roofline

$$
P_{\text{attainable}} = \min\big(P_{\text{peak}},\; I \cdot B_{\text{mem}}\big), \qquad I^{\star} = \frac{P_{\text{peak}}}{B_{\text{mem}}}
$$

*In words:* You cannot beat the chip's peak, and you cannot compute faster than memory can feed you.

*Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb)

### Epoch time of an input pipeline

$$
T_{\text{epoch}} \approx n_{\text{batches}} \big(t_{\text{load}} + t_{\text{step}}\big) \;\;(\text{no workers}), \qquad
T_{\text{epoch}} \approx n_{\text{batches}} \max\!\Big(\frac{t_{\text{load}}}{k},\; t_{\text{step}}\Big) \;\;(k \text{ workers})
$$

*In words:* Without workers, loading and compute add up; with $k$ workers they overlap and the slower of the two sets the pace.

*Taught in:* [L3 · 01 GPU fundamentals & profiling](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/01_gpu_fundamentals_and_profiling.ipynb)

### Padding waste

$$
\text{waste} = 1 - \frac{\sum_i \ell_i}{\sum_{\text{batches } b} \lvert b \rvert \cdot \max_{i \in b} \ell_i}
$$

*In words:* The share of computed positions that are padding.

*Taught in:* [L3 · 04 Pretrained transformers & tokenisation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/04_pretrained_transformers_and_tokenization.ipynb)

### Model-state memory

$$
M_{\text{state}} = \Psi \,\big(b_{w} + b_{g} + b_{\text{opt}}\big)
$$

*In words:* Every parameter pays for its weight, its gradient and its optimizer state.

*Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/02_training_efficiency_and_memory.ipynb)

### Training memory with a frozen base

$$
M \approx b_w\, \Psi_{\text{total}} + 16\, \Psi_{\text{trained}} + M_{\text{act}}
$$

*In words:* Every weight must be stored ($b_w$ bytes each), but only trained weights need gradients and AdamW state (16 bytes each in fp32).

*Taught in:* [L3 · 06 Parameter-efficient fine-tuning](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/06_parameter_efficient_fine_tuning.ipynb)

### Activation memory of a transformer

$$
M_{\text{act}} \approx L \cdot B \cdot T \cdot \big(34\,h + 5\,a\,T\big) \;\text{bytes}
$$

*In words:* Linear in layers, batch and hidden size, and **quadratic in sequence length** through the attention probabilities.

*Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/02_training_efficiency_and_memory.ipynb)

### Gradient accumulation

$$
g = \sum_{i=1}^{k} \frac{n_i}{N}\, \nabla_\theta \bar{\ell}_i, \qquad N = \sum_{i=1}^{k} n_i
$$

*In words:* Weight each micro-batch's mean-loss gradient by its share of the examples.

*Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/02_training_efficiency_and_memory.ipynb)

### Global batch size

$$
B_{\text{global}} = N \cdot b_{\text{per GPU}} \cdot k_{\text{accum}}
$$

*In words:* The batch the optimiser sees is GPUs times per-GPU batch times accumulation steps.

*Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/09_distributed_data_parallel.ipynb)

### Activation checkpointing cost

$$
\text{step time} \approx F + 2F \;\longrightarrow\; F + 2F + F = \tfrac{4}{3} \times \text{(no checkpointing)}
$$

*In words:* Backward costs about two forwards; recomputing adds one more.

*Taught in:* [L3 · 02 Training efficiency & memory](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/02_training_efficiency_and_memory.ipynb)

### Loss scaling

$$
\nabla_\theta (S \cdot \mathcal L) = S \cdot \nabla_\theta \mathcal L, \qquad \text{fp16 normal range} \approx [6.1 \times 10^{-5},\; 65{,}504]
$$

*In words:* Scale the loss up so gradients are representable in fp16, then divide by $S$ in fp32 before the update.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

### Data parallelism averages gradients

$$
g = \frac{1}{N} \sum_{r=1}^{N} g_r, \qquad g_r = \nabla_\theta \frac{1}{b} \sum_{j \in \text{shard } r} \ell_j
$$

*In words:* With equal shards of size $b$, the mean of the ranks' mean-loss gradients is the gradient of the mean loss over the global batch $N b$.

*Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/09_distributed_data_parallel.ipynb)

### Ring all-reduce traffic

$$
\text{bytes sent per rank} = 2\,\frac{N - 1}{N}\, M
$$

*In words:* $N - 1$ reduce-scatter steps plus $N - 1$ all-gather steps, each sending $M / N$.

*Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/09_distributed_data_parallel.ipynb)

### All-reduce time and bus bandwidth

$$
T_{\text{comm}} \approx 2\,\frac{N-1}{N}\,\frac{M}{\text{BW}} + 2(N-1)\,\alpha, \qquad \text{busbw} = \text{algbw} \cdot 2\,\frac{N-1}{N}
$$

*In words:* A bandwidth term that barely depends on $N$, plus a latency term ($\alpha$ per hop) that grows with it.

*Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/09_distributed_data_parallel.ipynb)

### Scaling efficiency

$$
E = \frac{T_{\text{compute}}}{T_{\text{step}}}, \qquad T_{\text{step}} \approx T_{\text{compute}} + (1 - o)\, T_{\text{comm}}
$$

*In words:* The share of the step spent on useful compute; $o$ is the fraction of communication hidden behind backward.

*Taught in:* [L3 · 09 Distributed data parallel](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/09_distributed_data_parallel.ipynb)

### Model-state memory per GPU (mixed-precision Adam, $N$ data-parallel GPUs)

$$
M_{\text{DDP}} = 16\Psi, \quad M_{\text{Z1}} = 4\Psi + \frac{12\Psi}{N}, \quad M_{\text{Z2}} = 2\Psi + \frac{14\Psi}{N}, \quad M_{\text{Z3}} = \frac{16\Psi}{N}
$$

*In words:* Each stage shards one more of optimizer state (12 bytes), gradients (2) and parameters (2).

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

### Pipeline bubble

$$
\text{bubble} = \frac{p - 1}{m + p - 1}
$$

*In words:* $p - 1$ slots of fill and drain out of $m + p - 1$.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

### Tensor parallelism

$$
\text{column: } y = \big[\, x W_1^\top,\; x W_2^\top \,\big] \;(\text{all-gather}), \qquad
\text{row: } y = x_1 W_1^\top + x_2 W_2^\top \;(\text{all-reduce})
$$

*In words:* Split the weight by output columns and all-gather the halves, or by input rows and all-reduce the partial sums.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

### Training compute and time

$$
C \approx 6\,N D, \qquad t \approx \frac{C}{n_{\text{GPU}} \cdot P_{\text{peak}} \cdot \text{MFU}}
$$

*In words:* 6 FLOPs per parameter per token; time is that divided by the FLOP/s you actually get.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb) · [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

### Chinchilla compute-optimal sizing

$$
D^{\star} \approx 20\, N^{\star} \;\;\Rightarrow\;\; C = 6 N^{\star} (20 N^{\star}) \;\;\Rightarrow\;\; N^{\star} = \sqrt{C / 120}
$$

*In words:* Compute-optimal training uses about 20 tokens per parameter, so the best model size grows with the square root of compute.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

### Optimal checkpoint interval (Young 1974; Daly 2006)

$$
\text{waste}(\tau) \approx \frac{C}{\tau} + \frac{\tau/2 + R}{M}, \qquad \tau^{\star} = \sqrt{2\,C\,M} \;\;(\text{Daly: } \approx \sqrt{2CM} - C)
$$

*In words:* Checkpoint cost per interval plus, per failure, half an interval of lost work and a restart.

*Taught in:* [L3 · 10 Scaling to large models](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/10_scaling_to_large_models.ipynb)

## Serving and systems

### Symmetric int8 quantisation

$$
q = \operatorname{clamp}\!\big(\operatorname{round}(x / s),\,-127,\,127\big), \qquad s = \frac{\max_i \lvert x_i \rvert}{127}, \qquad \hat{x} = q\,s
$$

*In words:* Divide by the step size, round to the nearest integer, and multiply back.

*Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/11_quantization_and_distillation.ipynb)

### Affine (asymmetric) int8 quantisation

$$
s = \frac{x_{\max} - x_{\min}}{255}, \qquad z = \operatorname{round}\!\left(-128 - \frac{x_{\min}}{s}\right), \qquad q = \operatorname{clip}\!\left(\operatorname{round}(x/s) + z,\ -128,\ 127\right), \qquad \hat x = s\,(q - z)
$$

*In words:* A scale turns float units into integer steps, a zero-point shifts so that 0.0 lands exactly on an integer; the rounding error per value is at most $s/2$.

*Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb)

### Rounding error bound

$$
\lvert x - \hat{x} \rvert \le \frac{s}{2}
$$

*In words:* Rounding moves a value by at most half a step.

*Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/11_quantization_and_distillation.ipynb)

### Model size in memory

$$
\text{bytes} \approx \text{parameters} \times \frac{\text{bits per parameter}}{8}
$$

*In words:* Memory is parameters times bytes per parameter; int8 is a quarter of fp32.

*Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb) · [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/11_quantization_and_distillation.ipynb)

### Distillation loss

$$
\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(p_t^{(T)} \,\Vert\, p_s^{(T)}\right) + (1 - \alpha)\, \mathrm{CE}(z_s, y), \qquad p^{(T)} = \operatorname{softmax}(z / T)
$$

*In words:* Match the teacher's softened distribution (weight $\alpha$) and the true label (weight $1 - \alpha$).

*Taught in:* [L3 · 11 Quantisation & distillation](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/11_quantization_and_distillation.ipynb) · [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb)

### Parity criterion

$$
\max_i \left\lvert p^{\text{offline}}_i - p^{\text{API}}_i \right\rvert \le \varepsilon,
\qquad \varepsilon = 10^{-6}
$$

*In words:* The largest disagreement must be below what rounding to six decimals can explain ($0.5 \times 10^{-6}$), with a little slack.

*Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/10_consistent_serving.ipynb) · [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb)

### How a skewed feature moves a score

$$
\Delta \eta = \beta_j\, \delta_j,
\qquad
p_{\text{served}} = \sigma\big(\eta + \beta_j \delta_j\big)
$$

*In words:* The error in the feature, times its coefficient, is added straight to the log-odds.

*Taught in:* [L1 · 10 Consistent serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/10_consistent_serving.ipynb)

### Batching throughput

$$
\text{throughput}(B) = \frac{B}{t(B)}, \qquad \text{latency} \approx t_{\text{wait}} + t(B)
$$

*In words:* A batch of $B$ requests finishes together, so throughput rises as long as the call time $t(B)$ grows more slowly than $B$.

*Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)

### Arithmetic intensity of a Linear layer

$$
I(B) \approx \frac{2\,B\,d_{\text{in}}\,d_{\text{out}}}{b\,d_{\text{in}}\,d_{\text{out}}} = \frac{2B}{b}\ \ \text{FLOPs per byte}
$$

*In words:* Each weight read ($b$ bytes) is used for one multiply-add per request in the batch. It becomes compute-bound when $I(B)$ exceeds the hardware's peak FLOP/s ÷ memory bandwidth.

*Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)

### KV-cache size

$$
\text{bytes} = 2 \cdot L \cdot H \cdot d_{\text{head}} \cdot T \cdot B \cdot b
$$

*In words:* A key and a value (the 2) per layer, head, channel, position and sequence.

*Taught in:* [L3 · 03 A transformer from scratch](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/03_transformer_from_scratch.ipynb) · [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)

### Percentile

$$
p_q = x_{(\lceil q N / 100 \rceil)}
$$

*In words:* The value with $q\%$ of observations at or below it. NumPy's default interpolates between neighbours, so the two differ slightly on small samples.

*Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/13_health_logs_and_latency.ipynb)

### Little's law

$$
L = \lambda\, W
$$

*In words:* Average requests in flight = arrival rate × average time in the system.

*Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/13_health_logs_and_latency.ipynb) · [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb) · [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb) · [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)

### Why latency explodes near saturation

$$
W = \frac{S}{1 - \rho}, \qquad \rho = \lambda S
$$

*In words:* As utilisation ρ approaches 1, waiting time grows without bound, which is why capacity plans target 50–70% utilisation.

*Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb)

### Single-process ceiling

$$
\lambda_{\max} \approx \frac{1}{t_{\text{CPU per request}}}
$$

*In words:* CPU-bound Python serves at most one request per CPU-time per process, however many threads it has.

*Taught in:* [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/13_health_logs_and_latency.ipynb)

### Fan-out and tail latency

$$
P(\text{request waits for a slow shard}) = 1 - (1 - q)^N
$$

*In words:* With N parallel calls, each slow with probability q, the request is slow if any one is.

*Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb) · [L1 · 13 Health, logs & latency](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/13_health_logs_and_latency.ipynb)

### The latency budget

$$
\sum_{\text{stages}} b_s + b_{\text{outside}} \le \text{SLO}, \qquad p_{99}\Big(\sum_s t_s\Big) \le \sum_s p_{99}(t_s)
$$

*In words:* Stage budgets plus slack must fit the SLO; because tail latencies rarely coincide, the p99 of the sum is below the sum of p99s, which makes per-stage budgets conservative.

*Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb) · [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb) · [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb)

### Cache hit rate and its effect on latency

$$
h \le 1 - \frac{\text{distinct keys}}{\text{requests}}, \qquad \mathbb{E}[t] = h\, t_{\text{hit}} + (1 - h)\, t_{\text{miss}}
$$

*In words:* A cache can serve at most the repeated keys; expected latency mixes hit and miss times.

*Taught in:* [L2 · 17 Serving a ranking cascade](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/17_serving_a_ranking_cascade.ipynb)

### Amdahl's law: what a speed-up is worth end to end

$$
\text{overall speed-up} = \frac{1}{(1 - f) + f / k}
$$

*In words:* Speeding up a stage that is a fraction $f$ of the time can never beat $1/(1-f)$ overall.

*Taught in:* [L2 · 18 Model optimisation for inference](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/18_model_optimization_for_inference.ipynb)

### Exponential backoff

$$
d_n = b \cdot 2^{n} \quad (\times\ \text{a random factor, with jitter})
$$

*In words:* Double the wait after each failed attempt, with random jitter so clients do not retry in lockstep.

*Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/11_api_clients_and_tests.ipynb)

### Worst-case time of one logical call

$$
T_{\max} = (R + 1)\,\tau + \sum_{n=0}^{R-1} b\, 2^{n} = (R + 1)\,\tau + b\,(2^{R} - 1)
$$

*In words:* Every attempt may hang until the timeout, plus all the waits.

*Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/11_api_clients_and_tests.ipynb)

### Why retries help with transient failures

$$
P(\text{give up}) = q^{\,R + 1}
$$

*In words:* If attempts fail independently with probability $q$, all $R + 1$ fail with probability $q^{R+1}$; during an outage they are not independent.

*Taught in:* [L1 · 11 API clients & tests](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/11_api_clients_and_tests.ipynb)

### Capacity: QPS and replicas

$$
\text{QPS}_{\text{avg}} = \frac{\text{DAU} \times r}{86{,}400},
\qquad
\text{QPS}_{\text{peak}} = f \cdot \text{QPS}_{\text{avg}},
\qquad
\text{replicas} = \left\lceil \frac{\text{QPS}_{\text{peak}}}{u \cdot \text{QPS}_{\text{replica}}} \right\rceil
$$

*In words:* Average QPS from daily users, times a peak factor, divided by what one replica handles at target utilisation.

*Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb) · [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb) · [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb)

### Fleet size with headroom

$$
\text{replicas} = \max\!\left(2,\ \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS}_{\text{replica}} \cdot (1 - h)} \right\rceil \right)
$$

*In words:* At least two replicas, and enough to carry peak traffic while keeping a fraction $h$ of each spare.

*Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb) · [L3 · 20 Full interview loop & portfolio](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/20_full_interview_loop_and_portfolio.ipynb)

### Scoring load and servers

$$
\text{rows/s} = \text{QPS} \times \text{candidates scored}, \qquad \text{servers} = \left\lceil \frac{\text{rows/s}}{\text{rows/s per core} \times \text{cores} \times \text{utilisation}} \right\rceil
$$

*In words:* Rows scored per second is QPS times candidates; divide by what the fleet scores per second at target utilisation.

*Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb)

### Batch-scoring time

$$
t = \frac{N}{\text{rows per second}}
$$

*In words:* If the whole population scores in minutes, a nightly batch job beats an online service.

*Taught in:* [L1 · 17 System design & mock interview](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/17_system_design_and_mock_interview.ipynb)

### Embedding and index memory

$$
\text{bytes} = n \times d \times b \quad (\text{flat}), \qquad n\,(d\,b + 2M \cdot 4) \quad (\text{HNSW}), \qquad n\,(\text{code bytes} + 8) \quad (\text{IVF-PQ})
$$

*In words:* Flat storage is $n \times d \times$ bytes; HNSW adds graph links; product quantisation compresses each vector to a few bytes.

*Taught in:* [L2 · 19 Recommendation & search system design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/19_recsys_and_search_system_design.ipynb)

### Embedding-table memory

$$
\text{bytes} = \text{rows} \times \text{dim} \times \text{bytes per value} \times (1 + \text{optimiser slots})
$$

*In words:* Embedding tables dominate recommender memory, and Adam's two slots triple them during training.

*Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb)

### Online feature-store size

$$
\text{GB} = \frac{E \cdot (k + F \cdot b) \cdot o \cdot r}{10^9}
$$

*In words:* Entities times (key plus features times bytes), times store overhead and replicas.

*Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

### Cost per 1,000 requests

$$
\text{cost}_{1k} = \frac{1000 \cdot p}{3600 \cdot r \cdot u}
$$

*In words:* The hourly price divided by the number of requests actually served in that hour.

*Taught in:* [L3 · 12 High-throughput serving](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/12_high_throughput_serving.ipynb)

### The price of idle capacity

$$
\text{cost per useful GPU-hour} = \frac{\text{price per hour}}{u}
$$

*In words:* An accelerator used a fraction $u$ of the time costs $1/u$ times its price per useful hour.

*Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

### Cost per request

$$
\text{API-style: } c = n_{\text{in}}\, p_{\text{in}} + n_{\text{out}}\, p_{\text{out}}, \qquad \text{self-hosted: } c = \frac{p_{\text{hour}}}{3600} \cdot t_{\text{request}}
$$

*In words:* API: tokens in and out times their prices; self-hosted: the hourly price times the seconds each request occupies.

*Taught in:* [L3 · 16 Evaluating LLM systems & guardrails](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/16_evaluating_llm_systems_and_guardrails.ipynb)

### LLM cost per conversation

$$
P_t = S + R + U + (t - 1)(U + A), \qquad \text{cost} = \sum_{t=1}^{n} \big( P_t\, p_{\text{in}} + A\, p_{\text{out}} \big)
$$

*In words:* Every turn re-sends the system prompt, retrieved text and the whole history, so input tokens grow with each turn.

*Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb)

### Blended cost with a human fallback

$$
c = c_{\text{LLM}} + (1 - d)\, c_{\text{human}}
$$

*In words:* The LLM cost plus the human cost of every conversation it fails to deflect.

*Taught in:* [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb)

### Risk score

$$
\text{risk} = \text{likelihood} \times \text{impact} \qquad (\text{each } 1\text{–}5)
$$

*In words:* Rate each risk 1–5 on both axes and deal with the largest products first.

*Taught in:* [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

## Monitoring

### Population Stability Index

$$
\operatorname{PSI} = \sum_{b=1}^{B} (a_b - e_b)\, \ln\frac{a_b}{e_b}
$$

*In words:* For each bin, the change in share times the log of the ratio of shares; both factors have the same sign, so every term is $\ge 0$.

*Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/14_monitoring_and_rollback.ipynb)

### Score-mean ratio (prediction drift)

$$
r = \frac{\bar{s}_{\text{current}}}{\bar{s}_{\text{reference}}}
$$

*In words:* A cheap prediction-drift alarm: the current mean score over the reference mean.

*Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/14_monitoring_and_rollback.ipynb)

### Calibration ratio on a matured cohort

$$
\text{calibration ratio} = \frac{\bar{s}}{\bar{y}}
$$

*In words:* Mean predicted over observed rate, once labels have matured; 1 is perfect.

*Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/14_monitoring_and_rollback.ipynb)

### Rollback rule

$$
\text{roll back if } \text{PR-AUC}_{\text{cand}} < (1 - \delta)\,\text{PR-AUC}_{\text{base}}
$$

*In words:* Agree in advance how much worse a candidate may be before it is rolled back automatically.

*Taught in:* [L1 · 14 Monitoring & rollback](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/14_monitoring_and_rollback.ipynb)

### Delayed feedback and label maturity

$$
P(y_{\text{obs}} = 1 \mid x, e) = \mathrm{pCVR}(x) \cdot F(e), \qquad F(e) = 1 - e^{-e/\tau}\ \ \text{(exponential delays)}
$$

*In words:* A conversion is visible only once it has happened; with exponential delays, a share $1 - e^{-e/\tau}$ is visible after time $e$.

*Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb) · [L3 · 17 ML platform design](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/17_ml_platform_design.ipynb)

## Ads, CTR and bandits

### Re-calibration after negative downsampling

$$
\frac{q}{1 - q} = \frac{1}{w} \cdot \frac{p}{1 - p} \quad\Longrightarrow\quad p = \frac{q}{q + (1 - q)/w}
$$

*In words:* Keeping a random fraction $w$ of negatives divides their count by $1/w$, so the odds the model learns are $1/w$ times the true odds; multiply them back by $w$.

*Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb)

### Hash collisions

$$
P(\text{a key shares its bucket}) = 1 - \left(1 - \tfrac{1}{n}\right)^{k - 1} \approx 1 - e^{-k/n}
$$

*In words:* With $k$ keys hashed into $n$ buckets, a key shares its bucket with probability about $1 - e^{-k/n}$.

*Taught in:* [L3 · 13 CTR prediction at scale](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/13_ctr_prediction_at_scale.ipynb)

### ESMM

$$
\mathrm{pCTCVR}(x) = \mathrm{pCTR}(x) \times \mathrm{pCVR}(x), \qquad \mathcal{L} = \mathrm{BCE}(\mathrm{pCTR}, \text{click}) + \mathrm{BCE}(\mathrm{pCTCVR}, \text{click} \wedge \text{conv})
$$

*In words:* Both losses are defined on **every** impression; pCVR is never fitted on clicked rows alone.

*Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb)

### Blended ranking score

$$
\text{score} = \mathrm{pCTR}^{a} \cdot \mathrm{pCVR}^{b} \cdot \text{value}
$$

*In words:* With calibrated predictions and $a = b = 1$, the score is expected value per impression.

*Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb)

### eCPM and the second price

$$
\mathrm{eCPM}_i = \mathrm{pCTR}_i \cdot \text{bid}_i \cdot 1000, \qquad \text{price per click} = \max\!\left(\text{reserve},\ \frac{\text{bid}_2 \cdot \mathrm{pCTR}_2}{\mathrm{pCTR}_1}\right) \le \text{bid}_1
$$

*In words:* Rank by expected revenue per 1,000 impressions; the winner pays per click the smallest bid that would still beat the runner-up.

*Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb) · [L3 · 18 Large-scale design cases](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/18_large_scale_design_cases.ipynb)

### Revenue equivalence

$$
\mathbb{E}[\text{revenue}] = \frac{n-1}{n+1} \qquad (n \text{ bidders, values i.i.d. } U[0, 1])
$$

*In words:* With $n$ bidders and uniform values, first-price and second-price auctions earn the same expected revenue.

*Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb)

### Bandits: regret, UCB1 and Thompson sampling

$$
\mathrm{Regret}(T) = \sum_{t=1}^{T} \big( p^{*} - p_{a_t} \big), \qquad a_t^{\mathrm{UCB1}} = \arg\max_k \left( \hat{p}_k + \sqrt{\frac{2 \ln t}{n_k}} \right)
$$

$$
\theta_k \sim \mathrm{Beta}(\alpha_k, \beta_k), \quad a_t = \arg\max_k \theta_k, \qquad \alpha_{a_t} \mathrel{+}= r_t, \quad \beta_{a_t} \mathrel{+}= 1 - r_t
$$

*In words:* Regret counts the clicks lost by not always playing the best arm; UCB1 adds an optimism bonus that shrinks with plays; Thompson sampling draws from each arm's Beta posterior and plays the best draw.

*Taught in:* [L3 · 14 Multi-task ranking & exploration](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/14_multitask_ranking_and_exploration.ipynb)

## Privacy and fairness

### Bias AUCs (subgroup, BPSN, BNSP)

$$
\mathrm{AUC}(P, N) = \Pr\big(s(x^{+}) > s(x^{-})\big), \quad x^{+} \sim P,\ x^{-} \sim N \quad (\text{ties count } \tfrac12)
$$

$$
\begin{aligned}
\text{Subgroup AUC} &= \mathrm{AUC}(\text{SP},\ \text{SN}) \\
\text{BPSN AUC} &= \mathrm{AUC}(\text{BP},\ \text{SN}) \\
\text{BNSP AUC} &= \mathrm{AUC}(\text{SP},\ \text{BN})
\end{aligned}
$$

*In words:* AUC is the chance a positive outscores a negative; the bias AUCs restrict the pairs to mixes of subgroup and background examples.

*Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/15_fairness_robustness_and_privacy.ipynb)

### Error rates per group at threshold $t$

$$
\mathrm{FPR}_g(t) = \Pr(s \ge t \mid y = 0, g), \qquad \mathrm{FNR}_g(t) = \Pr(s < t \mid y = 1, g)
$$

*In words:* False-positive and false-negative rates computed separately per group, at the same threshold.

*Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/15_fairness_robustness_and_privacy.ipynb)

### Power mean over groups

$$
M_p(m_1, \dots, m_G) = \Big(\frac{1}{G} \sum_{g=1}^{G} m_g^{\,p}\Big)^{1/p}, \qquad p = -5
$$

*In words:* A mean that, for negative $p$, is pulled towards the worst group.

*Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/15_fairness_robustness_and_privacy.ipynb)

### Differential privacy

$$
\Pr[A(D) \in S] \le e^{\varepsilon}\, \Pr[A(D') \in S] + \delta \quad \text{for all neighbouring } D, D' \text{ and all } S
$$

*In words:* Adding or removing one record changes the probability of any outcome by at most a factor $e^{\varepsilon}$ (plus a small slack $\delta$).

*Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/15_fairness_robustness_and_privacy.ipynb)

### One DP-SGD step

$$
\bar{g}_i = g_i \cdot \min\!\left(1, \frac{C}{\lVert g_i \rVert_2}\right), \qquad \tilde{g} = \frac{1}{n}\left( \sum_{i=1}^{n} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) \right), \qquad \theta \leftarrow \theta - \eta\, \tilde{g}
$$

*In words:* Clip each example's gradient to norm $C$, add Gaussian noise scaled to $C$, then average and step.

*Taught in:* [L3 · 15 Fairness, robustness & privacy](https://github.com/berdakh/MLBOOTCAMP/blob/main/level3/notebooks/15_fairness_robustness_and_privacy.ipynb)

## Coding interviews: complexity

### Reading growth from timings

$$
k \approx \frac{\log(t_2 / t_1)}{\log(n_2 / n_1)}
$$

*In words:* The local exponent that `growth_table` prints: if doubling n multiplies the time by 4, the algorithm is about O(n²); by 2, O(n).

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb)

### Input size → the complexity you need

$$
n \le 20 \Rightarrow O(2^n) \qquad n \le 10^3\text{–}10^4 \Rightarrow O(n^2) \qquad n \approx 10^5\text{–}10^6 \Rightarrow O(n \log n) \text{ or } O(n)
$$

*In words:* Pure Python does about $10^7$ simple steps per second, so $n$ tells you which complexity you can afford.

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb) · [L1 · 16 Python & SQL drills](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/16_python_and_sql_drills.ipynb)

### Why the sliding window is O(n)

$$
\text{total steps} = \underbrace{n}_{\text{right moves}} + \underbrace{\le n}_{\text{left moves}} \le 2n
$$

*In words:* Count pointer moves, not loop nesting. A `while` inside a `for` is not automatically quadratic.

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb)

### Heap top-k versus sorting

$$
O(n \log k) \ \text{vs}\ O(n \log n)
$$

*In words:* A size-$k$ heap keeps the top $k$ of a stream in $O(n \log k)$ time and $O(k)$ memory.

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb)

### Prefix sums and "subarray sum equals k"

$$
\sum_{t=i}^{j} x_t = P_{j+1} - P_i, \qquad P_0 = 0, \qquad \#\{\text{subarrays with sum } k\} = \sum_{j} \#\{\, i \le j : P_i = P_{j+1} - k \,\}
$$

*In words:* Any subarray sum is a difference of two prefix sums, so count earlier prefixes equal to the current one minus $k$.

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb)

### Output-sensitive bounds

$$
O(n + p), \qquad p \le \binom{n}{2} = \frac{n(n-1)}{2}
$$

*In words:* If the answer itself can have $p$ items, no algorithm beats $O(n + p)$.

*Taught in:* [L2 · 15 Coding patterns I](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/15_coding_patterns_1.ipynb)

### Graph traversal cost

$$
T_{\text{BFS}} = T_{\text{DFS}} = O(V + E)
$$

*In words:* Each node is enqueued (or pushed) once and each edge is looked at once (twice if undirected). A grid of R × C cells has V = RC and E ≤ 4RC.

*Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/16_coding_patterns_2.ipynb)

### Dijkstra's relaxation

$$
d(v) \leftarrow \min\big(d(v),\ d(u) + w(u, v)\big), \qquad T = O\big((V + E) \log V\big) \text{ with a binary heap}
$$

*In words:* When you pop the closest unfinished node u, try every edge out of it; correct because, with $w \ge 0$, nothing popped later can be closer.

*Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/16_coding_patterns_2.ipynb)

### Binary search iterations

$$
\text{iterations} = \lceil \log_2(\text{hi} - \text{lo} + 1) \rceil
$$

*In words:* Each check halves the range of candidates.

*Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/16_coding_patterns_2.ipynb)

### The DP recurrences in this notebook

$$
\text{coins}(a) = 1 + \min_{c \le a} \text{coins}(a - c), \qquad
D(i, j) = \min\big(D(i-1, j) + 1,\ D(i, j-1) + 1,\ D(i-1, j-1) + \mathbb{1}[a_i \ne b_j]\big)
$$

*In words:* The best answer for a state is built from the best answers of smaller states; the base cases are $\text{coins}(0) = 0$, $D(i, 0) = i$, $D(0, j) = j$.

*Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/16_coding_patterns_2.ipynb)

### Union-find

$$
T(\text{m operations on n elements}) = O\big(m\, \alpha(n)\big), \qquad \alpha(n) \le 4 \text{ for any } n \text{ you will meet}
$$

*In words:* With path compression and union by size each operation is effectively constant time; with neither, a chain makes `find` O(n).

*Taught in:* [L2 · 16 Coding patterns II](https://github.com/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/16_coding_patterns_2.ipynb)

---

See also: the [glossary](glossary.md) for definitions, the [interview question bank](interview-questions.md) for practice, and the [diagram index](diagrams.md) for every figure.
