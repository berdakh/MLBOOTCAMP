# Capstone 1 · Predictive maintenance (NASA C-MAPSS)

**Recommended order: 1st** · MVP ≈ 4 days at 3 h/day · CPU only (GPU optional)

**Guided solution:** [walkthrough.ipynb](walkthrough.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap1-predictive-maintenance/walkthrough.ipynb) — a
step-by-step worked solution of the MVP below, written so that you attempt
each step first and then compare. It was executed end to end on the official
FD001 files (about 1–7 minutes on a CPU, depending on load); its final step
produced the [results table](#results) below.

A public-data demonstration inspired by industrial and energy-asset
maintenance problems. It uses **simulated** turbofan degradation data
published by NASA; it is not work for NASA, an airline, an engine maker or an
energy company, and it says nothing about any real fleet.


---

## Business question

*For each engine currently in service, how many more operating cycles can it
run before it fails — and is that estimate safe to plan maintenance on?*

**Who acts on it:** a maintenance planner who schedules shop visits and spare
parts. A prediction that is too **late** (over-estimates remaining life) risks
an in-service failure; one that is too **early** wastes useful life and shop
capacity. The two errors do not cost the same, which is why the metric below
is asymmetric.

## Why this project

* **Entity-level evaluation**, the industrial twin of cross-subject
  evaluation in EEG/BCI: engines are subjects, cycles are trials. You can
  explain why a window-level random split is leakage in one sentence — and
  then prove it with a number.
* **A label you have to construct.** Remaining useful life (RUL) is not in the
  training file; you derive it, and the cap you choose changes the results.
* **A compact temporal model** (1D-CNN or GRU in PyTorch) trained on windows,
  compared honestly against tree models on hand-made features.
* **An asymmetric, decision-driven metric** and an early/late analysis — the
  kind of reasoning Applied Scientist interviews probe.

## Data

| Fact | Status |
|---|---|
| Landing page: <https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data> (NASA Prognostics Center of Excellence, PCoE) | **verified from an official page snippet** |
| Distributed as a ZIP of space-separated text files with 26 numeric columns | **verified from an official page snippet** |
| A separate catalogue entry for the C-MAPSS *simulator software* exists and is not downloadable — do not confuse the two | **verified from an official page snippet** |
| Training trajectories run to failure; test trajectories stop some time before failure; a separate file gives the true remaining cycles for each test engine | **verified from an official page snippet** |
| Download used by the walkthrough: <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip> (the link NASA's PCoE data repository points to), 12,429,152 bytes, SHA-256 `c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2` | **verified by download** (November 2026); confirm on the landing page that it is still the current link |
| The outer ZIP contains a second ZIP, `CMAPSSData.zip`, with `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` (x = 1–4), `readme.txt` and the Saxena et al. PDF | **verified by opening the archive** |
| Columns: unit number, cycle, 3 operational settings, 21 sensor measurements | **verified** from the archive's `readme.txt` and the data (its readme lists "sensor measurement 26", a typo: the files have 21 sensors) |
| FD001 and FD003: one operating condition; FD002 and FD004: six. FD001/FD002: one fault mode; FD003/FD004: two | **verified** from the archive's `readme.txt` |
| FD001: 100 training engines (20,631 rows) and 100 test engines (13,096 rows); 100 RUL values | **verified by counting** in the walkthrough |
| The S3 "Data Set 2" on NASA's page is **N-CMAPSS** (Chao et al., 2021), a different, much larger dataset — not FD001–FD004 | **verified from an official page snippet** |

**Licence / terms.** NASA data is generally a US Government work and in the
public domain in the US — **to verify**: confirm the terms shown on the
landing page (and any linked NASA open-data policy) before use, and record
what you found in `DATA_CARD.md`. Even if no licence restricts you, cite it.

**Citation** (**verified from an official page snippet**):

> A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for
> Aircraft Engine Run-to-Failure Simulation", *International Conference on
> Prognostics and Health Management (PHM08)*, 2008.
> DOI: 10.1109/PHM.2008.4711414

**Known gotchas** (checked on FD001 in the walkthrough; **to verify** for
the other subsets):

* Lines end with two trailing spaces, so a naive split on single spaces
  produces 28 fields. Use `sep=r"\s+"` and assert 26 columns.
* Several sensors are (near-)constant in FD001/FD003. Drop them by a rule
  computed on **training engines only** (e.g. std below a threshold), not by
  copying a list from a blog. In FD001, `setting3`, `s1`, `s5`, `s10`, `s16`,
  `s18` and `s19` take a single value and `s6` two.
* Unit IDs restart at 1 in every file: test engine 1 is not training engine 1.
  Never join files on `unit`.
* Test trajectories have different lengths (31–303 cycles in FD001); in
  other subsets some are shorter than a typical window. Decide how to pad
  (repeat first row, or zero-pad with a mask) and test it.
* The RUL file has one number per test engine, aligned to the **last** cycle
  of that engine's test trajectory — not to every row.
* MVP uses **FD001 only** (one condition, one fault mode). FD002/FD004 need
  normalisation per operating condition, which is an extension.

## Problem framing

* **Unit of prediction:** one engine at its latest observed cycle.
* **Input:** the last `W` cycles of that engine (e.g. `W = 30`, a choice to
  justify and test) — 3 settings + the non-constant sensors.
* **Output:** predicted RUL in cycles (a non-negative number).
* **Label (training):** `RUL = max_cycle(unit) − cycle`. Most published work
  uses a **piecewise-linear cap**: `min(RUL, R_max)` with `R_max` commonly
  around 125–130, on the argument that degradation is not observable early
  in life. **The cap is a modelling choice, not a fact.** Pick one, state it,
  and show how results change for 2–3 values.
* **Label (evaluation):** the true RUL from `RUL_FD001.txt` at the last cycle
  of each test engine. Report metrics against the **uncapped** truth as the
  primary result (that is what a planner experiences); you may also report
  capped-truth metrics for comparison with literature, clearly labelled.

## Baselines

1. **Constant baseline:** predict the mean (capped) RUL of training windows.
   Trivial, but it calibrates how impressive everything else is.
2. **Linear / ridge on last-cycle features:** the latest value of each sensor.
3. **Gradient-boosted trees on window features:** per-sensor mean, std,
   min, max, last value and least-squares slope over the window. This is
   often a strong baseline; the temporal model must beat it to justify itself.

## Improvements

* **Compact temporal model (MVP):** a small **1D-CNN** (2–3 conv layers,
  ~32 channels, global average pooling, a linear head) **or** a 1-layer
  **GRU** (hidden size ~32–64), in PyTorch, trained on sliding windows with
  MSE (or Huber) loss on capped RUL. FD001 is small (tens of thousands of
  rows — **count it**), so this is **CPU-trainable** in minutes per run;
  measure and report your actual wall-clock time. A GPU is optional and not
  needed for the MVP.
* Early stopping on a **validation split of engines**, never of windows.
* Ensembling 3–5 seeds and reporting the spread is a cheap, honest
  improvement — it also tells you how much of a difference is noise.
* **Extension:** asymmetric loss (penalise over-prediction more) and compare
  early/late behaviour.

## Evaluation protocol

**Splits — engine-level, like cross-subject.**

* `GroupKFold` over **training engine IDs** (e.g. 5 folds) for model
  selection: window length, cap, architecture, hyperparameters.
  All windows of one engine are in exactly one fold.
* Feature scaling and constant-sensor removal are fitted **inside** each
  fold, on that fold's training engines.
* The official **test set is evaluated once**, at the end, with the chosen
  configuration retrained on all training engines.
* Uncertainty: bootstrap over **test engines** (not windows) for a 95% CI.

**Metrics.**

* **RMSE** in cycles, at the last cycle of each test engine.
* **NASA asymmetric score** (from the PHM08 challenge; check the exact form
  against Saxena et al. 2008). With `d = RUL_pred − RUL_true`:
  `s = Σ (exp(−d/13) − 1)` for `d < 0` (early) and `Σ (exp(d/10) − 1)` for
  `d ≥ 0` (late). Lower is better; late predictions are penalised more, and
  the sum is dominated by a few very late engines — so always show the
  per-engine distribution, not just the total.
* **Early/late analysis:** share of engines predicted late; mean error
  for early vs late; a scatter of predicted vs true RUL; errors binned by true
  RUL (e.g. `< 25`, `25–75`, `> 75` cycles) — accuracy near failure matters
  most to a planner.

## Leakage traps

* **Window-level random split.** Overlapping windows from the same engine in
  train and validation. Demonstrate it: train once with a random window split
  and once with `GroupKFold`, and report both. The gap is your interview
  story.
* **Scaling on all data**, including test engines.
* **Sensor selection or cap tuned on the test set.**
* **Using `cycle` or "fraction of life" as a feature.** In training, the
  current cycle is correlated with RUL by construction; at test time you do
  not know total life. `cycle` itself is observable and can be used with
  care, but anything derived from `max_cycle` cannot.
* **Evaluating on every test row** instead of the last cycle — the RUL file
  does not label the other rows.

## MVP (≈ 4 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repo from the template; `make data` downloads and checksums the ZIP; parser with column-count assertion; RUL label; EDA of sensor trends for a few engines; constant-sensor rule; **engine-level split test** |
| 2 | Window builder (+ padding) with tests; constant and tree baselines with `GroupKFold`; random-split vs group-split comparison; RMSE + NASA score implemented and unit-tested on hand-computed cases |
| 3 | 1D-CNN or GRU in PyTorch on CPU; early stopping on validation engines; 3 seeds; cap sensitivity (2–3 values) |
| 4 | Final test evaluation **once**; bootstrap CIs; early/late analysis; API; README, model card, data card |

**Extensions** (optional):

* FD002/FD004 with per-operating-condition normalisation (cluster the 3
  settings, normalise sensors within cluster) — CPU feasible.
* Prediction intervals (quantile loss or conformal prediction over engines).
* A health-index / anomaly view of degradation onset.
* Larger temporal models or N-CMAPSS — **GPU recommended**; N-CMAPSS is much
  larger and is a separate dataset with its own documentation to verify.

## API design

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model loaded; returns `model_version`, `window`, `rul_cap` |
| POST | `/predict` | RUL for 1–100 engines, each given its recent cycles |

Request:

```json
{
  "engines": [
    {
      "unit_id": "engine-17",
      "cycles": [
        {"cycle": 101, "settings": [0.0012, -0.0003, 100.0], "sensors": {"s2": 642.1, "s3": 1589.7, "s4": 1400.6}},
        {"cycle": 102, "settings": [0.0008, 0.0001, 100.0], "sensors": {"s2": 642.4, "s3": 1590.2, "s4": 1402.3}}
      ]
    }
  ]
}
```

(Values in both examples are illustrative placeholders, not from the dataset
or a model; the real request lists every sensor your model uses.)

Response:

```json
{
  "model_version": "v3",
  "predictions": [
    {"unit_id": "engine-17", "rul_cycles": 112.5, "padded": true, "n_cycles_used": 2}
  ]
}
```

**Validation rules:** cycles strictly increasing and unique per engine; at
least 1 and at most e.g. 500 cycles; all required sensors present (unknown
keys rejected, `extra="forbid"`); values finite; settings length 3. If fewer
than `W` cycles are sent, the service pads exactly as in training and says so
(`padded: true`). Output clipped at 0.

**Precomputed vs online:** nothing is precomputed except the fitted scaler,
sensor list and model weights, which ship in the model bundle. Windowing and
scaling are computed online with **the same function** used in training
(import it, do not copy it). Load the PyTorch model once at startup, run with
`torch.no_grad()` and `model.eval()`.

## Results

Produced by Step 10 of the [walkthrough](walkthrough.ipynb) from its own
computed numbers, on the official FD001 files, executed on a CPU in November
2026 (python 3.12, scikit-learn 1.6, PyTorch 2.8). Test metrics use the
**uncapped** truth at the last cycle of each test engine; intervals are 95%
bootstrap intervals over test engines. Your numbers will differ slightly on
other hardware (PyTorch) and will differ more if you make other choices —
which is the point: your repository's table must come from your run.

| Model | Split | RMSE, cycles (95% CI) | NASA score | Late predictions (95% CI) |
|---|---|---|---:|---|
| constant (mean capped RUL) | official FD001 test, 100 engines, uncapped truth | 43.1 (38.6–47.4) | 33629 | 50% (40%–60%) |
| ridge on window features | official FD001 test, 100 engines, uncapped truth | 17.2 (15.0–19.4) | 472 | 53% (43%–63%) |
| gradient boosting | official FD001 test, 100 engines, uncapped truth | 14.4 (12.2–16.6) | 338 | 61% (51%–70%) |
| 1D-CNN (3-seed mean) | official FD001 test, 100 engines, uncapped truth | 16.1 (13.7–18.5) | 500 | 59% (49%–69%) |
| constant (mean capped RUL) | 5-fold engine-level CV on train, 500 cuts | 45.5 | – | 60% |
| ridge on window features | 5-fold engine-level CV on train, 500 cuts | 19.3 | – | 60% |
| gradient boosting | 5-fold engine-level CV on train, 500 cuts | 16.8 | – | 56% |
| 1D-CNN, pool head (1 seed per fold) | 5-fold engine-level CV on train, 500 cuts | 23.7 | – | 52% |
| 1D-CNN, flatten head (1 seed per fold) | 5-fold engine-level CV on train, 500 cuts | 18.6 | – | 54% |
| gradient boosting, **random window split** | 5-fold, window-level, capped truth | 7.0 (vs 13.7 engine-level) | – | – |

RUL cap 125 (chosen by CV among 100/125/150), window 30 cycles, served model: gradient boosting. CNN: flatten head, 8,705 parameters, 3 seeds; 5-fold CV took 9s and all final training 70s on CPU.

What the numbers say, in one paragraph: the engine-level split matters (a
random window split roughly halves the apparent error); gradient boosting on
window features was the best model on cross-validation and on the test set;
the compact CNN improved markedly when its head kept the temporal position
(flatten rather than global average pooling) but still did not beat the
trees, and on the test set the paired difference was not significant (see
Step 7). The NASA score is
dominated by a handful of late predictions in the 25–75-cycle range (Step 8).

## Limitations

* **Simulated data.** C-MAPSS is generated by a simulator with injected
  degradation; real engines have maintenance actions, sensor faults,
  different usage and censoring (most engines are removed before failure).
  Results do not transfer to a real fleet.
* **RUL cap is arbitrary** and changes RMSE substantially; comparisons with
  papers using a different cap are not like-for-like.
* **Small test set** (on the order of 100 engines — count it): CIs are wide,
  and the NASA score is dominated by a few late predictions.
* **No censoring in training** — every training engine runs to failure,
  which is rarely true in practice; a real system needs survival-style
  methods.
* FD001 only in the MVP: one operating condition, one fault mode.

## Interview questions

1. **Why can't you split windows randomly?** Strong answer: overlapping
   windows share most of their samples; the model memorises engine-specific
   offsets; analogy to within- vs cross-subject EEG; you measured the gap.
2. **How did you choose the RUL cap, and what does it do to the metrics?**
   Covers: early-life degradation unobservable; cap as a modelling assumption;
   sensitivity table; evaluated against uncapped truth.
3. **Explain the NASA score. Why asymmetric? What does it reward?** Covers:
   late = failure risk; exponential so dominated by worst engines; can be
   gamed by systematic under-prediction — show the early/late split.
4. **Did the CNN/GRU beat gradient boosting on window features? By how much,
   with what uncertainty?** Covers: same folds, bootstrap over engines, seeds;
   honest if it did not.
5. **How do you handle test engines shorter than your window?** Covers:
   padding strategy identical in training and serving, a test for it.
6. **How would this change with six operating conditions (FD002)?** Covers:
   condition-wise normalisation; clustering settings; leakage-safe fitting.
7. **What would you need to deploy this on a real fleet?** Covers: censored
   data, maintenance events, sensor drift/faults, per-engine baselines,
   decision thresholds tied to inspection cost, monitoring with delayed labels.
8. **How do you turn RUL into a maintenance decision?** Covers: threshold or
   quantile (e.g. schedule when the 10th percentile of RUL < lead time);
   asymmetric costs; capacity.
9. **How would you monitor this model after deployment?** Covers: input drift
   per sensor, prediction drift, labels only arrive at failure/removal,
   periodic backtests.
10. **What in this project is closest to your research, and what is new?**
    Covers: cross-subject ↔ cross-engine; new: API, contracts, reproducibility,
    asymmetric decision metrics.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `cmapss_rul`)
- [ ] `make data` downloads from the NASA page, checks SHA-256, asserts 26 columns
- [ ] `DATA_CARD.md` with terms checked on the landing page and the PHM08 citation
- [ ] Test: no engine ID in both train and validation in any fold
- [ ] Unit tests for RUL label, windowing/padding, NASA score
- [ ] Baselines + temporal model, same folds, seeds reported
- [ ] Results table (RMSE, NASA score, % late, 95% CI over engines) — filled by the walkthrough's final step from computed numbers (see [Results](#results)); in your repository, **TBD — fill from your run**
- [ ] Random-split vs engine-split comparison — likewise produced by the walkthrough (Step 6b)
- [ ] Early/late analysis figure and RUL-binned errors
- [ ] FastAPI service with validation tests and not-ready test
- [ ] `MODEL_CARD.md` stating "simulated data" and the RUL cap prominently
- [ ] CI green (ruff + pytest on toy data, no download)

## Stretch: connect to the churnlab patterns of Part I

* Save each trained configuration (cap, window, architecture) as an
  immutable registry version with metadata, using the pattern in
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py);
  promote the best and practise a rollback.
* Monitoring: compute PSI per sensor between training engines and a batch of
  "incoming" test engines with
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py);
  then simulate a sensor offset on a few engines and check whether drift
  alerts fire before RMSE degrades.
* Add a `/metrics` endpoint with latency percentiles, as in the Part I API
  (Chapter 14).
