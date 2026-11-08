# Capstone 3 · Retail demand forecasting (UCI Online Retail)

**Recommended order: 3rd** · MVP ≈ 5 days at 3 h/day · CPU only

**Guided solution:** [walkthrough.ipynb](walkthrough.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap3-retail-demand-forecasting/walkthrough.ipynb)
— a complete worked solution of the MVP, step by step: attempt each step yourself, then compare with
ours. It runs on the real UCI data in about two minutes on a laptop CPU.

A public-data demonstration inspired by retail and e-commerce forecasting problems. It uses the
transaction data of one UK online gift retailer published on the UCI Machine Learning Repository; it
is not work for, or endorsed by, Amazon or any other retailer, and nothing here describes their
systems.

---

## Business question

*How many units of each product will be sold in each of the next four weeks, so that a buyer can
decide how much to reorder?*

**Who acts on it:** a replenishment buyer or inventory planner. An over-forecast ties up cash in stock;
an under-forecast causes stock-outs and lost sales, and a stock-out also **hides** demand from the data
used for the next model. A forecast is useful only if its errors are understood by type of product: a
few high-volume items carry most of the units, while most items sell rarely and irregularly.

## Why this project

* **Messy transactional data:** cancellations, returns, non-product stock codes, missing customer IDs
  and extreme wholesale orders. You make and document cleaning decisions under ambiguity, which is a
  large part of applied work.
* **SQL feature building** (DuckDB, as in [Chapter 4](../../part1-foundations/ch04-sql-for-ml/README.md)):
  aggregating invoice lines to a product × week panel with explicit zero weeks.
* **Intermittent demand:** many zeros, where MAPE breaks down and simple methods are hard to beat.
* **Sales ≠ demand:** a clear dataset for discussing censoring, a standard question in supply-chain
  and retail science interviews.

## Data

| Fact | Status |
|---|---|
| Online Retail: <https://archive.ics.uci.edu/dataset/352/online+retail> | **verified from an official page snippet** |
| Online Retail II: <https://archive.ics.uci.edu/dataset/502/online+retail+ii> | **verified from an official page snippet**; downloaded by the walkthrough |
| Licence for both: **Creative Commons Attribution 4.0 International (CC BY 4.0)** | **verified from an official page snippet** |
| DOIs: 10.24432/C5BW33 (Online Retail), 10.24432/C5CG6D (Online Retail II) | **verified from an official page snippet** |
| Online Retail II: 1,067,371 rows, 2009-12-01 to 2011-12-09 | **verified by the walkthrough run** |
| Online Retail: 541,909 rows | **verified from an official page snippet**; its date range 2010-12-01 to 2011-12-09 is **to verify** |
| A UK-based, non-store online retailer of gifts; many customers are wholesalers | **verified from an official page snippet** |
| Online Retail II columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` (Online Retail uses `InvoiceNo`, `UnitPrice`, `CustomerID`) | Online Retail II **verified by the walkthrough run**; Online Retail names **to verify** |
| Online Retail II is a zip (45,622,418 bytes) holding one Excel file with two sheets, `Year 2009-2010` and `Year 2010-2011` | **verified by the walkthrough run** (SHA-256 recorded in the notebook) |

**Use Online Retail II for the MVP.** Two years of history allow a 52-week seasonal baseline and a test
period with a full prior year. Do **not** concatenate it with Online Retail: their periods overlap,
which would duplicate transactions. Note also that **the two sheets of Online Retail II overlap**
(2010-12-01 to 2010-12-09): the walkthrough found 22,523 rows that appear, identically, in both sheets
(**verified by the walkthrough run**).

**Licence gist (CC BY 4.0):** you may share and adapt the data, including commercially, **provided you
give appropriate credit**, link to the licence and indicate changes. Confirm on the UCI page. Some
mirrors label the data differently (for example OpenML dataset 43368 is labelled CC0, **verified from
an official page snippet** of that mirror); **follow the UCI licence**, which is the original
publisher's.

**Citation** (**verified from an official page snippet**): cite the UCI dataset DOI above, and the paper

> D. Chen, S. L. Sain, K. Guo, "Data mining for the online retail industry: A case study of RFM
> model-based customer segmentation using data mining", *Journal of Database Marketing & Customer
> Strategy Management* 19(3), 197–208, 2012. DOI: 10.1057/dbm.2012.17

**Known gotchas.** Each was measured in the walkthrough (Step 3) on Online Retail II, after removing the
sheet overlap (1,044,848 rows); record the counts in your data card.

* **Cancellations:** invoice numbers starting with `C`, with negative quantities: 19,165 rows (1.8%).
  Some cancel an earlier line; some have no matching sale in the data.
* **Non-product stock codes:** postage, manual adjustments, bank charges, marketplace fees, gift
  vouchers and similar (`POST`, `DOT`, `M`, `C2`, `D`, `BANK CHARGES`, `AMAZONFEE`, …): 5,993 rows
  (0.6%). List every code that does not match the product pattern and decide one by one.
* **Missing customer IDs:** 235,287 rows (22.5%). They do not matter for product-level demand but do
  for any customer-level feature.
* **Zero or negative prices:** 6,024 zero prices and 5 negative ones ("Adjust bad debt"); 3,393
  negative quantities without a `C` invoice, with descriptions such as "damages", "check" and
  "missing" (stock adjustments).
* **Extreme orders:** the two largest lines (80,995 and 74,215 units) were each cancelled by the same
  customer 12 and 16 minutes later. Decide whether such orders are demand.
* **Partial weeks:** the data start on a Tuesday and end on Friday 2011-12-09, so the first and last
  weeks are incomplete; forecast only complete weeks.
* **Description drift:** the same `StockCode` can have several descriptions; key on `StockCode`.

## Problem framing

* **Unit of prediction:** one product (`StockCode`) × one week (Monday to Sunday).
* **Scope:** products with at least one sale in the training period and active in the last N weeks
  before the forecast origin (define N). Optionally restrict the MVP to the top few hundred products by
  training-period volume plus a random sample of the long tail, so that both regimes are evaluated.
  (The walkthrough uses the top 300 plus a random 200 of the products active in the last 13 training
  weeks.)
* **Forecast origin:** end of week *t*. **Horizon:** weeks *t+1 … t+4*.
* **Label (decision for this project):** weekly **gross units sold** = the sum of `Quantity` on
  non-cancelled invoice lines with a positive price and a product stock code. Cancellations and returns
  are **not** subtracted from the sales week (they often land in a different week and would create
  negative "demand"); they are reported separately as a returns rate per product. Justify this in the
  README and show the alternative (net units) once.
* **Zero weeks are explicit rows:** a product with no sales in a week has a label of 0, not a missing
  row. Build the panel with a calendar cross join.

## Baselines

1. **Naive:** last week's units.
2. **Moving average** of the last 4 and 8 weeks.
3. **Seasonal naive:** the same week last year (possible with Online Retail II).
4. **For intermittent items:** Croston's method or its SBA variant (a few lines of NumPy; implement
   and unit-test it), and the all-zeros forecast, which can win on MAE for very sparse items and must
   be reported.

## Improvements

* **One global gradient-boosted model** across products (scikit-learn `HistGradientBoostingRegressor`
  with Poisson loss, or LightGBM with Tweedie loss) on: lags 1–4 and 52; rolling means and standard
  deviations over 4, 8 and 13 weeks; weeks since the last sale; share of zero weeks; recent price;
  returns rate; week of year; product-level aggregates computed on the training window only.
* **Direct multi-horizon:** one model per horizon *h* = 1…4, or horizon as a feature; features are
  always computed at the origin.
* **Segment-wise evaluation and model choice:** classify products by average demand interval (ADI) and
  squared coefficient of variation (CV²) into smooth / erratic / intermittent / lumpy (the
  Syntetos–Boylan scheme, with the common cut-offs ADI = 1.32 and CV² = 0.49), computed on the training
  window only. Using different methods for different segments is legitimate if the choice is made on
  validation.

## Evaluation protocol

**Chronological, rolling-origin.**

* **Final test period:** the last 12 complete weeks before 2011-12-09. This includes the pre-Christmas
  ramp, which is both the hardest and the most relevant period for the business; say so.
* **Validation:** a rolling-origin backtest over the 12–16 weeks before the test period, with origins
  every week or two; retrain at each origin (or every 4 weeks, stated).
* Everything (product scope, segment labels, encodings) is computed using data up to the origin only.
* The final test period is evaluated **once**. Uncertainty: bootstrap over **products** (and the
  spread across origins).

**Metrics.**

* **WAPE** = Σ|y − ŷ| / Σ|y| over all product-weeks in the evaluation: primary. It is volume-weighted,
  defined with zeros and easy to explain.
* **MAE** per product-week, overall and by segment.
* **Bias** = Σ(ŷ − y) / Σy, overall and by segment: systematic over- or under-forecasting matters more
  than noise for ordering.
* **For intermittent and lumpy items:** MAE and bias by segment, and a scaled error such as MASE
  (scaled by the in-sample naive MAE) so that items of very different volume can be compared; MAPE is
  undefined with zeros. Optionally evaluate cumulative 4-week units, which is what an order covers.
* **Returns handling:** report the returns rate by segment and check whether products with high
  return rates have higher forecast error.

## Results

The table below is generated by Step 10 of the [walkthrough](walkthrough.ipynb) from its run on the
real Online Retail II data (Python 3.12, scikit-learn 1.6.1, DuckDB 1.2.2, 2 CPU threads); it was not
typed by hand. Your own run should reproduce it up to small differences from library versions.

Data: UCI Online Retail II. Test period: 12 weeks 2011-09-12 to 2011-11-28 (week starts), rolling weekly origins, horizons 1–4; 500 products (300 top + 200 random active tail); 21,000 product-weeks. 95% CIs: bootstrap over products (500 resamples).

| Method | WAPE (95% CI) | MAE | Bias | MASE |
|---|---|---|---|---|
| zeros | 1.000 | 128.26 | -1.000 | 4.082 |
| naive | 0.724 | 92.90 | -0.058 | 3.509 |
| ma_4 | 0.617 | 79.13 | -0.116 | 3.162 |
| ma_8 | 0.611 (0.578–0.649) | 78.32 | -0.155 | 3.014 |
| seasonal_naive | 1.053 | 135.10 | +0.374 | 4.200 |
| sba | 0.649 | 83.22 | -0.273 | 3.596 |
| gbm | 0.633 (0.596–0.677) | 81.25 | -0.089 | 3.398 |
| gbm_ma8_mean | 0.611 (0.577–0.652) | 78.36 | -0.122 | 3.140 |
| per_segment_choice | 0.618 (0.583–0.659) | 79.30 | -0.137 | 3.315 |

| Segment | Products | WAPE gbm | WAPE ma_8 | WAPE sba | WAPE gbm_ma8_mean | MAE gbm | Bias gbm |
|---|---|---|---|---|---|---|---|
| smooth | 57 | 0.486 | 0.490 | 0.466 | 0.479 | 112.60 | -0.011 |
| erratic | 224 | 0.670 | 0.651 | 0.648 | 0.647 | 92.28 | -0.053 |
| intermittent | 44 | 0.746 | 0.626 | 0.791 | 0.676 | 86.27 | -0.261 |
| lumpy | 175 | 0.651 | 0.629 | 0.764 | 0.630 | 55.66 | -0.174 |

GBM (poisson loss) vs ma_8: WAPE difference -0.023 (95% CI -0.040 to -0.008; positive = GBM better). gbm_ma8_mean vs gbm: -0.023 (-0.032 to -0.015). 4-week cumulative WAPE: gbm 0.471, ma_8 0.470.

**Reading the table.** On this test period the global gradient-boosted model did **not** beat an
8-week moving average (WAPE 0.633 vs 0.611; the interval of the difference excludes zero), and an
equal-weight combination of the two matched the moving average. MASE above 1 everywhere means that
the 12 autumn weeks were harder than an average training week. These are results, not failures: with
about 100 weeks of noisy wholesale history, a well-chosen moving average is a strong baseline. Your
own repository's table must come from your own run; if you have not run it, the cell says
*TBD — fill from your run*.

## Leakage traps

* **Rolling features without `shift`**, so that the current week's sales leak into its own features.
* **Product scope chosen with the future:** selecting "top products" by volume over the *whole*
  dataset, test period included (a survivorship leak). Rank on the training window only.
* **Segment labels (ADI/CV²) computed on all data.**
* **Cancellations matched to future invoices** when cleaning training data: a cancellation dated after
  the origin must not remove a sale before it in the training set at that origin.
* **Price features from the future** (an average price over the whole period).
* **Random splits** of product-weeks.
* **Merging Online Retail and Online Retail II**, or the two sheets of Online Retail II, and
  double-counting the overlap.

## MVP (≈ 5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repository from the template; read the UCI page and licence; `make data` downloads and checksums; load both Excel sheets into DuckDB; profile row counts, `C` invoices, non-product codes, missing IDs, price ≤ 0 and extreme orders, each with a count |
| 2 | Cleaning rules as SQL with a test per rule; product × week panel with explicit zero weeks (calendar cross join); returns table; label decision written in the README |
| 3 | Baselines (naive, MA, seasonal naive, Croston/SBA, zeros); ADI/CV² segments on the training window; backtest harness; WAPE/MAE/bias/MASE with unit tests |
| 4 | Global GBM with Poisson/Tweedie loss; backtest model selection; per-segment comparison; final 12-week test run once; bootstrap CIs over products |
| 5 | Error analysis (top-volume misses, Christmas weeks, high-return items); API serving precomputed forecasts; README, model card, data card |

**Extensions** (optional, all CPU):

* Hierarchical forecasting: product → total, with reconciliation.
* Price effects and promotions (price changes are sparse and confounded).
* Probabilistic forecasts and a simple order-up-to policy simulation evaluating service level against
  stock, clearly labelled as a simulation with stated assumptions, not a business result.
* Customer-level features (wholesale vs retail customers), using only rows with a customer ID, with
  the bias this introduces discussed.

## API design

Forecasts are produced weekly in batch; the API serves them.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | forecast table loaded; returns `model_version`, `forecast_origin` |
| GET | `/forecast/{stock_code}?horizon=4` | precomputed forecasts for one product |
| POST | `/forecast/batch` | up to 500 stock codes at once |

`POST /forecast/batch` request:

```json
{"stock_codes": ["85123A", "22423"], "horizon_weeks": 4}
```

Response:

```json
{
  "model_version": "v1",
  "forecast_origin": "2011-09-11",
  "results": [
    {"stock_code": "85123A", "segment": "smooth", "method": "gbm_poisson",
     "weeks": [{"week_start": "2011-09-12", "units": 412.0}]},
    {"stock_code": "22423", "segment": "intermittent", "method": "sba",
     "weeks": [{"week_start": "2011-09-12", "units": 3.4}]}
  ],
  "unknown_stock_codes": []
}
```

(Stock codes and numbers are illustrative placeholders, not results.)

**Validation rules:** `horizon_weeks` in 1–4; 1–500 unique stock codes; codes are trimmed strings of
bounded length; non-product codes (postage, fees) are rejected with a clear message; unknown products
are listed in `unknown_stock_codes` rather than failing the whole request (and never receive an
invented forecast). Forecasts are non-negative.

**Precomputed vs online:** everything is precomputed in a weekly batch job (add a `make forecast`
target to the template's Makefile) and stored as a versioned table with the origin date and model
version; the API is a lookup. The reasons: features need the full weekly panel, forecasts are consumed
weekly, and a lookup is cheap, cacheable and easy to roll back.

## Limitations

* **Sales ≠ demand.** Recorded sales are censored by stock: weeks with zero sales may be stock-outs,
  not zero demand, and stock levels are not in the data. A forecast trained on sales under-estimates
  demand for items that often stock out.
* **Returns and cancellations** blur what was actually bought; the label choice (gross vs net) changes
  the results.
* **One retailer, mostly wholesale customers, UK-centric, 2009–2011:** not representative of modern
  e-commerce.
* **No promotions, inventory, competitor prices or marketing data.**
* **Short history** (about two years), so annual seasonality is estimated from one prior year.
* Large-order outliers dominate WAPE for some products; the decision to keep or cap them is a
  judgement that you must state.

## Interview questions

1. **What exactly is your label, and why not net units?** Strong answer: gross units on valid lines;
   cancellations land in other weeks and create negative demand; both were shown; returns are tracked
   separately.
2. **Why WAPE and not MAPE?** Covers: zeros make MAPE undefined; WAPE is volume-weighted, which also
   means that top sellers dominate it, so you report by segment too.
3. **How did you handle intermittent demand?** Covers: ADI/CV² segments computed on training data;
   Croston/SBA; the zeros baseline; why point forecasts for lumpy items are of limited use; the
   probabilistic alternative.
4. **Is your data demand? What would you need to estimate true demand?** Covers: censoring by
   stock-outs; the need for inventory or availability data; uncensoring methods; why a model can lock
   in under-stocking.
5. **Walk me through your cleaning rules. Which one changed results most?** Covers: counts per rule;
   tests; extreme orders; non-product codes.
6. **How did you build the weekly panel in SQL?** Covers: calendar × product cross join, left join of
   aggregated sales, `COALESCE` to 0, window functions for lags, with windows that end at the origin.
7. **Your global model beats naive overall but not on intermittent items. What do you ship?** Covers:
   per-segment method choice, simplicity, monitoring per segment.
8. **How would you turn forecasts into order quantities?** Covers: lead time, review period, quantile
   forecasts, service level, holding vs stock-out cost.
9. **What changes in production?** Covers: weekly batch, late-arriving data, new products (cold start
   via category or description similarity), monitoring bias per segment.
10. **Where could leakage hide in this project?** Covers: scope selection by future volume, segment
    labels, rolling features, cancellations after the origin.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `retailcast`)
- [ ] `DATA_CARD.md`: UCI URL, DOI, CC BY 4.0 with attribution gist, citation, counts for every data issue
- [ ] Cleaning in SQL (DuckDB) with one test per rule
- [ ] Panel with explicit zeros; a test that every product has every week in range
- [ ] Leakage tests: features use only weeks ≤ origin; scope chosen on the training window
- [ ] Baselines including seasonal naive and Croston/SBA; GBM; the same backtest origins
- [ ] Results table: WAPE, MAE, bias, MASE overall and by segment, filled from your own run (the walkthrough's final step generates it; until you run it, **TBD — fill from your run**)
- [ ] Returns-rate analysis and error analysis of the Christmas weeks
- [ ] API serving precomputed, versioned forecasts; validation tests
- [ ] Limitations section leading with "sales ≠ demand"
- [ ] CI green on a small synthetic transactions fixture

## Stretch: connect to the course's churnlab patterns

* **Registry and batch outputs:** store each weekly forecast run as a versioned artifact (model,
  forecast table and origin) and point the API at `CURRENT`, so that rolling back a bad week's
  forecasts is one pointer change —
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py)
  ([Chapter 15](../../part1-foundations/ch15-monitoring-and-rollback/README.md)).
* **Monitoring with delayed labels:** each week, score the forecasts made 1–4 weeks earlier against
  actuals (bias and WAPE per segment) and alert on sustained bias; add PSI on the weekly-units
  distribution to catch a broken feed — patterns from
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py).
