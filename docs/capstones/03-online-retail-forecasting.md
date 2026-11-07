# Capstone 3 · Weekly product demand forecasting on UCI Online Retail II

**Recommended order: 3rd** · MVP ≈ 5 days at 3 h/day · CPU only

A public-data demonstration inspired by retail and e-commerce forecasting
problems. It uses transaction data from one UK online gift retailer published
on the UCI Machine Learning Repository; it is not work for, or endorsed by,
Amazon or any other retailer, and nothing here describes their systems.


---

## Business question

*How many units of each product will be sold in each of the next four weeks,
so a buyer can decide how much to reorder?*

**Who acts on it:** a replenishment buyer or inventory planner. Over-forecast
ties up cash in stock; under-forecast causes stockouts and lost sales — and,
crucially, a stockout also **hides** demand from the data you will train on
next time. A forecast is useful only if its errors are understood per product
type: a few high-volume items carry most of the units, while most items sell
rarely and irregularly.

## Why this project

* **Messy transactional data**: cancellations, returns, non-product stock
  codes, missing customer IDs, extreme wholesale orders. You make and document
  cleaning decisions under ambiguity — the daily reality of applied roles.
* **SQL-heavy feature building** (DuckDB, as in bootcamp notebook 03):
  aggregating invoice lines to a product × week panel with explicit zero weeks.
* **Intermittent demand**: many zeros, where MAPE breaks and naive methods are
  hard to beat.
* **Sales ≠ demand**: the clearest dataset for discussing censoring, which
  interviewers for supply-chain and retail science roles ask about.

## Data

| Fact | Status |
|---|---|
| Online Retail: <https://archive.ics.uci.edu/dataset/352/online+retail> | **verified from an official page snippet** |
| Online Retail II: <https://archive.ics.uci.edu/dataset/502/online+retail+ii> | **verified from an official page snippet** |
| Licence for both: **Creative Commons Attribution 4.0 International (CC BY 4.0)** | **verified from an official page snippet** |
| DOIs: 10.24432/C5BW33 (Online Retail), 10.24432/C5CG6D (Online Retail II) | **verified from an official page snippet** |
| Online Retail II: 1,067,371 rows, 2009-12-01 to 2011-12-09 | **verified from an official page snippet** |
| Online Retail: 541,909 rows | **verified from an official page snippet**; its date range 2010-12-01 to 2011-12-09 is **to verify** |
| A UK-based, non-store online retailer of gifts; many customers are wholesalers | **verified from an official page snippet** |
| Columns `Invoice`/`InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`/`UnitPrice`, `Customer ID`/`CustomerID`, `Country` (names differ between the two datasets) | **to verify** |
| Distributed as Excel (Online Retail II with one sheet per year) | **to verify** |

**Use Online Retail II for the MVP.** Two years of history allows a
52-week seasonal baseline and a test period that has a full prior year.
Do **not** concatenate the two datasets: their periods overlap
(**to verify** exactly how), which would duplicate transactions.

**Licence gist (CC BY 4.0):** you may share and adapt the data, including
commercially, **provided you give appropriate credit**, link to the licence
and indicate changes. Confirm on the UCI page. Some mirrors label the data
differently (e.g. OpenML dataset 43368 is labelled CC0 — **verified from an
official page snippet** of that mirror); **follow the UCI licence**, which is
the original publisher's.

**Citation** (**verified from an official page snippet**): cite the UCI
dataset DOI above, and the paper

> D. Chen, S. L. Sain, K. Guo, "Data mining for the online retail industry: A
> case study of RFM model-based customer segmentation using data mining",
> *Journal of Database Marketing & Customer Strategy Management* 19(3),
> 197–208, 2012. DOI: 10.1057/dbm.2012.17

**Known gotchas** (all **to verify** — measure each, record counts in the
data card):

* **Cancellations:** invoice numbers starting with `C`, with negative
  quantities. Some cancel an earlier line; some have no matching sale in the
  data.
* **Non-product stock codes:** postage, manual adjustments, bank charges,
  marketplace fees and similar (`POST`, `M`, `BANK CHARGES`, `AMAZONFEE`,
  …). List every non-numeric-pattern code and decide one by one.
* **Missing customer IDs** on a large share of rows (often quoted around a
  quarter). They do not matter for product-level demand but do for any
  customer-level feature.
* **Zero or negative prices**, and stock adjustment lines with descriptions
  like "damaged" or "lost".
* **Extreme orders:** a handful of very large wholesale quantities, some
  cancelled shortly after. Find them; decide whether they are demand.
* **Partial final week:** the data ends on 2011-12-09; the last week is
  incomplete — drop it or forecast only complete weeks.
* **Description drift:** the same `StockCode` can have several descriptions;
  key on `StockCode`.

## Problem framing

* **Unit of prediction:** one product (`StockCode`) × one ISO week.
* **Scope:** products with at least one sale in the training period and
  active in the last N weeks before the forecast origin (define N). Optionally
  restrict the MVP to the top few hundred products by training-period volume
  plus a random sample of the long tail, so that both regimes are evaluated.
* **Forecast origin:** end of week *t*. **Horizon:** weeks *t+1 … t+4*.
* **Label (decision for this project):** weekly **gross units sold** =
  sum of `Quantity` on non-cancelled invoice lines with positive price and a
  product stock code. Cancellations/returns are **not** subtracted from the
  sales week (they often land in a different week and would create negative
  "demand"); they are reported separately as a returns rate per product.
  Justify this in the README and show the alternative (net units) once.
* **Zero weeks are explicit rows**: a product with no sales in a week has a
  label of 0, not a missing row. Build the panel with a calendar cross join.

## Baselines

1. **Naive:** last week's units.
2. **Moving average** of the last 4 and 8 weeks.
3. **Seasonal naive:** same week last year (possible with Online Retail II).
4. **For intermittent items:** Croston's method or its SBA variant (a few
   lines of NumPy; implement and unit-test it), and the all-zeros forecast —
   which can win on MAE for very sparse items and must be reported.

## Improvements

* **One global gradient-boosted model** across products (scikit-learn
  `HistGradientBoostingRegressor` with Poisson loss, or LightGBM with Tweedie
  loss) on: lags 1–4, 52; rolling means/std of 4, 8, 13 weeks; weeks since
  last sale; share of zero weeks; recent price; returns rate; week of year;
  product-level aggregates computed on the training window only.
* **Direct multi-horizon:** one model per horizon *h* = 1…4, or horizon as a
  feature; features always computed at the origin.
* **Segment-wise evaluation and model choice:** classify products by average
  demand interval (ADI) and squared coefficient of variation (CV²) into
  smooth / erratic / intermittent / lumpy (the Syntetos–Boylan scheme, cut-offs
  commonly ADI = 1.32 and CV² = 0.49), computed on the training window only.
  It is fine — and honest — to use different methods per segment.

## Evaluation protocol

**Chronological, rolling-origin.**

* **Final test period:** the last 12 complete weeks before 2011-12-09 (this
  includes the pre-Christmas ramp, which is both the hardest and the most
  business-relevant period — say so).
* **Validation:** a rolling-origin backtest over the 12–16 weeks before the
  test period, origins every week or two; retrain at each origin (or every
  4 weeks, stated).
* Everything (product scope, segment labels, encodings) is computed using
  data up to the origin only.
* Final test evaluated **once**. Uncertainty: block bootstrap over
  **products** (and spread across origins).

**Metrics.**

* **WAPE** = Σ|y − ŷ| / Σ|y| over all product-weeks in the evaluation —
  primary. Volume-weighted, defined with zeros, easy to explain.
* **MAE** per product-week, overall and by segment.
* **Bias** = Σ(ŷ − y) / Σy, overall and by segment — systematic over- or
  under-forecasting matters more than noise for ordering.
* **For intermittent / lumpy items:** MAE and bias by segment, and a scaled
  error such as MASE (scaled by in-sample naive MAE) so items of very
  different volume can be compared; note that MAPE is undefined with zeros.
  Optionally evaluate cumulative 4-week units, which is what an order covers.
* **Returns handling:** report the returns rate by segment and check whether
  products with high return rates have higher forecast error.

## Leakage traps

* **Rolling features without `shift`**, so the current week's sales leak
  into its own features.
* **Product scope chosen with the future:** selecting "top products" by
  volume over the *whole* dataset, including the test period (a survivorship
  leak). Rank on the training window only.
* **Segment labels (ADI/CV²) computed on all data.**
* **Cancellations matched to future invoices** when cleaning training data:
  a cancellation dated after the origin must not remove a sale before it in
  the training set at that origin.
* **Price features from the future** (average price over the whole period).
* **Random splits** of product-weeks.
* **Merging Online Retail and Online Retail II** and double-counting overlap.

## MVP (≈ 5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repo from template; read the UCI page and licence; `make data` downloads and checksums; load both Excel sheets into DuckDB; profile: row counts, `C` invoices, non-product codes, missing IDs, price ≤ 0, extreme orders — each with a count |
| 2 | Cleaning rules as SQL with a test per rule; product × week panel with explicit zero weeks (calendar cross join); returns table; label decision written in the README |
| 3 | Baselines (naive, MA, seasonal naive, Croston/SBA, zeros); ADI/CV² segments on training window; backtest harness; WAPE/MAE/bias/MASE with unit tests |
| 4 | Global GBM with Poisson/Tweedie loss; backtest model selection; per-segment comparison; final 12-week test run once; bootstrap CIs over products |
| 5 | Error analysis (top-volume misses, Christmas weeks, high-return items); API serving precomputed forecasts; README, model card, data card |

**Extensions** (optional, all CPU):

* Hierarchical forecasting: product → total, with reconciliation.
* Price effects and promotions (careful: price changes are sparse and
  confounded).
* Probabilistic forecasts and a simple order-up-to policy simulation
  evaluating service level vs stock — clearly labelled as a simulation with
  stated assumptions, not a business result.
* Customer-level features (wholesale vs retail customers), using only rows
  with a customer ID, with the bias that introduces discussed.

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

**Validation rules:** `horizon_weeks` in 1–4; 1–500 unique stock codes;
codes are trimmed strings of bounded length; non-product codes (postage,
fees) rejected with a clear message; unknown products are listed in
`unknown_stock_codes` rather than failing the whole request (and never get an
invented forecast). Forecasts are non-negative.

**Precomputed vs online:** everything is precomputed in a weekly batch job
(add a `make forecast` target to the template's Makefile) and stored as a versioned table with the origin date and
model version; the API is a lookup. Explain why: features need the full
weekly panel, forecasts are consumed weekly, and a lookup is cheap, cacheable
and easy to roll back.

## Limitations

* **Sales ≠ demand.** Recorded sales are censored by stock: weeks with zero
  sales may be stockouts, not zero demand, and stock levels are not in the
  data. A forecast trained on sales under-estimates demand for items that
  frequently stock out.
* **Returns and cancellations** blur what was actually bought; the label
  choice (gross vs net) changes results.
* **One retailer, mostly wholesale customers, UK-centric, 2009–2011** — not
  representative of modern e-commerce.
* **No promotions, inventory, prices of competitors or marketing data.**
* **Short history** (about two years), so annual seasonality is estimated from
  one prior year.
* Large-order outliers dominate WAPE for some products; the decision to keep
  or cap them is a judgement call you must state.

## Interview questions

1. **What exactly is your label, and why not net units?** Strong answer:
   gross units on valid lines; cancellations land in other weeks and create
   negative demand; showed both; returns tracked separately.
2. **Why WAPE and not MAPE?** Covers: zeros make MAPE undefined; WAPE is
   volume-weighted — which also means it is dominated by top sellers, so you
   report by segment too.
3. **How did you handle intermittent demand?** Covers: ADI/CV² segments
   computed on training data; Croston/SBA; zeros baseline; why point forecasts
   for lumpy items are of limited use; probabilistic alternative.
4. **Is your data demand? What would you need to estimate true demand?**
   Covers: censoring by stockouts; need inventory/availability; uncensoring
   methods; why a model can lock in under-stocking.
5. **Walk me through your cleaning rules. Which one changed results most?**
   Covers: counts per rule; tests; extreme orders; non-product codes.
6. **How did you build the weekly panel in SQL?** Covers: calendar × product
   cross join, left join of aggregated sales, `COALESCE` to 0, window functions
   for lags — with care that windows end at the origin.
7. **Your global model beats naive overall but not on intermittent items.
   What do you ship?** Covers: per-segment method choice, simplicity,
   monitoring per segment.
8. **How would you turn forecasts into order quantities?** Covers: lead time,
   review period, quantile forecasts, service level, holding vs stockout cost.
9. **What changes in production?** Covers: weekly batch, late-arriving data,
   new products (cold start via category/description similarity), monitoring
   bias per segment.
10. **Where could leakage hide in this project?** Covers: scope selection by
    future volume, segment labels, rolling features, cancellations after the
    origin.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `retailcast`)
- [ ] `DATA_CARD.md`: UCI URL, DOI, CC BY 4.0 with attribution gist, citation, counts for every data issue
- [ ] Cleaning in SQL (DuckDB) with one test per rule
- [ ] Panel with explicit zeros; test that every product has every week in range
- [ ] Leakage tests: features use only weeks ≤ origin; scope chosen on training window
- [ ] Baselines incl. seasonal naive and Croston/SBA; GBM; same backtest origins
- [ ] Results table: WAPE, MAE, bias, MASE overall and by segment — **TBD — fill from your run**
- [ ] Returns-rate analysis and error analysis of the Christmas weeks
- [ ] API serving precomputed, versioned forecasts; validation tests
- [ ] Limitations section leading with "sales ≠ demand"
- [ ] CI green on a small synthetic transactions fixture

## Stretch: connect to the bootcamp's churnlab patterns

* **Registry + batch outputs:** store each weekly forecast run as a versioned
  artifact (model + forecast table + origin) and point the API at `CURRENT`,
  so rolling back a bad week's forecasts is one pointer change —
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py).
* **Monitoring with delayed labels:** each week, score the forecasts made 1–4
  weeks earlier against actuals (bias and WAPE per segment) and alert on
  sustained bias; add PSI on the weekly-units distribution to catch a broken
  feed — patterns from
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py).
