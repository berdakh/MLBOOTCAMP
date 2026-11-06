---
title: "Capstone 1 · ERCOT load forecasting"
parent: Capstones
nav_order: 1
---

# Capstone 1 · Day-ahead hourly load forecasting for the Texas grid (ERCOT)
{: .no_toc }

**Recommended order: 2nd** · MVP ≈ 4–5 days at 3 h/day · CPU only

A public-data demonstration inspired by energy-industry forecasting problems.
It uses load data that ERCOT publishes on its website; it is not work for,
or endorsed by, ERCOT or any utility, retailer or trading company.

1. TOC
{:toc}

---

## Business question

*At the end of each day, what will ERCOT system load be in each hour of the
next day — and how wrong are we during the hours that matter most, the
afternoon peaks?*

**Who acts on it:** in a real setting, a load forecast feeds generation
scheduling, retail energy procurement and trading desks. Under-forecasting
a summer afternoon peak is the expensive error (scarcity prices, reliability
risk); a forecast that is excellent at 3 a.m. and poor at 5 p.m. is not a good
forecast. This project measures both overall and peak-hour error for that
reason.

## Why this project

* **Rolling-origin backtesting**: the evaluation protocol that time-series
  interviews probe hardest, with forecasts issued at a fixed time using only
  information available then.
* **Seasonal-naive baselines** that are surprisingly hard to beat, and the
  discipline to report that honestly.
* **Real time-series data plumbing**: hour-ending conventions, Central
  Prevailing Time, daylight-saving days with 23 or 25 hours, and revisions.
* **Decision-aware error metrics** (peak-hour error), not just one global MAE.
* Directly relevant to the Houston energy job market — framed as a
  demonstration, not as experience at an energy company.

## Data

| Fact | Status |
|---|---|
| Historical load archive: <https://www.ercot.com/gridinfo/load/load_hist> | **verified from an official page snippet** |
| One ZIP per year, going back to 2002 | **verified from an official page snippet** |
| The previous month is posted around the 9th of the following month, after settlement | **verified from an official page snippet** |
| Values are hourly load in MW, by weather zone since April 2003 (before that, by 11 control areas) | **verified from an official page snippet** |
| There are 8 weather zones | **verified from an official page snippet** |
| Zone column names COAST, EAST, FWEST, NORTH, NCENT, SOUTH, SCENT, WEST plus an ERCOT total | **to verify** |
| Each ZIP contains an Excel file, typically named like `Native_Load_YYYY.xlsx`; first column "Hour Ending" | **to verify** — file names and formats have changed across years |
| Timestamps are in Central Prevailing Time (i.e. local time with DST) | **to verify** (reported by a third party, not seen on the official page) |
| Hour-ending convention 01:00–24:00 | **to verify** |
| Daily report "Actual System Load by Weather Zone" (NP6-345-CD) exists | **verified** via the gridstatus documentation (<https://opensource.gridstatus.io>), not ERCOT itself |

**Terms of use.** ERCOT's website Terms of Use
(<https://www.ercot.com/help/terms>, updated 07/20/2023 — **verified from an
official page snippet**) say, in gist, that information is provided **"as
is"**, without warranties, and that ERCOT does not guarantee the accuracy of
compilations. No explicit open-data licence was found. The data portal / API
has **separate** terms (<https://www.ercot.com/help/terms/data-portal> —
**verified from an official page snippet**). Treat the data as *publicly
posted, as-is, with no stated open licence*: read both pages yourself before
downloading, do not commit the raw files to your repository, and download
them in `make data` instead.

**Citation / attribution.** Cite the source page and your download date,
e.g. *"ERCOT, Hourly Load Data Archive, https://www.ercot.com/gridinfo/load/load_hist,
accessed YYYY-MM-DD."*

**Mirror.** The open-source `gridstatus` Python library can fetch ERCOT load
(**verified from its documentation**). It is convenient for extensions, but
for the MVP prefer the official yearly archive so that your data card has one
authoritative source; if you use `gridstatus`, record its version.

**Size.** ~8,760 hourly rows per year (8,784 in leap years) × (8 zones +
total). Use **at least 5 recent complete years** so that every backtest fold
has several years of history. Measure and record your exact range.

**Known gotchas** (**to verify** on your files):

* **Daylight saving:** the spring-forward day has 23 hours, the fall-back day
  25, with one hour label duplicated (it may be flagged with a DST marker
  column, or simply repeated). Decide: convert to UTC (recommended — a
  continuous hourly index) and keep local time only for calendar features and
  peak-hour definitions.
* **Hour ending 24:00** means midnight of the *next* day in most date
  parsers. Parse explicitly; write a test with a hand-checked row.
* **Format drift across years** (column names, Excel vs CSV, sheet names).
  Write one parser per format, normalise to a common schema, assert row
  counts per year.
* **Revisions:** recent months may be revised after settlement; record your
  download date and do not mix files downloaded months apart without checking.
* **Anomalies:** Winter Storm Uri (February 2021) involved load shedding,
  and 2020 had pandemic-era demand changes. Plot those periods; decide
  whether to keep them in training and say so. Do not silently delete them.

**Weather (extension only).** Temperature is the main driver of load, but
weather data is not part of the MVP. Candidate sources (NOAA, Open-Meteo
historical APIs) are **to verify**: check their terms and the difference
between observed weather and *forecast* weather (see leakage traps).

## Problem framing

* **Unit of prediction:** one hour of ERCOT **system total** load (MW). Zones
  are an extension.
* **Forecast origin:** end of day *D* (after hour-ending 24:00 local).
* **Horizon:** the 24 hours of day *D+1* (horizons 1–24 h).
* **Information set:** all load up to and including the origin. In a real
  control room, telemetry is near real-time; the archive's monthly publication
  lag is a property of the *public* archive, not of the operator. State this
  assumption in your README.
* **Label:** actual hourly system load for day *D+1*, in MW.
* **Consequence for features:** for every target hour in *D+1*, lag 24 h is
  available (same hour on day *D*), lag 1 h is **not**. Use lags ≥ 24 h only
  (24, 48, 168, …), rolling means that end at the origin, and calendar
  features (hour, day of week, month, holidays).

## Baselines

1. **Seasonal naive, daily:** load at the same hour yesterday (lag 24).
2. **Seasonal naive, weekly:** load at the same hour one week ago (lag 168).
   Often stronger than (1) because weekdays and weekends differ.
3. **Average of the last k same-weekday-same-hour values** (e.g. k = 4).

Every improvement is reported **relative to the best of these**, per fold.

## Improvements

* **Gradient-boosted trees** (scikit-learn `HistGradientBoostingRegressor`,
  or LightGBM) on lag-24/48/168 features, rolling statistics up to the origin,
  hour × day-of-week, month, and a US federal holiday flag. CPU, seconds per
  fold.
* **Direct multi-horizon** setup: one model with "hour of the target day" as a
  feature, or 24 per-hour models — compare, and explain the trade-off.
* **Ridge with Fourier terms** for daily/weekly/annual seasonality as a
  transparent alternative.
* **Extension:** temperature features (see weather above), quantile
  regression for P10/P50/P90 forecasts, zone-level forecasts reconciled to
  the total.

## Evaluation protocol

**Rolling-origin backtest (expanding window).**

* Hold out the most recent complete year as the **final test year**; do not
  look at it until the end.
* On the years before it, run a rolling-origin backtest for model selection:
  e.g. 12 monthly folds — for each month *m*, train on everything before *m*,
  then produce day-ahead forecasts for every day in *m* (each day's forecast
  uses only data up to the previous midnight). Retrain monthly, not daily, to
  keep it cheap; say so.
* Final: the same procedure over the test year, run **once**.
* Report each metric per fold and overall; uncertainty from the spread over
  folds and a block bootstrap over **days** (not hours — hours in a day are
  correlated).

**Metrics.**

* **MAE (MW)** over all forecast hours — primary, easy to explain.
* **MAPE** as a secondary, interpretable percentage (load is never near zero,
  so MAPE is well-defined here, unlike retail).
* **Peak-hour MAE (MW).** *Definition chosen for this project:* hours ending
  **16:00–19:00 Central Prevailing Time on weekdays from June to September**
  (i.e. 15:00–19:00 clock time). Justification: it is defined *in advance*
  from the calendar, so it does not depend on the outcome being evaluated,
  and it covers the summer late-afternoon window when Texas system load is
  typically highest. **Check this on your data** — compute the hour of each
  day's maximum load by month and adjust the window if your data disagrees,
  documenting the change.
  * *Why not "top 5% load hours"?* Selecting hours by their **actual** load
    conditions on the outcome: it picks hours that were unexpectedly high,
    so every model looks biased low there (regression to the mean). Report it
    as a secondary diagnostic if you like, with that caveat.
* **Daily peak error:** `max(forecast for day) − max(actual for day)` in MW,
  and the signed bias — operators care whether the peak was under-forecast.
* Signed bias overall and by hour of day.

## Leakage traps

* **Lags shorter than the horizon.** Lag 1 h is not available when
  forecasting hour 18 of tomorrow at midnight. A test should assert every lag
  ≥ 24 h relative to the origin.
* **Rolling features computed over the target day** (centred windows,
  `rolling` without `shift`).
* **Random K-fold** on hours: the model sees the hours on either side of each
  test hour.
* **Scalers or target encodings fitted on all years.**
* **Observed weather as a feature.** Actual temperature for tomorrow is not
  known at the origin; using it gives an oracle forecast. If you add weather,
  either use archived *forecasts* or label the result "with perfect weather
  knowledge (upper bound)".
* **Holiday or anomaly handling decided after looking at the test year.**
* **DST duplicates** silently dropped or summed, shifting a day's hours.

## MVP (≈ 4–5 days × 3 h)

| Day | Milestone |
|---|---|
| 1 | Repo from template; read the terms; `make data` downloads ≥ 5 yearly ZIPs with checksums; parser per format; tests for hour-ending 24:00 and DST days; continuous UTC index; row counts per year |
| 2 | EDA: daily/weekly/annual seasonality, hour of daily peak by month (validates the peak window), Uri/2020 inspection; seasonal-naive baselines; backtest harness with an "origin" abstraction and a lag-≥-24 h test |
| 3 | Metrics (MAE, MAPE, peak-hour MAE, daily peak error, bias) with unit tests; baselines through the backtest; gradient boosting with lag/calendar features |
| 4 | Model selection on backtest folds only; final test year run once; block-bootstrap CIs; error by hour/month; API |
| 5 (buffer) | README, model card, data card; limitations; interview answers with your numbers |

**Extensions** (optional, all CPU):

* Weather: population-weighted temperature for a few major Texas cities,
  with observed-vs-forecast discussion (sources and terms **to verify**).
* Probabilistic forecasts (quantile GBM) with pinball loss and interval
  coverage.
* Zone-level forecasts and reconciliation to the system total.
* A small neural model (MLP or 1D-CNN over the last 7 days) — CPU is enough;
  report whether it beats the tree model, with the same folds.

## API design

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | model loaded; returns `model_version`, `trained_through` |
| POST | `/forecast` | day-ahead forecast given recent history |
| GET | `/forecast/{date}` | (optional) precomputed forecast for a date in the backtest |

`POST /forecast` request — the client sends the history it has:

```json
{
  "target_date": "2024-07-16",
  "timezone": "America/Chicago",
  "history": [
    {"hour_ending_utc": "2024-07-08T05:00:00Z", "load_mw": 50123.0},
    {"hour_ending_utc": "2024-07-08T06:00:00Z", "load_mw": 48810.5}
  ]
}
```

Response:

```json
{
  "model_version": "v2",
  "target_date": "2024-07-16",
  "forecasts": [
    {"hour_ending_local": "2024-07-16T01:00:00-05:00", "load_mw": 51234.7, "is_peak_window": false}
  ]
}
```

(Values are illustrative placeholders, not ERCOT data or model output.)

**Validation rules:** `target_date` must be the day after the last history
hour (local time) — anything else would need a different horizon; history
must cover at least the 168 h (+ 24 h) the features need, hourly, strictly
increasing, no duplicates in UTC; `load_mw` finite and within a plausible
range (e.g. 0 < x < 150,000 — choose from training data and document it);
timezone must be `America/Chicago`. Respond with 24, 23 or 25 forecasts on
DST days, and test all three cases.

**Precomputed vs online:** the model bundle carries the fitted model,
feature list and holiday calendar. Features are computed online from the
posted history with the **same** feature function as training. For a
dashboard, forecasts for the backtest period can be precomputed in batch and
served by `GET /forecast/{date}`.

## Limitations

* **No weather in the MVP**, so errors on unusually hot or cold days will be
  large; that is expected and should be shown, not hidden.
* **Archive vs real-time:** the public archive is published monthly and may
  be revised; a real operator has real-time telemetry and weather forecasts.
* **DST and hour-ending conventions** are easy to get subtly wrong; your
  tests are the only evidence they are right.
* **Structural change:** Texas load has grown over time (new large loads,
  population); the expanding-window model may lag trend shifts.
* **Extreme events** (e.g. Uri) are rare; any metric over a year averages
  them away; report them separately.
* System total only; zone-level and price forecasting are different problems.

## Interview questions

1. **Walk me through your backtest. What does the model know at each
   origin?** Strong answer: origin at end of day D, lags ≥ 24 h, expanding
   window, monthly retrain, test year untouched; a test enforces it.
2. **Why is the seasonal-naive baseline hard to beat, and by how much did you
   beat it?** Covers: strong weekly periodicity; your per-fold numbers and CI;
   where you did not beat it.
3. **How did you define peak hours and why not "top 5% of hours"?** Covers:
   ex-ante calendar window; selection on the outcome biases the metric;
   checked against hour-of-daily-max.
4. **How did you handle daylight saving time?** Covers: UTC index, local
   calendar features, 23/25-hour days, a test per case, API returns 23/25 rows.
5. **What would you add first to improve accuracy?** Covers: weather
   *forecasts* not actuals; the leakage risk; expected effect measured, not
   promised.
6. **How would you produce uncertainty, and how would you check it?**
   Covers: quantile models or residual bootstrap; pinball loss; coverage of
   intervals per season and on peaks.
7. **Your MAE is fine but the operator is unhappy. Why?** Covers: errors
   concentrated at peaks; bias sign; daily-peak error; asymmetric costs.
8. **How would you monitor this in production?** Covers: labels arrive fast
   (hourly), so performance monitoring is direct; per-hour bias alerts; data
   gaps; retraining cadence; rollback.
9. **How did you choose your training window? Should old years be used?**
   Covers: trend vs data quantity; compared windows in the backtest; anomalies.
10. **What is different between this and forecasting load for one
    building?** Covers: aggregation smooths noise; individual loads are
    spikier; different baselines and metrics.

## GitHub deliverables checklist

- [ ] Repository from the template; package renamed (e.g. `ercotload`)
- [ ] `DATA_CARD.md`: archive URL, years used, download date, terms page and gist, checksums, format differences per year
- [ ] Parser tests: hour-ending 24:00, spring-forward and fall-back days, row counts per year
- [ ] Backtest harness with an explicit origin; test that all lags ≥ 24 h
- [ ] Baselines (lag 24, lag 168) and at least one model on identical folds
- [ ] Results table: MAE, MAPE, peak-hour MAE, daily-peak bias, per fold and final test year — **TBD — fill from your run**
- [ ] Error by hour of day and month; extreme-event section
- [ ] FastAPI `/forecast` with DST-aware output and validation tests
- [ ] `MODEL_CARD.md` and Limitations (no weather, archive lag, revisions)
- [ ] CI green without downloading ERCOT data (toy series in tests)

## Stretch: connect to the bootcamp's churnlab patterns

* **Registry:** register a model per monthly retrain with its training cut-off
  in the metadata; promote only if the last backtest month beats the current
  model; practise a rollback with
  [`registry.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/registry.py).
* **Monitoring:** labels arrive within an hour, so implement *performance*
  monitoring directly: rolling 7-day MAE and peak-hour bias against alert
  thresholds (see `evaluate_rules` in
  [`monitoring.py`](https://github.com/berdakh/MLBOOTCAMP/blob/main/src/churnlab/monitoring.py)),
  plus PSI on lag features to catch data-pipeline breaks (e.g. a unit change
  from MW to GW).
