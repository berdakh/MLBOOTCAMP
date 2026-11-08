# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap3-retail-demand-forecasting/walkthrough.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Capstone 3 · Retail demand forecasting (UCI Online Retail) — guided walkthrough
#
# **Capstone · about 2 minutes to run on a laptop CPU (plus a 45 MB download) · CPU only**
#
# This notebook is a complete worked solution of the capstone's minimum viable product (MVP), written as
# if a tutor were sitting next to you. It forecasts how many units of each product a UK online gift
# retailer will sell in each of the next four weeks, from the public UCI *Online Retail II*
# transactions. The data are messy in instructive ways (cancellations, postage lines, duplicated rows,
# a few enormous orders that were cancelled minutes later) and most products sell rarely and
# irregularly, which is where naive forecasts are hard to beat.
#
# **How to use this walkthrough.** Attempt each step yourself first. Every step states its goal, then a
# **Your turn.** box asks you to make a decision and suggests how long to spend on it. Only then read
# **Our solution.**, which shows our code and explains why we chose it, which alternatives we
# considered and which mistakes are common. **Check.** cells assert properties that any correct
# solution must satisfy (for example, that no feature uses a week after the forecast origin); if you
# replace our code with yours, keep the checks.
#
# The steps follow the brief's milestones ([README.md](README.md)):
#
# 1. framing; 2. data acquisition; 3. data audit and cleaning; 4. a leakage-safe split; 5. baselines;
# 6. a gradient-boosted model; 7. evaluation with uncertainty; 8. error and slice analysis; 9. a small
# API; 10. write-up (results table generated from this run, limitations, next steps).
#
# **Data and licence.** *Online Retail II*, UCI Machine Learning Repository,
# <https://archive.ics.uci.edu/dataset/502/online+retail+ii>, DOI 10.24432/C5CG6D, licensed
# **CC BY 4.0** (you may share and adapt it, including commercially, if you give appropriate credit,
# link to the licence and indicate changes). Cite also D. Chen, S. L. Sain and K. Guo (2012),
# *Journal of Database Marketing & Customer Strategy Management* 19(3), 197–208. This is a public-data
# demonstration; it is not work for, or endorsed by, any retailer.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Imports and run mode ----------------------------------------------------
import hashlib
import json
import time
import urllib.request
import zipfile

import duckdb
import numpy as np
import pandas as pd

try:  # pandas reads .xlsx through openpyxl; Colab has it, a bare local env may not
    import openpyxl  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openpyxl"], check=True)

# MLBOOTCAMP_CAPSTONE_SAMPLE=1 is for automated checks only: it replaces the UCI download with a small
# SYNTHETIC file of the same format (Step 2). Numbers from that mode mean nothing about the retailer.
SAMPLE = os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1"
OUT = ART / "capstones" / "cap3"
OUT.mkdir(parents=True, exist_ok=True)
T_START = time.time()
pd.set_option("display.width", 140)
print("mode:", "SYNTHETIC SAMPLE (automated checks only)" if SAMPLE else "real UCI data")
print("outputs:", OUT)

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Forecast origin** | the last week whose sales are known when the forecast is made | end of week *t* (Sunday) |
# | **Horizon** | how many weeks ahead the forecast is for | *h* = 1, 2, 3, 4 |
# | **Product × week panel** | one row per product per week, including weeks with zero sales | 500 products × 104 weeks |
# | **Gross units** | units sold on valid sales lines, before subtracting returns | our label |
# | **Net units** | gross units minus cancelled/returned units in the same week | shown once, then rejected |
# | **Intermittent demand** | a series with many zero weeks between sales | a gift that sells every 3–4 weeks |
# | **ADI / CV²** | average interval between non-zero weeks / squared variation of the non-zero sizes | the two axes of the segment chart |
# | **Croston / SBA** | forecast size and interval of non-zero sales separately; SBA removes Croston's bias | the classic intermittent baseline |
# | **WAPE** | total absolute error divided by total units | 0.40 = errors equal to 40% of volume |
# | **Rolling origin** | evaluate by moving the origin forward week by week, using only the past each time | 12 test origins |
# | **Censoring (sales ≠ demand)** | recorded sales are capped by stock; a stock-out looks like low demand | a zero week for a best-seller |

# %% [markdown]
# ## Key equations
#
# Let $y_{i,t}$ be the units of product $i$ in week $t$ and $\hat y_{i,t}$ the forecast. Sums run over
# every product-week in the evaluation set $E$.
#
# **WAPE** (weighted absolute percentage error), our primary metric:
#
# $$
# \operatorname{WAPE} = \frac{\sum_{(i,t)\in E} \lvert y_{i,t} - \hat y_{i,t} \rvert}{\sum_{(i,t)\in E} y_{i,t}}
# $$
#
# *In words:* all the units we got wrong, as a share of all the units sold. It is defined when many
# weeks are zero (unlike MAPE, which divides by each $y_{i,t}$) and it is volume-weighted, so the
# best-sellers dominate it. Example: actuals (10, 0, 30), forecasts (12, 1, 20): (2 + 1 + 10) / 40 = 0.325.
#
# **MAE** and **bias**:
#
# $$
# \operatorname{MAE} = \frac{1}{\lvert E\rvert}\sum_{E} \lvert y - \hat y \rvert,
# \qquad
# \operatorname{bias} = \frac{\sum_{E} (\hat y - y)}{\sum_{E} y}
# $$
#
# *In words:* MAE is the average miss in units per product-week; bias is the net over- (positive) or
# under-forecast (negative) as a share of volume. Same example: MAE = 13/3 ≈ 4.33; bias = (2 + 1 − 10)/40 = −0.175.
#
# **MASE** (mean absolute scaled error) divides each product's errors by that product's in-sample
# one-step naive error $s_i = \operatorname{mean}_t \lvert y_{i,t} - y_{i,t-1} \rvert$ (training weeks only):
#
# $$
# \operatorname{MASE} = \frac{1}{\lvert E\rvert}\sum_{(i,t)\in E} \frac{\lvert y_{i,t} - \hat y_{i,t}\rvert}{s_i}
# $$
#
# *In words:* below 1 means better than the naive forecast was in-sample, whatever the product's volume.
#
# **Demand classification** (Syntetos–Boylan), on the training window only, with $p$ = number of
# non-zero weeks and $z_1, \dots, z_p$ their sizes:
#
# $$
# \operatorname{ADI} = \frac{\text{number of weeks}}{p},
# \qquad
# \operatorname{CV}^2 = \left(\frac{\operatorname{sd}(z)}{\operatorname{mean}(z)}\right)^2
# $$
#
# smooth: ADI < 1.32 and CV² < 0.49; erratic: ADI < 1.32, CV² ≥ 0.49; intermittent: ADI ≥ 1.32,
# CV² < 0.49; lumpy: both above. Example: 104 weeks with 26 sales weeks gives ADI = 4 (intermittent or lumpy).
#
# **Croston's method** keeps two exponentially smoothed quantities, updated only in weeks with a sale:
# the size $z$ and the interval $q$ since the previous sale. With smoothing constant $\alpha$:
#
# $$
# \begin{aligned}
# \hat z &\leftarrow \hat z + \alpha\,(y_t - \hat z), \qquad \hat q \leftarrow \hat q + \alpha\,(q_t - \hat q) \quad \text{if } y_t > 0\\
# \hat y^{\text{Croston}} &= \frac{\hat z}{\hat q}, \qquad \hat y^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right)\frac{\hat z}{\hat q}
# \end{aligned}
# $$
#
# *In words:* "about $\hat z$ units every $\hat q$ weeks", i.e. $\hat z / \hat q$ per week; the SBA factor
# corrects Croston's known upward bias. Example: $\hat z = 12$, $\hat q = 4$, $\alpha = 0.1$: Croston 3.0,
# SBA 2.85 units per week.

# %% [markdown]
# ## Step 1 · Framing
#
# **Goal.** Write down, before touching data, exactly what is predicted, for whom, from what, and how
# it will be judged. Vague framing is the most common reason capstones end with numbers nobody can
# interpret.
#
# > **Your turn (15 minutes).** Write one sentence each for: the decision supported; the unit of
# > prediction; the label; the forecast origin and horizon; the primary metric and why. Decide whether
# > the label is *gross* units or *net* units (gross minus returns) and justify it.
#
# **Our solution.**
#
# * **Decision supported:** a buyer reorders stock weekly and needs units per product for the next
#   four weeks.
# * **Unit of prediction:** one product (`StockCode`) × one week (Monday–Sunday).
# * **Label:** weekly **gross units sold** = the sum of `Quantity` over sales lines that are not
#   cancellations, have a positive price and a product stock code. Cancellations are *not* subtracted:
#   they often land in a different week from the sale and would create negative "demand". We report a
#   returns rate per product instead, and show the net alternative once (Step 3).
# * **Origin and horizon:** the forecast is made after week *t* closes, for weeks *t+1 … t+4*. We
#   allow one day of data latency: a sale cancelled within 24 hours is treated as never having
#   happened (Step 3), so the label of week *t* is final on Monday evening and forecasts run on Tuesday.
# * **Metric:** WAPE over all product-weeks (primary), with MAE, bias and MASE overall and per demand
#   segment. MAPE is unusable here: most product-weeks are zero.
# * **Scope:** the 300 best-selling products of the training window plus a random 200 of the other
#   products that were active in the last 13 training weeks, so that both the high-volume and the
#   long-tail regimes are evaluated. (Smaller in sample mode.)
#
# **Alternatives considered.** Daily forecasts (too sparse, and ordering is weekly); net units (rejected
# above); forecasting revenue (mixes price and quantity, and the buyer orders units).
#
# **Common mistakes.** Choosing "top products" by volume over the *whole* period, test weeks included
# (a survivorship leak), and leaving zero weeks out of the panel, which silently turns the problem into
# "how much, given that something sold".

# %%
FORECAST_HORIZON = 4
N_TOP, N_TAIL = (40, 30) if SAMPLE else (300, 200)
TEST_WEEKS, VALID_WEEKS = 12, 12
SEED = 0
print(f"horizon {FORECAST_HORIZON} weeks | scope: top {N_TOP} + random {N_TAIL} active tail products | "
      f"test {TEST_WEEKS} weeks, validation {VALID_WEEKS} weeks")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap3-rolling-origin.svg" alt="Timeline of 104 weeks split into training, 12 validation weeks and 12 test weeks; at each origin the features use weeks up to the origin and the targets are the next four weeks" width="100%">
#
# *What to notice:* features never look past the origin, and both the product scope and the demand
# segments are fixed using only weeks before the validation period starts.

# %% [markdown]
# ## Step 2 · Data acquisition
#
# **Goal.** Download the official file once, verify it is the file this walkthrough was written
# against, cache it in the artifact directory, and print the licence.
#
# > **Your turn (15 minutes).** Open <https://archive.ics.uci.edu/dataset/502/online+retail+ii> and read
# > the licence. Write a function that downloads the zip, checks its size and SHA-256 and caches it.
# > What should happen when the checksum differs?
#
# **Our solution.** We pin the size and SHA-256 we measured. A mismatch does not stop the notebook (UCI
# could repackage the same data) but prints a loud warning, because every number below would then need
# re-checking. Reading the Excel workbook takes about a minute, so we also cache a Parquet copy.
#
# The offline sample path writes a small **synthetic** workbook with the same sheet names, columns and
# kinds of defects (cancellations, postage lines, missing customer IDs, zero prices, an overlap between
# the two sheets and one huge order cancelled minutes later). It exists only so the notebook can be
# tested without network access; it is not the retailer's data.

# %%
UCI_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
EXPECTED_BYTES, EXPECTED_SHA256 = 45_622_418, "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
SHEETS = ["Year 2009-2010", "Year 2010-2011"]
COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]


def make_synthetic_online_retail(path: Path, seed: int = 0) -> Path:
    """SYNTHETIC stand-in with the official file's format (zip > xlsx > two sheets). For tests only."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2009-12-01", "2011-12-09", freq="D")
    days = days[days.dayofweek != 5]  # the real data has almost no Saturday sales
    codes = [f"{20000 + i}" + ("A" if i % 9 == 0 else "") for i in range(110)]
    base = np.where(np.arange(110) < 50, rng.uniform(2, 25, 110), rng.uniform(0.05, 0.4, 110))
    rows, inv = [], 500000
    for d in days:
        season = 1 + 0.8 * np.exp(-((d.dayofyear - 320) / 30) ** 2)
        for j in np.flatnonzero(rng.random(110) < np.minimum(base * season / 8, 0.95)):
            inv += 1
            q = int(rng.poisson(8 * season) + 1)
            cust = float(rng.integers(12000, 18000)) if rng.random() > 0.25 else np.nan
            price = 0.0 if rng.random() < 0.005 else round(float(rng.uniform(0.4, 9)), 2)
            t = d + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
            rows.append([str(inv), codes[j], f"ITEM {codes[j]}", q, t, price, cust, "United Kingdom"])
            if rng.random() < 0.02:  # a later cancellation of part of the line
                rows.append([f"C{inv}", codes[j], f"ITEM {codes[j]}", -max(1, q // 2), t + pd.Timedelta(days=3),
                             price, cust, "United Kingdom"])
        if rng.random() < 0.3:  # postage and other non-product lines
            inv += 1
            code = rng.choice(["POST", "M", "BANK CHARGES", "AMAZONFEE", "DOT"])
            rows.append([str(inv), code, str(code), 1, d + pd.Timedelta(hours=12), 18.0, np.nan, "France"])
        if rng.random() < 0.02:  # stock adjustment without a C invoice
            rows.append([str(inv), codes[0], "damaged", -3, d + pd.Timedelta(hours=9), 0.0, np.nan, "United Kingdom"])
    big = pd.Timestamp("2011-01-18 10:01")  # one enormous order, cancelled 16 minutes later
    rows += [["540001", codes[1], "ITEM BIG", 70000, big, 1.04, 12346.0, "United Kingdom"],
             ["C540002", codes[1], "ITEM BIG", -70000, big + pd.Timedelta(minutes=16), 1.04, 12346.0, "United Kingdom"]]
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("InvoiceDate", kind="stable")
    sheet1 = df[df.InvoiceDate < "2010-12-10"]  # the two official sheets overlap on 2010-12-01..09
    sheet2 = df[df.InvoiceDate >= "2010-12-01"]
    xlsx = path.with_suffix(".xlsx")
    with pd.ExcelWriter(xlsx) as w:
        sheet1.to_excel(w, sheet_name=SHEETS[0], index=False)
        sheet2.to_excel(w, sheet_name=SHEETS[1], index=False)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(xlsx, "online_retail_II.xlsx")
    xlsx.unlink()
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_online_retail(out: Path) -> Path:
    if SAMPLE:
        return make_synthetic_online_retail(out / "online_retail_II_SYNTHETIC.zip")
    path = out / "online_retail_II.zip"
    if not path.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, path)
    size, digest = path.stat().st_size, sha256(path)
    print(f"{path.name}: {size:,} bytes, sha256 {digest[:16]}…")
    if (size, digest) != (EXPECTED_BYTES, EXPECTED_SHA256):
        print("WARNING: this file differs from the one the walkthrough was written against; re-check every number.")
    return path


def load_raw(zip_path: Path) -> pd.DataFrame:
    cache = zip_path.with_suffix(".parquet")
    if cache.exists():
        return pd.read_parquet(cache)
    with zipfile.ZipFile(zip_path) as z:
        with z.open("online_retail_II.xlsx") as f:
            sheets = pd.read_excel(f, sheet_name=None, dtype={"Invoice": str, "StockCode": str, "Description": str})
    frames = [s.assign(sheet=name) for name, s in sheets.items()]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(cache)
    return raw


t0 = time.time()
zip_path = fetch_online_retail(OUT)
raw = load_raw(zip_path)
print(f"loaded {len(raw):,} rows from {raw.sheet.nunique()} sheets in {time.time() - t0:.0f} s")
print(f"licence: CC BY 4.0, {UCI_PAGE} (DOI 10.24432/C5CG6D). Credit UCI and Chen, Sain & Guo (2012); "
      "state that the data were cleaned and aggregated.")

# %% [markdown]
# **Expected (real data):** 1,067,371 rows from 2 sheets (the row count on the UCI page), read in
# roughly 40–90 seconds the first time. The second run reads the Parquet cache in about a second.

# %% [markdown]
# ## Step 3 · Data audit and cleaning
#
# **Goal.** Measure every known defect with a count before deciding what to do about it, then write
# the cleaning rules as SQL and test each rule on a tiny hand-made table.
#
# > **Your turn (45 minutes).** In DuckDB, count: rows and date range per sheet; rows that appear in
# > both sheets; cancellation invoices (starting with `C`); negative quantities that are *not*
# > cancellations; missing `Customer ID`; prices ≤ 0; stock codes that are not products; the largest
# > orders and whether they were cancelled. For each, decide: drop, keep, or keep separately.
#
# **Our solution.** First the schema and the two sheets.

# %%
con = duckdb.connect()
con.register("raw", raw)
q = lambda sql: con.sql(sql).df()  # noqa: E731
print(raw.dtypes.to_string())
q("""SELECT sheet, COUNT(*) AS rows, MIN(InvoiceDate) AS first, MAX(InvoiceDate) AS last
     FROM raw GROUP BY sheet ORDER BY sheet""")

# %% [markdown]
# The two sheets overlap: the first runs to 9 December 2010 and the second starts on 1 December 2010.
# Are the overlapping rows the same transactions? We compare them as multisets (`INTERSECT ALL`).

# %%
overlap = q(f"""
  SELECT (SELECT COUNT(*) FROM raw WHERE sheet = '{SHEETS[0]}' AND InvoiceDate >= TIMESTAMP '2010-12-01') AS sheet1_rows,
         (SELECT COUNT(*) FROM raw WHERE sheet = '{SHEETS[1]}' AND InvoiceDate <  TIMESTAMP '2010-12-10') AS sheet2_rows,
         (SELECT COUNT(*) FROM (
             SELECT Invoice, StockCode, Quantity, InvoiceDate, Price FROM raw
              WHERE sheet = '{SHEETS[0]}' AND InvoiceDate >= TIMESTAMP '2010-12-01'
             INTERSECT ALL
             SELECT Invoice, StockCode, Quantity, InvoiceDate, Price FROM raw WHERE sheet = '{SHEETS[1]}')) AS identical
""")
overlap

# %% [markdown]
# **Check.** Every overlapping row of the first sheet is an exact copy of a row in the second, so we
# keep the second sheet's copy and drop the first sheet's rows from 2010-12-01 on. Concatenating the
# sheets naively would double-count nine days of sales, just before Christmas.

# %%
o = overlap.iloc[0]
assert o.sheet1_rows == o.sheet2_rows == o.identical > 0, "the sheet overlap is not a pure duplicate"
print(f"duplicated rows across sheets: {int(o.identical):,}")

# %% [markdown]
# Now the other defects, each with a count. `PRODUCT_CODE` is our product rule: five digits optionally
# followed by letters (colour or size variants such as `85123A`). Everything else (postage, manual
# adjustments, bank charges, marketplace fees, gift vouchers, a test product, a handful of `DCGS…`
# web-shop codes) is not a product the buyer reorders.

# %%
PRODUCT_CODE = r"^[0-9]{5}[A-Za-z]*$"
con.sql(f"""CREATE OR REPLACE VIEW lines AS SELECT * FROM raw
            WHERE NOT (sheet = '{SHEETS[0]}' AND InvoiceDate >= TIMESTAMP '2010-12-01')""")
audit = q(f"""
  SELECT COUNT(*)                                                    AS rows_after_dedup,
         SUM(CASE WHEN Invoice LIKE 'C%' THEN 1 ELSE 0 END)          AS cancellation_rows,
         SUM(CASE WHEN Quantity < 0 AND Invoice NOT LIKE 'C%' THEN 1 ELSE 0 END) AS negative_qty_not_C,
         SUM(CASE WHEN "Customer ID" IS NULL THEN 1 ELSE 0 END)      AS missing_customer_id,
         SUM(CASE WHEN Price = 0 THEN 1 ELSE 0 END)                  AS zero_price,
         SUM(CASE WHEN Price < 0 THEN 1 ELSE 0 END)                  AS negative_price,
         SUM(CASE WHEN NOT regexp_matches(StockCode, '{PRODUCT_CODE}') THEN 1 ELSE 0 END) AS non_product_rows,
         COUNT(DISTINCT CASE WHEN regexp_matches(StockCode, '{PRODUCT_CODE}') THEN StockCode END) AS product_codes
  FROM lines""")
counts = audit.drop(columns="product_codes").T.rename(columns={0: "rows"})
print(counts.assign(share=lambda d: (d["rows"] / audit.rows_after_dedup[0]).round(4)).to_string())
print("distinct product codes:", int(audit.product_codes[0]))

# %%
q(f"""SELECT StockCode, ANY_VALUE(Description) AS description, COUNT(*) AS rows, SUM(Quantity) AS units
      FROM lines WHERE NOT regexp_matches(StockCode, '{PRODUCT_CODE}')
      GROUP BY StockCode ORDER BY rows DESC LIMIT 12""")

# %%
print("descriptions on negative-quantity lines that are not cancellations (stock adjustments):")
print(q("""SELECT COALESCE(Description, '(missing)') AS description, COUNT(*) AS rows FROM lines
           WHERE Quantity < 0 AND Invoice NOT LIKE 'C%' GROUP BY 1 ORDER BY rows DESC LIMIT 8""").to_string(index=False))
print("\nthe largest lines by absolute quantity:")
print(q("""SELECT Invoice, StockCode, Quantity, InvoiceDate, Price, "Customer ID" AS customer FROM lines
           ORDER BY ABS(Quantity) DESC LIMIT 6""").to_string(index=False))

# %% [markdown]
# **Expected (real data, measured):** after removing the 22,523 duplicated rows, 1,044,848 rows remain:
# 19,165 cancellation lines (1.8%), 3,393 negative quantities without a `C` invoice (0.3%), 235,287
# rows without a customer ID (22.5%), 6,024 zero prices (0.6%), 5 negative prices ("Adjust bad debt")
# and 5,993 non-product lines (0.6%: `POST` and `DOT` postage, `M` manual, `C2` carriage, `D`
# discount, …); 5,242 distinct product codes. Negative quantities without a `C` invoice mostly have no
# description, or descriptions such as "check", "damages" and "missing": stock adjustments, not sales.
# The two largest lines (80,995 and 74,215 units) were each cancelled by the same customer 12 and 16
# minutes later.
#
# **Decisions.**
#
# | Defect | Decision | Why |
# |---|---|---|
# | Sheet overlap | drop the first sheet's copy | exact duplicates (checked above) |
# | `C` invoices | not sales; kept separately as returns | they cancel earlier lines, often in another week |
# | Negative quantity without `C` | drop | stock adjustments ("damaged", "missing") |
# | Missing customer ID | **keep** | irrelevant for product-level units; would matter for customer features |
# | Price ≤ 0 | drop | free samples, adjustments, bad debt: not demand at a price |
# | Non-product codes | drop | postage and fees are not reordered stock |
# | Sale cancelled by the same customer within 24 h | drop the sale | order-entry errors such as the 80,995-unit line |
# | Exact repeated lines within an invoice | keep | cannot tell a double scan from a genuine repeat; stated as a limitation |
#
# The 24-hour window is what makes the extreme-order rule leakage-safe: a cancellation that arrives
# days later is a *return*, and at a forecast origin we would not yet know about it. We use only
# cancellations that arrive within the one day of data latency stated in Step 1.
#
# > **Intuition.** Cleaning is deciding which rows describe *the thing you forecast*. A postage line
# > is a real transaction but not a unit of stock; an order typed with three extra zeros and cancelled
# > fifteen minutes later is a real row but not demand. *Where this breaks down:* some decisions are
# > genuinely ambiguous (a large order cancelled a week later may be real demand that went elsewhere),
# > so state them and measure how much they change.
#
# The rules as one SQL function, so the same code cleans the real data and a hand-made test table.

# %%
def clean_sales_sql(source: str) -> str:
    """Valid product sales lines from `source`, minus sales cancelled by the same customer within 24 h."""
    return f"""
    WITH product AS (SELECT * FROM {source} WHERE regexp_matches(StockCode, '{PRODUCT_CODE}')),
         cancels AS (SELECT * FROM product WHERE Invoice LIKE 'C%')
    SELECT s.* FROM product s
    WHERE s.Invoice NOT LIKE 'C%' AND s.Quantity > 0 AND s.Price > 0
      AND NOT EXISTS (SELECT 1 FROM cancels c
                      WHERE c.StockCode = s.StockCode AND c."Customer ID" = s."Customer ID"
                        AND c.Quantity = -s.Quantity AND c.InvoiceDate >= s.InvoiceDate
                        AND c.InvoiceDate <= s.InvoiceDate + INTERVAL 24 HOUR)"""


def returns_sql(source: str) -> str:
    return f"""SELECT StockCode, InvoiceDate, -Quantity AS returned FROM {source}
               WHERE Invoice LIKE 'C%' AND Quantity < 0 AND regexp_matches(StockCode, '{PRODUCT_CODE}')"""


# A unit test per rule on a hand-made table: each row is labelled with whether it should survive.
ts = pd.Timestamp("2011-03-01 10:00")
toy = pd.DataFrame([
    ["1", "85123A", 6, ts, 2.5, 1.0, True],                       # a normal sale
    ["2", "POST", 1, ts, 18.0, 1.0, False],                        # postage
    ["3", "22423", 4, ts, 0.0, 1.0, False],                        # zero price
    ["4", "22423", -3, ts, 0.0, None, False],                      # stock adjustment
    ["C5", "22423", -2, ts, 3.0, 1.0, False],                      # a cancellation is not a sale
    ["6", "23843", 80995, ts, 2.08, 7.0, False],                   # cancelled 12 minutes later
    ["C7", "23843", -80995, ts + pd.Timedelta(minutes=12), 2.08, 7.0, False],
    ["8", "21212", 10, ts, 0.5, 9.0, True],                        # cancelled 5 days later = a return
    ["C9", "21212", -10, ts + pd.Timedelta(days=5), 0.5, 9.0, False],
    ["10", "84879", 2, ts, 1.7, None, True],                       # missing customer ID is kept
], columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "keep"])
con.register("toy", toy)
kept = set(q(clean_sales_sql("toy")).Invoice)
assert kept == set(toy.Invoice[toy.keep]), kept
assert q(returns_sql("toy")).returned.sum() == 2 + 80995 + 10
print("cleaning rules: all", len(toy), "hand-made cases pass")

# %%
con.sql(f"CREATE OR REPLACE TABLE sales AS {clean_sales_sql('lines')}")
con.sql(f"CREATE OR REPLACE TABLE returns AS {returns_sql('lines')}")
funnel = q(f"""
  SELECT 'raw rows' AS stage, (SELECT COUNT(*) FROM raw) AS rows UNION ALL
  SELECT 'after sheet de-duplication', (SELECT COUNT(*) FROM lines) UNION ALL
  SELECT 'product codes only', (SELECT COUNT(*) FROM lines WHERE regexp_matches(StockCode, '{PRODUCT_CODE}')) UNION ALL
  SELECT 'not C, quantity > 0, price > 0', (SELECT COUNT(*) FROM lines WHERE regexp_matches(StockCode, '{PRODUCT_CODE}')
         AND Invoice NOT LIKE 'C%' AND Quantity > 0 AND Price > 0) UNION ALL
  SELECT 'minus sales cancelled within 24 h', (SELECT COUNT(*) FROM sales)""")
removed = q("""SELECT COUNT(*) AS n, SUM(Quantity) AS units FROM (SELECT Invoice, StockCode, Quantity FROM lines
               WHERE Invoice NOT LIKE 'C%' AND Quantity > 0 AND Price > 0 EXCEPT ALL
               SELECT Invoice, StockCode, Quantity FROM sales) """).iloc[0]
print(funnel.to_string(index=False))
print(f"quick-cancel rule removed {int(removed.n):,} sales lines carrying {int(removed.units or 0):,} units")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap3-cleaning-funnel.svg" alt="Cleaning funnel: raw invoice lines, de-duplicate the sheet overlap, keep product codes, keep valid sales, drop sales cancelled within 24 hours, aggregate to a product-by-week panel with explicit zeros; cancellations go to a separate returns table" width="100%">
#
# *What to notice:* cancellations are not thrown away; they leave the sales path and become a
# returns table, which feeds a feature and the error analysis.
#
# **Expected (real data, measured):** 1,013,498 clean sales lines. The 24-hour rule removes 6,156 lines
# (0.6% of sales lines) carrying 240,901 units, about a third of them from the two enormous orders.
# Without this rule those two weeks would be the largest in the history of their products.
#
# ### The weekly panel, with explicit zero weeks
#
# We aggregate to ISO weeks (Monday start) and keep only **complete** weeks: the data start on a
# Tuesday and end on a Friday, so the first and last weeks are partial. The panel is a calendar ×
# product cross join, left-joined to the weekly sums, with `COALESCE(…, 0)`: a week without sales is a
# row with 0, not a missing row.

# %%
span = q("SELECT MIN(InvoiceDate) AS a, MAX(InvoiceDate) AS b FROM lines").iloc[0]
first_week = (span.a.normalize() - pd.Timedelta(days=span.a.dayofweek)) + pd.Timedelta(weeks=1 if span.a.dayofweek else 0)
last_week = (span.b.normalize() - pd.Timedelta(days=span.b.dayofweek)) - pd.Timedelta(weeks=0 if span.b.dayofweek == 6 else 1)
print(f"data {span.a.date()} → {span.b.date()} | complete weeks {first_week.date()} → {last_week.date()} (week starts)")

panel = q(f"""
  WITH weeks AS (SELECT CAST(w AS TIMESTAMP) AS week FROM range(TIMESTAMP '{first_week}', TIMESTAMP '{last_week}' + INTERVAL 1 DAY,
                                                INTERVAL 7 DAY) t(w)),
       products AS (SELECT DISTINCT StockCode FROM sales),
       wk AS (SELECT StockCode, date_trunc('week', InvoiceDate) AS week, SUM(Quantity) AS units,
                     ROUND(SUM(Quantity * Price) / SUM(Quantity), 4) AS avg_price  -- rounded: parallel float sums vary in the last digit
              FROM sales GROUP BY 1, 2),
       rt AS (SELECT StockCode, date_trunc('week', InvoiceDate) AS week, SUM(returned) AS returned FROM returns GROUP BY 1, 2)
  SELECT p.StockCode, w.week, COALESCE(wk.units, 0) AS units, wk.avg_price, COALESCE(rt.returned, 0) AS returned
  FROM products p CROSS JOIN weeks w
  LEFT JOIN wk ON wk.StockCode = p.StockCode AND wk.week = w.week
  LEFT JOIN rt ON rt.StockCode = p.StockCode AND rt.week = w.week
  ORDER BY p.StockCode, w.week""")
WEEKS = pd.DatetimeIndex(sorted(panel.week.unique()))
n_prod = panel.StockCode.nunique()
print(f"panel: {n_prod:,} products × {len(WEEKS)} weeks = {len(panel):,} rows; zero weeks {np.mean(panel.units == 0):.1%}")

# %% [markdown]
# **Check.** Every product has every week, exactly once, and there are no negative labels.

# %%
assert len(panel) == n_prod * len(WEEKS)
assert panel.groupby("StockCode").week.nunique().eq(len(WEEKS)).all()
assert (panel.units >= 0).all()
net = panel.units - panel.returned
print(f"net-units alternative: {np.mean(net < 0):.2%} of product-weeks would be negative "
      f"(returns {panel.returned.sum() / panel.units.sum():.1%} of gross units overall)")

# %% [markdown]
# **Expected (real data, measured):** 4,872 products × 104 complete weeks (2009-12-07 to the week
# starting 2011-11-28) = 506,688 rows, 61.9% of them zero weeks. Under the net-units label 0.47% of
# product-weeks would be *negative demand*, which is the reason we chose gross units. Returns are 3.6%
# of gross units.
#
# **Common mistakes here.** Aggregating only the weeks that have sales (no zero rows), using
# `date_trunc('month', …)` by habit, and forgetting the partial first and last weeks, which then look
# like a collapse in demand.

# %% [markdown]
# ## Step 4 · A leakage-safe split for this problem
#
# **Goal.** Rehearse exactly how the forecast will be used: at each origin only the past is known.
#
# > **Your turn (20 minutes).** Decide: the test period; the validation period; how the product scope
# > and the demand segments are chosen without looking at those periods; and how often the model is
# > retrained. Why is a random split of product-weeks wrong here?
#
# **Our solution.** Test = the last 12 complete weeks (September to early December 2011, including the
# pre-Christmas ramp, the hardest and most important period). Validation = the 12 weeks before that.
# Everything else is training. The **scope** (top 300 by training-window units + a random 200 of the
# remaining products active in the last 13 training weeks) and the **segments** (ADI/CV²) use only
# weeks before validation starts, so the same scope and segments serve both periods without leakage.
#
# The model is trained once per period (on targets that end before that period starts) and then used
# at every weekly origin inside it, with features recomputed from the latest known week. Retraining
# every week is possible but slower; we state the simplification.
#
# A random split would put week 40 of a product in training and week 39 in test: the model would
# interpolate between known neighbours, which never happens in production.

# %%
VALID_START = len(WEEKS) - TEST_WEEKS - VALID_WEEKS  # index of the first validation week
TEST_START = len(WEEKS) - TEST_WEEKS
wide = panel.pivot(index="StockCode", columns="week", values="units")
train_units = wide.iloc[:, :VALID_START].sum(axis=1)
active = wide.iloc[:, VALID_START - 13:VALID_START].sum(axis=1) > 0
top = train_units[active].sort_values(ascending=False).index[:N_TOP]
rest = train_units[active].index.difference(top)
tail = pd.Index(np.random.default_rng(SEED).choice(rest, size=min(N_TAIL, len(rest)), replace=False))
SCOPE = top.append(tail).sort_values()
Y = wide.loc[SCOPE].to_numpy(dtype=float)  # products × weeks
RET = panel.pivot(index="StockCode", columns="week", values="returned").loc[SCOPE].to_numpy(dtype=float)
PRICE = panel.pivot(index="StockCode", columns="week", values="avg_price").loc[SCOPE].ffill(axis=1).to_numpy(dtype=float)
is_top = np.isin(SCOPE, top)
print(f"train weeks {WEEKS[0].date()}..{WEEKS[VALID_START - 1].date()} ({VALID_START}) | "
      f"valid {WEEKS[VALID_START].date()}..{WEEKS[TEST_START - 1].date()} | test {WEEKS[TEST_START].date()}..{WEEKS[-1].date()}")
print(f"scope: {len(SCOPE)} products ({is_top.sum()} top, {(~is_top).sum()} tail); "
      f"share of training-window units covered: {train_units[SCOPE].sum() / train_units.sum():.1%}")

# %% [markdown]
# Segments by ADI and CV² on the training window only, as in the Key equations.

# %%
def demand_segment(y: np.ndarray) -> str:
    nz = y[y > 0]
    if len(nz) == 0:
        return "lumpy"
    adi = len(y) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
    if adi < 1.32:
        return "smooth" if cv2 < 0.49 else "erratic"
    return "intermittent" if cv2 < 0.49 else "lumpy"


SEGMENT = np.array([demand_segment(row[:VALID_START]) for row in Y])
seg_table = pd.DataFrame({"segment": SEGMENT, "top": is_top, "units": Y[:, :VALID_START].sum(1)}).groupby("segment").agg(
    products=("top", "size"), of_which_top=("top", "sum"), training_units=("units", "sum"))
seg_table["unit_share"] = seg_table.training_units / seg_table.training_units.sum()
seg_table

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap3-demand-segments.svg" alt="ADI on the horizontal axis and CV squared on the vertical axis, split at 1.32 and 0.49 into four quadrants: smooth, erratic, intermittent and lumpy, with the candidate forecasting method for each" width="100%">
#
# *What to notice:* the two cut-offs split products by *how often* they sell and *how variable* the
# order sizes are; the segment decides which baseline is hard to beat.
#
# **Expected (real data, measured):** training 2009-12-07 to 2011-06-13 (80 weeks), validation from
# 2011-06-20, test from 2011-09-12. The 500 products cover 53% of training-window units. Segments:
# 224 erratic (184 of them top products, 57% of units), 175 lumpy, 57 smooth (56 top, 25% of units)
# and 44 intermittent (2% of units). Smooth and erratic products carry over 80% of the volume.
#
# > **Intuition.** ADI is "how many weeks do I wait between sales"; CV² is "how surprised am I by the
# > size when a sale comes". A product that sells 20 units nearly every week is smooth; a product that
# > sells 0, 0, 0, 120 is lumpy, and no point forecast will be good for it.

# %% [markdown]
# ### Features available at the forecast origin
#
# The panel is a matrix `Y[product, week]`. For origin index `o` and horizon `h`, every feature reads
# only columns `≤ o`; the target is `Y[:, o + h]`. The one exception is the seasonal lag, the same week
# last year (`o + h − 52`), which is still in the past because `h ≤ 4`.

# %%
def features_at(o: int, h: int, Y=Y, RET=RET, PRICE=PRICE) -> pd.DataFrame:
    past = Y[:, : o + 1]
    sold = past > 0
    f = {f"lag_{k}": Y[:, o - k + 1] for k in range(1, 5)}
    f["same_week_last_year"] = Y[:, o + h - 52] if o + h - 52 >= 0 else np.full(len(Y), np.nan)
    for w in (4, 8, 13):
        f[f"mean_{w}"] = past[:, -w:].mean(1)
    f["std_8"] = past[:, -8:].std(1)
    f["zero_share_13"] = (past[:, -13:] == 0).mean(1)
    f["weeks_since_sale"] = np.where(sold.any(1), np.argmax(sold[:, ::-1], axis=1), o + 1)
    f["mean_to_date"] = past.mean(1)
    f["price"] = PRICE[:, o]
    f["returns_rate_26"] = RET[:, max(0, o - 25): o + 1].sum(1) / np.maximum(past[:, -26:].sum(1), 1)
    f["week_of_year"] = np.full(len(Y), (WEEKS[0] + pd.Timedelta(weeks=o + h)).isocalendar().week)
    f["horizon"] = np.full(len(Y), h)
    return pd.DataFrame(f)


def rows_for(origins, horizons=range(1, FORECAST_HORIZON + 1), last_target=None):
    out = []
    for o in origins:
        for h in horizons:
            if last_target is not None and o + h > last_target:
                continue
            X = features_at(o, h)
            X["product"], X["origin"], X["target_week"], X["y"] = np.arange(len(Y)), o, o + h, Y[:, o + h]
            out.append(X)
    return pd.concat(out, ignore_index=True)


FEATURES = list(features_at(60, 1).columns)
print(len(FEATURES), "features:", FEATURES)

# %% [markdown]
# **Check (leakage).** Overwrite every week after an origin with random numbers: the features at that
# origin must not change. This is the test that catches a rolling mean computed without a shift.

# %%
o_chk = VALID_START - 1
Y_fake = Y.copy()
Y_fake[:, o_chk + 1:] = np.random.default_rng(1).integers(0, 1000, Y_fake[:, o_chk + 1:].shape)
for h in range(1, FORECAST_HORIZON + 1):
    a = features_at(o_chk, h)
    b = features_at(o_chk, h, Y=Y_fake)
    pd.testing.assert_frame_equal(a, b)
print("features at an origin are unchanged when every later week is replaced: no look-ahead")

# %% [markdown]
# Training rows: origins from week 13 onwards (so the 13-week windows are full), with targets strictly
# before the period being evaluated. Evaluation rows: every origin whose targets fall inside the period.

# %%
def split_rows(period_start: int, period_end: int):
    train = rows_for(range(13, period_start), last_target=period_start - 1)
    evalr = rows_for(range(period_start - 1, period_end - 1), last_target=period_end - 1)
    evalr = evalr[evalr.target_week >= period_start].reset_index(drop=True)
    return train, evalr


tr_valid, ev_valid = split_rows(VALID_START, TEST_START)
tr_test, ev_test = split_rows(TEST_START, len(WEEKS))
assert tr_valid.target_week.max() < VALID_START <= ev_valid.target_week.min()
assert tr_test.target_week.max() < TEST_START <= ev_test.target_week.min()
assert ev_test.target_week.max() == len(WEEKS) - 1
print(f"validation: {len(tr_valid):,} training rows, {len(ev_valid):,} evaluation rows")
print(f"test:       {len(tr_test):,} training rows, {len(ev_test):,} evaluation rows "
      f"({ev_test.origin.nunique()} origins × up to {FORECAST_HORIZON} horizons × {len(Y)} products)")

# %% [markdown]
# ## Step 5 · Baselines
#
# **Goal.** Establish what "good" means before any model: the simplest forecasts a buyer could compute
# in a spreadsheet, plus the standard intermittent-demand method.
#
# > **Your turn (40 minutes).** Implement: all zeros; naive (last week); moving averages of 4 and 8
# > weeks; seasonal naive (same week last year); and Croston/SBA. Unit-test Croston on a series where
# > you can compute the answer by hand.
#
# **Our solution.** All baselines are functions of `(o, h)` that read only `Y[:, :o + 1]` (plus the
# seasonal lag). Croston/SBA is a few lines of NumPy, initialised with the first sale.

# %%
def croston(y: np.ndarray, alpha: float = 0.1, sba: bool = True) -> float:
    """One-step (= flat multi-step) Croston/SBA forecast from the history y."""
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z, q, last = float(y[nz[0]]), float(nz[0] + 1), nz[0]
    for t in nz[1:]:
        z += alpha * (y[t] - z)
        q += alpha * ((t - last) - q)
        last = t
    return (1 - alpha / 2 if sba else 1.0) * z / q


# hand-computed case: sales of 10 every 2 weeks -> z = 10, q = 2 -> Croston 5, SBA 4.75
assert abs(croston(np.array([0, 10, 0, 10, 0, 10.0]), sba=False) - 5.0) < 1e-9
assert abs(croston(np.array([0, 10, 0, 10, 0, 10.0])) - 4.75) < 1e-9
assert croston(np.zeros(5)) == 0.0


def baseline_forecasts(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    p, o = rows["product"].to_numpy(), rows["origin"].to_numpy()
    out["zeros"] = 0.0
    out["naive"] = rows["lag_1"].to_numpy()
    out["ma_4"] = rows["mean_4"].to_numpy()
    out["ma_8"] = rows["mean_8"].to_numpy()
    out["seasonal_naive"] = np.nan_to_num(rows["same_week_last_year"].to_numpy())
    sba_cache = {(i, oo): croston(Y[i, : oo + 1]) for i, oo in set(zip(p, o))}
    out["sba"] = [sba_cache[(i, oo)] for i, oo in zip(p, o)]
    return out


t0 = time.time()
base_valid = baseline_forecasts(ev_valid)
print(f"baselines on validation in {time.time() - t0:.1f} s")

# %% [markdown]
# Metric functions, unit-tested on the worked example from the Key equations.

# %%
def wape(y, f):
    return float(np.abs(y - f).sum() / max(y.sum(), 1e-9))


def bias(y, f):
    return float((f - y).sum() / max(y.sum(), 1e-9))


SCALE = np.abs(np.diff(Y[:, :VALID_START], axis=1)).mean(1)  # in-sample naive MAE, training weeks only


def mase(y, f, products):
    s = SCALE[products]
    ok = s > 0
    return float(np.mean(np.abs(y - f)[ok] / s[ok]))


y_ex, f_ex = np.array([10.0, 0, 30]), np.array([12.0, 1, 20])
assert abs(wape(y_ex, f_ex) - 0.325) < 1e-12 and abs(bias(y_ex, f_ex) + 0.175) < 1e-12


def score_table(rows: pd.DataFrame, preds: pd.DataFrame, by_segment: bool = True) -> pd.DataFrame:
    y, prod = rows["y"].to_numpy(), rows["product"].to_numpy()
    seg = SEGMENT[prod]
    out = []
    for name in preds.columns:
        f = preds[name].to_numpy()
        groups = [("all", np.ones(len(y), bool))]
        if by_segment:
            groups += [(s, seg == s) for s in ["smooth", "erratic", "intermittent", "lumpy"]]
        for g, m in groups:
            if m.sum() == 0:
                continue
            out.append({"method": name, "segment": g, "rows": int(m.sum()), "WAPE": wape(y[m], f[m]),
                        "MAE": float(np.abs(y[m] - f[m]).mean()), "bias": bias(y[m], f[m]),
                        "MASE": mase(y[m], f[m], prod[m])})
    return pd.DataFrame(out)


val_scores = score_table(ev_valid, base_valid)
val_scores[val_scores.segment == "all"].set_index("method").round(3)

# %% [markdown]
# **Expected (real data, validation, measured):** all zeros 1.000 (WAPE is exactly 1 by definition);
# naive 0.841; MA(4) 0.701; **MA(8) 0.675**; SBA 0.690; seasonal naive 1.058. Single weeks are noisy,
# so averaging more of them helps. Seasonal naive is the worst non-trivial method: one prior year is a
# noisy estimate of seasonality, and many products sold little or nothing a year earlier.
#
# **Common mistakes.** Computing the moving average *including* the target week; scoring Croston on
# the product's whole history including the period being evaluated; and leaving out the all-zeros
# forecast, which can win on MAE for very sparse products.

# %% [markdown]
# ## Step 6 · The improved model: one global gradient-boosted model
#
# **Goal.** Learn one model across all products that combines recent level, trend, sparsity, price and
# seasonality, and predicts each horizon directly.
#
# > **Your turn (45 minutes).** Train `HistGradientBoostingRegressor` on the training rows of the
# > validation period. Compare a Poisson loss with squared error on validation WAPE. Why might Poisson
# > suit counts that are often zero?
#
# **Our solution.** One global model, horizon as a feature (a "direct" multi-horizon strategy: no
# recursive feeding of forecasts). The Poisson loss models the log of the mean, so predictions are
# always positive and the model's errors scale with volume, which fits counts; squared error lets the
# few largest products dominate the fit. We choose the loss on validation, then retrain on all weeks
# before the test period.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor


def fit_gbm(rows: pd.DataFrame, loss: str = "poisson") -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=300, max_leaf_nodes=31, early_stopping=False,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    return model.fit(rows[FEATURES], rows["y"])


t0 = time.time()
gbm_val = {}
for loss in ["poisson", "squared_error"]:
    m = fit_gbm(tr_valid, loss)
    gbm_val[f"gbm_{loss}"] = np.clip(m.predict(ev_valid[FEATURES]), 0, None)
print(f"two models trained and scored in {time.time() - t0:.0f} s")
val_all = pd.concat([base_valid, pd.DataFrame(gbm_val, index=ev_valid.index)], axis=1)
val_scores = score_table(ev_valid, val_all)
val_scores[val_scores.segment == "all"].set_index("method").round(3).sort_values("WAPE")

# %% [markdown]
# **A forecast combination.** The GBM and the 8-week moving average make different kinds of error (the
# GBM reacts to recent weeks and seasonality, MA(8) is stable). The equal-weight average of two
# reasonable forecasts is a classic, hard-to-overfit improvement: it has no parameter to tune. We add
# it as a candidate and judge it on validation like every other method.

# %%
wape_of = lambda m: val_scores.query("segment == 'all' and method == @m").WAPE.iloc[0]  # noqa: E731
BEST_LOSS = "poisson" if wape_of("gbm_poisson") <= wape_of("gbm_squared_error") else "squared_error"
val_all["gbm_ma8_mean"] = 0.5 * (val_all[f"gbm_{BEST_LOSS}"] + val_all["ma_8"])
val_scores = score_table(ev_valid, val_all)
val_scores[val_scores.segment == "all"].set_index("method").round(3).sort_values("WAPE")

# %% [markdown]
# **Per-segment method choice (on validation).** It is honest, and common in practice, to use the
# method that works best for each demand segment. We pick, for each segment, the method with the lowest
# validation WAPE, and fix that choice before looking at the test period.

# %%
seg_val = val_scores[val_scores.segment != "all"].pivot(index="segment", columns="method", values="WAPE")
CHOICE = seg_val.idxmin(axis=1).to_dict()
print("validation WAPE by segment:")
print(seg_val.round(3).to_string())
print("\nchosen loss:", BEST_LOSS, "| chosen method per segment:", CHOICE)

# %% [markdown]
# **Expected (real data, validation, measured):** the Poisson GBM (WAPE ≈ 0.69) only ties MA(8) (0.675)
# and SBA (0.690); squared error is clearly worse (≈ 0.75). The equal-weight combination of the GBM and
# MA(8) is the best candidate (≈ 0.67). Per segment, validation picks MA(8) for smooth and lumpy
# products, the combination for erratic ones and the GBM for intermittent ones.
#
# This is a typical, honest outcome on weekly retail data: with ~100 weeks of history and noisy
# wholesale orders, a well-chosen moving average is very hard to beat, and the model's value shows
# mostly in combination. We do **not** go back and tune the GBM until it wins: every extra round of
# tuning on validation makes the validation number less trustworthy.
#
# **Alternatives considered.** LightGBM with a Tweedie loss (similar, slightly faster; in
# `requirements-part2.txt`); one model per horizon (four times the training cost for little gain at
# this size); a deep sequence model (overkill for ~500 series of 100 weeks). **Common mistakes:**
# tuning many hyperparameters on validation and then reporting validation numbers as the result;
# product-level aggregates (such as mean units) computed over the whole history including test weeks.

# %% [markdown]
# ## Step 7 · Evaluation on the test period, once, with uncertainty
#
# **Goal.** Retrain on everything before the test period, forecast the 12 test weeks from rolling
# origins, and report WAPE with a confidence interval from a bootstrap over **products** (the unit
# that is sampled; product-weeks of one product are not independent).
#
# > **Your turn (30 minutes).** Write the bootstrap. Why resample products and not rows?
#
# **Our solution.**

# %%
t0 = time.time()
gbm = fit_gbm(tr_test, BEST_LOSS)
test_pred = baseline_forecasts(ev_test)
test_pred["gbm"] = np.clip(gbm.predict(ev_test[FEATURES]), 0, None)
seg_of_row = SEGMENT[ev_test["product"].to_numpy()]
test_pred["gbm_ma8_mean"] = 0.5 * (test_pred["gbm"] + test_pred["ma_8"])


def column_for(method: str) -> str:
    """Validation names gbm_poisson / gbm_squared_error both refer to the retrained test model `gbm`."""
    return "gbm" if method in ("gbm_poisson", "gbm_squared_error") else method


test_pred["per_segment_choice"] = [test_pred.at[i, column_for(CHOICE.get(s, "gbm"))]
                                   for i, s in zip(test_pred.index, seg_of_row)]
print(f"trained on {len(tr_test):,} rows and scored {len(ev_test):,} test rows in {time.time() - t0:.0f} s")
test_scores = score_table(ev_test, test_pred)
test_scores[test_scores.segment == "all"].set_index("method").round(3).sort_values("WAPE")

# %%
def bootstrap_wape(rows, preds, methods, B=500, seed=SEED):
    """95% intervals for WAPE of each method, and for differences to the first method, resampling products."""
    rng = np.random.default_rng(seed)
    prod = rows["product"].to_numpy()
    y = rows["y"].to_numpy()
    n = len(Y)
    sum_y = np.bincount(prod, weights=y, minlength=n)
    abs_err = {m: np.bincount(prod, weights=np.abs(y - preds[m].to_numpy()), minlength=n) for m in methods}
    draws = {m: [] for m in methods}
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n)  # how often each product was drawn
        for m in methods:
            draws[m].append((w * abs_err[m]).sum() / (w * sum_y).sum())
    out = {}
    for m in methods:
        d = np.array(draws[m])
        out[m] = (wape(y, preds[m].to_numpy()), *np.percentile(d, [2.5, 97.5]))
        if m != methods[0]:
            diff = d - np.array(draws[methods[0]])
            out[f"{m} − {methods[0]}"] = (out[m][0] - out[methods[0]][0], *np.percentile(diff, [2.5, 97.5]))
    return pd.DataFrame(out, index=["WAPE", "ci_low", "ci_high"]).T


best_baseline = test_scores.query("segment == 'all' and method in ['naive', 'ma_4', 'ma_8', 'seasonal_naive', 'sba']") \
    .sort_values("WAPE").method.iloc[0]
METHODS = ["gbm", best_baseline, "gbm_ma8_mean", "per_segment_choice"]
ci = bootstrap_wape(ev_test, test_pred, METHODS)
print(f"best baseline on test: {best_baseline}")
ci.round(3)

# %% [markdown]
# **Check.** The gradient-boosted model's improvement over the best baseline is reported with its
# interval; if the interval for the difference contains 0, the honest statement is "no clear
# improvement".

# %%
diff = ci.loc[f"{best_baseline} − gbm"]
verdict = "the GBM is better" if diff.ci_low > 0 else ("the baseline is better" if diff.ci_high < 0 else "no clear difference")
print(f"WAPE({best_baseline}) − WAPE(gbm) = {diff.WAPE:+.3f}, 95% CI [{diff.ci_low:+.3f}, {diff.ci_high:+.3f}] → {verdict}")
seg_test = test_scores[test_scores.segment != "all"].pivot(index="segment", columns="method", values="WAPE")
print("\ntest WAPE by segment:")
print(seg_test[["naive", "ma_8", "seasonal_naive", "sba", "gbm", "gbm_ma8_mean", "per_segment_choice"]].round(3).to_string())
comb = ci.loc["gbm_ma8_mean − gbm"]
print(f"\nWAPE(gbm_ma8_mean) − WAPE(gbm) = {comb.WAPE:+.3f}, 95% CI [{comb.ci_low:+.3f}, {comb.ci_high:+.3f}]")

# %% [markdown]
# **Expected (real data, test, measured):** MA(8) WAPE 0.611 (95% CI 0.578–0.649), GBM 0.633
# (0.596–0.677), combination 0.611, per-segment choice ≈ 0.61–0.62. The GBM is *worse* than MA(8) by
# 0.023 WAPE, and the interval of that difference (−0.040 to −0.008) excludes zero: on this test period
# the baseline wins. The combination beats the GBM alone by 0.023 (interval excludes zero) and ties
# MA(8). By segment, SBA is best on smooth products (0.466) and MA(8) on intermittent ones. Bias is
# negative for every method except seasonal naive: the autumn ramp is under-forecast.
#
# What to write in a README: "a global Poisson GBM did not beat an 8-week moving average on the
# 12-week test period (WAPE 0.633 vs 0.611); an equal-weight combination matched the moving average".
# That is a result. A capstone that reports it honestly is stronger than one that tuned on the test
# set until the model won.
#
# > **Intuition.** Bootstrapping products asks "if the retailer had stocked a different but similar
# > set of 500 products, how much would WAPE move?" Resampling rows would pretend that 48 weeks of one
# > product are 48 independent products and give intervals that are far too narrow.

# %% [markdown]
# ## Step 8 · Error and slice analysis
#
# **Goal.** Find *where* the errors are: which weeks, which products, and whether returns or stock-outs
# explain them.
#
# > **Your turn (40 minutes).** Break test WAPE down by target week and by horizon; list the products
# > with the largest absolute errors; check whether high-return products have higher error; and look
# > for evidence that sales are not demand.
#
# **Our solution.** First by target week and horizon.

# %%
ev = ev_test.assign(gbm=test_pred.gbm, base=test_pred[best_baseline], segment=seg_of_row)
by_week = ev.groupby("target_week").apply(lambda d: pd.Series({
    "week_start": WEEKS[d.name].date(), "units": d.y.sum(), "WAPE_gbm": wape(d.y, d.gbm),
    "WAPE_base": wape(d.y, d.base), "bias_gbm": bias(d.y, d.gbm)}), include_groups=False)
by_h = ev.groupby("horizon").apply(lambda d: pd.Series({"WAPE_gbm": wape(d.y, d.gbm), "WAPE_base": wape(d.y, d.base)}),
                                   include_groups=False)
print(by_week.round(3).to_string())
print()
print(by_h.round(3).to_string())

# %% [markdown]
# Largest misses, the returns relationship, and cumulative four-week error (what an order covers).

# %%
desc = q("SELECT StockCode, ANY_VALUE(Description) AS description FROM sales GROUP BY 1").set_index("StockCode").description
per_prod = ev.groupby("product").apply(lambda d: pd.Series({"units": d.y.sum(), "abs_err": np.abs(d.y - d.gbm).sum(),
                                                            "bias": bias(d.y, d.gbm)}), include_groups=False)
per_prod["StockCode"] = SCOPE[per_prod.index]
per_prod["description"] = per_prod.StockCode.map(desc).str.slice(0, 32)
per_prod["segment"] = SEGMENT[per_prod.index]
print(per_prod.sort_values("abs_err", ascending=False).head(8).round(2).to_string(index=False))

rr = RET[:, :TEST_START].sum(1) / np.maximum(Y[:, :TEST_START].sum(1), 1)
per_prod["returns_rate"] = rr[per_prod.index]
per_prod["returns_band"] = pd.cut(per_prod.returns_rate, [-0.01, 0.0, 0.02, 0.05, 10], labels=["0", "0-2%", "2-5%", ">5%"])
print("\nWAPE by training returns rate:")
print(per_prod.groupby("returns_band", observed=True).apply(
    lambda d: pd.Series({"products": len(d), "WAPE_gbm": d.abs_err.sum() / max(d.units.sum(), 1)}),
    include_groups=False).round(3).to_string())

cum4 = ev.groupby(["product", "origin"]).filter(lambda d: len(d) == FORECAST_HORIZON).groupby(["product", "origin"])[["y", "gbm", "base"]].sum()
print(f"\n4-week cumulative WAPE (what one order covers): gbm {wape(cum4.y, cum4.gbm):.3f}, "
      f"{best_baseline} {wape(cum4.y, cum4.base):.3f}  (weekly: gbm {wape(ev.y, ev.gbm):.3f})")

# %% [markdown]
# **Sales ≠ demand: evidence from the data.** Stock levels are not in the data, so we cannot see
# stock-outs directly. But we can count a symptom: a *smooth* product (it sold in nearly every training
# week) that records a zero week sandwiched between two strong weeks. Zero demand from hundreds of
# customers for one week is implausible; an empty shelf is a likelier explanation.

# %%
smooth = SEGMENT == "smooth"
Ys = Y[smooth]
med = np.median(Ys[:, :TEST_START], axis=1, keepdims=True)
sandwich = (Ys[:, 1:-1] == 0) & (Ys[:, :-2] >= med) & (Ys[:, 2:] >= med)
print(f"smooth products: {smooth.sum()} | zero weeks between two weeks at or above the product's median: "
      f"{sandwich.sum()} ({sandwich.any(1).mean():.0%} of smooth products have at least one)")

# %% [markdown]
# **Expected (real data, measured):** units per week rise from about 53,000 to about 290,000 over the
# test period, and the GBM's bias is negative in 9 of the 12 weeks (the ramp is under-forecast). Its
# worst weeks (2011-09-26 and 2011-10-10, WAPE 0.69 and 0.84) are *over*-forecasts after a very large
# week, i.e. the model chased a spike. Error grows slowly with horizon (0.62 → 0.65). The single largest
# miss is product 23084 (a night light) with about 90,000 test units: it barely sold before the
# validation period, so it was ranked as an *intermittent* tail product, a reminder that segment labels
# frozen on the training window misclassify new launches. Products with no recorded returns have the
# highest WAPE (≈ 0.93): they are mostly new or sparse items. Cumulative 4-week WAPE (≈ 0.47) is much
# lower than weekly WAPE (0.63): weekly errors partly cancel, and the 4-week total is what an order
# covers. Only 4 "sandwiched" zero weeks appear among the 57 smooth products: weak evidence of
# stock-outs, which is all this data can give; without stock levels, censoring cannot be measured.
#
# **Common mistakes.** Reporting only the overall WAPE (it is dominated by a few products); treating the
# list of biggest misses as anecdotes instead of a pattern (here: wholesale spikes and the Christmas ramp).

# %% [markdown]
# ## Step 9 · A small API serving precomputed forecasts
#
# **Goal.** Serve versioned weekly forecasts by lookup, with the validation rules from the brief.
#
# > **Your turn (40 minutes).** Refit on all complete weeks, produce forecasts for the four weeks after
# > the last complete week, store them as a table with `model_version` and `forecast_origin`, and write a
# > FastAPI app factory with `/health`, `/ready`, `GET /forecast/{stock_code}` and `POST /forecast/batch`.
# > Test it in-process with `TestClient`.
#
# **Our solution.** Forecasts are produced in a weekly batch and the API is a lookup: features need the
# full weekly panel, forecasts are consumed weekly, and a lookup is cheap, cacheable and easy to roll
# back (point the API at last week's table). The per-segment method choice from validation decides
# which forecast each product gets.

# %%
from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

final_origin = len(WEEKS) - 1
tr_all = rows_for(range(13, len(WEEKS)), last_target=len(WEEKS) - 1)
gbm_final = fit_gbm(tr_all, BEST_LOSS)
recs = []
for h in range(1, FORECAST_HORIZON + 1):
    X = features_at(final_origin, h)
    g = np.clip(gbm_final.predict(X[FEATURES]), 0, None)
    for i, code in enumerate(SCOPE):
        method = column_for(CHOICE.get(SEGMENT[i], "gbm"))
        units = {"gbm": g[i], "sba": croston(Y[i]), "ma_4": Y[i, -4:].mean(), "ma_8": Y[i, -8:].mean(),
                 "gbm_ma8_mean": 0.5 * (g[i] + Y[i, -8:].mean()),
                 "naive": Y[i, -1], "zeros": 0.0, "seasonal_naive": Y[i, final_origin + h - 52]}[method]
        recs.append({"stock_code": code, "segment": SEGMENT[i], "method": method, "horizon": h,
                     "week_start": (WEEKS[-1] + pd.Timedelta(weeks=h)).date().isoformat(), "units": round(float(units), 2)})
FORECASTS = pd.DataFrame(recs)
META = {"model_version": "v1", "forecast_origin": (WEEKS[-1] + pd.Timedelta(days=6)).date().isoformat()}
FORECASTS.to_parquet(OUT / "forecasts_v1.parquet")
print(f"{len(FORECASTS):,} forecasts for {len(SCOPE)} products; origin {META['forecast_origin']}; methods "
      f"{FORECASTS.drop_duplicates('stock_code').method.value_counts().to_dict()}")

# %%
NON_PRODUCT = {"POST", "DOT", "M", "C2", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK", "ADJUST", "B", "PADS"}


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stock_codes: list[str] = Field(min_length=1, max_length=500)
    horizon_weeks: int = Field(default=4, ge=1, le=4)

    @field_validator("stock_codes")
    @classmethod
    def clean_codes(cls, v):
        codes = list(dict.fromkeys(c.strip() for c in v))
        if any(not c or len(c) > 20 for c in codes):
            raise ValueError("stock codes must be 1-20 characters")
        bad = [c for c in codes if c.upper() in NON_PRODUCT]
        if bad:
            raise ValueError(f"non-product codes are not forecast: {bad}")
        return codes


def create_app(forecasts: pd.DataFrame | None, meta: dict) -> FastAPI:
    app = FastAPI(title="retail demand forecasts", version="1.0")
    table = None if forecasts is None else {c: g.sort_values("horizon") for c, g in forecasts.groupby("stock_code")}

    def result(code: str, horizon: int) -> dict:
        g = table[code]
        g = g[g.horizon <= horizon]
        return {"stock_code": code, "segment": g.segment.iloc[0], "method": g.method.iloc[0],
                "weeks": [{"week_start": w, "units": max(0.0, float(u))} for w, u in zip(g.week_start, g.units)]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if table is None:
            raise HTTPException(503, "forecast table not loaded")
        return {"status": "ready", **meta}

    @app.get("/forecast/{stock_code}")
    def one(stock_code: str, horizon: int = Query(4, ge=1, le=4)):
        code = stock_code.strip()
        if code.upper() in NON_PRODUCT:
            raise HTTPException(422, f"{code} is not a product")
        if table is None or code not in table:
            raise HTTPException(404, f"no forecast for {code}")
        return {**meta, "results": [result(code, horizon)]}

    @app.post("/forecast/batch")
    def batch(req: BatchRequest):
        if table is None:
            raise HTTPException(503, "forecast table not loaded")
        known = [c for c in req.stock_codes if c in table]
        return {**meta, "results": [result(c, req.horizon_weeks) for c in known],
                "unknown_stock_codes": [c for c in req.stock_codes if c not in table]}

    return app


client = TestClient(create_app(FORECASTS, META))
code0 = SCOPE[0]
assert client.get("/health").json() == {"status": "ok"}
assert client.get("/ready").json()["forecast_origin"] == META["forecast_origin"]
r = client.post("/forecast/batch", json={"stock_codes": [code0, " 99999ZZ "], "horizon_weeks": 2}).json()
assert len(r["results"][0]["weeks"]) == 2 and r["unknown_stock_codes"] == ["99999ZZ"]
assert all(w["units"] >= 0 for w in r["results"][0]["weeks"])
assert client.post("/forecast/batch", json={"stock_codes": ["POST"]}).status_code == 422
assert client.post("/forecast/batch", json={"stock_codes": [code0], "horizon_weeks": 9}).status_code == 422
assert client.post("/forecast/batch", json={"stock_codes": [code0], "extra": 1}).status_code == 422
assert client.post("/forecast/batch", json={"stock_codes": []}).status_code == 422
assert client.get(f"/forecast/{code0}?horizon=5").status_code == 422
assert client.get("/forecast/POST").status_code == 422
assert TestClient(create_app(None, META)).get("/ready").status_code == 503
print("API: 10 in-process checks pass; example:", json.dumps(client.get(f"/forecast/{code0}?horizon=1").json())[:200])

# %% [markdown]
# **Common mistakes.** Computing features inside the request (slow, and a second copy of the feature
# code that drifts from training); inventing a forecast for an unknown product instead of listing it in
# `unknown_stock_codes`; returning negative units.

# %% [markdown]
# ## Step 10 · Write-up: results table, limitations, next steps
#
# **Goal.** Produce the README results table *from this run's numbers*, never by hand.
#
# > **Your turn (30 minutes).** Generate the markdown table in code. Then write the limitations,
# > leading with "sales ≠ demand", and two or three next steps.
#
# **Our solution.** The table below is printed from the variables computed above. When this notebook
# runs on the real UCI data, this is the table to paste into the capstone's README; in sample mode it
# is labelled as synthetic and must not be pasted anywhere.

# %%
def results_markdown() -> str:
    src = "SYNTHETIC SAMPLE — not a result" if SAMPLE else "UCI Online Retail II"
    lines = [f"Data: {src}. Test period: {TEST_WEEKS} weeks {WEEKS[TEST_START].date()} to {WEEKS[-1].date()} "
             f"(week starts), rolling weekly origins, horizons 1–{FORECAST_HORIZON}; {len(SCOPE)} products "
             f"({N_TOP} top + {N_TAIL} random active tail); {len(ev_test):,} product-weeks. 95% CIs: bootstrap "
             "over products (500 resamples).", "",
             "| Method | WAPE (95% CI) | MAE | Bias | MASE |", "|---|---|---|---|---|"]
    for m in ["zeros", "naive", "ma_4", "ma_8", "seasonal_naive", "sba", "gbm", "gbm_ma8_mean", "per_segment_choice"]:
        r = test_scores.query("segment == 'all' and method == @m").iloc[0]
        w = f"{r.WAPE:.3f}"
        if m in ci.index:
            w += f" ({ci.loc[m].ci_low:.3f}–{ci.loc[m].ci_high:.3f})"
        lines.append(f"| {m} | {w} | {r.MAE:.2f} | {r.bias:+.3f} | {r.MASE:.3f} |")
    lines += ["", "| Segment | Products | WAPE gbm | WAPE " + best_baseline + " | WAPE sba | WAPE gbm_ma8_mean | MAE gbm | Bias gbm |",
              "|---|---|---|---|---|---|---|---|"]
    for s in ["smooth", "erratic", "intermittent", "lumpy"]:
        t = test_scores[test_scores.segment == s].set_index("method")
        if t.empty:
            continue
        lines.append(f"| {s} | {int((SEGMENT == s).sum())} | {t.WAPE['gbm']:.3f} | {t.WAPE[best_baseline]:.3f} | "
                     f"{t.WAPE['sba']:.3f} | {t.WAPE['gbm_ma8_mean']:.3f} | {t.MAE['gbm']:.2f} | {t.bias['gbm']:+.3f} |")
    lines += ["", f"GBM ({BEST_LOSS} loss) vs {best_baseline}: WAPE difference {diff.WAPE:+.3f} "
              f"(95% CI {diff.ci_low:+.3f} to {diff.ci_high:+.3f}; positive = GBM better). gbm_ma8_mean vs gbm: "
              f"{comb.WAPE:+.3f} ({comb.ci_low:+.3f} to {comb.ci_high:+.3f}). 4-week cumulative WAPE: gbm "
              f"{wape(cum4.y, cum4.gbm):.3f}, {best_baseline} {wape(cum4.y, cum4.base):.3f}."]
    return "\n".join(lines)


RESULTS_MD = results_markdown()
(OUT / "results.md").write_text(RESULTS_MD + "\n", encoding="utf-8")
print(RESULTS_MD)
print(f"\nwhole notebook: {time.time() - T_START:.0f} s")

# %% [markdown]
# **Limitations** (state them in this order in your README):
#
# 1. **Sales ≠ demand.** Sales are censored by stock; Step 8 found zero weeks among steady sellers that
#    look like stock-outs. A model trained on sales learns to repeat past under-stocking. Estimating
#    true demand needs availability data (stock levels or out-of-stock flags), which this dataset lacks.
# 2. **Label choices change results.** Gross vs net units, keeping exact repeated lines, and the 24-hour
#    rule for cancelled orders are judgement calls, each counted above.
# 3. **One retailer, mostly wholesale customers, UK-centric, 2009–2011**; about two years of history,
#    so yearly seasonality is estimated from one previous year.
# 4. **No promotions, inventory, competitor prices or marketing data**; price features are sparse.
# 5. **Volume-weighted metric:** WAPE is dominated by the top products; we report segments for that
#    reason. The model was trained once per period rather than at every origin.
#
# **Next steps (extensions).** Probabilistic (quantile) forecasts and a simple order-up-to policy
# simulation with stated lead time and service level, clearly labelled as a simulation; hierarchical
# reconciliation (product → total); a demand-uncensoring experiment that treats "sandwiched" zero
# weeks of smooth products as missing rather than zero; weekly retraining in the backtest.
#
# **What to put in your repository.** The cleaning SQL with the hand-made test above, the panel and
# leakage checks as `pytest` tests, the results table printed by this step, a data card with the
# counts from Step 3 and the CC BY 4.0 attribution, and the API with its tests.
#
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ImportError: Missing optional dependency 'openpyxl'` | a bare local environment | `pip install openpyxl` (Colab has it) |
# | Download fails or is very slow | UCI is temporarily unavailable | retry later; the zip is cached in the artifact directory once downloaded |
# | Checksum warning | UCI repackaged the file | compare row counts with Step 2's expected line; re-check every number |
# | Results differ slightly from the expected lines | different library versions | the conclusions, not the third decimal, should match |
