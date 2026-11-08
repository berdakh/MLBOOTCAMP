# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap2-electricity-load-forecasting/walkthrough.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up. No GPU is needed.*

# %% [markdown]
# # Capstone 2 · Electricity load forecasting (ERCOT) — guided walkthrough
#
# **About 5–10 minutes to run on a laptop CPU · 4–5 days of work if you attempt every step yourself first**
#
# This notebook is a complete, worked solution of the capstone's minimum
# viable product (MVP), written as if a tutor were sitting next to you. It
# follows the milestones of the [capstone brief](README.md): framing, data,
# audit, a leakage-safe protocol, baselines, an improved model, evaluation with
# uncertainty, error analysis, a small API and the write-up.
#
# ## How to use this walkthrough
#
# Attempt each step yourself before you read our solution. Every step has the
# same three parts:
#
# 1. **Your turn.** The goal of the step, the decision you have to make, and
#    roughly how long it should take. Write your own code in a scratch cell
#    first.
# 2. **Our solution.** Code, followed by *why* we did it this way, the
#    alternatives we considered, and the mistakes we see most often.
# 3. **Check.** Assertions that must hold whatever your design.
#
# Your numbers will differ from ours if you made different choices. What
# matters is that every choice is stated and every number comes from a run on
# a split you can name.
#
# > **Data and honesty notice.** The load data is published by ERCOT on its
# > [historical load archive](https://www.ercot.com/gridinfo/load/load_hist).
# > This is a public-data exercise; it is not work for, or endorsed by, ERCOT
# > or any utility, retailer or trading company. ERCOT's website
# > [Terms of Use](https://www.ercot.com/help/terms) say, in gist, that
# > information is provided "as is" without warranties; no explicit open-data
# > licence was found. Read the terms yourself before downloading, do not
# > commit the raw files, and cite the page and your download date.
# >
# > **Where this walkthrough was verified.** ERCOT's website could not be
# > reached from the machine on which this walkthrough was written and tested.
# > Every cell was executed on a **synthetic sample** with the same file layout
# > (see Step 2); **the run on the official data was not executed there**.
# > The download and parsing code follows the archive's documented layout, but
# > details marked *to verify* (file names, column names, how the repeated
# > daylight-saving hour is marked) must be checked against your files.

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
# --- Libraries and run mode --------------------------------------------------
import datetime as dt
import hashlib
import io
import json
import math
import re
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

T_START = time.time()
SEED = 0
TZ = "America/Chicago"           # Central Prevailing Time (CST/CDT) - to verify for your files

# MLBOOTCAMP_CAPSTONE_SAMPLE=1 runs every step on a SYNTHETIC hourly series with the
# archive's file layout, so the course's automated checks can execute this notebook
# without reaching ercot.com. Its numbers mean nothing.
SAMPLE = os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1"
DATA_DIR = ART / "cap2-ercot" / ("sample" if SAMPLE else "official")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
print("mode:", "SYNTHETIC SAMPLE (numbers are not results)" if SAMPLE else "official ERCOT archive")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Native load** | electricity consumed in the ERCOT region in one hour, in MW | the `ERCOT` column |
# | **Hour ending (HE)** | an hour is labelled by its *end*: HE 01:00 covers 00:00–01:00 | HE 24:00 is the last hour of a day |
# | **Central Prevailing Time** | local Texas clock time, switching between CST and CDT | 23-hour and 25-hour days |
# | **Forecast origin** | the moment the forecast is issued; nothing after it may be used | end of day D (midnight) |
# | **Day-ahead horizon** | the 24 (23, 25) hours of day D+1 | horizons 1–25 h |
# | **Seasonal naive** | "the same hour, one season ago" as the forecast | same hour last week |
# | **Rolling-origin backtest** | repeatedly move the origin forward, train on the past, forecast the next block | 12 monthly folds |
# | **Peak window** | hours defined *in advance* as those that matter most | HE 16–19, weekdays, June–September |
# | **Block bootstrap** | resample whole blocks of consecutive days to keep their correlation | 7-day blocks |
# | **MAE / MAPE** | mean absolute error in MW / in percent of the actual load | "MAE 1,100 MW" |

# %% [markdown]
# ## Step 1 · Framing the problem
#
# **Your turn (about 20 minutes).** Write down the question, who acts on the
# answer, the unit of prediction, the forecast origin, the horizon, the
# information set, the label and the metrics. Then decide which lags are
# allowed.
#
# **Our solution.**
#
# * **Question.** At the end of each day, what will ERCOT system load be in
#   each hour of the next day, and how wrong are we in the hours that matter
#   most?
# * **Who acts.** In a real setting: generation scheduling, retail
#   procurement and trading desks. Under-forecasting a summer afternoon peak is
#   the expensive error.
# * **Unit of prediction.** One hour of ERCOT **system total** load in MW
#   (zones are an extension).
# * **Origin and horizon.** Issued at the end of day D (after HE 24:00 local);
#   forecasts every hour of day D+1: 24 hours, or 23/25 on daylight-saving days.
# * **Information set.** All load up to and including the origin. A real
#   control room has telemetry within minutes; the *public* archive is posted
#   monthly. We assume the operator's situation and say so in the README.
# * **Label.** Actual hourly load on day D+1.
# * **Metrics.** MAE in MW over all hours (primary); MAPE; MAE in the
#   **peak window** defined in advance in the brief — hours ending 16:00–19:00
#   on weekdays from June to September; daily peak error and signed bias.
#
# **Which lags are allowed?** For a target hour on day D+1, the most recent
# hour we know is HE 24:00 of day D. So "the same hour yesterday" (lag 1 day)
# is the shortest seasonal lag we may use; "one hour ago" is not known at the
# origin. We express lags in **local calendar days**, not in 24 UTC hours: on a
# 25-hour day, "24 hours earlier" for the last hour of D+1 would land *after* the
# origin. That is a real leak that only shows up twice a year.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap2-day-ahead.svg" alt="Days D-6, D-1 and D feed the features; the origin is midnight; day D+1 is the target; a 1-hour lag is not known" width="100%">
#
# *What to notice:* the vertical line is the origin. Everything a feature uses
# must lie to its left.
#
# > **Intuition.** Day-ahead forecasting is a weather forecaster writing
# > tomorrow's hourly forecast tonight. She may use today's observations, the
# > same day last week and the calendar, but not tomorrow's 3 p.m. temperature.
# > *Where the analogy breaks down:* a weather forecaster also has physical
# > models of the atmosphere; our MVP has no weather input at all, so it cannot
# > anticipate a heat wave that has not started yet.

# %% [markdown]
# ## Key equations
#
# With $y_t$ the actual load and $\hat{y}_t$ the forecast for hour $t$, over
# $n$ forecast hours:
#
# $$
# \operatorname{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left\lvert \hat{y}_t - y_t \right\rvert,
# \qquad
# \operatorname{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{\left\lvert \hat{y}_t - y_t \right\rvert}{y_t}
# $$
#
# *In words:* the average miss in MW, and the average miss as a share of the
# load. Example: actual $(40\,000, 50\,000)$ MW, forecast $(41\,000, 48\,000)$:
# MAE $= (1000 + 2000)/2 = 1500$ MW; MAPE $= (2.5\% + 4\%)/2 = 3.25\%$. Load is
# never near zero, so MAPE is well defined here.
#
# **Peak-window MAE** is the same MAE restricted to the set $P$ of peak hours,
# defined from the calendar before looking at any outcome:
#
# $$
# \operatorname{MAE}_{\text{peak}} = \frac{1}{\lvert P \rvert} \sum_{t \in P} \left\lvert \hat{y}_t - y_t \right\rvert
# $$
#
# **Daily peak error** for day $d$ with hours $H_d$:
#
# $$
# e_d = \max_{t \in H_d} \hat{y}_t - \max_{t \in H_d} y_t
# $$
#
# *In words:* did we forecast the day's highest load too high ($e_d > 0$) or
# too low ($e_d < 0$)? Example: forecast peak 70,500 MW, actual 72,000 MW
# gives $e_d = -1500$ MW, an under-forecast.
#
# **Seasonal naive (weekly)**: $\hat{y}_{d,h} = y_{d-7,h}$ — the same local
# hour, seven days earlier.
#
# We code the metrics now and check them on these numbers.

# %%
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(y, float))))


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(100 * np.mean(np.abs(p - y) / y))


assert mae([40_000, 50_000], [41_000, 48_000]) == 1500
assert abs(mape([40_000, 50_000], [41_000, 48_000]) - 3.25) < 1e-12
print("metric checks passed")

# %% [markdown]
# ## Step 2 · Data acquisition
#
# **Your turn (about 45 minutes).** Find the yearly ZIP links on the archive
# page, download at least five recent complete years, cache them, and record a
# SHA-256 per file. Decide what to do when a file you downloaded earlier has a
# different hash today.
#
# **Our solution.** ERCOT posts one ZIP per year on the archive page; the file
# names inside have changed over the years (**to verify**: recent years are
# reported as `Native_Load_YYYY.xlsx`). Rather than hard-code URLs that move,
# we read the archive page and pick the links whose file name contains the
# year. If that fails (the page layout changed, or the site blocks scripted
# access), download the yearly ZIPs in a browser and put them in `RAW_DIR`
# with the year in the file name; the rest of the notebook works the same.
#
# ERCOT revises recent months after settlement, so a changed hash is not
# necessarily an error. We keep a manifest of hashes and **warn** when a file
# changes, so you decide consciously whether to mix old and new downloads.

# %%
ARCHIVE_PAGE = "https://www.ercot.com/gridinfo/load/load_hist"
LAST_YEAR = 2025                          # the most recent COMPLETE year when you run this
YEARS = list(range(LAST_YEAR - 6, LAST_YEAR + 1))   # 7 years: 5 of history, 1 validation, 1 test
TEST_YEAR, VAL_YEAR = YEARS[-1], YEARS[-2]
MANIFEST = DATA_DIR / "manifest.json"
print("years:", YEARS, "| validation year", VAL_YEAR, "| test year", TEST_YEAR)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (course notebook)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def find_year_links(html: str, base: str = "https://www.ercot.com") -> dict[int, str]:
    links = {}
    for href in re.findall(r'href="([^"]+?\.zip)"', html, flags=re.I):
        name = href.rsplit("/", 1)[-1]
        m = re.search(r"(19|20)\d{2}", name)
        if m:
            links.setdefault(int(m.group(0)), href if href.startswith("http") else base + href)
    return links


def download_years(years: list[int]) -> dict[int, Path]:
    have = {int(m.group(0)): p for p in RAW_DIR.glob("*.zip") if (m := re.search(r"(19|20)\d{2}", p.name))}
    missing = [y for y in years if y not in have]
    if missing:
        links = find_year_links(fetch(ARCHIVE_PAGE).decode("utf-8", "replace"))
        for y in missing:
            if y not in links:
                raise RuntimeError(f"no ZIP for {y} found on {ARCHIVE_PAGE}; download it by hand into {RAW_DIR}")
            dest = RAW_DIR / f"ercot_load_{y}.zip"
            dest.write_bytes(fetch(links[y]))
            print("downloaded", y, links[y])
            have[y] = dest
    return {y: have[y] for y in years}


# %% [markdown]
# **The synthetic sample (automated checks only).** With
# `MLBOOTCAMP_CAPSTONE_SAMPLE=1` the function below writes one ZIP per year,
# each containing `Native_Load_YYYY.xlsx` with a first column `Hour Ending`
# (text such as `01/01/2023 01:00`, with `24:00` for the last hour of a day),
# the eight weather-zone columns and an `ERCOT` total — the layout the brief
# describes. Daylight saving is real: the spring day has 23 rows and the fall
# day 25, with the repeated hour marked by a ` DST` suffix (a guess at one
# possible marking; **to verify** how your files mark it — the parser below
# does not depend on it). The load itself is a made-up sum of daily, weekly
# and annual cycles, holidays, a persistent day-to-day "weather" disturbance
# and noise. Writing `.xlsx` needs `openpyxl` (preinstalled in Colab); if it
# is missing, the sample is written as CSV inside the ZIP instead, and the
# parser reads either. **The official files are Excel; the CSV fallback exists
# only for the synthetic sample.**

# %%
ZONES = ["COAST", "EAST", "FWEST", "NORTH", "NCENT", "SOUTH", "SCENT", "WEST"]
ZONE_SHARE = np.array([0.27, 0.035, 0.07, 0.02, 0.36, 0.075, 0.14, 0.03])


def synthetic_load(utc_end: pd.DatetimeIndex, rng) -> np.ndarray:
    local = (utc_end - pd.Timedelta(hours=1)).tz_convert(TZ)      # hour-beginning, local clock
    h = local.hour.to_numpy() + 0.5
    doy = local.dayofyear.to_numpy()
    years = local.year.to_numpy() - 2019 + doy / 365
    summer = np.exp(-(((doy - 205) / 45) ** 2))
    winter = np.exp(-(((np.minimum(doy, 365 - doy) + 10) / 30) ** 2))
    base = 41_000 * 1.03 ** years + 15_000 * summer + 6_000 * winter
    shape = (0.5 - 0.5 * np.cos(2 * np.pi * (h - 4.5) / 24)) * (0.32 + 0.12 * summer)  # peak ~16:30
    shape += winter * 0.08 * np.exp(-((h - 8) / 2) ** 2)                                # winter morning
    week = np.where(local.dayofweek.to_numpy() >= 5, 0.92, 1.0)
    dates = pd.DatetimeIndex(local.date)
    hol = dates.isin(USFederalHolidayCalendar().holidays(dates.min(), dates.max()))
    week = np.where(hol, 0.93, week)
    # a persistent day-to-day disturbance standing in for weather (AR(1) over days)
    days = pd.factorize(dates)[0]
    ar = np.zeros(days.max() + 1)
    for i in range(1, len(ar)):
        ar[i] = 0.75 * ar[i - 1] + rng.normal(0, 1)
    weather = ar[days] * (900 + 2_500 * summer * shape / 0.44)
    return (base * (0.62 + shape) * week + weather + rng.normal(0, 350, len(h))).round(2)


def hour_ending_labels(utc_end: pd.DatetimeIndex) -> list[str]:
    local = utc_end.tz_convert(TZ)
    labels, seen = [], set()
    for t in local:
        if t.hour == 0:                                       # midnight -> "24:00" of the previous day
            d = t.tz_localize(None).normalize() - pd.Timedelta(days=1)   # calendar arithmetic, not 24 h
            lab = f"{d:%m/%d/%Y} 24:00"
        else:
            lab = f"{t:%m/%d/%Y %H}:00"
        if lab in seen:                                       # the repeated fall-back hour
            lab += " DST"
        seen.add(lab)
        labels.append(lab)
    return labels


def write_synthetic_archive(years: list[int], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    try:
        import openpyxl  # noqa: F401
        fmt = "xlsx"
    except ImportError:
        fmt = "csv"
    for y in years:
        start = pd.Timestamp(f"{y}-01-01", tz=TZ).tz_convert("UTC")
        end = pd.Timestamp(f"{y + 1}-01-01", tz=TZ).tz_convert("UTC")
        utc_end = pd.date_range(start + pd.Timedelta(hours=1), end, freq="h")
        total = synthetic_load(utc_end, rng)
        zones = np.outer(total, ZONE_SHARE) * rng.normal(1, 0.01, (len(total), len(ZONES)))
        zones *= (total / zones.sum(1))[:, None]
        df = pd.DataFrame(zones.round(2), columns=ZONES)
        df.insert(0, "Hour Ending", hour_ending_labels(utc_end))
        df["ERCOT"] = df[ZONES].sum(1).round(2)
        buf = io.BytesIO()
        if fmt == "xlsx":
            df.to_excel(buf, index=False, sheet_name="Sheet1")
        else:
            buf.write(df.to_csv(index=False).encode())
        with zipfile.ZipFile(RAW_DIR / f"Native_Load_{y}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"Native_Load_{y}.{fmt}", buf.getvalue())
    return fmt


# %%
if SAMPLE:
    if not list(RAW_DIR.glob("*.zip")):
        print("synthetic sample written as", write_synthetic_archive(YEARS).upper(), "inside each ZIP")
zips = download_years(YEARS)

manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
for y, p in zips.items():
    h = sha256(p)
    old = manifest.get(str(y), {}).get("sha256")
    if old and old != h:
        print(f"WARNING {y}: file changed since {manifest[str(y)]['downloaded']} - a revision? check before mixing")
    if old != h:
        manifest[str(y)] = {"file": p.name, "bytes": p.stat().st_size, "sha256": h,
                            "downloaded": dt.date.today().isoformat()}
MANIFEST.write_text(json.dumps(manifest, indent=1))
print(pd.DataFrame(manifest).T[["file", "bytes", "sha256"]].assign(sha256=lambda d: d.sha256.str[:16] + "…").to_string())
print("source:", ARCHIVE_PAGE, "| terms: https://www.ercot.com/help/terms (as-is, no open licence found)")

# %% [markdown]
# **Why this way.** The manifest is the evidence for your data card: file,
# size, hash and download date per year. *Alternatives considered:* the
# `gridstatus` library (convenient, but a second source to document — fine for
# extensions); the ERCOT data-portal API (separate terms). *Common mistakes:*
# downloading the current, still-changing year; mixing files downloaded months
# apart without noticing a revision.

# %% [markdown]
# ## Step 3 · Data audit: parsing, hour-ending 24:00 and daylight saving
#
# **Your turn (about 90 minutes — this is where the time goes).** Parse every
# year into one table with a continuous UTC index. Handle `24:00`, the
# 23-hour spring day and the 25-hour fall day without dropping or summing
# anything. Assert row counts per day and per year. Then look at the series.
#
# **Our solution.** We do not try to interpret each hour label inside a day.
# We take the **local date** of each row and its **order within that date**:
# row $k$ of local date $d$ ends at (local midnight of $d$) $+ k$ hours, which
# we convert to UTC. This works whatever the repeated hour is called, and the
# number of rows per date must then equal the length of that day — 23, 24 or
# 25 hours — which we assert.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap2-dst.svg" alt="Spring, ordinary and fall days are mapped to a continuous UTC index, from which local date and hour and a row-count check are derived" width="100%">
#
# *What to notice:* local time is derived from UTC, never the other way
# round, except at this one parsing step, where the day length is checked.

# %%
def read_year_file(zip_path: Path) -> pd.DataFrame:
    """Read the single data file inside a yearly ZIP; normalise column names."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls", ".csv")))
        data = z.read(name)
    if name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    else:
        df = pd.read_excel(io.BytesIO(data))      # .xlsx needs openpyxl; old .xls files need xlrd
    df.columns = [str(c).strip().upper().replace("_", " ") for c in df.columns]
    hour_col = next(c for c in df.columns if "HOUR" in c)
    total_col = next(c for c in df.columns if c in ("ERCOT", "TOTAL", "ERCOT TOTAL"))
    out = df.rename(columns={hour_col: "hour_ending", total_col: "ERCOT"})
    out["source_file"] = name
    return out


def local_date_of(hour_ending: pd.Series) -> pd.Series:
    """Local calendar date that each hour-ending label belongs to (HE 24:00 -> that same date)."""
    if pd.api.types.is_datetime64_any_dtype(hour_ending):
        # Excel datetimes: 24:00 is stored as 00:00 of the next day; step back one hour.
        return (hour_ending - pd.Timedelta(hours=1)).dt.normalize()
    s = hour_ending.astype(str).str.strip()
    m = s.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})")
    if m.isna().any().any():
        raise ValueError(f"unparsed hour labels, e.g. {s[m.isna().any(axis=1)].head(3).tolist()}")
    date = pd.to_datetime(m[2] + "-" + m[0] + "-" + m[1], format="%Y-%m-%d")
    hour = m[3].astype(int)
    return date.where(hour > 0, date - pd.Timedelta(days=1))    # a "00:00" label closes the previous day


def day_length_hours(dates: pd.DatetimeIndex) -> np.ndarray:
    start = dates.tz_localize(TZ)
    end = (dates + pd.Timedelta(days=1)).tz_localize(TZ)
    return ((end - start) / pd.Timedelta(hours=1)).to_numpy().astype(int)


def to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Continuous UTC hour-ending index from (local date, order within date)."""
    out = df.copy()
    out["local_date"] = local_date_of(out["hour_ending"])
    assert out["local_date"].is_monotonic_increasing, "rows are not in time order"
    k = out.groupby("local_date").cumcount() + 1
    midnight_utc = pd.DatetimeIndex(out["local_date"]).tz_localize(TZ).tz_convert("UTC")
    out["utc_end"] = midnight_utc + pd.to_timedelta(k.to_numpy(), unit="h")
    counts = out.groupby("local_date").size()
    expected = pd.Series(day_length_hours(pd.DatetimeIndex(counts.index)), index=counts.index)
    bad = counts[counts != expected]
    if len(bad):
        raise ValueError(f"days with the wrong number of hours: {bad.head().to_dict()}")
    return out


# Check: hand-computed rows. HE 01:00 on 1 Jan 2023 (CST = UTC-6) ends at 07:00 UTC;
# HE 24:00 on 1 Jan ends at 06:00 UTC on 2 Jan.
toy = pd.DataFrame({"hour_ending": [f"01/01/2023 {h:02d}:00" for h in range(1, 25)], "ERCOT": 1.0})
t = to_utc_index(toy)["utc_end"]
assert t.iloc[0] == pd.Timestamp("2023-01-01 07:00", tz="UTC")
assert t.iloc[-1] == pd.Timestamp("2023-01-02 06:00", tz="UTC")
assert list(day_length_hours(pd.DatetimeIndex(["2023-03-12", "2023-07-01", "2023-11-05"]))) == [23, 24, 25]
print("hour-ending and DST day-length checks passed")

# %%
frames = []
for y, p in zips.items():
    f = to_utc_index(read_year_file(p))
    n_exp = int(day_length_hours(pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")).sum())
    assert len(f) == n_exp, f"{y}: {len(f)} rows, expected {n_exp}"
    frames.append(f)
raw = pd.concat(frames, ignore_index=True)
print(raw.groupby(raw.local_date.dt.year).agg(rows=("ERCOT", "size"), file=("source_file", "first"),
                                              mean_MW=("ERCOT", "mean"), max_MW=("ERCOT", "max")).round(0))

# continuous hourly UTC index: no gaps, no duplicates
assert raw.utc_end.is_unique and (raw.utc_end.diff().dropna() == pd.Timedelta(hours=1)).all()
print("UTC index is continuous:", raw.utc_end.min(), "→", raw.utc_end.max())

# %% [markdown]
# **Expected:** 8,760 rows per year (8,784 in leap years), a continuous UTC
# index with no gaps or duplicates. On the official files, mean load is
# typically in the 40,000–55,000 MW range and grows from year to year (**to
# verify** — measure it). If an assertion fails, print the offending day: it
# is usually a format change between years (another column name, Excel
# datetimes instead of text) and needs its own parsing branch, not a deletion.
#
# Now the schema and value checks.

# %%
zone_cols = [c for c in ZONES if c in raw.columns]
print("zones found:", zone_cols)
print("missing values per column:", raw[zone_cols + ["ERCOT"]].isna().sum().to_dict())
if zone_cols:
    gap = (raw[zone_cols].sum(1) - raw["ERCOT"]).abs()
    print(f"|sum of zones - ERCOT|: median {gap.median():.2f} MW, max {gap.max():.2f} MW")
print("non-positive or absurd values:", int(((raw.ERCOT <= 0) | (raw.ERCOT > 150_000)).sum()))

load = raw.set_index("utc_end")["ERCOT"].astype(float)
local = load.index.tz_convert(TZ)
cal = pd.DataFrame({"local_date": raw["local_date"].to_numpy(),
                    "hour": (load.index - pd.Timedelta(hours=1)).tz_convert(TZ).hour,   # hour-beginning, local
                    "he": (load.index - pd.Timedelta(hours=1)).tz_convert(TZ).hour + 1,
                    "y": load.to_numpy()}, index=load.index)
dst_days = cal.groupby("local_date").size().loc[lambda s: s != 24]
print("days that are not 24 hours long:\n", dst_days.tail(4).to_string())

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
daily = cal.groupby("local_date")["y"].mean()
axes[0].plot(daily.index, daily.to_numpy(), lw=0.6)
axes[0].set(title="daily mean load (annual cycle and trend)", ylabel="MW")
prof = cal[cal.local_date.dt.year < VAL_YEAR].assign(m=lambda d: d.local_date.dt.month)
for mth, col in [(1, "tab:blue"), (4, "tab:green"), (7, "tab:red"), (10, "tab:orange")]:
    g = prof[prof.m == mth].groupby("he")["y"].mean()
    axes[1].plot(g.index, g.to_numpy(), label=f"month {mth}", color=col)
axes[1].set(title="mean daily profile by month", xlabel="hour ending (local)")
axes[1].legend(fontsize=8)
wk = prof.assign(dow=prof.local_date.dt.dayofweek).groupby("dow")["y"].mean()
axes[2].bar(wk.index, wk.to_numpy())
axes[2].set(title="mean load by day of week (0 = Monday)", ylim=(wk.min() * 0.95, wk.max() * 1.02))
plt.show()

# %% [markdown]
# **What to notice:** three cycles — annual (summer peak, a smaller winter
# bump), weekly (weekends lower) and daily (an afternoon peak, plus a morning
# bump in winter). The seasonal-naive baselines exploit the weekly and daily
# cycles directly.
#
# **Validating the peak window.** The brief defines the peak window as HE
# 16–19 on weekdays from June to September, and asks you to check it on data
# that is *not* the test year. We look at the hour of each day's maximum.

# %%
hist = cal[cal.local_date.dt.year < VAL_YEAR]
peak_he = hist.loc[hist.groupby("local_date")["y"].idxmax()]
by_month = peak_he.groupby(peak_he.local_date.dt.month)["he"].agg(lambda s: s.mode().iloc[0])
summer_wd = peak_he[peak_he.local_date.dt.month.isin([6, 7, 8, 9]) & (peak_he.local_date.dt.dayofweek < 5)]
print("most common hour ending of the daily peak, by month:", by_month.to_dict())
print(f"summer weekdays with the daily peak inside HE 16-19: {summer_wd.he.between(16, 19).mean():.0%}")

# %% [markdown]
# **Expected:** in summer the daily maximum falls in HE 16–19 on the large
# majority of weekdays; in winter it can move to the morning or evening. If
# your data disagrees, change the window **now**, document why, and do not
# revisit it after seeing test results.
#
# **Extreme periods.** Winter Storm Uri (February 2021) involved load
# shedding: the recorded load is not the demand that would have occurred. We
# keep those days in training (deleting history is itself a decision that
# must be justified) and look at them separately in the error analysis.

# %%
uri = cal[(cal.local_date >= "2021-02-08") & (cal.local_date <= "2021-02-24")]
if len(uri):
    print(uri.groupby("local_date")["y"].agg(["min", "max"]).round(0).T.to_string())

# %% [markdown]
# ## Step 4 · A leakage-safe protocol: rolling origin
#
# **Your turn (about 60 minutes).** Design the backtest: what is trained when,
# what is forecast, which year selects models and which is touched once. Write
# a test that would fail if any feature used data after the origin.
#
# **Our solution.**
#
# * **Final test year** = the last complete year, run once at the end.
# * **Validation year** = the year before, used for every decision (features,
#   settings, model choice).
# * **Folds** = calendar months. For month $m$ we train on all target days
#   before $m$ (expanding window) and forecast every day of $m$, each day from
#   its own origin (midnight before it). Retraining monthly instead of daily
#   keeps it cheap; the model is a month stale at most, which we state.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap2-backtest.svg" alt="Rolling-origin backtest: training bars grow; each fold forecasts one month; validation year then test year" width="100%">
#
# *What to notice:* the training window always ends where the evaluated month
# starts; no fold ever trains on its own future.
#
# **Features.** We build a table `P[date, hour]` of load by local date and
# local hour (the repeated fall-back hour is averaged; the missing spring hour
# is interpolated — in this table only, never in the labels). Every feature
# for target date $T$ is read from rows $T-1$ or earlier of that table, plus
# the calendar of $T$ (known in advance).

# %%
HOLIDAYS = USFederalHolidayCalendar().holidays(f"{YEARS[0] - 1}-01-01", f"{LAST_YEAR + 2}-12-31")


def profile_table(cal_df: pd.DataFrame) -> pd.DataFrame:
    """P[local date, local hour 0-23]: one row per calendar day, continuous dates."""
    p = cal_df.groupby(["local_date", "hour"])["y"].mean().unstack("hour")
    p = p.reindex(columns=range(24))
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="D"))
    return p.interpolate(axis=1, limit_direction="both")


def target_hours(dates) -> pd.DataFrame:
    """Every hour of the given local dates, with its UTC hour-ending and local hour."""
    rows = []
    for d in pd.DatetimeIndex(dates):
        n = int(day_length_hours(pd.DatetimeIndex([d]))[0])
        start = pd.Timestamp(d).tz_localize(TZ).tz_convert("UTC")
        utc_end = start + pd.to_timedelta(np.arange(1, n + 1), unit="h")
        rows.append(pd.DataFrame({"utc_end": utc_end, "local_date": d,
                                  "hour": (utc_end - pd.Timedelta(hours=1)).tz_convert(TZ).hour}))
    return pd.concat(rows, ignore_index=True)


def build_features(P: pd.DataFrame, hours: pd.DataFrame) -> pd.DataFrame:
    """Day-ahead features for each target hour. Uses P only at dates <= target - 1 day."""
    T = pd.DatetimeIndex(hours["local_date"])
    h = hours["hour"].to_numpy()

    def at(days_back: int) -> np.ndarray:
        src = T - pd.Timedelta(days=days_back)
        return P.reindex(src).to_numpy()[np.arange(len(h)), h] if len(h) else np.array([])

    day_d = P.reindex(T - pd.Timedelta(days=1))
    week = np.stack([P.reindex(T - pd.Timedelta(days=k)).mean(axis=1).to_numpy() for k in range(1, 8)])
    f = pd.DataFrame({
        "lag_1d": at(1), "lag_2d": at(2), "lag_7d": at(7), "lag_14d": at(14),
        "same_wd_mean_4w": np.mean([at(7 * k) for k in range(1, 5)], axis=0),
        "day_d_mean": day_d.mean(axis=1).to_numpy(), "day_d_max": day_d.max(axis=1).to_numpy(),
        "day_d_min": day_d.min(axis=1).to_numpy(), "last_7d_mean": week.mean(axis=0),
        "hour": h, "dow": T.dayofweek, "month": T.month,
        "doy_sin": np.sin(2 * np.pi * T.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * T.dayofyear / 365.25),
        "holiday": T.isin(HOLIDAYS).astype(int), "day_after_holiday": (T - pd.Timedelta(days=1)).isin(HOLIDAYS).astype(int),
    }, index=hours.index)
    return f


P_all = profile_table(cal)
all_dates = pd.DatetimeIndex(sorted(cal.local_date.unique()))
H = target_hours(all_dates[28:])                 # first 28 days have no 4-week history
H = H.merge(cal.reset_index()[["utc_end", "y"]], on="utc_end", how="left")
X = build_features(P_all, H)
assert H["y"].notna().all() and len(H) == len(X)
print("target hours with features:", len(H), "| features:", list(X.columns))

# %% [markdown]
# **Check — the leakage test.** Pick an origin. Replace every value *after* it
# with nonsense, rebuild the features for the next day, and require them to be
# identical. A feature that peeks at day D+1 (a centred rolling mean, a
# `shift` in hours on a 25-hour day, a forgotten `shift(1)`) fails this test.
# We run it on an ordinary day and on both daylight-saving transitions.

# %%
def assert_no_future(target: pd.Timestamp) -> None:
    origin_date = target - pd.Timedelta(days=1)
    corrupted = cal.copy()
    corrupted.loc[corrupted.local_date > origin_date, "y"] = 1e12
    hours = target_hours([target])
    a = build_features(P_all, hours)
    b = build_features(profile_table(corrupted), hours)
    pd.testing.assert_frame_equal(a, b)


spring = [d for d in dst_days.index if dst_days[d] == 23 and d.year == VAL_YEAR][0]
fall = [d for d in dst_days.index if dst_days[d] == 25 and d.year == VAL_YEAR][0]
for target in [pd.Timestamp(f"{VAL_YEAR}-07-15"), spring, fall]:
    assert_no_future(pd.Timestamp(target))
print("no-future check passed for", pd.Timestamp(f"{VAL_YEAR}-07-15").date(), spring.date(), fall.date(),
      "| hours on those days:", [len(target_hours([d])) for d in (f"{VAL_YEAR}-07-15", spring, fall)])

# %% [markdown]
# **Why this way.** Lags are taken by *local calendar day*, so a 23- or 25-hour
# day can never move a lag across the origin. *Alternatives considered:*
# `series.shift(24)` on the UTC series — simpler, but on the 25-hour day it
# lets the last target hour see a value after the origin, and on the 23-hour
# day "24 hours ago" is a different clock hour; and 24 separate models, one
# per hour of the day (the brief asks you to compare; we use one model with
# `hour` as a feature, which shares data across hours). *Common mistakes:*
# random K-fold over hours; scalers or target encodings fitted on all years;
# deciding holiday handling after looking at the test year.

# %% [markdown]
# ## Step 5 · Baselines through the backtest
#
# **Your turn (about 45 minutes).** Write the backtest harness once, as a
# function that takes a "forecaster" and a list of months, and run the three
# seasonal-naive baselines through it.
#
# **Our solution.** A forecaster is `fit_predict(train_rows, test_rows) ->
# predictions`. Baselines ignore the training rows.

# %%
PEAK_MONTHS, PEAK_HE = [6, 7, 8, 9], range(16, 20)
H["he"] = H["hour"] + 1
H["is_peak"] = (H.local_date.dt.month.isin(PEAK_MONTHS) & (H.local_date.dt.dayofweek < 5)
                & H.he.isin(PEAK_HE))


def backtest(fit_predict, year: int, name: str) -> pd.DataFrame:
    preds = []
    for month in range(1, 13):
        start = pd.Timestamp(f"{year}-{month:02d}-01")
        end = start + pd.offsets.MonthBegin(1)
        tr = H.local_date < start
        te = (H.local_date >= start) & (H.local_date < end)
        preds.append(pd.DataFrame({"pred": fit_predict(tr.to_numpy(), te.to_numpy()), "fold": month},
                                  index=H.index[te]))
    out = pd.concat(preds)
    out["model"] = name
    return out


def naive(col):
    return lambda tr, te: X.loc[te, col].to_numpy()


BASELINES = {"seasonal naive, daily (lag 1 day)": naive("lag_1d"),
             "seasonal naive, weekly (lag 7 days)": naive("lag_7d"),
             "mean of last 4 same weekdays": naive("same_wd_mean_4w")}


def metrics(res: pd.DataFrame) -> dict:
    d = H.loc[res.index].assign(pred=res["pred"])
    daily_peak = d.groupby("local_date").agg(f=("pred", "max"), a=("y", "max"))
    pk = d[d.is_peak]
    return {"MAE": mae(d.y, d.pred), "MAPE %": mape(d.y, d.pred),
            "peak MAE": mae(pk.y, pk.pred) if len(pk) else np.nan,
            "daily peak bias": float((daily_peak.f - daily_peak.a).mean()),
            "bias": float((d.pred - d.y).mean()),
            "fold MAE sd": float(d.assign(e=(d.pred - d.y).abs()).groupby(res["fold"])["e"].mean().std())}


val_results = {name: backtest(fp, VAL_YEAR, name) for name, fp in BASELINES.items()}
val_table = pd.DataFrame({n: metrics(r) for n, r in val_results.items()}).T
print(f"validation year {VAL_YEAR}, 12 monthly folds:\n", val_table.round(1).to_string())
BEST_BASELINE = val_table["MAE"].idxmin()
print("best baseline on validation:", BEST_BASELINE)

# %% [markdown]
# **Expected:** on real load, the weekly naive is usually better than the
# daily naive on Mondays and Saturdays (it gets the weekday right) and worse
# during fast weather changes; the 4-week mean is smoother. Whichever wins on
# the *validation* year is "the baseline" every improvement is measured
# against.

# %% [markdown]
# ## Step 6 · The improved model: gradient-boosted trees
#
# **Your turn (about 90 minutes).** Train a gradient-boosted tree model on the
# lag and calendar features through the same backtest. Decide one design
# question on the validation year only.
#
# **Our solution.** `HistGradientBoostingRegressor` with absolute-error loss
# (the metric is MAE). The design question we decide on validation: do the
# "day D" summary features (`day_d_mean`, `day_d_max`, `day_d_min`,
# `last_7d_mean`) help beyond the pure lags? They carry the level of the most
# recent day, which tracks weather persistence.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

LAGS_CAL = ["lag_1d", "lag_2d", "lag_7d", "lag_14d", "same_wd_mean_4w", "hour", "dow", "month",
            "doy_sin", "doy_cos", "holiday", "day_after_holiday"]
WITH_DAY_D = LAGS_CAL + ["day_d_mean", "day_d_max", "day_d_min", "last_7d_mean"]
CATEGORICAL = ["hour", "dow", "month"]


def gbm_forecaster(cols):
    def fit_predict(tr, te):
        model = HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.08, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=40, categorical_features=[c in CATEGORICAL for c in cols], random_state=SEED)
        model.fit(X.loc[tr, cols], H.loc[tr, "y"])
        return model.predict(X.loc[te, cols])
    return fit_predict


CANDIDATES = {"GBM, lags + calendar": LAGS_CAL, "GBM, + day-D statistics": WITH_DAY_D}
t0 = time.time()
for name, cols in CANDIDATES.items():
    val_results[name] = backtest(gbm_forecaster(cols), VAL_YEAR, name)
GBM_VAL_SECONDS = time.time() - t0
val_table = pd.DataFrame({n: metrics(r) for n, r in val_results.items()}).T
print(f"validation year {VAL_YEAR} ({GBM_VAL_SECONDS:.0f}s for 24 GBM fits):\n", val_table.round(1).to_string())
CHOSEN = val_table.loc[list(CANDIDATES), "MAE"].idxmin()
print("chosen on validation MAE:", CHOSEN)

# %% [markdown]
# **How to read it.** The tree model should beat the best seasonal-naive
# baseline clearly on overall MAE; the question is by how much in the peak
# window, where errors are largest in MW. Our choice between the two feature
# sets is made here and frozen. *Alternatives considered:* ridge on Fourier
# terms (transparent, try it as an exercise); LightGBM (faster, same idea);
# per-hour models. *Common mistake:* trying ten feature sets on the validation
# year and reporting the best one as if it were a test result — that is why
# the test year exists.

# %% [markdown]
# ## Step 7 · Final evaluation on the test year (run once)
#
# **Your turn (about 40 minutes).** Run the best baseline and the chosen model
# through the same monthly backtest on the test year. Report MAE, MAPE,
# peak-window MAE and daily-peak bias, with a block-bootstrap interval over
# days for the MAE and for the improvement over the baseline.
#
# **Our solution.** A moving-block bootstrap with 7-day blocks: neighbouring
# days share weather, so resampling single hours (or even single days) would
# understate the uncertainty.

# %%
t0 = time.time()
test_results = {BEST_BASELINE: backtest(BASELINES[BEST_BASELINE], TEST_YEAR, BEST_BASELINE),
                CHOSEN: backtest(gbm_forecaster(CANDIDATES[CHOSEN]), TEST_YEAR, CHOSEN)}
TEST_SECONDS = time.time() - t0
test_table = pd.DataFrame({n: metrics(r) for n, r in test_results.items()}).T


def block_bootstrap_days(day_errors: pd.DataFrame, n_boot=2000, block=7, seed=SEED) -> np.ndarray:
    """day_errors: one row per day, columns = models' daily mean absolute error. Returns resampled means."""
    g = np.random.default_rng(seed)
    n = len(day_errors)
    starts = np.arange(n - block + 1)
    k = math.ceil(n / block)
    vals = day_errors.to_numpy()
    out = np.empty((n_boot, vals.shape[1]))
    for b in range(n_boot):
        idx = (g.choice(starts, size=k)[:, None] + np.arange(block)).ravel()[:n]
        out[b] = vals[idx].mean(axis=0)
    return out


def daily_abs_error(res):
    d = H.loc[res.index].assign(e=(res["pred"] - H.loc[res.index, "y"]).abs())
    return d.groupby("local_date")["e"].mean()


de = pd.DataFrame({n: daily_abs_error(r) for n, r in test_results.items()})
boot = block_bootstrap_days(de)
for j, n in enumerate(de.columns):
    test_table.loc[n, "MAE lo"], test_table.loc[n, "MAE hi"] = np.percentile(boot[:, j], [2.5, 97.5])
# Day-weighted MAE (each day counts once) equals the hour-weighted MAE up to the few 23/25-hour days.
diff = boot[:, 1] - boot[:, 0]
DIFF = (float(de.iloc[:, 1].mean() - de.iloc[:, 0].mean()), *np.percentile(diff, [2.5, 97.5]))
print(f"test year {TEST_YEAR} ({TEST_SECONDS:.0f}s):\n", test_table.round(1).to_string())
print(f"\nMAE({CHOSEN}) - MAE({BEST_BASELINE}) = {DIFF[0]:+.0f} MW, "
      f"95% block-bootstrap CI [{DIFF[1]:+.0f}, {DIFF[2]:+.0f}] MW "
      f"({100 * DIFF[0] / de.iloc[:, 0].mean():+.1f}% of the baseline MAE)")

# %% [markdown]
# **Why this way.** The same harness, the same folds and the same origin rule
# for both models; one run; an interval that respects the correlation between
# neighbouring days. Report the improvement as a measurement — "reduced MAE
# by X MW (Y%) against a seasonal-naive baseline on the 12 monthly folds of
# the test year" — never as money saved.

# %% [markdown]
# ## Step 8 · Error analysis: hours, months, peaks and extreme days
#
# **Your turn (about 45 minutes).** Where are the errors? By hour of day, by
# month, in the peak window, and on the worst days. Is the peak under- or
# over-forecast? Form one hypothesis per bad slice.
#
# **Our solution.**

# %%
d = H.loc[test_results[CHOSEN].index].assign(pred=test_results[CHOSEN]["pred"])
d["base"] = test_results[BEST_BASELINE]["pred"]
d["err"] = d.pred - d.y
by_hour = d.groupby("he").agg(MAE=("err", lambda e: e.abs().mean()), bias=("err", "mean"))
by_month = d.groupby(d.local_date.dt.month).agg(
    MAE=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
    baseline_MAE=("base", lambda b: (b - d.loc[b.index, "y"]).abs().mean()))
print("by month:\n", by_month.round(0).T.to_string())
worst = d.groupby("local_date").agg(MAE=("err", lambda e: e.abs().mean()), bias=("err", "mean"),
                                    peak_actual=("y", "max")).nlargest(5, "MAE")
print("\nfive worst days:\n", worst.round(0).to_string())
dst_test = [x for x in dst_days.index if x.year == TEST_YEAR]
print("\nDST days in the test year:",
      {str(x.date()): (int((d.local_date == x).sum()), round(float(d[d.local_date == x].err.abs().mean()), 0))
       for x in dst_test}, "(hours, MAE)")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
axes[0].bar(by_hour.index, by_hour["MAE"], color="grey", label="MAE")
axes[0].plot(by_hour.index, by_hour["bias"], "r.-", label="bias (forecast - actual)")
axes[0].axvspan(15.5, 19.5, color="orange", alpha=0.15, label="peak window hours")
axes[0].set(xlabel="hour ending (local)", ylabel="MW", title=f"{CHOSEN}: error by hour, {TEST_YEAR}")
axes[0].legend(fontsize=8)
wd = d.groupby("local_date")[["y", "pred"]].max()
axes[1].scatter(wd.y, wd.pred - wd.y, s=8)
axes[1].axhline(0, color="k", lw=1)
axes[1].set(xlabel="actual daily peak (MW)", ylabel="forecast peak - actual peak", title="daily peak error")
plt.show()

# %% [markdown]
# **How to read it (check each against your output).**
#
# * **By hour:** errors usually grow through the day and are largest in the
#   afternoon, when load is highest and most weather-sensitive. Day-ahead
#   forecasts also get worse with horizon: hour 1 of D+1 is one hour after the
#   last observation, hour 18 is eighteen hours after it.
# * **Daily peak:** if the points on the right of the scatter (the highest
#   peaks) lie below zero, the model under-forecasts the hottest days — the
#   expensive error. Hypothesis: without temperature, the model can only infer
#   a heat wave from the days before it, so the *first* hot day of a spell is
#   missed. Test it by adding weather (an extension; mind the
#   observed-versus-forecast leakage trap).
# * **Worst days** are typically holidays (few examples per year) and sudden
#   weather changes. Look them up; write them in the README.
# * **DST days** should not stand out because of their length. If one has a
#   large error, print its hours next to the day before: a systematic one-hour
#   shift between forecast and actual points to a parsing or alignment bug; a
#   miss of similar size in every hour is ordinary model error (a weekend, a
#   weather change).

# %% [markdown]
# ## Step 9 · A small API
#
# **Your turn (about 60 minutes).** A FastAPI app factory with `/health`,
# `/ready` and `POST /forecast`. The client sends recent history; the service
# answers with 23, 24 or 25 hourly forecasts for the next local day. Validate:
# time zone `America/Chicago`; history hourly, strictly increasing, no
# duplicates; at least the 28 local days the features need; the target date is
# the day after the last history hour; loads finite and within a plausible
# range. Test all three day lengths.
#
# **Our solution.** The service rebuilds `P` from the posted history and calls
# **the same** `build_features` function used in the backtest.

# %%
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_HISTORY = 24 * 60


class HourIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour_ending_utc: dt.datetime
    load_mw: float = Field(gt=0, lt=150_000)

    @field_validator("hour_ending_utc")
    @classmethod
    def must_be_utc(cls, v):
        if v.tzinfo is None or v.utcoffset() != dt.timedelta(0):
            raise ValueError("hour_ending_utc must be a UTC timestamp (…Z)")
        return v


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_date: dt.date
    timezone: str
    history: list[HourIn] = Field(min_length=24 * 29, max_length=MAX_HISTORY)

    @field_validator("timezone")
    @classmethod
    def chicago_only(cls, v):
        if v != TZ:
            raise ValueError(f"only {TZ} is supported")
        return v


def create_app(bundle: dict | None) -> FastAPI:
    app = FastAPI(title="ERCOT day-ahead load (walkthrough)")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if bundle is None:
            return JSONResponse({"status": "not_ready", "detail": "no model loaded"}, status_code=503)
        return {"status": "ready", "model_version": bundle["model_version"],
                "trained_through": bundle["trained_through"]}

    @app.post("/forecast")
    def forecast(req: ForecastRequest):
        if bundle is None:
            raise HTTPException(503, "model not loaded")
        ts = pd.DatetimeIndex([h.hour_ending_utc for h in req.history]).tz_convert("UTC")
        if not ts.is_unique or not ts.is_monotonic_increasing:
            raise HTTPException(422, "history must be strictly increasing with no duplicate hours")
        if (np.diff(ts.asi8) != 3_600 * 10**9).any():
            raise HTTPException(422, "history must be hourly with no gaps")
        last_local_date = (ts[-1] - pd.Timedelta(hours=1)).tz_convert(TZ).normalize().tz_localize(None)
        target = pd.Timestamp(req.target_date)
        if target != last_local_date + pd.Timedelta(days=1):
            raise HTTPException(422, "target_date must be the day after the last history hour (local time)")
        if ts[-1] != target.tz_localize(TZ).tz_convert("UTC"):
            raise HTTPException(422, "history must end at the forecast origin (local midnight before target_date)")
        hist = pd.DataFrame({"y": [h.load_mw for h in req.history]}, index=ts)
        hb = (ts - pd.Timedelta(hours=1)).tz_convert(TZ)
        hist["local_date"], hist["hour"] = hb.normalize().tz_localize(None), hb.hour
        if hist.local_date.nunique() < 29:
            raise HTTPException(422, "history must cover the 28 local days before the target day")
        hours = target_hours([target])
        feats = build_features(profile_table(hist), hours)[bundle["columns"]]
        pred = bundle["model"].predict(feats)
        local_end = hours.utc_end.dt.tz_convert(TZ)
        return {"model_version": bundle["model_version"], "target_date": str(req.target_date),
                "forecasts": [{"hour_ending_local": t.isoformat(), "load_mw": round(float(p), 1),
                               "is_peak_window": bool(t.month in PEAK_MONTHS and target.dayofweek < 5
                                                      and (t - pd.Timedelta(hours=1)).hour + 1 in PEAK_HE)}
                              for t, p in zip(local_end, pred)]}

    return app


# Train the served model on everything before the test year (the model the backtest's
# January fold would use), so the API can be checked against the offline forecasts.
served_cols = CANDIDATES[CHOSEN]
tr = (H.local_date < pd.Timestamp(f"{TEST_YEAR}-01-01")).to_numpy()
served = HistGradientBoostingRegressor(
    loss="absolute_error", learning_rate=0.08, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
    categorical_features=[c in CATEGORICAL for c in served_cols], random_state=SEED)
served.fit(X.loc[tr, served_cols], H.loc[tr, "y"])
bundle = {"model": served, "columns": served_cols, "model_version": "ercot-dayahead-gbm-v1",
          "trained_through": f"{TEST_YEAR - 1}-12-31"}
client = TestClient(create_app(bundle))


def payload(target: str, days: int = 35) -> dict:
    target = pd.Timestamp(target)
    end = target.tz_localize(TZ).tz_convert("UTC")
    start = (target - pd.Timedelta(days=days)).tz_localize(TZ).tz_convert("UTC")
    h = load[(load.index > start) & (load.index <= end)]
    return {"target_date": str(target.date()), "timezone": TZ,
            "history": [{"hour_ending_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "load_mw": float(v)}
                        for t, v in h.items()]}


assert client.get("/health").json() == {"status": "ok"}
assert client.get("/ready").json()["model_version"] == "ercot-dayahead-gbm-v1"
spring_t = [x for x in dst_test if dst_days[x] == 23][0]
fall_t = [x for x in dst_test if dst_days[x] == 25][0]
jan = f"{TEST_YEAR}-01-17"
for target, n in [(jan, 24), (str(spring_t.date()), 23), (str(fall_t.date()), 25)]:
    r = client.post("/forecast", json=payload(target))
    assert r.status_code == 200, r.text
    assert len(r.json()["forecasts"]) == n, (target, len(r.json()["forecasts"]))
r = client.post("/forecast", json=payload(jan)).json()
offline = test_results[CHOSEN]["pred"][H.loc[test_results[CHOSEN].index, "local_date"] == pd.Timestamp(jan)]
assert np.allclose([f["load_mw"] for f in r["forecasts"]], offline.to_numpy(), atol=0.11)
print("first two forecasts for", jan, r["forecasts"][:2])

bad = payload(jan)
bad["timezone"] = "UTC"
assert client.post("/forecast", json=bad).status_code == 422                     # wrong time zone
bad = payload(jan)
del bad["history"][100]
assert client.post("/forecast", json=bad).status_code == 422                     # a gap
bad = payload(jan)
bad["history"][101] = bad["history"][100]
assert client.post("/forecast", json=bad).status_code == 422                     # duplicate hour
bad = payload(jan)
bad["target_date"] = f"{TEST_YEAR}-01-19"
assert client.post("/forecast", json=bad).status_code == 422                     # not the next day
bad = payload(jan)
bad["history"][5]["load_mw"] = -1.0
assert client.post("/forecast", json=bad).status_code == 422                     # implausible value
assert client.post("/forecast", json=payload(jan, days=10)).status_code == 422   # too little history
cold = TestClient(create_app(None))
assert cold.get("/health").status_code == 200 and cold.get("/ready").status_code == 503
print("API checks passed: 24/23/25-hour days, matches offline forecasts, 6 validation rules, not-ready state")

# %% [markdown]
# **Why this way.** The API test "the service's forecast for 17 January equals
# the backtest's forecast for 17 January" is the cheapest protection against
# training/serving skew. Timestamps travel in UTC; local time appears only in
# the response, with its offset, so a client never has to guess which
# 01:00 is meant on the fall-back day. *Common mistakes:* accepting naive
# timestamps; always returning 24 values; computing features in the API with
# a re-implementation instead of the training function. In your repository
# this moves to `src/<package>/api.py` with these checks as `pytest` tests
# (Chapters 10 and 12 show the pattern).

# %% [markdown]
# ## Step 10 · Write-up: results table, limitations, next steps
#
# **Your turn (about 60 minutes).** Produce the README results table from the
# computed numbers and write the limitations.
#
# **Our solution.** The cell prints a markdown table to paste into your README.
# On the synthetic sample it prints a warning first: those numbers are not
# results.

# %%
def results_markdown() -> str:
    lines = ["| Model | Evaluation | MAE, MW (95% CI) | MAPE | Peak-window MAE, MW | Daily-peak bias, MW |",
             "|---|---|---|---:|---:|---:|"]
    for n, r in test_table.iterrows():
        lines.append(f"| {n} | test year {TEST_YEAR}, 12 monthly folds | {r['MAE']:,.0f} "
                     f"({r['MAE lo']:,.0f}–{r['MAE hi']:,.0f}) | {r['MAPE %']:.2f}% | {r['peak MAE']:,.0f} | "
                     f"{r['daily peak bias']:+,.0f} |")
    for n, r in val_table.iterrows():
        lines.append(f"| {n} | validation year {VAL_YEAR}, 12 monthly folds (fold sd {r['fold MAE sd']:,.0f}) | "
                     f"{r['MAE']:,.0f} | {r['MAPE %']:.2f}% | {r['peak MAE']:,.0f} | {r['daily peak bias']:+,.0f} |")
    lines += ["", f"Improvement on the test year: {DIFF[0]:+,.0f} MW MAE versus {BEST_BASELINE} "
              f"(95% block-bootstrap CI {DIFF[1]:+,.0f} to {DIFF[2]:+,.0f} MW, 7-day blocks). "
              f"Peak window: HE 16–19 local, weekdays, June–September. Data: ERCOT system total, "
              f"{YEARS[0]}–{TEST_YEAR}, monthly retraining, origin at local midnight."]
    return "\n".join(lines)


if SAMPLE:
    print("SYNTHETIC SAMPLE RUN - these numbers are NOT results. Do not copy them into a README.\n")
print(results_markdown())
print(f"\nwhole notebook: {time.time() - T_START:.0f}s")

# %% [markdown]
# **Limitations (write these in your own words, with your numbers).**
#
# * **No weather in the MVP**, so unusually hot or cold days carry large
#   errors; show them, do not hide them.
# * **Archive versus real time.** The public archive is posted monthly and
#   may be revised; a real operator has telemetry and weather forecasts.
# * **Daylight saving and hour-ending conventions** are easy to get subtly
#   wrong; the row-count assertions and the no-future test are the only
#   evidence that ours are right. The repeated fall-back hour receives the same
#   features twice, so its two forecasts are identical.
# * **Structural change.** Texas load grows (new large loads, population); an
#   expanding window may lag a trend shift. Compare training windows as an
#   extension.
# * **Extreme events** (Winter Storm Uri) are rare and contain load shedding;
#   a yearly average hides them.
# * **System total only**; zones and prices are different problems.
#
# **What to do next (extensions).**
#
# 1. Weather: population-weighted temperature for a few large Texas cities —
#    archived *forecasts* for an honest result, or observed values labelled as
#    an upper bound.
# 2. Probabilistic forecasts: quantile gradient boosting for P10/P50/P90,
#    pinball loss and interval coverage per month.
# 3. Zone-level forecasts reconciled to the system total.
# 4. Performance monitoring: labels arrive within the hour, so track a rolling
#    7-day MAE and peak-hour bias against alert thresholds, and register one
#    model per monthly retrain with promote/rollback — the patterns of
#    Chapter 15.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `no ZIP for 2025 found` | the archive page changed or blocks scripts | download the yearly ZIPs in a browser into `RAW_DIR` (year in the file name) |
# | `days with the wrong number of hours` | a format change between years, or a dropped DST row | print that day's rows; add a parsing branch for that year |
# | `unparsed hour labels` | a year stores labels differently (e.g. `1/1/2019 1:00`, or extra text) | widen the regular expression; test it on a hand-checked row |
# | `ImportError: openpyxl` / `xlrd` | Excel reader missing outside Colab | `pip install openpyxl` (and `xlrd` for old `.xls`) |
# | Model "beats" the baseline by an implausible margin | a feature uses day D+1 | run `assert_no_future` on the new feature |
#
# **Interview practice:** answer the ten questions at the end of the
# [brief](README.md) aloud, in two minutes each, using the numbers your run
# printed above.
