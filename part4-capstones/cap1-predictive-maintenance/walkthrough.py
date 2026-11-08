# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap1-predictive-maintenance/walkthrough.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up. No GPU is needed.*

# %% [markdown]
# # Capstone 1 · Predictive maintenance (NASA C-MAPSS) — guided walkthrough
#
# **About 10 minutes to run on a laptop CPU · 4 days of work if you attempt every step yourself first**
#
# This notebook is a complete, worked solution of the capstone's minimum
# viable product (MVP), written as if a tutor were sitting next to you. It
# follows the milestones of the [capstone brief](README.md): framing, data,
# audit, a leakage-safe split, baselines, an improved model, evaluation with
# uncertainty, error analysis, a small API and the write-up.
#
# ## How to use this walkthrough
#
# Attempt each step yourself before you read our solution. Every step has the
# same three parts:
#
# 1. **Your turn.** The goal of the step, the decision you have to make, and
#    roughly how long it should take. Close the solution (or cover the screen)
#    and write your own code in a scratch cell.
# 2. **Our solution.** Code, followed by *why* we did it this way, the
#    alternatives we considered, and the mistakes we see most often.
# 3. **Check.** Assertions that must hold whatever your design. If your
#    version fails one of them, the problem is real, not cosmetic.
#
# Your numbers will differ from ours if you made different choices. That is
# expected; what matters is that every choice is stated and every number
# comes from a run on a split you can name.
#
# > **Data and honesty notice.** C-MAPSS is **simulated** turbofan degradation
# > data published by the NASA Prognostics Center of Excellence (PCoE). This
# > walkthrough is a public-data exercise. It is not work for or endorsed by
# > NASA, an airline or an engine maker, and it says nothing about any real
# > fleet. Cite: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation
# > Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008,
# > DOI 10.1109/PHM.2008.4711414. Terms: NASA data is generally a US
# > Government work in the public domain in the US — **to verify** on the
# > [landing page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
# > before you use it.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

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
import hashlib
import io
import math
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

T_START = time.time()
SEED = 0
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

# MLBOOTCAMP_CAPSTONE_SAMPLE=1 runs every step on a small SYNTHETIC dataset with
# exactly the official file layout. It exists so the course's automated checks
# can execute this notebook without network access. Its numbers mean nothing.
SAMPLE = os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1"
DATA_DIR = ART / "cap1-cmapss" / ("sample" if SAMPLE else "official")
DATA_DIR.mkdir(parents=True, exist_ok=True)
print("mode:", "SYNTHETIC SAMPLE (numbers are not results)" if SAMPLE else "official NASA data")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Engine (unit)** | one simulated turbofan; all its rows form one trajectory | unit 17 of `train_FD001.txt` |
# | **Cycle** | one flight's worth of operation; one row per cycle | cycle 101 |
# | **Run to failure** | a training trajectory that ends at the failure cycle | every training engine |
# | **RUL** | remaining useful life: cycles left before failure | 112 cycles |
# | **RUL cap ($R_{\max}$)** | an upper limit on the training label, because early wear is invisible | 125 cycles |
# | **Window** | the last $W$ cycles of an engine, the model's input | 30 cycles × 16 channels |
# | **Engine-level (group) split** | every window of one engine lies on the same side of the split | `GroupKFold` over engine IDs |
# | **Early / late prediction** | predicted RUL below / above the truth; late is the dangerous one | $d = \hat{y} - y > 0$ is late |
# | **NASA (PHM08) score** | an asymmetric, exponential penalty that punishes late predictions harder | lower is better |
# | **Bootstrap CI** | resample *engines* with replacement to see how much a metric could move | 95% interval for RMSE |

# %% [markdown]
# ## Step 1 · Framing the problem
#
# **Your turn (about 20 minutes).** Before touching data, write down in one
# sentence each: the question, who acts on the answer, the unit of
# prediction, the input, the label, the horizon and the primary metric. Decide
# whether the error should be symmetric.
#
# **Our solution.**
#
# * **Question.** For each engine in service, how many more cycles can it run
#   before failure, and is that estimate safe to plan maintenance on?
# * **Who acts.** A maintenance planner who books shop visits and spare parts.
# * **Unit of prediction.** One engine, at its latest observed cycle. Not a
#   row, not a window: the RUL file gives one truth per test engine, at its
#   last cycle, so that is the only place we can be scored.
# * **Input.** The last $W = 30$ cycles of that engine (settings and the
#   sensors that actually vary). Thirty is long enough to estimate a trend and
#   short enough that every FD001 test engine has that many cycles (we check
#   this in Step 3).
# * **Label.** $\text{RUL} = \text{last cycle} - \text{cycle}$ in training,
#   capped at $R_{\max}$ (Step 4 explains the cap); in test, the uncapped
#   number in `RUL_FD001.txt`.
# * **Horizon.** There is no fixed horizon: the model answers "how long from
#   now?" for whatever cycle the engine has reached.
# * **Metrics.** RMSE in cycles against the **uncapped** truth (what a planner
#   experiences) as the primary metric, the NASA asymmetric score as the
#   decision-aware secondary metric, and the share of late predictions.
#
# **Why asymmetric?** A late prediction (we say 80 cycles left, the engine
# fails at 60) risks an in-service failure; an early one wastes 20 cycles of
# useful life. Those costs are not equal, so we report a metric that is not
# symmetric, and we always show *which side* the errors fall on.
#
# > **Intuition.** You have seen this structure before. In cross-subject EEG
# > decoding, each subject contributes many trials, trials of one subject are
# > strongly correlated, and the honest question is "how well does the decoder
# > do on a *new* subject?". Here each engine contributes many windows, windows
# > of one engine overlap, and the honest question is "how well does the model
# > do on a *new* engine?". Engines are subjects; cycles are trials. *Where the
# > analogy breaks down:* an EEG subject's state does not drift towards a
# > terminal event, whereas every engine here degrades monotonically to
# > failure, so the label itself is a function of time within the trajectory.

# %% [markdown]
# ## Key equations
#
# **Capped RUL label** for engine $u$ at cycle $t$, with $T_u$ its last
# (failure) cycle:
#
# $$
# y_{u,t} = \min\left(T_u - t,\; R_{\max}\right)
# $$
#
# *In words:* the number of cycles left, but never more than $R_{\max}$.
# Example: $T_u = 200$, $t = 50$, $R_{\max} = 125$ gives
# $\min(150, 125) = 125$; at $t = 150$ it gives $50$.
#
# **RMSE** over $n$ test engines, with $d_i = \hat{y}_i - y_i$:
#
# $$
# \operatorname{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} d_i^{2}}
# $$
#
# *In words:* the typical size of an error, in cycles, with large errors
# weighted more. Example: errors $-10, 0, 20$ give
# $\sqrt{(100 + 0 + 400)/3} \approx 12.9$ cycles.
#
# **NASA (PHM08) score**:
#
# $$
# s = \sum_{i=1}^{n} s_i, \qquad
# s_i = \begin{cases} e^{-d_i/13} - 1 & d_i < 0 \ \text{(early)} \\ e^{d_i/10} - 1 & d_i \ge 0 \ \text{(late)} \end{cases}
# $$
#
# *In words:* each engine adds an exponential penalty; being 10 cycles late
# costs as much as being 13 cycles early, and the gap widens fast. Example:
# $d = -13$ gives $e^{1} - 1 \approx 1.72$; $d = +13$ gives
# $e^{1.3} - 1 \approx 2.67$; $d = +50$ gives $e^{5} - 1 \approx 147$.
# Because it is a *sum*, a few very late engines dominate it — always look at
# the per-engine values too. (Check the exact form against Saxena et al.,
# 2008; this is the form used in the PHM08 challenge.)
#
# **Least-squares slope** of a sensor $x$ over a window of $W$ cycles,
# $t = 1, \dots, W$:
#
# $$
# \beta = \frac{\sum_{t} (t - \bar{t})(x_t - \bar{x})}{\sum_{t} (t - \bar{t})^{2}}
# $$
#
# *In words:* how fast the sensor is drifting per cycle inside the window.
# Example: $x = (1, 2, 4)$ gives $\beta = ((-1)(-1.33) + 0 + (1)(1.67))/2 = 1.5$.
#
# We code the two metrics now and check them on the hand-computed cases above.

# %%
def rmse(y_true, y_pred) -> float:
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def nasa_terms(y_true, y_pred) -> np.ndarray:
    """Per-engine PHM08 penalty; d = pred - true, late (d >= 0) is penalised harder."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def nasa_score(y_true, y_pred) -> float:
    return float(nasa_terms(y_true, y_pred).sum())


def slope(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Least-squares slope along `axis` (time)."""
    w = x.shape[axis]
    t = np.arange(w, dtype=float) - (w - 1) / 2
    shape = [1] * x.ndim
    shape[axis] = w
    t = t.reshape(shape)
    return (t * (x - x.mean(axis=axis, keepdims=True))).sum(axis=axis) / (t ** 2).sum()


# Check: hand-computed cases from the equations above.
assert abs(rmse([0, 0, 0], [-10, 0, 20]) - math.sqrt(500 / 3)) < 1e-9
assert abs(nasa_score([0], [-13]) - (math.e - 1)) < 1e-9
assert abs(nasa_score([0], [13]) - (math.exp(1.3) - 1)) < 1e-9
assert nasa_score([5], [5]) == 0.0
assert abs(slope(np.array([[1.0, 2.0, 4.0]]))[0] - 1.5) < 1e-9
print("metric checks passed")

# %% [markdown]
# ## Step 2 · Data acquisition
#
# **Your turn (about 30 minutes).** Write a function that downloads the
# official archive once, caches it in the artifact directory, checks its size
# and SHA-256, and extracts only the FD001 files. Decide what should happen
# when the checksum does not match.
#
# **Our solution.** The NASA PCoE data repository distributes C-MAPSS as
# `6. Turbofan Engine Degradation Simulation Data Set.zip`, which contains a
# second ZIP, `CMAPSSData.zip`, with the text files (we found this nesting by
# opening the archive; the brief marked the file names as *to verify*). The
# download link below is the one NASA's PCoE repository points to; open the
# [landing page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
# and confirm it is still current — **to verify** each time you start.
#
# The checksum below is the one we measured on the archive we downloaded on
# the date this walkthrough was written. If NASA re-packages the file, the hash
# changes: the code then stops, so that you look at what changed before
# trusting any number computed from it.

# %%
CMAPSS_URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
              "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
LANDING_PAGE = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
EXPECTED_BYTES = 12_429_152
EXPECTED_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"
FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", "readme.txt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print("cached:", dest.name)
        return dest
    tmp = dest.with_suffix(".part")
    print("downloading", url)
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    tmp.rename(dest)
    return dest


def extract_fd001(outer_zip: Path, out_dir: Path) -> dict[str, Path]:
    """Open the outer ZIP, find CMAPSSData.zip inside it, extract the FD001 files."""
    with zipfile.ZipFile(outer_zip) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        inner_bytes = outer.read(inner_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        names = inner.namelist()
        print("inner archive:", len(names), "files, e.g.", sorted(names)[:4])
        for name in FILES:
            (out_dir / name).write_bytes(inner.read(name))
    return {name: out_dir / name for name in FILES}


# %% [markdown]
# **The synthetic sample (automated checks only).** When
# `MLBOOTCAMP_CAPSTONE_SAMPLE=1`, the function below writes a fake archive with
# the same nesting, file names and text format as the official one: 26
# space-separated numbers per row, two trailing spaces on every line, a few
# constant sensors, run-to-failure training engines, truncated test engines and
# a one-number-per-engine RUL file. The degradation is a made-up smooth trend
# plus noise. It lets the course's CI run every cell without the network; it
# is **not** C-MAPSS, and no number computed from it is a result.

# %%
# name: (base value, total drift at failure, noise sd, decimals); drift 0 = constant
SYNTH_SENSORS = {
    "s1": (518.67, 0, 0, 2), "s2": (642.2, 1.6, 0.35, 2), "s3": (1588.0, 25, 4.5, 2),
    "s4": (1404.0, 36, 5.5, 2), "s5": (14.62, 0, 0, 2), "s6": (21.61, 0, 0, 2),
    "s7": (554.0, -3.2, 0.6, 2), "s8": (2388.05, 0.25, 0.05, 2), "s9": (9060.0, 45, 12, 2),
    "s10": (1.30, 0, 0, 2), "s11": (47.35, 1.3, 0.17, 2), "s12": (521.8, -2.7, 0.5, 2),
    "s13": (2388.05, 0.25, 0.05, 2), "s14": (8140.0, 30, 11, 2), "s15": (8.42, 0.12, 0.03, 4),
    "s16": (0.03, 0, 0, 2), "s17": (392.0, 5.5, 1.2, 0), "s18": (2388.0, 0, 0, 0),
    "s19": (100.0, 0, 0, 2), "s20": (38.95, -0.65, 0.14, 2), "s21": (23.37, -0.38, 0.08, 4),
}


def _synthetic_engine(rng, life: int) -> list[str]:
    onset = max(5, life - int(rng.integers(110, 170)))
    t = np.arange(1, life + 1)
    wear = np.clip((t - onset) / (life - onset), 0, None) ** 1.6 + rng.normal(0, 0.02)
    lines = []
    for k, cyc in enumerate(t):
        row = [f"{cyc}", f"{rng.normal(0, 0.002):.4f}", f"{rng.normal(0, 0.0003):.4f}", "100.0"]
        for name, (base, drift, sd, dec) in SYNTH_SENSORS.items():
            if name == "s6":
                value = 21.61 if rng.random() < 0.98 else 21.60
            else:
                value = base + drift * wear[k] + rng.normal(0, sd) if sd else base
            row.append(f"{value:.{dec}f}")
        lines.append(row)
    return lines


def write_synthetic_cmapss(outer_zip: Path, n_train: int = 100, n_test: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    train, test, rul = [], [], []
    for unit in range(1, n_train + 1):
        life = int(np.clip(128 + rng.gamma(2.5, 32), 128, 362))
        train += [f"{unit} " + " ".join(r[0:1] + r[1:]) + "  " for r in _synthetic_engine(rng, life)]
    for unit in range(1, n_test + 1):
        life = int(np.clip(128 + rng.gamma(2.5, 32), 128, 362))
        true_rul = int(rng.integers(7, min(146, life - 31)))
        rows = _synthetic_engine(rng, life)[: life - true_rul]
        test += [f"{unit} " + " ".join(r) + "  " for r in rows]
        rul.append(f"{true_rul} ")
    readme = "SYNTHETIC SAMPLE written by the walkthrough for automated checks. Not C-MAPSS data.\n"
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("train_FD001.txt", "\n".join(train) + "\n")
        z.writestr("test_FD001.txt", "\n".join(test) + "\n")
        z.writestr("RUL_FD001.txt", "\n".join(rul) + "\n")
        z.writestr("readme.txt", readme)
    with zipfile.ZipFile(outer_zip, "w") as z:
        z.writestr("6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip", inner.getvalue())


# %%
outer = DATA_DIR / "cmapss_outer.zip"
if SAMPLE:
    if not outer.exists():
        write_synthetic_cmapss(outer)
    print("synthetic archive written; checksum check not applicable")
else:
    download(CMAPSS_URL, outer)
    size, digest = outer.stat().st_size, sha256(outer)
    print(f"size {size:,} bytes | sha256 {digest}")
    if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise RuntimeError("The archive differs from the one this walkthrough was checked against. "
                           "Inspect it (file list, readme), then update EXPECTED_* deliberately.")
paths = extract_fd001(outer, DATA_DIR)
print("source:", LANDING_PAGE)
print("terms : NASA data, generally a US Government work (public domain in the US) - to verify on the page")
print("cite  : Saxena, Goebel, Simon, Eklund, PHM08 (2008), DOI 10.1109/PHM.2008.4711414")
print(paths["readme.txt"].read_text(encoding="latin-1")[:400])

# %% [markdown]
# **Expected (official data):** 12,429,152 bytes, the hash above, an inner
# archive of 14 files, and a readme that lists FD001 as 100 training and 100
# test trajectories, one operating condition (sea level) and one fault mode
# (HPC degradation).
#
# **Why this way.** The raw files are cached under the artifact directory and
# never committed. A size-plus-hash check catches truncated downloads and silent
# re-packaging. *Alternative considered:* checking only the inner text files'
# hashes; that survives re-zipping, but we prefer to know when the published
# archive changes. *Common mistake:* downloading the much larger N-CMAPSS
# ("Data Set 2"), a different dataset, by following the wrong link.

# %% [markdown]
# ## Step 3 · Data audit
#
# **Your turn (about 45 minutes).** Load the three FD001 files. Assert the
# column count, count engines and rows, look for missing values, find the
# constant sensors, measure trajectory lengths, and plot a few sensors against
# "cycles before failure". Write down anything surprising.
#
# **Our solution.** First the trailing-space trap: splitting a line on single
# spaces does not give 26 fields.

# %%
first_line = paths["train_FD001.txt"].read_text().splitlines()[0]
print(repr(first_line[-30:]))
print("naive split(' ') gives", len(first_line.split(" ")), "fields; split() gives", len(first_line.split()))

COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]


def read_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    assert df.shape[1] == 26, f"{path.name}: expected 26 columns, got {df.shape[1]}"
    df.columns = COLS
    return df.astype({"unit": int, "cycle": int})


train_raw = read_cmapss(paths["train_FD001.txt"])
test_raw = read_cmapss(paths["test_FD001.txt"])
rul_test = pd.read_csv(paths["RUL_FD001.txt"], sep=r"\s+", header=None)[0].to_numpy()
print("train:", train_raw.shape, "engines", train_raw.unit.nunique(),
      "| test:", test_raw.shape, "engines", test_raw.unit.nunique(), "| RUL rows", len(rul_test))
print("missing values:", int(train_raw.isna().sum().sum() + test_raw.isna().sum().sum()))

# %% [markdown]
# **Expected (official data):** a naive split gives 28 fields (two empty
# strings from the trailing spaces); 20,631 training rows from 100 engines,
# 13,096 test rows from 100 engines, 100 RUL values, no missing values.

# %%
life = train_raw.groupby("unit")["cycle"].max()
test_len = test_raw.groupby("unit")["cycle"].max()
print("training life (cycles):  min", life.min(), "median", int(life.median()), "max", life.max())
print("test trajectory length: min", test_len.min(), "median", int(test_len.median()), "max", test_len.max())
print("test RUL truth:         min", rul_test.min(), "median", int(np.median(rul_test)), "max", rul_test.max())

# cycles must run 1, 2, 3, ... without gaps inside every engine
for name, df in [("train", train_raw), ("test", test_raw)]:
    ok = df.groupby("unit")["cycle"].apply(lambda c: (c.diff().dropna() == 1).all() and c.iloc[0] == 1)
    assert ok.all(), f"{name}: cycles are not consecutive from 1"
print("cycles are consecutive from 1 in every engine")

stats = pd.DataFrame({"std": train_raw[COLS[2:]].std(), "n_unique": train_raw[COLS[2:]].nunique()})
print(stats.T.round(4).to_string())

# %% [markdown]
# **Expected (official data):** training lives from 128 to 362 cycles (median
# 199); test trajectories from 31 to 303 cycles; test truth from 7 to 145.
# `setting3`, `s1`, `s5`, `s10`, `s16`, `s18` and `s19` have a single value and
# `s6` only two.
#
# **Oddities worth writing in the data card.**
#
# 1. **Unit IDs restart in each file.** Test engine 1 is *not* training engine
#    1; they are different simulated engines that share a number. Any join on
#    `unit` across files is a bug. We prefix IDs when we combine them.
# 2. **The readme lists "sensor measurement 26"**, but the files contain 21
#    sensors (26 columns minus unit, cycle and three settings). Trust the data
#    and the column count, not the typo.
# 3. **Settings 1 and 2 vary slightly but FD001 has one operating condition**:
#    their variation is noise around sea-level conditions. We keep them (the
#    rule below decides) but do not expect them to matter.
#
# Now the constant-sensor rule. It is computed on training engines only and
# re-computed inside every fold later, so no information flows from
# validation or test engines into the choice.

# %%
def informative_columns(df: pd.DataFrame) -> list[str]:
    """Settings and sensors with more than two distinct values in THESE rows."""
    return [c for c in COLS[2:] if df[c].nunique() > 2]


FEATURES = informative_columns(train_raw)
print(len(FEATURES), "informative columns:", FEATURES)
assert 26 - 2 - len(FEATURES) >= 7 or SAMPLE

# %%
train_raw["rul"] = train_raw.groupby("unit")["cycle"].transform("max") - train_raw["cycle"]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, s in zip(axes, ["s2", "s4", "s7", "s11"]):
    for u in train_raw.unit.unique()[:8]:
        g = train_raw[train_raw.unit == u]
        ax.plot(-g["rul"], g[s], lw=0.8, alpha=0.8)
    ax.axvline(-125, color="grey", ls="--", lw=1)
    ax.set_title(s)
    ax.set_xlabel("cycles before failure")
plt.suptitle("Four sensors for eight training engines; dashed line: 125 cycles before failure", y=1.05)
plt.show()

# %% [markdown]
# **What to notice:** for most of an engine's life the sensors wander around a
# flat level set by the engine's initial wear; the drift towards failure
# becomes visible only in the last 100–150 cycles, and engines start at
# different offsets. This is the evidence behind a capped label (Step 4) and
# behind per-window features such as slope (Step 6).

# %% [markdown]
# ## Step 4 · A leakage-safe split for this problem
#
# **Your turn (about 40 minutes).** Decide how to split the 100 training
# engines for model selection, how validation should mimic the official test
# set, and what the training label is. Write the assertion that would catch an
# engine on both sides.
#
# **Our solution.**
#
# * **Split by engine.** Five folds of 20 engines each; all windows of an
#   engine fall in one fold. This is `GroupKFold` with the engine ID as the
#   group — exactly the leave-subjects-out scheme you would use for a
#   cross-subject EEG result. Scaling and the constant-sensor rule are fitted
#   on each fold's training engines only.
# * **Validation mimics the test set.** The official test engines are cut
#   off some time before failure and scored once, at their last cycle. So for
#   each validation engine we pick 5 random "cut" cycles and score the window
#   ending there against the **uncapped** true RUL. We restrict cuts to true
#   RUL between 1 and 150 cycles: the range in which a planner is actually
#   making a decision. (That restriction is a choice; we state it rather than
#   tune it.)
# * **Training label.** RUL capped at $R_{\max}$. We start from 125 — the plot
#   above shows no visible drift earlier than roughly 125–150 cycles before
#   failure, and values near 125–130 are the common choice in the literature —
#   and then compare 100, 125 and 150 in cross-validation (Step 6) instead of
#   taking it on trust.
# * **The official test set** is touched once, in Step 7.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap1-engine-split.svg" alt="Random window split puts windows of one engine on both sides; group split keeps each engine on one side" width="100%">
#
# *What to notice:* in the left panel neighbouring windows of engine 7 sit on
# both sides of the split and share 29 of their 30 cycles.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap1-rul-cap.svg" alt="The capped RUL label is flat at R_max and then falls linearly to zero at failure" width="100%">
#
# *What to notice:* the capped label only differs from the truth early in
# life, where no sensor can tell 180 cycles left from 140.

# %%
W = 30
N_FOLDS = 5
rng = np.random.default_rng(SEED)
train_units = np.sort(train_raw.unit.unique())
perm = rng.permutation(train_units)
FOLD_UNITS = [np.sort(perm[k::N_FOLDS]) for k in range(N_FOLDS)]

# Check: every engine is in exactly one validation fold, and never on both sides.
all_val = np.concatenate(FOLD_UNITS)
assert len(all_val) == len(set(all_val)) == len(train_units)
for k, val_u in enumerate(FOLD_UNITS):
    tr_u = np.setdiff1d(train_units, val_u)
    assert len(np.intersect1d(tr_u, val_u)) == 0, f"fold {k}: engine on both sides"
print("folds:", [len(f) for f in FOLD_UNITS], "engines; no engine on both sides")

# Validation cut points: 5 per engine, with true (uncapped) RUL in [1, 150] and cycle >= W.
cut_rng = np.random.default_rng(SEED + 1)
cands = train_raw[(train_raw.rul.between(1, 150)) & (train_raw.cycle >= W)]
CUTS = (cands.groupby("unit", group_keys=False)
        .apply(lambda g: g.sample(n=min(5, len(g)), random_state=int(cut_rng.integers(1 << 30))))
        [["unit", "cycle", "rul"]].reset_index(drop=True))
print(len(CUTS), "validation cut points; true RUL range", CUTS.rul.min(), "-", CUTS.rul.max())

# %% [markdown]
# **Why this way.** The thing we want to estimate is the error on an engine the
# model has never seen. *Alternatives considered:* (a) a random split of windows
# — leaks, and we measure by how much in Step 6; (b) a single 80/20 engine split
# — honest but noisy with only 20 validation engines, so we use five folds;
# (c) scoring every window of a validation engine — it over-weights long-lived
# engines and early-life windows, which the test protocol never scores.
# *Common mistakes:* fitting the scaler on all rows before splitting; choosing
# the cap by looking at test RMSE; joining train and test on `unit`.

# %% [markdown]
# ## Step 5 · Windows and baselines
#
# **Your turn (about 60 minutes).** Write one function that turns an engine's
# rows into windows of length $W$, including engines (or early cycles) with
# fewer than $W$ rows, and use it everywhere: training, validation, test and the
# API. Then fit two baselines on the same folds: a constant and a linear model.
#
# **Our solution.** Windows end at every cycle. When fewer than $W$ cycles
# exist, we **repeat the first row** at the front. Zero-padding would insert
# physically impossible sensor values (a temperature of 0) that the model must
# learn to ignore; repeating the first row says "assume the engine looked like
# this before we started watching", which is mild and identical in training
# and serving. We test the padding on a 3-cycle toy engine.

# %%
from numpy.lib.stride_tricks import sliding_window_view


def engine_windows(values: np.ndarray, w: int = W) -> np.ndarray:
    """All windows ending at each cycle of ONE engine: (n_cycles, w, n_channels).

    Cycles before the start are filled by repeating the first row.
    """
    pad = np.repeat(values[:1], w - 1, axis=0)
    full = np.concatenate([pad, values], axis=0)
    return sliding_window_view(full, w, axis=0).transpose(0, 2, 1).copy()


def make_windows(df: pd.DataFrame, cols: list[str], w: int = W, last_only: bool = False):
    """Windows for every engine in `df`. Returns X (n, w, c), unit, cycle, n_observed."""
    xs, units, cycles, n_obs = [], [], [], []
    for unit, g in df.groupby("unit", sort=True):
        win = engine_windows(g[cols].to_numpy(np.float32), w)
        cyc = g["cycle"].to_numpy()
        idx = [len(g) - 1] if last_only else range(len(g))
        xs.append(win[idx])
        units += [unit] * len(idx)
        cycles += list(cyc[idx])
        n_obs += [min(i + 1, w) for i in idx]
    return np.concatenate(xs), np.array(units), np.array(cycles), np.array(n_obs)


# Check: the padding rule on a toy engine with 3 cycles and 2 channels.
toy = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
tw = engine_windows(toy, w=5)
assert tw.shape == (3, 5, 2)
assert (tw[2, :, 0] == [1, 1, 1, 2, 3]).all() and (tw[0, :, 1] == 10).all()
print("padding check passed; last toy window:", tw[2, :, 0])

# %%
def window_features(X: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Hand-made summary of each window: mean, std, min, max, last value, slope."""
    parts = {"mean": X.mean(1), "std": X.std(1), "min": X.min(1), "max": X.max(1),
             "last": X[:, -1, :], "slope": slope(X, axis=1)}
    return pd.DataFrame(np.concatenate(list(parts.values()), axis=1),
                        columns=[f"{c}_{k}" for k in parts for c in cols])


X_all, U_all, C_all, NOBS_all = make_windows(train_raw, FEATURES)
RUL_all = (train_raw.groupby("unit")["cycle"].transform("max") - train_raw["cycle"]).to_numpy()
# make_windows walks the engines in sorted order; train_raw is already sorted by unit and cycle
assert (U_all == train_raw.unit.to_numpy()).all() and (C_all == train_raw.cycle.to_numpy()).all()
F_all = window_features(X_all, FEATURES)
cut_key = set(zip(CUTS.unit, CUTS.cycle))
IS_CUT = np.array([(u, c) in cut_key for u, c in zip(U_all, C_all)])
print("windows:", X_all.shape, "| features:", F_all.shape, "| padded windows:", int((NOBS_all < W).sum()))

# %% [markdown]
# **Expected (official data):** 20,631 windows of shape (30, 16) — one per
# training row — 96 summary features, and 2,900 padded windows (the first 29
# cycles of each of the 100 engines).
#
# Now the cross-validation harness. Every model is a function
# `fit_predict(train_mask, cut_mask, cap) -> predictions at the cuts`, so all
# models see exactly the same folds and are scored the same way.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CAP = 125


def cv_evaluate(fit_predict, cap=CAP, name="", folds=FOLD_UNITS) -> dict:
    """Engine-level CV. Score at the validation cuts against the UNCAPPED truth."""
    preds = np.full(len(U_all), np.nan)
    for val_u in folds:
        val = np.isin(U_all, val_u)
        tr = ~val
        cut = val & IS_CUT
        preds[cut] = np.clip(fit_predict(tr, cut, cap), 0, None)
    m = IS_CUT
    out = {"model": name, "cap": cap, "cv_rmse": rmse(RUL_all[m], preds[m]),
           "cv_nasa_per_engine": nasa_score(RUL_all[m], preds[m]) / m.sum(),
           "cv_late_share": float(np.mean(preds[m] > RUL_all[m]))}
    return out, preds


def fp_mean(tr, cut, cap):
    return np.full(cut.sum(), np.minimum(RUL_all[tr], cap).mean())


def fp_ridge(tr, cut, cap):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(F_all[tr], np.minimum(RUL_all[tr], cap))
    return model.predict(F_all[cut])


t0 = time.time()
cv_rows, cv_preds = [], {}
for name, fp in [("constant (mean capped RUL)", fp_mean), ("ridge on window features", fp_ridge)]:
    row, cv_preds[name] = cv_evaluate(fp, name=name)
    cv_rows.append(row)
print(pd.DataFrame(cv_rows).round(3).to_string(index=False), f"\n({time.time() - t0:.1f}s)")

# %% [markdown]
# **Expected:** the constant baseline is poor (an RMSE of roughly 40–50 cycles
# on the official data — it predicts the same number for an engine about to fail
# and one with 150 cycles left); ridge should roughly halve it.
#
# **Why these baselines.** The constant tells you how much signal there is at
# all. The linear model on the same window features as the tree model isolates
# *one* change at a time: when the gradient-boosted model wins, it wins because
# of non-linearity, not because it saw better features. *Alternative
# considered:* ridge on the last cycle only (the brief's version); try it — it
# is noisier because a single cycle carries all the sensor noise.

# %% [markdown]
# ## Step 6 · Improved models
#
# **Your turn (about 2 sessions).** (a) Fit gradient-boosted trees on the
# window features; (b) compare RUL caps 100, 125 and 150; (c) measure how much
# a random window split flatters the score; (d) train a compact temporal model
# in PyTorch with early stopping on *engines*, with several seeds.
#
# ### 6a · Gradient-boosted trees on window features

# %%
def fp_gbm(tr, cut, cap):
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=50, random_state=SEED)
    model.fit(F_all[tr], np.minimum(RUL_all[tr], cap))
    return model.predict(F_all[cut])


t0 = time.time()
cap_rows = []
for cap in (100, 125, 150):
    row, p = cv_evaluate(fp_gbm, cap=cap, name="gradient boosting")
    cap_rows.append(row)
    if cap == CAP:
        cv_preds["gradient boosting"] = p
cap_table = pd.DataFrame(cap_rows)
print(cap_table.round(3).to_string(index=False), f"\n({time.time() - t0:.1f}s)")
CAP = int(cap_table.loc[cap_table.cv_rmse.idxmin(), "cap"])
cv_rows.append(cap_table[cap_table.cap == 125].iloc[0].to_dict())
print("cap chosen by engine-level CV RMSE (uncapped truth):", CAP)

# %% [markdown]
# **Expected (official data):** gradient boosting beats ridge by a few cycles of
# RMSE. Caps of 125 and 150 are usually close, and 100 is worse, because cuts
# with true RUL up to 150 can never be predicted above 100. Your chosen cap is
# whatever CV says; the rest of the notebook uses the printed value.
#
# **Common mistake:** choosing the cap by comparing *capped* errors. A smaller
# cap shrinks both the labels and the errors, so it always looks better; only
# the uncapped truth makes the caps comparable.
#
# ### 6b · How much does a random window split flatter the score?
#
# We train the same model twice: once with a 5-fold split of *windows* at
# random, once with our engine-level folds, and in both cases score every held-out
# window against the capped label (the like-for-like comparison).

# %%
from sklearn.model_selection import KFold


def window_level_rmse(split_masks) -> float:
    errs = []
    y = np.minimum(RUL_all, CAP)
    for val in split_masks:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=50, random_state=SEED)
        model.fit(F_all[~val], y[~val])
        errs.append(model.predict(F_all[val]) - y[val])
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))


t0 = time.time()
random_masks = []
for _, val_idx in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(F_all):
    m = np.zeros(len(F_all), bool)
    m[val_idx] = True
    random_masks.append(m)
group_masks = [np.isin(U_all, f) for f in FOLD_UNITS]
RMSE_RANDOM = window_level_rmse(random_masks)
RMSE_GROUP = window_level_rmse(group_masks)
print(f"random window split : RMSE {RMSE_RANDOM:.2f} cycles (optimistic)")
print(f"engine-level split  : RMSE {RMSE_GROUP:.2f} cycles (honest)")
print(f"the random split understates the error by {100 * (1 - RMSE_RANDOM / RMSE_GROUP):.0f}%  ({time.time() - t0:.1f}s)")

# %% [markdown]
# **What to notice:** the random split reports a smaller error, for the same
# model and features. Its validation windows have near-identical neighbours in
# training (same engine, 29 shared cycles, same initial-wear offset), so the
# model partly recognises the engine instead of reading its degradation. This
# gap is the number to quote when an interviewer asks "why not split randomly?"
# — the same story as within-subject versus cross-subject accuracy in BCI.
#
# ### 6c · A compact 1D-CNN in PyTorch
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap1-cnn.svg" alt="Window, two Conv1d layers, global average pooling, linear head" width="100%">
#
# *What to notice:* the network sees the raw window, not our hand-made
# summaries, and it is tiny on purpose.
#
# Design decisions: channels standardised with means and standard deviations
# from the training engines of the current fold; target divided by the cap so
# the output is near 0–1; mean squared error; Adam; early stopping on 20% of the
# **training engines** of the fold (never on windows, never on the validation
# fold that is being scored).

# %%
import torch.nn as nn


class RULCNN(nn.Module):
    def __init__(self, n_channels: int, width: int = 32, kernel: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, width, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(width, width, kernel, padding=kernel // 2), nn.ReLU())
        self.head = nn.Linear(width, 1)

    def forward(self, x):                      # x: (batch, W, channels)
        h = self.conv(x.transpose(1, 2))       # (batch, width, W)
        return self.head(h.mean(dim=2)).squeeze(-1)


def train_cnn(X, y, units, cap, seed, max_epochs=40, patience=6, batch=256):
    """Train on windows; early-stop on 20% of the given engines. Returns (model, scaler, info)."""
    g = np.random.default_rng(seed)
    eng = np.unique(units)
    stop_units = g.choice(eng, size=max(1, len(eng) // 5), replace=False)
    es = np.isin(units, stop_units)
    mu = X[~es].reshape(-1, X.shape[2]).mean(0)
    sd = X[~es].reshape(-1, X.shape[2]).std(0) + 1e-6
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(np.minimum(y, cap) / cap, dtype=torch.float32)
    torch.manual_seed(seed)
    model = RULCNN(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    tr_idx = np.where(~es)[0]
    best, best_state, best_epoch, waited = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for b in np.array_split(g.permutation(tr_idx), max(1, len(tr_idx) // batch)):
            opt.zero_grad()
            loss = ((model(Xs[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(Xs[es]) - yt[es]) ** 2).mean())
        if val_loss < best - 1e-5:
            best, best_epoch, waited = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, (mu, sd), {"best_epoch": best_epoch + 1, "epochs_run": epoch + 1}


def cnn_predict(model, scaler, X, cap) -> np.ndarray:
    mu, sd = scaler
    with torch.no_grad():
        return model(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy() * cap


n_params = sum(p.numel() for p in RULCNN(len(FEATURES)).parameters())
print("CNN parameters:", n_params)

CNN_INFO = []


def fp_cnn(tr, cut, cap):
    model, scaler, info = train_cnn(X_all[tr], RUL_all[tr], U_all[tr], cap, seed=SEED)
    CNN_INFO.append(info)
    return cnn_predict(model, scaler, X_all[cut], cap)


t0 = time.time()
row, cv_preds["1D-CNN"] = cv_evaluate(fp_cnn, cap=CAP, name="1D-CNN (1 seed per fold)")
cv_rows.append(row)
CNN_CV_SECONDS = time.time() - t0
print(f"CNN 5-fold CV: {CNN_CV_SECONDS:.0f}s on CPU;", "best epochs per fold:",
      [i["best_epoch"] for i in CNN_INFO])
cv_table = pd.DataFrame(cv_rows)
print(cv_table.round(3).to_string(index=False))

# %% [markdown]
# **Expected (official data):** about 7,500 parameters; well under a minute per
# fold on a laptop CPU; a CV RMSE in the same range as gradient boosting —
# sometimes a little better, sometimes not. If yours is far worse, check the
# scaling and that the target is divided by the cap.
#
# **Is the difference real?** Both models were scored on the same 500 cuts, so
# we can bootstrap the *paired* difference over validation **engines**.

# %%
def engine_bootstrap(units, stat_fn, n_boot=2000, seed=SEED) -> np.ndarray:
    """Resample engines (not rows) with replacement; return the statistic per resample."""
    g = np.random.default_rng(seed)
    eng = np.unique(units)
    rows_of = {e: np.where(units == e)[0] for e in eng}
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows_of[e] for e in g.choice(eng, size=len(eng), replace=True)])
        out[b] = stat_fn(idx)
    return out


m = IS_CUT
yc, uc = RUL_all[m], U_all[m]
pg, pc = cv_preds["gradient boosting"][m], cv_preds["1D-CNN"][m]
diff = engine_bootstrap(uc, lambda i: rmse(yc[i], pc[i]) - rmse(yc[i], pg[i]))
print(f"CV RMSE(CNN) - RMSE(GBM) = {rmse(yc, pc) - rmse(yc, pg):+.2f} cycles, "
      f"95% CI [{np.percentile(diff, 2.5):+.2f}, {np.percentile(diff, 97.5):+.2f}] (bootstrap over engines)")

# %% [markdown]
# **How to read it.** If the interval contains 0, the cross-validation cannot
# tell the two models apart, and the simpler model (trees: faster, no PyTorch
# in serving) is the sensible default. We still carry both to the test set,
# because the brief asks for an honest comparison, but **the model we serve is
# chosen here, on CV, before the test set is opened**.

# %%
cv_rmse = {"gradient boosting": rmse(yc, pg), "1D-CNN": rmse(yc, pc)}
lo, hi = np.percentile(diff, [2.5, 97.5])
SERVED = "1D-CNN" if hi < 0 else "gradient boosting"
print("model chosen for serving (decided on CV only):", SERVED)

# %% [markdown]
# ## Step 7 · Final evaluation on the official test set (run once)
#
# **Your turn (about 40 minutes).** Retrain the chosen configuration on all
# 100 training engines, predict the last window of each test engine, and report
# RMSE, the NASA score and the share of late predictions with 95% bootstrap
# intervals over test engines. Do not change anything after you see the numbers.
#
# **Our solution.** Gradient boosting is retrained on all engines. The CNN is
# trained with three seeds (each with its own 20% early-stopping engines) and
# the three predictions are averaged; we also report the spread across seeds.

# %%
F_cols = FEATURES  # re-derive the constant-sensor rule on all training engines (unchanged here)
Xte, Ute, Cte, NOBS_te = make_windows(test_raw, F_cols, last_only=True)
assert (Ute == np.arange(1, len(rul_test) + 1)).all(), "RUL file is aligned to test units 1..n in order"
y_test = rul_test.astype(float)
print("test engines:", len(Ute), "| padded test windows:", int((NOBS_te < W).sum()))

t0 = time.time()
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                    min_samples_leaf=50, random_state=SEED)
gbm.fit(F_all, np.minimum(RUL_all, CAP))
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(F_all, np.minimum(RUL_all, CAP))
const = float(np.minimum(RUL_all, CAP).mean())

cnn_models, seed_preds = [], []
for s in range(3):
    model, scaler, info = train_cnn(X_all, RUL_all, U_all, CAP, seed=SEED + 10 + s)
    cnn_models.append((model, scaler))
    seed_preds.append(np.clip(cnn_predict(model, scaler, Xte, CAP), 0, None))
FINAL_TRAIN_SECONDS = time.time() - t0

test_preds = {
    "constant (mean capped RUL)": np.full(len(y_test), const),
    "ridge on window features": np.clip(ridge.predict(window_features(Xte, F_cols)), 0, None),
    "gradient boosting": np.clip(gbm.predict(window_features(Xte, F_cols)), 0, None),
    "1D-CNN (3-seed mean)": np.mean(seed_preds, axis=0),
}
print(f"final training: {FINAL_TRAIN_SECONDS:.0f}s; CNN single-seed test RMSE:",
      [round(rmse(y_test, p), 2) for p in seed_preds])

# %%
def summarise(y, p, units, n_boot=2000) -> dict:
    idx_all = np.arange(len(y))
    boot = engine_bootstrap(units, lambda i: rmse(y[i], p[i]), n_boot=n_boot)
    late = engine_bootstrap(units, lambda i: np.mean(p[i] > y[i]), n_boot=n_boot)
    return {"rmse": rmse(y, p), "rmse_lo": np.percentile(boot, 2.5), "rmse_hi": np.percentile(boot, 97.5),
            "nasa": nasa_score(y, p), "late": float(np.mean(p[idx_all] > y[idx_all])),
            "late_lo": np.percentile(late, 2.5), "late_hi": np.percentile(late, 97.5)}


test_rows = []
for name, p in test_preds.items():
    test_rows.append({"model": name, **summarise(y_test, p, Ute)})
test_table = pd.DataFrame(test_rows)
print(test_table.round(2).to_string(index=False))

d_test = engine_bootstrap(Ute, lambda i: rmse(y_test[i], test_preds["1D-CNN (3-seed mean)"][i])
                          - rmse(y_test[i], test_preds["gradient boosting"][i]))
print(f"\ntest RMSE(CNN) - RMSE(GBM): {test_rows[3]['rmse'] - test_rows[2]['rmse']:+.2f} cycles, "
      f"95% CI [{np.percentile(d_test, 2.5):+.2f}, {np.percentile(d_test, 97.5):+.2f}]")
print("capped-truth RMSE (literature-style, for comparison only):",
      {k: round(rmse(np.minimum(y_test, CAP), v), 2) for k, v in test_preds.items()})

# %% [markdown]
# **Expected (official data):** gradient boosting and the CNN both far ahead of
# the constant; RMSE intervals about ±3–4 cycles wide, because there are only
# 100 test engines. The NASA score is dominated by a handful of engines (next
# step). The capped-truth RMSE is lower than the uncapped one; papers usually
# report the capped version, so compare with care and say which one you quote.
#
# **Why this way.** One configuration, fixed before the test set was opened;
# bootstrap over *engines* because the engine is the unit of prediction;
# paired differences because both models were scored on the same engines.
# *Common mistake:* evaluating every row of the test file. The RUL file labels
# only the last cycle of each engine; other rows have no truth.

# %% [markdown]
# ## Step 8 · Error and slice analysis: early versus late
#
# **Your turn (about 40 minutes).** For the served model: what share of
# engines is predicted late? How large are early and late errors? How does the
# error depend on the true RUL? Which engines dominate the NASA score, and why?
#
# **Our solution.**

# %%
p_srv = test_preds["1D-CNN (3-seed mean)" if SERVED == "1D-CNN" else "gradient boosting"]
d = p_srv - y_test
bins = pd.cut(y_test, [0, 25, 75, np.inf], labels=["< 25", "25-75", "> 75"], right=False)
slices = (pd.DataFrame({"bin": bins, "d": d, "late": d > 0, "abs": np.abs(d)})
          .groupby("bin", observed=True)
          .agg(engines=("d", "size"), rmse=("d", lambda x: float(np.sqrt(np.mean(x ** 2)))),
               mean_error=("d", "mean"), late_share=("late", "mean")))
print("served model:", SERVED)
print(slices.round(2).to_string())
print(f"\nlate: {np.mean(d > 0):.0%} of engines, mean late error {d[d > 0].mean():+.1f} cycles; "
      f"early: mean {d[d < 0].mean():+.1f} cycles")
terms = nasa_terms(y_test, p_srv)
top = np.argsort(terms)[::-1][:5]
print(f"top 5 engines contribute {terms[top].sum() / terms.sum():.0%} of the NASA score:")
print(pd.DataFrame({"test unit": Ute[top], "true RUL": y_test[top], "pred": p_srv[top].round(1),
                    "error": d[top].round(1), "penalty": terms[top].round(1),
                    "cycles observed": test_len.to_numpy()[top]}).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_test, p_srv, s=14, c=np.where(d > 0, "tab:red", "tab:blue"))
axes[0].plot([0, y_test.max()], [0, y_test.max()], "k--", lw=1)
axes[0].axhline(CAP, color="grey", lw=0.8, ls=":")
axes[0].set(xlabel="true RUL (cycles)", ylabel="predicted RUL", title="red = late, blue = early")
axes[1].hist(d, bins=25, color="grey")
axes[1].axvline(0, color="k", lw=1)
axes[1].set(xlabel="error = predicted - true (cycles)", title="error distribution")
plt.show()

# %% [markdown]
# **How to read it (the patterns we expect, check them against your output).**
#
# * **Near failure (< 25 cycles) errors are smallest** — the degradation is
#   unmistakable — and this is the slice a planner cares about most.
# * **Far from failure (> 75 cycles) predictions sit below the truth and flatten
#   at the cap.** That is the cap working as designed: the model says "at least
#   about $R_{\max}$" and is scored against the uncapped truth, so these engines
#   are predicted *early*. Early is the safe side.
# * **The NASA score's worst engines** are typically late predictions on
#   engines with short observed histories or with unusually slow-looking
#   degradation. Hypothesis to test: a short history gives a noisy slope; the
#   `cycles observed` column lets you check it.
#
# **Do not game the score.** Subtracting a constant from every prediction
# lowers the NASA score (fewer late engines) while raising RMSE. If a planner
# wants a safety margin, make it an explicit decision rule — e.g. "schedule a
# visit when the predicted RUL minus 15 cycles falls below the lead time" — and
# report both metrics, not a quietly biased model.

# %% [markdown]
# ## Step 9 · A small API
#
# **Your turn (about 60 minutes).** Build a FastAPI app factory with
# `/health`, `/ready` and `/predict` for 1–100 engines. Validation: cycles
# strictly increasing; 1–500 cycles; exactly the sensors the model uses (unknown
# keys rejected); finite values; three settings. Pad short histories the same
# way as in training and say so. Test it in-process.
#
# **Our solution.** The app takes a *bundle* (model, columns, window, cap,
# version) so that tests can build it with or without a model. Windowing and
# features are the same functions as above — imported in a real package, here
# simply called.

# %%
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SENSOR_COLS = [c for c in FEATURES if c.startswith("s")]
SETTING_COLS = ["setting1", "setting2", "setting3"]


class CycleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cycle: int = Field(ge=1)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: dict[str, float]

    @field_validator("settings")
    @classmethod
    def finite_settings(cls, v):
        if not all(math.isfinite(x) for x in v):
            raise ValueError("settings must be finite")
        return v

    @field_validator("sensors")
    @classmethod
    def finite_sensors(cls, v):
        if not all(math.isfinite(x) for x in v.values()):
            raise ValueError("sensor values must be finite")
        return v


class EngineIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str = Field(min_length=1, max_length=64)
    cycles: list[CycleIn] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def increasing(self):
        c = [x.cycle for x in self.cycles]
        if any(b <= a for a, b in zip(c, c[1:])):
            raise ValueError("cycles must be strictly increasing and unique")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engines: list[EngineIn] = Field(min_length=1, max_length=100)


def make_bundle(kind: str) -> dict:
    if kind == "gradient boosting":
        predict = lambda X: gbm.predict(window_features(X, FEATURES))  # noqa: E731
    else:
        predict = lambda X: np.mean([cnn_predict(m, s, X, CAP) for m, s in cnn_models], axis=0)  # noqa: E731
    return {"predict": predict, "columns": FEATURES, "window": W, "rul_cap": CAP,
            "model_version": f"cmapss-fd001-{'gbm' if kind == 'gradient boosting' else 'cnn'}-cap{CAP}-v1"}


def create_app(bundle: dict | None) -> FastAPI:
    app = FastAPI(title="C-MAPSS RUL (walkthrough)")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if bundle is None:
            return JSONResponse({"status": "not_ready", "detail": "no model loaded"}, status_code=503)
        return {"status": "ready", "model_version": bundle["model_version"],
                "window": bundle["window"], "rul_cap": bundle["rul_cap"]}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if bundle is None:
            raise HTTPException(503, "model not loaded")
        sensors_needed = [c for c in bundle["columns"] if c.startswith("s")]
        windows, meta = [], []
        for eng in req.engines:
            for c in eng.cycles:
                if set(c.sensors) != set(sensors_needed):
                    missing = sorted(set(sensors_needed) - set(c.sensors))
                    extra = sorted(set(c.sensors) - set(sensors_needed))
                    raise HTTPException(422, f"{eng.unit_id} cycle {c.cycle}: missing {missing}, unknown {extra}")
            rows = pd.DataFrame([{**dict(zip(SETTING_COLS, c.settings)), **c.sensors} for c in eng.cycles])
            win = engine_windows(rows[bundle["columns"]].to_numpy(np.float32), bundle["window"])[-1:]
            windows.append(win)
            meta.append((eng.unit_id, len(eng.cycles)))
        preds = np.clip(bundle["predict"](np.concatenate(windows)), 0, None)
        return {"model_version": bundle["model_version"],
                "predictions": [{"unit_id": u, "rul_cycles": round(float(p), 1),
                                 "padded": n < bundle["window"], "n_cycles_used": min(n, bundle["window"])}
                                for (u, n), p in zip(meta, preds)]}

    return app


def engine_payload(df: pd.DataFrame, unit: int, last_n: int, name: str) -> dict:
    g = df[df.unit == unit].tail(last_n)
    return {"unit_id": name, "cycles": [
        {"cycle": int(r.cycle), "settings": [float(r.setting1), float(r.setting2), float(r.setting3)],
         "sensors": {s: float(getattr(r, s)) for s in SENSOR_COLS}} for r in g.itertuples()]}


client = TestClient(create_app(make_bundle(SERVED)))
assert client.get("/health").json() == {"status": "ok"}
assert client.get("/ready").json()["rul_cap"] == CAP
body = {"engines": [engine_payload(test_raw, 1, 60, "test-1"), engine_payload(test_raw, 2, 5, "test-2")]}
r = client.post("/predict", json=body)
assert r.status_code == 200, r.text
out = r.json()["predictions"]
print(r.json()["model_version"], out)
# The API must agree with the offline pipeline on the same engine.
offline = test_preds["1D-CNN (3-seed mean)" if SERVED == "1D-CNN" else "gradient boosting"][0]
assert abs(out[0]["rul_cycles"] - offline) < 0.1 and out[1]["padded"] is True

# Validation rules: each must be rejected with 422.
bad = engine_payload(test_raw, 1, 3, "bad")
bad["cycles"][1]["cycle"] = bad["cycles"][0]["cycle"]                     # duplicate cycle
assert client.post("/predict", json={"engines": [bad]}).status_code == 422
bad = engine_payload(test_raw, 1, 3, "bad")
del bad["cycles"][0]["sensors"][SENSOR_COLS[0]]                           # missing sensor
assert client.post("/predict", json={"engines": [bad]}).status_code == 422
bad = engine_payload(test_raw, 1, 3, "bad")
bad["cycles"][0]["sensors"]["s99"] = 1.0                                  # unknown sensor
assert client.post("/predict", json={"engines": [bad]}).status_code == 422
bad = engine_payload(test_raw, 1, 3, "bad")
bad["cycles"][0]["settings"] = [0.0, 0.0]                                 # wrong settings length
assert client.post("/predict", json={"engines": [bad]}).status_code == 422
assert client.post("/predict", json={"engines": []}).status_code == 422   # empty batch
# Not-ready state: health still answers, ready and predict refuse.
cold = TestClient(create_app(None))
assert cold.get("/health").status_code == 200 and cold.get("/ready").status_code == 503
assert cold.post("/predict", json=body).status_code == 503
print("API checks passed: prediction matches offline, padding flagged, 5 validation rules, not-ready state")

# %% [markdown]
# **Why this way.** The service and the offline evaluation share the windowing
# function, so the test "API prediction equals offline prediction for test
# engine 1" catches training/serving skew. Readiness is separate from liveness
# so an orchestrator does not route traffic to a process without a model.
# *Common mistakes:* silently ignoring unknown sensor keys (a renamed sensor
# would then be treated as missing data); padding differently in the API; not
# clipping negative RUL. In your repository this moves to `src/<package>/api.py`
# with these checks as `pytest` tests (Chapters 10 and 12 show the pattern).

# %% [markdown]
# ## Step 10 · Write-up: results table, limitations, next steps
#
# **Your turn (about 60 minutes).** Produce the README results table *from
# the computed numbers* — never type a number by hand — and write the
# limitations a sceptical reviewer would raise.
#
# **Our solution.** The cell below prints a markdown table you can paste into
# your README. On the synthetic sample it prints a warning instead of a table
# you could mistake for results.

# %%
def results_markdown() -> str:
    lines = ["| Model | Split | RMSE, cycles (95% CI) | NASA score | Late predictions (95% CI) |",
             "|---|---|---|---:|---|"]
    for r in test_rows:
        lines.append(f"| {r['model']} | official FD001 test, 100 engines, uncapped truth | "
                     f"{r['rmse']:.1f} ({r['rmse_lo']:.1f}–{r['rmse_hi']:.1f}) | {r['nasa']:.0f} | "
                     f"{r['late']:.0%} ({r['late_lo']:.0%}–{r['late_hi']:.0%}) |")
    for r in cv_rows:
        lines.append(f"| {r['model']} | 5-fold engine-level CV on train, 500 cuts | {r['cv_rmse']:.1f} | "
                     f"– | {r['cv_late_share']:.0%} |")
    lines.append(f"| gradient boosting, **random window split** | 5-fold, window-level, capped truth | "
                 f"{RMSE_RANDOM:.1f} (vs {RMSE_GROUP:.1f} engine-level) | – | – |")
    lines.append("")
    lines.append(f"RUL cap {CAP} (chosen by CV among 100/125/150), window {W} cycles, "
                 f"served model: {SERVED}. CNN: {n_params} parameters, 3 seeds, "
                 f"5-fold CV took {CNN_CV_SECONDS:.0f}s and final training {FINAL_TRAIN_SECONDS:.0f}s on CPU.")
    return "\n".join(lines)


if SAMPLE:
    print("SYNTHETIC SAMPLE RUN - these numbers are NOT results. Do not copy them into a README.\n")
print(results_markdown())
print(f"\nwhole notebook: {time.time() - T_START:.0f}s")

# %% [markdown]
# **Limitations (write these in your own words, with your numbers).**
#
# * **Simulated data.** C-MAPSS comes from a simulator with injected
#   degradation. Real engines have maintenance actions, sensor faults,
#   different usage and censoring. Nothing here transfers to a real fleet.
# * **The cap is a modelling choice.** It changes RMSE by several cycles;
#   results against papers that use another cap, or report capped truth, are
#   not comparable.
# * **100 test engines.** The intervals are several cycles wide, and the NASA
#   score is driven by a few engines; differences of one or two cycles between
#   models are not evidence.
# * **No censoring in training**: every training engine runs to failure. A real
#   system needs survival-style methods for engines removed before failure.
# * **FD001 only**: one operating condition and one fault mode.
#
# **What to do next (extensions).**
#
# 1. FD002/FD004: cluster the three settings into operating conditions and
#    standardise each sensor within its condition, fitted on training engines
#    only.
# 2. Prediction intervals: quantile loss, or conformal intervals calibrated on
#    engines, then a decision rule on the lower bound.
# 3. An asymmetric training loss (penalise over-prediction more) and its effect
#    on the early/late split and on RMSE.
# 4. Register each configuration (cap, window, model) as an immutable version
#    with a promote/rollback step, and add PSI-based drift checks per sensor —
#    the patterns of Chapter 15.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `expected 26 columns, got 28` | splitting on single spaces; trailing spaces at line ends | `sep=r"\s+"` |
# | `RuntimeError: The archive differs…` | NASA re-packaged the file, or a truncated download | delete the cached ZIP and retry; if it persists, inspect and update the hash deliberately |
# | Download fails | the host moved the file | open the landing page, find the current link, update `CMAPSS_URL` |
# | CNN RMSE much worse than trees | unscaled inputs, or target not divided by the cap | check `train_cnn` scaling and `cap` |
# | Test RMSE looks too good to be true | evaluated against capped truth, or on all rows | score the last cycle of each test engine against `RUL_FD001.txt` |
#
# **Interview practice:** answer the ten questions at the end of the
# [brief](README.md) aloud, in two minutes each, using the numbers your run
# printed above.
