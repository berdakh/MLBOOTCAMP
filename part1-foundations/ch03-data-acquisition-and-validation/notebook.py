# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch03-data-acquisition-and-validation/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 3 · Data acquisition & validation
#
# **Day 1 · ~45 minutes · CPU only**
#
# Most production model failures start upstream of the model: a file that
# downloaded half-way, a re-export with different rows, a code that a parser
# quietly turned into "missing". This notebook gets the project's raw extract,
# proves it is the right data, writes down what the data must look like, and
# then finds - with evidence - everything that is wrong with it.
#
# By the end you will be able to:
#
# 1. verify a download with a checksum and explain why downloads must be atomic;
# 2. spot a parser silently changing your data, and fix it at read time;
# 3. express expectations as a data contract and run it;
# 4. investigate each defect with evidence before deciding how to fix it;
# 5. recognise columns that describe the *export date* and would leak the future into features.
#
# **You will produce:** `artifacts/data_quality.json`, a data-quality summary (checksums, defects found, cleaning decisions, contract results).

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: only the raw extract is needed (it ships with the repo) ------
import pandas as pd

from churnlab import bootstrap
from churnlab import data as data_mod
from grading import ch03 as grade

RAW = bootstrap.ensure_raw()

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Raw extract** | the files exactly as exported from the source system, never edited by hand | four `.csv.gz` files in `data/raw/` |
# | **Checksum (SHA-256)** | a 64-character fingerprint of a file's bytes; one changed byte changes it completely | `verify_raw` compares each file with a pinned hash |
# | **Atomic download** | write to a temporary name and rename only when complete and verified | `fetch` writes `<name>.part`, then renames |
# | **Default NA strings** | strings pandas turns into missing values unless told not to | `"NA"`, `"N/A"`, `"null"` |
# | **Data contract** | a written, checkable statement of what each column must look like | `plan` must be one of `basic`, `pro`, `team` |
# | **Sentinel** | a fake value standing for "unknown" | `sessions = -1` |
# | **Unique key** | the columns that should identify exactly one row | (`customer_id`, `week_start`) in `usage_weekly` |
# | **Referential integrity** | every foreign key points at a row that exists | each `customer_id` in `tickets` is in `customers` |
# | **Freshness** | the time span a table actually covers | usage runs to the end of December 2024 |
# | **Export-time column** | a value describing the account on the day of the export, not on an earlier date | `status`, `last_active_week` |

# %% [markdown]
# ## 2.1 · Acquire, then verify
#
# The raw extract is four gzipped CSV files in `data/raw/`. "I downloaded it"
# is not evidence that you have the right bytes. A **SHA-256 checksum** is: any
# change to the file - one flipped bit, a truncated download, a re-export -
# changes the hash completely. The expected hashes are pinned in the code,
# next to the code that uses the data.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-validation-flow.svg" alt="Download to a .part file, verify the checksum, read with only empty fields missing, check the contract, investigate each defect, clean with stated decisions, re-validate" width="100%">
#
# *What to notice:* the first three steps prove you have the right bytes, read the right way; the last four turn defects into written decisions. This notebook walks the row left to right.
#
# > 💡 **Intuition.** A checksum is a tamper-evident seal on a jar: you do not
# > need to taste the contents to know nobody opened it. *Where it breaks
# > down:* the seal proves the bytes are the ones you pinned, not that those
# > bytes were correct in the first place; that is what the contract is for.

# %%
for path in sorted(RAW.glob("*.csv.gz")):
    print(f"{path.name:<20}{path.stat().st_size:>10,} bytes   sha256 {data_mod.sha256_file(path)[:16]}...")

print()
print("pinned:", {k: v[:16] + "..." for k, v in data_mod.RAW_SHA256.items()})
print("verify_raw:", data_mod.verify_raw(RAW))

# %% [markdown]
# **Expected:** four files from about 57 KB (`customers`) to about 1 MB
# (`usage_weekly`), and `verify_raw` reporting `True` for every table.
#
# `sha256_file` reads in 1 MB chunks, so it works on a 50 GB file without
# loading it into memory. Read it in
# [`src/churnlab/data.py`](../../src/churnlab/data.py) - it is eight lines.
#
# ### Atomic downloads
#
# In Colab, `bootstrap.ensure_raw()` downloads any missing file from
# `RAW_URL` with `churnlab.data.fetch`. Two details make that safe:
#
# 1. it writes to `<name>.part` and **renames only when complete and
#    verified**, so an interrupted download never looks like a finished file;
# 2. it checks the hash **before** the rename, and deletes the `.part` on a
#    mismatch.
#
# To see it work without touching the network, point `fetch` at the local file
# with a `file://` URL (in Colab the URL would be `RAW_URL`):

# %%
import shutil
import tempfile

print("in Colab it would fetch:", data_mod.RAW_URL.format(name="tickets"))

tmp = Path(tempfile.mkdtemp())
url = (RAW / "tickets.csv.gz").resolve().as_uri()
print("here we fetch          :", url)

dest = data_mod.fetch(url, tmp / "tickets.csv.gz", sha256=data_mod.RAW_SHA256["tickets"])
print("fetched and verified   :", dest.name, dest.stat().st_size, "bytes")

try:
    data_mod.fetch(url, tmp / "tickets_wrong.csv.gz", sha256="0" * 64)
except ValueError as exc:
    print("wrong checksum         :", exc)
print("files left behind      :", sorted(p.name for p in tmp.iterdir()))

# %% [markdown]
# **Expected:** the first fetch succeeds; the second raises `checksum mismatch`
# and leaves **no** `tickets_wrong.csv.gz` and no `.part` file behind - only the
# good download.
#
# Now corrupt a copy: flip a single byte in the middle of one file, and verify
# the directory again.

# %%
corrupt_dir = tmp / "raw_copy"
shutil.copytree(RAW, corrupt_dir)

victim = corrupt_dir / "invoices.csv.gz"
blob = bytearray(victim.read_bytes())
blob[len(blob) // 2] ^= 0xFF  # flip every bit of one byte
victim.write_bytes(bytes(blob))

print(data_mod.verify_raw(corrupt_dir))
shutil.rmtree(tmp)

# %% [markdown]
# **Expected:** `invoices: False`, the other three `True`. One byte out of
# ~350,000 changed and the check caught it. Without the check, the best case is
# a gzip error; the worst case is data that parses fine and is wrong.

# %% [markdown]
# ## 2.2 · A first look, and the first trap
#
# Read the customers table the way almost everyone does, then look at
# `region`. Include missing values in the count (`dropna=False`) - without it,
# pandas hides them.

# %%
naive = pd.read_csv(RAW / "customers.csv.gz")
print(naive["region"].value_counts(dropna=False))
print(f"\nregion missing: {naive['region'].isna().mean():.1%}")

# %% [markdown]
# **Expected:** about 2,092 missing regions (≈42%) and **no** `NA` region at
# all, although the company sells in North America.
#
# Before reading on: is 42% missing plausible for a field the sign-up form
# requires? When a number surprises you, look at the raw bytes:

# %%
import gzip

with gzip.open(RAW / "customers.csv.gz", "rt") as fh:
    for _, line in zip(range(4), fh):
        print(line.rstrip())

# %% [markdown]
# The file says `NA` - the region code for North America. `pd.read_csv` treats
# the strings `"NA"`, `"N/A"`, `"null"`, `"nan"` and a dozen others as missing
# **by default**. The fix is at read time: only empty fields are missing.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l1-na-trap.svg" alt="The same raw line read with pandas defaults turns the region code NA into NaN; read with keep_default_na=False it stays NA and only 87 regions are truly missing" width="100%">
#
# *What to notice:* nothing fails in the top row. The loader quietly produces a different dataset, which is why the fix belongs at read time.

# %%
fixed = pd.read_csv(RAW / "customers.csv.gz", keep_default_na=False, na_values=[""])
print(fixed["region"].value_counts(dropna=False))

# %% [markdown]
# **Expected:** `NA` ≈ 2,005 customers and only **87** truly missing regions.
#
# `churnlab.data.load_raw` reads every table this way and parses the date
# columns. From here on we use it.
#
# > 🧠 **Research bridge.** This is the tabular cousin of a mis-specified
# > channel montage or a wrong sampling-rate header in an EEG file: the loader
# > does not fail, it silently gives you different data. The defence is the
# > same - check summary statistics against what you know must be true before
# > you analyse anything.

# %%
raw = data_mod.load_raw(RAW)
for name, df in raw.items():
    print(f"{name:<14}{df.shape[0]:>8,} rows  {df.shape[1]:>2} cols  {list(df.columns)}")

# %% [markdown]
# ## 2.3 · Data contracts
#
# A **data contract** writes down what each column must look like - type,
# nullability, allowed values, ranges, uniqueness - so that checking it is a
# function call, not a feeling. `churnlab.validate` is a deliberately small
# version of what pandera or Great Expectations do. The customers contract:

# %%
from churnlab import validate as V

for col in V.CONTRACTS["customers"]:
    print(col)

# %% [markdown]
# Run every contract against the raw tables. `validate` never raises; it
# returns a report you can print, assert on in a test, or save.

# %%
reports = V.validate_all(raw)
for report in reports.values():
    print(report)

# %% [markdown]
# **Expected:** `customers` with 2 issues (55 `plan` values outside the
# allowed set, e.g. `'Basic'`; 87 null `region`), `usage_weekly` with 2 issues
# (199 `sessions` below 0; 630 duplicate `customer_id+week_start` keys),
# `invoices` and `tickets` PASS.
#
# A contract tells you **that** something is wrong. It does not tell you
# **what to do about it**. That needs an investigation per defect.
#
# > 💡 **Intuition.** A data contract is a smoke alarm for a table: cheap,
# > always on, and it tells you which room, not what is burning. *Where it
# > breaks down:* it only checks what you thought to write down. Export-time
# > columns (section 2.6) pass every contract and are still wrong to use.

# %% [markdown]
# ## 2.4 · Investigate each defect, with evidence
#
# ### Duplicated usage rows
#
# The unique key of `usage_weekly` is (`customer_id`, `week_start`): one row
# per account per week. To decide what to do with duplicates you need to see
# **all** copies of each duplicated key, side by side.

# %% [markdown]
# > **Exercise 1.** Write `find_duplicate_keys(df, key)` that returns **every**
# > row whose key value appears more than once (both copies, not just the
# > second), sorted by the key. `key` is a column name or a list of names. Do
# > not modify `df`.

# %%
def find_duplicate_keys(df: pd.DataFrame, key) -> pd.DataFrame:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(find_duplicate_keys)

# %% [markdown]
# Now the evidence: how many duplicated keys are there, and are the copies
# identical? (This cell uses pandas directly, so it runs whether or not you
# have finished the exercise.)

# %%
usage = raw["usage_weekly"]
key = ["customer_id", "week_start"]
dups = usage[usage.duplicated(subset=key, keep=False)].sort_values(key)
print("rows involved in duplicated keys:", len(dups))
print("extra copies (rows to drop)     :", int(usage.duplicated(subset=key).sum()))
print("exact duplicate rows            :", int(usage.duplicated().sum()))

differs = dups.groupby(key).nunique().gt(1).any(axis=1)
print("keys whose copies DISAGREE      :", int(differs.sum()))
dups.set_index(key).loc[differs[differs].index]

# %% [markdown]
# **Expected:** 1,260 rows involved, 630 extra copies, 629 of them exact
# duplicates, and **one** key whose two copies disagree: one says `sessions =
# -1`, the other a real count.
#
# Exact copies are a pipeline artefact (a retried batch, a double export):
# dropping them loses nothing. The one disagreement is worth noticing - it
# ties two defects together - and is a reminder that "drop duplicates" is a
# decision about *which* copy wins.
#
# ### The `-1` sentinel
#
# `sessions` is a count, so `-1` cannot be a real value. Is it "no activity"
# or "unknown"? Look at what else was recorded in the same rows:

# %%
print("sessions value counts (lowest):")
print(usage["sessions"].value_counts().sort_index().head(4))
print()
print("rows with sessions == -1, other columns:")
print(usage.loc[usage["sessions"] == -1, ["active_users", "features_used"]].describe().round(1))

# %% [markdown]
# **Expected:** 199 rows with `-1`, **no** rows with `0` (inactive weeks are
# simply not logged), and the `-1` rows have a median of about 2 active users
# and 2 features used. Weeks with active users and zero sessions are
# impossible, so `-1` means **"the session count is unknown"**, not "none".
#
# > 💡 **Intuition.** A sentinel is a blank on a form that someone filled with
# > "-1" because the field would not accept empty. Reading it as 0 is like
# > reading "did not answer" as "answered no". *Where it breaks down:* some
# > systems really do use 0 for "unknown"; only evidence from other columns
# > (here, active users in the same row) tells you which it is.

# %% [markdown]
# ### Plan casing
#
# 55 customers have `Basic` / `Pro` / `Team`. A different product, or the same
# plan written differently? Evidence: what do they pay?

# %%
inv = raw["invoices"].merge(raw["customers"][["customer_id", "plan"]], on="customer_id")
print(raw["customers"]["plan"].value_counts())
print()
print(inv.groupby("plan")["amount_usd"].median().rename("median invoice (USD)"))

# %% [markdown]
# **Expected:** `Basic` and `basic` both have a median invoice of \$29, `Pro`
# and `pro` \$79. Same price list, so it is a formatting bug: lower-case it.
# (`team` medians differ because team invoices scale with seats.)
#
# ### Truly missing region
#
# After the read-time fix, 87 regions are genuinely empty in the file:

# %%
missing = raw["customers"][raw["customers"]["region"].isna()]
print(len(missing), "customers with an empty region field")
missing[["customer_id", "signup_date", "plan", "region", "acquisition_channel"]].head(3)

# %% [markdown]
# **Expected:** 87 rows. Dropping them would bias the data toward complete
# records (and you would have to drop them at prediction time too). Keeping
# them as an explicit `"unknown"` category lets the model learn whether
# "missing" itself carries signal.
#
# ### Referential integrity and freshness
#
# Two checks every new table deserves: does every foreign key point at
# something that exists, and what time span does each table actually cover?

# %% [markdown]
# > **Exercise 2.** Write `orphan_ids(tables)` that returns a dict mapping each
# > of `usage_weekly`, `invoices` and `tickets` to the **set** of `customer_id`
# > values that do not appear in `customers` (an empty set if there are none).

# %%
def orphan_ids(tables: dict) -> dict:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(orphan_ids)

try:
    print({name: len(ids) for name, ids in orphan_ids(raw).items()})
except NotImplementedError:
    print("(finish Exercise 2 to run it on the real tables)")

# %% [markdown]
# **Expected (once solved):** `{'usage_weekly': 0, 'invoices': 0, 'tickets': 0}`
# on the real data. An empty result is a finding too - write it down.
#
# > **Exercise 3.** Write `freshness(tables)` that returns a DataFrame indexed
# > by table name with columns `min` and `max`: the earliest and latest value of
# > that table's timestamp column given in `TIMESTAMP_COLUMNS` (below). Ignore
# > missing timestamps.

# %%
TIMESTAMP_COLUMNS = {
    "customers": "signup_date",
    "usage_weekly": "week_start",
    "invoices": "invoice_date",
    "tickets": "created_at",
}


def freshness(tables: dict) -> pd.DataFrame:
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(freshness)

try:
    print(freshness(raw))
except NotImplementedError:
    print("(finish Exercise 3 to run it on the real tables)")

# %% [markdown]
# **Expected (once solved):** sign-ups from 2022-05-16 to 2024-10-31, and
# usage, invoices and tickets all running from around 1 January 2023 (the first
# ticket is from 2022-12-31) to the end of December 2024. That end date - the **export date** - matters in
# section 2.6.

# %% [markdown]
# ## 2.5 · Clean, with stated decisions, then re-validate
#
# `churnlab.validate.clean` applies one reviewable decision per defect. Read
# its docstring: every decision is written where the code is.

# %%
import inspect

print(inspect.getdoc(V.clean))

# %%
tables = V.clean(raw)
clean_reports = V.validate_clean(tables)
for report in clean_reports.values():
    print(report)

print("\nsessions now NaN (was -1):", int(tables["usage_weekly"]["sessions"].isna().sum()))
print("region 'unknown'         :", int((tables["customers"]["region"] == "unknown").sum()))

# %% [markdown]
# **Expected:** all four tables PASS; `usage_weekly` has 209,995 rows (630
# fewer); 199 `sessions` are now NaN; 87 regions are `"unknown"`.
#
# Notice something a reviewer should catch: for the one disagreeing duplicate
# (`C03550`, 2024-12-02), "keep the first copy" kept the `-1` copy and threw
# away the real count of 3. One row does not move any result here, but a
# better rule would prefer the non-sentinel copy. Passing contracts do not
# mean every decision was the best one.
#
# `validate_clean` uses the same contracts, relaxed **only** where cleaning
# chose to (sessions may be NaN; region may be `"unknown"`). A relaxation is a
# decision too, so it lives in code where a reviewer can see it.
#
# Save the evidence next to the data, so whoever trains a model on these tables
# can see what was done to them:

# %%
import json

quality = {
    "raw_sha256": {name: data_mod.sha256_file(RAW / f"{name}.csv.gz") for name in data_mod.TABLES},
    "checksums_ok": data_mod.verify_raw(RAW),
    "raw_rows": {name: len(df) for name, df in raw.items()},
    "clean_rows": {name: len(df) for name, df in tables.items()},
    "raw_issues": {name: [{"column": i.column, "rule": i.rule, "rows": i.n_rows} for i in r.issues]
                   for name, r in reports.items()},
    "decisions": {
        "region 'NA'": "read with keep_default_na=False, na_values=[''] (North America, not missing)",
        "plan casing": "lower-case (same price list => formatting bug)",
        "missing region": "'unknown' category (keeps 87 customers, lets the model use missingness)",
        "duplicate usage keys": "keep first copy per (customer_id, week_start)",
        "sessions == -1": "NaN (unknown), imputed inside the model pipeline",
        "leaky columns": "kept for labels only; excluded from features (see 2.6)",
    },
    "clean_contracts_pass": {name: r.ok for name, r in clean_reports.items()},
}
(ART / "data_quality.json").write_text(json.dumps(quality, indent=2, default=str))
print(json.dumps(quality, indent=2, default=str)[:800])

# %% [markdown]
# > **Exercise 4 (written).** Write a short decision memo (4-6 sentences) for a
# > colleague: what should happen to `sessions == -1`, and why **not** replace
# > it with 0? Use the evidence from section 2.4, and say what a model would learn if you
# > used 0. Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 2.6 · Columns that describe the export date
#
# Three columns in `customers` look like gifts: `status`, `churn_date` and
# `last_active_week`. They describe each account **as of the export** (end of
# December 2024), not as of any earlier date. Pick one monthly snapshot,
# T = 2024-01-01, and look at the accounts that were still active then:

# %%
c = tables["customers"]
T = pd.Timestamp("2024-01-01")
active_at_T = c[(c["signup_date"] < T) & (c["churn_date"].isna() | (c["churn_date"] >= T))]
will_churn = (active_at_T["churn_date"] >= T) & (active_at_T["churn_date"] < T + pd.Timedelta(days=30))

print("accounts active at T                    :", len(active_at_T))
print("  ... whose status says 'cancelled'      :", int((active_at_T["status"] == "cancelled").sum()))
print("  ... whose last_active_week is AFTER T  :", f"{(active_at_T['last_active_week'] >= T).mean():.1%}")
print()
print(pd.crosstab(will_churn.rename("churns in [T, T+30d)"), active_at_T["status"]))

# %% [markdown]
# **Expected:** about 2,173 accounts active at T; about half already "cancelled"
# according to `status`; over 99% with a `last_active_week` after T; and every
# one of the ~121 accounts that churn in the next 30 days has `status =
# cancelled`.
#
# At T, nobody could have known any of that. A model given `status` would look
# brilliant in validation and be useless in production, where every account it
# scores is, by definition, still active. The rule this course follows:
#
# * `churn_date` is used **only to build the label** (notebook 03);
# * `status` and `last_active_week` are **never** features;
# * features for snapshot T use only rows timestamped **before** T.
#
# Notebook 04 measures how badly these columns inflate validation scores.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `verify_raw` shows `False` for a table | a partial or corrupted file, or someone edited it | delete that file and re-run `bootstrap.ensure_raw()` (in Colab it re-downloads and re-verifies) |
# | `ValueError: checksum mismatch` from `fetch` | the remote file changed, or the download was corrupted | do not "fix" the pinned hash to make it pass; find out which file is right first |
# | `region` is ~42% missing | read with pandas defaults, `"NA"` parsed as missing | `pd.read_csv(..., keep_default_na=False, na_values=[""])`, or use `load_raw` |
# | `TypeError: '<' not supported between 'str' and 'Timestamp'` | date columns read as strings | use `load_raw`, which parses them, or `pd.to_datetime` |
# | `value_counts()` shows no missing values but the contract says there are | `value_counts` drops NaN by default | `value_counts(dropna=False)` |
# | `validate_clean` fails after you changed `clean` | the contract and the cleaning decision disagree | change both together, in the same commit, and say why |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. How do you prove a downloaded file is the one you expect, and why download to a `.part` file first?
# 2. A column is 42% missing in a field the form requires. Walk through how you would investigate.
# 3. What is a data contract, and where in a pipeline would you run it?
# 4. Why is `sessions = -1` not the same as `sessions = 0`, and what would a model learn if you treated it as 0?
# 5. Name a column that is safe to use as a label input but not as a feature, and explain why.
#
# ➡️ **Next:** [`03_sql_for_ml.ipynb`](../ch04-sql-for-ml/notebook.ipynb) — turn the clean tables into point-in-time features and labels with SQL.
