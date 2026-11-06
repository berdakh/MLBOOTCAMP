"""The bootcamp dataset: a fictional subscription business called *TaskFlow*.

Why synthetic?
--------------
The bootcamp needs one dataset that is (1) available in a fresh Colab runtime
with no account, key or licence question, (2) small enough to train on a CPU in
seconds, (3) relational, so SQL is genuinely needed, and (4) honest about its
own ground truth, so that we can *plant* the problems real data has - leakage,
duplicates, sentinels, inconsistent categories - and check that you find them.

No public dataset satisfies all four. So the raw extract in ``data/raw/`` was
produced by :func:`generate` below, committed once, and is treated from then on
as if it came from a production database. The generator stays in the repo for
two reasons: provenance (you can read exactly how the data was made) and
simulation (Day 4 uses it to produce a *new* quarter of data with a shift in it).

Nothing learned on this data says anything about real customers. It teaches
engineering, not business facts.

The tables
----------
``customers``      one row per account, as exported *today* (note: today!)
``usage_weekly``   one row per account per completed week of activity
``invoices``       one row per monthly invoice, paid or failed
``tickets``        one row per support ticket

The prediction task
-------------------
At a monthly *snapshot date* T, for every account still active at T, predict
whether it cancels in the 30 days after T. Features may use only information
available strictly before T. See ``churnlab.features``.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = ("customers", "usage_weekly", "invoices", "tickets")

PLANS = {"basic": 29.0, "pro": 79.0, "team": 199.0}  # list price per seat-block
REGIONS = ["NA", "EU", "APAC", "LATAM"]
CHANNELS = ["organic", "paid_search", "referral", "partner"]
SIZES = ["1-10", "11-50", "51-200", "200+"]
TICKET_CATEGORIES = ["how_to", "bug", "billing", "feature_request", "cancellation"]


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs for :func:`generate`. The committed extract uses the defaults."""

    n_customers: int = 5000
    start: str = "2023-01-02"  # a Monday: weeks start on Mondays
    n_weeks: int = 104
    seed: int = 407
    # Scenario knobs, used on Day 4 to simulate a changed world.
    failed_payment_multiplier: float = 1.0
    engagement_shift: float = 0.0
    signup_window_days: int = 900
    inject_quality_issues: bool = True


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(config: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Simulate the four raw tables.

    The simulation is a weekly loop. Each account has a latent *engagement*
    level that drifts (AR(1)), and some accounts enter a disengaging state. Engagement drives usage,
    low engagement drives support tickets, and the weekly cancellation hazard
    rises with low or falling engagement, recent failed payments and recent
    tickets. Accounts that cancel stop producing data.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    start = pd.Timestamp(cfg.start)
    weeks = pd.date_range(start, periods=cfg.n_weeks, freq="7D")
    end = weeks[-1] + pd.Timedelta(days=7)

    # ---- static account attributes -------------------------------------------
    plan = rng.choice(list(PLANS), size=n, p=[0.5, 0.35, 0.15])
    seat_hi = np.select([plan == "basic", plan == "pro"], [3, 10], 40)
    seats = rng.integers(1, seat_hi + 1)
    region = rng.choice(REGIONS, size=n, p=[0.42, 0.30, 0.18, 0.10])
    channel = rng.choice(CHANNELS, size=n, p=[0.40, 0.30, 0.18, 0.12])
    size_idx = np.clip((np.log2(seats + 1) + rng.normal(0, 0.8, n)).round().astype(int) - 1, 0, 3)
    company_size = np.array(SIZES)[size_idx]
    # Signups spread over ~2.5 years ending shortly before the window ends,
    # so some accounts pre-date the data and some arrive during it.
    signup_offset = rng.integers(0, cfg.signup_window_days, size=n)
    signup = end - pd.Timedelta(days=60) - pd.to_timedelta(signup_offset, unit="D")
    signup = signup.normalize()

    base = (
        0.9
        + np.select([plan == "basic", plan == "pro"], [0.0, 0.35], 0.6)
        + np.select([channel == "referral", channel == "partner", channel == "paid_search"],
                    [0.25, 0.2, -0.25], 0.0)
        + rng.normal(0, 0.45, n)
        + cfg.engagement_shift
    )
    fail_rate = (
        np.select([region == "LATAM", region == "APAC"], [0.07, 0.045], 0.03)
        * np.where(plan == "basic", 1.4, 1.0)
        * cfg.failed_payment_multiplier
    )

    engagement = base + rng.normal(0, 0.3, n)
    prev_engagement = engagement.copy()
    disengaging = np.zeros(n, dtype=bool)
    churn_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    alive = np.ones(n, dtype=bool)
    recent_failed = np.zeros(n)  # decays weekly
    recent_tickets = np.zeros(n)

    usage_rows, invoice_rows, ticket_rows = [], [], []
    signup_np = signup.values

    for week in weeks:
        wk = np.datetime64(week)
        started = signup_np <= wk
        active = alive & started
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue

        # Engagement dynamics: a mean-reverting walk, plus a "disengaging" state.
        # Accounts enter it at random (more often after failed payments), and
        # while in it their engagement slides week after week - the observable
        # precursor that makes cancellation partly predictable. Some recover.
        enter = rng.random(idx.size) < (0.012 + 0.03 * (recent_failed[idx] > 0))
        disengaging[idx] |= enter
        recover = rng.random(idx.size) < 0.08
        disengaging[idx] &= ~recover
        drift = np.where(disengaging[idx], -0.22, 0.0)
        prev_engagement[idx] = engagement[idx]
        engagement[idx] = (
            0.88 * engagement[idx] + 0.12 * base[idx] + rng.normal(0, 0.15, idx.size) + drift
        )
        e = engagement[idx]

        # Usage this week.
        sessions = rng.poisson(np.exp(np.clip(e, -3, 3.5)) * (1 + 0.6 * np.log1p(seats[idx])))
        active_users = np.minimum(seats[idx], rng.binomial(seats[idx], _sigmoid(e - 0.2)))
        features_used = np.minimum(12, rng.poisson(np.clip(1.5 + 1.2 * e, 0.1, None)))
        usage_rows.append(pd.DataFrame({
            "customer_id": idx, "week_start": week, "sessions": sessions,
            "active_users": active_users, "features_used": features_used,
        }))

        # Monthly invoices on the signup day-of-month (approximated as every 4th week
        # since signup), failure probability rises with low engagement too.
        weeks_since = ((wk - signup_np[idx]) / np.timedelta64(7, "D")).astype(int)
        billed = (weeks_since % 4) == 0
        if billed.any():
            b_idx = idx[billed]
            p_fail = np.clip(fail_rate[b_idx] * np.exp(-0.3 * (engagement[b_idx] - 1)), 0, 0.5)
            failed = rng.random(b_idx.size) < p_fail
            recent_failed[b_idx] += failed
            amount = np.array([PLANS[p] for p in plan[b_idx]]) * np.ceil(seats[b_idx] / 5)
            invoice_rows.append(pd.DataFrame({
                "customer_id": b_idx,
                "invoice_date": week + pd.to_timedelta(rng.integers(0, 7, b_idx.size), unit="D"),
                "amount_usd": amount.round(2),
                "status": np.where(failed, "failed", "paid"),
            }))

        # Support tickets: more likely when engagement is low or a payment failed.
        lam = 0.04 + 0.08 * _sigmoid(-(e - 0.5)) + 0.15 * (recent_failed[idx] > 0)
        n_tickets = rng.poisson(lam)
        recent_tickets[idx] = 0.8 * recent_tickets[idx] + n_tickets

        # Weekly cancellation hazard.
        decline = np.clip(prev_engagement[idx] - e, 0, None)
        tenure_w = weeks_since
        logit = (
            -5.6
            - 1.2 * (e - 1.0)
            + 1.5 * decline
            + 1.6 * disengaging[idx]
            + 1.3 * np.minimum(recent_failed[idx], 2)
            + 0.45 * np.minimum(recent_tickets[idx], 3)
            + np.where(plan[idx] == "basic", 0.45, 0.0)
            - 0.25 * np.log1p(np.maximum(tenure_w, 0) / 26)
        )
        churns = rng.random(idx.size) < _sigmoid(logit)
        # A "cancellation" ticket precedes some cancellations by a few days.
        will_churn = idx[churns]
        churn_day = week + pd.to_timedelta(rng.integers(0, 7, will_churn.size), unit="D")
        churn_date[will_churn] = churn_day.values
        alive[will_churn] = False

        t_idx = np.repeat(idx, n_tickets)
        if t_idx.size:
            cats = rng.choice(TICKET_CATEGORIES[:4], size=t_idx.size, p=[0.4, 0.3, 0.2, 0.1])
            cats = np.where((recent_failed[t_idx] > 0) & (rng.random(t_idx.size) < 0.5),
                            "billing", cats)
            ticket_rows.append(pd.DataFrame({
                "customer_id": t_idx,
                "created_at": week + pd.to_timedelta(rng.integers(0, 7 * 24 * 60, t_idx.size), unit="min"),
                "category": cats,
                "resolution_hours": rng.gamma(2.0, 9.0, t_idx.size).round(1),
            }))
        cancel_ticket = will_churn[rng.random(will_churn.size) < 0.35]
        if cancel_ticket.size:
            ticket_rows.append(pd.DataFrame({
                "customer_id": cancel_ticket,
                "created_at": pd.Series(churn_date[cancel_ticket]) - pd.to_timedelta(
                    rng.integers(60, 3 * 24 * 60, cancel_ticket.size), unit="min"),
                "category": "cancellation",
                "resolution_hours": rng.gamma(1.5, 4.0, cancel_ticket.size).round(1),
            }))

        recent_failed[idx] *= 0.85

    # ---- assemble -------------------------------------------------------------
    ids = np.array([f"C{i:05d}" for i in range(n)])
    usage = pd.concat(usage_rows, ignore_index=True)
    usage = usage[usage["sessions"] > 0]
    last_seen = usage.groupby("customer_id")["week_start"].max()

    customers = pd.DataFrame({
        "customer_id": ids,
        "signup_date": signup,
        "plan": plan,
        "seats": seats,
        "region": region,
        "acquisition_channel": channel,
        "company_size": company_size,
        # The two columns below are *as of the export date*. They are the
        # classic leakage trap: they describe the future relative to any
        # snapshot taken before the export.
        "status": np.where(alive, "active", "cancelled"),
        "churn_date": pd.to_datetime(churn_date),
        "last_active_week": pd.to_datetime(
            pd.Series(np.arange(n)).map(last_seen).values
        ),
    })
    # Accounts that signed up after the window closed never appear.
    customers = customers[customers["signup_date"] < end].reset_index(drop=True)

    usage["customer_id"] = ids[usage["customer_id"].to_numpy()]
    invoices = pd.concat(invoice_rows, ignore_index=True)
    invoices["customer_id"] = ids[invoices["customer_id"].to_numpy()]
    invoices = invoices[invoices["invoice_date"] < end]
    tickets = pd.concat(ticket_rows, ignore_index=True)
    tickets["customer_id"] = ids[tickets["customer_id"].to_numpy()]
    tickets = tickets[tickets["created_at"] < end].sort_values("created_at")
    tickets.insert(0, "ticket_id", [f"T{i:06d}" for i in range(len(tickets))])
    invoices = invoices.sort_values(["invoice_date", "customer_id"])
    invoices.insert(0, "invoice_id", [f"I{i:06d}" for i in range(len(invoices))])

    tables = {
        "customers": customers,
        "usage_weekly": usage.sort_values(["week_start", "customer_id"]).reset_index(drop=True),
        "invoices": invoices.reset_index(drop=True),
        "tickets": tickets.reset_index(drop=True),
    }
    if cfg.inject_quality_issues:
        tables = _inject_quality_issues(tables, np.random.default_rng(cfg.seed + 1))
    return tables


def _inject_quality_issues(tables: dict[str, pd.DataFrame], rng) -> dict[str, pd.DataFrame]:
    """Plant the defects a real export has, at known rates, so Day 1 can find them."""
    out = {k: v.copy() for k, v in tables.items()}

    c = out["customers"]
    miss = rng.random(len(c)) < 0.02
    c["region"] = c["region"].astype(object)
    c.loc[miss, "region"] = None
    shout = rng.random(len(c)) < 0.01
    c.loc[shout, "plan"] = c.loc[shout, "plan"].str.capitalize()  # "Pro" vs "pro"

    u = out["usage_weekly"]
    dup = u.sample(frac=0.003, random_state=int(rng.integers(1_000_000)))
    sentinel = rng.random(len(u)) < 0.001
    u.loc[sentinel, "sessions"] = -1  # the logging pipeline's "unknown" marker
    out["usage_weekly"] = (
        pd.concat([u, dup]).sort_values(["week_start", "customer_id"]).reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Reading and writing the committed extract
# ---------------------------------------------------------------------------

#: SHA-256 of each committed ``data/raw/*.csv.gz`` file. ``verify_raw`` checks
#: these, so a corrupted download or an accidental edit is caught immediately.
#: Regenerate with ``python tools/make_dataset.py`` (and update these values).
RAW_SHA256: dict[str, str] = {
    "customers": "40319090a261e20191d07566fed2cc7c4458cc8d68c385ce5705b6918051263d",
    "usage_weekly": "5821a809d3ecfb359494a09e73858d37f69d88ad5c23262bda89700e8e5251b8",
    "invoices": "6e39705e339ba5d3a4689d4a885ee7a458c5d157c08bc71830d755eaa76dcd8b",
    "tickets": "a11762f8211ec4a52ea66743de14fc581ebceb12ddc03bcd13f7570fab149528",
}

DATE_COLUMNS = {
    "customers": ["signup_date", "churn_date", "last_active_week"],
    "usage_weekly": ["week_start"],
    "invoices": ["invoice_date"],
    "tickets": ["created_at"],
}

RAW_URL = "https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/data/raw/{name}.csv.gz"


def write_raw(tables: dict[str, pd.DataFrame], directory: Path) -> dict[str, str]:
    """Write tables as deterministic gzip CSVs; return their SHA-256 digests."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in TABLES:
        buf = io.StringIO()
        tables[name].to_csv(buf, index=False, date_format="%Y-%m-%d %H:%M:%S")
        raw = buf.getvalue().encode("utf-8")
        path = directory / f"{name}.csv.gz"
        # mtime=0 makes the gzip bytes reproducible, so the hash is stable.
        with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
            gz.write(raw)
        digests[name] = sha256_file(path)
    return digests


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large files do not load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_raw(directory: Path, expected: dict[str, str] | None = None) -> dict[str, bool]:
    """Return {table: matches_expected_hash}. Missing files count as False."""
    expected = expected if expected is not None else RAW_SHA256
    result = {}
    for name in TABLES:
        path = Path(directory) / f"{name}.csv.gz"
        result[name] = path.exists() and (name not in expected or sha256_file(path) == expected[name])
    return result


def fetch(url: str, destination: Path, sha256: str | None = None, timeout: float = 60) -> Path:
    """Download ``url`` to ``destination`` atomically, optionally verifying a hash.

    Downloads go to a ``.part`` file first and are renamed only once complete
    and verified, so an interrupted download never looks like a finished one.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, open(tmp, "wb") as fh:
        shutil.copyfileobj(response, fh)
    if sha256 is not None:
        got = sha256_file(tmp)
        if got != sha256:
            tmp.unlink()
            raise ValueError(f"checksum mismatch for {url}: expected {sha256[:12]}..., got {got[:12]}...")
    tmp.replace(destination)
    return destination


def load_raw(directory: Path) -> dict[str, pd.DataFrame]:
    """Read the four raw tables with dates parsed. No cleaning is done here."""
    directory = Path(directory)
    tables = {}
    for name in TABLES:
        # keep_default_na=False: otherwise pandas reads the region code "NA"
        # (North America) as a missing value. Only empty fields are missing.
        df = pd.read_csv(directory / f"{name}.csv.gz", keep_default_na=False, na_values=[""])
        for col in DATE_COLUMNS[name]:
            df[col] = pd.to_datetime(df[col])
        tables[name] = df
    return tables
