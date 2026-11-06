"""Data contracts: write down what the data must look like, then check it.

A *contract* is a list of column rules. Checking it produces a report you can
print, assert on in a test, or save next to a model as evidence of what it was
trained on. This is a deliberately small, readable version of what libraries
such as pandera or Great Expectations do - so you can see there is no magic.

Usage::

    report = validate(df, CONTRACTS["usage_weekly"])
    print(report)
    report.raise_if_failed()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Column:
    """Rules for one column. Every rule is optional."""

    name: str
    dtype: str | None = None  # "int", "float", "str", "datetime"
    nullable: bool = True
    allowed: tuple | None = None
    min: float | None = None
    max: float | None = None
    unique: bool = False


@dataclass
class Issue:
    column: str
    rule: str
    n_rows: int
    example: object = None

    def __str__(self) -> str:
        ex = "" if self.example is None else f"  e.g. {self.example!r}"
        return f"{self.column:<20} {self.rule:<28} {self.n_rows:>7} rows{ex}"


@dataclass
class ValidationReport:
    table: str
    n_rows: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def __str__(self) -> str:
        head = f"[{self.table}] {self.n_rows} rows - " + ("PASS" if self.ok else f"{len(self.issues)} issue(s)")
        return "\n".join([head, *("  " + str(i) for i in self.issues)])

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(str(self))


_DTYPE_CHECKS = {
    "int": pd.api.types.is_integer_dtype,
    "float": pd.api.types.is_numeric_dtype,
    "datetime": pd.api.types.is_datetime64_any_dtype,
    "str": lambda s: pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s),
}


def validate(df: pd.DataFrame, contract: list[Column], table: str = "table",
             unique_key: list[str] | None = None) -> ValidationReport:
    """Check ``df`` against ``contract`` and return a report (never raises)."""
    report = ValidationReport(table, len(df))
    for col in contract:
        if col.name not in df.columns:
            report.issues.append(Issue(col.name, "missing column", len(df)))
            continue
        s = df[col.name]
        if col.dtype and not _DTYPE_CHECKS[col.dtype](s):
            report.issues.append(Issue(col.name, f"dtype is {s.dtype}, want {col.dtype}", len(df)))
            continue
        nulls = s.isna()
        if not col.nullable and nulls.any():
            report.issues.append(Issue(col.name, "null values", int(nulls.sum())))
        present = s[~nulls]
        if col.allowed is not None:
            bad = present[~present.isin(col.allowed)]
            if len(bad):
                report.issues.append(Issue(col.name, "value not in allowed set", len(bad), bad.iloc[0]))
        if col.min is not None:
            bad = present[present < col.min]
            if len(bad):
                report.issues.append(Issue(col.name, f"below min {col.min}", len(bad), bad.iloc[0]))
        if col.max is not None:
            bad = present[present > col.max]
            if len(bad):
                report.issues.append(Issue(col.name, f"above max {col.max}", len(bad), bad.iloc[0]))
        if col.unique and present.duplicated().any():
            report.issues.append(Issue(col.name, "duplicate values", int(present.duplicated().sum())))
    if unique_key:
        dup = df.duplicated(subset=unique_key)
        if dup.any():
            report.issues.append(Issue("+".join(unique_key), "duplicate key", int(dup.sum())))
    return report


PLANS = ("basic", "pro", "team")
REGIONS = ("NA", "EU", "APAC", "LATAM")

CONTRACTS: dict[str, list[Column]] = {
    "customers": [
        Column("customer_id", "str", nullable=False, unique=True),
        Column("signup_date", "datetime", nullable=False),
        Column("plan", "str", nullable=False, allowed=PLANS),
        Column("seats", "int", nullable=False, min=1, max=500),
        Column("region", "str", nullable=False, allowed=REGIONS),
        Column("acquisition_channel", "str", nullable=False,
               allowed=("organic", "paid_search", "referral", "partner")),
        Column("company_size", "str", nullable=False, allowed=("1-10", "11-50", "51-200", "200+")),
    ],
    "usage_weekly": [
        Column("customer_id", "str", nullable=False),
        Column("week_start", "datetime", nullable=False),
        Column("sessions", "int", nullable=False, min=0),
        Column("active_users", "int", nullable=False, min=0),
        Column("features_used", "int", nullable=False, min=0, max=12),
    ],
    "invoices": [
        Column("invoice_id", "str", nullable=False, unique=True),
        Column("customer_id", "str", nullable=False),
        Column("invoice_date", "datetime", nullable=False),
        Column("amount_usd", "float", nullable=False, min=0),
        Column("status", "str", nullable=False, allowed=("paid", "failed")),
    ],
    "tickets": [
        Column("ticket_id", "str", nullable=False, unique=True),
        Column("customer_id", "str", nullable=False),
        Column("created_at", "datetime", nullable=False),
        Column("category", "str", nullable=False,
               allowed=("how_to", "bug", "billing", "feature_request", "cancellation")),
        Column("resolution_hours", "float", nullable=True, min=0),
    ],
}

UNIQUE_KEYS = {"usage_weekly": ["customer_id", "week_start"]}


def validate_all(tables: dict[str, pd.DataFrame]) -> dict[str, ValidationReport]:
    return {
        name: validate(df, CONTRACTS[name], name, UNIQUE_KEYS.get(name))
        for name, df in tables.items() if name in CONTRACTS
    }


def clean(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fix the known defects, each with a stated, reviewable decision.

    * plan casing      -> lower-case (a formatting bug, not a different plan)
    * missing region   -> "unknown"  (kept as its own category; dropping the
                          customers would bias the data toward complete records)
    * duplicate usage  -> keep the first row per (customer, week)
    * sessions == -1   -> the logging sentinel for "unknown"; set to NaN and let
                          the feature pipeline impute, rather than treating -1
                          as a real count
    The leakage columns (status, churn_date, last_active_week) are *kept* here:
    the label is built from churn_date. They are excluded from features later.
    """
    out = {k: v.copy() for k, v in tables.items()}
    c = out["customers"]
    c["plan"] = c["plan"].str.lower()
    c["region"] = c["region"].fillna("unknown")
    u = out["usage_weekly"].drop_duplicates(subset=["customer_id", "week_start"], keep="first")
    u = u.astype({"sessions": "float64"})
    u.loc[u["sessions"] < 0, "sessions"] = float("nan")
    out["usage_weekly"] = u.reset_index(drop=True)
    return out


CLEAN_CONTRACT_OVERRIDES = {
    "customers": {"region": Column("region", "str", nullable=False, allowed=(*REGIONS, "unknown"))},
    "usage_weekly": {"sessions": Column("sessions", "float", nullable=True, min=0)},
}


def validate_clean(tables: dict[str, pd.DataFrame]) -> dict[str, ValidationReport]:
    """Validate after :func:`clean`, with the contract relaxed where cleaning chose to."""
    reports = {}
    for name, df in tables.items():
        if name not in CONTRACTS:
            continue
        overrides = CLEAN_CONTRACT_OVERRIDES.get(name, {})
        contract = [overrides.get(c.name, c) for c in CONTRACTS[name]]
        reports[name] = validate(df, contract, name, UNIQUE_KEYS.get(name))
    return reports
