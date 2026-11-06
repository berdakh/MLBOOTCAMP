"""Point-in-time features and labels, built in SQL with DuckDB.

The single most important rule in this module:

    A feature for snapshot T may use only rows whose timestamp is < T.
    The label for snapshot T uses only rows in [T, T + 30 days).

Usage is recorded per *week starting* on ``week_start``. A week is only known
once it is over, so a week counts as "before T" when ``week_start + 7 days <= T``.
Forgetting that detail leaks up to six days of the future into every feature.

The SQL lives in module-level strings so notebooks can print it, and so that a
reviewer can read the whole feature definition in one place.
"""

from __future__ import annotations

import duckdb
import pandas as pd

LABEL = "churned_30d"
KEYS = ["snapshot_date", "customer_id"]

CATEGORICAL = ["plan", "region", "acquisition_channel", "company_size"]
NUMERIC = [
    "seats",
    "tenure_days",
    "mrr_usd",
    "sessions_4w",
    "sessions_prev_4w",
    "sessions_trend",
    "seat_utilization_4w",
    "features_used_4w",
    "weeks_since_last_active",
    "tickets_30d",
    "tickets_90d",
    "billing_tickets_90d",
    "cancellation_tickets_30d",
    "avg_resolution_hours_90d",
    "failed_payments_90d",
]
FEATURES = CATEGORICAL + NUMERIC

#: Columns that look useful and are *not allowed*: each describes the account
#: as of the export date, i.e. the future relative to a snapshot.
LEAKY_COLUMNS = ["status", "churn_date", "last_active_week"]

SNAPSHOT_SQL = """
-- One row per (snapshot, account still active at the snapshot), with its label.
SELECT
    s.snapshot_date,
    c.customer_id,
    c.plan, c.seats, c.region, c.acquisition_channel, c.company_size,
    date_diff('day', c.signup_date, s.snapshot_date) AS tenure_days,
    CASE WHEN c.churn_date >= s.snapshot_date
          AND c.churn_date <  s.snapshot_date + INTERVAL 30 DAY
         THEN 1 ELSE 0 END AS churned_30d
FROM snapshots AS s
JOIN customers AS c
  ON c.signup_date < s.snapshot_date
WHERE c.churn_date IS NULL OR c.churn_date >= s.snapshot_date
"""

USAGE_SQL = """
-- Usage aggregates over the last 4 and the previous 4 *completed* weeks.
SELECT
    b.snapshot_date,
    b.customer_id,
    SUM(CASE WHEN u.week_start + INTERVAL 7 DAY >  b.snapshot_date - INTERVAL 28 DAY
             THEN u.sessions END)                                AS sessions_4w,
    SUM(CASE WHEN u.week_start + INTERVAL 7 DAY <= b.snapshot_date - INTERVAL 28 DAY
             THEN u.sessions END)                                AS sessions_prev_4w,
    SUM(CASE WHEN u.week_start + INTERVAL 7 DAY >  b.snapshot_date - INTERVAL 28 DAY
             THEN u.active_users END) / 4.0                      AS active_users_4w,
    AVG(CASE WHEN u.week_start + INTERVAL 7 DAY >  b.snapshot_date - INTERVAL 28 DAY
             THEN u.features_used END)                           AS features_used_4w
FROM base AS b
JOIN usage_weekly AS u
  ON  u.customer_id = b.customer_id
  AND u.week_start + INTERVAL 7 DAY <= b.snapshot_date            -- week is complete
  AND u.week_start + INTERVAL 7 DAY >  b.snapshot_date - INTERVAL 56 DAY
GROUP BY b.snapshot_date, b.customer_id
"""

LAST_ACTIVE_SQL = """
-- Most recent completed week with any activity, strictly before the snapshot.
SELECT b.snapshot_date, b.customer_id,
       MAX(u.week_start + INTERVAL 7 DAY) AS last_active_end
FROM base AS b
JOIN usage_weekly AS u
  ON u.customer_id = b.customer_id
 AND u.week_start + INTERVAL 7 DAY <= b.snapshot_date
GROUP BY b.snapshot_date, b.customer_id
"""

TICKETS_SQL = """
SELECT
    b.snapshot_date,
    b.customer_id,
    COUNT(*) FILTER (WHERE t.created_at >= b.snapshot_date - INTERVAL 30 DAY)  AS tickets_30d,
    COUNT(*)                                                                    AS tickets_90d,
    COUNT(*) FILTER (WHERE t.category = 'billing')                             AS billing_tickets_90d,
    COUNT(*) FILTER (WHERE t.category = 'cancellation'
                       AND t.created_at >= b.snapshot_date - INTERVAL 30 DAY)  AS cancellation_tickets_30d,
    AVG(t.resolution_hours)                                                     AS avg_resolution_hours_90d
FROM base AS b
JOIN tickets AS t
  ON  t.customer_id = b.customer_id
  AND t.created_at <  b.snapshot_date
  AND t.created_at >= b.snapshot_date - INTERVAL 90 DAY
GROUP BY b.snapshot_date, b.customer_id
"""

INVOICES_SQL = """
SELECT
    b.snapshot_date,
    b.customer_id,
    COUNT(*) FILTER (WHERE i.status = 'failed'
                       AND i.invoice_date >= b.snapshot_date - INTERVAL 90 DAY) AS failed_payments_90d,
    arg_max(i.amount_usd, i.invoice_date)                                       AS mrr_usd
FROM base AS b
JOIN invoices AS i
  ON  i.customer_id = b.customer_id
  AND i.invoice_date < b.snapshot_date
GROUP BY b.snapshot_date, b.customer_id
"""

ASSEMBLE_SQL = """
SELECT
    b.*,
    COALESCE(inv.mrr_usd, 0)                       AS mrr_usd,
    COALESCE(u.sessions_4w, 0)                     AS sessions_4w,
    COALESCE(u.sessions_prev_4w, 0)                AS sessions_prev_4w,
    (COALESCE(u.sessions_4w, 0) + 1) / (COALESCE(u.sessions_prev_4w, 0) + 1) AS sessions_trend,
    COALESCE(u.active_users_4w, 0) / b.seats       AS seat_utilization_4w,
    u.features_used_4w                             AS features_used_4w,
    COALESCE(date_diff('day', la.last_active_end, b.snapshot_date) / 7.0,
             b.tenure_days / 7.0)                  AS weeks_since_last_active,
    COALESCE(t.tickets_30d, 0)                     AS tickets_30d,
    COALESCE(t.tickets_90d, 0)                     AS tickets_90d,
    COALESCE(t.billing_tickets_90d, 0)             AS billing_tickets_90d,
    COALESCE(t.cancellation_tickets_30d, 0)        AS cancellation_tickets_30d,
    t.avg_resolution_hours_90d                     AS avg_resolution_hours_90d,
    COALESCE(inv.failed_payments_90d, 0)           AS failed_payments_90d
FROM base AS b
LEFT JOIN usage_agg   AS u   USING (snapshot_date, customer_id)
LEFT JOIN last_active AS la  USING (snapshot_date, customer_id)
LEFT JOIN ticket_agg  AS t   USING (snapshot_date, customer_id)
LEFT JOIN invoice_agg AS inv USING (snapshot_date, customer_id)
ORDER BY b.snapshot_date, b.customer_id
"""

LEAKY_SQL = """
-- DO NOT USE IN A MODEL. Built only to demonstrate leakage on Day 1.
SELECT f.*,
       CASE WHEN c.status = 'cancelled' THEN 1 ELSE 0 END                       AS status_cancelled,
       date_diff('day', f.snapshot_date, c.last_active_week)                     AS days_until_last_active
FROM features AS f JOIN customers AS c USING (customer_id)
"""

DEFAULT_SNAPSHOTS = pd.date_range("2023-03-01", "2024-11-01", freq="MS")


def connect(tables: dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB connection with each DataFrame registered as a view."""
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con


def build_features(
    tables: dict[str, pd.DataFrame],
    snapshots=DEFAULT_SNAPSHOTS,
    include_leaky: bool = False,
) -> pd.DataFrame:
    """Build the (snapshot, customer) training table from *cleaned* raw tables."""
    con = connect(tables)
    con.register("snapshots", pd.DataFrame({"snapshot_date": pd.to_datetime(list(snapshots))}))
    con.execute(f"CREATE TEMP TABLE base AS {SNAPSHOT_SQL}")
    con.execute(f"CREATE TEMP TABLE usage_agg AS {USAGE_SQL}")
    con.execute(f"CREATE TEMP TABLE last_active AS {LAST_ACTIVE_SQL}")
    con.execute(f"CREATE TEMP TABLE ticket_agg AS {TICKETS_SQL}")
    con.execute(f"CREATE TEMP TABLE invoice_agg AS {INVOICES_SQL}")
    df = con.execute(ASSEMBLE_SQL).df()
    if include_leaky:
        con.register("features", df)
        df = con.execute(LEAKY_SQL).df()
    con.close()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df[LABEL] = df[LABEL].astype(int)
    for col in NUMERIC:
        df[col] = df[col].astype(float)
    return df
