"""Reference solutions for notebook 03. Graded by tests/test_solutions.py."""

from __future__ import annotations


def ex1() -> str:
    return """
    SELECT customer_id, week_start, sessions,
           LAG(sessions) OVER w            AS prev_sessions,
           sessions - LAG(sessions) OVER w AS sessions_change
    FROM usage_weekly
    WINDOW w AS (PARTITION BY customer_id ORDER BY week_start)
    ORDER BY customer_id, week_start
    """


def ex2(as_of: str) -> str:
    return f"""
    SELECT customer_id, invoice_date, amount_usd, status
    FROM invoices
    WHERE invoice_date < TIMESTAMP '{as_of}'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_date DESC) = 1
    ORDER BY customer_id
    """


def ex3(snapshot: str) -> str:
    return f"""
    WITH active AS (
        SELECT customer_id
        FROM customers
        WHERE signup_date < TIMESTAMP '{snapshot}'
          AND (churn_date IS NULL OR churn_date >= TIMESTAMP '{snapshot}')
    )
    SELECT a.customer_id, COUNT(t.ticket_id) AS tickets_30d
    FROM active AS a
    LEFT JOIN tickets AS t
      ON  t.customer_id = a.customer_id
      AND t.created_at >= TIMESTAMP '{snapshot}' - INTERVAL 30 DAY
      AND t.created_at <  TIMESTAMP '{snapshot}'
    GROUP BY a.customer_id
    ORDER BY a.customer_id
    """
