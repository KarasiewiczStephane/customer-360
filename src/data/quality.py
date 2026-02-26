"""Data quality assessment for the Customer 360 platform.

Generates a quality report covering record counts, missing-value
rates, duplicate rates, and orphan transaction detection.
"""

from typing import Any

import duckdb

from src.utils.logger import get_logger

logger = get_logger(__name__)


def assess_data_quality(db_path: str) -> dict[str, Any]:
    """Generate a data-quality report from DuckDB tables.

    Args:
        db_path: Path to the DuckDB database file.

    Returns:
        Dictionary containing quality metrics for each table.
    """
    report: dict[str, Any] = {
        "total_customers": 0,
        "duplicate_rate": 0.0,
        "missing_email_rate": 0.0,
        "orphan_transaction_rate": 0.0,
        "tables": {},
    }

    with duckdb.connect(db_path) as conn:
        for table in [
            "crm_customers",
            "transactions",
            "web_sessions",
            "support_tickets",
        ]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            report["tables"][table] = {"count": count}

        report["total_customers"] = report["tables"]["crm_customers"]["count"]

        # Orphan transactions
        orphans = conn.execute(
            """
            SELECT COUNT(*) FROM transactions t
            LEFT JOIN crm_customers c ON t.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
            """
        ).fetchone()[0]
        txn_count = report["tables"]["transactions"]["count"]
        report["orphan_transaction_rate"] = orphans / txn_count if txn_count else 0.0

        # Missing emails
        missing_emails = conn.execute(
            "SELECT COUNT(*) FROM crm_customers WHERE email IS NULL"
        ).fetchone()[0]
        cust_count = report["total_customers"]
        report["missing_email_rate"] = missing_emails / cust_count if cust_count else 0.0

        # Duplicate rate (IDs containing DUP)
        dup_count = conn.execute(
            "SELECT COUNT(*) FROM crm_customers WHERE customer_id LIKE '%DUP%'"
        ).fetchone()[0]
        report["duplicate_rate"] = dup_count / cust_count if cust_count else 0.0

    logger.info("Data quality report: %s", report)
    return report
