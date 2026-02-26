"""Tests for data quality assessment."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataLoader
from src.data.quality import assess_data_quality


@pytest.fixture()
def db_with_data(tmp_path: Path) -> str:
    """Create a DuckDB with sample data and return its path."""
    db_path = str(tmp_path / "quality_test.duckdb")
    loader = DataLoader(db_path)

    crm = pd.DataFrame(
        {
            "customer_id": ["CRM_000001", "CRM_000002", "CRM_DUP_000001"],
            "name": ["Alice", "Bob", "alice"],
            "email": ["alice@test.com", None, "alice@test.com"],
            "phone": ["1234567890", "0987654321", "1234567890"],
        }
    )
    txn = pd.DataFrame(
        {
            "transaction_id": ["TXN_1", "TXN_2", "TXN_3"],
            "customer_id": ["CRM_000001", "CRM_000002", "ORPHAN_999"],
            "amount": [100.0, 200.0, 50.0],
        }
    )
    web = pd.DataFrame(
        {
            "session_id": ["SES_1", "SES_2"],
            "customer_id": ["CRM_000001", None],
        }
    )
    support = pd.DataFrame(
        {
            "ticket_id": ["TKT_1"],
            "customer_id": ["CRM_000001"],
        }
    )

    loader.load_all_data(
        {
            "crm_customers": crm,
            "transactions": txn,
            "web_sessions": web,
            "support_tickets": support,
        }
    )
    return db_path


class TestAssessDataQuality:
    """Tests for assess_data_quality function."""

    def test_report_keys(self, db_with_data: str) -> None:
        report = assess_data_quality(db_with_data)
        assert "total_customers" in report
        assert "tables" in report
        assert "orphan_transaction_rate" in report
        assert "missing_email_rate" in report
        assert "duplicate_rate" in report

    def test_record_counts(self, db_with_data: str) -> None:
        report = assess_data_quality(db_with_data)
        assert report["tables"]["crm_customers"]["count"] == 3
        assert report["tables"]["transactions"]["count"] == 3
        assert report["tables"]["web_sessions"]["count"] == 2
        assert report["tables"]["support_tickets"]["count"] == 1

    def test_orphan_rate(self, db_with_data: str) -> None:
        report = assess_data_quality(db_with_data)
        # 1 orphan out of 3 transactions
        assert abs(report["orphan_transaction_rate"] - 1 / 3) < 0.01

    def test_missing_email_rate(self, db_with_data: str) -> None:
        report = assess_data_quality(db_with_data)
        # 1 missing out of 3 customers
        assert abs(report["missing_email_rate"] - 1 / 3) < 0.01

    def test_duplicate_rate(self, db_with_data: str) -> None:
        report = assess_data_quality(db_with_data)
        # 1 DUP out of 3 customers
        assert abs(report["duplicate_rate"] - 1 / 3) < 0.01
