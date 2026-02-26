"""Tests for RFM analysis and segmentation."""

from datetime import datetime

import pandas as pd
import pytest

from src.analytics.rfm import RFMAnalyzer


@pytest.fixture()
def analyzer() -> RFMAnalyzer:
    return RFMAnalyzer(reference_date=datetime(2025, 1, 1))


@pytest.fixture()
def transactions() -> pd.DataFrame:
    """Synthetic transactions for 5 customers."""
    rows = []
    # Customer A: recent, frequent, high spend → Champion candidate
    for i in range(20):
        rows.append(
            {
                "transaction_id": f"T_A_{i}",
                "customer_id": "A",
                "date": "2024-12-15",
                "amount": 200.0,
            }
        )
    # Customer B: old, infrequent, low spend → Lost candidate
    rows.append(
        {
            "transaction_id": "T_B_0",
            "customer_id": "B",
            "date": "2023-01-01",
            "amount": 10.0,
        }
    )
    # Customers C, D, E: mid-range
    for cid, date, amt, count in [
        ("C", "2024-10-01", 100.0, 5),
        ("D", "2024-06-01", 50.0, 3),
        ("E", "2024-03-01", 30.0, 2),
    ]:
        for i in range(count):
            rows.append(
                {
                    "transaction_id": f"T_{cid}_{i}",
                    "customer_id": cid,
                    "date": date,
                    "amount": amt,
                }
            )
    return pd.DataFrame(rows)


class TestCalculateRFM:
    """Tests for raw RFM metric calculation."""

    def test_output_columns(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        assert {"customer_id", "recency", "frequency", "monetary"}.issubset(set(rfm.columns))

    def test_customer_count(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        assert len(rfm) == 5

    def test_recency_values(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        a_row = rfm[rfm["customer_id"] == "A"].iloc[0]
        b_row = rfm[rfm["customer_id"] == "B"].iloc[0]
        assert a_row["recency"] < b_row["recency"]

    def test_frequency_values(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        a_row = rfm[rfm["customer_id"] == "A"].iloc[0]
        assert a_row["frequency"] == 20


class TestAssignScores:
    """Tests for quintile scoring."""

    def test_score_range(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        for col in ["r_score", "f_score", "m_score"]:
            assert scored[col].min() >= 1
            assert scored[col].max() <= 5

    def test_rfm_score_string(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        assert scored["rfm_score"].str.len().eq(3).all()


class TestAssignSegments:
    """Tests for segment labeling."""

    def test_all_customers_segmented(
        self, analyzer: RFMAnalyzer, transactions: pd.DataFrame
    ) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        segmented = analyzer.assign_segments(scored)
        assert segmented["segment"].notna().all()

    def test_known_segment_mapping(self, analyzer: RFMAnalyzer) -> None:
        """Direct test of segment mapping with known scores."""
        rfm = pd.DataFrame(
            {
                "customer_id": ["X"],
                "recency": [10],
                "frequency": [50],
                "monetary": [5000],
                "r_score": [5],
                "f_score": [5],
                "m_score": [5],
                "rfm_score": ["555"],
            }
        )
        result = analyzer.assign_segments(rfm)
        assert result.iloc[0]["segment"] == "Champions"


class TestGetSegmentSummary:
    """Tests for segment summary statistics."""

    def test_summary_columns(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        segmented = analyzer.assign_segments(scored)
        summary = analyzer.get_segment_summary(segmented)
        assert "customer_count" in summary.columns
        assert "pct_customers" in summary.columns
        assert "pct_revenue" in summary.columns

    def test_pct_customers_sum(self, analyzer: RFMAnalyzer, transactions: pd.DataFrame) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        segmented = analyzer.assign_segments(scored)
        summary = analyzer.get_segment_summary(segmented)
        assert abs(summary["pct_customers"].sum() - 100.0) < 1.0

    def test_customer_count_matches(
        self, analyzer: RFMAnalyzer, transactions: pd.DataFrame
    ) -> None:
        rfm = analyzer.calculate_rfm(transactions)
        scored = analyzer.assign_scores(rfm)
        segmented = analyzer.assign_segments(scored)
        summary = analyzer.get_segment_summary(segmented)
        assert summary["customer_count"].sum() == 5
