"""Tests for cohort analysis."""

import pandas as pd
import pytest

from src.analytics.cohort import CohortAnalyzer


@pytest.fixture()
def analyzer() -> CohortAnalyzer:
    return CohortAnalyzer()


@pytest.fixture()
def customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "signup_date": ["2024-01-15", "2024-01-20", "2024-02-10", "2024-03-05"],
        }
    )


@pytest.fixture()
def transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(8)],
            "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C1"],
            "date": [
                "2024-01-20",
                "2024-02-15",
                "2024-01-25",
                "2024-03-10",
                "2024-02-15",
                "2024-03-20",
                "2024-03-10",
                "2024-03-20",
            ],
            "amount": [100, 150, 200, 50, 80, 120, 90, 300],
        }
    )


class TestCreateCohorts:
    def test_cohort_month_assigned(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        result = analyzer.create_cohorts(customers, transactions)
        assert "cohort_month" in result.columns
        assert "cohort_age" in result.columns

    def test_cohort_age_nonnegative(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        result = analyzer.create_cohorts(customers, transactions)
        assert (result["cohort_age"] >= 0).all()


class TestCalculateRetention:
    def test_retention_shape(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        analyzer.create_cohorts(customers, transactions)
        matrix = analyzer.calculate_retention()
        # Should have rows for cohort months and columns for ages
        assert matrix.shape[0] > 0
        assert matrix.shape[1] > 0

    def test_month_zero_is_100(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        analyzer.create_cohorts(customers, transactions)
        matrix = analyzer.calculate_retention()
        if 0 in matrix.columns:
            assert (matrix[0] == 100.0).all()

    def test_raises_without_data(self, analyzer: CohortAnalyzer) -> None:
        with pytest.raises(ValueError):
            analyzer.calculate_retention()


class TestHeatmapData:
    def test_keys_present(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        analyzer.create_cohorts(customers, transactions)
        analyzer.calculate_retention()
        data = analyzer.get_retention_heatmap_data()
        assert "z" in data
        assert "x" in data
        assert "y" in data

    def test_raises_without_retention(self, analyzer: CohortAnalyzer) -> None:
        with pytest.raises(ValueError):
            analyzer.get_retention_heatmap_data()


class TestCohortMetrics:
    def test_metric_columns(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        metrics = analyzer.calculate_cohort_metrics(customers, transactions)
        assert "ltv" in metrics.columns
        assert "orders_per_customer" in metrics.columns
        assert "total_revenue" in metrics.columns


class TestAverageRetentionCurve:
    def test_curve_length(
        self, analyzer: CohortAnalyzer, customers: pd.DataFrame, transactions: pd.DataFrame
    ) -> None:
        analyzer.create_cohorts(customers, transactions)
        matrix = analyzer.calculate_retention()
        curve = analyzer.get_average_retention_curve()
        assert len(curve) == matrix.shape[1]
