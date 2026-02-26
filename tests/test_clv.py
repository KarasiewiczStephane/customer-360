"""Tests for predictive CLV module."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.clv import CLVPredictor


@pytest.fixture()
def predictor() -> CLVPredictor:
    return CLVPredictor(prediction_period=12)


@pytest.fixture()
def transactions() -> pd.DataFrame:
    """Synthetic transactions spanning 2 years."""
    np.random.seed(42)
    rows = []
    for cid in range(20):
        n_txn = np.random.randint(2, 15)
        for t in range(n_txn):
            rows.append(
                {
                    "transaction_id": f"T_{cid}_{t}",
                    "customer_id": f"C_{cid}",
                    "date": pd.Timestamp("2023-01-01")
                    + pd.Timedelta(days=int(np.random.randint(0, 700))),
                    "amount": round(np.random.lognormal(3.5, 1.0), 2),
                    "product_category": np.random.choice(["Electronics", "Clothing", "Food"]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def web_sessions() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(40):
        rows.append(
            {
                "session_id": f"S_{i}",
                "customer_id": f"C_{i % 20}",
                "pages_visited": np.random.randint(1, 15),
                "time_on_site": np.random.randint(10, 600),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def support_tickets() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(15):
        rows.append(
            {
                "ticket_id": f"TK_{i}",
                "customer_id": f"C_{i % 20}",
                "satisfaction_score": np.random.randint(1, 6),
            }
        )
    return pd.DataFrame(rows)


class TestPrepareRFMSummary:
    def test_columns_present(self, predictor: CLVPredictor, transactions: pd.DataFrame) -> None:
        summary = predictor.prepare_rfm_summary(transactions)
        assert "frequency" in summary.columns
        assert "monetary_value" in summary.columns

    def test_only_repeat_customers(
        self, predictor: CLVPredictor, transactions: pd.DataFrame
    ) -> None:
        summary = predictor.prepare_rfm_summary(transactions)
        assert (summary["frequency"] > 0).all()


class TestProbabilisticModels:
    def test_fit_and_predict(self, predictor: CLVPredictor, transactions: pd.DataFrame) -> None:
        summary = predictor.prepare_rfm_summary(transactions)
        predictor.fit_probabilistic_models(summary)
        result = predictor.predict_clv_probabilistic(summary)
        assert "predicted_clv" in result.columns
        assert "predicted_purchases" in result.columns
        assert len(result) == len(summary)


class TestMLFeatures:
    def test_feature_columns(
        self,
        predictor: CLVPredictor,
        transactions: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> None:
        features = predictor.prepare_ml_features(transactions, web_sessions, support_tickets)
        expected = {
            "customer_id",
            "total_spend",
            "avg_order",
            "order_count",
            "category_diversity",
            "tenure_days",
            "recency_days",
        }
        assert expected.issubset(set(features.columns))

    def test_no_nans(
        self,
        predictor: CLVPredictor,
        transactions: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> None:
        features = predictor.prepare_ml_features(transactions, web_sessions, support_tickets)
        assert features.isna().sum().sum() == 0


class TestFitMLModel:
    def test_metrics_returned(
        self,
        predictor: CLVPredictor,
        transactions: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> None:
        features = predictor.prepare_ml_features(transactions, web_sessions, support_tickets)
        target = features["total_spend"]
        metrics = predictor.fit_ml_model(features, target)
        assert "mae" in metrics
        assert "r2" in metrics
        assert "feature_importance" in metrics

    def test_feature_importance_keys(
        self,
        predictor: CLVPredictor,
        transactions: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> None:
        features = predictor.prepare_ml_features(transactions, web_sessions, support_tickets)
        target = features["total_spend"]
        metrics = predictor.fit_ml_model(features, target)
        assert len(metrics["feature_importance"]) > 0


class TestSegmentByCLV:
    def test_tier_labels(self, predictor: CLVPredictor) -> None:
        df = pd.DataFrame(
            {
                "customer_id": [f"C_{i}" for i in range(100)],
                "predicted_clv": np.random.uniform(0, 1000, 100),
            }
        )
        result = predictor.segment_by_clv(df)
        assert "clv_tier" in result.columns
        assert set(result["clv_tier"].unique()).issubset(
            {"Low Value", "Medium Value", "High Value"}
        )

    def test_all_assigned(self, predictor: CLVPredictor) -> None:
        df = pd.DataFrame(
            {
                "customer_id": [f"C_{i}" for i in range(50)],
                "predicted_clv": np.random.uniform(10, 500, 50),
            }
        )
        result = predictor.segment_by_clv(df)
        assert result["clv_tier"].notna().all()
