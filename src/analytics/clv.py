"""Predictive Customer Lifetime Value (CLV) modelling.

Combines probabilistic models (BG/NBD + Gamma-Gamma) with a
Gradient Boosting regression to forecast 12-month customer value.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLVPredictor:
    """Predict customer lifetime value using ML and probabilistic models.

    Args:
        prediction_period: Forecast horizon in months.
    """

    def __init__(self, prediction_period: int = 12) -> None:
        self.prediction_period = prediction_period
        self.gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self._bgf_fitted = False
        self._ggf_fitted = False

    # ------------------------------------------------------------------
    # Probabilistic models (BG/NBD + Gamma-Gamma)
    # ------------------------------------------------------------------

    def prepare_rfm_summary(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Build RFM summary in the format expected by lifetimes.

        Args:
            transactions: Transaction DataFrame with ``customer_id``,
                ``date``, and ``amount``.

        Returns:
            Summary DataFrame with ``frequency``, ``recency``, ``T``,
            and ``monetary_value`` columns.
        """
        try:
            from lifetimes.utils import summary_data_from_transaction_data

            summary = summary_data_from_transaction_data(
                transactions,
                customer_id_col="customer_id",
                datetime_col="date",
                monetary_value_col="amount",
            )
            summary = summary[summary["frequency"] > 0]
            logger.info("RFM summary: %d repeat customers", len(summary))
            return summary
        except Exception as exc:
            logger.warning("lifetimes summary failed: %s — using fallback", exc)
            return self._fallback_rfm_summary(transactions)

    def fit_probabilistic_models(self, summary: pd.DataFrame) -> None:
        """Fit BG/NBD and Gamma-Gamma models.

        Args:
            summary: RFM summary from :meth:`prepare_rfm_summary`.
        """
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter

            self._bgf = BetaGeoFitter(penalizer_coef=0.01)
            self._bgf.fit(summary["frequency"], summary["recency"], summary["T"])
            self._bgf_fitted = True
            logger.info("BG/NBD model fitted")

            gg_data = summary[summary["frequency"] > 0].copy()
            self._ggf = GammaGammaFitter(penalizer_coef=0.01)
            self._ggf.fit(gg_data["frequency"], gg_data["monetary_value"])
            self._ggf_fitted = True
            logger.info("Gamma-Gamma model fitted")
        except Exception as exc:
            logger.warning("Probabilistic model fitting failed: %s", exc)

    def predict_clv_probabilistic(
        self,
        summary: pd.DataFrame,
        time_horizon: int | None = None,
        discount_rate: float = 0.01,
    ) -> pd.DataFrame:
        """Predict CLV using BG/NBD + Gamma-Gamma.

        Args:
            summary: RFM summary DataFrame.
            time_horizon: Prediction period in months.
            discount_rate: Monthly discount rate.

        Returns:
            Summary with ``predicted_purchases`` and ``predicted_clv``.
        """
        time_horizon = time_horizon or self.prediction_period
        result = summary.copy()

        if self._bgf_fitted and self._ggf_fitted:
            result["predicted_purchases"] = self._bgf.predict(
                time_horizon * 30,
                result["frequency"],
                result["recency"],
                result["T"],
            )
            result["predicted_clv"] = self._ggf.customer_lifetime_value(
                self._bgf,
                result["frequency"],
                result["recency"],
                result["T"],
                result["monetary_value"],
                time=time_horizon,
                discount_rate=discount_rate,
            )
        else:
            # Fallback: simple heuristic CLV
            result["predicted_purchases"] = (
                result["frequency"] / result["T"] * time_horizon * 30
            ).clip(lower=0)
            result["predicted_clv"] = result["predicted_purchases"] * result["monetary_value"]
            logger.info("Using heuristic CLV (probabilistic models unavailable)")

        return result

    # ------------------------------------------------------------------
    # ML-based CLV
    # ------------------------------------------------------------------

    def prepare_ml_features(
        self,
        transactions: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Engineer features for ML-based CLV prediction.

        Args:
            transactions: Transaction records.
            web_sessions: Web session records.
            support_tickets: Support ticket records.

        Returns:
            Feature DataFrame keyed by ``customer_id``.
        """
        txn = transactions.copy()
        txn["date"] = pd.to_datetime(txn["date"])

        txn_features = (
            txn.groupby("customer_id")
            .agg(
                total_spend=("amount", "sum"),
                avg_order=("amount", "mean"),
                std_order=("amount", "std"),
                order_count=("amount", "count"),
                first_purchase=("date", "min"),
                last_purchase=("date", "max"),
                category_diversity=("product_category", "nunique"),
            )
            .reset_index()
        )
        txn_features["std_order"] = txn_features["std_order"].fillna(0)

        now = pd.Timestamp.now()
        txn_features["tenure_days"] = (now - txn_features["first_purchase"]).dt.days
        txn_features["recency_days"] = (now - txn_features["last_purchase"]).dt.days
        txn_features["purchase_frequency"] = txn_features["order_count"] / (
            txn_features["tenure_days"].clip(lower=1) / 30
        )
        txn_features = txn_features.drop(columns=["first_purchase", "last_purchase"])

        # Web features
        web_features = (
            web_sessions.groupby("customer_id")
            .agg(
                total_sessions=("session_id", "count"),
                avg_pages=("pages_visited", "mean"),
                avg_time_on_site=("time_on_site", "mean"),
            )
            .reset_index()
        )
        features = txn_features.merge(web_features, on="customer_id", how="left")

        # Support features
        support_features = (
            support_tickets.groupby("customer_id")
            .agg(
                ticket_count=("ticket_id", "count"),
                avg_satisfaction=("satisfaction_score", "mean"),
            )
            .reset_index()
        )
        features = features.merge(support_features, on="customer_id", how="left")
        features = features.fillna(0)
        return features

    def fit_ml_model(self, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Fit a Gradient Boosting model and return evaluation metrics.

        Args:
            features: Feature DataFrame (``customer_id`` excluded
                automatically).
            target: Series of target CLV values.

        Returns:
            Dictionary with ``mae``, ``r2``, ``cv_scores``, and
            ``feature_importance``.
        """
        feature_cols = [c for c in features.columns if c != "customer_id"]
        X = features[feature_cols]
        y = target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.gb_model.fit(X_train, y_train)

        y_pred = self.gb_model.predict(X_test)
        metrics: dict[str, Any] = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "cv_scores": float(
                cross_val_score(self.gb_model, X, y, cv=min(5, len(X)), scoring="r2").mean()
            ),
        }

        importance = dict(zip(feature_cols, self.gb_model.feature_importances_, strict=False))
        metrics["feature_importance"] = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )
        logger.info("ML model — MAE: %.2f, R2: %.3f", metrics["mae"], metrics["r2"])
        return metrics

    def segment_by_clv(
        self,
        clv_predictions: pd.DataFrame,
        percentiles: tuple[float, float] = (0.33, 0.66),
    ) -> pd.DataFrame:
        """Assign customers to High / Medium / Low CLV tiers.

        Args:
            clv_predictions: DataFrame with ``predicted_clv`` column.
            percentiles: Quantile boundaries for tier assignment.

        Returns:
            DataFrame with ``clv_tier`` column added.
        """
        result = clv_predictions.copy()
        low_t = result["predicted_clv"].quantile(percentiles[0])
        high_t = result["predicted_clv"].quantile(percentiles[1])

        result["clv_tier"] = pd.cut(
            result["predicted_clv"],
            bins=[-np.inf, low_t, high_t, np.inf],
            labels=["Low Value", "Medium Value", "High Value"],
        )
        tier_counts = result["clv_tier"].value_counts()
        logger.info("CLV tiers:\n%s", tier_counts.to_string())
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_rfm_summary(transactions: pd.DataFrame) -> pd.DataFrame:
        """Simple RFM summary without lifetimes dependency."""
        txn = transactions.copy()
        txn["date"] = pd.to_datetime(txn["date"])
        now = txn["date"].max()

        summary = txn.groupby("customer_id").agg(
            frequency=("date", lambda x: len(x) - 1),
            recency=("date", lambda x: (x.max() - x.min()).days),
            T=("date", lambda x: (now - x.min()).days),
            monetary_value=("amount", "mean"),
        )
        summary = summary[summary["frequency"] > 0]
        return summary
