"""Cohort analysis and retention tracking.

Assigns customers to cohorts by signup month, calculates
period-over-period retention rates, and formats the results
for heatmap visualisation.
"""

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CohortAnalyzer:
    """Build cohort retention matrices from customer and transaction data."""

    def __init__(self) -> None:
        self.cohort_data: pd.DataFrame | None = None
        self.retention_matrix: pd.DataFrame | None = None

    def create_cohorts(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign customers to signup-month cohorts and merge with transactions.

        Args:
            customers: Customer DataFrame with ``customer_id`` and
                ``signup_date``.
            transactions: Transaction DataFrame with ``customer_id``
                and ``date``.

        Returns:
            Merged DataFrame with ``cohort_month`` and ``cohort_age``.
        """
        cust = customers.copy()
        cust["signup_date"] = pd.to_datetime(cust["signup_date"])
        cust["cohort_month"] = cust["signup_date"].dt.to_period("M")

        txn = transactions.copy()
        txn["date"] = pd.to_datetime(txn["date"])
        txn["transaction_month"] = txn["date"].dt.to_period("M")

        cohort_data = txn.merge(
            cust[["customer_id", "cohort_month"]],
            on="customer_id",
            how="left",
        )
        cohort_data = cohort_data.dropna(subset=["cohort_month"])
        cohort_data["cohort_age"] = cohort_data["transaction_month"].astype(int) - cohort_data[
            "cohort_month"
        ].astype(int)

        self.cohort_data = cohort_data
        logger.info(
            "Created cohorts spanning %d months",
            cust["cohort_month"].nunique(),
        )
        return cohort_data

    def calculate_retention(self, cohort_data: pd.DataFrame | None = None) -> pd.DataFrame:
        """Calculate the retention matrix (% active per period).

        Args:
            cohort_data: Output from :meth:`create_cohorts`.
                Uses stored data if *None*.

        Returns:
            Pivot table with cohort months as rows and cohort age as
            columns, values being retention percentages.
        """
        if cohort_data is None:
            cohort_data = self.cohort_data
        if cohort_data is None:
            raise ValueError("No cohort data available — call create_cohorts first")

        cohort_counts = (
            cohort_data.groupby(["cohort_month", "cohort_age"])
            .agg(customers=("customer_id", "nunique"))
            .reset_index()
        )

        cohort_sizes = cohort_counts[cohort_counts["cohort_age"] == 0][
            ["cohort_month", "customers"]
        ].rename(columns={"customers": "cohort_size"})

        retention = cohort_counts.merge(cohort_sizes, on="cohort_month")
        retention["retention_rate"] = (
            retention["customers"] / retention["cohort_size"] * 100
        ).round(1)

        self.retention_matrix = retention.pivot(
            index="cohort_month",
            columns="cohort_age",
            values="retention_rate",
        )
        logger.info(
            "Retention matrix: %d cohorts x %d periods",
            self.retention_matrix.shape[0],
            self.retention_matrix.shape[1],
        )
        return self.retention_matrix

    def get_retention_heatmap_data(self) -> dict[str, Any]:
        """Format the retention matrix for Plotly heatmap rendering.

        Returns:
            Dict with ``z``, ``x``, ``y``, ``colorscale`` keys.

        Raises:
            ValueError: If retention has not been calculated yet.
        """
        if self.retention_matrix is None:
            raise ValueError("Must calculate retention first")

        return {
            "z": self.retention_matrix.values.tolist(),
            "x": [f"Month {i}" for i in self.retention_matrix.columns],
            "y": [str(m) for m in self.retention_matrix.index],
            "colorscale": "RdYlGn",
        }

    def calculate_cohort_metrics(
        self,
        customers: pd.DataFrame,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute per-cohort revenue and engagement aggregates.

        Args:
            customers: Customer DataFrame.
            transactions: Transaction DataFrame.

        Returns:
            DataFrame with per-cohort metrics.
        """
        cust = customers.copy()
        cust["signup_date"] = pd.to_datetime(cust["signup_date"])
        cust["cohort_month"] = cust["signup_date"].dt.to_period("M").astype(str)

        txn_with_cohort = transactions.merge(
            cust[["customer_id", "cohort_month"]], on="customer_id"
        )

        metrics = (
            txn_with_cohort.groupby("cohort_month")
            .agg(
                unique_customers=("customer_id", "nunique"),
                total_revenue=("amount", "sum"),
                avg_order_value=("amount", "mean"),
                total_orders=("amount", "count"),
            )
            .round(2)
        )
        metrics["orders_per_customer"] = (
            metrics["total_orders"] / metrics["unique_customers"]
        ).round(2)
        metrics["ltv"] = (metrics["total_revenue"] / metrics["unique_customers"]).round(2)

        return metrics.reset_index()

    def get_average_retention_curve(self) -> pd.Series:
        """Average retention rate across all cohorts per period.

        Returns:
            Series indexed by cohort age.

        Raises:
            ValueError: If retention has not been calculated yet.
        """
        if self.retention_matrix is None:
            raise ValueError("Must calculate retention first")
        return self.retention_matrix.mean(axis=0)
