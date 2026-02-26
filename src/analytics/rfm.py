"""RFM (Recency, Frequency, Monetary) analysis and segmentation.

Calculates per-customer RFM metrics, assigns quintile-based scores,
and maps score combinations to meaningful segment labels.
"""

from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping of (R, F, M) score tuples to segment labels.
_SEGMENT_MAP: dict[tuple[int, int, int], str] = {
    (5, 5, 5): "Champions",
    (5, 5, 4): "Champions",
    (5, 4, 5): "Champions",
    (4, 5, 5): "Loyal Customers",
    (5, 4, 4): "Loyal Customers",
    (4, 4, 5): "Loyal Customers",
    (4, 4, 4): "Loyal Customers",
    (5, 3, 3): "Potential Loyalists",
    (4, 3, 3): "Potential Loyalists",
    (3, 3, 3): "Potential Loyalists",
    (5, 1, 1): "New Customers",
    (4, 1, 1): "New Customers",
    (3, 1, 1): "Promising",
    (3, 2, 2): "Promising",
    (2, 2, 2): "Need Attention",
    (2, 3, 3): "Need Attention",
    (1, 3, 3): "About to Sleep",
    (2, 2, 3): "About to Sleep",
    (1, 2, 2): "At Risk",
    (1, 2, 3): "At Risk",
    (2, 1, 1): "At Risk",
    (1, 1, 3): "Can't Lose Them",
    (1, 1, 4): "Can't Lose Them",
    (1, 1, 5): "Can't Lose Them",
    (1, 1, 2): "Hibernating",
    (1, 1, 1): "Lost",
}


class RFMAnalyzer:
    """Compute RFM scores and assign customer segments.

    Args:
        reference_date: Date used to calculate recency.  Defaults to
            ``datetime.now()``.
    """

    def __init__(self, reference_date: datetime | None = None) -> None:
        self.reference_date = reference_date or datetime.now()

    def calculate_rfm(
        self,
        transactions: pd.DataFrame,
        customer_id_col: str = "customer_id",
    ) -> pd.DataFrame:
        """Calculate raw RFM metrics per customer.

        Args:
            transactions: DataFrame with *customer_id*, *date*, *amount*,
                and *transaction_id* columns.
            customer_id_col: Column name identifying the customer.

        Returns:
            DataFrame with columns ``customer_id``, ``recency``,
            ``frequency``, ``monetary``.
        """
        txn = transactions.copy()
        txn["date"] = pd.to_datetime(txn["date"])

        rfm = (
            txn.groupby(customer_id_col)
            .agg(
                recency=("date", lambda x: (self.reference_date - x.max()).days),
                frequency=("transaction_id", "count"),
                monetary=("amount", "sum"),
            )
            .reset_index()
        )
        logger.info("Calculated RFM for %d customers", len(rfm))
        return rfm

    def assign_scores(self, rfm: pd.DataFrame, n_segments: int = 5) -> pd.DataFrame:
        """Assign quintile-based RFM scores (1–5).

        Args:
            rfm: DataFrame with ``recency``, ``frequency``, ``monetary``.
            n_segments: Number of quantile bins.

        Returns:
            DataFrame with additional ``r_score``, ``f_score``,
            ``m_score``, and ``rfm_score`` columns.
        """
        rfm = rfm.copy()

        # Recency: lower days = higher score (reverse)
        rfm["r_score"] = pd.qcut(
            rfm["recency"],
            q=n_segments,
            labels=range(n_segments, 0, -1),
            duplicates="drop",
        ).astype(int)

        # Frequency & monetary: higher = higher score
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            q=n_segments,
            labels=range(1, n_segments + 1),
            duplicates="drop",
        ).astype(int)

        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"),
            q=n_segments,
            labels=range(1, n_segments + 1),
            duplicates="drop",
        ).astype(int)

        rfm["rfm_score"] = (
            rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
        )
        return rfm

    def assign_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Map RFM score combinations to human-readable segment names.

        Args:
            rfm: DataFrame with ``r_score``, ``f_score``, ``m_score``.

        Returns:
            DataFrame with a ``segment`` column.
        """
        rfm = rfm.copy()

        def _get_segment(row: pd.Series) -> str:
            key = (int(row["r_score"]), int(row["f_score"]), int(row["m_score"]))
            if key in _SEGMENT_MAP:
                return _SEGMENT_MAP[key]
            # Fallback rules
            if row["r_score"] >= 4 and row["f_score"] >= 4:
                return "Loyal Customers"
            if row["r_score"] >= 4:
                return "Potential Loyalists"
            if row["r_score"] <= 2 and row["m_score"] >= 4:
                return "Can't Lose Them"
            if row["r_score"] <= 2:
                return "At Risk"
            return "Need Attention"

        rfm["segment"] = rfm.apply(_get_segment, axis=1)

        segment_counts = rfm["segment"].value_counts()
        logger.info("Segment distribution:\n%s", segment_counts.to_string())
        return rfm

    def get_segment_summary(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Summarise each segment's size and financial metrics.

        Args:
            rfm: Scored DataFrame with ``segment``, ``recency``,
                ``frequency``, ``monetary`` columns.

        Returns:
            Summary DataFrame indexed by segment.
        """
        summary = (
            rfm.groupby("segment")
            .agg(
                customer_count=("customer_id", "count"),
                avg_recency=("recency", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                total_monetary=("monetary", "sum"),
            )
            .round(2)
        )
        total = summary["customer_count"].sum()
        summary["pct_customers"] = (summary["customer_count"] / total * 100).round(1)
        summary["pct_revenue"] = (
            summary["total_monetary"] / summary["total_monetary"].sum() * 100
        ).round(1)
        return summary.sort_values("total_monetary", ascending=False)
