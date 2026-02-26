"""Match quality reporting for entity resolution.

Generates comprehensive reports on match rates, confidence
distributions, and cluster statistics.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MatchReport:
    """Summary of entity resolution results.

    Attributes:
        total_records: Number of input records.
        total_matches: Number of matched pairs.
        total_clusters: Number of match clusters.
        match_rate: Fraction of records involved in at least one match.
        uncertain_count: Pairs flagged for manual review.
        unmatched_count: Records with no matches.
        avg_cluster_size: Mean records per cluster.
        confidence_distribution: Histogram of confidence scores.
    """

    total_records: int
    total_matches: int
    total_clusters: int
    match_rate: float
    uncertain_count: int
    unmatched_count: int
    avg_cluster_size: float
    confidence_distribution: dict[str, int] = field(default_factory=dict)


def generate_match_report(
    original_count: int,
    matches: pd.MultiIndex,
    uncertain: pd.MultiIndex,
    clusters: list[set[int]],
    features: pd.DataFrame,
) -> MatchReport:
    """Generate a comprehensive match-quality report.

    Args:
        original_count: Total number of input records.
        matches: MultiIndex of confirmed match pairs.
        uncertain: MultiIndex of uncertain match pairs.
        clusters: List of record-index clusters.
        features: Comparison features with a ``score`` column.

    Returns:
        :class:`MatchReport` summarising the resolution results.
    """
    matched_indices: set[int] = set()
    for cluster in clusters:
        matched_indices.update(cluster)

    avg_size = sum(len(c) for c in clusters) / len(clusters) if clusters else 0.0

    # Confidence distribution
    conf_dist: dict[str, int] = {}
    if "score" in features.columns and not features.empty:
        bins = pd.cut(
            features["score"],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            include_lowest=True,
        )
        for interval, count in bins.value_counts().items():
            conf_dist[str(interval)] = int(count)

    report = MatchReport(
        total_records=original_count,
        total_matches=len(matches),
        total_clusters=len(clusters),
        match_rate=len(matched_indices) / original_count if original_count else 0.0,
        uncertain_count=len(uncertain),
        unmatched_count=original_count - len(matched_indices),
        avg_cluster_size=avg_size,
        confidence_distribution=conf_dist,
    )
    logger.info("Match report: %s", report)
    return report


def export_review_queue(
    uncertain: pd.MultiIndex,
    features: pd.DataFrame,
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """Export uncertain matches for manual review as CSV.

    Args:
        uncertain: MultiIndex of uncertain pairs.
        features: Comparison features DataFrame.
        df: Original customer DataFrame.
        output_path: File path for the CSV output.
    """
    rows: list[dict[str, Any]] = []
    for left, right in uncertain:
        scores = features.loc[(left, right)]
        rows.append(
            {
                "left_id": df.loc[left, "customer_id"] if left in df.index else left,
                "right_id": df.loc[right, "customer_id"] if right in df.index else right,
                "left_name": df.loc[left, "name"] if left in df.index else "",
                "right_name": df.loc[right, "name"] if right in df.index else "",
                "score": float(scores.get("score", 0)),
                "name_similarity": float(scores.get("name_similarity", 0)),
                "email_match": float(scores.get("email_match", 0)),
                "phone_similarity": float(scores.get("phone_similarity", 0)),
            }
        )

    review_df = pd.DataFrame(rows)
    review_df.to_csv(output_path, index=False)
    logger.info("Exported %d uncertain pairs to %s", len(rows), output_path)
