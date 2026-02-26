"""Entity resolution matcher for the Customer 360 platform.

Implements blocking, comparison, and classification of customer
record pairs using Jaro-Winkler, exact email, and phonetic matching.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import recordlinkage

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewCandidate:
    """A match pair flagged for manual review.

    Attributes:
        record_id_1: Index of the first record.
        record_id_2: Index of the second record.
        confidence_score: Overall match confidence (0–1).
        comparison_details: Per-field similarity scores.
    """

    record_id_1: int
    record_id_2: int
    confidence_score: float
    comparison_details: dict[str, float] = field(default_factory=dict)


class EntityMatcher:
    """Match customer records across sources using fuzzy comparison.

    Args:
        config: Resolution config section with thresholds and blocking
            field definitions.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.match_threshold = config.get("match_threshold", 0.85)
        self.uncertain_threshold = config.get("uncertain_threshold", 0.6)

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize fields for comparison.

        Args:
            df: Raw customer DataFrame.

        Returns:
            Copy of *df* with additional ``*_normalized`` columns.
        """
        df = df.copy()
        df["email_normalized"] = (
            df["email"].fillna("").str.lower().str.strip().str.replace(" ", "", regex=False)
        )
        df["phone_normalized"] = df["phone"].fillna("").str.replace(r"\D", "", regex=True)
        df["name_lower"] = df["name"].fillna("").str.lower().str.strip()
        df["first_letter"] = df["name_lower"].str[0].str.upper()
        return df

    def create_candidate_pairs(self, df: pd.DataFrame) -> pd.MultiIndex:
        """Create candidate pairs using blocking on first letter of name.

        Args:
            df: Normalized customer DataFrame (must have ``first_letter``).

        Returns:
            MultiIndex of candidate (left, right) index pairs.
        """
        indexer = recordlinkage.Index()
        indexer.block("first_letter")
        pairs = indexer.index(df)
        logger.info(
            "Blocking produced %d candidate pairs from %d records",
            len(pairs),
            len(df),
        )
        return pairs

    def compare_records(self, df: pd.DataFrame, candidate_pairs: pd.MultiIndex) -> pd.DataFrame:
        """Compare candidate record pairs using similarity metrics.

        Args:
            df: Normalized customer DataFrame.
            candidate_pairs: MultiIndex of pairs to compare.

        Returns:
            DataFrame with per-pair similarity scores.
        """
        compare = recordlinkage.Compare()
        compare.string("name_lower", "name_lower", method="jarowinkler", label="name_similarity")
        compare.exact("email_normalized", "email_normalized", label="email_match")
        compare.string(
            "phone_normalized",
            "phone_normalized",
            method="jarowinkler",
            label="phone_similarity",
        )
        features = compare.compute(candidate_pairs, df)
        logger.info("Compared %d candidate pairs", len(features))
        return features

    def classify_matches(
        self, features: pd.DataFrame
    ) -> tuple[pd.MultiIndex, pd.MultiIndex, pd.MultiIndex]:
        """Classify pairs into matches, uncertain, and non-matches.

        Args:
            features: DataFrame of per-pair similarity scores.

        Returns:
            Tuple of (matches, uncertain, non_matches) MultiIndex objects.
        """
        features = features.copy()

        # Weighted aggregate score
        features["score"] = (
            features["name_similarity"] * 0.4
            + features["email_match"] * 0.4
            + features["phone_similarity"] * 0.2
        )

        matches = features[features["score"] >= self.match_threshold].index
        uncertain = features[
            (features["score"] >= self.uncertain_threshold)
            & (features["score"] < self.match_threshold)
        ].index
        non_matches = features[features["score"] < self.uncertain_threshold].index

        logger.info(
            "Classification: %d matches, %d uncertain, %d non-matches",
            len(matches),
            len(uncertain),
            len(non_matches),
        )
        return matches, uncertain, non_matches

    def build_review_queue(
        self, uncertain: pd.MultiIndex, features: pd.DataFrame
    ) -> list[ReviewCandidate]:
        """Build a review queue from uncertain pairs.

        Args:
            uncertain: MultiIndex of uncertain pairs.
            features: Full features DataFrame with scores.

        Returns:
            List of :class:`ReviewCandidate` objects.
        """
        queue: list[ReviewCandidate] = []
        for left, right in uncertain:
            row = features.loc[(left, right)]
            queue.append(
                ReviewCandidate(
                    record_id_1=left,
                    record_id_2=right,
                    confidence_score=float(row["score"]),
                    comparison_details={
                        "name_similarity": float(row["name_similarity"]),
                        "email_match": float(row["email_match"]),
                        "phone_similarity": float(row["phone_similarity"]),
                    },
                )
            )
        return queue

    def run(
        self, df: pd.DataFrame
    ) -> tuple[pd.MultiIndex, pd.MultiIndex, pd.MultiIndex, pd.DataFrame]:
        """Execute the full matching pipeline.

        Args:
            df: Raw customer DataFrame.

        Returns:
            Tuple of (matches, uncertain, non_matches, features).
        """
        normalized = self.normalize_data(df)
        pairs = self.create_candidate_pairs(normalized)
        features = self.compare_records(normalized, pairs)
        matches, uncertain, non_matches = self.classify_matches(features)
        return matches, uncertain, non_matches, features
