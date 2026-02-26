"""Golden record creation by merging matched customer records.

Uses connected-component clustering to group matching pairs,
then selects the best field value from each cluster to form
a single unified customer record.
"""

from typing import Any

import networkx as nx
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RecordMerger:
    """Merge matched customer records into golden records.

    Args:
        priority_order: Source priority for conflict resolution.
    """

    def __init__(self, priority_order: list[str] | None = None) -> None:
        self.priority_order = priority_order or [
            "crm",
            "transactions",
            "web",
            "support",
        ]

    def build_match_clusters(self, matches: pd.MultiIndex) -> list[set[int]]:
        """Group matching pairs into clusters via connected components.

        Args:
            matches: MultiIndex of matched (left, right) index pairs.

        Returns:
            List of sets, each containing record indices that belong
            to the same entity.
        """
        if len(matches) == 0:
            return []

        graph = nx.Graph()
        graph.add_edges_from(matches.tolist())
        clusters = [set(c) for c in nx.connected_components(graph)]
        logger.info(
            "Built %d match clusters from %d pairs",
            len(clusters),
            len(matches),
        )
        return clusters

    def create_golden_record(self, cluster: set[int], df: pd.DataFrame) -> dict[str, Any]:
        """Create a unified golden record from a cluster.

        Args:
            cluster: Set of DataFrame indices belonging to one entity.
            df: Customer DataFrame.

        Returns:
            Dict representing the golden record.
        """
        records = df.loc[list(cluster)]
        unified_id = f"GOLD_{min(cluster):06d}"

        return {
            "unified_id": unified_id,
            "source_ids": ",".join(records["customer_id"].astype(str).tolist()),
            "name": self._best_name(records),
            "email": self._best_email(records),
            "phone": self._best_phone(records),
        }

    def merge_all(self, clusters: list[set[int]], df: pd.DataFrame) -> pd.DataFrame:
        """Create golden records for all clusters and unclustered records.

        Args:
            clusters: List of matched-record clusters.
            df: Full customer DataFrame.

        Returns:
            DataFrame of golden records.
        """
        golden_records: list[dict[str, Any]] = []
        clustered_indices: set[int] = set()

        for cluster in clusters:
            golden_records.append(self.create_golden_record(cluster, df))
            clustered_indices.update(cluster)

        # Add unclustered records as their own golden records
        for idx in df.index:
            if idx not in clustered_indices:
                row = df.loc[idx]
                golden_records.append(
                    {
                        "unified_id": f"GOLD_{idx:06d}",
                        "source_ids": str(row["customer_id"]),
                        "name": row.get("name"),
                        "email": row.get("email"),
                        "phone": row.get("phone"),
                    }
                )

        result = pd.DataFrame(golden_records)
        logger.info(
            "Created %d golden records (%d from clusters, %d singletons)",
            len(result),
            len(clusters),
            len(result) - len(clusters),
        )
        return result

    # ------------------------------------------------------------------
    # Best-value selection strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _best_name(records: pd.DataFrame) -> str | None:
        """Select the longest non-null name (most complete)."""
        names = records["name"].dropna()
        if names.empty:
            return None
        return names.loc[names.str.len().idxmax()]

    @staticmethod
    def _best_email(records: pd.DataFrame) -> str | None:
        """Select the first valid (contains @) email."""
        emails = records["email"].dropna()
        valid = emails[emails.str.contains("@", na=False)]
        if not valid.empty:
            return valid.iloc[0]
        return emails.iloc[0] if not emails.empty else None

    @staticmethod
    def _best_phone(records: pd.DataFrame) -> str | None:
        """Select the phone with the most digits."""
        phones = records["phone"].dropna()
        if phones.empty:
            return None
        digit_counts = phones.str.replace(r"\D", "", regex=True).str.len()
        return phones.loc[digit_counts.idxmax()]
