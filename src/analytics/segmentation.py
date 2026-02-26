"""K-Means customer segmentation with feature engineering.

Combines RFM scores, web engagement, and support metrics into
a unified feature matrix, finds the optimal cluster count, and
produces PCA projections for visualisation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerSegmentation:
    """K-Means segmentation over multi-source customer features.

    Args:
        max_clusters: Upper bound for the elbow-method search.
        random_state: Seed for reproducibility.
    """

    def __init__(self, max_clusters: int = 10, random_state: int = 42) -> None:
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.pca: PCA | None = None
        self.optimal_k: int | None = None

    def prepare_features(
        self,
        rfm: pd.DataFrame,
        web_sessions: pd.DataFrame,
        support_tickets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Combine data sources into a single feature matrix.

        Args:
            rfm: RFM-scored DataFrame (must include ``customer_id``,
                ``r_score``, ``f_score``, ``m_score``, ``monetary``).
            web_sessions: Web session records.
            support_tickets: Support ticket records.

        Returns:
            Feature DataFrame indexed by ``customer_id``.
        """
        features = rfm[["customer_id", "r_score", "f_score", "m_score", "monetary"]].copy()

        # Web engagement aggregates
        web_agg = (
            web_sessions.groupby("customer_id")
            .agg(
                web_sessions=("session_id", "count"),
                avg_pages=("pages_visited", "mean"),
                avg_time=("time_on_site", "mean"),
            )
            .reset_index()
        )
        features = features.merge(web_agg, on="customer_id", how="left")

        # Support aggregates
        support_agg = (
            support_tickets.groupby("customer_id")
            .agg(
                ticket_count=("ticket_id", "count"),
                avg_satisfaction=("satisfaction_score", "mean"),
                avg_resolution_time=("resolution_time_hours", "mean"),
            )
            .reset_index()
        )
        features = features.merge(support_agg, on="customer_id", how="left")
        features = features.fillna(0)

        logger.info(
            "Feature matrix: %d customers x %d features",
            len(features),
            len(features.columns) - 1,
        )
        return features

    def find_optimal_k(self, X: np.ndarray) -> tuple[int, dict[int, float], dict[int, float]]:
        """Find the optimal cluster count using elbow + silhouette.

        Args:
            X: Scaled feature array.

        Returns:
            Tuple of (optimal_k, inertias_dict, silhouettes_dict).
        """
        inertias: dict[int, float] = {}
        silhouettes: dict[int, float] = {}

        max_k = min(self.max_clusters, len(X) - 1)
        if max_k < 2:
            self.optimal_k = 2
            return 2, {}, {}

        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            inertias[k] = km.inertia_
            silhouettes[k] = silhouette_score(X, labels)

        best_silhouette_k = max(silhouettes, key=silhouettes.get)  # type: ignore[arg-type]

        # Elbow via second derivative
        k_vals = sorted(inertias.keys())
        inertia_vals = [inertias[k] for k in k_vals]
        if len(inertia_vals) >= 3:
            diffs = np.diff(inertia_vals)
            second_diffs = np.diff(diffs)
            elbow_idx = int(np.argmax(second_diffs)) + 2
        else:
            elbow_idx = k_vals[0]

        self.optimal_k = best_silhouette_k if abs(best_silhouette_k - elbow_idx) <= 2 else elbow_idx
        logger.info(
            "Optimal K=%d (elbow=%d, best silhouette=%d)",
            self.optimal_k,
            elbow_idx,
            best_silhouette_k,
        )
        return self.optimal_k, inertias, silhouettes

    def fit_clusters(self, features: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
        """Fit K-Means and assign cluster labels.

        Args:
            features: Feature DataFrame (from :meth:`prepare_features`).
            k: Number of clusters.  Auto-detected if *None*.

        Returns:
            Features DataFrame with a ``cluster`` column.
        """
        feature_cols = [c for c in features.columns if c != "customer_id"]
        X = features[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        if k is None:
            k, _, _ = self.find_optimal_k(X_scaled)

        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        features = features.copy()
        features["cluster"] = self.kmeans.fit_predict(X_scaled)
        logger.info("Fit K-Means with %d clusters", k)
        return features

    def get_cluster_profiles(self, features: pd.DataFrame) -> pd.DataFrame:
        """Compute average metrics per cluster.

        Args:
            features: DataFrame with ``cluster`` and feature columns.

        Returns:
            Profile DataFrame indexed by cluster.
        """
        feature_cols = [c for c in features.columns if c not in ("customer_id", "cluster")]
        profiles = features.groupby("cluster")[feature_cols].mean().round(2)
        profiles["size"] = features.groupby("cluster").size()
        profiles["pct_of_total"] = (profiles["size"] / len(features) * 100).round(1)
        return profiles

    def get_pca_projection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Project features to 2-D via PCA for visualisation.

        Args:
            features: Clustered feature DataFrame.

        Returns:
            DataFrame with ``customer_id``, ``cluster``, ``pca_x``,
            ``pca_y``.
        """
        feature_cols = [c for c in features.columns if c not in ("customer_id", "cluster")]
        X = features[feature_cols].values
        X_scaled = self.scaler.transform(X)

        self.pca = PCA(n_components=2, random_state=self.random_state)
        coords = self.pca.fit_transform(X_scaled)

        result = features[["customer_id", "cluster"]].copy()
        result["pca_x"] = coords[:, 0]
        result["pca_y"] = coords[:, 1]

        explained = self.pca.explained_variance_ratio_.sum()
        logger.info("PCA explains %.1f%% of variance", explained * 100)
        return result
