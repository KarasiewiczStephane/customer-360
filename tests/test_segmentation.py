"""Tests for K-Means customer segmentation."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.segmentation import CustomerSegmentation


@pytest.fixture()
def segmenter() -> CustomerSegmentation:
    return CustomerSegmentation(max_clusters=5, random_state=42)


@pytest.fixture()
def rfm_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "customer_id": [f"C_{i}" for i in range(n)],
            "r_score": np.random.randint(1, 6, n),
            "f_score": np.random.randint(1, 6, n),
            "m_score": np.random.randint(1, 6, n),
            "monetary": np.random.uniform(10, 5000, n).round(2),
        }
    )


@pytest.fixture()
def web_data() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(100):
        rows.append(
            {
                "session_id": f"S_{i}",
                "customer_id": f"C_{i % 50}",
                "pages_visited": np.random.randint(1, 20),
                "time_on_site": np.random.randint(10, 1800),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def support_data() -> pd.DataFrame:
    np.random.seed(42)
    rows = []
    for i in range(30):
        rows.append(
            {
                "ticket_id": f"T_{i}",
                "customer_id": f"C_{i % 50}",
                "satisfaction_score": np.random.randint(1, 6),
                "resolution_time_hours": round(np.random.exponential(24), 2),
            }
        )
    return pd.DataFrame(rows)


class TestPrepareFeatures:
    def test_all_columns_present(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        expected = {
            "customer_id",
            "r_score",
            "f_score",
            "m_score",
            "monetary",
            "web_sessions",
            "avg_pages",
            "avg_time",
            "ticket_count",
            "avg_satisfaction",
            "avg_resolution_time",
        }
        assert expected.issubset(set(features.columns))

    def test_no_nan_values(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        assert features.isna().sum().sum() == 0

    def test_row_count_matches_rfm(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        assert len(features) == len(rfm_data)


class TestFindOptimalK:
    def test_k_in_range(self, segmenter: CustomerSegmentation) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 5)
        k, inertias, silhouettes = segmenter.find_optimal_k(X)
        assert 2 <= k <= segmenter.max_clusters


class TestFitClusters:
    def test_cluster_labels_assigned(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        result = segmenter.fit_clusters(features, k=3)
        assert "cluster" in result.columns
        assert result["cluster"].nunique() == 3

    def test_deterministic(
        self,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        seg1 = CustomerSegmentation(random_state=42)
        seg2 = CustomerSegmentation(random_state=42)
        f1 = seg1.prepare_features(rfm_data, web_data, support_data)
        f2 = seg2.prepare_features(rfm_data, web_data, support_data)
        r1 = seg1.fit_clusters(f1, k=3)
        r2 = seg2.fit_clusters(f2, k=3)
        pd.testing.assert_series_equal(r1["cluster"], r2["cluster"])


class TestClusterProfiles:
    def test_size_sums_to_total(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        clustered = segmenter.fit_clusters(features, k=3)
        profiles = segmenter.get_cluster_profiles(clustered)
        assert profiles["size"].sum() == len(rfm_data)

    def test_pct_sums_to_100(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        clustered = segmenter.fit_clusters(features, k=3)
        profiles = segmenter.get_cluster_profiles(clustered)
        assert abs(profiles["pct_of_total"].sum() - 100.0) < 1.0


class TestPCAProjection:
    def test_output_shape(
        self,
        segmenter: CustomerSegmentation,
        rfm_data: pd.DataFrame,
        web_data: pd.DataFrame,
        support_data: pd.DataFrame,
    ) -> None:
        features = segmenter.prepare_features(rfm_data, web_data, support_data)
        clustered = segmenter.fit_clusters(features, k=3)
        pca = segmenter.get_pca_projection(clustered)
        assert "pca_x" in pca.columns
        assert "pca_y" in pca.columns
        assert len(pca) == len(rfm_data)
