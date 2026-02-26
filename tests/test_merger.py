"""Tests for golden record creation and match reporting."""

from pathlib import Path

import pandas as pd
import pytest

from src.resolution.merger import RecordMerger
from src.resolution.quality import MatchReport, export_review_queue, generate_match_report


@pytest.fixture()
def merger() -> RecordMerger:
    return RecordMerger()


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["CRM_001", "CRM_002", "CRM_DUP_001", "CRM_003"],
            "name": ["Robert Smith", "Jane Doe", "Bob Smith", "Alice Johnson"],
            "email": [
                "robert@test.com",
                "jane@test.com",
                "robert@test.com",
                "alice@test.com",
            ],
            "phone": [
                "+1-123-456-7890",
                "555-111-2222",
                "1234567890",
                "999-000-1111",
            ],
        }
    )


class TestBuildMatchClusters:
    """Tests for cluster building."""

    def test_single_pair(self, merger: RecordMerger) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 2)])
        clusters = merger.build_match_clusters(pairs)
        assert len(clusters) == 1
        assert {0, 2} in clusters

    def test_transitive_closure(self, merger: RecordMerger) -> None:
        """A-B and B-C → all three in the same cluster."""
        pairs = pd.MultiIndex.from_tuples([(0, 1), (1, 2)])
        clusters = merger.build_match_clusters(pairs)
        assert len(clusters) == 1
        assert clusters[0] == {0, 1, 2}

    def test_separate_clusters(self, merger: RecordMerger) -> None:
        pairs = pd.MultiIndex.from_tuples([(0, 1), (2, 3)])
        clusters = merger.build_match_clusters(pairs)
        assert len(clusters) == 2

    def test_empty_matches(self, merger: RecordMerger) -> None:
        pairs = pd.MultiIndex.from_tuples([], names=[None, None])
        clusters = merger.build_match_clusters(pairs)
        assert clusters == []


class TestCreateGoldenRecord:
    """Tests for golden record creation."""

    def test_golden_record_fields(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        golden = merger.create_golden_record({0, 2}, sample_df)
        assert golden["unified_id"] == "GOLD_000000"
        assert "CRM_001" in golden["source_ids"]
        assert "CRM_DUP_001" in golden["source_ids"]

    def test_best_name_longest(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        golden = merger.create_golden_record({0, 2}, sample_df)
        assert golden["name"] == "Robert Smith"  # longer than "Bob Smith"

    def test_best_email_valid(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        golden = merger.create_golden_record({0, 2}, sample_df)
        assert "@" in golden["email"]

    def test_best_phone_most_digits(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        golden = merger.create_golden_record({0, 2}, sample_df)
        # +1-123-456-7890 has 11 digits, 1234567890 has 10
        assert golden["phone"] == "+1-123-456-7890"


class TestMergeAll:
    """Tests for merging all clusters."""

    def test_includes_singletons(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        clusters = [{0, 2}]
        result = merger.merge_all(clusters, sample_df)
        # 1 cluster + 2 singletons (indices 1, 3) = 3 golden records
        assert len(result) == 3

    def test_all_records_covered(self, merger: RecordMerger, sample_df: pd.DataFrame) -> None:
        clusters = [{0, 2}]
        result = merger.merge_all(clusters, sample_df)
        all_source_ids = ",".join(result["source_ids"].tolist())
        for cid in sample_df["customer_id"]:
            assert cid in all_source_ids


class TestGenerateMatchReport:
    """Tests for match report generation."""

    def test_report_fields(self) -> None:
        matches = pd.MultiIndex.from_tuples([(0, 1)])
        uncertain = pd.MultiIndex.from_tuples([(2, 3)])
        clusters = [{0, 1}]
        features = pd.DataFrame(
            {"score": [0.9, 0.65]},
            index=pd.MultiIndex.from_tuples([(0, 1), (2, 3)]),
        )
        report = generate_match_report(10, matches, uncertain, clusters, features)

        assert isinstance(report, MatchReport)
        assert report.total_records == 10
        assert report.total_matches == 1
        assert report.total_clusters == 1
        assert report.uncertain_count == 1
        assert report.match_rate == 0.2  # 2 of 10 records matched
        assert report.unmatched_count == 8

    def test_empty_input(self) -> None:
        empty = pd.MultiIndex.from_tuples([], names=[None, None])
        features = pd.DataFrame({"score": []})
        report = generate_match_report(0, empty, empty, [], features)
        assert report.total_matches == 0
        assert report.match_rate == 0.0


class TestExportReviewQueue:
    """Tests for review queue export."""

    def test_csv_written(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "name": ["Alice", "Bob"],
            }
        )
        features = pd.DataFrame(
            {
                "score": [0.7],
                "name_similarity": [0.8],
                "email_match": [0.0],
                "phone_similarity": [0.6],
            },
            index=pd.MultiIndex.from_tuples([(0, 1)]),
        )
        uncertain = pd.MultiIndex.from_tuples([(0, 1)])
        out = str(tmp_path / "review.csv")
        export_review_queue(uncertain, features, df, out)

        result = pd.read_csv(out)
        assert len(result) == 1
        assert "left_name" in result.columns
        assert result.iloc[0]["score"] == pytest.approx(0.7)
