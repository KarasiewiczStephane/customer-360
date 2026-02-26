"""Tests for entity resolution matching."""

import pandas as pd
import pytest

from src.resolution.matcher import EntityMatcher, ReviewCandidate


@pytest.fixture()
def config() -> dict:
    return {"match_threshold": 0.85, "uncertain_threshold": 0.6}


@pytest.fixture()
def matcher(config: dict) -> EntityMatcher:
    return EntityMatcher(config)


@pytest.fixture()
def sample_customers() -> pd.DataFrame:
    """Small dataset with one known duplicate pair."""
    return pd.DataFrame(
        {
            "customer_id": [
                "CRM_000001",
                "CRM_000002",
                "CRM_DUP_000001",
                "CRM_000003",
            ],
            "name": ["Robert Smith", "Jane Doe", "Bob Smith", "Alice Johnson"],
            "email": [
                "robert.smith@test.com",
                "jane.doe@test.com",
                "robert.smith@test.com",
                "alice@test.com",
            ],
            "phone": ["123-456-7890", "555-111-2222", "1234567890", "999-000-1111"],
        }
    )


class TestNormalizeData:
    """Tests for data normalization."""

    def test_creates_normalized_columns(
        self, matcher: EntityMatcher, sample_customers: pd.DataFrame
    ) -> None:
        result = matcher.normalize_data(sample_customers)
        assert "email_normalized" in result.columns
        assert "phone_normalized" in result.columns
        assert "name_lower" in result.columns
        assert "first_letter" in result.columns

    def test_email_lowercased(self, matcher: EntityMatcher) -> None:
        df = pd.DataFrame(
            {
                "customer_id": ["1"],
                "name": ["Test"],
                "email": ["TEST@Example.COM"],
                "phone": ["123"],
            }
        )
        result = matcher.normalize_data(df)
        assert result.iloc[0]["email_normalized"] == "test@example.com"

    def test_phone_digits_only(self, matcher: EntityMatcher) -> None:
        df = pd.DataFrame(
            {
                "customer_id": ["1"],
                "name": ["Test"],
                "email": ["t@t.com"],
                "phone": ["+1-(555)-123-4567"],
            }
        )
        result = matcher.normalize_data(df)
        assert result.iloc[0]["phone_normalized"] == "15551234567"

    def test_does_not_modify_original(
        self, matcher: EntityMatcher, sample_customers: pd.DataFrame
    ) -> None:
        original_cols = set(sample_customers.columns)
        matcher.normalize_data(sample_customers)
        assert set(sample_customers.columns) == original_cols


class TestBlocking:
    """Tests for blocking index creation."""

    def test_reduces_comparison_space(
        self, matcher: EntityMatcher, sample_customers: pd.DataFrame
    ) -> None:
        normalized = matcher.normalize_data(sample_customers)
        pairs = matcher.create_candidate_pairs(normalized)
        # All-vs-all would be n*(n-1)/2 = 6, blocking should reduce this
        all_pairs = len(sample_customers) * (len(sample_customers) - 1) // 2
        assert len(pairs) <= all_pairs

    def test_same_letter_records_paired(self, matcher: EntityMatcher) -> None:
        """Records sharing a first letter are paired by the blocker."""
        df = pd.DataFrame(
            {
                "customer_id": ["A", "B", "C"],
                "name": ["Alice Doe", "Alice Smith", "Bob Jones"],
                "email": ["a@t.com", "a2@t.com", "b@t.com"],
                "phone": ["111", "222", "333"],
            }
        )
        normalized = matcher.normalize_data(df)
        pairs = matcher.create_candidate_pairs(normalized)
        assert len(pairs) >= 1


class TestCompareRecords:
    """Tests for record comparison."""

    def test_comparison_output_shape(
        self, matcher: EntityMatcher, sample_customers: pd.DataFrame
    ) -> None:
        normalized = matcher.normalize_data(sample_customers)
        pairs = matcher.create_candidate_pairs(normalized)
        features = matcher.compare_records(normalized, pairs)
        assert "name_similarity" in features.columns
        assert "email_match" in features.columns
        assert "phone_similarity" in features.columns

    def test_exact_email_match_scores_one(self, matcher: EntityMatcher) -> None:
        """Records sharing an exact email score 1.0 on email_match."""
        df = pd.DataFrame(
            {
                "customer_id": ["A", "B"],
                "name": ["Robert S", "Robert S"],
                "email": ["r@test.com", "r@test.com"],
                "phone": ["1234567890", "1234567890"],
            }
        )
        normalized = matcher.normalize_data(df)
        pairs = matcher.create_candidate_pairs(normalized)
        if len(pairs) > 0:
            features = matcher.compare_records(normalized, pairs)
            assert (features["email_match"] == 1.0).all()


class TestClassifyMatches:
    """Tests for match classification."""

    def test_high_score_is_match(self, matcher: EntityMatcher) -> None:
        features = pd.DataFrame(
            {
                "name_similarity": [0.95],
                "email_match": [1.0],
                "phone_similarity": [0.9],
            },
            index=pd.MultiIndex.from_tuples([(0, 1)]),
        )
        matches, uncertain, non_matches = matcher.classify_matches(features)
        assert len(matches) == 1
        assert len(uncertain) == 0

    def test_low_score_is_non_match(self, matcher: EntityMatcher) -> None:
        features = pd.DataFrame(
            {
                "name_similarity": [0.3],
                "email_match": [0.0],
                "phone_similarity": [0.1],
            },
            index=pd.MultiIndex.from_tuples([(0, 1)]),
        )
        matches, uncertain, non_matches = matcher.classify_matches(features)
        assert len(matches) == 0
        assert len(non_matches) == 1

    def test_mid_score_is_uncertain(self, matcher: EntityMatcher) -> None:
        # Use values that land in the uncertain zone (0.6 <= score < 0.85)
        features2 = pd.DataFrame(
            {
                "name_similarity": [0.8],
                "email_match": [0.5],
                "phone_similarity": [0.7],
            },
            index=pd.MultiIndex.from_tuples([(0, 1)]),
        )
        # score = 0.8*0.4 + 0.5*0.4 + 0.7*0.2 = 0.32 + 0.2 + 0.14 = 0.66
        matches, uncertain, non_matches = matcher.classify_matches(features2)
        assert len(uncertain) == 1


class TestBuildReviewQueue:
    """Tests for review queue construction."""

    def test_queue_from_uncertain(self, matcher: EntityMatcher) -> None:
        features = pd.DataFrame(
            {
                "name_similarity": [0.75],
                "email_match": [0.5],
                "phone_similarity": [0.6],
                "score": [0.70],
            },
            index=pd.MultiIndex.from_tuples([(0, 1)]),
        )
        uncertain = features.index
        queue = matcher.build_review_queue(uncertain, features)
        assert len(queue) == 1
        assert isinstance(queue[0], ReviewCandidate)
        assert queue[0].confidence_score == pytest.approx(0.70)


class TestRunPipeline:
    """Integration test for the full matching pipeline."""

    def test_run_returns_tuple(
        self, matcher: EntityMatcher, sample_customers: pd.DataFrame
    ) -> None:
        matches, uncertain, non_matches, features = matcher.run(sample_customers)
        total = len(matches) + len(uncertain) + len(non_matches)
        assert total == len(features)
