"""Tests for web session and support ticket generation."""

import pytest

from src.data.generator import SyntheticDataGenerator
from src.utils.config import load_config


@pytest.fixture()
def generator() -> SyntheticDataGenerator:
    config = load_config()
    return SyntheticDataGenerator(config, seed=42)


@pytest.fixture()
def customer_ids() -> list[str]:
    return [f"CRM_{i:06d}" for i in range(100)]


class TestGenerateWebSessions:
    """Tests for web session generation."""

    def test_session_count(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_web_sessions(customer_ids, n=500)
        assert len(df) == 500

    def test_required_columns(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_web_sessions(customer_ids, n=50)
        expected = {
            "session_id",
            "customer_id",
            "timestamp",
            "pages_visited",
            "time_on_site",
            "referrer",
            "pages_path",
        }
        assert expected.issubset(set(df.columns))

    def test_anonymous_sessions(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        """Approximately 10 % of sessions are anonymous."""
        df = generator.generate_web_sessions(customer_ids, n=2000)
        anon_rate = df["customer_id"].isna().mean()
        assert 0.04 <= anon_rate <= 0.18

    def test_pages_visited_range(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_web_sessions(customer_ids, n=200)
        assert (df["pages_visited"] >= 1).all()
        assert (df["pages_visited"] <= 20).all()

    def test_time_on_site_range(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_web_sessions(customer_ids, n=200)
        assert (df["time_on_site"] >= 10).all()
        assert (df["time_on_site"] <= 1800).all()

    def test_session_ids_unique(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_web_sessions(customer_ids, n=500)
        assert df["session_id"].is_unique


class TestGenerateSupportTickets:
    """Tests for support ticket generation."""

    def test_ticket_count(self, generator: SyntheticDataGenerator, customer_ids: list[str]) -> None:
        df = generator.generate_support_tickets(customer_ids, n=300)
        assert len(df) == 300

    def test_required_columns(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_support_tickets(customer_ids, n=50)
        expected = {
            "ticket_id",
            "customer_id",
            "created_at",
            "category",
            "resolution_time_hours",
            "satisfaction_score",
            "status",
        }
        assert expected.issubset(set(df.columns))

    def test_satisfaction_range(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_support_tickets(customer_ids, n=500)
        assert df["satisfaction_score"].isin([1, 2, 3, 4, 5]).all()

    def test_resolution_time_positive(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_support_tickets(customer_ids, n=200)
        assert (df["resolution_time_hours"] > 0).all()

    def test_categories_valid(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_support_tickets(customer_ids, n=200)
        assert df["category"].isin(SyntheticDataGenerator.TICKET_CATEGORIES).all()

    def test_status_values(
        self, generator: SyntheticDataGenerator, customer_ids: list[str]
    ) -> None:
        df = generator.generate_support_tickets(customer_ids, n=500)
        assert df["status"].isin(["resolved", "open", "pending"]).all()
