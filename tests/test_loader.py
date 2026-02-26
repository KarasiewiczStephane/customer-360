"""Tests for the DuckDB data loader."""

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import DataLoader


@pytest.fixture()
def loader(tmp_path: Path) -> DataLoader:
    return DataLoader(str(tmp_path / "test.duckdb"))


@pytest.fixture()
def sample_data() -> dict[str, pd.DataFrame]:
    return {
        "users": pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}),
        "orders": pd.DataFrame({"order_id": [10, 20], "user_id": [1, 2], "amount": [99.9, 49.5]}),
    }


class TestDataLoader:
    """Tests for DataLoader."""

    def test_load_and_retrieve(
        self, loader: DataLoader, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        loader.load_all_data(sample_data)
        users = loader.get_table("users")
        assert len(users) == 3
        assert set(users.columns) == {"id", "name"}

    def test_load_replaces_existing(
        self, loader: DataLoader, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        """Loading the same table twice replaces the old data."""
        loader.load_all_data(sample_data)
        new_data = {"users": pd.DataFrame({"id": [99], "name": ["Zara"]})}
        loader.load_all_data(new_data)
        users = loader.get_table("users")
        assert len(users) == 1
        assert users.iloc[0]["name"] == "Zara"

    def test_table_exists(self, loader: DataLoader, sample_data: dict[str, pd.DataFrame]) -> None:
        loader.load_all_data(sample_data)
        assert loader.table_exists("users") is True
        assert loader.table_exists("nonexistent") is False

    def test_multiple_tables(
        self, loader: DataLoader, sample_data: dict[str, pd.DataFrame]
    ) -> None:
        loader.load_all_data(sample_data)
        assert loader.table_exists("users")
        assert loader.table_exists("orders")
        orders = loader.get_table("orders")
        assert len(orders) == 2
