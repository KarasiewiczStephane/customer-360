"""Tests for dashboard helper functions.

Streamlit rendering is not tested directly; we test the query
helpers and data-formatting logic.
"""

import duckdb
import pandas as pd
import pytest

from src.dashboard.app import _safe_query


@pytest.fixture()
def conn(tmp_path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with sample tables."""
    db = duckdb.connect(str(tmp_path / "dash_test.duckdb"))
    db.execute(
        """
        CREATE TABLE golden_records AS SELECT * FROM (
            VALUES ('GOLD_000001', 'CRM_001', 'Alice Smith', 'alice@test.com', '123')
        ) t(unified_id, source_ids, name, email, phone)
        """
    )
    db.execute(
        """
        CREATE TABLE rfm_scores AS SELECT * FROM (
            VALUES ('GOLD_000001', 5, 5, 5, '555', 'Champions', 100, 20, 2000.0)
        ) t(customer_id, r_score, f_score, m_score, rfm_score, segment, recency, frequency, monetary)
        """
    )
    db.execute(
        """
        CREATE TABLE clv_predictions AS SELECT * FROM (
            VALUES ('GOLD_000001', 1500.0, 'High Value')
        ) t(customer_id, predicted_clv, clv_tier)
        """
    )
    return db


class TestSafeQuery:
    def test_valid_query(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(conn, "SELECT * FROM golden_records")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice Smith"

    def test_parameterized_query(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(
            conn,
            "SELECT * FROM golden_records WHERE unified_id = ?",
            ["GOLD_000001"],
        )
        assert len(result) == 1

    def test_bad_query_returns_empty(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(conn, "SELECT * FROM nonexistent_table")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_search_like(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(
            conn,
            "SELECT * FROM golden_records WHERE LOWER(name) LIKE LOWER(?)",
            ["%alice%"],
        )
        assert len(result) == 1

    def test_empty_params_uses_no_params_path(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(conn, "SELECT * FROM golden_records", params=[])
        assert len(result) == 1

    def test_none_params_uses_no_params_path(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(conn, "SELECT * FROM golden_records", params=None)
        assert len(result) == 1

    def test_rfm_query(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(
            conn,
            "SELECT segment, COUNT(*) AS customer_count, AVG(monetary) AS avg_monetary "
            "FROM rfm_scores GROUP BY segment",
        )
        assert len(result) == 1
        assert result.iloc[0]["segment"] == "Champions"

    def test_clv_query(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(conn, "SELECT predicted_clv, clv_tier FROM clv_predictions")
        assert len(result) == 1
        assert result.iloc[0]["clv_tier"] == "High Value"

    def test_join_query(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = _safe_query(
            conn,
            "SELECT g.unified_id, g.name, r.segment, c.predicted_clv "
            "FROM golden_records g "
            "LEFT JOIN rfm_scores r ON g.unified_id = r.customer_id "
            "LEFT JOIN clv_predictions c ON g.unified_id = c.customer_id "
            "WHERE g.unified_id = ?",
            ["GOLD_000001"],
        )
        assert len(result) == 1
        assert result.iloc[0]["name"] == "Alice Smith"
        assert result.iloc[0]["segment"] == "Champions"
