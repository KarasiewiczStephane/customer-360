"""Tests for DuckDB connection management."""

from pathlib import Path

import pytest

from src.utils.database import get_connection


class TestGetConnection:
    """Tests for the get_connection context manager."""

    def test_connection_opens_and_closes(self, tmp_path: Path) -> None:
        """Connection opens within context and closes on exit."""
        db_path = str(tmp_path / "test.duckdb")
        with get_connection(db_path) as conn:
            assert conn is not None
            result = conn.execute("SELECT 1 AS val").fetchone()
            assert result[0] == 1

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directories are created if they do not exist."""
        db_path = str(tmp_path / "subdir" / "nested" / "test.duckdb")
        with get_connection(db_path) as conn:
            conn.execute("CREATE TABLE t (id INT)")
        assert Path(db_path).exists()

    def test_table_creation_persists(self, tmp_path: Path) -> None:
        """Tables created inside the context persist on disk."""
        db_path = str(tmp_path / "persist.duckdb")

        with get_connection(db_path) as conn:
            conn.execute("CREATE TABLE demo (id INT, name VARCHAR)")
            conn.execute("INSERT INTO demo VALUES (1, 'Alice')")

        with get_connection(db_path) as conn:
            row = conn.execute("SELECT * FROM demo").fetchone()
            assert row == (1, "Alice")

    def test_connection_cleanup_on_exception(self, tmp_path: Path) -> None:
        """Connection is closed even when an exception occurs."""
        db_path = str(tmp_path / "error.duckdb")
        with pytest.raises(RuntimeError):
            with get_connection(db_path) as conn:
                conn.execute("SELECT 1")
                raise RuntimeError("intentional")

        # Should be able to reconnect after the error
        with get_connection(db_path) as conn:
            assert conn.execute("SELECT 42").fetchone()[0] == 42
