"""DuckDB connection management for the Customer 360 platform.

Provides a context-managed connection helper that ensures proper
resource cleanup after each database interaction.
"""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from src.utils.logger import get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection(
    db_path: str = "data/customer360.duckdb",
) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Yield a DuckDB connection, closing it on exit.

    Args:
        db_path: File-system path for the DuckDB database.

    Yields:
        An open :class:`duckdb.DuckDBPyConnection`.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
