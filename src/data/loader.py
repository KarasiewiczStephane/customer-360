"""Data loading utilities for DuckDB.

Handles bulk-loading DataFrames into DuckDB tables and retrieving
them back for downstream analytics.
"""

from pathlib import Path

import duckdb
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Load and retrieve DataFrames from a DuckDB database.

    Args:
        db_path: File-system path to the DuckDB database file.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def load_all_data(self, data: dict[str, pd.DataFrame]) -> None:
        """Load multiple DataFrames into DuckDB, replacing existing tables.

        Args:
            data: Mapping of ``table_name`` to ``DataFrame``.
        """
        with duckdb.connect(self.db_path) as conn:
            for table_name, df in data.items():
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
                logger.info("Loaded %d records into %s", len(df), table_name)

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Retrieve a full table as a DataFrame.

        Args:
            table_name: Name of the DuckDB table.

        Returns:
            DataFrame containing all rows from the table.
        """
        with duckdb.connect(self.db_path) as conn:
            return conn.execute(f"SELECT * FROM {table_name}").df()

    def table_exists(self, table_name: str) -> bool:
        """Check whether a table exists in the database.

        Args:
            table_name: Table name to check.

        Returns:
            ``True`` if the table exists, ``False`` otherwise.
        """
        with duckdb.connect(self.db_path) as conn:
            tables = conn.execute("SHOW TABLES").df()
            return table_name in tables["name"].values
