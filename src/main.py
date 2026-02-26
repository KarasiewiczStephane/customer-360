"""Entry point for the Customer 360 analytics pipeline.

Orchestrates data generation, entity resolution, analytics, and
stores results in DuckDB for dashboard consumption.
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the Customer 360 analytics pipeline."""
    logger.info("Customer 360 pipeline — ready for modules to be registered")


if __name__ == "__main__":
    main()
