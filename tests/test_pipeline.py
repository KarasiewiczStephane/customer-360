"""Tests for the end-to-end pipeline orchestration."""

from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

from src.main import _flatten_retention_matrix, main, run_pipeline


@pytest.fixture()
def config_file(tmp_path):
    """Create a minimal config pointing to a temp database."""
    db_path = str(tmp_path / "test.duckdb")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""\
data:
  num_customers: 50
  num_transactions: 200
  num_web_sessions: 100
  num_support_tickets: 30
  sample_customers: 20
  seed: 42

database:
  path: "{db_path}"

resolution:
  match_threshold: 0.85
  uncertain_threshold: 0.6

analytics:
  rfm_segments: 5
  kmeans_max_clusters: 5
  clv_prediction_months: 12
  clv_discount_rate: 0.01
"""
    )
    return str(config), db_path


class TestRunPipeline:
    def test_full_pipeline(self, config_file: tuple[str, str]) -> None:
        """Pipeline runs end-to-end and produces expected tables."""
        config_path, db_path = config_file
        summary = run_pipeline(config_path=config_path, sample=False)

        assert "total_time" in summary
        assert summary["total_time"] > 0
        assert "steps" in summary
        assert summary["steps"]["generate"]["customers"] > 0

        conn = duckdb.connect(db_path, read_only=True)
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        conn.close()

        for expected in [
            "crm_customers",
            "transactions",
            "web_sessions",
            "support_tickets",
            "golden_records",
            "rfm_scores",
            "clv_predictions",
            "cohort_retention",
        ]:
            assert expected in tables, f"Missing table: {expected}"

    def test_sample_mode(self, config_file: tuple[str, str]) -> None:
        """Sample mode reduces data volumes."""
        config_path, db_path = config_file
        summary = run_pipeline(config_path=config_path, sample=True)

        assert summary["steps"]["generate"]["customers"] > 0
        assert summary["total_time"] > 0

    def test_pipeline_creates_golden_records(self, config_file: tuple[str, str]) -> None:
        """Golden records table is populated after pipeline run."""
        config_path, db_path = config_file
        run_pipeline(config_path=config_path, sample=False)

        conn = duckdb.connect(db_path, read_only=True)
        golden = conn.execute("SELECT COUNT(*) AS n FROM golden_records").fetchone()[0]
        conn.close()

        assert golden > 0

    def test_pipeline_creates_rfm_scores(self, config_file: tuple[str, str]) -> None:
        """RFM scores table has segment assignments."""
        config_path, db_path = config_file
        run_pipeline(config_path=config_path, sample=False)

        conn = duckdb.connect(db_path, read_only=True)
        rfm = conn.execute("SELECT DISTINCT segment FROM rfm_scores").df()
        conn.close()

        assert len(rfm) > 0


class TestFlattenRetentionMatrix:
    def test_flatten(self) -> None:
        """Converts a pivot table to long format."""
        matrix = pd.DataFrame(
            {"0": [100.0, 100.0], "1": [80.0, 70.0]},
            index=["2024-01", "2024-02"],
        )
        matrix.index.name = "cohort_month"
        matrix.columns = [0, 1]
        matrix.columns.name = "cohort_age"

        result = _flatten_retention_matrix(matrix)

        assert len(result) == 4
        assert set(result.columns) == {"cohort_month", "cohort_age", "retention_rate"}

    def test_flatten_with_nans(self) -> None:
        """NaN values are excluded from the flattened output."""
        matrix = pd.DataFrame(
            {"0": [100.0, 100.0], "1": [80.0, float("nan")]},
            index=["2024-01", "2024-02"],
        )
        matrix.columns = [0, 1]

        result = _flatten_retention_matrix(matrix)

        assert len(result) == 3


class TestMainCLI:
    def test_main_with_args(self, config_file: tuple[str, str]) -> None:
        """CLI entry point parses args and runs pipeline."""
        config_path, _ = config_file
        with patch("sys.argv", ["src.main", "--config", config_path, "--sample"]):
            main()

    def test_main_default_args(self, config_file: tuple[str, str]) -> None:
        """CLI entry point works with config override."""
        config_path, _ = config_file
        with patch("sys.argv", ["src.main", "--config", config_path]):
            main()
