"""Tests for configuration loading and management."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import get_database_path, load_config


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self) -> None:
        """Default config file loads successfully."""
        config = load_config()
        assert isinstance(config, dict)
        assert "data" in config
        assert "database" in config
        assert "resolution" in config
        assert "analytics" in config

    def test_load_config_data_section(self) -> None:
        """Data section contains expected keys with correct types."""
        config = load_config()
        data = config["data"]
        assert data["num_customers"] == 10000
        assert data["num_transactions"] == 500000
        assert data["num_web_sessions"] == 100000
        assert data["num_support_tickets"] == 20000
        assert data["sample_customers"] == 1000

    def test_load_config_resolution_section(self) -> None:
        """Resolution section contains matching thresholds."""
        config = load_config()
        resolution = config["resolution"]
        assert 0 < resolution["match_threshold"] <= 1.0
        assert isinstance(resolution["blocking_fields"], list)

    def test_load_config_analytics_section(self) -> None:
        """Analytics section contains RFM and clustering settings."""
        config = load_config()
        analytics = config["analytics"]
        assert analytics["rfm_segments"] == 5
        assert analytics["kmeans_max_clusters"] == 10

    def test_load_config_custom_path(self, tmp_path: Path) -> None:
        """Loading config from a custom path works."""
        custom = {"test_key": "test_value", "nested": {"a": 1}}
        config_file = tmp_path / "custom.yaml"
        config_file.write_text(yaml.dump(custom))

        loaded = load_config(str(config_file))
        assert loaded["test_key"] == "test_value"
        assert loaded["nested"]["a"] == 1

    def test_load_config_missing_file(self) -> None:
        """FileNotFoundError is raised for a missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """YAMLError is raised for invalid YAML content."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("{{invalid yaml: [}")
        with pytest.raises(yaml.YAMLError):
            load_config(str(bad_file))


class TestGetDatabasePath:
    """Tests for get_database_path helper."""

    def test_returns_path_from_config(self) -> None:
        """Database path is extracted from config dict."""
        config = {"database": {"path": "data/test.duckdb"}}
        assert get_database_path(config) == "data/test.duckdb"
