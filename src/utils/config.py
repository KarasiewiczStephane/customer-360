"""Configuration management for the Customer 360 platform.

Loads and validates YAML configuration from the configs directory,
providing typed access to all configurable parameters.
"""

from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. Uses the default
            ``configs/config.yaml`` when *None*.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        config = yaml.safe_load(fh)

    return config


def get_database_path(config: dict[str, Any]) -> str:
    """Return the resolved database path from config.

    Args:
        config: Loaded configuration dictionary.

    Returns:
        Database file path as a string.
    """
    return config["database"]["path"]
