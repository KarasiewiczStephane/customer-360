"""Tests for structured logging utilities."""

import logging

from src.utils.logger import get_logger


class TestGetLogger:
    """Tests for the get_logger factory."""

    def test_returns_logger_instance(self) -> None:
        """Returns a proper logging.Logger."""
        log = get_logger("test_module")
        assert isinstance(log, logging.Logger)

    def test_logger_has_handler(self) -> None:
        """Logger has at least one handler attached."""
        log = get_logger("test_handler")
        assert len(log.handlers) >= 1

    def test_logger_default_level(self) -> None:
        """Default level is INFO."""
        log = get_logger("test_level")
        assert log.level == logging.INFO

    def test_logger_custom_level(self) -> None:
        """Custom level is applied."""
        log = get_logger("test_custom_level", level=logging.DEBUG)
        assert log.level == logging.DEBUG

    def test_logger_format(self) -> None:
        """Handler uses the expected format string."""
        log = get_logger("test_format_check")
        handler = log.handlers[0]
        fmt = handler.formatter._fmt
        assert "%(asctime)s" in fmt
        assert "%(name)s" in fmt
        assert "%(levelname)s" in fmt

    def test_no_duplicate_handlers(self) -> None:
        """Calling get_logger twice does not add duplicate handlers."""
        name = "test_no_dup"
        log1 = get_logger(name)
        count = len(log1.handlers)
        log2 = get_logger(name)
        assert len(log2.handlers) == count
        assert log1 is log2
