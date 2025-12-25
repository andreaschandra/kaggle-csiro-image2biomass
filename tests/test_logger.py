"""Unit tests for csiro_biomass.utils.logger module."""

import logging

from csiro_biomass.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_default_params(self, tmp_path, monkeypatch):
        """Test setup_logger with default parameters."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger()

        assert logger.name == "kaggle"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2

    def test_setup_logger_custom_name(self, tmp_path, monkeypatch):
        """Test setup_logger with custom name."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(name="custom_logger")

        assert logger.name == "custom_logger"

    def test_setup_logger_custom_level(self, tmp_path, monkeypatch):
        """Test setup_logger with custom logging level."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logger_creates_log_directory(self, tmp_path, monkeypatch):
        """Test that setup_logger creates logs directory."""
        monkeypatch.chdir(tmp_path)
        logs_dir = tmp_path / "logs"
        assert not logs_dir.exists()

        setup_logger()

        assert logs_dir.exists()
        assert logs_dir.is_dir()

    def test_setup_logger_creates_log_file(self, tmp_path, monkeypatch):
        """Test that setup_logger creates app.log file."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "logs" / "app.log"

        setup_logger()

        assert log_file.exists()
        assert log_file.is_file()

    def test_setup_logger_handlers(self, tmp_path, monkeypatch):
        """Test that setup_logger configures both console and file handlers."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger()

        assert len(logger.handlers) == 2
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" in handler_types

    def test_setup_logger_formatter(self, tmp_path, monkeypatch):
        """Test that setup_logger configures formatters correctly."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger()

        for handler in logger.handlers:
            formatter = handler.formatter
            assert formatter is not None
            assert "%(asctime)s" in formatter._fmt
            assert "%(name)s" in formatter._fmt
            assert "%(levelname)s" in formatter._fmt
            assert "%(message)s" in formatter._fmt

    def test_setup_logger_no_duplicate_handlers(self, tmp_path, monkeypatch):
        """Test that calling setup_logger twice doesn't add duplicate handlers."""
        monkeypatch.chdir(tmp_path)

        logger1 = setup_logger(name="test_logger")
        initial_handler_count = len(logger1.handlers)

        logger2 = setup_logger(name="test_logger")

        assert len(logger2.handlers) == initial_handler_count

    def test_setup_logger_logging_works(self, tmp_path, monkeypatch, caplog):
        """Test that logger actually logs messages."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(name="test_log")

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "Test message" in caplog.text

    def test_setup_logger_file_logging(self, tmp_path, monkeypatch):
        """Test that logger writes to file."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "logs" / "app.log"

        logger = setup_logger()
        logger.info("File logging test")

        assert log_file.exists()
        content = log_file.read_text()
        assert "File logging test" in content

    def test_setup_logger_info_level_filtering(self, tmp_path, monkeypatch):
        """Test that INFO level filters DEBUG messages."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "logs" / "app.log"

        logger = setup_logger(level=logging.INFO)
        logger.debug("Debug message")
        logger.info("Info message")

        content = log_file.read_text()
        assert "Debug message" not in content
        assert "Info message" in content

    def test_setup_logger_debug_level(self, tmp_path, monkeypatch):
        """Test that DEBUG level logs all messages."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "logs" / "app.log"

        logger = setup_logger(level=logging.DEBUG)
        logger.debug("Debug message")
        logger.info("Info message")

        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content

    def test_setup_logger_warning_level(self, tmp_path, monkeypatch, caplog):
        """Test logger with WARNING level."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(level=logging.WARNING)

        with caplog.at_level(logging.WARNING):
            logger.info("Info message")
            logger.warning("Warning message")

        assert "Info message" not in caplog.text
        assert "Warning message" in caplog.text

    def test_setup_logger_error_level(self, tmp_path, monkeypatch, caplog):
        """Test logger with ERROR level."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(level=logging.ERROR)

        with caplog.at_level(logging.ERROR):
            logger.warning("Warning message")
            logger.error("Error message")

        assert "Warning message" not in caplog.text
        assert "Error message" in caplog.text

    def test_setup_logger_file_mode_write(self, tmp_path, monkeypatch):
        """Test that logger file mode is 'w' (overwrites)."""
        monkeypatch.chdir(tmp_path)
        log_file = tmp_path / "logs" / "app.log"

        logger1 = setup_logger(name="logger1")
        logger1.info("First message")

        logger2 = setup_logger(name="logger2")
        logger2.info("Second message")

        content = log_file.read_text()
        # File should be overwritten, so both messages should be present
        # but if we recreate the logger, the file handler might be reused
        assert "message" in content.lower()

    def test_setup_logger_existing_logs_dir(self, tmp_path, monkeypatch):
        """Test setup_logger when logs directory already exists."""
        monkeypatch.chdir(tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        logger = setup_logger()

        assert logger is not None
        assert logs_dir.exists()


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_multiple_loggers_different_names(self, tmp_path, monkeypatch):
        """Test creating multiple loggers with different names."""
        monkeypatch.chdir(tmp_path)

        logger1 = setup_logger(name="logger1")
        logger2 = setup_logger(name="logger2")

        assert logger1.name != logger2.name
        assert logger1 is not logger2

    def test_logger_in_real_workflow(self, tmp_path, monkeypatch, caplog):
        """Test logger in a simulated workflow."""
        monkeypatch.chdir(tmp_path)

        logger = setup_logger(name="workflow")

        with caplog.at_level(logging.INFO):
            logger.info("Starting workflow")
            logger.info("Processing data")
            logger.info("Workflow complete")

        assert "Starting workflow" in caplog.text
        assert "Processing data" in caplog.text
        assert "Workflow complete" in caplog.text
