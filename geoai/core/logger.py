"""
Structured logging configuration for the GeoAI Platform.

Provides a single entry point for configuring the platform-wide logging
infrastructure. All production modules obtain their loggers via the standard
``logging.getLogger(__name__)`` pattern after this module has been initialised
at application startup.

Supports console output, file output with rotation, and an optional structured
JSON handler for production deployments. All configuration is read from the
logging YAML file; no values are hardcoded.

Position in dependency hierarchy: core (no internal imports).
"""

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    structured: bool = False,
) -> None:
    """Configure the GeoAI Platform logging infrastructure.

    Must be called once at application startup before any platform modules
    emit log records. Subsequent calls replace the existing handler configuration.

    Args:
        log_level: Logging level name as a string (DEBUG, INFO, WARNING, ERROR,
            CRITICAL). Case-insensitive.
        log_file: Absolute or relative path to the log file. If None, file
            logging is disabled and only console output is active.
        max_bytes: Maximum size in bytes for each log file before rotation.
            Default is 10 MB.
        backup_count: Number of rotated log files to retain. Default is 5.
        structured: If True, emit JSON-structured log records suitable for
            log aggregation systems. If False, emit human-readable text.

    Raises:
        ValueError: If ``log_level`` is not a recognised logging level name.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level '{log_level}'. Must be one of: "
            "DEBUG, INFO, WARNING, ERROR, CRITICAL."
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove any handlers added by previous calls or by external libraries.
    root_logger.handlers.clear()

    formatter = _build_formatter(structured)

    # Console handler — always active.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — active only when log_file is specified.
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress verbose third-party library loggers that pollute platform output.
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("fiona").setLevel(logging.WARNING)
    logging.getLogger("shapely").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured — level=%s file=%s structured=%s",
        log_level.upper(),
        log_file or "disabled",
        structured,
    )


def configure_logging_from_config(config: dict) -> None:
    """Configure logging from a parsed logging configuration dictionary.

    Convenience wrapper that extracts logging parameters from the dictionary
    produced by loading ``configs/logging.yaml`` and passes them to
    :func:`configure_logging`.

    Args:
        config: Dictionary containing keys ``level``, ``file`` (optional),
            ``max_bytes`` (optional), ``backup_count`` (optional), and
            ``structured`` (optional).

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    configure_logging(
        log_level=config.get("level", "INFO"),
        log_file=config.get("file", None),
        max_bytes=config.get("max_bytes", 10_485_760),
        backup_count=config.get("backup_count", 5),
        structured=config.get("structured", False),
    )


def get_run_logger(run_id: str) -> logging.Logger:
    """Return a child logger scoped to a specific pipeline run.

    Creates a named logger ``geoai.run.<run_id>`` that inherits the root
    configuration but can be targeted by handlers writing to a run-specific
    log file.

    Args:
        run_id: Unique identifier for the pipeline run (e.g. 'PS10_20260620_143012').

    Returns:
        A :class:`logging.Logger` instance scoped to the run.
    """
    return logging.getLogger(f"geoai.run.{run_id}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_formatter(structured: bool) -> logging.Formatter:
    """Build a log formatter appropriate for the requested output style.

    Args:
        structured: If True, return a JSON formatter. If False, return a
            human-readable text formatter.

    Returns:
        A :class:`logging.Formatter` instance.
    """
    if structured:
        return _JSONFormatter()
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects.

    Suitable for ingestion by log aggregation systems (e.g., ELK stack,
    CloudWatch Logs). Each record is a complete JSON object on a single line.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-encoded string representing the log record.
        """
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)
