"""Logging configuration for decline_curve package."""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
    log_file: Optional[str | Path] = None,
) -> None:
    """Configure logging for the decline_curve package.

    Args:
        level: Logging level (default: WARNING)
        format_string: Custom format string (default: standard format)
        stream: Output stream (default: stderr)
        log_file: Optional path to log file (logs to both console and file if provided)

    Example:
        >>> from decline_curve.logging_config import configure_logging
        >>> import logging
        >>> configure_logging(level=logging.INFO, log_file='batch_job.log')
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = []

    # Console handler
    if stream is None:
        stream = sys.stderr
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
