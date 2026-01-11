"""
Logging utility for DocQA AI system.
Provides structured logging, multiple output handlers, and log rotation.
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from functools import wraps
import traceback
import time

# Try to import colorlog for colored output
try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

# Try to import python-json-logger for JSON formatting
try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False


class CustomJsonFormatter(jsonlogger.JsonFormatter) if JSON_LOGGER_AVAILABLE else object:
    """Custom JSON formatter for structured logging."""

    def __init__(self, *args, **kwargs):
        if JSON_LOGGER_AVAILABLE:
            super().__init__(*args, **kwargs)
            self.rename_fields = {
                'asctime': 'timestamp',
                'levelname': 'level',
                'name': 'logger'
            }

    def add_fields(self, log_record, record, message_dict):
        if JSON_LOGGER_AVAILABLE:
            super().add_fields(log_record, record, message_dict)

        # Add hostname and process info
        if not hasattr(log_record, 'hostname'):
            log_record['hostname'] = os.uname().nodename if hasattr(os, 'uname') else 'unknown'

        if not hasattr(log_record, 'pid'):
            log_record['pid'] = os.getpid()

        if not hasattr(log_record, 'thread_id'):
            log_record['thread_id'] = threading.get_ident() if 'threading' in sys.modules else 0


class ColoredFormatter:
    """Custom colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def __init__(self, fmt: Optional[str] = None):
        self.fmt = fmt or '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Get color for level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Format the message
        formatted = self.fmt % {
            'asctime': self.formatTime(record),
            'levelname': f'{color}{record.levelname}{reset}',
            'name': record.name,
            'message': record.getMessage(),
            'filename': record.filename,
            'lineno': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            formatted += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return formatted

    def formatTime(self, record: logging.LogRecord) -> str:
        """Format timestamp."""
        return datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')


class LoggerConfig:
    """Configuration for logger setup."""

    def __init__(
        self,
        name: str = "docqa",
        level: str = "INFO",
        log_dir: str = "./logs",
        log_to_file: bool = True,
        log_to_console: bool = True,
        log_to_json: bool = False,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
        format_json: bool = False,
        enable_rotation: bool = True,
        colored_console: bool = True
    ):
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.log_dir = Path(log_dir)
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_to_json = log_to_json
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count
        self.format_json = format_json
        self.enable_rotation = enable_rotation
        self.colored_console = colored_console and COLORLOG_AVAILABLE


class LoggerManager:
    """Centralized logger manager for the application."""

    _instances: Dict[str, logging.Logger] = {}
    _default_config: Optional[LoggerConfig] = None

    @classmethod
    def setup_default_config(cls, config: LoggerConfig):
        """Set default configuration for all loggers."""
        cls._default_config = config

        # Create log directory if needed
        if config.log_to_file:
            config.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_logger(
        cls,
        name: Optional[str] = None,
        config: Optional[LoggerConfig] = None
    ) -> logging.Logger:
        """
        Get a logger instance.

        Args:
            name: Logger name (if None, uses root logger)
            config: Optional custom configuration

        Returns:
            Configured logger instance
        """
        logger_name = name or "docqa"

        # Return cached logger if exists
        if logger_name in cls._instances:
            return cls._instances[logger_name]

        # Use default config if none provided
        if config is None:
            if cls._default_config is None:
                cls.setup_default_config(LoggerConfig())
            config = cls._default_config

        # Create new logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(config.level)
        logger.propagate = False

        # Remove existing handlers
        logger.handlers.clear()

        # Add console handler
        if config.log_to_console:
            cls._add_console_handler(logger, config)

        # Add file handlers
        if config.log_to_file:
            cls._add_file_handlers(logger, config)

        # Add JSON handler
        if config.log_to_json and JSON_LOGGER_AVAILABLE:
            cls._add_json_handler(logger, config)

        cls._instances[logger_name] = logger
        return logger

    @classmethod
    def _add_console_handler(cls, logger: logging.Logger, config: LoggerConfig):
        """Add console (stdout) handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)

        # Choose formatter
        if config.colored_console and COLORLOG_AVAILABLE:
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        elif config.colored_console:
            formatter = ColoredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    @classmethod
    def _add_file_handlers(cls, logger: logging.Logger, config: LoggerConfig):
        """Add rotating file handlers."""
        # Main log file handler
        log_file = config.log_dir / f"{config.name}.log"

        if config.enable_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')

        file_handler.setLevel(config.level)

        # Choose formatter for file
        if config.format_json and JSON_LOGGER_AVAILABLE:
            formatter = CustomJsonFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s',
                rename_fields={
                    'timestamp': 'timestamp',
                    'level': 'level',
                    'name': 'logger'
                }
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Error log file handler (only errors and above)
        error_log_file = config.log_dir / f"{config.name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    @classmethod
    def _add_json_handler(cls, logger: logging.Logger, config: LoggerConfig):
        """Add JSON formatter handler for structured logging."""
        json_log_file = config.log_dir / f"{config.name}.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=config.max_file_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(config.level)

        if JSON_LOGGER_AVAILABLE:
            formatter = CustomJsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
            json_handler.setFormatter(formatter)

        logger.addHandler(json_handler)


# Global logger instance
_default_logger = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (e.g., 'src.ingestion.loader')

    Returns:
        Configured logger
    """
    global _default_logger

    if name:
        return LoggerManager.get_logger(name)

    if _default_logger is None:
        _default_logger = LoggerManager.get_logger("docqa")

    return _default_logger


def setup_logging(
    level: str = "INFO",
    log_dir: str = "./logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    format_json: bool = False,
    colored: bool = True
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_to_file: Whether to log to files
        log_to_console: Whether to log to console
        format_json: Whether to use JSON format for files
        colored: Whether to use colored console output
    """
    config = LoggerConfig(
        name="docqa",
        level=level,
        log_dir=log_dir,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        format_json=format_json,
        colored_console=colored
    )
    LoggerManager.setup_default_config(config)

    global _default_logger
    _default_logger = LoggerManager.get_logger("docqa")
    _default_logger.info(f"Logging configured: level={level}, log_dir={log_dir}")


class LoggerContext:
    """Context manager for temporary log level changes."""

    def __init__(self, logger: logging.Logger, level: Union[str, int], name: Optional[str] = None):
        self.logger = logger if name is None else get_logger(name)
        self.new_level = getattr(logging, level.upper()) if isinstance(level, str) else level
        self.old_level = self.logger.level

    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def log_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments and return values.

    Args:
        logger: Logger to use (defaults to module logger)
        level: Log level for function calls
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            # Build argument string
            args_str = ', '.join([repr(a) for a in args])
            kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))

            # Log function entry
            log_level = getattr(logging, level.upper(), logging.DEBUG)
            logger.log(log_level, f"Calling {func.__name__}({all_args})")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log return value (truncate if too long)
                result_str = repr(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."

                logger.log(
                    log_level,
                    f"{func.__name__} completed in {duration:.3f}s, returned {result_str}"
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s: {e}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_async_function_call(logger: Optional[logging.Logger] = None, level: str = "DEBUG"):
    """Decorator for async function logging."""
    import asyncio

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            # Build argument string
            args_str = ', '.join([repr(a) for a in args])
            kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))

            log_level = getattr(logging, level.upper(), logging.DEBUG)
            logger.log(log_level, f"Async calling {func.__name__}({all_args})")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                result_str = repr(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."

                logger.log(
                    log_level,
                    f"{func.__name__} completed in {duration:.3f}s, returned {result_str}"
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s: {e}",
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


class PerformanceLogger:
    """Utility for logging performance metrics."""

    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.start_time = None
        self.metrics = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics['duration_seconds'] = duration

        if exc_type is None:
            self.logger.info(f"Performance: {self.name} - {duration:.3f}s", extra=self.metrics)
        else:
            self.logger.error(f"Performance: {self.name} failed after {duration:.3f}s", extra=self.metrics)

    def add_metric(self, key: str, value: Any):
        """Add custom metric."""
        self.metrics[key] = value


def performance_logger(logger_name: str, operation_name: str):
    """Context manager for performance logging."""
    return PerformanceLogger(get_logger(logger_name), operation_name)


class StructuredLogger:
    """Wrapper for structured logging with extra fields."""

    def __init__(self, logger: logging.Logger, default_fields: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.default_fields = default_fields or {}

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Internal logging method with structured fields."""
        log_extra = self.default_fields.copy()
        if extra:
            log_extra.update(extra)

        if log_extra:
            self.logger.log(level, msg, extra=log_extra, **kwargs)
        else:
            self.logger.log(level, msg, **kwargs)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.DEBUG, msg, extra, **kwargs)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.INFO, msg, extra, **kwargs)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.WARNING, msg, extra, **kwargs)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.ERROR, msg, extra, **kwargs)

    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.CRITICAL, msg, extra, **kwargs)

    def exception(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.ERROR, msg, extra, exc_info=True, **kwargs)

    def with_fields(self, **fields) -> 'StructuredLogger':
        """Create a new logger with additional default fields."""
        new_fields = self.default_fields.copy()
        new_fields.update(fields)
        return StructuredLogger(self.logger, new_fields)


def get_structured_logger(name: str, **default_fields) -> StructuredLogger:
    """Get a structured logger with default fields."""
    logger = get_logger(name)
    return StructuredLogger(logger, default_fields)


# Import threading for thread ID in JSON logs
import threading

# Initialize default logging on module import
setup_logging()


if __name__ == "__main__":
    # Example usage
    import time

    # Get logger
    logger = get_logger("example")

    # Basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Structured logging with extra fields
    logger.info("User action", extra={"user_id": 123, "action": "login"})

    # Performance logging
    with performance_logger("example", "database_query"):
        time.sleep(0.5)  # Simulate work

    # Function decorator
    @log_function_call(level="INFO")
    def test_function(x, y):
        return x + y

    test_function(5, 3)

    # Structured logger
    struct_logger = get_structured_logger("api", service="docqa", version="1.0.0")
    struct_logger.info("Request received", extra={"method": "GET", "path": "/health"})

    # Child logger with additional fields
    child_logger = struct_logger.with_fields(endpoint="/query")
    child_logger.info("Processing query", extra={"query_length": 42})

    print("\nLogging examples completed. Check ./logs/ directory for log files.")
