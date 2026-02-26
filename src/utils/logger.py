"""
Enhanced logging utility for DocQA AI system.
Provides structured logging, error categorization, stack traces, and improved error messages.
"""

import os
import sys
import json
import logging
import logging.handlers
import traceback
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable
from datetime import datetime
from functools import wraps
import time
import hashlib
import socket
import platform

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

# Try to import structlog for structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# Error Categories
# ============================================================

class ErrorCategory:
    """Error categories for better classification."""
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    FILE_IO = "file_io"
    PARSING = "parsing"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class ErrorSeverity:
    """Error severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


# ============================================================
# Enhanced JSON Formatter
# ============================================================

class EnhancedJsonFormatter(jsonlogger.JsonFormatter) if JSON_LOGGER_AVAILABLE else object:
    """
    Enhanced JSON formatter with additional context and error details.
    """

    def __init__(self, *args, **kwargs):
        if JSON_LOGGER_AVAILABLE:
            super().__init__(*args, **kwargs)
            self.rename_fields = {
                'asctime': 'timestamp',
                'levelname': 'level',
                'name': 'logger',
                'module': 'module',
                'funcName': 'function',
                'lineno': 'line'
            }

    def add_fields(self, log_record, record, message_dict):
        if JSON_LOGGER_AVAILABLE:
            super().add_fields(log_record, record, message_dict)

        # Add system info
        if not hasattr(log_record, 'hostname'):
            log_record['hostname'] = socket.gethostname()

        if not hasattr(log_record, 'pid'):
            log_record['pid'] = os.getpid()

        if not hasattr(log_record, 'thread_id'):
            log_record['thread_id'] = threading.get_ident() if 'threading' in sys.modules else 0

        # Add process info
        if not hasattr(log_record, 'process_name'):
            log_record['process_name'] = os.path.basename(sys.argv[0]) if sys.argv else 'unknown'

        # Add error details if present
        if hasattr(record, 'exc_info') and record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            if exc_value:
                log_record['error_type'] = exc_type.__name__ if exc_type else 'Unknown'
                log_record['error_message'] = str(exc_value)

                # Add stack trace
                if hasattr(record, 'stack_info') and record.stack_info:
                    log_record['stack_trace'] = record.stack_info
                elif exc_traceback:
                    log_record['stack_trace'] = ''.join(traceback.format_tb(exc_traceback))

        # Add custom fields from extra
        if hasattr(record, 'extra'):
            for key, value in record.extra.items():
                if key not in log_record:
                    log_record[key] = value

        # Ensure timestamp format
        if 'timestamp' in log_record and isinstance(log_record['timestamp'], datetime):
            log_record['timestamp'] = log_record['timestamp'].isoformat()


# ============================================================
# Colored Formatter
# ============================================================

class ColoredFormatter:
    """Custom colored formatter for console output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'FATAL': '\033[41m\033[37m', # Red background, white text
        'RESET': '\033[0m'        # Reset
    }

    # Additional styles
    STYLES = {
        'bold': '\033[1m',
        'dim': '\033[2m',
        'underline': '\033[4m',
        'blink': '\033[5m'
    }

    def __init__(self, fmt: Optional[str] = None, show_timestamp: bool = True,
                 show_level: bool = True, show_logger: bool = True,
                 show_location: bool = False):
        """
        Initialize colored formatter.

        Args:
            fmt: Format string
            show_timestamp: Show timestamp
            show_level: Show log level
            show_logger: Show logger name
            show_location: Show file location
        """
        self.show_timestamp = show_timestamp
        self.show_level = show_level
        self.show_logger = show_logger
        self.show_location = show_location

        self.fmt = fmt or self._build_format()

    def _build_format(self) -> str:
        """Build format string based on settings."""
        parts = []

        if self.show_timestamp:
            parts.append('%(asctime)s')

        if self.show_level:
            parts.append('%(levelname)-8s')

        if self.show_logger:
            parts.append('%(name)s')

        if self.show_location:
            parts.append('%(filename)s:%(lineno)d')

        parts.append('%(message)s')

        return ' | '.join(parts)

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
            'lineno': record.lineno,
            'funcName': record.funcName,
            'module': record.module
        }

        # Add exception info if present
        if record.exc_info:
            formatted += '\n' + ''.join(traceback.format_exception(*record.exc_info))

        return formatted

    def formatTime(self, record: logging.LogRecord) -> str:
        """Format timestamp."""
        return datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


# ============================================================
# Error Message Enhancer
# ============================================================

class ErrorMessageEnhancer:
    """
    Enhance error messages with context, suggestions, and categorization.
    """

    # Common error patterns and suggestions
    ERROR_PATTERNS = {
        r'Connection refused|Failed to connect|ConnectionError': {
            'category': ErrorCategory.NETWORK,
            'suggestion': 'Check if the service is running and reachable. Verify network connectivity.',
            'severity': ErrorSeverity.ERROR
        },
        r'Timeout|timed out|timeout': {
            'category': ErrorCategory.TIMEOUT,
            'suggestion': 'The operation took too long. Try increasing the timeout or reducing the workload.',
            'severity': ErrorSeverity.ERROR
        },
        r'Permission denied|Access denied|Forbidden|403': {
            'category': ErrorCategory.AUTHORIZATION,
            'suggestion': 'Check your permissions and authentication credentials.',
            'severity': ErrorSeverity.WARNING
        },
        r'Unauthorized|401|Invalid token|Invalid credentials': {
            'category': ErrorCategory.AUTHENTICATION,
            'suggestion': 'Verify your API key or authentication token is valid and not expired.',
            'severity': ErrorSeverity.WARNING
        },
        r'Rate limit|429|Too many requests': {
            'category': ErrorCategory.RATE_LIMIT,
            'suggestion': 'You have exceeded the rate limit. Wait and try again or reduce request frequency.',
            'severity': ErrorSeverity.WARNING
        },
        r'Out of memory|MemoryError|Memory limit': {
            'category': ErrorCategory.MEMORY,
            'suggestion': 'The system is running out of memory. Try processing smaller batches or reducing data size.',
            'severity': ErrorSeverity.CRITICAL
        },
        r'File not found|No such file|FileNotFoundError': {
            'category': ErrorCategory.FILE_IO,
            'suggestion': 'Check that the file path is correct and the file exists.',
            'severity': ErrorSeverity.ERROR
        },
        r'Parse error|Invalid JSON|Invalid format|Parsing failed': {
            'category': ErrorCategory.PARSING,
            'suggestion': 'Check the data format and ensure it matches the expected schema.',
            'severity': ErrorSeverity.ERROR
        },
        r'Validation error|Invalid value|Schema validation': {
            'category': ErrorCategory.VALIDATION,
            'suggestion': 'Check the input data against the expected schema or requirements.',
            'severity': ErrorSeverity.WARNING
        },
        r'Missing dependency|ModuleNotFoundError|ImportError': {
            'category': ErrorCategory.DEPENDENCY,
            'suggestion': 'Install the required dependency using pip or check your environment setup.',
            'severity': ErrorSeverity.ERROR
        },
        r'Database|SQL|Postgres|MySQL|connection|pool': {
            'category': ErrorCategory.DATABASE,
            'suggestion': 'Check database connection, credentials, and ensure the database is running.',
            'severity': ErrorSeverity.CRITICAL
        },
        r'Configuration|Config|setting': {
            'category': ErrorCategory.CONFIGURATION,
            'suggestion': 'Check your configuration file or environment variables for correct values.',
            'severity': ErrorSeverity.WARNING
        }
    }

    @classmethod
    def enhance_error(cls, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance an error with category, suggestion, and context.

        Args:
            error: The exception
            context: Additional context

        Returns:
            Enhanced error information
        """
        error_str = str(error)
        error_type = type(error).__name__

        # Find matching pattern
        category = ErrorCategory.UNKNOWN
        suggestion = "Check the error details and logs for more information."
        severity = ErrorSeverity.ERROR

        for pattern, info in cls.ERROR_PATTERNS.items():
            if re.search(pattern, error_str, re.IGNORECASE):
                category = info.get('category', ErrorCategory.UNKNOWN)
                suggestion = info.get('suggestion', suggestion)
                severity = info.get('severity', ErrorSeverity.ERROR)
                break

        # Build enhanced error info
        enhanced = {
            'error_type': error_type,
            'error_message': error_str,
            'category': category,
            'severity': severity,
            'suggestion': suggestion,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }

        # Add stack trace if available
        if hasattr(error, '__traceback__'):
            enhanced['stack_trace'] = ''.join(traceback.format_tb(error.__traceback__))

        return enhanced

    @classmethod
    def format_error_message(cls, enhanced: Dict[str, Any]) -> str:
        """
        Format enhanced error for logging.

        Args:
            enhanced: Enhanced error information

        Returns:
            Formatted error message
        """
        parts = []

        parts.append(f"[{enhanced.get('category', 'unknown').upper()}]")
        parts.append(f"{enhanced.get('error_type', 'Error')}: {enhanced.get('error_message', '')}")

        if enhanced.get('suggestion'):
            parts.append(f"💡 {enhanced['suggestion']}")

        if enhanced.get('context'):
            context_str = ', '.join(f"{k}={v}" for k, v in enhanced['context'].items())
            parts.append(f"Context: {context_str}")

        return ' | '.join(parts)


# ============================================================
# Structured Logger
# ============================================================

class StructuredLogger:
    """
    Enhanced logger with structured logging, error enhancement, and context management.
    """

    def __init__(self, logger: logging.Logger, default_fields: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.default_fields = default_fields or {}
        self.context_stack: List[Dict[str, Any]] = []
        self._error_enhancer = ErrorMessageEnhancer()

    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None,
             exc_info: bool = False, **kwargs):
        """Internal logging method with structured fields."""
        log_extra = self.default_fields.copy()

        # Add context from stack
        for ctx in self.context_stack:
            log_extra.update(ctx)

        if extra:
            log_extra.update(extra)

        # Enhance error messages
        if exc_info and isinstance(exc_info, (Exception, tuple)):
            if isinstance(exc_info, Exception):
                enhanced = self._error_enhancer.enhance_error(
                    exc_info,
                    context=log_extra
                )
                msg = self._error_enhancer.format_error_message(enhanced)
                log_extra['error_enhanced'] = enhanced

        if log_extra:
            self.logger.log(level, msg, extra=log_extra, exc_info=exc_info, **kwargs)
        else:
            self.logger.log(level, msg, exc_info=exc_info, **kwargs)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.DEBUG, msg, extra, **kwargs)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.INFO, msg, extra, **kwargs)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.WARNING, msg, extra, **kwargs)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None,
              exc_info: bool = True, **kwargs):
        self._log(logging.ERROR, msg, extra, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None,
                 exc_info: bool = True, **kwargs):
        self._log(logging.CRITICAL, msg, extra, exc_info=exc_info, **kwargs)

    def exception(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.ERROR, msg, extra, exc_info=True, **kwargs)

    def fatal(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        self._log(logging.CRITICAL, msg, extra, exc_info=True, **kwargs)

    def push_context(self, **fields):
        """Push context fields onto the stack."""
        self.context_stack.append(fields)

    def pop_context(self):
        """Pop context fields from the stack."""
        if self.context_stack:
            return self.context_stack.pop()
        return {}

    def with_fields(self, **fields) -> 'StructuredLogger':
        """Create a new logger with additional default fields."""
        new_fields = self.default_fields.copy()
        new_fields.update(fields)
        return StructuredLogger(self.logger, new_fields)


# ============================================================
# Logger Configuration
# ============================================================

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
        colored_console: bool = True,
        enable_structured: bool = True,
        enable_error_enhancement: bool = True,
        show_location: bool = True,
        show_timestamp: bool = True
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
        self.enable_structured = enable_structured
        self.enable_error_enhancement = enable_error_enhancement
        self.show_location = show_location
        self.show_timestamp = show_timestamp


class LoggerManager:
    """Centralized logger manager for the application."""

    _instances: Dict[str, StructuredLogger] = {}
    _default_config: Optional[LoggerConfig] = None
    _error_enhancer = ErrorMessageEnhancer()

    @classmethod
    def setup_default_config(cls, config: LoggerConfig):
        """Set default configuration for all loggers."""
        cls._default_config = config

        # Create log directory if needed
        if config.log_to_file:
            config.log_dir.mkdir(parents=True, exist_ok=True)

        # Set global log level
        logging.basicConfig(level=config.level)

    @classmethod
    def get_logger(
        cls,
        name: Optional[str] = None,
        config: Optional[LoggerConfig] = None
    ) -> StructuredLogger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name
            config: Optional custom configuration

        Returns:
            StructuredLogger instance
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

        # Create base logger
        base_logger = logging.getLogger(logger_name)
        base_logger.setLevel(config.level)
        base_logger.propagate = False

        # Remove existing handlers
        base_logger.handlers.clear()

        # Add console handler
        if config.log_to_console:
            cls._add_console_handler(base_logger, config)

        # Add file handlers
        if config.log_to_file:
            cls._add_file_handlers(base_logger, config)

        # Add JSON handler
        if config.log_to_json and JSON_LOGGER_AVAILABLE:
            cls._add_json_handler(base_logger, config)

        # Create structured logger
        structured_logger = StructuredLogger(base_logger)
        cls._instances[logger_name] = structured_logger

        return structured_logger

    @classmethod
    def _add_console_handler(cls, logger: logging.Logger, config: LoggerConfig):
        """Add console (stdout) handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)

        # Choose formatter
        if config.colored_console:
            formatter = ColoredFormatter(
                show_timestamp=config.show_timestamp,
                show_level=True,
                show_logger=True,
                show_location=config.show_location
            )
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
            formatter = EnhancedJsonFormatter(
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
            formatter = EnhancedJsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
            json_handler.setFormatter(formatter)

        logger.addHandler(json_handler)


# ============================================================
# Decorators
# ============================================================

def log_function_call(logger: Optional[StructuredLogger] = None, level: str = "DEBUG"):
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
            logger.debug(f"Calling {func.__name__}({all_args})", extra={"function": func.__name__})

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log return value (truncate if too long)
                result_str = repr(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."

                logger.debug(
                    f"{func.__name__} completed in {duration:.3f}s, returned {result_str}",
                    extra={"function": func.__name__, "duration": duration}
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s: {e}",
                    extra={"function": func.__name__, "duration": duration},
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


def log_async_function_call(logger: Optional[StructuredLogger] = None, level: str = "DEBUG"):
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
            logger.debug(f"Async calling {func.__name__}({all_args})", extra={"function": func.__name__})

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                result_str = repr(result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."

                logger.debug(
                    f"{func.__name__} completed in {duration:.3f}s, returned {result_str}",
                    extra={"function": func.__name__, "duration": duration}
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s: {e}",
                    extra={"function": func.__name__, "duration": duration},
                    exc_info=True
                )
                raise

        return wrapper
    return decorator


# ============================================================
# Convenience Functions
# ============================================================

_default_logger: Optional[StructuredLogger] = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (e.g., 'src.ingestion.loader')

    Returns:
        StructuredLogger instance
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
    colored: bool = True,
    show_location: bool = True,
    show_timestamp: bool = True
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
        show_location: Whether to show file location in logs
        show_timestamp: Whether to show timestamp in logs
    """
    config = LoggerConfig(
        name="docqa",
        level=level,
        log_dir=log_dir,
        log_to_file=log_to_file,
        log_to_console=log_to_console,
        format_json=format_json,
        colored_console=colored,
        show_location=show_location,
        show_timestamp=show_timestamp
    )
    LoggerManager.setup_default_config(config)

    global _default_logger
    _default_logger = LoggerManager.get_logger("docqa")
    _default_logger.info(f"Logging configured: level={level}, log_dir={log_dir}")


# ============================================================
# Log Context Manager
# ============================================================

class LogContext:
    """Context manager for adding context to logs."""

    def __init__(self, logger: StructuredLogger, **context):
        self.logger = logger
        self.context = context

    def __enter__(self):
        self.logger.push_context(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.pop_context()


def log_context(**context):
    """
    Context manager for adding context to logs.

    Args:
        **context: Context fields to add

    Returns:
        LogContext context manager
    """
    logger = get_logger()
    return LogContext(logger, **context)


# ============================================================
# Error Logging Helper
# ============================================================

def log_error(
    error: Exception,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[StructuredLogger] = None
):
    """
    Log an error with enhanced information.

    Args:
        error: The exception
        message: Optional message
        context: Optional context
        logger: Logger to use
    """
    if logger is None:
        logger = get_logger()

    enhanced = ErrorMessageEnhancer.enhance_error(error, context)
    formatted = ErrorMessageEnhancer.format_error_message(enhanced)

    msg = f"{message}: {formatted}" if message else formatted

    logger.error(msg, extra={"error_enhanced": enhanced}, exc_info=True)


# ============================================================
# Initialization
# ============================================================

# Initialize default logging on module import
setup_logging()


if __name__ == "__main__":
    # Example usage
    import time

    # Get logger
    logger = get_logger("example")

    # Basic logging
    logger.info("This is an info message")
    logger.warning("This is a warning message")

    # Log with context
    with log_context(user_id="123", action="test"):
        logger.info("Action performed with context")

    # Log with extra fields
    logger.info("User action", extra={"user_id": "456", "action": "login", "duration_ms": 150})

    # Test error logging
    try:
        raise ValueError("Invalid input value: 'test'")
    except Exception as e:
        log_error(e, "Failed to process request", {"input": "test", "source": "api"})

    # Test decorated function
    @log_function_call(level="INFO")
    def test_function(x, y):
        return x + y

    test_function(5, 3)

    print("\n✅ Enhanced logging ready! Check ./logs directory for log files.")
