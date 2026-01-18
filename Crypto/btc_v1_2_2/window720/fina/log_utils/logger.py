"""Logging - System logging and monitoring"""

# ======================================== #
#     python -m fina.log_utils.logger      #
# ======================================== #


import json
import logging
import logging.handlers
import os
import sqlite3
import threading
from typing import Any, Dict, Optional
from ..utils.time_utils import get_now


# ======================================= #
#            Custom Log Handler           #
# ======================================= #


class DatabaseHandler(logging.Handler):
    """Custom logging handler that writes to SQLite database."""

    def __init__(self, db_path: str = "data/database.db"):
        super().__init__()
        self.db_path = db_path
        self._connected = False
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database and create tables if needed."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tradinglogs table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tradinglogs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()
            conn.close()
            self._connected = True
            print(f"Database initialized at: {self.db_path}")

        except Exception as e:
            print(f"Database initialization error: {e}")
            self._connected = False

    def emit(self, record):
        """Emit a log record to the database."""
        # This is called synchronously, so we need to handle separately
        # We'll store the record and process it later
        pass

    def async_emit(self, record: dict):
        """Insert record into SQLite database."""
        if not self._connected:
            print("Database not connected, skipping log entry")
            return

        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO tradinglogs (data, algorithm)
                    VALUES (?, ?)
                """,
                    (record.get("data", ""), record.get("algorithm", "unknown")),
                )

                conn.commit()
                conn.close()

            except Exception as e:
                print(f"Database logging error: {e}")
                # Try to reconnect on next attempt
                self._connected = False
                self._init_database()

    def disconnect(self):
        """Disconnect from database (nothing to do for SQLite)."""
        self._connected = False


class FileFormatter(logging.Formatter):
    """Custom formatter for file output without colors."""

    def format(self, record):
        # Format timestamp
        timestamp = get_now().strftime("%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Include milliseconds
        message = record.getMessage()

        # Combine all parts
        formatted = f"[{timestamp}][{record.levelname}] {message}"

        # Add data if present
        if hasattr(record, "data") and record.data:
            data_str = json.dumps(record.data, indent=2, ensure_ascii=False)
            formatted += f"\n{data_str}"

        return formatted


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds gray timestamp and colors to log messages."""

    # Color codes
    GRAY = "\033[90m"  # Gray for timestamp
    CYAN = "\033[96m"  # Cyan for info
    YELLOW = "\033[93m"  # Yellow for debug and warning
    RED = "\033[91m"  # Red for error and critical
    GREEN = "\033[92m"  # Green for other levels
    RESET = "\033[0m"  # Reset

    def format(self, record):
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S.%f")[
            :-3
        ]  # Include milliseconds
        colored_timestamp = f"{self.GRAY}[{timestamp}]{self.RESET}"

        # Get the message
        message = record.getMessage()
        level = record.levelname

        match record.levelname:
            case "DEBUG" | "WARNING":
                level = f"{self.YELLOW}[{level}]{self.RESET}"
                message = f"{self.YELLOW}{message}{self.RESET}"
            case "INFO":
                level = f"{self.CYAN}[{level}]{self.RESET}"
            case "ERROR" | "CRITICAL":
                level = f"{self.RED}[{level}]{self.RESET}"
                message = f"{self.RED}{message}{self.RESET}"
            case _:
                level = f"{self.GREEN}[{level}]{self.RESET}"

        # Combine timestamp and message
        formatted = f"{colored_timestamp} {level} {message}"

        # Add data if present
        if hasattr(record, "data") and record.data:
            data_str = json.dumps(record.data, indent=2, ensure_ascii=False)
            formatted += f"\n{data_str}"

        return formatted


# ======================================= #
#              TradingLogger              #
# ======================================= #


class Logger:
    def __init__(self, log_file: str = "trading.log"):
        self.log_file = log_file
        self.db_handler = DatabaseHandler()
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with multiple handlers."""
        # Create logger
        self.logger = logging.getLogger("trading_logger")
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for persistent logging
        log_file_path = self._get_log_file_path()
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setLevel(logging.INFO)
        file_formatter = FileFormatter()
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    def connect(self):
        """Connect to database handler (SQLite connects automatically)."""
        if not self.db_handler._connected:
            self.db_handler._init_database()

    def disconnect(self):
        """Disconnect from database handler."""
        self.db_handler.disconnect()

    # Public logging methods
    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        extra = {"data": data} if data else {}
        self.logger.info(message, extra=extra)

    def debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        extra = {"data": data} if data else {}
        self.logger.debug(message, extra=extra)

    def error(self, message: str, data: Optional[Dict[str, Any]] = None):
        extra = {"data": data} if data else {}
        self.logger.error(message, extra=extra)

    def track(self, message: str, raw_data: Optional[Dict[str, Any]] = None):
        """Track messages to terminal, file, and database."""
        algorithm = raw_data.get("algorithm", "unknown") if raw_data else "unknown"
        data = raw_data.get("data", None) if raw_data else None

        # Log to console and file
        self.logger.info(
            message,
            extra={
                "data": raw_data or {},
            },
        )

        # Log to database if data contains algorithm info
        if data and algorithm != "unknown":
            self.db_handler.async_emit(
                {
                    "data": json.dumps(data, ensure_ascii=False),
                    "algorithm": algorithm,
                }
            )
        else:
            self.logger.warning(
                "Missing algorithm or data information in raw_data",
                extra={
                    "data": {
                        "message": message,
                        "raw_data": raw_data,
                    }
                },
            )

    def warning(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning level messages."""
        extra = {"data": data} if data else {}
        self.logger.warning(message, extra=extra)

    def critical(self, message: str, data: Optional[Dict[str, Any]] = None):
        """Log critical level messages."""
        extra = {"data": data} if data else {}
        self.logger.critical(message, extra=extra)

    def _get_log_file_path(self) -> str:
        """Return the path to the log file."""
        log_path = f"data/logs/trading_{get_now().strftime('%Y-%m-%d')}.log"
        if not os.path.exists(os.path.dirname(log_path)):
            os.makedirs(os.path.dirname(log_path))
        return log_path


# Optional: Singleton helper
_logger_instance: Optional[Logger] = None


def get_logger() -> Logger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger()
        _logger_instance.connect()
    return _logger_instance


# ======================================= #
#              Example Usage              #
# ======================================= #

if __name__ == "__main__":
    """Run example logger functionality"""
    # Example 1: Using the singleton
    logger = get_logger()

    logger.track(
        "Track trading event, saved to log file and Database",
        {"algorithm": "test", "data": {"key": "value"}},
    )
    logger.info("Simple information message, used to track the process")
    logger.info("Simple information message with data", {"data": "example data"})
    logger.error(
        "Error message",
        {"error": "Error message example"},
    )
    logger.debug("Debugging message with data", {"data": 194.5})

    logger.disconnect()
