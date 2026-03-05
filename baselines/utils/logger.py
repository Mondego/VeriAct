import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone


_SKIP_ATTRS = frozenset([
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message",
    "pathname", "process", "processName", "relativeCreated",
    "thread", "threadName", "exc_info", "exc_text", "stack_info",
    "taskName",
])


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _SKIP_ATTRS:
                log_data[key] = value

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):

    def format(self, record):
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        color = colors.get(record.levelname, "")

        extras = []
        for key, value in record.__dict__.items():
            if key not in _SKIP_ATTRS:
                extras.append(f"{key}={value}")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        log_line = f"{timestamp} - {color}{record.levelname:8s}{reset} - [{record.threadName}] - {record.name} - {record.getMessage()}{extra_str}"

        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)

        return log_line


def create_logger(name, thread_id, output_dir):

    logs_dir = os.path.join(output_dir, "logs")
    Path(logs_dir).mkdir(exist_ok=True)
    
    logger_name = f"{name}.Thread-{thread_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_file = f"{logs_dir}/{name}.Thread-{thread_id}.jsonl"
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file
