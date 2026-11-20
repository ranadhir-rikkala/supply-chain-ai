"""JSON logging with request ids."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime

REQUEST_ID_HEADER = "X-Request-ID"
# Only safe characters, so a header value cannot inject text into the logs.
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

STANDARD_RECORD_FIELDS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def resolve_request_id(candidate: str | None) -> str:
    """Use the caller's X-Request-ID if it is safe, otherwise create a new one."""
    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return str(uuid.uuid4())


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line with the request id."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record as JSON, including any extra fields."""
        document = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": os.getenv("SERVICE_NAME", "forecast"),
        }
        request_id = request_id_var.get()
        if request_id:
            document["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in STANDARD_RECORD_FIELDS and not key.startswith("_"):
                document[key] = value
        if record.exc_info:
            document["exception"] = self.formatException(record.exc_info)
        return json.dumps(document, default=str)


def configure_logging(level: str | None = None) -> None:
    """Send all logs, including uvicorn's, to stdout as JSON."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level or os.getenv("LOG_LEVEL", "INFO"))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "pika"):
        logging.getLogger(name).handlers[:] = []
        logging.getLogger(name).propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
