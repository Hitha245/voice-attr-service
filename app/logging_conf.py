"""Structured logging setup.

Every request gets a request_id and a processing_ms timing field logged as
structured JSON (or plain text in LOG_JSON=false for local dev readability),
which is what Task 4 ("observability hooks: logging, timing") asks for.

Crucially: we log audio *metadata* (duration, sample rate, quality label,
byte size) and *results* (predictions, confidences, timings) — never the
raw audio bytes or any derived waveform. See README > Privacy.
"""
from __future__ import annotations

import logging
import sys

from app.config import settings

try:
    from pythonjsonlogger import jsonlogger
    _HAS_JSON_LOGGER = True
except ImportError:  # pragma: no cover
    _HAS_JSON_LOGGER = False


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)

    if settings.LOG_JSON and _HAS_JSON_LOGGER:
        fmt = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(processing_ms)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
            defaults={"request_id": None, "processing_ms": None},
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )

    handler.setFormatter(fmt)
    root.handlers = [handler]

    # Quiet down noisy third-party loggers unless we're at DEBUG.
    if settings.LOG_LEVEL.upper() != "DEBUG":
        for noisy in ("uvicorn.access", "httpx", "urllib3", "transformers"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


class RequestLogAdapter(logging.LoggerAdapter):
    """Injects request_id/processing_ms into every log line without every
    call site having to remember to pass `extra=`."""

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {})
        kwargs["extra"].setdefault("request_id", self.extra.get("request_id"))
        kwargs["extra"].setdefault("processing_ms", self.extra.get("processing_ms"))
        return msg, kwargs
