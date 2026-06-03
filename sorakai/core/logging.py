"""Logging utilities backed by structlog (Wave 8).

We previously used the stdlib ``logging`` module directly; Wave 8 moves
to ``structlog`` so every log line is a structured event with bound
context vars (``request_id``, OTel ``trace_id`` / ``span_id``), and
renders as either JSON (``LOG_FORMAT=json``, the default in containers)
or a human-friendly tinted console format (``LOG_FORMAT=console``).

The public surface stays tiny:

- :func:`configure_logging` - install/reconfigure once per process; safe
  to call from tests and from every service's ``lifespan``.
- :func:`get_logger` - returns a ``structlog`` :class:`BoundLogger` that
  also accepts printf-style positional args (so Wave 1-7 call sites keep
  working without rewriting every ``logger.info("foo %s", x)``).
- :func:`bind_request_id` / :func:`clear_request_context` - middleware
  helpers; every subsequent log line in the same task gets the bound
  vars automatically.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from opentelemetry import trace
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars
from structlog.types import EventDict, Processor, WrappedLogger

_CONFIGURED: bool = False


def _add_otel_trace_context(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Attach the current OTel trace_id + span_id to every log record.

    The IDs are zero when no span is active; we strip those so the
    output stays clean during boot / shutdown.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context() if span else None
    if ctx is not None and ctx.is_valid:
        event_dict.setdefault("trace_id", format(ctx.trace_id, "032x"))
        event_dict.setdefault("span_id", format(ctx.span_id, "016x"))
    return event_dict


def _resolve_renderer(log_format: str) -> Processor:
    """Pick the right final processor (renderer).

    ``json`` -> :class:`structlog.processors.JSONRenderer` (production /
    container friendly). ``console`` -> :class:`structlog.dev.ConsoleRenderer`
    (developer friendly, colourful when stderr is a TTY).
    """
    fmt = (log_format or "json").lower()
    if fmt == "console":
        return structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    return structlog.processors.JSONRenderer()


def configure_logging(level: str | int | None = None, *, log_format: str | None = None) -> None:
    """Install the structlog config and route stdlib loggers through it.

    Wave 8 uses the **stdlib-bridge** idiom: a single
    :class:`structlog.stdlib.ProcessorFormatter` runs the processor chain
    so a ``logger.info("foo")`` from sorakai code and a stdlib
    ``logging.getLogger("uvicorn").info("bar")`` end up rendered identically.

    Resolution order for ``level``:
    1. Explicit argument.
    2. ``LOG_LEVEL`` env var.
    3. ``"INFO"``.

    Resolution order for ``log_format``:
    1. Explicit argument.
    2. ``LOG_FORMAT`` env var.
    3. ``"json"`` (zero-config containers ship JSON; flip to ``console``
       locally with ``LOG_FORMAT=console``).
    """
    global _CONFIGURED

    resolved_level = _coerce_level(level if level is not None else os.environ.get("LOG_LEVEL", "INFO"))
    resolved_format = log_format if log_format is not None else os.environ.get("LOG_FORMAT", "json")

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_otel_trace_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = _resolve_renderer(resolved_format)

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processor=renderer,
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Replace existing handlers on reconfigure so successive tests don't
    # accumulate duplicates.
    root.handlers = [handler]
    root.setLevel(resolved_level)
    logging.getLogger("sorakai").setLevel(resolved_level)

    _CONFIGURED = True


def _coerce_level(level: str | int) -> int:
    """Translate a user-facing level (``"INFO"`` / ``20``) into stdlib int."""
    if isinstance(level, int):
        return level
    return logging.getLevelNamesMapping().get(level.upper(), logging.INFO)


def get_logger(name: str | None = None) -> Any:
    """Return a structlog :class:`BoundLogger` bound to ``name``.

    Compatible with the legacy stdlib API: ``logger.info("text %s", arg)``
    is rewritten by the PositionalArgumentsFormatter processor, and
    ``logger.exception(...)`` still attaches the traceback through
    :func:`structlog.processors.format_exc_info`.
    """
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()


def bind_request_id(request_id: str) -> None:
    """Bind ``request_id`` to the current async context.

    Called from the FastAPI middleware; every log line emitted further
    down the request chain gets the same id automatically through
    :func:`structlog.contextvars.merge_contextvars`.
    """
    bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    """Clear all bound context vars; pair with :func:`bind_request_id` in
    the middleware's ``finally`` block."""
    clear_contextvars()


def get_request_id() -> str | None:
    """Read the currently bound request id (returns ``None`` outside a request)."""
    ctx = get_contextvars()
    rid = ctx.get("request_id")
    return rid if isinstance(rid, str) else None


__all__ = [
    "bind_request_id",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "get_request_id",
]
