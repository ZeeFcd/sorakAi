"""Wave 8 structlog migration tests.

The legacy ``sorakai/core/logging.py`` returned stdlib ``Logger`` instances;
Wave 8 swaps those for structlog ``BoundLogger`` wrappers that emit a
structured event per call (JSON in containers, console-tinted on a TTY).
These tests pin the new contract.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pytest
import structlog

from sorakai.core.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_id,
)


@pytest.fixture(autouse=True)
def _reset_structlog() -> Any:
    """Each test gets a fresh configuration so a previous test's renderer
    can't bleed into the next assertion."""
    yield
    structlog.reset_defaults()
    clear_request_context()


def _capture_root_handler() -> tuple[io.StringIO, Any]:
    """Replace the root handler with one writing to a StringIO."""
    buf = io.StringIO()
    root = logging.getLogger()
    new_handler = logging.StreamHandler(buf)
    if root.handlers:
        new_handler.setFormatter(root.handlers[0].formatter)
    root.handlers = [new_handler]
    return buf, new_handler


def test_get_logger_emits_json_when_format_is_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging("INFO", log_format="json")
    buf, _ = _capture_root_handler()

    logger = get_logger("sorakai.test")
    logger.info("hello", k=1)

    text = buf.getvalue().strip()
    assert text, "structlog produced no output"
    parsed = json.loads(text)
    assert parsed["event"] == "hello"
    assert parsed["k"] == 1
    assert parsed["level"] == "info"
    assert parsed["logger"] == "sorakai.test"
    assert "timestamp" in parsed


def test_get_logger_console_format_is_human_friendly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "console")
    configure_logging("INFO", log_format="console")
    buf, _ = _capture_root_handler()

    logger = get_logger("sorakai.test")
    logger.info("hello", k=1)

    text = buf.getvalue()
    assert "hello" in text
    assert "sorakai.test" in text
    # The console renderer does NOT emit raw JSON braces.
    assert not text.lstrip().startswith("{")


def test_printf_style_args_keep_working() -> None:
    """Existing call sites use ``logger.info('foo %s', x)``; the
    PositionalArgumentsFormatter processor must rewrite that into the
    event string so we don't break Wave 1-7 code."""
    configure_logging("INFO", log_format="json")
    buf, _ = _capture_root_handler()

    logger = get_logger("sorakai.test")
    logger.info("user=%s top_k=%d", "alice", 5)

    parsed = json.loads(buf.getvalue().strip())
    assert parsed["event"] == "user=alice top_k=5"


def test_bind_request_id_propagates_through_contextvars() -> None:
    configure_logging("INFO", log_format="json")
    buf, _ = _capture_root_handler()

    bind_request_id("req-xyz")
    logger = get_logger("sorakai.test")
    logger.info("with-context")

    parsed = json.loads(buf.getvalue().strip())
    assert parsed["request_id"] == "req-xyz"
    assert get_request_id() == "req-xyz"


def test_clear_request_context_drops_request_id() -> None:
    configure_logging("INFO", log_format="json")
    buf, _ = _capture_root_handler()

    bind_request_id("req-zzz")
    clear_request_context()
    logger = get_logger("sorakai.test")
    logger.info("after-clear")

    parsed = json.loads(buf.getvalue().strip())
    assert "request_id" not in parsed
    assert get_request_id() is None


def test_stdlib_loggers_are_routed_through_structlog() -> None:
    """uvicorn / opentelemetry / mlflow log through stdlib; they must end
    up in the same JSON line shape so observability tooling stays uniform."""
    configure_logging("INFO", log_format="json")
    buf, _ = _capture_root_handler()

    logging.getLogger("uvicorn.access").warning("upstream timeout url=%s", "/v1/query")
    parsed = json.loads(buf.getvalue().strip())
    assert parsed["logger"] == "uvicorn.access"
    assert parsed["level"] == "warning"
    assert "upstream timeout url=/v1/query" in parsed["event"]


def test_exception_traceback_serialised_into_event() -> None:
    """``logger.exception('boom: %s', exc)`` must keep the traceback so we
    don't lose debug context after migrating."""
    configure_logging("ERROR", log_format="json")
    buf, _ = _capture_root_handler()

    logger = get_logger("sorakai.test")
    try:
        raise ValueError("planned")
    except ValueError as exc:
        logger.exception("boom: %s", exc)

    parsed = json.loads(buf.getvalue().strip())
    assert parsed["event"].startswith("boom: planned")
    assert "Traceback" in parsed.get("exception", "")
