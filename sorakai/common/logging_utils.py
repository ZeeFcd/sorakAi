"""Legacy logging shim.

The canonical implementations live in :mod:`sorakai.core.logging`; this
module is kept so existing imports keep working. Wave 8 added a
``ContextVar``-backed request id; we now bridge that into structlog so
both old call sites (``request_id_ctx.set(...)``) and new ones
(``bind_request_id(...)``) emit the same id.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from sorakai.core.logging import (
    bind_request_id,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_id,
)

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return str(uuid.uuid4())


__all__ = [
    "bind_request_id",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "request_id_ctx",
]
