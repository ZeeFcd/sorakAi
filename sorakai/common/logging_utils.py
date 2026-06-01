"""Legacy logging shim.

The canonical implementations live in :mod:`sorakai.core.logging`; this
module is kept around so existing imports (``from sorakai.common.logging_utils
import get_logger``) keep working while the codebase migrates wave-by-wave.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

from sorakai.core.logging import configure_logging, get_logger

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return str(uuid.uuid4())


__all__ = ["configure_logging", "get_logger", "new_request_id", "request_id_ctx"]
