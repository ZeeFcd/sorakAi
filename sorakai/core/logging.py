"""Logging utilities.

The current MVP obtained loggers without ever calling :func:`logging.basicConfig`,
which silently dropped every ``sorakai.*`` ``logger.info(...)`` line because the
root logger defaults to ``WARNING``. :func:`configure_logging` is the single
source of truth - it is called once per service from its FastAPI ``lifespan``.

Future logging upgrades (structlog, JSON output, OTel correlation) only need to
change this module; call sites continue to use :func:`get_logger`.
"""

from __future__ import annotations

import logging
import os
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s %(message)s"
_CONFIGURED: bool = False


def get_logger(name: str) -> logging.Logger:
    """Return a project-scoped logger.

    Prefer ``get_logger(__name__)`` over ``logging.getLogger(__name__)`` so
    the import surface for future logging swaps stays small.
    """
    return logging.getLogger(name)


def configure_logging(level: str | int | None = None) -> None:
    """Install a sensible default logging config for sorakAi services.

    Called once per service from its FastAPI ``lifespan``. Safe to call
    multiple times - subsequent calls only adjust the ``sorakai`` logger
    level so we never fight with uvicorn's own root handlers.

    The level is resolved in this order:

    1. Explicit ``level`` argument.
    2. ``LOG_LEVEL`` environment variable.
    3. ``"INFO"``.
    """
    global _CONFIGURED

    resolved = level if level is not None else os.environ.get("LOG_LEVEL", "INFO")

    if not _CONFIGURED:
        logging.basicConfig(level=resolved, format=_DEFAULT_FORMAT)
        _CONFIGURED = True

    logging.getLogger("sorakai").setLevel(resolved)
