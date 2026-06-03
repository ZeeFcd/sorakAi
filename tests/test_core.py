"""Smoke tests for the new ``sorakai.core`` package introduced in Wave 0."""

from __future__ import annotations

import logging

from sorakai.core import (
    ConfigError,
    EmbeddingError,
    LLMError,
    RetrievalError,
    SorakaiError,
    StoreError,
    configure_logging,
    get_logger,
)


def test_error_hierarchy_is_catchable_as_base():
    """Every typed error must be catchable as the single ``SorakaiError`` base."""
    for cls in (ConfigError, StoreError, LLMError, EmbeddingError, RetrievalError):
        try:
            raise cls("boom")
        except SorakaiError as exc:
            assert isinstance(exc, cls)
            assert isinstance(exc, SorakaiError)
        else:  # pragma: no cover - defensive, raise always triggers except
            raise AssertionError(f"{cls.__name__} did not propagate as SorakaiError")


def test_get_logger_returns_named_logger():
    """Wave 8 swapped stdlib loggers for structlog BoundLoggers.

    The returned object is no longer ``logging.Logger`` but still exposes
    the standard ``.info`` / ``.warning`` API and carries the logger name
    on the structlog proxy so the ``add_logger_name`` processor produces it.
    """
    logger = get_logger("sorakai.test")
    assert callable(getattr(logger, "info", None))
    assert callable(getattr(logger, "warning", None))
    assert getattr(logger, "name", None) == "sorakai.test"
    # stdlib loggers with the same name still answer normally.
    assert logging.getLogger("sorakai.test").name == "sorakai.test"


def test_configure_logging_sets_sorakai_level(monkeypatch):
    """``configure_logging`` must lift the ``sorakai`` logger above WARNING.

    Before Wave 0 nothing called :func:`logging.basicConfig`, so ``logger.info(...)``
    on ``sorakai.*`` was silently dropped. This test pins the contract.
    """
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    configure_logging("DEBUG")
    assert logging.getLogger("sorakai").getEffectiveLevel() == logging.DEBUG

    configure_logging("WARNING")
    assert logging.getLogger("sorakai").getEffectiveLevel() == logging.WARNING


def test_configure_logging_reads_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    configure_logging()
    assert logging.getLogger("sorakai").getEffectiveLevel() == logging.ERROR
