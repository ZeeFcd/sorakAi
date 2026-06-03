"""OpenTelemetry wiring for the FastAPI services (Wave 8).

This module is the only place that touches the OTel SDK; the rest of the
codebase calls :func:`get_tracer` + :func:`span` so future changes (e.g.
switching to a different exporter, plugging in metrics, swapping samplers)
land in one file.

Design notes
------------

- **Idempotent**: tests and uvicorn both call ``configure_tracing`` more
  than once. We set the global ``TracerProvider`` only on first call and
  return early after that; the instrumentors below do their own
  ``already_instrumented`` guard.
- **Exporter switch**: the default exporter is ``console`` so a fresh
  dev install gets traces in stdout without any infra. When the user sets
  ``OTEL_EXPORTER_OTLP_ENDPOINT`` (or flips ``OTEL_EXPORTER=otlp``) we
  swap to an OTLP gRPC exporter aimed at Jaeger (compose has an optional
  Jaeger service behind the ``otel`` profile).
- **Sampling**: ``OTEL_SAMPLER_RATIO=1.0`` by default; lower it (e.g.
  ``0.05``) in prod-like deployments. The ratio is parent-based so child
  spans inherit the trace-level decision.
- **No global side effects on import**: tracing is opt-out per service
  (``OTEL_ENABLED``) and only configured from each app's ``lifespan``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from sorakai.common.config import Settings

_logger = logging.getLogger(__name__)
_CONFIGURED = False
_NOOP_INSTRUMENTORS = False


def _build_exporter(settings: Settings) -> SpanExporter:
    """Pick the right exporter for the current settings.

    ``OTEL_EXPORTER=otlp`` (or any non-empty ``OTEL_EXPORTER_OTLP_ENDPOINT``)
    flips us to OTLP/gRPC; everything else falls back to the console
    exporter so ``make dev`` works without a collector in the loop.
    """
    exporter_name = (settings.otel_exporter or "console").lower()
    if exporter_name == "otlp" or settings.otel_exporter_otlp_endpoint:
        endpoint = settings.otel_exporter_otlp_endpoint or "http://localhost:4317"
        _logger.info("OTel: OTLP exporter -> %s", endpoint)
        return OTLPSpanExporter(endpoint=endpoint, insecure=True)
    _logger.info("OTel: console exporter (set OTEL_EXPORTER=otlp to switch)")
    return ConsoleSpanExporter()


def configure_tracing(service_name: str, settings: Settings, *, version: str | None = None) -> None:
    """Install the global TracerProvider for this process.

    Safe to call multiple times: the second + onwards calls early-out so
    we don't stack BatchSpanProcessors on the same provider (which would
    duplicate every span).
    """
    global _CONFIGURED

    if not settings.otel_enabled:
        _logger.info("OTel: disabled via OTEL_ENABLED=false")
        return

    if _CONFIGURED:
        return

    resource = Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name or service_name,
            SERVICE_VERSION: version or "0.0.0",
        }
    )
    sampler = ParentBased(TraceIdRatioBased(float(settings.otel_sampler_ratio)))
    provider = TracerProvider(resource=resource, sampler=sampler)
    provider.add_span_processor(BatchSpanProcessor(_build_exporter(settings)))
    trace.set_tracer_provider(provider)
    _CONFIGURED = True


def instrument_fastapi(app: FastAPI) -> None:
    """Auto-instrument an existing FastAPI app.

    Calling twice on the same app is a no-op (the instrumentor itself
    guards against double-wrapping).
    """
    if not _CONFIGURED:
        return
    try:
        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("OTel FastAPI instrumentation skipped: %s", exc)


def instrument_httpx() -> None:
    """Auto-instrument all ``httpx`` client calls process-wide.

    Idempotent: the instrumentor sets a sentinel after first install and
    silently returns on subsequent calls.
    """
    global _NOOP_INSTRUMENTORS
    if not _CONFIGURED or _NOOP_INSTRUMENTORS:
        return
    try:
        HTTPXClientInstrumentor().instrument()
        _NOOP_INSTRUMENTORS = True
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning("OTel httpx instrumentation skipped: %s", exc)


def get_tracer(name: str = "sorakai") -> Tracer:
    """Return a tracer; if OTel is off we still get the SDK's no-op tracer."""
    return trace.get_tracer(name)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a manual span with the given name + initial attributes.

    Usage::

        async with span("rag.query", session=sid, top_k=k) as s:
            answer = await chain.ainvoke(...)
            s.set_attribute("sources_used", n)

    On unhandled exception the span status is flipped to ERROR and the
    exception recorded; we re-raise so the handler still sees it.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        for key, value in attributes.items():
            if value is None:
                continue
            try:
                s.set_attribute(key, value)
            except Exception:  # pragma: no cover - bad attribute type
                s.set_attribute(key, str(value))
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            raise


def reset_for_tests() -> None:
    """Reset the module-level guard between tests.

    Tests that exercise multiple lifespans (e.g. switching exporters)
    need to be able to re-run ``configure_tracing`` with a fresh provider.
    Not part of the public surface.
    """
    global _CONFIGURED, _NOOP_INSTRUMENTORS
    _CONFIGURED = False
    _NOOP_INSTRUMENTORS = False


__all__ = [
    "configure_tracing",
    "get_tracer",
    "instrument_fastapi",
    "instrument_httpx",
    "reset_for_tests",
    "span",
]
