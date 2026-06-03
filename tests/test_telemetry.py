"""Wave 8 OpenTelemetry plumbing tests.

These exercise the public surface of :mod:`sorakai.common.telemetry`
without ever spinning up an OTLP collector: we install an in-memory span
provider directly, run a manual span, and assert the SDK saw it.

The OTel API forbids replacing the global tracer provider once it's set,
so we **avoid** calling ``configure_tracing`` from the span-recording
tests - that helper is exercised on its own in a dedicated test that
shuts the provider down afterwards.
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from sorakai.common.config import get_settings
from sorakai.common.telemetry import (
    configure_tracing,
    get_tracer,
    instrument_fastapi,
    instrument_httpx,
    reset_for_tests,
    span,
)


def _install_in_memory_provider() -> InMemorySpanExporter:
    """Install a fresh in-memory exporter as the global tracer provider.

    OTel uses a one-shot ``Once`` guard around ``set_tracer_provider``; once
    something has installed a provider in the process the second call
    silently no-ops. For test isolation we reach into the private state
    and reset the guard so each test sees a clean SDK. This is the same
    pattern the OTel test suite uses; nothing here ships to production.
    """
    from threading import Lock

    from opentelemetry.util._once import Once

    trace._TRACER_PROVIDER = None
    trace._TRACER_PROVIDER_SET_ONCE = Once()
    _ = Lock  # silence unused-import: kept to document the Once internals

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def in_memory_exporter() -> InMemorySpanExporter:
    return _install_in_memory_provider()


def test_configure_tracing_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_ENABLED", "false")
    get_settings.cache_clear()
    reset_for_tests()
    configure_tracing("sorakai-test", get_settings())
    # The helper must return cleanly; span() falls back to the SDK's
    # no-op tracer so call sites are safe.
    with span("disabled.span") as s:
        assert s is not None


def test_span_records_attributes_and_status(in_memory_exporter: InMemorySpanExporter) -> None:
    with span("rag.query", session=True, top_k=5) as s:
        s.set_attribute("sources_used", 3)
    spans = in_memory_exporter.get_finished_spans()
    rag_span = next(sp for sp in spans if sp.name == "rag.query")
    assert rag_span.attributes is not None
    assert rag_span.attributes.get("session") is True
    assert rag_span.attributes.get("top_k") == 5
    assert rag_span.attributes.get("sources_used") == 3


def test_span_records_exception_status(in_memory_exporter: InMemorySpanExporter) -> None:
    with pytest.raises(ValueError), span("rag.query.fail"):
        raise ValueError("boom")
    spans = in_memory_exporter.get_finished_spans()
    bad = next(sp for sp in spans if sp.name == "rag.query.fail")
    assert bad.status.status_code.name == "ERROR"
    assert any(ev.name == "exception" for ev in bad.events)


def test_span_drops_none_attributes(in_memory_exporter: InMemorySpanExporter) -> None:
    with span("with_none", actual_value="ok", noisy=None):
        pass
    spans = in_memory_exporter.get_finished_spans()
    s = next(sp for sp in spans if sp.name == "with_none")
    assert s.attributes is not None
    assert "noisy" not in s.attributes
    assert s.attributes.get("actual_value") == "ok"


def test_get_tracer_returns_a_tracer() -> None:
    tracer = get_tracer("sorakai.test")
    assert tracer is not None


def test_instrument_helpers_noop_without_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """``instrument_fastapi`` / ``instrument_httpx`` must early-return when
    OTel was never configured - otherwise importing the apps in tests with
    ``OTEL_ENABLED=false`` would blow up."""
    monkeypatch.setenv("OTEL_ENABLED", "false")
    get_settings.cache_clear()
    reset_for_tests()
    from fastapi import FastAPI

    app = FastAPI()
    instrument_fastapi(app)
    instrument_httpx()


def test_otlp_exporter_chosen_when_endpoint_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the OTLP endpoint flips the exporter even without
    ``OTEL_EXPORTER=otlp`` (the endpoint is the strong signal)."""
    from sorakai.common.telemetry import _build_exporter

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger.local:4317")
    get_settings.cache_clear()
    exporter = _build_exporter(get_settings())
    assert "OTLP" in type(exporter).__name__
