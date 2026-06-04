"""Regression tests for docker-compose launch wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yml"


def _compose() -> dict[str, Any]:
    payload = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def test_qdrant_does_not_require_shell_tools_inside_database_image() -> None:
    """Qdrant image does not reliably ship wget/curl/bash."""
    services = _compose()["services"]
    qdrant = services["qdrant"]
    assert "healthcheck" not in qdrant
    assert "qdrant-ready" in services
    assert services["qdrant-ready"]["image"].startswith("curlimages/curl:")


def test_ingest_and_rag_wait_for_qdrant_ready_sidecar() -> None:
    services = _compose()["services"]
    for service_name in ("ingest", "rag"):
        depends_on = services[service_name]["depends_on"]
        assert depends_on["qdrant-ready"]["condition"] == "service_completed_successfully"


def test_ui_service_installs_from_requirements_ui_txt() -> None:
    services = _compose()["services"]
    ui = services["ui"]
    env = ui["environment"]
    volumes = "\n".join(ui["volumes"])
    command = "\n".join(ui["command"])
    assert env["PYTHONPATH"] == "/app"
    assert "requirements-ui.txt" in volumes
    assert "pip install --quiet -r requirements-ui.txt" in command


def test_local_compose_uses_host_network_on_linux() -> None:
    """Avoid Docker's published-port proxy path; it broke on older compose plugins."""
    services = _compose()["services"]
    for service_name in ("mlflow", "redis", "qdrant", "ollama", "ingest", "rag", "gateway"):
        service = services[service_name]
        assert service["network_mode"] == "host"
        assert "ports" not in service


def test_otel_endpoint_is_empty_unless_otlp_is_requested() -> None:
    """Setting the endpoint implicitly switches tracing to OTLP."""
    services = _compose()["services"]
    for service_name in ("ingest", "rag", "gateway"):
        env = services[service_name]["environment"]
        assert env["OTEL_EXPORTER"] == "${OTEL_EXPORTER:-console}"
        assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "${OTEL_EXPORTER_OTLP_ENDPOINT:-}"
