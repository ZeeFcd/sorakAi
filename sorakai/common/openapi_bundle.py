"""Serve committed/build-time OpenAPI files alongside live /openapi.json."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.responses import FileResponse


def register_bundled_openapi_routes(app: FastAPI, service_name: str) -> None:
    """
    If OPENAPI_DIR/<service>.openapi.yaml exists, expose GET /openapi.bundled.yaml
    (and .json) for gateways, contract portals, and sidecars that need a stable file URL.
    """
    root = Path(os.environ.get("OPENAPI_DIR", "/app/openapi"))
    name = (os.environ.get("SORAKAI_SERVICE") or service_name).lower().strip()
    yaml_path = root / f"{name}.openapi.yaml"
    json_path = root / f"{name}.openapi.json"

    if yaml_path.is_file():

        @app.get("/openapi.bundled.yaml", include_in_schema=False)
        async def bundled_openapi_yaml() -> FileResponse:
            return FileResponse(yaml_path, media_type="application/yaml")

    if json_path.is_file():

        @app.get("/openapi.bundled.json", include_in_schema=False)
        async def bundled_openapi_json() -> FileResponse:
            return FileResponse(json_path, media_type="application/json")
