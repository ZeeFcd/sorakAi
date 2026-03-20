#!/usr/bin/env python3
"""Export OpenAPI 3 schemas for gateway, ingest, and RAG apps."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SERVICES: dict[str, tuple[str, str]] = {
    "gateway": ("sorakai.gateway.app", "create_app"),
    "ingest": ("sorakai.ingest.app", "create_app"),
    "rag": ("sorakai.rag.app", "create_app"),
}


def build_app(module_name: str, factory_name: str):
    mod = importlib.import_module(module_name)
    factory = getattr(mod, factory_name)
    return factory()


def openapi_schema(name: str) -> dict:
    module_name, factory_name = SERVICES[name]
    app = build_app(module_name, factory_name)
    return app.openapi()


def canonical_json(schema: dict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_artifacts(name: str, out_dir: Path, write_yaml: bool) -> None:
    schema = openapi_schema(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{name}.openapi.json"
    json_path.write_text(canonical_json(schema), encoding="utf-8")
    if write_yaml:
        import yaml

        yaml_path = out_dir / f"{name}.openapi.yaml"
        yaml_path.write_text(
            yaml.dump(schema, sort_keys=False, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "openapi",
        help="Output directory",
    )
    parser.add_argument("--yaml", action="store_true", help="Also write *.openapi.yaml files")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if committed JSON specs differ from code (CI)",
    )
    args = parser.parse_args()
    out_dir: Path = args.output

    if args.check:
        drift = False
        for name in SERVICES:
            fresh = canonical_json(openapi_schema(name))
            json_path = out_dir / f"{name}.openapi.json"
            if not json_path.exists():
                print(f"Missing {json_path}", file=sys.stderr)
                drift = True
                continue
            if json_path.read_text(encoding="utf-8") != fresh:
                print(f"OpenAPI drift: {json_path} (run: python scripts/export_openapi.py --yaml)", file=sys.stderr)
                drift = True
        return 1 if drift else 0

    for name in SERVICES:
        write_artifacts(name, out_dir, write_yaml=args.yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
