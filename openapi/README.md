# OpenAPI specifications

Generated from FastAPI for **gateway**, **ingest**, and **RAG** (`*.openapi.json` canonical for CI; `*.openapi.yaml` for humans and many API gateways).

## Regenerate (after API changes)

```bash
python scripts/export_openapi.py --yaml --output openapi
```

## CI / contract check

```bash
python scripts/export_openapi.py --check --output openapi
```

Fails with exit code 1 if committed JSON does not match the current code.

## Deployment

### Live spec (always matches running code)

Each service exposes the generated schema from the app:

- `GET /openapi.json` — OpenAPI 3 JSON (FastAPI default)
- `GET /docs` — Swagger UI

### Bundled spec (stable file in the container)

Docker images run `scripts/export_openapi.py` at build time and ship specs under `/app/openapi/`. At runtime:

- `GET /openapi.bundled.json`
- `GET /openapi.bundled.yaml` (if the YAML file was produced at build)

Use these URLs from API portals, ingress controllers, or policy engines that want a **file-backed** contract. Override the directory with **`OPENAPI_DIR`**. **`SORAKAI_SERVICE`** selects which file is served (`gateway`, `ingest`, `rag`).

### Local dev (bundled routes)

```bash
export OPENAPI_DIR=<PATH_TO_REPO>/openapi
uvicorn sorakai.gateway.app:app --port 8000
# curl http://127.0.0.1:8000/openapi.bundled.json
```

### Kubernetes ConfigMap

Publish specs for tools that do not call the pod:

```bash
kubectl -n sorakai create configmap sorakai-openapi \
  --from-file=gateway.openapi.yaml=openapi/gateway.openapi.yaml \
  --from-file=ingest.openapi.yaml=openapi/ingest.openapi.yaml \
  --from-file=rag.openapi.yaml=openapi/rag.openapi.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

Mount the ConfigMap into a sidecar or documentation pod as needed.
