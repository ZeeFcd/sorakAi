# Multi-service image: set SORAKAI_SERVICE to gateway | ingest | rag
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sorakai ./sorakai
COPY pyproject.toml .
COPY scripts/export_openapi.py ./scripts/export_openapi.py

RUN mkdir -p openapi && python scripts/export_openapi.py --yaml --output openapi

ENV OPENAPI_DIR=/app/openapi

ARG SORAKAI_SERVICE=gateway
ENV SORAKAI_SERVICE=${SORAKAI_SERVICE}
ENV PYTHONUNBUFFERED=1

# Default port; override per Deployment (8000 gateway, 8001 ingest, 8002 rag)
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${PORT}/health" || exit 1

CMD sh -c 'exec uvicorn "sorakai.${SORAKAI_SERVICE}.app:app" --host 0.0.0.0 --port "${PORT}"'
