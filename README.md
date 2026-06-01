# sorakAi

Microservices MVP: **gateway** (8000), **ingest** (8001), **RAG** (8002), **Redis**, **Ollama** (self-hosted LLM), **MLflow** hooks. See `docker-compose.yml` and `k8s/`.

## Environment

Linux-only. Tested on Python 3.12 (see `python:3.12-slim` in [Dockerfile](Dockerfile)).
Create and use a virtualenv (path is up to you, e.g. `.venv` or `<YOUR_VENV>`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt          # runtime + tests + lint
make lint typecheck test                     # full pre-merge check
```

## Run services (dev)

With the venv activated:

```bash
uvicorn sorakai.ingest.app:app --reload --port 8001
uvicorn sorakai.rag.app:app --reload --port 8002
export INGEST_SERVICE_URL=http://127.0.0.1:8001 RAG_SERVICE_URL=http://127.0.0.1:8002
uvicorn sorakai.gateway.app:app --reload --port 8000
```

Set the same `REDIS_URL` on ingest and RAG when running as separate processes.

### Embeddings (ingest + RAG)

`EMBEDDING_PROVIDER` controls how chunks and queries are vectorized:

| Provider | Env | Notes |
|----------|-----|--------|
| `char` (default) | — | Pseudo-vectors; no semantics; good for **tests** / offline. |
| `ollama` | `OLLAMA_EMBED_BASE_URL` (e.g. `http://ollama:11434`), `OLLAMA_EMBEDDING_MODEL` (e.g. `nomic-embed-text`) | **Docker Compose** sets this and pulls the embed model with the chat model. |
| `openai` | `OPENAI_API_KEY`, `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`) | Optional `OPENAI_EMBEDDINGS_BASE_URL` for Azure/proxies (not the Ollama chat URL). |

**Important:** query and stored chunks must use the **same** provider and model dimensions; changing provider after ingesting requires **re-ingesting** or `replace_kb: true`.

### LLM backends (RAG service)

Priority in `sorakai/common/llm.py`:

1. **`OPENAI_BASE_URL`** set → OpenAI-compatible HTTP API (e.g. **Ollama** at `http://127.0.0.1:11434/v1`). `OPENAI_API_KEY` optional (Ollama uses a dummy value if unset). Set **`OPENAI_CHAT_MODEL`** to the Ollama tag (e.g. `llama3.2:1b`).
2. Else **`OPENAI_API_KEY`** set → OpenAI cloud (default base URL).
3. Else → **stub** answer (no model).

Local dev with Ollama on the host: run `ollama serve`, `ollama pull llama3.2:1b`, then start RAG with  
`OPENAI_BASE_URL=http://127.0.0.1:11434/v1` and `OPENAI_CHAT_MODEL=llama3.2:1b`.

### Storage (Redis)

- **Knowledge base** and **chat history** use **different keys** (they are not one blob):
  - Chunks live in Redis hash **`sorakai:kb:chunks`** (one hash *field* per chunk: embedding + text + `doc_id` / `filename`). New documents are appended with **`HSET`** only.
  - Sessions use **`sorakai:chat:<session_id>`** (see `sorakai/common/chat_history.py`).
- **Retrieval** still loads all chunk vectors into the RAG process and runs cosine similarity (fine for small/medium KBs). For **large** corpora or **ANN** search, plug in a **vector database** (Qdrant, Milvus, pgvector, Redis Search with VECTOR, Pinecone, …) and keep Redis for metadata or caching only—details in the module docstring of `sorakai/common/store.py`.

## OpenAPI

- **Versioned specs**: `openapi/*.openapi.json` (CI-checked) and optional `*.openapi.yaml` — regenerate with `python scripts/export_openapi.py --yaml --output openapi`.
- **Runtime**: each service serves `GET /openapi.json` and `/docs`; Docker images also expose **`GET /openapi.bundled.json`** and **`GET /openapi.bundled.yaml`** from the files baked at build (`OPENAPI_DIR`, default `/app/openapi`).
- Details: **`openapi/README.md`**.

## Docker

```bash
docker compose up --build
```

Includes **Ollama**, **MLflow** (UI at **http://127.0.0.1:5000**), Redis, ingest, RAG (Ollama + `MLFLOW_TRACKING_URI`), and gateway. On first start, **`ollama-model`** pulls **`llama3.2:1b`** (can take a few minutes). Use another small model:

```bash
OLLAMA_MODEL=phi3:mini docker compose up --build
```

**OpenAI cloud instead of Ollama:** use a `docker-compose.override.yml` (not committed) to remove `OPENAI_BASE_URL` from `rag`, set `OPENAI_API_KEY`, and drop `depends_on` / `ollama-model` if you want a slimmer stack—or stop the Ollama services and point RAG at the cloud env vars only.
