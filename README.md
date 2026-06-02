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

### Fully local - never calls a cloud LLM provider

sorakAi only ships adapters for local Ollama (chat + embeddings) and a deterministic
test stub. There are no cloud SDKs in `requirements.txt` and no `OPENAI_*` env vars.
The architecture is provider-pluggable behind LangChain's `BaseChatModel` and
`Embeddings` interfaces, so swapping or adding a host is one new file plus one
registry entry - see "Adding a new provider" below.

| Layer | Default provider | Env knob | Notes |
|-------|------------------|----------|-------|
| Chat model | `ollama` | `LLM_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL` | Tests flip `LLM_PROVIDER=stub` via conftest. |
| Embeddings | `ollama` | `EMBEDDING_PROVIDER`, `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL` | `EMBEDDING_PROVIDER=char` gives offline pseudo-vectors for tests. |

Local dev: `ollama serve`, `ollama pull llama3.2:1b`, `ollama pull nomic-embed-text`,
then the RAG / ingest services Just Work with their defaults.

**Important:** query and stored chunks must use the **same** embeddings provider
and model dimensions. Changing provider after ingesting requires re-ingesting or
`replace_kb: true` (Wave 2 of the overhaul plan adds an automatic dim-guard).

### Adding a new provider

The factories live under `sorakai/infra/llm/` and `sorakai/infra/embeddings/` and
are tiny:

```python
# sorakai/infra/llm/<your_provider>.py
from langchain_<your_provider> import Chat<YourProvider>
from sorakai.common.config import Settings
from sorakai.infra.llm.base import BaseChatModel

def build_<your_provider>_chat(settings: Settings) -> BaseChatModel:
    return Chat<YourProvider>(model=settings.<your_provider>_chat_model, ...)
```

```python
# sorakai/infra/llm/factory.py  (add one line)
from sorakai.infra.llm.<your_provider> import build_<your_provider>_chat
CHAT_MODEL_REGISTRY["<your_provider>"] = build_<your_provider>_chat
```

Then extend `LLMProvider = Literal["ollama","stub",...]` in `sorakai/common/config.py`
and you're done; no chain, agent, ingest, or RAG handler touches the change.

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

