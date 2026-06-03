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

**Dim-guard (Wave 2).** The KB stamps the `{provider, model, dim}` triple it was
built with into `sorakai:kb:meta` on first ingest. Every later ingest and every
query is verified against that triple; a mismatch returns **`409 Conflict`** with
a JSON body explaining `expected` vs `actual`. The fix is either to re-ingest the
corpus with the current provider/model, or to POST `/v1/documents` again with
`replace_kb: true` (which atomically clears the chunks and the meta record).
This replaces the silent zero-padding that used to mask cross-model vector
arithmetic.

### Ollama embeddings tuning

The Ollama adapter (`sorakai/infra/embeddings/ollama.py`) is batched and
concurrent. Defaults are sensible for a single-host setup; the knobs:

| Env | Default | What it does |
|-----|---------|--------------|
| `OLLAMA_EMBED_BATCH` | `64` | Max inputs per `/api/embed` request body. |
| `OLLAMA_EMBED_CONCURRENCY` | `4` | Max in-flight embed requests (bounded by an asyncio.Semaphore). |
| `OLLAMA_EMBED_TIMEOUT_SECONDS` | `60.0` | Per-request timeout, separate from the gateway proxy timeout. |
| `OLLAMA_EMBED_USE_BATCH_ENDPOINT` | `true` | Set to `false` to force the legacy per-input `/api/embeddings` (older Ollama). The adapter also falls back automatically on a 404. |

Empty / whitespace-only chunks are filtered before the network call and
zero-vectors are reinserted at the original indices, so the returned list
always aligns 1:1 with the input list.

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

### Chunker (Wave 3)

`POST /v1/documents` runs the content through a LangChain
`RecursiveCharacterTextSplitter` (`sorakai/common/ingest.py`). Splitter
selection is language-aware:

| Hint | Splitter |
|------|----------|
| `mime_type=text/x-python` or `filename=*.py / *.pyi` | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` (keeps `def` / `class` / blocks intact when possible) |
| `mime_type=text/markdown` or `filename=*.md / *.markdown / *.mdx` | `from_language(Language.MARKDOWN)` (prefers heading + paragraph boundaries) |
| anything else | generic recursive splitter (newlines -> sentences -> words -> chars) |

`mime_type` wins over `filename` so explicit caller intent beats guessing.
Adding a new language is one entry in each of `_FILENAME_LANGUAGE_MAP` and
`_MIME_LANGUAGE_MAP` plus one in `_LANGUAGE_STRATEGY`.

Request knobs (see `DocumentIngestRequest`):

| Field | Default | Notes |
|-------|---------|-------|
| `chunk_size` | `500` | Hard limit per chunk; bounded `[50, 10_000]`. |
| `chunk_overlap` | `50` | Characters shared between neighbouring chunks; must be strictly `< chunk_size` (Pydantic validator enforces it). |
| `mime_type` | `null` | Optional override for splitter selection; charset parameters (e.g. `; charset=utf-8`) are stripped before lookup. |

Stored chunks carry `{doc_id, filename, chunk_index, chunk_total, mime}` for
downstream filtering / attribution. Reads tolerate legacy entries that
predate the `chunk_total` + `mime` fields (Wave 3 onwards) - missing fields
read as `-1` / `null` respectively.

### Document API (Wave 4)

Beyond `POST /v1/documents`, the ingest service exposes:

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/v1/documents` | List every document in the KB with `chunk_count` and `mime`. |
| `DELETE` | `/v1/documents/{doc_id}` | Drop every chunk for that document; returns the count removed. 404 if unknown. |

The gateway proxies both as `/api/v1/documents` and `/api/v1/documents/{doc_id}`.

Re-ingest semantics: `POST /v1/documents` with an existing `document_id`
**atomically replaces** that document's chunks (Redis pipeline pairs
`HDEL ck:<doc_id>:*` with the new `HSET`). No duplicates, no half-written state.

`replace_kb=true` runs the same atomicity contract at the whole-KB level
(single `MULTI`/`EXEC` of `DEL sorakai:kb:chunks` + `HSET` of the new doc),
and only after the chunks land does the dim-guard meta get rewritten. If the
meta swap fails for any reason, the next request gets a clean 409 instead of
the silent corruption the meta-first ordering used to produce.

### Vector store (Wave 5)

`sorakai/infra/vector_store/` exposes the same factory pattern as the LLM /
embeddings layers: callers depend on the `VectorStore` Protocol and pick a
backend with `VECTOR_STORE`.

| `VECTOR_STORE` | Backend | Notes |
|----------------|---------|-------|
| `memory` *(test default)* | `KnowledgeStoreVectorStore(InMemoryKnowledgeStore())` | Single-process; resets on restart. |
| `redis` *(prod default)* | `KnowledgeStoreVectorStore(RedisKnowledgeStore(REDIS_URL))` | Same Wave 4 hash layout (`ck:<doc_id>:<chunk_index>`). |
| `qdrant` | `QdrantVectorStore` | Real ANN backend; collection per env (`QDRANT_COLLECTION`, default `sorakai_kb`), cosine distance, payload carries the Wave 3 chunk metadata. |

The Protocol surface is intentionally small:

```python
class VectorStore(Protocol):
    async def upsert(self, docs: list[VectorDoc]) -> None: ...
    async def delete_doc(self, doc_id: str) -> int: ...
    async def list_docs(self) -> list[DocSummary]: ...
    async def search(self, query_vec, k, filters=None) -> list[Hit]: ...
    async def ping(self) -> bool: ...
    async def aclose(self) -> None: ...
```

Re-ingesting the same `doc_id` overwrites cleanly on every backend (Wave 4
contract carried into the Protocol). For Qdrant we drop any existing points
matching `doc_id` before upserting the new batch so a smaller re-ingest
doesn't orphan tail chunks.

#### Adding a new vector store

1. Drop an adapter file under `sorakai/infra/vector_store/<name>.py` that
   satisfies the Protocol.
2. Add the backend literal to `VectorStoreBackend` in `sorakai/common/config.py`.
3. Register a builder in `sorakai/infra/vector_store/factory.py`:
   ```python
   VECTOR_STORE_REGISTRY["milvus"] = _build_milvus
   ```
4. Add it to the `vstore` matrix in `tests/test_vector_store.py` — the
   behavioural matrix runs every test against every backend automatically.

#### Running Qdrant locally

```bash
docker compose up -d qdrant
# then point ingest + rag at it:
VECTOR_STORE=qdrant QDRANT_URL=http://127.0.0.1:6333 make dev
```

In tests the `qdrant-client` library supports the in-process `":memory:"`
transport (`AsyncQdrantClient(":memory:")`) so the suite exercises the real
Qdrant code path without needing a running server.

### Storage (Redis, Wave 4 layout)

Chunks are stored under a single hash `sorakai:kb:chunks`. Each field is named
`ck:<doc_id>:<chunk_index>` (Wave 4) so:

- Re-ingesting the same `doc_id` only writes the new fields and `HDEL`s the
  stale tail (or zero fields if the chunk count grows) — never duplicates.
- `DELETE /v1/documents/{doc_id}` is a single `HSCAN` + multi-field `HDEL`.
- Listing is one `HGETALL` plus an in-process group-by.
- Reads tolerate pre-Wave-4 `ck:<uuid>` keys (and infer `chunk_total` from
  siblings sharing the same `doc_id`), so live KBs survive an upgrade.

Other Redis keys used by the project:

- `sorakai:kb:meta` — the dim-guard identity hash (`provider`, `model`, `dim`).
- `sorakai:chat:<session_id>` — per-session chat history (see `sorakai/common/chat_history.py`).

Retrieval still loads all chunk vectors into the RAG process and runs cosine
similarity (fine for small / medium KBs). For large corpora or ANN search,
swap the storage backend out via the Wave 5 `VectorStore` Protocol
(Qdrant / Milvus / pgvector / Redis Search-VECTOR / Pinecone / …) and keep
Redis as a metadata + cache layer.

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

