# Changelog

All notable changes to **sorakAi** land here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-03

The "local RAG + agent overhaul" release. Twelve waves (0-11) turned
the original openai-only prototype into a fully local, provider-agnostic
RAG + agent platform with an evaluation harness, OTel observability,
a hardened gateway, and a Streamlit chat UI.

### Breaking changes

- **OpenAI is gone.** All cloud-provider code, env vars (`OPENAI_API_KEY`,
  `OPENAI_*`), and dependencies were removed in Wave 1. The default
  install is fully local against Ollama. **Migration:** set
  `LLM_PROVIDER=ollama` + `EMBEDDING_PROVIDER=ollama` (the new
  defaults), run `ollama pull llama3.2:1b nomic-embed-text`, and remove
  any leftover `OPENAI_*` from your `.env`. Or add an out-of-tree
  provider via `register_chat_model` / `register_embeddings`
  (see [`docs/providers.md`](docs/providers.md)).
- **API surface consolidation (Wave 10).** The gateway's canonical
  surface is now `/v1/*`; `/api/v1/*` paths still resolve via
  `308 Permanent Redirect` for one release. New integrations should
  target `/v1/*` directly.
- **Settings.populate_by_name=True (Wave 10).** Means a stray env var
  whose name *happens* to collide with a settings field name (rather
  than its alias) is now picked up. Audit your `.env` for any keys you
  did not intend to be read.

### Added

- **Provider-agnostic infra (Wave 2).** New factories +
  `register_*` hooks under `sorakai/infra/{llm,embeddings,vector_store}/`.
  Registered today: `ollama`, `stub` (LLM); `ollama`, `char`
  (embeddings); `qdrant`, `redis`, `memory` (vector stores).
- **Smarter chunker (Wave 3).** Markdown- and code-aware token-budgeted
  chunker with deterministic IDs, persisted under `KBMeta` so old and
  new ingests stay compatible.
- **Document API (Wave 4).** Stable `/v1/documents` CRUD on ingest, with
  `DocumentIngestRequest` / `DocumentListResponse` schemas, idempotent
  upserts via `document_id`, and a sibling `KnowledgeStore` interface.
- **Vector store abstraction (Wave 5).** `KnowledgeStoreVectorStore`,
  `RedisKnowledgeStore`, and a native Qdrant adapter (`qdrant` is the
  compose default).
- **LCEL RAG chain (Wave 6).** `build_rag_chain` returns a chain +
  retriever pair; hybrid retrieval (BM25 + vector with RRF), lazy BM25
  indexing for dynamic corpora, pluggable reranker protocol, history
  adapter (`SorakaiChatMessageHistory`) bridging
  `RedisChatHistoryStore` to LangChain's `BaseChatMessageHistory`.
- **LangGraph agent (Wave 7).** `build_agent_graph` wires a
  route -> retrieve -> grade -> generate -> critique loop with a
  `ToolRegistry` (`KBSearchTool`, `CalcTool` with safe AST evaluation,
  feature-flagged `WebSearchTool` stub). `/v1/agent` and
  `/v1/agent/stream` (SSE) endpoints expose the graph.
- **Observability (Wave 8).** OpenTelemetry auto-instrumentation for
  FastAPI + httpx, manual chain/agent spans, structlog with request-id
  binding, `MlflowChainCallback` that logs latency / token counts /
  retrievals / tool calls per run.
- **Evaluation harness (Wave 9).** `scripts/eval.py` runs the chain or
  agent against `tests/eval/golden.jsonl` (16 cases), scores with
  `answer_contains_expected` and `context_precision_at_k`, optionally
  logs to MLflow, and gates regressions via `--min-pass-rate`. The
  optional `ragas` extra is wired in but defaults off.
- **Gateway hardening (Wave 10).** Shared
  `sorakai/common/middleware.py` (CORS, request-id, exception handler,
  request-size limit), `sorakai/common/security.py` (bearer auth +
  slowapi rate limit, Redis-backed when `REDIS_URL` is set, in-memory
  otherwise).
- **Streamlit chat UI (Wave 10).** `ui/streamlit_app.py` talks to the
  gateway via `/v1/query` and `/v1/agent`. Optional dep under
  `requirements-ui.in`; `--profile ui` adds a Streamlit service to
  `docker-compose`.
- **Dev ergonomics (Wave 11).** `scripts/dev_up.sh` brings up compose,
  waits for every `/health` to go green, then runs `scripts/seed.py` to
  ingest a tiny sample corpus and smoke a sample query. `make dev`
  wraps the lot.
- **Docs (Wave 11).** [`docs/providers.md`](docs/providers.md) lists the
  registered adapters and includes a 10-line template for adding a new
  one. README rewritten with a Mermaid architecture diagram, an
  explicit "fully local, never calls a cloud LLM provider" statement,
  and quickstart, eval, agent, and provider howtos.

### Changed

- Mostly internal: ten waves of refactors. The visible deltas are the
  ones called out under **Added** and **Breaking changes**; everything
  else (Pydantic v2 migration, switching to structlog, removing
  `BLE001` blanket excepts) was line-noise that's easier to read in
  the per-wave commit messages.

### Removed

- All OpenAI provider code, env vars, and dependencies (Wave 1).

### Fixed

- Too many to enumerate; tracked in the per-wave PR bodies. The most
  visible: embedding-dimension mismatch now surfaces as a 409 instead
  of a corrupted vector write; the request-id is now reflected on every
  response; `populate_by_name=True` unblocks ergonomic test
  construction of `Settings`.

### Stats

- 389 tests passing (was ~70 at v0.1.0).
- `mypy --strict` clean across `sorakai/`, `tests/`, `scripts/`, `ui/`.
- `ruff check + format` clean; no inside-function imports, no `noqa`,
  no `type: ignore` introduced by the overhaul.
- Three FastAPI services + Qdrant + Redis + Ollama + MLflow + optional
  Jaeger + optional Streamlit, all wired together by one
  `docker-compose.yml`.

## [0.1.0] - 2025

Initial prototype. OpenAI-only, single FastAPI app, in-memory
retrieval, no eval harness, no agent. Kept for reference; not
maintained.

[0.2.0]: https://github.com/ZeeFcd/sorakAi/releases/tag/v0.2.0
[0.1.0]: https://github.com/ZeeFcd/sorakAi/releases/tag/v0.1.0
