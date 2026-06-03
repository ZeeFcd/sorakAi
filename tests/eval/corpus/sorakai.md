# sorakAi project conventions

sorakAi is a local-only Retrieval-Augmented Generation playground built on
FastAPI, LangChain, LangGraph, and Ollama. It is structured as three
microservices: an ingest service that chunks and embeds documents, a RAG
service that runs the LCEL chain and the LangGraph agent, and a thin
gateway that proxies the public API.

All providers (LLM, embeddings, vector store, chat history) are accessed
through factory functions in `sorakai.infra.*`, so swapping Ollama for a
different LLM is a one-environment-variable change. The default vector
store is Qdrant; the default LLM is `llama3.2:1b` and the default
embeddings model is `nomic-embed-text`.

The hybrid retriever combines BM25 lexical scores with vector similarity
using Reciprocal Rank Fusion. Streaming uses Server-Sent Events on the
`/v1/query/stream` and `/v1/agent/stream` endpoints. Observability is
provided by OpenTelemetry tracing, structured `structlog` logs, and a
LangChain MLflow callback that records one MLflow run per chain or agent
invocation.
