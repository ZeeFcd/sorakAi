#!/usr/bin/env python3
"""Wave 11: seed a small corpus into the gateway and (optionally) ask it a question.

Designed for two callers:

- ``scripts/dev_up.sh`` invokes this with the default gateway URL to verify
  the freshly-booted stack works end-to-end (ingest -> Qdrant -> RAG chain).
- A developer runs it manually after editing a prompt or chain wiring to
  smoke-check the full path without firing up the Streamlit UI.

The corpus shipped with the script is intentionally tiny (a handful of
synthetic facts about sorakAi) so a smoke run completes in seconds against
a local Ollama; for a real evaluation use ``scripts/eval.py`` with the
Wave 9 golden set.

Exit codes:
    0 - all documents ingested (and, when ``--no-query`` is unset, the
        sample query returned a non-empty answer).
    1 - the gateway returned an error (missing auth, embedding mismatch,
        upstream 5xx, etc.). The HTTP body is printed verbatim so a
        compose-level failure surfaces in the dev_up.sh log.
    2 - invalid CLI arguments.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import httpx

EXIT_OK = 0
EXIT_HTTP = 1
EXIT_USAGE = 2

DEFAULT_GATEWAY_URL = "http://127.0.0.1:8000"
DEFAULT_SAMPLE_QUESTION = "What is sorakAi and which providers does it support?"
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclasses.dataclass(frozen=True, slots=True)
class SeedDocument:
    """One row of the seed corpus.

    ``mime_type`` defaults to ``text/markdown`` because the bundled
    sample docs are markdown; callers that ingest plain prose can leave
    it on the default - the chunker treats markdown as text.
    """

    filename: str
    content: str
    mime_type: str = "text/markdown"


SAMPLE_CORPUS: tuple[SeedDocument, ...] = (
    SeedDocument(
        filename="sorakai-overview.md",
        content=(
            "# sorakAi overview\n\n"
            "sorakAi is a local-first RAG and agent platform that ships three FastAPI\n"
            "services: ingest, RAG, and a gateway BFF. It is intentionally provider-agnostic\n"
            "and never calls out to a cloud LLM provider in the default install.\n"
        ),
    ),
    SeedDocument(
        filename="sorakai-providers.md",
        content=(
            "# Registered providers (Wave 11)\n\n"
            "LLM: ollama (default), stub (deterministic test double).\n"
            "Embeddings: ollama (default), char (no-network deterministic embeddings).\n"
            "Vector stores: qdrant (default in compose), redis, memory.\n"
            "Chat history: redis (default), in-memory.\n"
        ),
    ),
    SeedDocument(
        filename="sorakai-eval.md",
        content=(
            "# Evaluation harness\n\n"
            "Wave 9 added a golden Q/A set under tests/eval/ and a runner that\n"
            "scores `answer_contains_expected` and `context_precision_at_k`. The\n"
            "CLI lives at scripts/eval.py and can target either the LCEL chain\n"
            "or the LangGraph agent. When the optional ragas extra is installed\n"
            "the runner pipes the same cases through ragas as well.\n"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_seed_script.py)
# ---------------------------------------------------------------------------


def build_ingest_payload(doc: SeedDocument) -> dict[str, str]:
    """Render a :class:`SeedDocument` as the gateway's JSON ingest body."""
    return {
        "filename": doc.filename,
        "content": doc.content,
        "mime_type": doc.mime_type,
    }


def build_query_payload(question: str) -> dict[str, object]:
    """Render the sample-query body. Stateless (no chat history)."""
    return {"question": question, "use_chat_history": False}


def build_headers(api_key: str | None) -> dict[str, str]:
    """Bearer header builder; empty dict when no key is supplied."""
    key = (api_key or "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def load_corpus(path: Path | None) -> tuple[SeedDocument, ...]:
    """Load the seed corpus from JSONL or fall back to :data:`SAMPLE_CORPUS`.

    Each non-empty, non-comment JSONL line must decode to an object with
    ``filename``/``content`` keys; ``mime_type`` is optional.
    """
    if path is None:
        return SAMPLE_CORPUS
    raw = path.read_text(encoding="utf-8")
    docs: list[SeedDocument] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{lineno}: expected an object, got {type(record).__name__}")
        if "filename" not in record or "content" not in record:
            raise ValueError(f"{path}:{lineno}: missing required 'filename' or 'content'")
        docs.append(
            SeedDocument(
                filename=str(record["filename"]),
                content=str(record["content"]),
                mime_type=str(record.get("mime_type", "text/markdown")),
            )
        )
    if not docs:
        raise ValueError(f"{path} contained no documents")
    return tuple(docs)


# ---------------------------------------------------------------------------
# HTTP plumbing (exercised end-to-end against an httpx.MockTransport)
# ---------------------------------------------------------------------------


def ingest_documents(
    docs: Iterable[SeedDocument],
    *,
    gateway_url: str,
    api_key: str | None,
    client: httpx.Client,
) -> list[dict[str, object]]:
    """POST each document to ``/v1/documents`` and return the parsed responses.

    Raises :class:`httpx.HTTPStatusError` on a non-2xx so the CLI can
    print a useful error and exit non-zero. The caller owns the
    :class:`httpx.Client` so tests can inject a ``MockTransport``.
    """
    base = gateway_url.rstrip("/")
    headers = build_headers(api_key)
    results: list[dict[str, object]] = []
    for doc in docs:
        response = client.post(
            f"{base}/v1/documents",
            json=build_ingest_payload(doc),
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):  # pragma: no cover - guard for hostile servers
            raise httpx.HTTPError(f"expected JSON object, got {type(body).__name__}")
        results.append(body)
    return results


def sample_query(
    question: str,
    *,
    gateway_url: str,
    api_key: str | None,
    client: httpx.Client,
) -> dict[str, object]:
    """POST the sample question to ``/v1/query`` and return the parsed body."""
    base = gateway_url.rstrip("/")
    response = client.post(
        f"{base}/v1/query",
        json=build_query_payload(question),
        headers=build_headers(api_key),
    )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):  # pragma: no cover
        raise httpx.HTTPError(f"expected JSON object, got {type(body).__name__}")
    return body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/seed.py",
        description="Seed a small corpus into the sorakAi gateway and smoke a sample query.",
    )
    parser.add_argument(
        "--gateway-url",
        default=DEFAULT_GATEWAY_URL,
        help=f"Gateway base URL (default: {DEFAULT_GATEWAY_URL}).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token to send when the gateway has GATEWAY_API_KEY set.",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_SAMPLE_QUESTION,
        help="Sample question to fire after seeding (skipped with --no-query).",
    )
    parser.add_argument(
        "--no-query",
        action="store_true",
        help="Skip the sample query (just seed the corpus).",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Optional JSONL path of {filename, content, mime_type?} rows; falls back "
            "to the bundled SAMPLE_CORPUS when omitted."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    return parser


def run(args: argparse.Namespace, *, client: httpx.Client | None = None) -> int:
    """Drive the seed + sample flow. ``client`` is injected by the tests."""
    try:
        docs = load_corpus(args.corpus)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    owns_client = client is None
    if client is None:
        client = httpx.Client(timeout=args.timeout)
    try:
        try:
            ingested = ingest_documents(
                docs,
                gateway_url=args.gateway_url,
                api_key=args.api_key,
                client=client,
            )
        except httpx.HTTPStatusError as exc:
            print(
                f"error: ingest failed with HTTP {exc.response.status_code}: {exc.response.text}",
                file=sys.stderr,
            )
            return EXIT_HTTP
        except httpx.RequestError as exc:
            print(f"error: gateway unreachable: {exc}", file=sys.stderr)
            return EXIT_HTTP

        for doc, payload in zip(docs, ingested, strict=True):
            chunks = payload.get("num_chunks", "?")
            print(f"ingested {doc.filename!r}: {chunks} chunks")

        if args.no_query:
            return EXIT_OK

        try:
            answer = sample_query(
                args.question,
                gateway_url=args.gateway_url,
                api_key=args.api_key,
                client=client,
            )
        except httpx.HTTPStatusError as exc:
            print(
                f"error: sample query failed with HTTP {exc.response.status_code}: {exc.response.text}",
                file=sys.stderr,
            )
            return EXIT_HTTP
        except httpx.RequestError as exc:
            print(f"error: gateway unreachable: {exc}", file=sys.stderr)
            return EXIT_HTTP

        print()
        print(f"Q: {args.question}")
        print(f"A: {answer.get('answer', '')}")
        sources = answer.get("sources_used", 0)
        print(f"sources_used: {sources}")
        return EXIT_OK
    finally:
        if owns_client:
            client.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
