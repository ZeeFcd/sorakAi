"""Wave 11 tests for scripts/seed.py.

These pin the pure helpers (payload/headers/corpus loading) and the
end-to-end CLI flow using ``httpx.MockTransport`` so the suite never
needs a running gateway.

The script is imported via ``importlib`` (rather than ``from
scripts.seed import ...``) so mypy doesn't see the same source file
under two module names; the same dance is used by
``tests/test_eval_cli.py`` for ``scripts/eval.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "seed.py"
_MODULE_NAME = "sorakai_seed_script"


def _load_seed_module() -> ModuleType:
    """Load ``scripts/seed.py`` under a synthetic module name.

    ``sys.modules`` is updated *before* ``exec_module`` so dataclass
    ``slots=True`` resolution (which looks the module up in
    ``sys.modules`` during processing) succeeds.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


seed = _load_seed_module()

DEFAULT_GATEWAY_URL: str = seed.DEFAULT_GATEWAY_URL
DEFAULT_SAMPLE_QUESTION: str = seed.DEFAULT_SAMPLE_QUESTION
EXIT_HTTP: int = seed.EXIT_HTTP
EXIT_OK: int = seed.EXIT_OK
EXIT_USAGE: int = seed.EXIT_USAGE
SAMPLE_CORPUS = seed.SAMPLE_CORPUS
SeedDocument = seed.SeedDocument
build_headers = seed.build_headers
build_ingest_payload = seed.build_ingest_payload
build_query_payload = seed.build_query_payload
ingest_documents = seed.ingest_documents
load_corpus = seed.load_corpus
run = seed.run
sample_query = seed.sample_query

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_ingest_payload_round_trips_doc_fields() -> None:
    doc = SeedDocument(filename="a.md", content="# x", mime_type="text/markdown")
    assert build_ingest_payload(doc) == {
        "filename": "a.md",
        "content": "# x",
        "mime_type": "text/markdown",
    }


def test_build_query_payload_is_stateless() -> None:
    body = build_query_payload("hello?")
    assert body == {"question": "hello?", "use_chat_history": False}


def test_build_headers_returns_empty_for_no_key() -> None:
    assert build_headers(None) == {}
    assert build_headers("") == {}
    assert build_headers("   ") == {}


def test_build_headers_emits_bearer_when_key_present() -> None:
    assert build_headers("hunter2") == {"Authorization": "Bearer hunter2"}


def test_sample_corpus_is_non_empty() -> None:
    assert len(SAMPLE_CORPUS) >= 3
    assert all(doc.filename.endswith(".md") for doc in SAMPLE_CORPUS)


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------


def test_load_corpus_returns_sample_when_path_is_none() -> None:
    assert load_corpus(None) is SAMPLE_CORPUS


def test_load_corpus_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "seed.jsonl"
    path.write_text(
        "# comments are skipped\n"
        "\n"
        '{"filename": "a.md", "content": "alpha"}\n'
        '{"filename": "b.txt", "content": "beta", "mime_type": "text/plain"}\n',
        encoding="utf-8",
    )
    docs = load_corpus(path)
    assert [d.filename for d in docs] == ["a.md", "b.txt"]
    assert docs[1].mime_type == "text/plain"


def test_load_corpus_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"content": "missing filename"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        load_corpus(path)


def test_load_corpus_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_corpus(path)


def test_load_corpus_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no documents"):
        load_corpus(path)


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ingest_documents_posts_each_doc_with_bearer() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            {
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
                "body": request.read().decode("utf-8"),
            }
        )
        return httpx.Response(201, json={"num_chunks": 2, "filename": "x", "document_id": "id"})

    docs = (
        SeedDocument(filename="a.md", content="alpha"),
        SeedDocument(filename="b.md", content="beta"),
    )
    with _client(handler) as client:
        results = ingest_documents(
            docs,
            gateway_url="http://gw.test/",
            api_key="hunter2",
            client=client,
        )

    assert len(results) == 2
    assert seen[0]["url"] == "http://gw.test/v1/documents"
    assert seen[0]["auth"] == "Bearer hunter2"
    assert '"filename":"a.md"' in str(seen[0]["body"]).replace(" ", "")


def test_ingest_documents_raises_on_4xx() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Missing bearer credentials"})

    with _client(handler) as client, pytest.raises(httpx.HTTPStatusError):
        ingest_documents(
            (SAMPLE_CORPUS[0],),
            gateway_url="http://gw.test",
            api_key=None,
            client=client,
        )


def test_sample_query_returns_parsed_body() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answer": "ok", "sources_used": 2, "context_preview": "...", "session_id": None},
        )

    with _client(handler) as client:
        body = sample_query(
            "what?",
            gateway_url="http://gw.test",
            api_key=None,
            client=client,
        )
    assert body["answer"] == "ok"
    assert body["sources_used"] == 2


# ---------------------------------------------------------------------------
# run() end-to-end
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "gateway_url": DEFAULT_GATEWAY_URL,
        "api_key": None,
        "question": DEFAULT_SAMPLE_QUESTION,
        "no_query": False,
        "corpus": None,
        "timeout": 5.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_seed_only_short_circuits_before_query(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents"
        return httpx.Response(201, json={"num_chunks": 1, "filename": "x", "document_id": "id"})

    rc = run(_args(no_query=True), client=_client(handler))
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "ingested" in out
    assert "Q:" not in out


def test_run_end_to_end_prints_answer(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/documents":
            return httpx.Response(201, json={"num_chunks": 1, "filename": "x", "document_id": "id"})
        return httpx.Response(
            200,
            json={"answer": "42", "sources_used": 1, "context_preview": "...", "session_id": None},
        )

    rc = run(_args(), client=_client(handler))
    assert rc == EXIT_OK
    out = capsys.readouterr().out
    assert "Q: " in out
    assert "A: 42" in out


def test_run_returns_exit_http_on_ingest_401(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Missing bearer credentials"})

    rc = run(_args(), client=_client(handler))
    assert rc == EXIT_HTTP
    err = capsys.readouterr().err
    assert "HTTP 401" in err


def test_run_returns_exit_usage_on_bad_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    rc = run(_args(corpus=path), client=_client(lambda _: httpx.Response(500)))
    assert rc == EXIT_USAGE
    err = capsys.readouterr().err
    assert "invalid JSON" in err
