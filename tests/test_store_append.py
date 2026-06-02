from fastapi.testclient import TestClient

from sorakai.rag.app import create_app as create_rag


def test_two_documents_in_kb_top_k_merge(run_async, seed_kb):
    rag = create_rag()
    with TestClient(rag) as rc:
        run_async(rag.state.store.clear_all())
        seed_kb(rag, ["The secret code is ALPHA."], doc_id="d1", filename="a.txt")
        seed_kb(rag, ["The backup code is BRAVO."], doc_id="d2", filename="b.txt")
        r = rc.post("/v1/query", json={"question": "What is the backup code?", "top_k": 3})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["sources_used"] >= 1
        assert body["sources_used"] <= 2


def test_session_memory_persists_across_turns(run_async, seed_kb):
    """Two POSTs with the same ``session_id`` produce a 2-turn history in the chat store.

    The assertion is provider-agnostic - it inspects the chat store directly
    instead of pattern-matching the stub answer text - so it stays valid the
    day we swap the stub for a real model.
    """
    rag = create_rag()
    with TestClient(rag) as rc:
        seed_kb(rag, ["Paris is the capital of France."], doc_id="d", filename="f.txt")
        r1 = rc.post(
            "/v1/query",
            json={"question": "What is the capital?", "session_id": "u1", "use_chat_history": True},
        )
        assert r1.status_code == 200, r1.text
        r2 = rc.post(
            "/v1/query",
            json={"question": "Repeat that in one word.", "session_id": "u1", "use_chat_history": True},
        )
        assert r2.status_code == 200, r2.text
        assert r2.json()["session_id"] == "u1"

        history = run_async(rag.state.chat_store.get_messages("u1"))
        roles = [m["role"] for m in history]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert history[0]["content"] == "What is the capital?"
        assert history[2]["content"] == "Repeat that in one word."
