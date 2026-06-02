import numpy as np
from fastapi.testclient import TestClient

from sorakai.rag.app import create_app as create_rag


def test_two_documents_in_kb_top_k_merge(run_async):
    rag = create_rag()
    with TestClient(rag) as rc:
        store = rag.state.store
        run_async(store.clear_all())
        run_async(
            store.append_document(
                "d1",
                "a.txt",
                ["The secret code is ALPHA."],
                [np.array([1.0, 0.0, 0.0])],
            )
        )
        run_async(
            store.append_document(
                "d2",
                "b.txt",
                ["The backup code is BRAVO."],
                [np.array([0.0, 1.0, 0.0])],
            )
        )
        r = rc.post("/v1/query", json={"question": "What is the backup code?", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["sources_used"] >= 1
        assert body["sources_used"] <= 2


def test_session_memory_persists_across_turns(run_async):
    """Two POSTs with the same ``session_id`` produce a 2-turn history in the chat store.

    The assertion is provider-agnostic - it inspects the chat store directly
    instead of pattern-matching the stub answer text - so it stays valid the
    day we swap the stub for a real model.
    """
    rag = create_rag()
    with TestClient(rag) as rc:
        run_async(
            rag.state.store.append_document(
                "d",
                "f.txt",
                ["Paris is the capital of France."],
                [np.array([1.0, 2.0])],
            )
        )
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
