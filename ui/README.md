# sorakAi chat UI (Wave 10)

A minimal Streamlit chat front-end that talks to the sorakAi **gateway**
(default `http://127.0.0.1:8000`). It's a zero-JS demo of the end-to-end
flow: type a question, the gateway forwards to the RAG service, the
answer streams back.

## Install + run

```bash
# The runtime services don't depend on Streamlit; install the UI extra
# separately so the gateway / ingest / RAG containers stay slim.
pip install -r requirements-ui.txt

# Point UI_GATEWAY_URL at a non-default gateway if needed.
PYTHONPATH=. UI_GATEWAY_URL=http://127.0.0.1:8000 streamlit run ui/streamlit_app.py
```

Or use the Makefile shortcuts:

```bash
make install-ui      # pip install -r requirements-ui.txt
make ui              # PYTHONPATH=. streamlit run ui/streamlit_app.py
```

## Features

- **Chain vs Agent toggle**: pick the LCEL chain (`/v1/query`) or the
  LangGraph agent (`/v1/agent`) from the sidebar.
- **Session-aware history**: a fixed `session_id` (configurable in the
  sidebar) so multi-turn context persists across the visible chat
  scroll.
- **Bearer auth**: paste the gateway's `GATEWAY_API_KEY` into the
  sidebar to talk to a hardened deployment.
- **Agent meta footer**: when in agent mode the footer shows `route`,
  `steps_used`, `sources_used`, and the node trace so you can see the
  graph's decision path inline.

## Tests

The pure helpers (`build_query_payload`, `parse_chain_response`,
`ask_gateway`, etc.) live in `ui/client.py` and are unit-tested under
`tests/test_ui_client.py` **without** Streamlit installed.
`ui/streamlit_app.py` imports Streamlit at module scope and is only
invoked by `streamlit run`; if you change UI layout, run it locally and
sanity-check by eye.

## Configuration

| Setting               | Source                | Default                  | Purpose                              |
| --------------------- | --------------------- | ------------------------ | ------------------------------------ |
| Gateway URL           | sidebar / env         | `UI_GATEWAY_URL`         | Where to send chat requests.         |
| API key               | sidebar               | empty                    | Bearer auth (matches `GATEWAY_API_KEY`). |
| Mode                  | sidebar               | `chain`                  | `chain` or `agent`.                  |
| Session ID            | sidebar               | `ui-default`             | Multi-turn history key.              |
| Max agent steps       | sidebar (agent only)  | `4`                      | Caps the LangGraph loop.             |
