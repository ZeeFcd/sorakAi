"""Streamlit chat UI for sorakAi (Wave 10).

The UI is split into two layers:

- pure Python helpers (request bodies, response parsers, header builders)
  in :mod:`ui.client`, unit-tested in ``tests/test_ui_client.py``;
- a Streamlit entrypoint in :mod:`ui.streamlit_app`, invoked by
  ``streamlit run ui/streamlit_app.py``.

Splitting the two means our test suite doesn't need ``streamlit``
installed (it's an optional extra under ``requirements-ui.txt``) and a
broken layout never breaks the CI gate.
"""
