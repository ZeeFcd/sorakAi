"""Streamlit chat UI for sorakAi (Wave 10).

The implementation in :mod:`ui.streamlit_app` is split into two layers:

- pure Python helpers (request bodies, response parsers, header builders)
  that are unit-tested in ``tests/test_ui_client.py``;
- a ``main()`` entrypoint that calls into the ``streamlit`` runtime to
  paint widgets; this is what ``streamlit run ui/streamlit_app.py``
  invokes.

Splitting the two means our test suite doesn't need ``streamlit``
installed (it's an optional extra under ``requirements-ui.txt``) and a
broken layout never breaks the CI gate.
"""
