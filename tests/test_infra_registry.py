"""Tests for the provider factories - registry lookup, env-driven selection, errors."""

from __future__ import annotations

import pytest

from sorakai.common.config import Settings
from sorakai.core.errors import ConfigError
from sorakai.infra.embeddings import EMBEDDINGS_REGISTRY, get_embeddings, register_embeddings
from sorakai.infra.embeddings.base import Embeddings
from sorakai.infra.embeddings.char import CharPseudoEmbeddings
from sorakai.infra.llm import CHAT_MODEL_REGISTRY, get_chat_model, register_chat_model
from sorakai.infra.llm.base import BaseChatModel
from sorakai.infra.llm.stub import StubChatModel


def test_chat_registry_has_default_providers() -> None:
    assert {"ollama", "stub"}.issubset(CHAT_MODEL_REGISTRY.keys())


def test_embeddings_registry_has_default_providers() -> None:
    assert {"ollama", "char"}.issubset(EMBEDDINGS_REGISTRY.keys())


def test_get_chat_model_returns_stub_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    settings = Settings()
    model = get_chat_model(settings)
    assert isinstance(model, StubChatModel)
    assert isinstance(model, BaseChatModel)


def test_get_embeddings_returns_char_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "char")
    settings = Settings()
    emb = get_embeddings(settings)
    assert isinstance(emb, CharPseudoEmbeddings)
    assert isinstance(emb, Embeddings)


def test_get_chat_model_raises_on_unknown_provider() -> None:
    settings = Settings.model_construct(llm_provider="nope")  # bypass Literal validation
    with pytest.raises(ConfigError, match="Unknown LLM_PROVIDER"):
        get_chat_model(settings)


def test_get_embeddings_raises_on_unknown_provider() -> None:
    settings = Settings.model_construct(embedding_provider="nope")
    with pytest.raises(ConfigError, match="Unknown EMBEDDING_PROVIDER"):
        get_embeddings(settings)


def test_register_chat_model_allows_extension() -> None:
    sentinel = StubChatModel()
    try:
        register_chat_model("custom", lambda _s: sentinel)
        settings = Settings.model_construct(llm_provider="custom")
        assert get_chat_model(settings) is sentinel
    finally:
        CHAT_MODEL_REGISTRY.pop("custom", None)


def test_register_embeddings_allows_extension() -> None:
    sentinel = CharPseudoEmbeddings()
    try:
        register_embeddings("custom", lambda _s: sentinel)
        settings = Settings.model_construct(embedding_provider="custom")
        assert get_embeddings(settings) is sentinel
    finally:
        EMBEDDINGS_REGISTRY.pop("custom", None)
