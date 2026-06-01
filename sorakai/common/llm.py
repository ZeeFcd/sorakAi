from sorakai.common.config import get_settings
from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.llm")


def ask_llm(
    question: str,
    context: str,
    *,
    conversation: list[dict[str, str]] | None = None,
) -> str:
    """
    LLM routing (RAG service only):

    1. ``OPENAI_BASE_URL`` set → self-hosted OpenAI-compatible API (e.g. Ollama at ``.../v1``).
    2. Else ``OPENAI_API_KEY`` set → OpenAI cloud.
    3. Else → deterministic stub (no network).

    ``conversation`` is prior user/assistant turns (OpenAI message shape); the current
    question is sent in the final user message together with the retrieved KB context.
    """
    settings = get_settings()
    base_url = settings.openai_base_url
    api_key = settings.openai_api_key
    model = settings.openai_chat_model

    if not base_url and not api_key:
        logger.warning("No OPENAI_BASE_URL or OPENAI_API_KEY; using stub answer")
        snippet = (context[:120] + "…") if len(context) > 120 else context
        hist_note = ""
        if conversation:
            hist_note = f" [+{len(conversation)} prior msgs]"
        return f"[stub] Based on context ({snippet!r}), Q: {question!r}{hist_note}"

    try:
        from openai import OpenAI

        if base_url:
            key = api_key or "ollama"
            client = OpenAI(api_key=key, base_url=base_url.rstrip("/"))
            logger.info("LLM via self-hosted endpoint model=%s", model)
        else:
            assert api_key is not None
            client = OpenAI(api_key=api_key)
            logger.info("LLM via OpenAI cloud model=%s", model)

        system = (
            "You answer using the knowledge base context provided in the user's message. "
            "Earlier messages in this chat are the same conversation—stay consistent with them. "
            "If the answer is not in the context, say you do not have that information."
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if conversation:
            for m in conversation:
                role = m.get("role", "")
                content = m.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append(
            {
                "role": "user",
                "content": f"Knowledge base context:\n{context}\n\nQuestion:\n{question}",
            }
        )

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("LLM call failed")
        return f"[error] LLM failed: {e}"
