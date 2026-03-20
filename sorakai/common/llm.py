import os

from sorakai.common.logging_utils import get_logger

logger = get_logger("sorakai.llm")


def ask_llm(question: str, context: str) -> str:
    """Calls OpenAI Chat Completions when OPENAI_API_KEY is set; otherwise returns a deterministic stub."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        logger.warning("No OPENAI_API_KEY; using stub answer")
        snippet = (context[:120] + "…") if len(context) > 120 else context
        return f"[stub] Based on context ({snippet!r}), Q: {question!r}"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer using only the provided context. Be concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            max_tokens=256,
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001 — surface to client in MVP
        logger.exception("LLM call failed")
        return f"[error] LLM failed: {e}"
