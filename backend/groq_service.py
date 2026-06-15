"""
Groq service: query rewriting + streaming answer generation.
"""

from groq import AsyncGroq
from config import get_settings
from typing import AsyncGenerator

settings = get_settings()
_client: AsyncGroq | None = None


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        _client = AsyncGroq(api_key=settings.groq_api_key)
    return _client


async def rewrite_query(original_question: str) -> str:
    """
    Use Groq to rewrite the user question into a better search query.
    A short, keyword-dense rephrasing improves vector search recall.
    """
    client = get_groq_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. "
                    "Rewrite the user question into a concise, keyword-focused search query "
                    "that will retrieve the most relevant passages from a vector database. "
                    "Return ONLY the rewritten query. No explanation, no punctuation at end."
                ),
            },
            {"role": "user", "content": original_question},
        ],
        temperature=0.1,
        max_tokens=80,
    )
    rewritten = response.choices[0].message.content or original_question
    return rewritten.strip()


async def generate_answer(
    question: str,
    context_chunks: list[str],
    chat_history: list[dict],
) -> str:
    """
    Generate a final answer using retrieved context.
    Non-streaming version used when framework handles its own chain.
    """
    context = "\n\n---\n\n".join(context_chunks)
    client = get_groq_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions based on the provided context. "
                "If the answer isn't in the context, say so clearly. "
                "Always cite which part of the context you used."
            ),
        }
    ]

    # Add conversation history
    for msg in chat_history[-6:]:  # last 3 turns
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


async def stream_answer(
    question: str,
    context_chunks: list[str],
    chat_history: list[dict],
) -> AsyncGenerator[str, None]:
    """
    Stream the final answer token by token.
    Used by the /chat/stream endpoint.
    """
    context = "\n\n---\n\n".join(context_chunks)
    client = get_groq_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer questions based on the provided context. "
                "If the answer isn't in the context, say so clearly."
            ),
        }
    ]

    for msg in chat_history[-6:]:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }
    )

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta