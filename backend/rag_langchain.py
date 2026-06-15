"""
rag_langchain.py
────────────────
Knowledge-base management and RAG chat using LangChain.

Key LangChain concepts demonstrated here
-----------------------------------------
1. RecursiveCharacterTextSplitter  – smarter chunking than naive word split
2. OpenAIEmbeddings                – wraps the OpenAI embedding endpoint
3. QdrantVectorStore               – LangChain's Qdrant integration
4. ChatGroq                        – LangChain wrapper around Groq
5. RetrievalQA / custom chain      – standard RAG chain composition
"""
from __future__ import annotations

from typing import AsyncIterator

import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import get_settings
from collection_utils import framework_prefix

cfg = get_settings()

# ── Shared clients (initialised once) ──────────────────────────────────────
_qdrant_client: QdrantClient | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_cohere_client: cohere.Client | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
        )
    return _qdrant_client


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        # HuggingFaceEmbeddings — runs fully locally, zero API cost, no rate limits.
        # normalize_embeddings=True is required for cosine similarity with BGE models.
        _embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(cfg.cohere_api_key)
    return _cohere_client


# ── Collection bootstrap ────────────────────────────────────────────────────

def _ensure_collection(collection: str, vector_size: int = 384) -> None:
    """Create Qdrant collection if it doesn't exist yet."""
    client = _get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ── Ingest ──────────────────────────────────────────────────────────────────

def ingest_documents(
    texts: list[str],
    source_url: str,
    source_title: str,
    collection_name: str,
) -> int:
    """
    Split *texts* with LangChain's RecursiveCharacterTextSplitter, embed them,
    and upsert into a per-URL Qdrant collection.

    Returns the number of chunks stored.
    """
    _ensure_collection(collection_name)

    raw_docs = [
        Document(page_content=t, metadata={"source": source_url, "title": source_title})
        for t in texts
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size * 4,
        chunk_overlap=cfg.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    vectorstore = QdrantVectorStore(
        client=_get_qdrant(),
        collection_name=collection_name,
        embedding=_get_embeddings(),
    )
    vectorstore.add_documents(docs)
    return len(docs)


# ── Retrieval + Rerank ──────────────────────────────────────────────────────

def _retrieve_and_rerank(query: str) -> list[Document]:
    """Vector search across all per-URL collections → Cohere rerank → top-k docs."""
    client = _get_qdrant()
    prefix = framework_prefix("langchain")
    collections = [c.name for c in client.get_collections().collections if c.name.startswith(prefix)]

    if not collections:
        return []

    all_candidates: list[Document] = []
    for cname in collections:
        try:
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=cname,
                embedding=_get_embeddings(),
            )
            all_candidates.extend(vectorstore.similarity_search(query, k=cfg.top_k_retrieve))
        except Exception:
            continue

    if not all_candidates:
        return []

    co = _get_cohere()
    response = co.rerank(
        model=cfg.rerank_model,
        query=query,
        documents=[d.page_content for d in all_candidates],
        top_n=cfg.top_k_rerank,
    )
    return [all_candidates[r.index] for r in response.results]


# ── Question rewriting ──────────────────────────────────────────────────────

def _rewrite_query(question: str, history: list[dict]) -> str:
    from groq import Groq
    client = Groq(api_key=cfg.groq_api_key)
    history_text = ""
    if history:
        # Skip failed "not covered" assistant turns — they mislead the rewriter
        useful = [
            m for m in history[-6:]
            if not (m["role"] == "assistant" and "isn't covered" in m.get("content", ""))
        ]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in useful)
    system = (
        "You are a search query optimizer. "
        "Given the conversation history and the latest question, rewrite the question "
        "into a concise, keyword-rich search query that resolves any vague references "
        "(like 'the url', 'it', 'that page') using the conversation context. "
        "Return ONLY the rewritten query, nothing else."
    )
    messages = [{"role": "system", "content": system}]
    if history_text:
        messages.append({"role": "user", "content": f"Conversation so far:\n{history_text}\n\nLatest question: {question}"})
    else:
        messages.append({"role": "user", "content": question})
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=messages,
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ── Chat (streaming) ────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = (
    "You are a strict knowledge-base assistant. "
    "You MUST answer using ONLY the context provided. "
    "FORBIDDEN: using any outside knowledge, training data, or general information. "
    "If the context does not contain enough information to answer, reply ONLY with: "
    "'This topic isn't covered in the ingested content. Try ingesting a relevant URL first.' "
    "Do not add caveats, extra explanations, or general knowledge under any circumstances."
)

_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="Context:\n{context}\n\nQuestion: {question}",
)


def chat_stream(question: str, history: list[dict] | None = None) -> tuple[AsyncIterator[str], list[dict]]:
    """
    Synchronous wrapper that:
    1. Rewrites the question for better retrieval (using history to resolve vague refs)
    2. Retrieves + reranks context chunks
    3. Returns (token_generator, source_chunks_metadata)
    """
    rewritten = _rewrite_query(question, history or [])
    docs = _retrieve_and_rerank(rewritten)

    sources = [
        {"content": d.page_content, "source": d.metadata.get("source", ""), "title": d.metadata.get("title", "")}
        for d in docs
    ]

    context_text = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = _RAG_PROMPT.format(context=context_text, question=question)

    from groq import Groq
    client = Groq(api_key=cfg.groq_api_key)

    def _generator():
        stream = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _generator(), sources
