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
) -> int:
    """
    Split *texts* further with LangChain's RecursiveCharacterTextSplitter,
    embed them, and upsert into Qdrant.

    Returns the number of chunks stored.
    """
    _ensure_collection(cfg.qdrant_collection_langchain)

    # LangChain-style: wrap raw strings in Document objects so we can attach metadata
    raw_docs = [
        Document(page_content=t, metadata={"source": source_url, "title": source_title})
        for t in texts
    ]

    # RecursiveCharacterTextSplitter is smarter than naive word-splitting:
    # it tries to split on paragraphs → sentences → words in that order.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size * 4,  # characters ≈ words × 4
        chunk_overlap=cfg.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    # Use pre-created collection so vector dimensions are always correct (1024-dim Cohere)
    vectorstore = QdrantVectorStore(
        client=_get_qdrant(),
        collection_name=cfg.qdrant_collection_langchain,
        embedding=_get_embeddings(),
    )
    vectorstore.add_documents(docs)
    return len(docs)


# ── Retrieval + Rerank ──────────────────────────────────────────────────────

def _retrieve_and_rerank(query: str) -> list[Document]:
    """Vector search → Cohere rerank → top-k docs."""
    vectorstore = QdrantVectorStore(
        client=_get_qdrant(),
        collection_name=cfg.qdrant_collection_langchain,
        embedding=_get_embeddings(),
    )
    # Retrieve wider candidate set first
    candidates: list[Document] = vectorstore.similarity_search(query, k=cfg.top_k_retrieve)

    if not candidates:
        return []

    # Cohere rerank narrows candidates to the most relevant chunks
    co = _get_cohere()
    response = co.rerank(
        model=cfg.rerank_model,
        query=query,
        documents=[d.page_content for d in candidates],
        top_n=cfg.top_k_rerank,
    )
    reranked = [candidates[r.index] for r in response.results]
    return reranked


# ── Question rewriting ──────────────────────────────────────────────────────

def _rewrite_query(question: str) -> str:
    """
    Use Groq (via LangChain's ChatOpenAI-compatible interface) to rephrase
    the user question into a better search query.
    """
    from groq import Groq
    client = Groq(api_key=cfg.groq_api_key)
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. "
                    "Rewrite the user's question into a concise, keyword-rich search query. "
                    "Return ONLY the rewritten query, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ── Chat (streaming) ────────────────────────────────────────────────────────

_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:""",
)


def chat_stream(question: str) -> tuple[AsyncIterator[str], list[dict]]:
    """
    Synchronous wrapper that:
    1. Rewrites the question for better retrieval
    2. Retrieves + reranks context chunks
    3. Returns (token_generator, source_chunks_metadata)

    We use Groq directly for streaming (LangChain streaming support for Groq
    requires extra setup; this keeps it simple and easy to understand).
    """
    rewritten = _rewrite_query(question)
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
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _generator(), sources
