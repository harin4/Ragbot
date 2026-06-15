"""
rag_llamaindex.py
─────────────────
Knowledge-base management and RAG chat using LlamaIndex.

Key LlamaIndex concepts demonstrated here
------------------------------------------
1. SimpleDirectoryReader / Document  – LlamaIndex's data loading primitives
2. SentenceSplitter                  – LlamaIndex's node parser
3. OpenAIEmbedding                   – LlamaIndex embedding model wrapper
4. QdrantVectorStore + VectorStoreIndex – LlamaIndex index over Qdrant
5. RetrieverQueryEngine              – LlamaIndex's RAG query engine
6. Groq LLM wrapper                  – custom LLM integration

Comparing LlamaIndex vs LangChain
──────────────────────────────────
| Aspect             | LangChain                        | LlamaIndex                        |
|--------------------|----------------------------------|-----------------------------------|
| Primary focus      | Chains / agents / tool use       | Data indexing / retrieval (RAG)   |
| Abstraction level  | Lower – more composable pieces   | Higher – opinionated pipeline     |
| Text splitting     | TextSplitter classes             | NodeParser (e.g. SentenceSplitter)|
| Vector store API   | VectorStore.from_documents()     | VectorStoreIndex.from_documents() |
| Query interface    | Chain.run() / invoke()           | QueryEngine.query()               |
| Streaming          | LLM.stream() / callbacks         | QueryEngine with stream_chat()    |
| Metadata           | Document.metadata dict           | Node.metadata dict                |
"""
from __future__ import annotations

from typing import Generator

import cohere
from llama_index.core import (
    Document,
    Settings as LISettings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq as LlamaGroq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import get_settings

cfg = get_settings()

# ── Shared singletons (created once, reused across requests) ─────────────────
_qdrant_client: QdrantClient | None = None
_cohere_client: cohere.Client | None = None
_hf_embed_model: HuggingFaceEmbedding | None = None   # cached — avoids 40s reload per request


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    return _qdrant_client


def _get_cohere() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.Client(cfg.cohere_api_key)
    return _cohere_client


def _get_hf_embed() -> HuggingFaceEmbedding:
    global _hf_embed_model
    if _hf_embed_model is None:
        _hf_embed_model = HuggingFaceEmbedding(
            model_name=cfg.embedding_model,
            embed_batch_size=32,
        )
    return _hf_embed_model


def _configure_llamaindex() -> None:
    """Set global LlamaIndex settings — uses cached embedding model."""
    LISettings.embed_model = _get_hf_embed()
    LISettings.llm = LlamaGroq(
        model=cfg.llm_model,
        api_key=cfg.groq_api_key,
    )
    LISettings.chunk_size = cfg.chunk_size * 4
    LISettings.chunk_overlap = cfg.chunk_overlap * 4


# ── Collection bootstrap ────────────────────────────────────────────────────

def _ensure_collection(collection: str, vector_size: int = 384) -> None:
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
    Convert *texts* to LlamaIndex Documents, parse into nodes with
    SentenceSplitter, embed, and store in Qdrant.

    Returns the number of nodes stored.
    """
    _configure_llamaindex()
    _ensure_collection(cfg.qdrant_collection_llamaindex)

    # LlamaIndex calls its document units "Documents" too, but they're slightly
    # different from LangChain Documents – they carry an id_ and extra_info dict.
    documents = [
        Document(
            text=t,
            metadata={"source": source_url, "title": source_title},  # type: ignore[arg-type]
        )
        for t in texts
    ]

    # SentenceSplitter respects sentence boundaries; compare to LangChain's
    # RecursiveCharacterTextSplitter which falls back through separators.
    parser = SentenceSplitter(
        chunk_size=cfg.chunk_size * 4,
        chunk_overlap=cfg.chunk_overlap * 4,
    )
    nodes = parser.get_nodes_from_documents(documents)

    # QdrantVectorStore (LlamaIndex flavour) + StorageContext wires everything together
    vector_store = QdrantVectorStore(
        client=_get_qdrant(),
        collection_name=cfg.qdrant_collection_llamaindex,
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # VectorStoreIndex.from_documents embeds nodes and stores them
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_ctx,
        show_progress=False,
    )
    return len(nodes)


# ── Retrieval + Rerank ──────────────────────────────────────────────────────

def _retrieve_and_rerank(query: str) -> list[dict]:
    """Retrieve candidates via LlamaIndex, rerank with Cohere."""
    _configure_llamaindex()

    vector_store = QdrantVectorStore(
        client=_get_qdrant(),
        collection_name=cfg.qdrant_collection_llamaindex,
    )
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_ctx,
    )

    # LlamaIndex retriever – similarity_top_k is equivalent to LangChain's k
    retriever = index.as_retriever(similarity_top_k=cfg.top_k_retrieve)
    nodes = retriever.retrieve(query)

    if not nodes:
        return []

    # Cohere rerank (same as LangChain path – framework-agnostic step)
    co = _get_cohere()
    response = co.rerank(
        model=cfg.rerank_model,
        query=query,
        documents=[n.node.get_content() for n in nodes],
        top_n=cfg.top_k_rerank,
    )
    reranked = [
        {
            "content": nodes[r.index].node.get_content(),
            "source": nodes[r.index].node.metadata.get("source", ""),
            "title": nodes[r.index].node.metadata.get("title", ""),
        }
        for r in response.results
    ]
    return reranked


# ── Question rewriting ──────────────────────────────────────────────────────

def _rewrite_query(question: str, history: list[dict]) -> str:
    from groq import Groq
    client = Groq(api_key=cfg.groq_api_key)
    history_text = ""
    if history:
        recent = history[-6:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
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

_SYSTEM_PROMPT = """You are a helpful assistant. Use the context below to answer the question.
If the context contains relevant information, prioritize it and cite it.
If the context does not cover the question, answer using your general knowledge and mention that the answer is not from the knowledge base.

Context:
{context}"""


def chat_stream(question: str, history: list[dict] | None = None) -> tuple[Generator[str, None, None], list[dict]]:
    """
    1. Rewrite query
    2. Retrieve + rerank with LlamaIndex/Qdrant + Cohere
    3. Stream answer from Groq

    Returns (token_generator, sources_list).
    """
    rewritten = _rewrite_query(question, history or [])
    sources = _retrieve_and_rerank(rewritten)

    context_text = "\n\n---\n\n".join(s["content"] for s in sources)
    full_prompt = _SYSTEM_PROMPT.format(context=context_text)

    from groq import Groq
    client = Groq(api_key=cfg.groq_api_key)

    def _generator() -> Generator[str, None, None]:
        stream = client.chat.completions.create(
            model=cfg.llm_model,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=1024,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _generator(), sources