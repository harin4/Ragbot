from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    cohere_api_key: str

    # OpenAI key is optional — only needed if switching back to OpenAI embeddings
    openai_api_key: str = ""

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieve: int = 6   # fewer candidates = faster Qdrant + Cohere
    top_k_rerank: int = 3

    # Embeddings: BAAI/BGE runs fully locally — no API key, no cost, no rate limits.
    # bge-small-en-v1.5 = 384-dim  (~130 MB download, fastest)
    # bge-base-en-v1.5  = 768-dim  (~440 MB download, better quality)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384           # must match the model above
    llm_model: str = "llama-3.3-70b-versatile"
    rerank_model: str = "rerank-v3.5"

    qdrant_collection_llamaindex: str = "rag_llamaindex"
    qdrant_collection_langchain: str = "rag_langchain"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore