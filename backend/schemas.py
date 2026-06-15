from pydantic import BaseModel
from typing import Optional


class IngestRequest(BaseModel):
    url: str
    framework: str = "llamaindex"  # "llamaindex" or "langchain"
    # Optional: paste raw text directly instead of scraping (for sites that block bots)
    raw_text: Optional[str] = None
    title: Optional[str] = None


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    url: str
    framework: str
    message: str
    title: str = ""


class ChatRequest(BaseModel):
    question: str
    framework: str = "llamaindex"
    chat_history: list[dict] = []


class SourceChunk(BaseModel):
    text: str
    url: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    sources: list[SourceChunk]
    framework: str


class KnowledgeBaseInfo(BaseModel):
    llamaindex_count: int
    langchain_count: int


class ErrorResponse(BaseModel):
    detail: str