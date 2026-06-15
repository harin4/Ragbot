import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest, chat, kb
from config import get_settings

settings = get_settings()
logger = logging.getLogger("rag_chatbot")

def _warmup_embeddings() -> None:
    from rag_langchain import _get_embeddings
    from rag_llamaindex import _get_hf_embed
    _get_embeddings()
    _get_hf_embed()
    _get_embeddings().embed_query("warmup")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        client.get_collections()
        print("✅ Qdrant connected")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")

    # Warmup embeddings
    logger.info("Pre-warming BGE embedding model...")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _warmup_embeddings)
        logger.info("✅ Embedding model ready.")
    except Exception as e:
        logger.warning(f"⚠️ Embedding pre-warm failed: {e}")
    yield

# ✅ app is defined BEFORE any @app decorators
app = FastAPI(
    title="RAG Chatbot API",
    description="Internship Assessment — RAG pipeline with LlamaIndex & LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(kb.router)

@app.get("/")
def root():
    return {"status": "Ragbot API is running!", "docs": "/docs"}

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "message": "RAG Chatbot API is running"}
