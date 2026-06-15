from fastapi import APIRouter, HTTPException

from kb_service import get_collection_count, delete_collection
from config import get_settings

router = APIRouter(tags=["knowledge-base"])
cfg = get_settings()


@router.get("/collections")
async def get_collections() -> dict:
    """Return chunk counts for both framework collections."""
    return {
        cfg.qdrant_collection_llamaindex: get_collection_count(cfg.qdrant_collection_llamaindex),
        cfg.qdrant_collection_langchain: get_collection_count(cfg.qdrant_collection_langchain),
    }


@router.delete("/collections/{framework}")
async def delete_kb(framework: str) -> dict:
    """Delete all vectors for a given framework's collection."""
    if framework == "llamaindex":
        collection = cfg.qdrant_collection_llamaindex
    elif framework == "langchain":
        collection = cfg.qdrant_collection_langchain
    else:
        raise HTTPException(status_code=400, detail="framework must be 'llamaindex' or 'langchain'")

    delete_collection(collection)
    return {"status": "deleted", "framework": framework, "collection": collection}
