from fastapi import APIRouter, HTTPException

from kb_service import get_collection_count, get_sources, delete_collection, delete_url_collection

router = APIRouter(tags=["knowledge-base"])

_FRAMEWORKS = {"llamaindex", "langchain"}


def _check_framework(framework: str) -> None:
    if framework not in _FRAMEWORKS:
        raise HTTPException(status_code=400, detail="framework must be 'llamaindex' or 'langchain'")


@router.get("/collections")
async def get_collections() -> dict:
    """Return total chunk counts keyed by framework name."""
    return {
        "llamaindex": get_collection_count("llamaindex"),
        "langchain": get_collection_count("langchain"),
    }


@router.get("/collections/{framework}/sources")
async def get_collection_sources(framework: str) -> list:
    """Return one entry per ingested URL for a framework."""
    _check_framework(framework)
    return get_sources(framework)


@router.delete("/collections/{framework}")
async def delete_kb(framework: str) -> dict:
    """Delete all URL collections for a framework."""
    _check_framework(framework)
    delete_collection(framework)
    return {"status": "deleted", "framework": framework}


@router.delete("/collections/{framework}/url/{collection_name}")
async def delete_url_kb(framework: str, collection_name: str) -> dict:
    """Delete a single URL's collection by its exact collection name."""
    _check_framework(framework)
    delete_url_collection(collection_name)
    return {"status": "deleted", "collection": collection_name}
