"""
Utility: query Qdrant for collection point counts.
"""

from qdrant_client import QdrantClient
from config import get_settings

settings = get_settings()


def _get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def get_collection_count(collection_name: str) -> int:
    """Return number of vectors stored in a collection. Returns 0 if not found."""
    client = _get_client()
    try:
        info = client.get_collection(collection_name)
        return info.points_count or 0
    except Exception:
        return 0


def delete_collection(collection_name: str) -> bool:
    """Delete a collection entirely. Returns True on success."""
    client = _get_client()
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False