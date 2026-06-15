"""
kb_service.py – Qdrant helpers that work across per-URL collections.

Each ingested URL lives in its own collection named:
  rag_{framework}_{domain}_{md5_12}

All functions here accept a framework string ("llamaindex" or "langchain")
and operate across every collection whose name starts with the matching prefix.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from config import get_settings
from collection_utils import framework_prefix

settings = get_settings()


def _get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def get_collection_count(framework: str) -> int:
    """Return total vectors stored across all URL collections for a framework."""
    client = _get_client()
    prefix = framework_prefix(framework)
    total = 0
    try:
        for col in client.get_collections().collections:
            if col.name.startswith(prefix):
                info = client.get_collection(col.name)
                total += info.points_count or 0
    except Exception:
        pass
    return total


def get_sources(framework: str) -> list[dict]:
    """Return one entry per ingested URL for a framework's collections."""
    client = _get_client()
    prefix = framework_prefix(framework)
    sources: list[dict] = []
    try:
        for col in client.get_collections().collections:
            if not col.name.startswith(prefix):
                continue
            info = client.get_collection(col.name)
            chunks = info.points_count or 0
            # Pull title/source from the first stored point
            points, _ = client.scroll(
                collection_name=col.name,
                with_payload=True,
                with_vectors=False,
                limit=1,
            )
            source = col.name
            title = col.name
            if points:
                payload = points[0].payload or {}
                meta = payload.get("metadata") or payload
                source = meta.get("source", col.name)
                title = meta.get("title", source)
            sources.append({
                "collection": col.name,
                "source": source,
                "title": title,
                "chunks": chunks,
            })
    except Exception:
        pass
    return sorted(sources, key=lambda x: x["chunks"], reverse=True)


def delete_collection(framework: str) -> bool:
    """Delete every URL collection for a framework. Returns True on success."""
    client = _get_client()
    prefix = framework_prefix(framework)
    try:
        for col in client.get_collections().collections:
            if col.name.startswith(prefix):
                client.delete_collection(col.name)
        return True
    except Exception:
        return False


def delete_url_collection(collection_name: str) -> bool:
    """Delete a single URL collection by its exact name."""
    client = _get_client()
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
