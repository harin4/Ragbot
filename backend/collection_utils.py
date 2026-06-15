"""
collection_utils.py – helpers for per-URL Qdrant collection naming.

Each ingested URL gets its own collection so sources stay isolated and can be
deleted or inspected independently.

Naming scheme:  rag_{framework}_{domain_slug}_{md5_12}
Example:        rag_llamaindex_www_f22labs_com_a1b2c3d4e5f6
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse


def url_to_collection_name(url: str, framework: str) -> str:
    prefix = "rag_llamaindex" if framework == "llamaindex" else "rag_langchain"
    parsed = urlparse(url)
    domain = re.sub(r"[^a-z0-9]+", "_", parsed.netloc.lower()).strip("_")[:20]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{prefix}_{domain}_{url_hash}"


def framework_prefix(framework: str) -> str:
    return "rag_llamaindex" if framework == "llamaindex" else "rag_langchain"
