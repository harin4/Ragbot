"""
scraper.py – fetch and clean HTML from a URL.
Returns a list of plain-text chunks plus page metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup


_SKIP_TAGS = {
    "script", "style", "noscript", "nav", "footer",
    "header", "aside", "form", "button", "svg", "img",
}

# Full browser headers — mimics Chrome 124 on Windows
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# Sites known to block all HTTP clients (need a real browser / JavaScript)
_JS_ONLY_SITES = {
    "medium.com",
    "substack.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "nytimes.com",
}


def _check_known_blocker(url: str) -> None:
    """Raise a helpful error before attempting if the site is known to block scrapers."""
    for domain in _JS_ONLY_SITES:
        if domain in url:
            raise ValueError(
                f"{domain} blocks automated scrapers — it requires JavaScript to render. "
                f"Use the 'Paste text' option in the sidebar instead, or try a different URL "
                f"(Wikipedia, arXiv, GitHub README, official docs, etc.)."
            )


@dataclass
class ScrapedPage:
    url: str
    title: str
    text: str
    chunks: list[str] = field(default_factory=list)


def _clean_text(raw: str) -> str:
    text = re.sub(r"[​­﻿]", "", raw)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def text_to_page(text: str, title: str, source_url: str,
                 chunk_size: int = 500, overlap: int = 50) -> ScrapedPage:
    """Build a ScrapedPage directly from pasted plain text (no HTTP request)."""
    clean = _clean_text(text)
    chunks = _split_into_chunks(clean, chunk_size, overlap)
    return ScrapedPage(url=source_url, title=title, text=clean, chunks=chunks)


async def scrape_url(
    url: str,
    chunk_size: int = 500,
    overlap: int = 50,
    timeout: float = 30.0,
) -> ScrapedPage:
    """Fetch *url*, strip boilerplate, return a ScrapedPage with text chunks."""
    _check_known_blocker(url)

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        http2=True,
    ) as client:
        resp = await client.get(url, headers=_HEADERS)

        # Retry without Sec-Fetch-* headers if first attempt is blocked
        if resp.status_code == 403:
            fallback = {k: v for k, v in _HEADERS.items() if not k.startswith("Sec-")}
            resp = await client.get(url, headers=fallback)

        if resp.status_code == 403:
            raise ValueError(
                f"403 Forbidden — {url} is blocking automated access. "
                f"Use the 'Paste text' option in the sidebar, or try a publicly accessible URL."
            )

        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(list(_SKIP_TAGS)):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else url

    body = (
        soup.find("article")
        or soup.find("main")
        or soup.find("body")
        or soup
    )

    raw_text = body.get_text(separator="\n")
    clean = _clean_text(raw_text)
    chunks = _split_into_chunks(clean, chunk_size, overlap)

    return ScrapedPage(url=url, title=title, text=clean, chunks=chunks)
