"""
scraper.py – fetch and clean HTML from a URL using Playwright.
Returns a list of plain-text chunks plus page metadata.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import trafilatura
from playwright.async_api import async_playwright


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
    """Fetch *url* with a real browser, strip boilerplate, return a ScrapedPage with text chunks."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=int(timeout * 1000))

            title = await page.title() or url
            html = await page.content()

            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )

            if not extracted:
                # Fall back to raw body text when trafilatura yields nothing
                extracted = await page.evaluate("() => document.body.innerText")
        finally:
            await browser.close()

    clean = _clean_text(extracted or "")
    if not clean:
        raise ValueError(f"Could not extract any text content from {url}")

    chunks = _split_into_chunks(clean, chunk_size, overlap)
    return ScrapedPage(url=url, title=title, text=clean, chunks=chunks)
