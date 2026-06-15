from fastapi import APIRouter, HTTPException

from schemas import IngestRequest, IngestResponse
from scraper import scrape_url, text_to_page
from collection_utils import url_to_collection_name
import rag_llamaindex
import rag_langchain
from config import get_settings

router = APIRouter(tags=["ingest"])
cfg = get_settings()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    if req.raw_text and req.raw_text.strip():
        page = text_to_page(
            text=req.raw_text,
            title=req.title or req.url,
            source_url=req.url,
            chunk_size=cfg.chunk_size,
            overlap=cfg.chunk_overlap,
        )
    else:
        try:
            page = await scrape_url(req.url, cfg.chunk_size, cfg.chunk_overlap)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape URL: {e}")

    if not page.text:
        raise HTTPException(status_code=422, detail="No meaningful text found at that URL.")

    collection_name = url_to_collection_name(req.url, req.framework)

    try:
        if req.framework == "llamaindex":
            n = rag_llamaindex.ingest_documents([page.text], page.url, page.title, collection_name)
        elif req.framework == "langchain":
            n = rag_langchain.ingest_documents([page.text], page.url, page.title, collection_name)
        else:
            raise HTTPException(status_code=400, detail="framework must be 'llamaindex' or 'langchain'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

    return IngestResponse(
        status="success",
        chunks_stored=n,
        url=req.url,
        framework=req.framework,
        message=f"Stored {n} chunks from {req.url}",
        title=page.title,
    )
