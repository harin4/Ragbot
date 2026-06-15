import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from schemas import ChatRequest
import rag_llamaindex
import rag_langchain

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    SSE streaming chat endpoint.
    Events:
      data: {"type": "token",   "data": "<token>"}
      data: {"type": "sources", "data": [{"content":..., "source":..., "title":...}, ...]}
      data: {"type": "done"}
      data: {"type": "error",   "data": "<message>"}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    framework = req.framework

    def _generate():
        try:
            if framework == "llamaindex":
                gen, sources = rag_llamaindex.chat_stream(req.question, req.chat_history)
            elif framework == "langchain":
                gen, sources = rag_langchain.chat_stream(req.question, req.chat_history)
            else:
                yield f"data: {json.dumps({'type': 'error', 'data': 'framework must be llamaindex or langchain'})}\n\n"
                return

            for token in gen:
                yield f"data: {json.dumps({'type': 'token', 'data': token})}\n\n"

            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
