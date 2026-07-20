"""POST /chat (non-streaming), POST /chat/stream (SSE-style, fetch-driven)."""

import json

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.limiter import limiter
from api.schemas import ChatRequest, ChatResponse, CitationOut, LatencyOut, TokensOut
from config import settings
from sage.db.conversations import append_message, conversation_belongs_to_session, get_history
from sage.generation.answer_engine import AnswerResult, generate_answer, generate_answer_stream

router = APIRouter()

# The model is instructed to end its response with a ```citations fenced
# block, but doesn't always include the backtick fence (see
# sage/generation/answer_engine.py's citation-parsing fallbacks) -- watch for
# either form so the raw block never leaks into the live stream.
FENCE_MARKERS = ("```citations", "\ncitations\n[")


def _to_chat_response(result: AnswerResult, session_id: int | None) -> ChatResponse:
    return ChatResponse(
        answer=result.answer_text,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        model=result.model,
        latency_ms=LatencyOut(
            retrieval_ms=result.retrieval_latency_ms,
            generation_ms=result.generation_latency_ms,
            total_ms=result.total_latency_ms,
        ),
        tokens=TokensOut(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
        cache_hit=result.cache_hit,
        cost_usd=result.cost_usd,
        session_id=session_id,
    )


def _persist_turn(conversation_id: int, query: str, result: AnswerResult) -> None:
    append_message(conversation_id, "user", query)
    append_message(
        conversation_id,
        "assistant",
        result.answer_text,
        citations=[
            {
                "n": c.n,
                "chunk_id": c.chunk_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "company": c.company,
            }
            for c in result.citations
        ],
    )


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.CHAT_RATE_LIMIT)
def chat(
    request: Request,
    body: ChatRequest,
    x_session_token: str = Header(default="", alias="X-Session-Token"),
) -> ChatResponse:
    if body.session_id is not None and not conversation_belongs_to_session(
        body.session_id, x_session_token
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")

    history = get_history(body.session_id) if body.session_id is not None else None
    try:
        result = generate_answer(
            body.query,
            top_k=body.top_k,
            companies=body.companies,
            fiscal_year=body.fiscal_year,
            doc_type=body.doc_type,
            history=history,
            session_id=str(body.session_id) if body.session_id is not None else None,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate an answer") from None

    if body.session_id is not None:
        _persist_turn(body.session_id, body.query, result)

    try:
        return _to_chat_response(result, body.session_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate an answer") from None


@router.post("/chat/stream")
@limiter.limit(settings.CHAT_RATE_LIMIT)
def chat_stream(
    request: Request,
    body: ChatRequest,
    x_session_token: str = Header(default="", alias="X-Session-Token"),
):
    # POST with the query/filters in the JSON body and the session token in
    # a header -- same shape as POST /chat -- rather than GET with all of
    # that in the URL. A native EventSource can only ever GET and can't set
    # custom headers, which is exactly why this used to carry the session
    # token (and, via DemoKeyMiddleware, the demo access key) as URL query
    # params: both would land in server access logs, browser history, and
    # any Referer header a query could leak through. The frontend
    # (frontend/src/api/chat.ts) now drives this with fetch() + a streamed
    # ReadableStream instead of EventSource, so it can send real headers.
    if body.session_id is not None and not conversation_belongs_to_session(
        body.session_id, x_session_token
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")

    history = get_history(body.session_id) if body.session_id is not None else None

    def event_source():
        # See FENCE_MARKERS: a delta chunk could contain only part of a
        # marker, so a tail of unsent text (as long as the longest marker) is
        # always held back until either more text rules out a match, or a
        # marker is found -- ported from the reference project's chat_stream.
        buffer = ""
        sent_len = 0
        fence_found = False
        max_marker_len = max(len(m) for m in FENCE_MARKERS)
        result: AnswerResult | None = None

        try:
            for item in generate_answer_stream(
                body.query,
                top_k=body.top_k,
                companies=body.companies,
                fiscal_year=body.fiscal_year,
                doc_type=body.doc_type,
                history=history,
                session_id=str(body.session_id) if body.session_id is not None else None,
            ):
                if isinstance(item, str):
                    if fence_found:
                        continue
                    buffer += item
                    fence_idx = min(
                        (i for i in (buffer.find(m) for m in FENCE_MARKERS) if i != -1),
                        default=-1,
                    )
                    if fence_idx != -1:
                        safe_text = buffer[sent_len:fence_idx]
                        if safe_text:
                            yield f"data: {json.dumps({'delta': safe_text})}\n\n"
                        fence_found = True
                    else:
                        safe_upto = max(sent_len, len(buffer) - max_marker_len)
                        if safe_upto > sent_len:
                            yield f"data: {json.dumps({'delta': buffer[sent_len:safe_upto]})}\n\n"
                            sent_len = safe_upto
                else:
                    result = item
        except Exception:
            error_payload = {"detail": "Failed to generate an answer"}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return

        # If the stream ended without ever finding a citations fence (e.g. the
        # no-relevant-context refusal message, which is a fixed string with no
        # fence at all), the last `max_marker_len` chars of `buffer` are still
        # sitting unsent -- held back on the chance they were the start of a
        # fence that never arrived. Flush them now, otherwise every such
        # answer's live-streamed text is silently missing its last few
        # characters even though `result.answer_text` (used below and in the
        # `done` payload) always has the correct, untruncated text.
        if not fence_found and sent_len < len(buffer):
            yield f"data: {json.dumps({'delta': buffer[sent_len:]})}\n\n"

        if body.session_id is not None and result is not None:
            _persist_turn(body.session_id, body.query, result)

        # Deltas (and the turn, above) are already sent/persisted by this
        # point -- a response-shape mismatch here must still reach the
        # client as an `error` event rather than silently dropping the
        # `done` event and leaving the request hanging with no signal that
        # generation actually succeeded.
        try:
            payload = _to_chat_response(result, body.session_id).model_dump()
        except Exception:
            error_payload = {"detail": "Failed to generate an answer"}
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            return
        yield f"event: done\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
