"""FastAPI app: CORS, rate limiting, demo-key middleware, and route registration."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.middleware import DemoKeyMiddleware, MaxUploadBodySizeMiddleware
from api.routes import chat, conversations, documents
from sage.db.database import get_session, init_db
from sage.db.models import Document
from sage.embed.local_embedder import _get_model as _get_embedding_model
from sage.retrieval.reranker import _get_model as _get_reranker_model

logger = logging.getLogger(__name__)

# Built frontend assets (frontend/dist, produced by `npm run build` -- see
# the root Dockerfile's frontend-build stage). Not present in local dev
# unless the frontend has been built, and never present in the test
# environment -- `check_dir=False` below (rather than a plain StaticFiles
# mount, which raises at import time if pointed at a missing directory) is
# what lets this module still import cleanly either way.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _warn_if_corpus_empty() -> None:
    """Log a loud warning at startup if no documents are ingested yet.

    Doesn't fail startup -- an empty corpus is completely normal for a fresh
    local dev environment before the first `sage ingest`. But the Hugging
    Face demo image (see Dockerfile / deploy/huggingface/prebuilt/README.md)
    is meant to ship with a pre-ingested corpus baked in; if that image ever
    gets built with the still-placeholder `sage.db`/`chroma/` (see that
    README's "Current status"), every query would silently fall through the
    relevance gate with no answer, for a reason that isn't obvious. This
    check surfaces that specific misconfiguration in the deployment's boot
    logs instead of only being discoverable by trying a query by hand.
    """
    session = get_session()
    try:
        document_count = session.query(Document).count()
    finally:
        session.close()

    if document_count == 0:
        logger.warning(
            "Startup check: no documents are ingested (db/sage.db has zero Document "
            "rows). Every query will hit the relevance gate and refuse to answer "
            "until at least one document is ingested (`sage ingest` or "
            "POST /documents/upload). If this is the Hugging Face demo image, this "
            "means it was built with the placeholder corpus -- see "
            "deploy/huggingface/prebuilt/README.md."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _warn_if_corpus_empty()
    # Load both lazy-singleton models once at startup, off the request path
    # entirely -- otherwise the first requests after a cold start race to
    # build them (see each module's _get_model() locking) and cold-start
    # latency lands on whichever user happens to ask first.
    _get_embedding_model()
    _get_reranker_model()
    yield


app = FastAPI(title="Sage API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Added in this order (Starlette wraps middleware in *reverse* add order,
# so CORSMiddleware -- added last -- ends up outermost, MaxUploadBodySize
# next, DemoKey innermost, then routes):
#   1. DemoKeyMiddleware: header check, no body access.
#   2. MaxUploadBodySizeMiddleware: rejects an oversized /documents/upload
#      request as bytes arrive, before Starlette's own multipart parser
#      (which has no size cap for actual file parts) buffers the whole
#      thing -- see api/middleware.py's docstring. Placed before DemoKey in
#      wrapping order (i.e. checked first) so an oversized request is
#      rejected without spending any effort on the demo-key check.
#   3. CORSMiddleware outermost so a 401/413 either of the above produces
#      still carries CORS headers a browser JS client can actually read.
app.add_middleware(DemoKeyMiddleware)
app.add_middleware(MaxUploadBodySizeMiddleware, path_prefix="/documents/upload")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(documents.router)

# `app.frontend()` (not a plain `app.mount(StaticFiles(...))`) registers the
# built frontend as explicitly *low-priority*: FastAPI checks /chat,
# /conversations, /documents first and only falls back to serving a static
# file if nothing above matched at all. A plain root-mounted `Mount` doesn't
# have that distinction -- Starlette's router treats it as a full match for
# every path, including ones an API route already owns under a different
# HTTP method (e.g. GET /chat, which only has a POST handler), so it used to
# win the routing race and produce a static-file 404 instead of the correct
# 405. `fallback="auto"` (the default) also serves index.html for
# client-side-routed paths like /conversations/123 on a hard refresh, if
# frontend/dist has one.
app.frontend("/", directory=str(_FRONTEND_DIST), check_dir=False)
