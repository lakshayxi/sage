"""FastAPI app: CORS, rate limiting, demo-key middleware, and route registration."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.limiter import limiter
from api.middleware import DemoKeyMiddleware
from api.routes import chat, conversations, documents
from sage.db.database import init_db
from sage.embed.local_embedder import _get_model as _get_embedding_model
from sage.retrieval.reranker import _get_model as _get_reranker_model

# Built frontend assets (frontend/dist, produced by `npm run build` -- see
# the root Dockerfile's frontend-build stage). Not present in local dev
# unless the frontend has been built, and never present in the test
# environment -- `check_dir=False` below (rather than a plain StaticFiles
# mount, which raises at import time if pointed at a missing directory) is
# what lets this module still import cleanly either way.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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

# DemoKeyMiddleware added first so CORSMiddleware (added last, and so
# outermost -- Starlette wraps middleware in reverse add order) still attaches
# CORS headers to a 401 it produces.
app.add_middleware(DemoKeyMiddleware)
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
