"""Shared demo-access-key gate for the public deployment.

When `settings.DEMO_ACCESS_KEY` is unset (the local-dev default), this is a
complete no-op -- the request is passed straight through with no header
check. When set, every request under /chat, /conversations, or /documents
must carry a matching key or get a 401, via either the `X-Demo-Key` header
(fetch-based callers: conversations.ts, documents.ts) or a `key` query
param (GET /chat/stream specifically -- the browser's native EventSource,
used by chat.ts, cannot set custom headers, so that's the only way it can
send this at all). Static assets (once a frontend is served from this same
app) are intentionally not covered by the prefixes below.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import settings

_PROTECTED_PREFIXES = ("/chat", "/conversations", "/documents")


class DemoKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.DEMO_ACCESS_KEY:
            return await call_next(request)

        if request.url.path.startswith(_PROTECTED_PREFIXES):
            supplied = request.headers.get("X-Demo-Key") or request.query_params.get("key")
            if supplied != settings.DEMO_ACCESS_KEY:
                return JSONResponse(
                    {"detail": "Missing or invalid demo access key"}, status_code=401
                )

        return await call_next(request)
