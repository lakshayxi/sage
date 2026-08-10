"""ASGI-level public input guards for access, body size, and header size.

When `settings.DEMO_ACCESS_KEY` is unset (the local-dev default), its gate is
a complete no-op. When set, every request under /chat, /conversations, or
/documents must carry a matching key via the `X-Demo-Key` header or get a
401. Static assets are intentionally not covered by the protected prefixes.

IMPORTANT -- this is a shared casual-access deterrent, not a real secret or
an authentication boundary: `VITE_DEMO_ACCESS_KEY` (the frontend's copy of
this same value, see frontend/src/api/session.ts) is compiled into the
public JS bundle by Vite at build time, so anyone can read it straight out
of the deployed site's own source. It stops a search-engine crawler or a
casual visitor from stumbling onto an unlisted demo URL and burning the
shared Gemini quota; it does not stop a deliberate attacker, who can just
copy the key out of the bundle. The real defense against abuse of the
costly Gemini-backed endpoints is `CHAT_RATE_LIMIT` (per-IP, applied to
every chat/upload route) -- see README's Security notes. Only a header is
accepted (no `key` query-param fallback): a query param would land in
server access logs, browser history, and any Referer header a request
leaks through, for no benefit now that every client here (chat.ts included)
drives requests with `fetch()` and can set real headers.
"""

import json

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config import settings

_PROTECTED_PREFIXES = ("/chat", "/conversations", "/documents")


class DemoKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.DEMO_ACCESS_KEY:
            return await call_next(request)

        if request.url.path.startswith(_PROTECTED_PREFIXES):
            supplied = request.headers.get("X-Demo-Key")
            if supplied != settings.DEMO_ACCESS_KEY:
                return JSONResponse(
                    {"detail": "Missing or invalid demo access key"}, status_code=401
                )

        return await call_next(request)


class _BodyTooLarge(Exception):
    pass


class _MaxBodySizeMiddleware:
    """Shared streaming body guard used by the API and upload limits."""

    app: ASGIApp

    def _applies(self, scope: Scope) -> bool:
        raise NotImplementedError

    def _max_bytes(self) -> int:
        raise NotImplementedError

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._applies(scope):
            await self.app(scope, receive, send)
            return

        max_bytes = self._max_bytes()
        content_length = _parse_content_length(scope.get("headers", []))
        if content_length is not None and content_length > max_bytes:
            await _reject_413(send, max_bytes)
            return

        total = 0

        async def guarded_receive() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, guarded_receive, send)
        except _BodyTooLarge:
            await _reject_413(send, max_bytes)


class MaxRequestBodySizeMiddleware(_MaxBodySizeMiddleware):
    """Bound ordinary request bodies before FastAPI buffers and validates them.

    Upload paths are explicitly excluded because multipart requests have a
    practical, larger limit enforced by `MaxUploadBodySizeMiddleware`. The
    setting is read for each request so tests and runtime configuration behave
    consistently with the upload guard.
    """

    def __init__(self, app: ASGIApp, excluded_path_prefixes: tuple[str, ...] = ()) -> None:
        self.app = app
        self.excluded_path_prefixes = excluded_path_prefixes

    def _applies(self, scope: Scope) -> bool:
        return not scope["path"].startswith(self.excluded_path_prefixes)

    def _max_bytes(self) -> int:
        return settings.MAX_REQUEST_BODY_BYTES


class MaxUploadBodySizeMiddleware(_MaxBodySizeMiddleware):
    """Rejects a request body over `settings.MAX_UPLOAD_BYTES` for paths
    under `path_prefix`, enforced as bytes actually arrive from the client.

    This has to be a plain ASGI middleware, not `BaseHTTPMiddleware`:
    `BaseHTTPMiddleware.dispatch` would need to call `request.body()` to
    inspect the body itself, which fully buffers it in memory first --
    defeating the entire point of a size guard.

    Why this exists at all, separately from `api/routes/documents.py`'s own
    `MAX_UPLOAD_BYTES` check: Starlette's multipart form parser
    (`starlette.formparsers.MultiPartParser.on_part_data`) enforces
    `max_part_size` only for plain text form fields, never for actual file
    parts. FastAPI resolves the `UploadFile` parameter (i.e. runs that
    parser to completion) *before* the route function body executes at
    all, so an oversized upload is already fully read and spooled to a
    temp file by Starlette regardless of any check inside the route --
    confirmed empirically (a 5MB fake upload against a parser with no
    explicit cap sails through with zero enforcement). This middleware is
    the layer that actually stops that, by capping bytes at the ASGI
    `receive()` level, before Starlette's form parser ever sees them.

    Two layers: an honest (or accidentally huge) `Content-Length` header is
    rejected immediately, before reading a single body byte; actual
    received bytes are also counted as they stream in, so a missing,
    understated, or absent (chunked transfer-encoding) `Content-Length`
    can't bypass the limit either.

    Reads `settings.MAX_UPLOAD_BYTES` fresh on every request rather than
    capturing it once in `__init__` -- `app.add_middleware(...)` runs at
    module-import time (`api/main.py`), so a captured value would be frozen
    at whatever the setting was then. That would silently ignore
    `monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", ...)` in tests (and
    any other runtime reconfiguration), unlike every other place in this
    codebase that treats `settings.X` as live.
    """

    def __init__(self, app: ASGIApp, path_prefix: str) -> None:
        self.app = app
        self.path_prefix = path_prefix

    def _applies(self, scope: Scope) -> bool:
        return scope["path"].startswith(self.path_prefix)

    def _max_bytes(self) -> int:
        return settings.MAX_UPLOAD_BYTES


class SessionTokenLengthMiddleware:
    """Reject an oversized session token before it reaches a database lookup."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and any(
            name.lower() == b"x-session-token" and len(value) > settings.MAX_SESSION_TOKEN_LENGTH
            for name, value in scope.get("headers", [])
        ):
            await _reject_json(send, 400, "Invalid request header")
            return
        await self.app(scope, receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply browser hardening to API errors and the single-page frontend."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "base-uri 'self'; object-src 'none'; form-action 'self'; "
            "frame-ancestors 'self' https://huggingface.co https://*.hf.space",
        )
        return response


def _parse_content_length(headers: list[tuple[bytes, bytes]]) -> int | None:
    for name, value in headers:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _reject_413(send: Send, max_bytes: int) -> None:
    await _reject_json(send, 413, f"Request body exceeds the maximum of {max_bytes} bytes")


async def _reject_json(send: Send, status_code: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
