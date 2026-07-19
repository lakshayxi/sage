"""Direct ASGI-level tests for api/middleware.py.

MaxUploadBodySizeMiddleware is tested here against a bare ASGI app (not the
full FastAPI app) specifically to exercise the byte-counting fallback path
independently of Content-Length -- TestClient/httpx (used in test_api.py)
always sends an accurate Content-Length for a files= upload, so a
TestClient-only test suite would never actually prove the "missing or
understated Content-Length" defense works, only the fast-path header check.
"""

import pytest

from api.middleware import MaxUploadBodySizeMiddleware
from config import settings


async def _draining_app(scope, receive, send):
    """A minimal downstream app that just reads the whole body, like
    Starlette's multipart parser would, with no size limit of its own."""
    total = 0
    while True:
        message = await receive()
        total += len(message.get("body", b""))
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": str(total).encode()})


def _make_receiver(chunks: list[tuple[bytes, bool]]):
    state = {"i": 0}

    async def receive():
        if state["i"] < len(chunks):
            body, more_body = chunks[state["i"]]
            state["i"] += 1
            return {"type": "http.request", "body": body, "more_body": more_body}
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


async def _run(middleware, headers, chunks):
    scope = {"type": "http", "path": "/documents/upload", "headers": headers}
    receive = _make_receiver(chunks)
    responses = []

    async def send(message):
        responses.append(message)

    await middleware(scope, receive, send)
    status = next(m["status"] for m in responses if m["type"] == "http.response.start")
    body = b"".join(m["body"] for m in responses if m["type"] == "http.response.body")
    return status, body


@pytest.mark.anyio
async def test_rejects_via_content_length_before_reading_any_body(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1000)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    status, body = await _run(
        middleware,
        headers=[(b"content-length", b"999999")],
        chunks=[(b"should never be read", False)],
    )

    assert status == 413
    assert b"1000" in body


@pytest.mark.anyio
async def test_rejects_via_byte_counting_when_content_length_missing(monkeypatch):
    """No Content-Length header at all (e.g. chunked transfer-encoding) --
    the fast-path header check can't catch this; the request must still be
    bounded by counting real bytes as they arrive."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1000)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    status, _body = await _run(
        middleware,
        headers=[],
        chunks=[(b"x" * 600, True), (b"y" * 600, False)],  # 1200 bytes, over the cap
    )

    assert status == 413


@pytest.mark.anyio
async def test_rejects_via_byte_counting_when_content_length_understated(monkeypatch):
    """A Content-Length header that lies (understates the real size) must
    not bypass the limit -- only the actual bytes received matter."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1000)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    status, _body = await _run(
        middleware,
        headers=[(b"content-length", b"10")],
        chunks=[(b"z" * 600, True), (b"w" * 600, False)],
    )

    assert status == 413


@pytest.mark.anyio
async def test_allows_upload_within_the_limit(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1000)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    status, body = await _run(
        middleware,
        headers=[(b"content-length", b"500")],
        chunks=[(b"a" * 500, False)],
    )

    assert status == 200
    assert body == b"500"


@pytest.mark.anyio
async def test_does_not_apply_to_other_paths(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    status, body = await _run(
        middleware,
        headers=[(b"content-length", b"500")],
        chunks=[(b"a" * 500, False)],
    )
    # Wrong path prefix -- middleware must be a no-op regardless of size.

    async def run_other_path():
        scope = {"type": "http", "path": "/chat", "headers": [(b"content-length", b"500")]}
        receive = _make_receiver([(b"a" * 500, False)])
        responses = []

        async def send(message):
            responses.append(message)

        await middleware(scope, receive, send)
        return responses

    responses = await run_other_path()
    status = next(m["status"] for m in responses if m["type"] == "http.response.start")
    assert status == 200


@pytest.mark.anyio
async def test_reads_max_upload_bytes_live_not_frozen_at_construction(monkeypatch):
    """Regression test: the middleware must not capture settings.MAX_UPLOAD_BYTES
    once in __init__ (module-import time in api/main.py) -- that would silently
    ignore monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", ...) in tests and
    any other runtime reconfiguration."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1_000_000)
    middleware = MaxUploadBodySizeMiddleware(_draining_app, path_prefix="/documents/upload")

    # Constructed while the limit was large; now shrink it before the request.
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 100)

    status, _body = await _run(
        middleware,
        headers=[(b"content-length", b"500")],
        chunks=[(b"a" * 500, False)],
    )

    assert status == 413
