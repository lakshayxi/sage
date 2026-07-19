"""GET /documents, POST /documents/upload.

Uploads are the one place this API accepts arbitrary user-supplied binary
content and hands it to a parser (PyMuPDF) and a synchronous, in-request
ingestion pipeline -- see settings.ALLOW_UPLOADS/MAX_UPLOAD_BYTES/
MAX_UPLOAD_PAGES for the hardening this implies: disabled by default,
streamed to disk with a hard size cap instead of buffered in memory via
`file.file.read()`, verified to actually be a readable PDF (not just named
*.pdf) before ingestion, and cleaned up on any failure so a rejected or
partially-ingested upload never leaves an orphaned file in data/raw/.

The request body's own size is additionally capped at the ASGI layer
(`api.middleware.MaxUploadBodySizeMiddleware`) -- Starlette's multipart
parser has no size limit on actual file parts and fully buffers/spools an
upload to disk *before* this route function ever runs, so MAX_UPLOAD_BYTES
alone (enforced only in `_stream_upload_to_disk` below) is not sufficient
by itself; see that middleware's docstring.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import fitz
from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from api.limiter import limiter
from api.schemas import DocumentOut, UploadResponse
from config import settings
from sage.db.database import get_session
from sage.db.models import Document
from sage.ingest.pipeline import ingest_pdf

router = APIRouter()

_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB per read, not the whole file at once
_PDF_MAGIC = b"%PDF-"
_MAX_METADATA_FIELD_LENGTH = 200
_STAGING_PREFIX = ".upload-"


def _validate_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if safe_name in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not (settings.RAW_DIR / safe_name).resolve().is_relative_to(settings.RAW_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe_name


def _stream_upload_to_staging_file(file: UploadFile) -> Path:
    """Write the upload to a uniquely-named staging file under RAW_DIR, in
    bounded chunks -- never loading the whole file into memory via
    `file.file.read()` -- enforcing MAX_UPLOAD_BYTES as bytes arrive so an
    oversized upload is rejected (and its partial bytes cleaned up) without
    ever fully landing on disk.

    Staging under a `tempfile.mkstemp`-style unique name (rather than
    writing directly to the upload's own filename) means two concurrent
    uploads -- of the same or different filenames -- can never write to the
    same path, and a still-in-progress or failed upload never touches an
    already-ingested document's file at its real name. `_claim_destination`
    below moves this staging file into place under that real name only
    once the upload is fully validated.
    """
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=settings.RAW_DIR, prefix=_STAGING_PREFIX, suffix=".pdf")
    staging_path = Path(tmp_name)
    total_bytes = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                block = file.file.read(_UPLOAD_READ_CHUNK_BYTES)
                if not block:
                    break
                total_bytes += len(block)
                if total_bytes > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the maximum upload size of "
                            f"{settings.MAX_UPLOAD_BYTES} bytes"
                        ),
                    )
                out.write(block)
    except HTTPException:
        staging_path.unlink(missing_ok=True)
        raise
    except OSError:
        staging_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file") from None
    return staging_path


def _claim_destination(staging_path: Path, original_filename: str) -> Path:
    """Move a fully-validated staging file into place under a name derived
    from `original_filename`, without ever overwriting an existing file.

    Claims the name atomically via O_CREAT|O_EXCL (portable, race-free --
    two concurrent uploads that both want the same final filename can't
    both "win"): if that exact name is already taken -- by an earlier
    successfully-ingested document, or by another upload that won a
    concurrent race for it -- a short random suffix is appended instead of
    silently overwriting it.
    """
    safe_name = _validate_filename(original_filename)
    dest_path = settings.RAW_DIR / safe_name
    try:
        fd = os.open(dest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
        dest_path = settings.RAW_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        fd = os.open(dest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    staging_path.replace(dest_path)
    return dest_path


def _validate_pdf(path: Path) -> None:
    """Confirm the uploaded file is an actually-readable PDF (not just named
    *.pdf) and within the page-count limit, before it reaches the ingest
    pipeline. Raises HTTPException(400/413) and does not clean up `path` --
    callers are responsible for that (see `upload_document`)."""
    with path.open("rb") as f:
        header = f.read(len(_PDF_MAGIC))
    if header != _PDF_MAGIC:
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    try:
        doc = fitz.open(path)
        try:
            page_count = doc.page_count
        finally:
            doc.close()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a readable PDF") from None

    if page_count > settings.MAX_UPLOAD_PAGES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the maximum of {settings.MAX_UPLOAD_PAGES} pages",
        )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    session = get_session()
    try:
        documents = session.query(Document).order_by(Document.ingested_at.desc()).all()
        return [
            DocumentOut(
                id=d.id,
                filename=d.filename,
                title=d.title,
                company=d.company,
                fiscal_year=d.fiscal_year,
                doc_type=d.doc_type,
                page_count=d.page_count,
                status=d.status,
                ingested_at=d.ingested_at,
            )
            for d in documents
        ]
    finally:
        session.close()


@router.post("/documents/upload", response_model=UploadResponse)
@limiter.limit(settings.CHAT_RATE_LIMIT)
def upload_document(
    request: Request,
    file: UploadFile,
    company: Annotated[str | None, Form(max_length=_MAX_METADATA_FIELD_LENGTH)] = None,
    fiscal_year: Annotated[str | None, Form(max_length=_MAX_METADATA_FIELD_LENGTH)] = None,
    doc_type: Annotated[str | None, Form(max_length=_MAX_METADATA_FIELD_LENGTH)] = None,
) -> UploadResponse:
    if not settings.ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploads are disabled on this deployment")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    _validate_filename(file.filename)  # fail fast on a bad filename before writing anything

    staging_path = _stream_upload_to_staging_file(file)
    dest_path: Path | None = None
    try:
        _validate_pdf(staging_path)
        dest_path = _claim_destination(staging_path, file.filename)
        overrides = {"company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}
        document = ingest_pdf(dest_path, metadata_overrides=overrides)
    except HTTPException:
        staging_path.unlink(missing_ok=True)
        if dest_path is not None:
            dest_path.unlink(missing_ok=True)
        raise
    except Exception:
        staging_path.unlink(missing_ok=True)
        if dest_path is not None:
            dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to ingest document") from None

    return UploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        page_count=document.page_count,
        company=document.company,
        fiscal_year=document.fiscal_year,
        doc_type=document.doc_type,
    )
