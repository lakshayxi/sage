"""GET /documents, POST /documents/upload."""

from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile

from api.limiter import limiter
from api.schemas import DocumentOut, UploadResponse
from config import settings
from sage.db.database import get_session
from sage.db.models import Document
from sage.ingest.pipeline import ingest_pdf

router = APIRouter()


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
    company: str | None = Form(None),
    fiscal_year: str | None = Form(None),
    doc_type: str | None = Form(None),
) -> UploadResponse:
    if not settings.ALLOW_UPLOADS:
        raise HTTPException(status_code=403, detail="Uploads are disabled on this deployment")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    safe_name = Path(file.filename).name
    if safe_name in ("", ".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest_path = settings.RAW_DIR / safe_name
    if not dest_path.resolve().is_relative_to(settings.RAW_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")

    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(file.file.read())

    overrides = {"company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}
    try:
        document = ingest_pdf(dest_path, metadata_overrides=overrides)
    except Exception:
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
