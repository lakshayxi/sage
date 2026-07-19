"""Full ingest pipeline: load -> chunk -> persist (SQLite) -> embed -> store (Chroma).

The single code path the CLI's `ingest` command and the API's upload route
(`api/routes/documents.py`) both go through.

SQLite and Chroma are two separate stores with no shared transaction, so a
failure partway through (embedding call, Chroma write, or the final SQLite
commit itself) can't be made atomic across both the way a single-database
transaction would be. What this function does instead:

- **Idempotency via checksum.** Before doing any real work, the raw file's
  sha256 is checked against already-`ready` documents; an exact repeat
  upload/re-ingest returns the existing document rather than creating a
  duplicate Document/Chunk set or re-spending an embedding pass.
- **Compensation on failure.** If anything raises after Chroma vectors have
  already been written for this attempt, those specific vectors are deleted
  before the exception propagates -- SQLite's own transaction already rolls
  back the Document/Chunk rows on any exception (see the `except` clause
  below), so without this, a failure after `store.add()` but before
  `session.commit()` would leave orphaned vectors in Chroma pointing at
  chunk ids that no longer exist anywhere in SQLite.
- **SQLite stays the source of truth for chunk text** (`store.add()` below
  passes no `documents=`) regardless of which side fails first.
"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from config import settings
from sage.db.database import get_session
from sage.db.models import Chunk, Document
from sage.embed.local_embedder import embed_texts
from sage.ingest.chunker import chunk_pages
from sage.ingest.metadata import DocumentMetadata, apply_overrides, parse_filename_metadata
from sage.ingest.pdf_loader import load_pdf_pages
from sage.retrieval import store

logger = logging.getLogger(__name__)


def _file_checksum(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _find_ready_document_by_checksum(checksum: str) -> Document | None:
    session = get_session()
    try:
        existing = (
            session.query(Document)
            .filter(Document.checksum == checksum, Document.status == "ready")
            .first()
        )
        if existing is not None:
            session.expunge(existing)
        return existing
    finally:
        session.close()


def ingest_pdf(
    pdf_path: Path,
    write_json: bool = True,
    metadata_overrides: dict | None = None,
) -> Document:
    """Ingest a single PDF: extract, chunk, embed, and persist it.

    Returns the persisted Document row (detached from its session). If a
    document with identical file content (by checksum) has already been
    successfully ingested, that existing row is returned unchanged instead
    of re-processing -- this function is safe to call again for the same
    file (e.g. a retried upload) without creating duplicates.
    """
    checksum = _file_checksum(pdf_path)
    existing = _find_ready_document_by_checksum(checksum)
    if existing is not None:
        logger.info(
            "Skipping ingestion of %s: identical content already ingested as document_id=%s (%s)",
            pdf_path.name,
            existing.id,
            existing.filename,
        )
        return existing

    pages = load_pdf_pages(pdf_path)
    meta: DocumentMetadata = parse_filename_metadata(pdf_path.name)
    if metadata_overrides:
        meta = apply_overrides(meta, **metadata_overrides)

    chunks = chunk_pages(pages, settings.CHUNK_TOKENS, settings.CHUNK_OVERLAP_TOKENS)

    if write_json:
        settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "filename": pdf_path.name,
            "company": meta.company,
            "fiscal_year": meta.fiscal_year,
            "doc_type": meta.doc_type,
            "page_count": len(pages),
            "chunks": [asdict(c) for c in chunks],
        }
        out_path = settings.PROCESSED_DIR / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(record, indent=2))

    session = get_session()
    chroma_ids: list[str] = []
    try:
        document = Document(
            filename=pdf_path.name,
            title=pdf_path.stem.replace("_", " "),
            company=meta.company,
            fiscal_year=meta.fiscal_year,
            doc_type=meta.doc_type,
            source_path=str(pdf_path),
            page_count=len(pages),
            embedding_model=settings.LOCAL_EMBEDDING_MODEL,
            status="processing",
            checksum=checksum,
        )
        session.add(document)
        session.flush()  # assigns document.id

        chunk_rows = []
        for c in chunks:
            row = Chunk(
                document_id=document.id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                text=c.text,
                char_start=c.char_start,
                char_end=c.char_end,
                token_count=c.token_count,
            )
            session.add(row)
            chunk_rows.append(row)
        session.flush()  # assigns each row.id

        if chunk_rows:
            # Local embedding call -- sentence-transformers batches
            # internally, no rate-limit/quota concern to chunk around.
            embeddings = embed_texts([row.text for row in chunk_rows])
            ids, metadatas = [], []
            for row in chunk_rows:
                row.embedding_id = str(row.id)
                ids.append(str(row.id))
                metadatas.append(
                    {
                        "chunk_id": row.id,
                        "document_id": document.id,
                        "company": meta.company or "",
                        "fiscal_year": meta.fiscal_year or "",
                        "doc_type": meta.doc_type or "",
                        "page_number": row.page_number or 0,
                    }
                )
            # No `documents=` -- SQLite (the Chunk rows just written above) is
            # the sole source of truth for chunk text; Chroma only needs ids,
            # vectors, and filter metadata to serve retrieval. `chroma_ids`
            # is recorded before the call so the except clause below can
            # compensate even if store.add() itself is what raises (e.g. a
            # partial batch write).
            chroma_ids = ids
            store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

        document.status = "ready"
        session.commit()
        session.refresh(document)
        session.expunge(document)
        return document
    except Exception:
        session.rollback()
        if chroma_ids:
            try:
                store.delete(ids=chroma_ids)
            except Exception:
                logger.warning(
                    "Failed to clean up orphaned Chroma vectors for %s after "
                    "ingest failure -- %d vector(s) may still reference "
                    "chunk ids that no longer exist in SQLite",
                    pdf_path.name,
                    len(chroma_ids),
                    exc_info=True,
                )
        raise
    finally:
        session.close()


def ingest_folder(input_dir: Path) -> list[Document]:
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    return [ingest_pdf(p) for p in pdf_paths]
