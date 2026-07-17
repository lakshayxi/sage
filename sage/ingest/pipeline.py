"""Full ingest pipeline: load -> chunk -> persist (SQLite) -> embed -> store (Chroma).

The single code path the CLI's `ingest` command (and, later, an API upload
route) goes through.
"""

import json
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


def ingest_pdf(
    pdf_path: Path,
    write_json: bool = True,
    metadata_overrides: dict | None = None,
) -> Document:
    """Ingest a single PDF: extract, chunk, embed, and persist it.

    Returns the persisted Document row (detached from its session).
    """
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
            # vectors, and filter metadata to serve retrieval.
            store.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

        document.status = "ready"
        session.commit()
        session.refresh(document)
        session.expunge(document)
        return document
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ingest_folder(input_dir: Path) -> list[Document]:
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    return [ingest_pdf(p) for p in pdf_paths]
