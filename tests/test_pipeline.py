from fpdf import FPDF

from sage.db.database import get_session
from sage.db.models import Chunk, Document
from sage.ingest import pipeline


def _make_sample_pdf(tmp_path, filename="Apple_FY24_10-K.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Apple reported FY24 results with strong revenue growth in Services.")
    pdf_path = tmp_path / filename
    pdf.output(str(pdf_path))
    return pdf_path


def test_ingest_pdf_persists_document_and_chunks_and_embeds(monkeypatch, tmp_path):
    embed_calls = []

    def fake_embed_texts(texts):
        embed_calls.append(texts)
        return [[0.1] * 8 for _ in texts]

    monkeypatch.setattr(pipeline, "embed_texts", fake_embed_texts)

    pdf_path = _make_sample_pdf(tmp_path)
    document = pipeline.ingest_pdf(pdf_path, write_json=False)

    assert document.company == "Apple"
    assert document.fiscal_year == "FY24"
    assert document.doc_type == "10-K"
    assert document.status == "ready"
    assert document.page_count == 1

    session = get_session()
    chunks = session.query(Chunk).filter(Chunk.document_id == document.id).all()
    session.close()

    assert len(chunks) >= 1
    assert all(c.embedding_id is not None for c in chunks)
    # embed_texts was called once with all chunk texts batched, not once per chunk.
    assert len(embed_calls) == 1
    assert len(embed_calls[0]) == len(chunks)


def test_ingest_pdf_does_not_duplicate_chunk_text_into_chroma(monkeypatch, tmp_path):
    """SQLite is the sole source of truth for chunk text (see
    sage/retrieval/store.py's module docstring) -- Chroma should only ever
    get ids, embeddings, and filter metadata from ingest."""
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.1] * 8 for _ in texts])

    add_calls = []
    real_add = pipeline.store.add

    def spying_add(*args, **kwargs):
        add_calls.append(kwargs)
        return real_add(*args, **kwargs)

    monkeypatch.setattr(pipeline.store, "add", spying_add)

    pdf_path = _make_sample_pdf(tmp_path)
    pipeline.ingest_pdf(pdf_path, write_json=False)

    assert len(add_calls) == 1
    assert "documents" not in add_calls[0] or add_calls[0]["documents"] is None


def test_ingest_pdf_applies_metadata_overrides(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.1] * 8 for _ in texts])

    pdf_path = _make_sample_pdf(tmp_path, filename="Unlabeled.pdf")
    document = pipeline.ingest_pdf(
        pdf_path,
        write_json=False,
        metadata_overrides={"company": "Apple Inc.", "fiscal_year": "FY24"},
    )

    assert document.company == "Apple Inc."
    assert document.fiscal_year == "FY24"


def test_ingest_folder_ingests_every_pdf_in_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.1] * 8 for _ in texts])

    _make_sample_pdf(tmp_path, filename="Apple_FY24_10-K.pdf")
    _make_sample_pdf(tmp_path, filename="Microsoft_FY24_10-K.pdf")

    documents = pipeline.ingest_folder(tmp_path)

    assert {d.company for d in documents} == {"Apple", "Microsoft"}
    session = get_session()
    assert session.query(Document).count() == 2
    session.close()


def test_ingest_pdf_rolls_back_on_embedding_failure(monkeypatch, tmp_path):
    def failing_embed_texts(texts):
        raise RuntimeError("simulated embedding API failure")

    monkeypatch.setattr(pipeline, "embed_texts", failing_embed_texts)

    pdf_path = _make_sample_pdf(tmp_path)
    try:
        pipeline.ingest_pdf(pdf_path, write_json=False)
        raised = False
    except RuntimeError:
        raised = True
    assert raised

    session = get_session()
    assert session.query(Document).count() == 0
    session.close()
