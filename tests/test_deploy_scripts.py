import sqlite3
from pathlib import Path

import chromadb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from config import settings
from sage.db.models import Base, Chunk, Conversation, Document, Message, QueryCache, QueryLog
from scripts.build_demo_corpus import (
    CorpusValidationError,
    build_demo_corpus,
    validate_demo_corpus,
)


def _make_source_corpus(root: Path) -> tuple[Path, Path]:
    database_path = root / "source.db"
    engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        document = Document(
            filename="Apple_FY25_filing.pdf",
            title="Apple FY25 filing",
            company="Apple",
            fiscal_year="FY25",
            doc_type="10-K",
            source_path="data/raw/Apple_FY25_filing.pdf",
            page_count=1,
            status="ready",
        )
        session.add(document)
        session.flush()
        session.add(
            Chunk(
                document_id=document.id,
                chunk_index=0,
                page_number=1,
                text="Apple reported revenue.",
                embedding_id="chunk-1",
            )
        )
        conversation = Conversation(title="private question", session_token="private-token")
        session.add(conversation)
        session.flush()
        session.add(
            Message(
                conversation_id=conversation.id,
                role="user",
                content="private question",
                ordered=0,
            )
        )
        session.add(
            QueryLog(
                query_text="private question",
                answer_text="private answer",
            )
        )
        session.add(
            QueryCache(
                cache_key="private-cache-key",
                query_text="private question",
                answer_text="private answer",
                citations_json=[],
            )
        )
        session.commit()
    engine.dispose()

    chroma_path = root / "source-chroma"
    client = chromadb.PersistentClient(path=str(chroma_path))
    client.create_collection(settings.CHROMA_COLLECTION).add(
        ids=["chunk-1"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"document_id": 1, "company": "Apple"}],
    )
    client.create_collection(settings.CHROMA_QUERY_CACHE_COLLECTION).add(
        ids=["private-cache-key"],
        embeddings=[[0.2, 0.1]],
        metadatas=[{"cache_key": "private-cache-key"}],
        documents=["private question"],
    )
    return database_path, chroma_path


def test_build_demo_corpus_exports_only_matched_filing_data(tmp_path):
    source_db, source_chroma = _make_source_corpus(tmp_path)
    output = tmp_path / "prebuilt"

    assert build_demo_corpus(source_db, source_chroma, output) == (1, 1)
    assert validate_demo_corpus(output / "sage.db", output / "chroma") == (1, 1)

    connection = sqlite3.connect(output / "sage.db")
    try:
        for table in ("messages", "conversations", "query_logs", "query_cache"):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1
    finally:
        connection.close()

    collections = chromadb.PersistentClient(path=str(output / "chroma")).list_collections()
    assert [collection.name for collection in collections] == [settings.CHROMA_COLLECTION]


def test_validate_demo_corpus_rejects_private_runtime_state(tmp_path):
    source_db, source_chroma = _make_source_corpus(tmp_path)

    with pytest.raises(CorpusValidationError, match="private runtime state"):
        validate_demo_corpus(source_db, source_chroma)


def test_validate_demo_corpus_rejects_empty_placeholder(tmp_path):
    placeholder = tmp_path / "placeholder"
    placeholder.mkdir()
    empty_database = placeholder / "sage.db"
    empty_database.touch()

    with pytest.raises(CorpusValidationError, match="missing or empty"):
        validate_demo_corpus(empty_database, placeholder / "chroma")
