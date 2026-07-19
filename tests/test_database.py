"""Tests for sage/db/database.py -- specifically the checksum-column
auto-migration (`_ensure_document_checksum_column`), since it's the one
piece of hand-rolled DDL in a project with no migration tool (Alembic)."""

import sqlite3

from sqlalchemy.exc import IntegrityError

from config import settings
from sage.db import database
from sage.db.database import get_session, init_db
from sage.db.models import Document


def _create_legacy_documents_table(db_path) -> None:
    """A `documents` table shaped like it was before the `checksum` column
    existed -- what init_db() must migrate forward from."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            filename VARCHAR NOT NULL,
            title VARCHAR,
            company VARCHAR,
            fiscal_year VARCHAR,
            doc_type VARCHAR,
            source_path VARCHAR NOT NULL,
            page_count INTEGER NOT NULL,
            embedding_model VARCHAR,
            ingested_at DATETIME,
            status VARCHAR
        )
        """
    )
    conn.commit()
    conn.close()


def test_migration_adds_checksum_column_to_a_preexisting_database(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(settings, "SQLITE_PATH", db_path)
    database.reset_engine()
    _create_legacy_documents_table(db_path)

    columns_before = {
        row[1] for row in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(documents)")
    }
    assert "checksum" not in columns_before

    init_db()

    columns_after = {
        row[1] for row in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(documents)")
    }
    assert "checksum" in columns_after


def test_migration_preserves_existing_rows_with_null_checksum(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(settings, "SQLITE_PATH", db_path)
    database.reset_engine()
    _create_legacy_documents_table(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO documents (filename, source_path, page_count, status) "
        "VALUES ('Apple_FY24_10-K.pdf', '/data/raw/Apple_FY24_10-K.pdf', 42, 'ready')"
    )
    conn.commit()
    conn.close()

    init_db()

    row = (
        sqlite3.connect(str(db_path))
        .execute("SELECT filename, status, checksum FROM documents")
        .fetchone()
    )
    assert row == ("Apple_FY24_10-K.pdf", "ready", None)


def test_migration_is_idempotent_across_repeated_init_db_calls(monkeypatch, tmp_path):
    db_path = tmp_path / "legacy.db"
    monkeypatch.setattr(settings, "SQLITE_PATH", db_path)
    database.reset_engine()
    _create_legacy_documents_table(db_path)

    init_db()
    init_db()
    init_db()

    columns = [
        row[1] for row in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(documents)")
    ]
    assert columns.count("checksum") == 1


def test_fresh_database_gets_checksum_column_via_create_all(monkeypatch, tmp_path):
    db_path = tmp_path / "fresh.db"
    monkeypatch.setattr(settings, "SQLITE_PATH", db_path)
    database.reset_engine()

    init_db()

    columns = {
        row[1] for row in sqlite3.connect(str(db_path)).execute("PRAGMA table_info(documents)")
    }
    assert "checksum" in columns


def test_multiple_null_checksums_do_not_violate_the_unique_constraint():
    """Legacy (pre-migration) rows all share checksum=NULL -- SQL NULLs are
    never equal to each other, so this must not collide."""
    session = get_session()
    session.add(Document(filename="a.pdf", source_path="/tmp/a.pdf", page_count=1, status="ready"))
    session.add(Document(filename="b.pdf", source_path="/tmp/b.pdf", page_count=1, status="ready"))
    session.commit()
    count = session.query(Document).count()
    session.close()
    assert count == 2


def test_duplicate_checksum_is_rejected_at_the_database_level():
    """Regression test for the concurrent-double-ingest race: two Document
    rows with the same real (non-NULL) checksum must never both commit --
    this is the safety net for two truly concurrent uploads of the same new
    file both passing ingest_pdf's own check-then-insert dedup before
    either has committed."""
    session = get_session()
    session.add(
        Document(
            filename="a.pdf",
            source_path="/tmp/a.pdf",
            page_count=1,
            status="ready",
            checksum="duplicate-hash",
        )
    )
    session.commit()
    session.close()

    session2 = get_session()
    session2.add(
        Document(
            filename="b.pdf",
            source_path="/tmp/b.pdf",
            page_count=1,
            status="ready",
            checksum="duplicate-hash",
        )
    )
    try:
        raised = False
        try:
            session2.commit()
        except IntegrityError:
            raised = True
            session2.rollback()
        assert raised
    finally:
        session2.close()
