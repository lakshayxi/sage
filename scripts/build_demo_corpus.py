"""Build a privacy-safe, internally consistent corpus for the public demo.

The development stores contain query logs, answer caches, conversations, and
session bearer tokens alongside the filing corpus. Copying them wholesale into
the public image would publish that private runtime state. This exporter instead
backs up SQLite, removes every non-corpus row, and recreates Chroma with only the
chunk collection before atomically replacing the deployment artifacts.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
from pathlib import Path

import chromadb

from config import settings

PRIVATE_TABLES = ("messages", "conversations", "query_logs", "query_cache")
BATCH_SIZE = 500


class CorpusValidationError(RuntimeError):
    """Raised when a demo artifact is empty, inconsistent, or contains private state."""


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _sqlite_counts(database_path: Path) -> tuple[int, int]:
    if not database_path.is_file() or database_path.stat().st_size == 0:
        raise CorpusValidationError(f"SQLite corpus is missing or empty: {database_path}")

    connection = sqlite3.connect(database_path)
    try:
        tables = _table_names(connection)
        required = {"documents", "chunks", *PRIVATE_TABLES}
        missing = required - tables
        if missing:
            raise CorpusValidationError(
                f"SQLite corpus is missing required tables: {', '.join(sorted(missing))}"
            )

        private_counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in PRIVATE_TABLES
        }
        populated = {table: count for table, count in private_counts.items() if count}
        if populated:
            detail = ", ".join(f"{table}={count}" for table, count in populated.items())
            raise CorpusValidationError(f"SQLite corpus contains private runtime state: {detail}")

        document_count = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'ready'"
        ).fetchone()[0]
        total_documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        missing_embedding_ids = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE embedding_id IS NULL OR embedding_id = ''"
        ).fetchone()[0]
        absolute_source_paths = connection.execute(
            "SELECT COUNT(*) FROM documents WHERE source_path LIKE '/%'"
        ).fetchone()[0]
    finally:
        connection.close()

    if document_count == 0 or chunk_count == 0:
        raise CorpusValidationError("Demo corpus must contain ready documents and chunks")
    if document_count != total_documents:
        raise CorpusValidationError("Every demo document must have status='ready'")
    if missing_embedding_ids:
        raise CorpusValidationError(f"{missing_embedding_ids} chunks have no embedding id")
    if absolute_source_paths:
        raise CorpusValidationError("Demo document source paths must be repository-relative")
    return document_count, chunk_count


def validate_demo_corpus(database_path: Path, chroma_path: Path) -> tuple[int, int]:
    """Prove that an artifact contains only a matched, non-empty filing corpus."""
    document_count, chunk_count = _sqlite_counts(database_path)
    if not chroma_path.is_dir():
        raise CorpusValidationError(f"Chroma corpus directory is missing: {chroma_path}")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection_names = {collection.name for collection in client.list_collections()}
    if settings.CHROMA_QUERY_CACHE_COLLECTION in collection_names:
        raise CorpusValidationError("Demo Chroma store contains the semantic query cache")
    if collection_names != {settings.CHROMA_COLLECTION}:
        raise CorpusValidationError(
            "Demo Chroma store must contain only "
            f"{settings.CHROMA_COLLECTION!r}; found {sorted(collection_names)}"
        )

    collection = client.get_collection(settings.CHROMA_COLLECTION)
    if collection.count() != chunk_count:
        raise CorpusValidationError(
            f"SQLite/Chroma chunk mismatch: sqlite={chunk_count}, chroma={collection.count()}"
        )

    connection = sqlite3.connect(database_path)
    try:
        sqlite_ids = {
            str(row[0]) for row in connection.execute("SELECT embedding_id FROM chunks").fetchall()
        }
    finally:
        connection.close()
    chroma_ids = set(collection.get(include=[])["ids"])
    if sqlite_ids != chroma_ids:
        raise CorpusValidationError("SQLite embedding ids do not match Chroma vector ids")

    return document_count, chunk_count


def _export_sqlite(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.execute("PRAGMA foreign_keys = ON")
        for table in PRIVATE_TABLES:
            destination_connection.execute(f"DELETE FROM {table}")
        if "sqlite_sequence" in _table_names(destination_connection):
            placeholders = ",".join("?" for _ in PRIVATE_TABLES)
            destination_connection.execute(
                f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})", PRIVATE_TABLES
            )
        destination_connection.commit()
        destination_connection.execute("VACUUM")
    finally:
        destination_connection.close()
        source_connection.close()


def _export_chroma(source: Path, destination: Path) -> None:
    source_client = chromadb.PersistentClient(path=str(source))
    source_collection = source_client.get_collection(settings.CHROMA_COLLECTION)
    destination_client = chromadb.PersistentClient(path=str(destination))
    destination_collection = destination_client.create_collection(
        settings.CHROMA_COLLECTION, metadata=source_collection.metadata
    )

    for offset in range(0, source_collection.count(), BATCH_SIZE):
        batch = source_collection.get(
            limit=BATCH_SIZE,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )
        kwargs: dict = {
            "ids": batch["ids"],
            "embeddings": batch["embeddings"],
            "metadatas": batch["metadatas"],
        }
        documents = batch.get("documents")
        if documents and any(document is not None for document in documents):
            kwargs["documents"] = documents
        destination_collection.add(**kwargs)


def build_demo_corpus(source_db: Path, source_chroma: Path, output_dir: Path) -> tuple[int, int]:
    """Create and atomically install a scrubbed deployment corpus."""
    if not source_db.is_file():
        raise FileNotFoundError(f"source SQLite database not found: {source_db}")
    if not source_chroma.is_dir():
        raise FileNotFoundError(f"source Chroma directory not found: {source_chroma}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sage-demo-corpus-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        staged_db = staging / "sage.db"
        staged_chroma = staging / "chroma"
        _export_sqlite(source_db, staged_db)
        _export_chroma(source_chroma, staged_chroma)
        counts = validate_demo_corpus(staged_db, staged_chroma)

        destination_db = output_dir / "sage.db"
        destination_chroma = output_dir / "chroma"
        previous_db = staging / "previous-sage.db"
        previous_chroma = staging / "previous-chroma"

        if destination_db.exists():
            destination_db.replace(previous_db)
        if destination_chroma.exists():
            destination_chroma.replace(previous_chroma)
        try:
            staged_db.replace(destination_db)
            staged_chroma.replace(destination_chroma)
        except Exception:
            if destination_db.exists():
                destination_db.unlink()
            if destination_chroma.exists():
                shutil.rmtree(destination_chroma)
            if previous_db.exists():
                previous_db.replace(destination_db)
            if previous_chroma.exists():
                previous_chroma.replace(destination_chroma)
            raise

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=settings.SQLITE_PATH)
    parser.add_argument("--source-chroma", type=Path, default=settings.CHROMA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.BASE_DIR / "deploy" / "huggingface" / "prebuilt",
    )
    args = parser.parse_args()

    documents, chunks = build_demo_corpus(
        args.source_db.resolve(), args.source_chroma.resolve(), args.output_dir.resolve()
    )
    print(f"Built clean demo corpus: {documents} documents, {chunks} chunks")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
