"""SQLAlchemy engine/session setup (SQLite), source of truth for chunk
text/metadata, cache rows, query logs, and conversation history. Chroma only
holds vectors plus filter metadata (see sage/retrieval/store.py).

The engine is a lazily-created, process-wide singleton built from
`settings.SQLITE_PATH` on first use rather than at import time -- this keeps
importing `sage.db.database` side-effect-free (no file created just by
importing the module) and, more importantly, gives tests a way to point at
an isolated tmp_path database: monkeypatch `settings.SQLITE_PATH` *and* call
`reset_engine()` before the first query in that test. Patching only the
setting without resetting the cached engine would silently keep using
whatever engine was already built (an already-open connection to the real
db path) -- the same class of bug documented in the reference project's
CLAUDE.md for its analogous Chroma client singleton.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config import settings

_engine = None
_SessionLocal: sessionmaker | None = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings.SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{settings.SQLITE_PATH}")
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def get_session() -> Session:
    _get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def init_db() -> None:
    from sage.db.models import Base

    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_document_checksum_column(engine)


def _ensure_document_checksum_column(engine) -> None:
    """Lightweight forward-compatible migration for the `documents.checksum`
    column added for ingest dedup (sage/ingest/pipeline.py).

    `Base.metadata.create_all()` only creates missing *tables* -- it never
    alters an existing one, so a `db/sage.db` created before this column
    existed would otherwise be permanently stuck without it. This project
    has no migration tool (Alembic) and doesn't need one yet for a single
    nullable+unique column on a single table; a plain `ALTER TABLE` plus a
    separate `CREATE UNIQUE INDEX` covers it -- SQLite's `ALTER TABLE ADD
    COLUMN` can't itself add a uniqueness constraint the way the ORM model's
    `Column(..., unique=True)` does for a table `create_all()` creates fresh,
    so that has to be a second explicit step here for a pre-existing table.
    Rows that already existed when this runs simply get `checksum = NULL`
    (no backfill -- the original file bytes aren't necessarily still
    available), which is safe: a NULL checksum never dedup-matches anything
    (SQL NULLs are never equal to each other either, so the unique index
    doesn't conflict across the potentially-many pre-existing NULL rows) --
    old documents just don't participate in dedup until re-ingested.
    """
    with engine.connect() as conn:
        columns = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(documents)")}
        if "checksum" not in columns:
            # A pre-existing table predates the column entirely, so it can't
            # already have a unique index on it either -- create_all() (which
            # already ran, above) only ever creates *missing* tables, so it
            # never touched this one. A freshly-created table already got
            # both the column and its unique index from the ORM model
            # definition directly, so this branch (and its own explicitly
            # named index, distinct from SQLAlchemy's auto-generated one) is
            # only ever reached for a genuinely pre-existing database.
            conn.exec_driver_sql("ALTER TABLE documents ADD COLUMN checksum VARCHAR")
            conn.exec_driver_sql(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_documents_checksum_unique "
                "ON documents(checksum)"
            )
            conn.commit()


def reset_engine() -> None:
    """Test-only hook: drop the cached engine/sessionmaker so the next call
    picks up a freshly monkeypatched `settings.SQLITE_PATH`."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
