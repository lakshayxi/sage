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

    Base.metadata.create_all(bind=_get_engine())


def reset_engine() -> None:
    """Test-only hook: drop the cached engine/sessionmaker so the next call
    picks up a freshly monkeypatched `settings.SQLITE_PATH`."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None
