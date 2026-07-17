"""Shared test fixtures.

Every test gets an isolated SQLite db, Chroma dir, and processed/raw data dir
under pytest's tmp_path -- never the real `data/` or `db/sage.db` a live run
would use. Both process-wide singletons (sage.db.database's cached engine,
sage.retrieval.store's cached Chroma client) are reset after monkeypatching
the underlying path, per the gotcha documented in both modules' docstrings:
patching the setting alone would leave an already-open connection/client
pointed at the old path. PROCESSED_DIR/RAW_DIR are patched too since
sage.ingest.pipeline.ingest_pdf writes a debug JSON dump there by default
(`write_json=True`) -- without this, running the ingest test suite would
leave real files behind in the repo's own `data/processed/`.
"""

import pytest

from config import settings
from sage.db import database
from sage.db.database import init_db
from sage.retrieval import store


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SQLITE_PATH", tmp_path / "sage.db")
    monkeypatch.setattr(settings, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(settings, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(settings, "PROCESSED_DIR", tmp_path / "processed")
    database.reset_engine()
    monkeypatch.setattr(store, "_client", None)
    init_db()
    yield
    database.reset_engine()
    monkeypatch.setattr(store, "_client", None)
