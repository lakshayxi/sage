"""Thin Chroma PersistentClient wrapper — the vector-store swap seam.

`_client` is a process-wide singleton built from `settings.CHROMA_DIR` on
first access. Tests that need an isolated Chroma dir must patch *both*
`settings.CHROMA_DIR` and reset this module's `_client` to `None` -- patching
only the setting leaves the already-open client pointed at whatever dir it
was first built with. (See sage/db/database.py's `reset_engine` docstring
for the SQLite analogue of this same gotcha.)
"""

import chromadb

from config import settings

_client = None


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))
    return _client


def get_collection(name: str = settings.CHROMA_COLLECTION):
    return get_client().get_or_create_collection(name=name)


def add(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str] | None = None,
    collection_name: str = settings.CHROMA_COLLECTION,
) -> None:
    # `documents` is optional: the main chunk collection deliberately omits
    # it (SQLite is the sole source of truth for chunk text -- see
    # sage/ingest/pipeline.py), but the semantic query-cache collection
    # (sage/generation/cache.py) legitimately stores query text here since
    # it has no SQLite-backed equivalent to look it up from.
    get_collection(collection_name).add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def query(
    embedding: list[float],
    top_k: int,
    where: dict | None = None,
    collection_name: str = settings.CHROMA_COLLECTION,
) -> dict:
    return get_collection(collection_name).query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where or None,
    )


def delete(ids: list[str], collection_name: str = settings.CHROMA_COLLECTION) -> None:
    get_collection(collection_name).delete(ids=ids)
