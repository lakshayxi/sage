"""Exact-match query cache: hash(query + metadata filters + model) -> cached
answer, plus a semantic (embedding-similarity) fallback cache checked on an
exact-match miss.

A cache hit avoids a live Gemini *chat* call entirely (the semantic-cache
lookup itself still costs one local embedding call, which is free -- see
sage/embed/local_embedder.py) -- this matters more for Sage than for the
reference (free, local Ollama) project, since it directly protects the free
Gemini chat quota rather than just latency.

`companies` is a `list[str] | None` here too (not singular `company`),
matching `retrieve_hybrid`'s signature: the same filters that scope
retrieval also scope the cache key, and a comparison query's cache entry
must not collide with (or be satisfied by) a single-company query's entry
for the same text. The list is sorted before hashing/storing so cache keys
are independent of the order companies were requested in.

Both layers expire entries older than `settings.CACHE_TTL_SECONDS` (checked
lazily at read time) rather than running a background eviction job --
reasonable at this project's expected query volume. Expired rows are left in
place rather than deleted, same tradeoff as the reference project made.
"""

import hashlib
import json
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError

from config import settings
from sage.db.database import get_session
from sage.db.models import QueryCache
from sage.embed.local_embedder import embed_query
from sage.retrieval import store

# Sentinel for an unset filter field, stored/matched explicitly rather than
# omitting the clause -- an omitted clause would make an unscoped query
# match ANY value of that field, including a specific company's cache entry.
_UNSET = "__unset__"


def _normalize_companies_for_key(companies: list[str] | None) -> list[str]:
    if not companies:
        return []
    return sorted({c for c in companies if c})


def make_cache_key(
    query: str,
    companies: list[str] | None,
    fiscal_year: str | None,
    doc_type: str | None,
    model: str,
) -> str:
    """Hash (query + metadata filters + model) into a cache key.

    `top_k` is deliberately not part of the key -- two requests differing
    only in top_k share a cache entry, matching the reference project's
    documented tradeoff.
    """
    payload = json.dumps(
        {
            "query": query.strip(),
            "companies": _normalize_companies_for_key(companies),
            "fiscal_year": fiscal_year,
            "doc_type": doc_type,
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_expired(created_at: datetime) -> bool:
    # SQLite drops tzinfo on write; a row read back has a naive `created_at`
    # that nonetheless represents UTC (per models.py's `_utcnow`).
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return datetime.now(UTC) - created_at > timedelta(seconds=settings.CACHE_TTL_SECONDS)


def get_cached(cache_key: str) -> QueryCache | None:
    session = get_session()
    try:
        row = session.query(QueryCache).filter(QueryCache.cache_key == cache_key).first()
        if row is not None and _is_expired(row.created_at):
            return None
        return row
    finally:
        session.close()


def _semantic_metadata(
    companies: list[str] | None, fiscal_year: str | None, doc_type: str | None, model: str
) -> dict:
    normalized = _normalize_companies_for_key(companies)
    return {
        "model": model,
        "companies": json.dumps(normalized) if normalized else _UNSET,
        "fiscal_year": fiscal_year or _UNSET,
        "doc_type": doc_type or _UNSET,
    }


def _semantic_where(
    companies: list[str] | None, fiscal_year: str | None, doc_type: str | None, model: str
) -> dict:
    normalized = _normalize_companies_for_key(companies)
    return {
        "$and": [
            {"model": model},
            {"companies": json.dumps(normalized) if normalized else _UNSET},
            {"fiscal_year": fiscal_year or _UNSET},
            {"doc_type": doc_type or _UNSET},
        ]
    }


def get_semantic_cached(
    query_text: str,
    companies: list[str] | None,
    fiscal_year: str | None,
    doc_type: str | None,
    model: str,
    query_embedding: list[float] | None = None,
) -> QueryCache | None:
    """Embedding-similarity fallback, checked on an exact-match miss.

    Finds the nearest previously-cached query, restricted to the same
    companies/fiscal_year/doc_type/model as the exact-match key, and returns
    its QueryCache row if within SEMANTIC_CACHE_THRESHOLD squared-L2
    distance. Delegates to `get_cached` for the final lookup, so an expired
    underlying row is treated as a miss here too.

    `query_embedding`, if given, is reused instead of re-embedding
    `query_text` -- the caller (answer_engine.generate_answer) computes this
    once per request and reuses it here and for retrieval, rather than
    embedding the identical query text twice.
    """
    embedding = query_embedding if query_embedding is not None else embed_query(query_text)
    where = _semantic_where(companies, fiscal_year, doc_type, model)
    result = store.query(
        embedding, top_k=1, where=where, collection_name=settings.CHROMA_QUERY_CACHE_COLLECTION
    )
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not ids or distances[0] > settings.SEMANTIC_CACHE_THRESHOLD:
        return None
    return get_cached(ids[0])


def store_cached(
    cache_key: str,
    query_text: str,
    model: str,
    answer_text: str,
    citations: list[dict],
    retrieved_chunk_ids: list[int],
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    companies: list[str] | None = None,
    fiscal_year: str | None = None,
    doc_type: str | None = None,
    query_embedding: list[float] | None = None,
) -> None:
    """Insert a fresh cache entry, or refresh an existing *expired* one in
    place -- a still-fresh row (written by a concurrent request that beat
    this one to it) is left alone.

    Without the expired-row refresh path, `get_cached()` correctly treats an
    expired row as a miss at read time, but this function's old `if existing
    is not None: return` treated the row's mere presence (expired or not) as
    "already cached" and silently discarded every regenerated answer for
    that key forever after -- the cache could never recover from staleness.

    The semantic Chroma cache is kept in step via `store.upsert` (not
    `store.add`): a plain add() for an id that already exists (the expired
    row's old semantic vector) would either raise or leave a stale vector
    still pointing at the citations/answer this call is about to replace.
    """
    session = get_session()
    should_refresh_semantic = False
    try:
        existing = session.query(QueryCache).filter(QueryCache.cache_key == cache_key).first()
        if existing is not None and not _is_expired(existing.created_at):
            return

        if existing is not None:
            existing.query_text = query_text
            existing.model_name = model
            existing.answer_text = answer_text
            existing.citations_json = citations
            existing.retrieved_chunk_ids = retrieved_chunk_ids
            existing.prompt_tokens = prompt_tokens
            existing.completion_tokens = completion_tokens
            existing.total_tokens = total_tokens
            existing.created_at = datetime.now(UTC)
        else:
            session.add(
                QueryCache(
                    cache_key=cache_key,
                    query_text=query_text,
                    model_name=model,
                    answer_text=answer_text,
                    citations_json=citations,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            )
        try:
            session.commit()
            should_refresh_semantic = True
        except IntegrityError:
            # Another concurrent request won the race to insert this same
            # new key first; nothing more to do.
            session.rollback()
    finally:
        session.close()

    if not should_refresh_semantic:
        return

    embedding = query_embedding if query_embedding is not None else embed_query(query_text)
    store.upsert(
        ids=[cache_key],
        embeddings=[embedding],
        documents=[query_text],
        metadatas=[_semantic_metadata(companies, fiscal_year, doc_type, model)],
        collection_name=settings.CHROMA_QUERY_CACHE_COLLECTION,
    )
