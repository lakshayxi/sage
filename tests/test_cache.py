from datetime import UTC, datetime, timedelta

from config import settings
from sage.db.database import get_session
from sage.db.models import QueryCache
from sage.generation import cache


def _fake_embedder(vectors: dict[str, list[float]]):
    def embed_query(text: str) -> list[float]:
        return vectors[text]

    return embed_query


def test_make_cache_key_is_deterministic():
    key1 = cache.make_cache_key(
        "What are Apple margins?", ["Apple"], "FY24", None, "gemini-2.5-flash"
    )
    key2 = cache.make_cache_key(
        "What are Apple margins?", ["Apple"], "FY24", None, "gemini-2.5-flash"
    )
    assert key1 == key2


def test_make_cache_key_is_order_independent_for_companies():
    key1 = cache.make_cache_key(
        "compare capex", ["Apple", "Microsoft"], None, None, "gemini-2.5-flash"
    )
    key2 = cache.make_cache_key(
        "compare capex", ["Microsoft", "Apple"], None, None, "gemini-2.5-flash"
    )
    assert key1 == key2


def test_make_cache_key_is_sensitive_to_filters_and_model():
    base = cache.make_cache_key(
        "What are Apple margins?", ["Apple"], "FY24", None, "gemini-2.5-flash"
    )

    other_companies = cache.make_cache_key(
        "What are Apple margins?", ["Apple", "Microsoft"], "FY24", None, "gemini-2.5-flash"
    )
    other_year = cache.make_cache_key(
        "What are Apple margins?", ["Apple"], "FY25", None, "gemini-2.5-flash"
    )
    other_model = cache.make_cache_key(
        "What are Apple margins?", ["Apple"], "FY24", None, "gemini-2.5-flash-lite"
    )
    no_companies = cache.make_cache_key(
        "What are Apple margins?", None, "FY24", None, "gemini-2.5-flash"
    )

    assert base != other_companies
    assert base != other_year
    assert base != other_model
    assert base != no_companies


def test_make_cache_key_is_sensitive_to_top_k():
    key_five = cache.make_cache_key(
        "compare capex", ["Apple", "Microsoft"], None, None, "gemini-2.5-flash", top_k=5
    )
    key_three = cache.make_cache_key(
        "compare capex", ["Apple", "Microsoft"], None, None, "gemini-2.5-flash", top_k=3
    )

    assert key_five != key_three


def test_get_cached_returns_none_for_unknown_key():
    assert cache.get_cached("no-such-key") is None


def test_store_and_get_cached_round_trip(monkeypatch):
    query_text = "round trip query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    citation = {
        "n": 1,
        "chunk_id": 42,
        "text": "chunk text",
        "page_number": 1,
        "company": "Apple",
        "fiscal_year": "FY24",
        "doc_type": "10-K",
        "filename": "f.pdf",
    }
    cache.store_cached(
        key, query_text, "gemini-2.5-flash", "The answer.", [citation], [42], 10, 5, 15
    )

    cached = cache.get_cached(key)
    assert cached is not None
    assert cached.answer_text == "The answer."
    assert cached.citations_json == [citation]
    assert cached.retrieved_chunk_ids == [42]
    assert cached.total_tokens == 15


def test_store_cached_does_not_duplicate_an_existing_key(monkeypatch):
    query_text = "dedup query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    cache.store_cached(key, query_text, "gemini-2.5-flash", "first", [], [], 1, 1, 2)
    cache.store_cached(key, query_text, "gemini-2.5-flash", "second", [], [], 1, 1, 2)

    session = get_session()
    count = session.query(QueryCache).filter(QueryCache.cache_key == key).count()
    session.close()

    assert count == 1
    assert cache.get_cached(key).answer_text == "first"


def test_get_cached_treats_expired_row_as_miss():
    query_text = "expired query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    expired_at = datetime.now(UTC) - timedelta(seconds=settings.CACHE_TTL_SECONDS + 1)

    session = get_session()
    session.add(
        QueryCache(
            cache_key=key,
            query_text=query_text,
            model_name="gemini-2.5-flash",
            answer_text="stale answer",
            citations_json=[],
            retrieved_chunk_ids=[],
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            created_at=expired_at,
        )
    )
    session.commit()
    session.close()

    assert cache.get_cached(key) is None


def test_get_cached_returns_row_within_ttl():
    query_text = "within ttl query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    fresh_at = datetime.now(UTC) - timedelta(seconds=settings.CACHE_TTL_SECONDS - 1)

    session = get_session()
    session.add(
        QueryCache(
            cache_key=key,
            query_text=query_text,
            model_name="gemini-2.5-flash",
            answer_text="still good",
            citations_json=[],
            retrieved_chunk_ids=[],
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            created_at=fresh_at,
        )
    )
    session.commit()
    session.close()

    cached = cache.get_cached(key)
    assert cached is not None
    assert cached.answer_text == "still good"


def test_store_cached_survives_concurrent_insert_race():
    query_text = "race query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")

    session = get_session()
    session.add(
        QueryCache(
            cache_key=key,
            query_text=query_text,
            model_name="gemini-2.5-flash",
            answer_text="winner",
            citations_json=[],
            retrieved_chunk_ids=[],
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )
    )
    session.commit()
    session.close()

    cache.store_cached(key, query_text, "gemini-2.5-flash", "loser", [], [], 1, 1, 2)

    session = get_session()
    count = session.query(QueryCache).filter(QueryCache.cache_key == key).count()
    session.close()

    assert count == 1
    assert cache.get_cached(key).answer_text == "winner"


def test_semantic_cache_hits_on_near_duplicate_query(monkeypatch):
    base_query = "semantic base query"
    near_query = "semantic near-duplicate query"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({base_query: [1.0, 0.0], near_query: [0.99, 0.01]})
    )

    key = cache.make_cache_key(base_query, None, None, None, "gemini-2.5-flash")
    cache.store_cached(
        key, base_query, "gemini-2.5-flash", "The margins declined.", [], [], 1, 1, 2
    )

    hit = cache.get_semantic_cached(near_query, None, None, None, "gemini-2.5-flash")

    assert hit is not None
    assert hit.answer_text == "The margins declined."


def test_semantic_cache_misses_below_threshold(monkeypatch):
    base_query = "semantic unrelated base"
    far_query = "semantic unrelated far"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({base_query: [1.0, 0.0], far_query: [0.0, 1.0]})
    )

    key = cache.make_cache_key(base_query, None, None, None, "gemini-2.5-flash")
    cache.store_cached(
        key, base_query, "gemini-2.5-flash", "The margins declined.", [], [], 1, 1, 2
    )

    miss = cache.get_semantic_cached(far_query, None, None, None, "gemini-2.5-flash")

    assert miss is None


def test_semantic_cache_respects_companies_filter(monkeypatch):
    base_query = "semantic apple query"
    near_query = "semantic microsoft query"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({base_query: [1.0, 0.0], near_query: [0.99, 0.01]})
    )

    key = cache.make_cache_key(base_query, ["Apple"], None, None, "gemini-2.5-flash")
    cache.store_cached(
        key,
        base_query,
        "gemini-2.5-flash",
        "Apple margins declined.",
        [],
        [],
        1,
        1,
        2,
        companies=["Apple"],
    )

    miss = cache.get_semantic_cached(near_query, ["Microsoft"], None, None, "gemini-2.5-flash")

    assert miss is None


def test_semantic_cache_respects_comparison_vs_single_company(monkeypatch):
    """A comparison query's cache entry (companies=["Apple","Microsoft"]) must
    not be reused by a single-company query for either company alone -- a
    Sage-specific case the reference project (singular `company`) can't have."""
    base_query = "semantic comparison base"
    near_query = "semantic comparison near"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({base_query: [1.0, 0.0], near_query: [0.99, 0.01]})
    )

    key = cache.make_cache_key(base_query, ["Apple", "Microsoft"], None, None, "gemini-2.5-flash")
    cache.store_cached(
        key,
        base_query,
        "gemini-2.5-flash",
        "Comparison answer.",
        [],
        [],
        1,
        1,
        2,
        companies=["Apple", "Microsoft"],
    )

    miss = cache.get_semantic_cached(near_query, ["Apple"], None, None, "gemini-2.5-flash")

    assert miss is None


def test_store_cached_replaces_expired_row_with_fresh_value(monkeypatch):
    """Regression test: store -> expire -> miss -> regenerate -> store
    replacement -> hit fresh value. The old `if existing is not None:
    return` treated an expired-but-present row as already cached, so a
    regenerated answer could never overwrite it -- get_cached() would keep
    returning None (miss) forever, even after a successful regeneration."""
    query_text = "expiring then regenerated query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    cache.store_cached(key, query_text, "gemini-2.5-flash", "stale answer", [], [], 1, 1, 2)
    assert cache.get_cached(key).answer_text == "stale answer"

    expired_at = datetime.now(UTC) - timedelta(seconds=settings.CACHE_TTL_SECONDS + 1)
    session = get_session()
    session.query(QueryCache).filter(QueryCache.cache_key == key).update({"created_at": expired_at})
    session.commit()
    session.close()

    assert cache.get_cached(key) is None  # expired -> miss

    cache.store_cached(
        key, query_text, "gemini-2.5-flash", "fresh regenerated answer", [], [], 3, 3, 6
    )

    fresh = cache.get_cached(key)
    assert fresh is not None
    assert fresh.answer_text == "fresh regenerated answer"
    assert fresh.total_tokens == 6

    # Exactly one row for this key -- the expired row was refreshed in
    # place, not left behind alongside a second inserted row.
    session = get_session()
    count = session.query(QueryCache).filter(QueryCache.cache_key == key).count()
    session.close()
    assert count == 1


def test_store_cached_does_not_replace_a_still_fresh_row(monkeypatch):
    """The expired-row refresh path must not also start clobbering rows that
    are still within TTL -- only an actually-expired row is eligible."""
    query_text = "still fresh query"
    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    cache.store_cached(key, query_text, "gemini-2.5-flash", "first answer", [], [], 1, 1, 2)
    cache.store_cached(key, query_text, "gemini-2.5-flash", "second answer", [], [], 1, 1, 2)

    assert cache.get_cached(key).answer_text == "first answer"


def test_semantic_cache_reflects_refreshed_answer_after_expired_replacement(monkeypatch):
    """The semantic Chroma cache must not keep pointing at the pre-refresh
    answer after an expired row is replaced -- store_cached upserts the
    semantic vector rather than leaving the old one in place."""
    query_text = "semantic refresh base"
    near_query = "semantic refresh near"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({query_text: [1.0, 0.0], near_query: [0.99, 0.01]})
    )

    key = cache.make_cache_key(query_text, None, None, None, "gemini-2.5-flash")
    cache.store_cached(key, query_text, "gemini-2.5-flash", "stale answer", [], [], 1, 1, 2)

    expired_at = datetime.now(UTC) - timedelta(seconds=settings.CACHE_TTL_SECONDS + 1)
    session = get_session()
    session.query(QueryCache).filter(QueryCache.cache_key == key).update({"created_at": expired_at})
    session.commit()
    session.close()

    cache.store_cached(key, query_text, "gemini-2.5-flash", "fresh answer", [], [], 1, 1, 2)

    hit = cache.get_semantic_cached(near_query, None, None, None, "gemini-2.5-flash")
    assert hit is not None
    assert hit.answer_text == "fresh answer"


def test_semantic_cache_treats_expired_underlying_row_as_miss(monkeypatch):
    base_query = "semantic expired base"
    near_query = "semantic expired near"
    monkeypatch.setattr(
        cache, "embed_query", _fake_embedder({base_query: [1.0, 0.0], near_query: [0.99, 0.01]})
    )

    key = cache.make_cache_key(base_query, None, None, None, "gemini-2.5-flash")
    cache.store_cached(
        key, base_query, "gemini-2.5-flash", "The margins declined.", [], [], 1, 1, 2
    )

    expired_at = datetime.now(UTC) - timedelta(seconds=settings.CACHE_TTL_SECONDS + 1)
    session = get_session()
    session.query(QueryCache).filter(QueryCache.cache_key == key).update({"created_at": expired_at})
    session.commit()
    session.close()

    miss = cache.get_semantic_cached(near_query, None, None, None, "gemini-2.5-flash")

    assert miss is None
