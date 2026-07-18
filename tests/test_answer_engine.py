from config import settings
from sage.db.conversations import HistoryTurn
from sage.db.database import get_session
from sage.db.models import QueryLog
from sage.generation import answer_engine, cache
from sage.generation.answer_engine import (
    NO_RELEVANT_CONTEXT_ANSWER,
    _resolve_citations,
    _split_answer_and_entries,
)
from sage.generation.gemini_client import ChatResult, StreamDone, StreamToken
from sage.retrieval.retriever import RetrievedChunk


def test_fenced_citations_are_parsed_and_stripped():
    raw = (
        "Margins declined due to component costs [1].\n"
        '```citations\n[{"n": 1, "chunk_id": 42}]\n```'
    )
    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text == "Margins declined due to component costs [1]."
    assert entries == [{"n": 1, "chunk_id": 42}]


def test_unfenced_citations_are_parsed_and_stripped():
    raw = 'Margins declined due to component costs [1].\ncitations\n[{"n": 1, "chunk_id": 42}]'
    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text == "Margins declined due to component costs [1]."
    assert entries == [{"n": 1, "chunk_id": 42}]


def test_truncated_fenced_citations_block_is_stripped_not_leaked():
    raw = 'Margins declined due to component costs [1].\n```citations\n[{"n": 1, "chunk_id'
    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text == "Margins declined due to component costs [1]."
    assert "```citations" not in clean_text
    assert entries == []


def test_bare_trailing_fence_is_stripped_not_leaked():
    raw = "Revenue grew 5% year-over-year [1].\n\n```"
    clean_text, entries = _split_answer_and_entries(raw)

    assert not clean_text.endswith("```")
    assert entries == []


def test_answer_with_no_citations_block_is_unaffected():
    raw = "Margins declined due to component costs."
    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text == "Margins declined due to component costs."
    assert entries == []


def test_resolve_citations_drops_entries_missing_a_valid_n():
    """Regression test: a malformed LLM citation entry with no (or a
    non-integer) "n" used to flow through into Citation(n=None, ...), which
    later crashed CitationOut(n: int) with an unhandled ValidationError."""
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2)]

    citations = _resolve_citations(
        [
            {"chunk_id": 1},  # missing "n" entirely
            {"n": "not-an-int", "chunk_id": 2},  # wrong type
        ],
        chunks,
    )

    assert citations == []


def test_resolve_citations_keeps_valid_entries_alongside_malformed_ones():
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2)]

    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 1}, {"chunk_id": 2}],
        chunks,
    )

    assert [c.n for c in citations] == [1]
    assert citations[0].chunk_id == 1


def _fake_chat_result(content: str, model: str = "gemini-test") -> ChatResult:
    return ChatResult(
        content=content, model=model, prompt_tokens=10, completion_tokens=5, total_tokens=15
    )


def _fake_chunk(chunk_id: int = 1, score: float = 0.99, company: str = "Apple") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=0,
        text=f"{company} margins declined due to component costs.",
        page_number=1,
        filename=f"{company}_FY24_10-K.pdf",
        company=company,
        fiscal_year="FY24",
        doc_type="10-K",
        score=score,
    )


class _FakeClient:
    model = "gemini-test"

    def __init__(self, content="An answer.", calls=None):
        self.content = content
        self.calls = calls if calls is not None else {"chat": 0, "chat_stream": 0}

    def chat(self, messages):
        self.calls["chat"] += 1
        self.last_messages = messages
        return _fake_chat_result(self.content, model=self.model)

    def chat_stream(self, messages):
        self.calls["chat_stream"] += 1
        self.last_messages = messages
        yield StreamToken(content=self.content)
        yield StreamDone(model=self.model, prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_generate_answer_runs_hybrid_retrieval_and_rerank_then_caches(monkeypatch):
    calls = {"retrieve_hybrid": 0, "rerank": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        calls["retrieve_hybrid"] += 1
        assert top_k == settings.RERANK_CANDIDATE_K
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        assert candidates == [_fake_chunk()]
        assert top_k == 5
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("The margins declined because of component costs.")
    result = answer_engine.generate_answer("first run query", top_k=5, client=client)

    assert calls == {"retrieve_hybrid": 1, "rerank": 1}
    assert client.calls["chat"] == 1
    assert result.cache_hit is False
    assert result.answer_text == "The margins declined because of component costs."


def test_generate_answer_second_call_hits_cache_and_skips_pipeline(monkeypatch):
    calls = {"retrieve_hybrid": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    query = "cache round trip query"
    client = _FakeClient("Cached-worthy answer.")

    first = answer_engine.generate_answer(query, client=client)
    assert first.cache_hit is False
    assert calls["retrieve_hybrid"] == 1
    assert client.calls["chat"] == 1

    second = answer_engine.generate_answer(query, client=client)
    assert second.cache_hit is True
    assert second.answer_text == first.answer_text
    # No new retrieval or generation call on a cache hit.
    assert calls["retrieve_hybrid"] == 1
    assert client.calls["chat"] == 1

    session = get_session()
    logs = session.query(QueryLog).filter(QueryLog.query_text == query).order_by(QueryLog.id).all()
    session.close()
    assert [log.cache_hit for log in logs] == [False, True]


def test_generate_answer_short_circuits_when_all_chunks_below_rerank_threshold(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer("irrelevant meta question", client=client)

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert result.citations == []
    assert result.retrieved_chunk_ids == []
    assert result.cost_usd == 0.0
    assert result.generation_latency_ms == 0.0


def test_generate_answer_filters_low_scoring_chunks_but_keeps_relevant_ones(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [
            _fake_chunk(chunk_id=1, score=0.95),
            _fake_chunk(chunk_id=2, score=settings.MIN_RERANK_SCORE - 0.01),
        ]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Margins declined [1].")
    result = answer_engine.generate_answer("partial relevance query", client=client)

    assert client.calls["chat"] == 1
    assert result.retrieved_chunk_ids == [1]


def test_generate_answer_retries_with_cleaned_query_when_first_attempt_is_empty(monkeypatch):
    """When the initial rerank gate comes back empty and the query has a
    trailing evaluative clause ("... were they good?"), the whole
    retrieve_hybrid -> rerank -> gate sequence is retried once with that
    clause stripped."""
    calls = {"retrieve_hybrid": 0, "rerank": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        if calls["rerank"] == 1:
            return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]
        return [_fake_chunk(chunk_id=99, score=0.9)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Apple's FY25 financials were strong [1].")
    result = answer_engine.generate_answer(
        "tell me about Apple's FY25 financials, were they good?", client=client
    )

    assert calls == {"retrieve_hybrid": 2, "rerank": 2}
    assert client.calls["chat"] == 1
    assert result.retrieved_chunk_ids == [99]
    assert result.answer_text != NO_RELEVANT_CONTEXT_ANSWER


def test_generate_answer_falls_back_when_retry_with_cleaned_query_is_also_empty(monkeypatch):
    """The retry happens exactly once -- if the cleaned query still clears
    nothing, there's no third attempt, just the existing fallback."""
    calls = {"retrieve_hybrid": 0, "rerank": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "tell me about Apple's FY25 filing's financials, were they good?", client=client
    )

    assert calls == {"retrieve_hybrid": 2, "rerank": 2}
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_generate_answer_no_retry_when_no_evaluative_clause_to_strip(monkeypatch):
    """A query with nothing to strip must not trigger a retry -- retrieval
    and rerank each run exactly once, same as before this fix."""
    calls = {"retrieve_hybrid": 0, "rerank": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "irrelevant meta question with no evaluative clause", client=client
    )

    assert calls == {"retrieve_hybrid": 1, "rerank": 1}
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_generate_answer_uses_comparison_prompt_when_chunks_span_multiple_companies(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
        ]

    def fake_rerank(query, candidates, top_k):
        # _retrieve_and_rerank reranks each company's own candidates
        # independently for a multi-company result set -- echo back
        # whatever single-company slice this call received.
        return candidates

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Apple grew capex. Microsoft grew capex more.")
    answer_engine.generate_answer(
        "compare capex query", companies=["Apple", "Microsoft"], client=client
    )

    from sage.generation.prompts import COMPARISON_SYSTEM_PROMPT

    assert client.last_messages[0]["content"] == COMPARISON_SYSTEM_PROMPT


def test_generate_answer_gives_each_company_its_own_rerank_budget(monkeypatch):
    """Regression test: a shared top_k reranked over the whole merged
    candidate pool let one company's chunks crowd out the others, so a
    3-company comparison could come back with only 1-2 chunks total instead
    of up to top_k per company. Each company must keep its own full top_k
    budget, independent of how many other companies are in the mix."""

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Apple"),
            _fake_chunk(chunk_id=3, company="Apple"),
            _fake_chunk(chunk_id=4, company="Microsoft"),
            _fake_chunk(chunk_id=5, company="NVIDIA"),
        ]

    def fake_rerank(query, candidates, top_k):
        assert top_k == 2  # each company gets the full per-call top_k, not a shared slice
        return candidates[:top_k]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Comparison answer.")
    result = answer_engine.generate_answer(
        "compare capex query", top_k=2, companies=["Apple", "Microsoft", "NVIDIA"], client=client
    )

    # Apple contributes its full budget (2 of its 3 candidates); Microsoft
    # and NVIDIA each contribute their only candidate. A shared top_k=2
    # cutoff over the merged pool would have returned only 2 chunks total.
    assert sorted(result.retrieved_chunk_ids) == [1, 2, 4, 5]


def test_generate_answer_keeps_untagged_chunks_in_multi_company_comparison(monkeypatch):
    """Regression test: a candidate with no company tag (company=None or "")
    isn't part of any distinct_companies group, so it used to be silently
    dropped from context whenever 2+ real companies were present -- unlike
    the single-company path, which includes everything."""

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
            _fake_chunk(chunk_id=3, company=None),
        ]

    def fake_rerank(query, candidates, top_k):
        return candidates

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Comparison answer.")
    result = answer_engine.generate_answer(
        "compare capex query", companies=["Apple", "Microsoft"], client=client
    )

    assert sorted(result.retrieved_chunk_ids) == [1, 2, 3]


def test_generate_answer_with_history_bypasses_cache(monkeypatch):
    """A history-bearing (multi-turn) query must not be served from -- or
    written to -- the plain query cache, since the same literal text can mean
    something different depending on prior turns."""
    call_count = {"retrieve": 0}

    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        call_count["retrieve"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    history = [HistoryTurn(role="user", content="What about last year?")]
    query = "same follow-up text"
    client = _FakeClient("Answer one.")
    first = answer_engine.generate_answer(query, history=history, client=client)
    client2 = _FakeClient("Answer two.")
    second = answer_engine.generate_answer(query, history=history, client=client2)

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert call_count["retrieve"] == 2  # retrieval ran fresh both times, no cache short-circuit


def test_generate_answer_stream_short_circuits_when_all_chunks_below_rerank_threshold(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk(score=0.0)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("should never stream")
    events = list(
        answer_engine.generate_answer_stream("irrelevant meta question stream", client=client)
    )

    assert client.calls["chat_stream"] == 0
    assert len(events) == 2
    assert events[0] == NO_RELEVANT_CONTEXT_ANSWER
    final = events[1]
    assert final.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert final.retrieved_chunk_ids == []
    assert final.citations == []


def test_generate_answer_stream_yields_tokens_then_final_answer(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    client = _FakeClient("Streamed answer text.")
    events = list(answer_engine.generate_answer_stream("stream query", client=client))

    assert events[0] == "Streamed answer text."
    assert events[-1].answer_text == "Streamed answer text."
    assert client.calls["chat_stream"] == 1


def test_generate_answer_raises_and_logs_error_on_generation_failure(monkeypatch):
    def fake_retrieve_hybrid(query, top_k, companies=None, fiscal_year=None, doc_type=None):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    class FailingClient:
        model = "gemini-test"

        def chat(self, messages):
            raise RuntimeError("simulated API failure")

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_text", lambda text: [0.0] * 8)

    query = "will fail query"
    try:
        answer_engine.generate_answer(query, client=FailingClient())
        raised = False
    except RuntimeError:
        raised = True
    assert raised

    session = get_session()
    log = session.query(QueryLog).filter(QueryLog.query_text == query).first()
    session.close()
    assert log is not None
    assert log.error == "simulated API failure"
