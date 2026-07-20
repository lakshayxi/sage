import pytest

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


@pytest.fixture(autouse=True)
def _fake_answer_engine_embedding(monkeypatch):
    """Answer-engine tests must never load the real local embedding model."""
    monkeypatch.setattr(answer_engine, "embed_query", lambda text: [0.0] * 8)


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


def test_real_gemini_comparison_output_with_markdown_headers_is_parsed():
    """Regression test using raw output actually captured from a live Gemini
    call (gemini-flash-lite-latest, comparison-mode prompt, 2 companies in
    context; see docs/llm-engineer-work-log.md's "Remaining limitations /
    TODOs" for the verification session). Confirms the ```citations fenced
    happy path still parses correctly through Gemini's real markdown-heavy
    comparison formatting (### section headers, multiple citation numbers
    packed into one bracket like "[2, 3]" in the prose body) -- none of that
    should be mistaken for the trailing citations fence itself."""
    raw = (
        "### Apple\n"
        "For the fiscal year ended September 27, 2025, Apple reported total net sales of "
        "$416.161 billion [3]. The company\u2019s net income for the same period was "
        "$112.010 billion [2, 3].\n\n"
        "### Microsoft\n"
        "For the fiscal year ended June 30, 2025, Microsoft reported total revenue of "
        "$281.724 billion [6]. The company\u2019s net income for the same period was "
        "$101.832 billion [6].\n\n"
        "### Comparison\n"
        "In fiscal year 2025, Apple generated higher total revenue ($416.161 billion) "
        "compared to Microsoft ($281.724 billion). Apple also reported a higher net income "
        "($112.010 billion) than Microsoft ($101.832 billion) for their respective fiscal "
        "years.\n\n"
        '```citations\n[{"n": 2, "chunk_id": 49}, {"n": 3, "chunk_id": 44}, '
        '{"n": 6, "chunk_id": 127}]\n```'
    )

    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text.startswith("### Apple")
    assert clean_text.endswith("for their respective fiscal years.")
    assert "```citations" not in clean_text
    assert entries == [
        {"n": 2, "chunk_id": 49},
        {"n": 3, "chunk_id": 44},
        {"n": 6, "chunk_id": 127},
    ]


def test_real_gemini_long_bulleted_output_with_multi_number_citations_is_parsed():
    """Regression test using raw output actually captured from a live Gemini
    call (gemini-flash-lite-latest): a long, multi-section answer with
    markdown headers, bold text, and bulleted lists where several bullets
    cite multiple chunk numbers in one bracket (e.g. "[2, 3, 5]"). Confirms
    the fenced-citations regex still isolates just the trailing fence rather
    than matching too early/greedily against those in-body brackets."""
    raw = (
        "NVIDIA\u2019s business faces a broad array of operational, geopolitical, and "
        "regulatory risks.\n\n"
        "### Geopolitical and Operational Risks\n"
        "*   **Export Controls:** NVIDIA faces complex and shifting export restrictions "
        "[2, 3, 5]. Worldwide controls create significant business uncertainty and "
        "competitive disadvantages [2, 3, 5].\n\n"
        "### Regulatory and Compliance Environment\n"
        "*   **Antitrust and AI Regulation:** Regulators have initiated inquiries into "
        "NVIDIA\u2019s business practices [2, 3].\n\n"
        '```citations\n[{"n": 1, "chunk_id": 219}, {"n": 2, "chunk_id": 229}, '
        '{"n": 3, "chunk_id": 228}, {"n": 4, "chunk_id": 241}, {"n": 5, "chunk_id": 197}]\n```'
    )

    clean_text, entries = _split_answer_and_entries(raw)

    assert clean_text.startswith("NVIDIA\u2019s business faces")
    assert clean_text.endswith("NVIDIA\u2019s business practices [2, 3].")
    assert "```citations" not in clean_text
    assert "chunk_id" not in clean_text
    assert entries == [
        {"n": 1, "chunk_id": 219},
        {"n": 2, "chunk_id": 229},
        {"n": 3, "chunk_id": 228},
        {"n": 4, "chunk_id": 241},
        {"n": 5, "chunk_id": 197},
    ]


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
        "Margins declined due to component costs [1] [2].",
    )

    assert citations == []


def test_resolve_citations_keeps_valid_entries_alongside_malformed_ones():
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2)]

    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 1}, {"chunk_id": 2}],
        chunks,
        "Margins declined due to component costs [1].",
    )

    assert [c.n for c in citations] == [1]
    assert citations[0].chunk_id == 1


def test_resolve_citations_drops_entry_never_referenced_in_answer_text():
    """Regression test for the Tesla-revenue bug (eval item
    unans-tesla-revenue): the model refused to answer but still listed a
    resolved citation entry pointing to a real chunk -- that entry's `n`
    never actually appeared as an inline marker in the answer text, so it
    must be dropped."""
    chunks = [_fake_chunk(chunk_id=1)]

    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 1}],
        chunks,
        "The provided context does not contain information regarding Tesla's "
        "total revenue for fiscal year 2025.",
    )

    assert citations == []


def test_resolve_citations_keeps_entry_referenced_only_inside_multi_number_bracket():
    """Regression test: Gemini legitimately packs multiple citation numbers
    into one bracket like "[1, 3]" -- a citation whose `n` only appears that
    way (never as a standalone "[n]") must still be kept."""
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2), _fake_chunk(chunk_id=3)]

    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 1}, {"n": 2, "chunk_id": 2}, {"n": 3, "chunk_id": 3}],
        chunks,
        "NVIDIA faces complex and shifting export restrictions [1, 3].",
    )

    assert [c.n for c in citations] == [1, 3]


def test_resolve_citations_maps_n_positionally_and_ignores_model_chunk_id():
    """Security regression: `n` must deterministically resolve to
    `chunks[n - 1]` -- the chunk actually shown to the model as "[n]" in the
    prompt (see prompts.py's build_context_block). A model-provided
    `chunk_id` that points at a *different* retrieved chunk must never be
    able to remap citation [1] to that other chunk."""
    chunk_a = _fake_chunk(chunk_id=101, company="Apple")
    chunk_b = _fake_chunk(chunk_id=202, company="Microsoft")
    chunks = [chunk_a, chunk_b]

    # The model claims citation [1] belongs to chunk_id=202 (chunk_b) -- but
    # [1] was shown to it as chunk_a (chunk_id=101). The resolved citation
    # must still point at chunk_a, not chunk_b.
    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 202}],
        chunks,
        "Some claim about Apple [1].",
    )

    assert len(citations) == 1
    assert citations[0].n == 1
    assert citations[0].chunk_id == chunk_a.chunk_id
    assert citations[0].company == "Apple"


def test_resolve_citations_drops_zero_negative_and_out_of_range_numbers():
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2)]

    citations = _resolve_citations(
        [{"n": 0, "chunk_id": 1}, {"n": -1, "chunk_id": 1}, {"n": 99, "chunk_id": 2}],
        chunks,
        "A claim [0] another [-1] and one more [99].",
    )

    assert citations == []


def test_resolve_citations_drops_duplicate_numbers():
    chunks = [_fake_chunk(chunk_id=1)]

    citations = _resolve_citations(
        [{"n": 1, "chunk_id": 1}, {"n": 1, "chunk_id": 1}],
        chunks,
        "Margins declined [1].",
    )

    assert len(citations) == 1
    assert citations[0].n == 1


def test_resolve_citations_accepts_bare_integer_entries():
    """The model is now instructed to prefer a flat list of citation numbers
    (`[1, 3]`) over the legacy `{"n": ..., "chunk_id": ...}` dict form -- both
    must resolve identically."""
    chunks = [_fake_chunk(chunk_id=1), _fake_chunk(chunk_id=2), _fake_chunk(chunk_id=3)]

    citations = _resolve_citations([1, 3], chunks, "Some claim [1, 3].")

    assert [c.n for c in citations] == [1, 3]
    assert [c.chunk_id for c in citations] == [chunks[0].chunk_id, chunks[2].chunk_id]


def test_resolve_citations_rejects_bool_as_citation_number():
    """bool is an int subclass in Python (`isinstance(True, int)` is True) --
    a malformed `{"n": true}` entry must not silently resolve to chunks[0]."""
    chunks = [_fake_chunk(chunk_id=1)]

    citations = _resolve_citations([{"n": True, "chunk_id": 1}], chunks, "A claim [1].")

    assert citations == []


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

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
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
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("The margins declined because of component costs.")
    result = answer_engine.generate_answer("first run query", top_k=5, client=client)

    assert calls == {"retrieve_hybrid": 1, "rerank": 1}
    assert client.calls["chat"] == 1
    assert result.cache_hit is False
    assert result.answer_text == "The margins declined because of component costs."


def test_generate_answer_second_call_hits_cache_and_skips_pipeline(monkeypatch):
    calls = {"retrieve_hybrid": 0}

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer("irrelevant meta question", client=client)

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert result.citations == []
    assert result.retrieved_chunk_ids == []
    assert result.cost_usd == 0.0
    assert result.generation_latency_ms == 0.0


def test_generate_answer_filters_low_scoring_chunks_but_keeps_relevant_ones(monkeypatch):
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [
            _fake_chunk(chunk_id=1, score=0.95),
            _fake_chunk(chunk_id=2, score=settings.MIN_RERANK_SCORE - 0.01),
        ]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        if calls["rerank"] == 1:
            chunk = _fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)
        else:
            chunk = _fake_chunk(chunk_id=99, score=0.9)
        chunk.fiscal_year = "FY25"
        return [chunk]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        calls["rerank"] += 1
        return [_fake_chunk(score=settings.MIN_RERANK_SCORE - 0.01)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "irrelevant meta question with no evaluative clause", client=client
    )

    assert calls == {"retrieve_hybrid": 1, "rerank": 1}
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_generate_answer_uses_comparison_prompt_when_chunks_span_multiple_companies(monkeypatch):
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
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
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
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
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("Comparison answer.")
    result = answer_engine.generate_answer(
        "compare capex query", top_k=2, companies=["Apple", "Microsoft", "NVIDIA"], client=client
    )

    # Apple contributes its full budget (2 of its 3 candidates); Microsoft
    # and NVIDIA each contribute their only candidate. A shared top_k=2
    # cutoff over the merged pool would have returned only 2 chunks total.
    assert sorted(result.retrieved_chunk_ids) == [1, 2, 4, 5]


def test_explicit_comparison_ignores_chunks_outside_requested_company_groups(monkeypatch):
    """Exact comparison scope must not be diluted by untagged evidence."""

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
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
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("Comparison answer.")
    result = answer_engine.generate_answer(
        "compare capex query", companies=["Apple", "Microsoft"], client=client
    )

    assert result.retrieved_chunk_ids == [1, 2]


def test_company_local_rerank_query_removes_comparison_frame_and_other_companies():
    query = (
        "Among Apple, Microsoft, and NVIDIA, which company reported the highest "
        "total revenue in its most recent fiscal year filing?"
    )
    companies = ["Apple", "Microsoft", "NVIDIA"]

    local_query = answer_engine._company_local_rerank_query(query, "Microsoft", companies)

    assert local_query == (
        "What was Microsoft's total revenue in its most recent fiscal year filing?"
    )
    assert "Apple" not in local_query
    assert "NVIDIA" not in local_query
    assert "highest" not in local_query


def test_long_comparison_query_localizes_metric_period_and_units():
    query = (
        "Compare total annual revenue for Apple, Microsoft, and NVIDIA using each "
        "company's ingested fiscal-year filing. State each fiscal year and amount "
        "in millions, then rank them from highest to lowest."
    )

    local_query = answer_engine._company_local_rerank_query(
        query, "Apple", ["Apple", "Microsoft", "NVIDIA"]
    )

    assert local_query == (
        "What was Apple's total annual revenue in its ingested fiscal-year filing. "
        "state the fiscal year and amount in millions?"
    )
    assert "rank" not in local_query


@pytest.mark.parametrize(
    "query",
    [
        "Compare total revenue for Apple, Microsoft, and NVIDIA.",
        "Compare Apple and Microsoft revenue.",
        "Compare Apple's iPhone revenue with Microsoft's Azure revenue.",
        "Which has more revenue, Apple or Microsoft?",
        "Between Apple and Microsoft, what was total revenue?",
    ],
)
def test_company_local_rerank_query_preserves_unsupported_comparison_shapes(query):
    assert (
        answer_engine._company_local_rerank_query(query, "Apple", ["Apple", "Microsoft"]) == query
    )


@pytest.mark.parametrize(
    "query",
    [
        "Between Apple and Microsoft, which company was higher in total revenue?",
        "Among Apple and Microsoft, which company had higher revenue and lower net income?",
        (
            "Compare Apple's total revenue and Microsoft's net income using each company's "
            "fiscal-year filing."
        ),
    ],
)
def test_company_local_rerank_query_preserves_unsafe_supported_frames(query):
    assert (
        answer_engine._company_local_rerank_query(query, "Apple", ["Apple", "Microsoft"]) == query
    )


def test_company_scope_normalization_does_not_activate_comparison_for_variants():
    assert answer_engine._is_explicit_comparison([" Apple ", "apple", "", " "]) is False
    assert answer_engine._is_explicit_comparison(["Apple", " microsoft "]) is True


def test_explicit_three_company_comparison_keeps_low_score_evidence_and_generates(monkeypatch):
    companies = ["Apple", "Microsoft", "NVIDIA"]

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [
            _fake_chunk(chunk_id=1, company="Apple", score=0.01),
            _fake_chunk(chunk_id=2, company="Microsoft", score=0.01),
            _fake_chunk(chunk_id=3, company="NVIDIA", score=0.01),
        ]

    rerank_queries = {}

    def fake_rerank(query, candidates, top_k):
        company = candidates[0].company
        rerank_queries[company] = query
        return [_fake_chunk(chunk_id=candidates[0].chunk_id, company=company, score=0.02)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("All three companies contributed evidence.")
    result = answer_engine.generate_answer(
        "Compare total annual revenue for Apple, Microsoft, and NVIDIA using each "
        "company's ingested fiscal-year filing.",
        companies=companies,
        client=client,
    )

    assert client.calls["chat"] == 1
    assert result.retrieved_chunk_ids == [1, 2, 3]
    assert set(rerank_queries) == set(companies)
    for company, local_query in rerank_queries.items():
        assert company in local_query
        assert all(other not in local_query for other in companies if other != company)
    assert "Compare total annual revenue" in client.last_messages[-1]["content"]


def test_explicit_two_company_comparison_keeps_score_just_below_global_gate(monkeypatch):
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
        ]

    def fake_rerank(query, candidates, top_k):
        candidate = candidates[0]
        return [
            _fake_chunk(
                chunk_id=candidate.chunk_id,
                company=candidate.company,
                score=0.093,
            )
        ]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("Comparison answer.")
    result = answer_engine.generate_answer(
        "Compare revenue.", companies=["Apple", "Microsoft"], client=client
    )

    assert client.calls["chat"] == 1
    assert result.retrieved_chunk_ids == [1, 2]


def test_single_company_score_just_below_global_gate_still_refuses(monkeypatch):
    monkeypatch.setattr(
        answer_engine,
        "retrieve_hybrid",
        lambda *a, **kw: [_fake_chunk(chunk_id=1, company="Microsoft")],
    )
    monkeypatch.setattr(
        answer_engine,
        "rerank",
        lambda *a, **kw: [_fake_chunk(chunk_id=1, company="Microsoft", score=0.093)],
    )
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "What was Microsoft's revenue?", companies=["Microsoft"], client=client
    )

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_unfiltered_mixed_company_candidates_use_one_global_threshold_gate(monkeypatch):
    candidates = [
        _fake_chunk(chunk_id=1, company="Apple"),
        _fake_chunk(chunk_id=2, company="Microsoft"),
    ]
    rerank_calls = []

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", lambda *a, **kw: candidates)

    def fake_rerank(query, received, top_k):
        rerank_calls.append(received)
        return [_fake_chunk(chunk_id=c.chunk_id, company=c.company, score=0.01) for c in received]

    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer("unfiltered comparison-like query", client=client)

    assert len(rerank_calls) == 1
    assert rerank_calls[0] == candidates
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_explicit_comparison_missing_requested_company_refuses_without_generation(monkeypatch):
    monkeypatch.setattr(
        answer_engine,
        "retrieve_hybrid",
        lambda *a, **kw: [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
        ],
    )
    monkeypatch.setattr(answer_engine, "rerank", lambda query, candidates, top_k: candidates)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "Compare revenue.",
        companies=["Apple", "Microsoft", "NVIDIA"],
        client=client,
    )

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert result.retrieved_chunk_ids == []


def test_explicit_out_of_corpus_company_refuses_before_generation(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        captured["companies"] = companies
        return []

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "Compare Apple and Tesla revenue.",
        companies=["Apple", "Tesla"],
        client=client,
    )

    assert captured["companies"] == ["Apple", "Tesla"]
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


@pytest.mark.parametrize("company", ["Tesla", "Amazon"])
def test_filtered_out_of_corpus_company_refuses_before_generation(monkeypatch, company):
    monkeypatch.setattr(answer_engine, "retrieve_hybrid", lambda *args, **kwargs: [])
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *args, **kwargs: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        f"What was {company}'s revenue in fiscal year 2025?",
        companies=[company],
        client=client,
    )

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


@pytest.mark.parametrize("company", ["Tesla", "Amazon"])
def test_unfiltered_named_out_of_corpus_company_refuses_before_generation(monkeypatch, company):
    apple = _fake_chunk(company="Apple", score=0.99)
    apple.text = "Apple reported revenue in fiscal year 2025."
    apple.fiscal_year = "FY25"
    monkeypatch.setattr(answer_engine, "retrieve_hybrid", lambda *args, **kwargs: [apple])
    monkeypatch.setattr(answer_engine, "rerank", lambda *args, **kwargs: [apple])
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *args, **kwargs: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        f"What was {company}'s revenue in fiscal year 2025?", client=client
    )

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_explicit_wrong_fiscal_year_scope_refuses_before_generation(monkeypatch):
    captured = {}

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        captured["fiscal_year"] = fiscal_year
        return []

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(
        "Compare revenue in FY20.",
        companies=["Apple", "Microsoft"],
        fiscal_year="FY20",
        client=client,
    )

    assert captured["fiscal_year"] == "FY20"
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER


def test_single_company_nonexistent_fact_does_not_use_comparison_fallback(monkeypatch):
    query = "How much revenue did NVIDIA's Automotive segment generate in fiscal year 2026?"
    rerank_queries = []

    monkeypatch.setattr(
        answer_engine,
        "retrieve_hybrid",
        lambda *a, **kw: [_fake_chunk(chunk_id=1, company="NVIDIA")],
    )

    def fake_rerank(rerank_query, candidates, top_k):
        rerank_queries.append(rerank_query)
        return [_fake_chunk(chunk_id=1, company="NVIDIA", score=0.998)]

    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(query, companies=["NVIDIA"], client=client)

    assert rerank_queries == [query]
    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert result.citations == []


def test_explicit_comparison_nonexistent_segment_refuses_before_generation(monkeypatch):
    query = "Compare Automotive segment revenue for Microsoft and NVIDIA in fiscal year 2026."
    candidates = [
        _fake_chunk(chunk_id=1, company="Microsoft"),
        _fake_chunk(chunk_id=2, company="NVIDIA"),
    ]
    for chunk in candidates:
        chunk.text = "The filing discusses products and consolidated revenue in fiscal year 2026."
        chunk.fiscal_year = "FY26"

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", lambda *a, **kw: candidates)
    monkeypatch.setattr(
        answer_engine,
        "rerank",
        lambda query, chunks, top_k: chunks,
    )
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    client = _FakeClient("should never be called")
    result = answer_engine.generate_answer(query, companies=["Microsoft", "NVIDIA"], client=client)

    assert client.calls["chat"] == 0
    assert result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert result.citations == []


def test_top_k_must_be_within_candidate_budget():
    client = _FakeClient("should never be called")

    with pytest.raises(ValueError, match="top_k must be between"):
        answer_engine.generate_answer("query", top_k=-1, client=client)
    with pytest.raises(ValueError, match="top_k must be between"):
        answer_engine.generate_answer("query", top_k=True, client=client)
    with pytest.raises(ValueError, match="top_k must be between"):
        answer_engine.generate_answer("query", top_k=3.0, client=client)
    with pytest.raises(ValueError, match="top_k must be between"):
        list(
            answer_engine.generate_answer_stream(
                "query", top_k=settings.RERANK_CANDIDATE_K + 1, client=client
            )
        )

    assert client.calls["chat"] == 0


def test_scope_validation_accepts_an_explicitly_named_segment():
    chunks = [
        _fake_chunk(company="NVIDIA"),
    ]
    chunks[0].text = (
        "The Compute & Networking segment reported revenue, while the Graphics segment "
        "reported a smaller amount."
    )

    reason = answer_engine._unsupported_scope_reason(
        "What revenue did the Graphics segment report?", chunks
    )

    assert reason is None


@pytest.mark.parametrize("modifier", ["reportable", "operating", "business"])
def test_scope_validation_rejects_missing_segment_with_financial_modifier(modifier):
    chunks = [_fake_chunk(company="NVIDIA")]
    chunks[0].text = "Automotive products contributed to consolidated revenue."

    reason = answer_engine._unsupported_scope_reason(
        f"How much revenue did the Automotive {modifier} segment generate?", chunks, ["NVIDIA"]
    )

    assert reason == "requested_segment_not_in_evidence:automotive"


def test_scope_validation_accepts_multi_word_segment_label():
    chunks = [_fake_chunk(company="NVIDIA")]
    chunks[0].text = "The Compute & Networking segment reported revenue."

    reason = answer_engine._unsupported_scope_reason(
        "What revenue did the Compute & Networking segment report?", chunks, ["NVIDIA"]
    )

    assert reason is None


def test_scope_validation_rejects_period_missing_from_text_and_metadata():
    chunks = [_fake_chunk(company="Apple")]
    chunks[0].text = "Apple reported fiscal year 2025 net sales."
    chunks[0].fiscal_year = "FY25"

    reason = answer_engine._unsupported_scope_reason(
        "What were Apple's net sales in fiscal year 2020?", chunks
    )

    assert reason == "requested_fiscal_year_not_in_evidence:2020"


@pytest.mark.parametrize("year_label", ["FY20", "FY2020", "fiscal-year FY2020"])
def test_scope_validation_rejects_standalone_wrong_fiscal_year_variants(year_label):
    chunks = [_fake_chunk(company="Apple")]
    chunks[0].text = "Apple also discusses trends that began in 2020."
    chunks[0].fiscal_year = "FY25"

    reason = answer_engine._unsupported_scope_reason(
        f"What were Apple's net sales in {year_label}?", chunks, ["Apple"]
    )

    assert reason == "requested_fiscal_year_not_in_evidence:2020"


def test_scope_validation_requires_every_year_mentioned_in_a_multi_year_query():
    chunks = [_fake_chunk(company="Apple")]
    chunks[0].text = "Apple reported fiscal year 2025 net sales."
    chunks[0].fiscal_year = "FY25"

    reason = answer_engine._unsupported_scope_reason(
        "Compare Apple's fiscal year 2020 revenue with its fiscal year 2021 revenue.",
        chunks,
        ["Apple"],
    )

    assert reason == "requested_fiscal_year_not_in_evidence:2020"


@pytest.mark.parametrize(
    ("query", "company"),
    [
        ("What was Bank of America's revenue?", "Bank of America"),
        ("What was Berkshire Hathaway’s revenue?", "Berkshire Hathaway"),
        ("What was The Walt Disney Company's revenue?", "The Walt Disney Company"),
    ],
)
def test_scope_validation_accepts_multi_word_possessive_company(query, company):
    chunks = [_fake_chunk(company=company)]
    chunks[0].text = f"{company} reported revenue."

    assert answer_engine._unsupported_scope_reason(query, chunks) is None


def test_filtered_scope_does_not_treat_ordinary_possessive_as_another_company():
    chunks = [_fake_chunk(company="Microsoft")]
    chunks[0].text = "Azure revenue is included in Microsoft's filing."

    reason = answer_engine._unsupported_scope_reason(
        "What did Microsoft's Azure's revenue include?", chunks, ["Microsoft"]
    )

    assert reason is None


def test_unfiltered_scope_does_not_treat_generic_possessive_as_a_company():
    chunks = [_fake_chunk(company="Apple")]
    chunks[0].text = "The CEO's compensation is described in the filing."

    assert (
        answer_engine._unsupported_scope_reason("What was the CEO's compensation?", chunks) is None
    )


def test_scope_validation_defers_mixed_company_year_association_to_generation():
    apple = _fake_chunk(company="Apple")
    apple.text = "Apple fiscal year 2025 revenue."
    apple.fiscal_year = "FY25"
    nvidia = _fake_chunk(company="NVIDIA")
    nvidia.text = "NVIDIA fiscal year 2026 revenue."
    nvidia.fiscal_year = "FY26"

    reason = answer_engine._unsupported_scope_reason(
        "Compare Apple fiscal year 2025 revenue with NVIDIA fiscal year 2026 revenue.",
        [apple, nvidia],
        ["Apple", "NVIDIA"],
    )

    assert reason is None


def test_explicit_comparison_citations_align_with_company_chunks(monkeypatch):
    candidates = [
        _fake_chunk(chunk_id=11, company="Apple"),
        _fake_chunk(chunk_id=22, company="Microsoft"),
        _fake_chunk(chunk_id=33, company="NVIDIA"),
    ]
    monkeypatch.setattr(answer_engine, "retrieve_hybrid", lambda *a, **kw: candidates)
    monkeypatch.setattr(answer_engine, "rerank", lambda query, chunks, top_k: chunks)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)

    content = (
        "Apple [1], Microsoft [2], and NVIDIA [3] all contributed evidence.\n"
        "```citations\n[1, 2, 3]\n```"
    )
    result = answer_engine.generate_answer(
        "Compare the three companies.",
        companies=["Apple", "Microsoft", "NVIDIA"],
        client=_FakeClient(content),
    )

    assert [(c.n, c.chunk_id, c.company, c.filename) for c in result.citations] == [
        (1, 11, "Apple", "Apple_FY24_10-K.pdf"),
        (2, 22, "Microsoft", "Microsoft_FY24_10-K.pdf"),
        (3, 33, "NVIDIA", "NVIDIA_FY24_10-K.pdf"),
    ]


def test_streaming_and_non_streaming_share_identical_retrieval_selection_scope(monkeypatch):
    calls = []

    def fake_cached_or_embed(*args, **kwargs):
        return None, [0.0] * 8

    def fake_retrieve_and_rerank(
        query, top_k, companies, fiscal_year, doc_type, query_embedding=None
    ):
        calls.append((query, top_k, companies, fiscal_year, doc_type, query_embedding))
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
        ], 1.0

    monkeypatch.setattr(answer_engine, "_cached_or_embed", fake_cached_or_embed)
    monkeypatch.setattr(answer_engine, "_retrieve_and_rerank", fake_retrieve_and_rerank)

    kwargs = {
        "query": "Compare revenue.",
        "top_k": 3,
        "companies": ["Apple", "Microsoft"],
        "fiscal_year": "FY25",
        "doc_type": "filing",
    }
    answer_engine.generate_answer(**kwargs, client=_FakeClient("answer"))
    list(answer_engine.generate_answer_stream(**kwargs, client=_FakeClient("answer")))

    assert calls == [
        ("Compare revenue.", 3, ["Apple", "Microsoft"], "FY25", "filing", [0.0] * 8),
        ("Compare revenue.", 3, ["Apple", "Microsoft"], "FY25", "filing", [0.0] * 8),
    ]


def test_streaming_and_non_streaming_both_refuse_unsupported_scope(monkeypatch):
    chunk = _fake_chunk(company="Apple")
    chunk.text = "Apple reported fiscal year 2025 net sales."
    chunk.fiscal_year = "FY25"
    monkeypatch.setattr(
        answer_engine,
        "_cached_or_embed",
        lambda *args, **kwargs: (None, [0.0] * 8),
    )
    monkeypatch.setattr(
        answer_engine,
        "_retrieve_and_rerank",
        lambda *args, **kwargs: ([chunk], 1.0),
    )

    client = _FakeClient("should never be called")
    kwargs = {
        "query": "What were Apple's net sales in fiscal year 2020?",
        "companies": ["Apple"],
        "client": client,
    }
    direct = answer_engine.generate_answer(**kwargs)
    streamed = list(answer_engine.generate_answer_stream(**kwargs))
    streamed_result = streamed[-1]

    assert direct.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert isinstance(streamed_result, answer_engine.AnswerResult)
    assert streamed_result.answer_text == NO_RELEVANT_CONTEXT_ANSWER
    assert client.calls["chat"] == 0
    assert client.calls["chat_stream"] == 0


def test_generate_answer_with_history_bypasses_cache(monkeypatch):
    """A history-bearing (multi-turn) query must not be served from -- or
    written to -- the plain query cache, since the same literal text can mean
    something different depending on prior turns."""
    call_count = {"retrieve": 0}

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        call_count["retrieve"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk(score=0.0)]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    client = _FakeClient("Streamed answer text.")
    events = list(answer_engine.generate_answer_stream("stream query", client=client))

    assert events[0] == "Streamed answer text."
    assert events[-1].answer_text == "Streamed answer text."
    assert client.calls["chat_stream"] == 1


def test_generate_answer_and_stream_resolve_mismatched_chunk_id_identically(monkeypatch):
    """Streaming and non-streaming both funnel through the same
    `_resolve_citations` -- a model response that tries to remap citation
    [1] to a different chunk_id must be corrected identically on both
    paths, not just the non-streaming one."""

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        return [_fake_chunk(chunk_id=101), _fake_chunk(chunk_id=202)]

    def fake_rerank(query, candidates, top_k):
        return candidates

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(answer_engine, "get_semantic_cached", lambda *a, **kw: None)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    malicious_content = (
        'Some claim about the first chunk [1].\n```citations\n[{"n": 1, "chunk_id": 202}]\n```'
    )

    non_streaming = answer_engine.generate_answer(
        "mismatch query one", client=_FakeClient(malicious_content)
    )
    stream_events = list(
        answer_engine.generate_answer_stream(
            "mismatch query two", client=_FakeClient(malicious_content)
        )
    )
    streaming_result = stream_events[-1]

    for result in (non_streaming, streaming_result):
        assert len(result.citations) == 1
        assert result.citations[0].n == 1
        assert result.citations[0].chunk_id == 101  # never remapped to 202


def test_generate_answer_use_cache_false_bypasses_read_and_write(monkeypatch):
    """use_cache=False (the eval harness's mode, eval/run_eval.py) must
    never serve a cached answer and must never write one -- otherwise a
    harness meant to re-test generation quality would silently just replay
    whatever the first run produced."""
    calls = {"retrieve_hybrid": 0, "chat": 0}

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        calls["retrieve_hybrid"] += 1
        return [_fake_chunk()]

    def fake_rerank(query, candidates, top_k):
        return [_fake_chunk()]

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

    query = "use-cache-false query"
    client = _FakeClient("First answer.")
    first = answer_engine.generate_answer(query, client=client, use_cache=False)
    assert first.cache_hit is False
    assert calls["retrieve_hybrid"] == 1

    # A second call with the same query and a different answer must NOT
    # come back as a cache hit -- if the first call had written to the
    # cache, this would incorrectly return "First answer." again.
    client2 = _FakeClient("Second, different answer.")
    second = answer_engine.generate_answer(query, client=client2, use_cache=False)
    assert second.cache_hit is False
    assert second.answer_text == "Second, different answer."
    assert calls["retrieve_hybrid"] == 2  # retrieval ran fresh both times

    assert (
        cache.get_cached(cache.make_cache_key(query, None, None, None, client.model)) is None
    )  # nothing was ever written


def test_generate_answer_embeds_query_exactly_once_end_to_end(monkeypatch):
    """A cache miss for an ordinary (including multi-company) query used to
    embed the identical query text multiple times: once for the semantic
    cache lookup, then again per company inside retrieve_hybrid. The whole
    request must now compute the embedding exactly once and reuse it for
    both the semantic cache check and retrieval."""
    embed_calls = []

    def counting_embed_query(text):
        embed_calls.append(text)
        return [0.0] * 8

    monkeypatch.setattr(answer_engine, "embed_query", counting_embed_query)
    monkeypatch.setattr(cache, "embed_query", counting_embed_query)

    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
        assert query_embedding == [0.0] * 8  # reused, not recomputed
        return [
            _fake_chunk(chunk_id=1, company="Apple"),
            _fake_chunk(chunk_id=2, company="Microsoft"),
            _fake_chunk(chunk_id=3, company="NVIDIA"),
        ]

    def fake_rerank(query, candidates, top_k):
        return candidates

    monkeypatch.setattr(answer_engine, "retrieve_hybrid", fake_retrieve_hybrid)
    monkeypatch.setattr(answer_engine, "rerank", fake_rerank)
    # get_semantic_cached runs for real here (a genuine miss against an
    # empty, test-isolated Chroma collection) -- it must consume the
    # already-computed embedding rather than calling embed_query itself.

    client = _FakeClient("Comparison answer.")
    answer_engine.generate_answer(
        "embed-once comparison query",
        companies=["Apple", "Microsoft", "NVIDIA"],
        client=client,
    )

    assert embed_calls == ["embed-once comparison query"]


def test_generate_answer_raises_and_logs_error_on_generation_failure(monkeypatch):
    def fake_retrieve_hybrid(
        query, top_k, companies=None, fiscal_year=None, doc_type=None, query_embedding=None
    ):
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
    monkeypatch.setattr(cache, "embed_query", lambda text: [0.0] * 8)

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
