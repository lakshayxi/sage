from sage.retrieval import reranker
from sage.retrieval.retriever import RetrievedChunk


def _make_chunk(chunk_id: int, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        chunk_index=0,
        text=text,
        page_number=1,
        filename="doc.pdf",
        company="Apple",
        fiscal_year="FY24",
        doc_type="10-K",
        score=score,
    )


def test_rerank_orders_by_cross_encoder_score(monkeypatch):
    chunks = [
        _make_chunk(1, "Unrelated text about the weather."),
        _make_chunk(2, "Apple margins declined due to higher component costs."),
    ]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())

    result = reranker.rerank("Why did Apple margins decline?", chunks, top_k=1)

    assert len(result) == 1
    assert result[0].chunk_id == 2
    assert result[0].score == 0.9
    assert chunks[0].score == 0.0
    assert chunks[1].score == 0.0
    assert result[0] is not chunks[1]


def test_rerank_truncates_to_top_k(monkeypatch):
    chunks = [_make_chunk(i, f"chunk {i}") for i in range(5)]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return list(range(len(pairs)))

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())

    result = reranker.rerank("query", chunks, top_k=2)

    assert len(result) == 2
    assert [c.chunk_id for c in result] == [4, 3]


def test_rerank_returns_empty_for_no_candidates(monkeypatch):
    def _fail():
        raise AssertionError("cross-encoder should not load when there are no candidates")

    monkeypatch.setattr(reranker, "_get_model", _fail)

    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_query_variants_do_not_overwrite_prior_results(monkeypatch):
    chunks = [_make_chunk(1, "first", score=0.25), _make_chunk(2, "second", score=0.5)]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.9, 0.1] if pairs[0][0] == "query A" else [0.2, 0.8]

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())

    result_a = reranker.rerank("query A", chunks, top_k=2)
    result_b = reranker.rerank("query B", chunks, top_k=2)

    assert [(c.chunk_id, c.score) for c in result_a] == [(1, 0.9), (2, 0.1)]
    assert [(c.chunk_id, c.score) for c in result_b] == [(2, 0.8), (1, 0.2)]
    assert [c.score for c in chunks] == [0.25, 0.5]


def test_min_rerank_score_gating_drops_low_scoring_chunks(monkeypatch):
    """Mirrors how answer_engine gates on settings.MIN_RERANK_SCORE after
    reranking -- this test exercises the gate itself, not just rerank()'s
    ordering, since that's the mechanism that keeps the pipeline from
    generating an answer off irrelevant context."""
    from config import settings

    chunks = [_make_chunk(1, "on-topic"), _make_chunk(2, "off-topic")]

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.8, 0.01]

    monkeypatch.setattr(reranker, "_get_model", lambda: FakeCrossEncoder())

    reranked = reranker.rerank("query", chunks, top_k=2)
    gated = [c for c in reranked if c.score >= settings.MIN_RERANK_SCORE]

    assert [c.chunk_id for c in gated] == [1]
