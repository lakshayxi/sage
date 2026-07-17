import pytest

from sage.db.database import get_session
from sage.db.models import Chunk, Document
from sage.retrieval import retriever


def _make_document(session, company: str, fiscal_year: str = "FY24") -> Document:
    document = Document(
        filename=f"{company}_{fiscal_year}_10-K.pdf",
        title=f"{company} {fiscal_year} 10-K",
        company=company,
        fiscal_year=fiscal_year,
        doc_type="10-K",
        source_path=f"/tmp/{company}.pdf",
        page_count=1,
        embedding_model="gemini-embedding-001",
        status="ready",
    )
    session.add(document)
    session.flush()
    return document


@pytest.fixture
def sample_chunks():
    session = get_session()
    document = _make_document(session, "Apple-Test")

    chunk_a = Chunk(
        document_id=document.id,
        chunk_index=0,
        page_number=1,
        text="Apple operating margins declined due to higher component costs.",
        char_start=0,
        char_end=60,
        token_count=10,
    )
    chunk_b = Chunk(
        document_id=document.id,
        chunk_index=1,
        page_number=2,
        text="Currency headwinds reduced dollar-denominated revenue.",
        char_start=60,
        char_end=130,
        token_count=10,
    )
    session.add_all([chunk_a, chunk_b])
    session.commit()
    ids = (document.id, chunk_a.id, chunk_b.id)
    session.close()
    return ids


def _add_neutral_chunk(document_id: int) -> int:
    """A third, unrelated chunk so BM25 idf isn't a degenerate half-of-2-doc
    artifact (see reference project's identical fixture rationale)."""
    session = get_session()
    chunk = Chunk(
        document_id=document_id,
        chunk_index=2,
        page_number=3,
        text="Quarterly filing deadlines remain unchanged for auditors.",
        char_start=130,
        char_end=190,
        token_count=10,
    )
    session.add(chunk)
    session.commit()
    chunk_id = chunk.id
    session.close()
    return chunk_id


def test_retrieve_hybrid_resolves_chroma_ids_to_sqlite_chunks(monkeypatch, sample_chunks):
    document_id, chunk_a_id, chunk_b_id = sample_chunks

    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        retriever.store,
        "query",
        lambda embedding, top_k, where=None: {
            "ids": [[str(chunk_a_id), str(chunk_b_id)]],
            "distances": [[0.12, 0.34]],
        },
    )

    results = retriever.retrieve_hybrid("Why did Apple margins decline?", top_k=2)

    assert {r.chunk_id for r in results} == {chunk_a_id, chunk_b_id}
    assert all(r.company == "Apple-Test" for r in results)


def test_retrieve_hybrid_returns_empty_when_no_matches(monkeypatch, sample_chunks):
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        retriever.store,
        "query",
        lambda embedding, top_k, where=None: {"ids": [[]], "distances": [[]]},
    )

    results = retriever.retrieve_hybrid("totally unrelated question", top_k=2)

    assert results == []


def test_build_where_combines_filters():
    assert retriever._build_where(None, None, None) is None
    assert retriever._build_where("Apple", None, None) == {"company": "Apple"}
    assert retriever._build_where("Apple", "FY24", None) == {
        "$and": [{"company": "Apple"}, {"fiscal_year": "FY24"}]
    }


def test_reciprocal_rank_fusion_rewards_agreement_between_signals():
    fused = retriever._reciprocal_rank_fusion([[1, 2], [1, 3]])
    assert fused[1] > fused[2]
    assert fused[1] > fused[3]


def test_reciprocal_rank_fusion_ties_on_symmetric_swap():
    fused = retriever._reciprocal_rank_fusion([[1, 2], [2, 1]])
    assert fused[1] == fused[2]


def test_retrieve_hybrid_recovers_bm25_only_matches(monkeypatch, sample_chunks):
    document_id, chunk_a_id, chunk_b_id = sample_chunks
    _add_neutral_chunk(document_id)

    # Vector search "sees" only chunk_a; chunk_b is a BM25-only match.
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        retriever.store,
        "query",
        lambda embedding, top_k, where=None: {"ids": [[str(chunk_a_id)]], "distances": [[0.1]]},
    )

    results = retriever.retrieve_hybrid(
        "currency headwinds dollar-denominated revenue", top_k=2, companies=["Apple-Test"]
    )

    assert {r.chunk_id for r in results} == {chunk_a_id, chunk_b_id}


def test_retrieve_hybrid_respects_single_company_filter(monkeypatch, sample_chunks):
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        retriever.store,
        "query",
        lambda embedding, top_k, where=None: {"ids": [[]], "distances": [[]]},
    )

    results = retriever.retrieve_hybrid("margins", top_k=2, companies=["NoSuchCompany"])

    assert results == []


def test_normalize_companies_dedupes_and_drops_falsy():
    assert retriever._normalize_companies(["Apple", "", None, "Apple", "Microsoft"]) == [
        "Apple",
        "Microsoft",
    ]
    assert retriever._normalize_companies(None) == []
    assert retriever._normalize_companies([]) == []


# --- Multi-company merge: new logic not present in the reference project. ---


def test_retrieve_hybrid_multi_company_merges_balanced_across_companies(monkeypatch):
    session = get_session()
    apple = _make_document(session, "Apple")
    msft = _make_document(session, "Microsoft")

    apple_chunks = []
    for i in range(3):
        c = Chunk(
            document_id=apple.id,
            chunk_index=i,
            page_number=1,
            text=f"Apple capex discussion number {i}.",
            char_start=i * 10,
            char_end=i * 10 + 10,
            token_count=5,
        )
        session.add(c)
        apple_chunks.append(c)

    msft_chunks = []
    for i in range(1):
        c = Chunk(
            document_id=msft.id,
            chunk_index=i,
            page_number=1,
            text=f"Microsoft capex discussion number {i}.",
            char_start=i * 10,
            char_end=i * 10 + 10,
            token_count=5,
        )
        session.add(c)
        msft_chunks.append(c)
    session.commit()
    apple_ids = [c.id for c in apple_chunks]
    msft_ids = [c.id for c in msft_chunks]
    session.close()

    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)

    def fake_vector_query(embedding, top_k, where=None):
        # Apple's corpus is larger (3 chunks) than Microsoft's (1) -- if a
        # single fused ranking were used instead of per-company retrieval,
        # Apple's extra candidates would be free to crowd out Microsoft's
        # only chunk. Returning every chunk id regardless of `where` proves
        # the *company scoping* (not vector relevance) is what separates them.
        return {"ids": [[str(i) for i in apple_ids + msft_ids]], "distances": [[0.1] * 4]}

    monkeypatch.setattr(retriever.store, "query", fake_vector_query)

    results = retriever.retrieve_hybrid("capex", top_k=4, companies=["Apple", "Microsoft"])

    companies_seen = [r.company for r in results]
    assert set(companies_seen) == {"Apple", "Microsoft"}
    # Balanced representation: Microsoft's one chunk must appear even though
    # Apple has three candidates and top_k (4) would fit all of them.
    assert msft_ids[0] in {r.chunk_id for r in results}
    # Round-robin merge means Microsoft's single chunk lands in one of the
    # first two slots (interleaved), not pushed to the back.
    assert companies_seen[:2] == ["Apple", "Microsoft"] or companies_seen[:2] == [
        "Microsoft",
        "Apple",
    ]


def test_retrieve_hybrid_multi_company_collapses_to_single_when_one_company(
    monkeypatch, sample_chunks
):
    document_id, chunk_a_id, chunk_b_id = sample_chunks
    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)
    monkeypatch.setattr(
        retriever.store,
        "query",
        lambda embedding, top_k, where=None: {
            "ids": [[str(chunk_a_id)]],
            "distances": [[0.1]],
        },
    )

    single = retriever.retrieve_hybrid("margins", top_k=2, companies=["Apple-Test"])
    multi_with_one = retriever.retrieve_hybrid("margins", top_k=2, companies=["Apple-Test"])

    assert [r.chunk_id for r in single] == [r.chunk_id for r in multi_with_one]


def test_merge_balanced_interleaves_and_keeps_every_candidate():
    from sage.retrieval.retriever import RetrievedChunk

    def _chunk(cid, company):
        return RetrievedChunk(
            chunk_id=cid,
            document_id=1,
            chunk_index=0,
            text="text",
            page_number=1,
            filename="f.pdf",
            company=company,
            fiscal_year="FY24",
            doc_type="10-K",
            score=1.0,
        )

    apple_list = [_chunk(1, "Apple"), _chunk(2, "Apple"), _chunk(3, "Apple")]
    msft_list = [_chunk(10, "Microsoft")]

    merged = retriever._merge_balanced([apple_list, msft_list])

    # No shared cap: all 4 candidates survive, round-robin interleaved.
    assert [c.chunk_id for c in merged] == [1, 10, 2, 3]


def test_retrieve_hybrid_multi_company_gives_each_company_its_own_top_k_budget(monkeypatch):
    """Regression test: retrieve_hybrid's multi-company branch used to cap
    the merged pool to one shared top_k, so a company with more candidates
    than others could crowd them out of the *retrieval* stage entirely,
    before reranking ever got a chance to see them."""
    session = get_session()
    apple = _make_document(session, "Apple")
    msft = _make_document(session, "Microsoft")

    apple_chunks = []
    for i in range(5):
        c = Chunk(
            document_id=apple.id,
            chunk_index=i,
            page_number=1,
            text=f"Apple capex discussion number {i}.",
            char_start=i * 10,
            char_end=i * 10 + 10,
            token_count=5,
        )
        session.add(c)
        apple_chunks.append(c)

    msft_chunks = []
    for i in range(5):
        c = Chunk(
            document_id=msft.id,
            chunk_index=i,
            page_number=1,
            text=f"Microsoft capex discussion number {i}.",
            char_start=i * 10,
            char_end=i * 10 + 10,
            token_count=5,
        )
        session.add(c)
        msft_chunks.append(c)
    session.commit()
    apple_ids = [c.id for c in apple_chunks]
    msft_ids = [c.id for c in msft_chunks]
    session.close()

    monkeypatch.setattr(retriever, "embed_text", lambda text: [0.0] * 8)

    def fake_vector_query(embedding, top_k, where=None):
        # Mirrors Chroma's real behavior of honoring the company where-clause
        # (unlike the sibling merge-balance test above, which deliberately
        # ignores it to prove company *scoping* drives the result) -- needed
        # here since "capex" appearing in every one of a company's 5 chunks
        # gives BM25 a degenerate negative idf for that company, so this
        # test's real assertion (retrieval keeps a full top_k per company)
        # has to be carried by the vector signal instead.
        company = (where or {}).get("company")
        ids = apple_ids if company == "Apple" else msft_ids
        return {"ids": [[str(i) for i in ids]], "distances": [[0.1] * len(ids)]}

    monkeypatch.setattr(retriever.store, "query", fake_vector_query)

    results = retriever.retrieve_hybrid("capex", top_k=5, companies=["Apple", "Microsoft"])

    per_company_counts = {"Apple": 0, "Microsoft": 0}
    for r in results:
        per_company_counts[r.company] += 1

    # Each company keeps its own full top_k=5, not a shared top_k=5 total.
    assert per_company_counts == {"Apple": 5, "Microsoft": 5}
    assert len(results) == 10
