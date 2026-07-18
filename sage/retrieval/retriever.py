"""Hybrid retrieval: vector search (Chroma) + BM25 keyword search, fused via
reciprocal rank fusion (RRF). Chroma is queried for nearest-neighbor chunk
ids/distances; SQLite (the source of truth for text/metadata) resolves those
ids into full chunk + document records.

Design difference from the reference project this was built alongside: the
company filter here is `companies: list[str] | None` (plural) from the start,
because Sage supports comparing multiple companies in a single query (e.g.
"Compare Apple vs Microsoft's capex"). When more than one company is given,
`retrieve_hybrid` runs the whole single-company hybrid pipeline once per
company (each call scoped by the existing single-company where-clause
internally) and merges the results round-robin, tagged by source company via
each RetrievedChunk's existing `.company` field. This guarantees every
requested company gets balanced representation in the candidate set handed
to the reranker -- letting RRF fuse across companies directly would instead
let corpus-size or topical-overlap differences between companies silently
starve one side out, since vector distance and BM25 score are not
comparable across independently-scored per-company runs anyway. When
`companies` has zero or one entries, this collapses exactly to a single
`_retrieve_hybrid_single` call -- today's single-company behavior.
"""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from config import settings
from sage.db.database import get_session
from sage.db.models import Chunk, Document
from sage.embed.local_embedder import embed_query
from sage.retrieval import store

_TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60  # standard reciprocal-rank-fusion smoothing constant


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    chunk_index: int
    text: str
    page_number: int | None
    filename: str
    company: str | None
    fiscal_year: str | None
    doc_type: str | None
    score: float


def _normalize_companies(companies: list[str] | None) -> list[str]:
    """Dedupe (order-preserving) and drop falsy entries."""
    if not companies:
        return []
    seen: set[str] = set()
    normalized = []
    for c in companies:
        if c and c not in seen:
            seen.add(c)
            normalized.append(c)
    return normalized


def _build_where(company: str | None, fiscal_year: str | None, doc_type: str | None) -> dict | None:
    clauses = []
    if company:
        clauses.append({"company": company})
    if fiscal_year:
        clauses.append({"fiscal_year": fiscal_year})
    if doc_type:
        clauses.append({"doc_type": doc_type})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _filtered_chunk_rows(
    session, company: str | None, fiscal_year: str | None, doc_type: str | None
) -> list[tuple[Chunk, Document]]:
    query = session.query(Chunk, Document).join(Document, Chunk.document_id == Document.id)
    if company:
        query = query.filter(Document.company == company)
    if fiscal_year:
        query = query.filter(Document.fiscal_year == fiscal_year)
    if doc_type:
        query = query.filter(Document.doc_type == doc_type)
    return query.order_by(Chunk.id).all()


def _bm25_ranked_ids(query_text: str, rows: list[tuple[Chunk, Document]]) -> list[int]:
    """Rank chunk ids by BM25 score against `query_text`.

    The index is rebuilt fresh on every call rather than cached: at this
    corpus's scale (tens to low hundreds of chunks per company), building it
    is sub-millisecond, and rebuilding sidesteps any staleness after an
    ingest/delete without needing cache-invalidation logic.
    """
    if not rows:
        return []
    corpus = [_tokenize(chunk.text) for chunk, _ in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    ranked = sorted(
        zip((chunk.id for chunk, _ in rows), scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    # A score of exactly 0.0 means no genuine keyword overlap; letting those
    # chunks through would give them arbitrary RRF credit off of DB row order.
    return [chunk_id for chunk_id, score in ranked if score > 0.0]


def _reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def _retrieve_hybrid_single(
    query_text: str,
    top_k: int,
    company: str | None,
    fiscal_year: str | None,
    doc_type: str | None,
) -> list[RetrievedChunk]:
    """Vector similarity + BM25 keyword search for a single company (or no
    company filter at all), fused via reciprocal rank fusion.

    Both signals are restricted to the same company/fiscal_year/doc_type
    filter so the fused ranking never mixes in chunks that don't match.
    """
    where = _build_where(company, fiscal_year, doc_type)
    query_embedding = embed_query(query_text)
    vector_result = store.query(query_embedding, top_k=top_k, where=where)
    vector_ids = [int(i) for i in vector_result.get("ids", [[]])[0]]

    session = get_session()
    try:
        rows = _filtered_chunk_rows(session, company, fiscal_year, doc_type)
        bm25_ids = _bm25_ranked_ids(query_text, rows)[:top_k]

        fused_scores = _reciprocal_rank_fusion([vector_ids, bm25_ids])
        if not fused_scores:
            return []
        ranked_chunk_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[
            :top_k
        ]

        by_id = {chunk.id: (chunk, document) for chunk, document in rows}
        retrieved = []
        for chunk_id in ranked_chunk_ids:
            if chunk_id not in by_id:
                continue
            chunk, document = by_id[chunk_id]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    filename=document.filename,
                    company=document.company,
                    fiscal_year=document.fiscal_year,
                    doc_type=document.doc_type,
                    score=fused_scores[chunk_id],
                )
            )
        return retrieved
    finally:
        session.close()


def _merge_balanced(per_company_results: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Round-robin merge each company's independently-ranked result list,
    preserving each list's internal rank order. Every candidate from every
    company is included -- there is no shared cap -- since each company's
    list already carries its own full `top_k` budget from
    `_retrieve_hybrid_single`, and capping the merge would take that budget
    back from whichever companies are later in `per_company_results`.

    Round-robin (rather than concatenation or re-sorting by raw score) is
    still what makes the *ordering* balanced: each company's list was scored
    by its own independent RRF run, so raw scores aren't comparable across
    companies, and a straight concatenation would front-load whichever
    company was processed first.
    """
    merged: list[RetrievedChunk] = []
    max_len = max((len(r) for r in per_company_results), default=0)
    for i in range(max_len):
        for results in per_company_results:
            if i < len(results):
                merged.append(results[i])
    return merged


def retrieve_hybrid(
    query_text: str,
    top_k: int = settings.DEFAULT_TOP_K,
    companies: list[str] | None = None,
    fiscal_year: str | None = None,
    doc_type: str | None = None,
) -> list[RetrievedChunk]:
    """Hybrid retrieval, company-aware for single- and multi-company queries.

    - `companies` is `None` or `[]`: no company filter, one retrieval call
      over the whole corpus.
    - `companies` has exactly one entry: identical to the reference
      project's single-company behavior, one retrieval call scoped to it.
    - `companies` has 2+ entries: one retrieval call per company (each fully
      scoped to that company alone, and each keeping its own full `top_k`
      budget -- not a `top_k` shared across companies), merged round-robin
      -- see `_merge_balanced`.
    """
    normalized = _normalize_companies(companies)

    if len(normalized) <= 1:
        company = normalized[0] if normalized else None
        return _retrieve_hybrid_single(query_text, top_k, company, fiscal_year, doc_type)

    per_company_results = [
        _retrieve_hybrid_single(query_text, top_k, company, fiscal_year, doc_type)
        for company in normalized
    ]
    return _merge_balanced(per_company_results)
