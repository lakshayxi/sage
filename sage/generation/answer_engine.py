"""Orchestrates cache-lookup -> hybrid retrieve -> rerank -> prompt -> generate ->
parse citations -> cache-store.

`companies: list[str] | None` flows through this whole path (retrieval,
cache key, and the query log) rather than a singular `company`, matching
`retrieve_hybrid`'s signature -- see sage/retrieval/retriever.py's module
docstring for why plural-from-the-start matters for Sage's comparison
feature.
"""

import json
import logging
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field

from config import settings
from sage.db.conversations import HistoryTurn
from sage.db.query_log import record_query_log
from sage.embed.local_embedder import embed_query
from sage.generation.cache import get_cached, get_semantic_cached, make_cache_key, store_cached
from sage.generation.cost import estimate_cost_usd
from sage.generation.gemini_client import GeminiChatClient, StreamDone, StreamToken
from sage.generation.prompts import build_messages
from sage.retrieval.reranker import rerank
from sage.retrieval.retriever import RetrievedChunk, _normalize_companies, retrieve_hybrid

CITATION_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

CITATIONS_FENCED_RE = re.compile(r"```citations\s*\n(.*?)```", re.DOTALL)
# Fallback for a model that drops the backtick fence but still labels the
# block and ends the response with a JSON array -- carried over defensively
# from the reference project (observed there with llama3.1 8B); Gemini's
# actual behavior here is unverified without a live key (see prompts.py).
CITATIONS_UNFENCED_RE = re.compile(r"\n?citations\s*\n(\[.*\])\s*\Z", re.DOTALL | re.IGNORECASE)
UNCLOSED_CITATIONS_RE = re.compile(r"```citations|\n?citations\s*\n\[", re.IGNORECASE)
BARE_TRAILING_FENCE_RE = re.compile(r"\n*```\s*\Z")

# Returned when ordinary reranking or deterministic scope validation finds no
# usable evidence. The LLM is never called in this case.
NO_RELEVANT_CONTEXT_ANSWER = (
    "I don't have information in the ingested documents that's relevant to this question."
)

# A trailing evaluative clause (", were they good?", ". is that good or bad?")
# tacked onto an otherwise on-topic question craters the cross-encoder
# reranker's score even though the underlying question is answerable from the
# corpus -- see CLAUDE.md for measured scores. Narrowly scoped to that
# pattern, not a general query-rewriting system: a short trailing was/were/
# is/are ... ? clause joined to the main clause by a period or comma.
TRAILING_EVALUATIVE_CLAUSE_RE = re.compile(
    r"[.,]\s*(?:was|were|is|are)\b[^?]*\?\s*\Z", re.IGNORECASE
)

# Comparison queries ask the cross-encoder to match one company's filing
# against wording dominated by a company list, superlative, and final
# synthesis instruction. Strip that frame deterministically so each company
# is scored against the fact it needs to contribute. No LLM is used for this
# transformation, and the original user query is preserved for generation.
COMPARISON_LEAD_RE = re.compile(
    r"^\s*(?:among|between)\b.*?\bwhich\s+(?:company|one)\b\s*", re.IGNORECASE | re.DOTALL
)
COMPARISON_REPORTING_VERB_RE = re.compile(
    r"^\s*(?:reported|had|generated|recorded|earned|spent)\s+", re.IGNORECASE
)
COMPARISON_SUPERLATIVE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:highest|higher|lowest|lower|largest|smaller|smallest|most|least)\s+",
    re.IGNORECASE,
)
COMPARISON_RANKING_CLAUSE_RE = re.compile(
    r"[,;]?\s*(?:then\s+)?rank(?:ing)?\b.*\Z", re.IGNORECASE | re.DOTALL
)
COMPARISON_METRIC_RE = re.compile(
    r"\b(?:total(?:\s+annual)?\s+revenue|net\s+sales|net\s+income|operating\s+income|"
    r"r\s*&\s*d\s+expense|research\s+and\s+development\s+expense|revenue)\b",
    re.IGNORECASE,
)
REQUESTED_FISCAL_YEAR_RE = re.compile(
    r"\b(?:(?:fiscal[\s-]+year)\s*(?:fy\s*)?|fy\s*)(20\d{2}|\d{2})\b", re.IGNORECASE
)
REQUESTED_POSSESSIVE_COMPANY_RE = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9.&-]*"
    r"(?:\s+(?:(?:of|the|and|&)\s+)?[A-Z][A-Za-z0-9.&-]*){0,5}"
    r")['’]s\b"
)
REQUESTED_SEGMENT_RE = re.compile(
    r"\b([a-z][a-z0-9&-]*(?:\s+(?:&|and)\s+[a-z][a-z0-9&-]*)?)\s+"
    r"(?:(?:business|operating|reportable)\s+)?segment\b",
    re.IGNORECASE,
)
GENERIC_SEGMENT_WORDS = {"a", "any", "business", "operating", "reportable", "the", "what", "which"}
GENERIC_POSSESSIVE_WORDS = {
    "board",
    "business",
    "ceo",
    "cfo",
    "company",
    "customer",
    "filing",
    "management",
    "segment",
}

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    n: int
    chunk_id: int
    text: str
    page_number: int | None
    company: str | None
    fiscal_year: str | None
    doc_type: str | None
    filename: str


@dataclass
class AnswerResult:
    answer_text: str
    citations: list[Citation]
    model: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    cache_hit: bool = False
    cost_usd: float = 0.0


def _split_answer_and_entries(raw_text: str) -> tuple[str, list[dict]]:
    """Split the model's raw output into (clean answer text, citation entries).

    Prefers the documented ```citations fenced format, then progressively
    looser fallbacks -- see module docstring for provenance; these fallbacks
    are unverified against real Gemini output.
    """
    match = CITATIONS_FENCED_RE.search(raw_text)
    if match:
        clean_text = CITATIONS_FENCED_RE.sub("", raw_text).rstrip()
        return clean_text, _safe_json_array(match.group(1))

    match = CITATIONS_UNFENCED_RE.search(raw_text)
    if match:
        clean_text = raw_text[: match.start()].rstrip()
        return clean_text, _safe_json_array(match.group(1))

    label_match = UNCLOSED_CITATIONS_RE.search(raw_text)
    if label_match:
        clean_text = raw_text[: label_match.start()].rstrip()
        array_blob = _extract_balanced_json_array(raw_text, label_match.start())
        if array_blob is not None:
            return clean_text, _safe_json_array(array_blob)
        return clean_text, []

    clean_text = raw_text.rstrip()
    bare_fence = BARE_TRAILING_FENCE_RE.search(clean_text)
    if bare_fence:
        return clean_text[: bare_fence.start()].rstrip(), []

    return clean_text, []


def _extract_balanced_json_array(text: str, search_from: int) -> str | None:
    """Return the first bracket-balanced `[...]` substring in `text` at or
    after `search_from`, or None if no "[" is found or it never closes."""
    start = text.find("[", search_from)
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _safe_json_array(blob: str) -> list[dict]:
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _referenced_citation_numbers(answer_text: str) -> set[int]:
    """Collect every citation number appearing in any `[n]` or `[n, m, ...]`
    inline marker in the answer text."""
    numbers = set()
    for match in CITATION_MARKER_RE.finditer(answer_text):
        numbers.update(int(n) for n in match.group(1).split(","))
    return numbers


def _entry_citation_number(entry: dict | int) -> int | None:
    """Pull a citation number out of a model-provided entry, accepting both
    the preferred bare-integer form (`[1, 3, 5]`) and the legacy
    `{"n": 1, "chunk_id": ...}` dict form. Returns None for anything that
    isn't a genuine int (including bool, which is a `int` subclass in
    Python but never a valid citation number)."""
    if isinstance(entry, dict):
        entry = entry.get("n")
    if isinstance(entry, bool) or not isinstance(entry, int):
        return None
    return entry


def _resolve_citations(
    entries: list[dict], chunks: list[RetrievedChunk], answer_text: str
) -> list[Citation]:
    """Resolve the model's trailing citation numbers against the retrieved
    chunks.

    Identity is deterministic and positional: citation number `n` always
    means `chunks[n - 1]` -- the exact chunk the model was shown as `[n]` in
    the prompt (see prompts.py's `build_context_block`, which labels chunks
    `[1]`, `[2]`, ... in this same order). A `chunk_id` the model echoes back
    in its citation JSON is never consulted to pick the chunk: trusting it
    would let the model (accidentally or adversarially) remap citation `[1]`
    in the visible answer text to point at a completely different retrieved
    chunk than the one actually shown as `[1]`, which is exactly the
    citation-integrity bug this function closes.

    An entry number is dropped (not resolved) if it's malformed (missing or
    non-int `n`), zero/negative, out of range for `chunks`, a duplicate of
    an already-resolved number, or never actually referenced inline as an
    `[n]`/`[n, m, ...]` marker in `answer_text` -- the model sometimes lists
    a citation it never actually used in the visible prose (e.g. when it
    declines to answer), and such entries must not survive into the
    response.
    """
    referenced_numbers = _referenced_citation_numbers(answer_text)
    seen: set[int] = set()
    citations = []
    for entry in entries:
        n = _entry_citation_number(entry)
        if n is None:
            logger.warning("Dropping malformed citation entry missing a valid 'n': %r", entry)
            continue
        if n in seen:
            continue
        if n < 1 or n > len(chunks):
            continue
        if n not in referenced_numbers:
            continue
        seen.add(n)
        chunk = chunks[n - 1]
        citations.append(
            Citation(
                n=n,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                page_number=chunk.page_number,
                company=chunk.company,
                fiscal_year=chunk.fiscal_year,
                doc_type=chunk.doc_type,
                filename=chunk.filename,
            )
        )
    return citations


def _safe_semantic_cached(
    query: str,
    companies: list[str] | None,
    fiscal_year: str | None,
    doc_type: str | None,
    model: str,
    query_embedding: list[float],
    top_k: int,
):
    # Chroma read on an already-computed embedding; a network hiccup or
    # Chroma lock here shouldn't turn what would otherwise be a normal
    # (uncached) generation into a failed request.
    try:
        return get_semantic_cached(
            query,
            companies,
            fiscal_year,
            doc_type,
            model,
            query_embedding,
            top_k=top_k,
        )
    except Exception:
        logger.warning("Semantic cache lookup failed; continuing without it", exc_info=True)
        return None


def _safe_store_cached(*args, **kwargs) -> None:
    # Runs after a successful generation -- a failure here must not turn a
    # good answer into an error for the caller.
    try:
        store_cached(*args, **kwargs)
    except Exception:
        logger.warning("Cache store failed; answer was still generated successfully", exc_info=True)


def _safe_record_query_log(*args, **kwargs) -> None:
    # Observability only -- a logging failure must never turn a successful
    # (or already-failed, already-raised) answer flow into a second error.
    try:
        record_query_log(*args, **kwargs)
    except Exception:
        logger.warning("Query log write failed", exc_info=True)


def _no_relevant_context_answer(
    model: str, retrieval_latency_ms: float, total_start: float
) -> AnswerResult:
    return AnswerResult(
        answer_text=NO_RELEVANT_CONTEXT_ANSWER,
        citations=[],
        model=model,
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=0.0,
        total_latency_ms=(time.perf_counter() - total_start) * 1000,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        retrieved_chunk_ids=[],
    )


def _history_as_messages(history: list[HistoryTurn] | None) -> list[dict] | None:
    if not history:
        return None
    return [{"role": h.role, "content": h.content} for h in history]


def _answer_from_cache(cached, total_start: float) -> AnswerResult:
    model = cached.model_name or settings.GEMINI_CHAT_MODEL
    prompt_tokens = cached.prompt_tokens or 0
    completion_tokens = cached.completion_tokens or 0
    return AnswerResult(
        answer_text=cached.answer_text,
        citations=[Citation(**c) for c in cached.citations_json],
        model=model,
        retrieval_latency_ms=0.0,
        generation_latency_ms=0.0,
        total_latency_ms=(time.perf_counter() - total_start) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=cached.total_tokens or 0,
        retrieved_chunk_ids=cached.retrieved_chunk_ids or [],
        cache_hit=True,
        cost_usd=estimate_cost_usd(model, prompt_tokens, completion_tokens),
    )


def _is_explicit_comparison(companies: list[str] | None) -> bool:
    return len(_normalize_companies(companies)) >= 2


def _validate_top_k(top_k: int) -> None:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= settings.RERANK_CANDIDATE_K
    ):
        raise ValueError(f"top_k must be between 1 and {settings.RERANK_CANDIDATE_K}")


def _company_local_rerank_query(query: str, company: str, requested_companies: list[str]) -> str:
    """Turn common comparison frames into a company-local fact request.

    Hybrid retrieval still uses the original query, so this helper only
    changes the cross-encoder's final ranking signal. The transformation is
    deliberately syntactic: remove the comparison lead/list, superlative,
    and ranking instruction while retaining the requested metric, period,
    units, and filing language.
    """
    stripped = query.strip()
    supported_shape = (
        bool(re.match(r"^(?:among|between)\b", stripped, re.I))
        and bool(re.search(r"\bwhich company\b", stripped, re.I))
    ) or (
        bool(re.match(r"^compare\b", stripped, re.I))
        and bool(re.search(r"\busing\s+each\s+company['’]s\b", stripped, re.I))
    )
    if not supported_shape:
        # A mixed-metric or free-form comparison cannot be decomposed safely
        # with string surgery. Preserve the original cross-encoder query.
        return query
    # The templates below localize one company-independent metric. Mixed or
    # company-specific metrics need semantic association that string surgery
    # cannot preserve safely, so keep the original query for those shapes.
    if len(COMPARISON_METRIC_RE.findall(stripped)) != 1:
        return query
    possessive_mentions = [
        match.group(1).casefold() for match in REQUESTED_POSSESSIVE_COMPANY_RE.finditer(stripped)
    ]
    if any(name in possessive_mentions for name in (c.casefold() for c in requested_companies)):
        return query

    focus = COMPARISON_LEAD_RE.sub("", stripped, count=1)
    focus = re.sub(r"^\s*compare\b\s*", "", focus, count=1, flags=re.IGNORECASE)
    for requested_company in requested_companies:
        focus = re.sub(rf"\b{re.escape(requested_company)}\b", "", focus, flags=re.IGNORECASE)
    focus = COMPARISON_REPORTING_VERB_RE.sub("", focus, count=1)
    focus = COMPARISON_SUPERLATIVE_RE.sub("", focus, count=1)
    # "which company was higher" is not one of the reporting-verb templates;
    # falling back is safer than emitting "What was Apple's was higher ...".
    if re.match(r"^\s*was\b", focus, re.IGNORECASE):
        return query
    focus = re.sub(r"\beach company['’]s\b", "its", focus, flags=re.IGNORECASE)
    focus = re.sub(r"\bfor\s*(?:(?:,\s*)|(?:and\s*))*?(?=using\b)", "", focus, flags=re.IGNORECASE)
    focus = re.sub(r"\busing\s+its\b", "in its", focus, flags=re.IGNORECASE)
    focus = re.sub(r"\bstate\s+each\s+fiscal year\b", "state the fiscal year", focus, flags=re.I)
    focus = COMPARISON_RANKING_CLAUSE_RE.sub("", focus)
    focus = re.sub(r"\s+([,.;:?])", r"\1", focus)
    focus = re.sub(r"\s+", " ", focus).strip(" ,.;?")
    if not focus:
        focus = "information needed for this comparison in its fiscal-year filing"
    return f"What was {company}'s {focus}?"


def _select_explicit_comparison_evidence(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    requested_companies: list[str],
) -> tuple[list[RetrievedChunk], float]:
    """Select a bounded rank-based evidence slice for every requested company.

    Raw BGE sigmoid outputs are intentionally logged but never treated as
    calibrated answerability probabilities here. Missing a requested group
    refuses the whole comparison instead of silently generating a partial
    ranking.
    """
    top_score = 0.0
    chunks: list[RetrievedChunk] = []
    for company in requested_companies:
        company_key = company.casefold()
        seen_ids: set[int] = set()
        company_candidates = []
        for candidate in candidates:
            if (
                candidate.company
                and candidate.company.casefold() == company_key
                and candidate.chunk_id not in seen_ids
            ):
                seen_ids.add(candidate.chunk_id)
                company_candidates.append(candidate)
        if not company_candidates:
            logger.info(
                "Comparison evidence refusal: company=%r candidate_count=0 reason=missing_scope",
                company,
            )
            return [], top_score

        rerank_query = _company_local_rerank_query(query, company, requested_companies)
        reranked = rerank(rerank_query, company_candidates, top_k=top_k)
        company_top_score = reranked[0].score if reranked else 0.0
        top_score = max(top_score, company_top_score)
        logger.info(
            "Comparison evidence: company=%r query_variant=%s "
            "candidate_count=%d top_score=%.4f selected_count=%d "
            "absolute_gate_bypassed=%s",
            company,
            "company_local" if rerank_query != query else "original_fallback",
            len(company_candidates),
            company_top_score,
            len(reranked),
            bool(reranked),
        )
        if not reranked:
            logger.info(
                "Comparison evidence refusal: company=%r reason=empty_rerank_selection",
                company,
            )
            return [], top_score
        chunks.extend(reranked)
    return chunks, top_score


def _rerank_and_gate(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    companies: list[str] | None = None,
) -> tuple[list[RetrievedChunk], float]:
    """Select evidence using explicit caller scope, not candidate metadata."""
    requested_companies = _normalize_companies(companies)
    if len(requested_companies) >= 2:
        return _select_explicit_comparison_evidence(query, candidates, top_k, requested_companies)

    reranked = rerank(query, candidates, top_k=top_k)
    top_score = reranked[0].score if reranked else 0.0
    chunks = [c for c in reranked if c.score >= settings.MIN_RERANK_SCORE]
    return chunks, top_score


def _unsupported_scope_reason(
    query: str,
    chunks: list[RetrievedChunk],
    companies: list[str] | None = None,
) -> str | None:
    """Return a deterministic reason when selected evidence misses explicit scope.

    The reranker cannot distinguish a requested period/category from a
    nearby one. Two narrow lexical checks are safe to make before generation:
    a stated fiscal year must occur in the selected text or document metadata,
    and a named singular ``X segment`` must actually be labeled ``X segment``
    rather than only appearing as an end market/platform/product category.
    """
    requested_companies = _normalize_companies(companies)
    evidence_groups = (
        [
            (
                company,
                [
                    chunk
                    for chunk in chunks
                    if chunk.company and chunk.company.casefold() == company.casefold()
                ],
            )
            for company in requested_companies
        ]
        if len(requested_companies) >= 2
        else [(None, chunks)]
    )

    # For an unfiltered singular possessive query ("Tesla's revenue"), a
    # company absent from every selected chunk is deterministically outside
    # the evidence scope. This is deliberately narrow, not general-purpose
    # company entity recognition.
    if not requested_companies:
        evidence_companies = {chunk.company.casefold() for chunk in chunks if chunk.company}
        for company_match in REQUESTED_POSSESSIVE_COMPANY_RE.finditer(query):
            named_company = company_match.group(1).strip()
            words = named_company.split()
            if named_company.casefold() in GENERIC_POSSESSIVE_WORDS or any(
                word.casefold() in GENERIC_POSSESSIVE_WORDS for word in words
            ):
                continue
            if named_company.casefold() not in evidence_companies:
                return f"requested_company_not_in_evidence:{named_company}"

    requested_years = list(
        dict.fromkeys(
            f"20{match.group(1)}" if len(match.group(1)) == 2 else match.group(1)
            for match in REQUESTED_FISCAL_YEAR_RE.finditer(query)
        )
    )

    def group_supports_year(group: list[RetrievedChunk], full_year: str) -> bool:
        short_year = full_year[-2:]
        metadata_years = {
            re.sub(r"[^a-z0-9]", "", str(chunk.fiscal_year).casefold())
            for chunk in group
            if chunk.fiscal_year
        }
        if metadata_years:
            return bool(
                {short_year, full_year, f"fy{short_year}", f"fy{full_year}"} & metadata_years
            )
        combined_text = " ".join(chunk.text for chunk in group).casefold()
        return full_year in combined_text or f"fy{short_year}" in combined_text

    if len(requested_years) == 1:
        full_year = requested_years[0]
        for company, group in evidence_groups:
            if not group_supports_year(group, full_year):
                suffix = f":{company}" if company else ""
                return f"requested_fiscal_year_not_in_evidence:{full_year}{suffix}"
    elif requested_years:
        # Multiple years may be independently scoped ("Apple FY25 vs NVIDIA
        # FY26"). Require every requested year somewhere in the total evidence
        # without incorrectly requiring every year in every company group.
        all_evidence = [chunk for _, group in evidence_groups for chunk in group]
        for full_year in requested_years:
            if not group_supports_year(all_evidence, full_year):
                return f"requested_fiscal_year_not_in_evidence:{full_year}"

    requested_segments = list(
        dict.fromkeys(match.group(1).casefold() for match in REQUESTED_SEGMENT_RE.finditer(query))
    )
    segment_groups = evidence_groups if len(requested_segments) == 1 else [(None, chunks)]
    for label in requested_segments:
        meaningful_words = [word for word in label.split() if word not in GENERIC_SEGMENT_WORDS]
        if not meaningful_words:
            continue
        for company, group in segment_groups:
            combined_text = " ".join(chunk.text for chunk in group).casefold()
            if f"{label} segment" not in combined_text:
                suffix = f":{company}" if company else ""
                return f"requested_segment_not_in_evidence:{label}{suffix}"
    return None


def _retrieve_and_rerank(
    query: str,
    top_k: int,
    companies: list[str] | None,
    fiscal_year: str | None,
    doc_type: str | None,
    query_embedding: list[float] | None = None,
) -> tuple[list[RetrievedChunk], float]:
    """`query_embedding`, if given, is the already-computed embedding for
    `query` (e.g. reused from a semantic cache lookup on the same text) --
    `retrieve_hybrid` reuses it across every company in a comparison query
    instead of re-embedding the identical text once per company. Not reused
    for the cleaned-query retry below: that's different text, so it needs
    its own embedding regardless.
    """
    retrieval_start = time.perf_counter()
    candidates = retrieve_hybrid(
        query,
        top_k=settings.RERANK_CANDIDATE_K,
        companies=companies,
        fiscal_year=fiscal_year,
        doc_type=doc_type,
        query_embedding=query_embedding,
    )
    comparison_mode = _is_explicit_comparison(companies)
    logger.info(
        "Retrieval selection: comparison_mode=%s requested_companies=%s candidate_count=%d",
        comparison_mode,
        _normalize_companies(companies),
        len(candidates),
    )
    chunks, top_score = _rerank_and_gate(query, candidates, top_k, companies)

    if not chunks:
        # Nothing cleared the gate -- try once more with a trailing
        # evaluative clause stripped, in case that's what tanked the score
        # (see TRAILING_EVALUATIVE_CLAUSE_RE). If there's nothing to strip,
        # or the retry is also empty, this falls straight through to the
        # existing NO_RELEVANT_CONTEXT_ANSWER behavior.
        cleaned_query = TRAILING_EVALUATIVE_CLAUSE_RE.sub("", query).rstrip()
        retry_top_score = None
        if cleaned_query and cleaned_query != query:
            retry_candidates = retrieve_hybrid(
                cleaned_query,
                top_k=settings.RERANK_CANDIDATE_K,
                companies=companies,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
            )
            chunks, retry_top_score = _rerank_and_gate(
                cleaned_query, retry_candidates, top_k, companies
            )
        logger.info(
            "Rerank gate returned no chunks for query_variant=original "
            "(top_score=%.4f); cleaned_retry_used=%s retry_top_score=%s",
            top_score,
            bool(cleaned_query and cleaned_query != query),
            retry_top_score,
        )

    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
    return chunks, retrieval_latency_ms


def _cached_or_embed(
    query: str,
    cache_key: str,
    companies: list[str] | None,
    fiscal_year: str | None,
    doc_type: str | None,
    model: str,
    history: list[HistoryTurn] | None,
    top_k: int,
    use_cache: bool = True,
) -> tuple[object | None, list[float] | None]:
    """Exact-cache lookup, then (only on a miss) a semantic-cache lookup that
    embeds `query` exactly once and reuses that vector.

    Returns `(cached_row_or_None, query_embedding_or_None)`. The embedding is
    None whenever it was never computed: a history-bearing query skips the
    cache entirely (see `generate_answer`'s docstring), and an exact-cache
    hit never needs one. Shared by both `generate_answer` and
    `generate_answer_stream` so their cache/embedding behavior can't drift
    apart from each other.

    `use_cache=False` (e.g. the eval harness re-testing generation quality,
    not cache plumbing) skips both cache lookups entirely -- same as a
    history-bearing query -- but still computes and returns the embedding
    so retrieval doesn't have to.
    """
    if history:
        return None, None
    if not use_cache:
        return None, embed_query(query)
    cached = get_cached(cache_key)
    if cached is not None:
        return cached, None
    query_embedding = embed_query(query)
    cached = _safe_semantic_cached(
        query, companies, fiscal_year, doc_type, model, query_embedding, top_k
    )
    return cached, query_embedding


def generate_answer(
    query: str,
    top_k: int = settings.DEFAULT_TOP_K,
    companies: list[str] | None = None,
    fiscal_year: str | None = None,
    doc_type: str | None = None,
    history: list[HistoryTurn] | None = None,
    client: GeminiChatClient | None = None,
    session_id: str | None = None,
    use_cache: bool = True,
) -> AnswerResult:
    """Non-streaming: cache lookup -> hybrid retrieve -> rerank -> prompt ->
    generate -> parse citations -> cache store.

    `history`, if given, is prior conversation turns included in the prompt
    for continuity; retrieval always runs fresh off just `query`.

    `use_cache=False` bypasses both the read (exact + semantic) and the
    write at the end -- for a caller re-testing generation/retrieval
    quality itself (the eval harness, `eval/run_eval.py`) rather than
    exercising the cache, a stale or freshly-written cache entry would
    silently make every subsequent identical run just replay the first
    run's answer instead of actually re-generating it.
    """
    total_start = time.perf_counter()
    _validate_top_k(top_k)
    chat_client = client or GeminiChatClient()
    cache_key = make_cache_key(
        query, companies, fiscal_year, doc_type, chat_client.model, top_k=top_k
    )

    log_kwargs = dict(
        query_text=query,
        session_id=session_id,
        companies=companies,
        top_k=top_k,
        embedding_model=settings.LOCAL_EMBEDDING_MODEL,
    )

    try:
        # A query with conversation history is inherently context-dependent
        # ("what about last year?"), so a bare cache lookup keyed only on the
        # literal query text would return a stale/wrong answer from a
        # different conversation. Only cache turn-independent queries.
        cached, query_embedding = _cached_or_embed(
            query,
            cache_key,
            companies,
            fiscal_year,
            doc_type,
            chat_client.model,
            history,
            top_k,
            use_cache,
        )
        if cached is not None:
            answer = _answer_from_cache(cached, total_start)
            _safe_record_query_log(
                result=answer, cache_hit=True, cost_usd=answer.cost_usd, **log_kwargs
            )
            return answer

        chunks, retrieval_latency_ms = _retrieve_and_rerank(
            query, top_k, companies, fiscal_year, doc_type, query_embedding
        )
        if not chunks:
            answer = _no_relevant_context_answer(
                chat_client.model, retrieval_latency_ms, total_start
            )
            _safe_record_query_log(result=answer, cache_hit=False, **log_kwargs)
            return answer
        unsupported_scope_reason = _unsupported_scope_reason(query, chunks, companies)
        if unsupported_scope_reason is not None:
            logger.info("Evidence scope refusal: reason=%s", unsupported_scope_reason)
            answer = _no_relevant_context_answer(
                chat_client.model, retrieval_latency_ms, total_start
            )
            _safe_record_query_log(result=answer, cache_hit=False, **log_kwargs)
            return answer

        messages = build_messages(
            query,
            chunks,
            history=_history_as_messages(history),
            comparison_mode=_is_explicit_comparison(companies),
        )

        generation_start = time.perf_counter()
        result = chat_client.chat(messages)
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000
        total_latency_ms = (time.perf_counter() - total_start) * 1000

        clean_text, entries = _split_answer_and_entries(result.content)
        citations = _resolve_citations(entries, chunks, clean_text)
        answer = AnswerResult(
            answer_text=clean_text,
            citations=citations,
            model=result.model,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            cost_usd=estimate_cost_usd(
                result.model, result.prompt_tokens, result.completion_tokens
            ),
        )

        if not history and use_cache:
            _safe_store_cached(
                cache_key,
                query,
                result.model,
                clean_text,
                [asdict(c) for c in citations],
                answer.retrieved_chunk_ids,
                result.prompt_tokens,
                result.completion_tokens,
                result.total_tokens,
                companies=companies,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
                query_embedding=query_embedding,
                top_k=top_k,
            )
    except Exception as e:
        _safe_record_query_log(
            result=None,
            error=str(e),
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
            **log_kwargs,
        )
        raise

    _safe_record_query_log(result=answer, cache_hit=False, cost_usd=answer.cost_usd, **log_kwargs)
    return answer


def generate_answer_stream(
    query: str,
    top_k: int = settings.DEFAULT_TOP_K,
    companies: list[str] | None = None,
    fiscal_year: str | None = None,
    doc_type: str | None = None,
    history: list[HistoryTurn] | None = None,
    client: GeminiChatClient | None = None,
    session_id: str | None = None,
) -> Iterator[str | AnswerResult]:
    """Streaming variant: yields raw text deltas, then a final AnswerResult.

    Callers should distinguish chunks by type: `str` deltas are live tokens
    (with the trailing ```citations block, if present in a delta, still
    included raw -- callers needing clean markdown should use the final
    AnswerResult.answer_text instead of concatenating deltas past the fence).
    """
    total_start = time.perf_counter()
    _validate_top_k(top_k)
    chat_client = client or GeminiChatClient()
    cache_key = make_cache_key(
        query, companies, fiscal_year, doc_type, chat_client.model, top_k=top_k
    )

    log_kwargs = dict(
        query_text=query,
        session_id=session_id,
        companies=companies,
        top_k=top_k,
        embedding_model=settings.LOCAL_EMBEDDING_MODEL,
    )

    try:
        cached, query_embedding = _cached_or_embed(
            query,
            cache_key,
            companies,
            fiscal_year,
            doc_type,
            chat_client.model,
            history,
            top_k,
        )
        if cached is not None:
            answer = _answer_from_cache(cached, total_start)
            yield answer.answer_text
            _safe_record_query_log(
                result=answer, cache_hit=True, cost_usd=answer.cost_usd, **log_kwargs
            )
            yield answer
            return

        chunks, retrieval_latency_ms = _retrieve_and_rerank(
            query, top_k, companies, fiscal_year, doc_type, query_embedding
        )
        if not chunks:
            answer = _no_relevant_context_answer(
                chat_client.model, retrieval_latency_ms, total_start
            )
            yield answer.answer_text
            _safe_record_query_log(result=answer, cache_hit=False, **log_kwargs)
            yield answer
            return
        unsupported_scope_reason = _unsupported_scope_reason(query, chunks, companies)
        if unsupported_scope_reason is not None:
            logger.info("Evidence scope refusal: reason=%s", unsupported_scope_reason)
            answer = _no_relevant_context_answer(
                chat_client.model, retrieval_latency_ms, total_start
            )
            yield answer.answer_text
            _safe_record_query_log(result=answer, cache_hit=False, **log_kwargs)
            yield answer
            return

        messages = build_messages(
            query,
            chunks,
            history=_history_as_messages(history),
            comparison_mode=_is_explicit_comparison(companies),
        )

        full_content = []
        generation_start = time.perf_counter()
        done: StreamDone | None = None
        for event in chat_client.chat_stream(messages):
            if isinstance(event, StreamToken):
                full_content.append(event.content)
                yield event.content
            elif isinstance(event, StreamDone):
                done = event
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000
        total_latency_ms = (time.perf_counter() - total_start) * 1000
        raw_text = "".join(full_content)

        clean_text, entries = _split_answer_and_entries(raw_text)
        citations = _resolve_citations(entries, chunks, clean_text)
        prompt_tokens = done.prompt_tokens if done else 0
        completion_tokens = done.completion_tokens if done else 0
        answer = AnswerResult(
            answer_text=clean_text,
            citations=citations,
            model=done.model if done else chat_client.model,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=done.total_tokens if done else 0,
            retrieved_chunk_ids=[c.chunk_id for c in chunks],
            cost_usd=estimate_cost_usd(
                done.model if done else chat_client.model, prompt_tokens, completion_tokens
            ),
        )

        if not history:
            _safe_store_cached(
                cache_key,
                query,
                answer.model,
                clean_text,
                [asdict(c) for c in citations],
                answer.retrieved_chunk_ids,
                answer.prompt_tokens,
                answer.completion_tokens,
                answer.total_tokens,
                companies=companies,
                fiscal_year=fiscal_year,
                doc_type=doc_type,
                query_embedding=query_embedding,
                top_k=top_k,
            )
    except Exception as e:
        _safe_record_query_log(
            result=None,
            error=str(e),
            total_latency_ms=(time.perf_counter() - total_start) * 1000,
            **log_kwargs,
        )
        raise

    _safe_record_query_log(result=answer, cache_hit=False, cost_usd=answer.cost_usd, **log_kwargs)
    yield answer
