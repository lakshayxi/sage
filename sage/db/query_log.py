"""Query logging into the query_logs table -- observability, not the
request/response path itself (a logging failure here should never turn a
successful answer into an error for the caller; see answer_engine.py's
call sites, which wrap this in the same defensive try/except pattern used
for cache writes).
"""

from typing import TYPE_CHECKING

from sage.db.database import get_session
from sage.db.models import QueryLog

if TYPE_CHECKING:
    from sage.generation.answer_engine import AnswerResult


def record_query_log(
    query_text: str,
    result: "AnswerResult | None",
    session_id: str | None = None,
    companies: list[str] | None = None,
    top_k: int | None = None,
    embedding_model: str | None = None,
    error: str | None = None,
    total_latency_ms: float | None = None,
    cache_hit: bool = False,
    cost_usd: float = 0.0,
) -> int:
    """Persist a QueryLog row for this query and return its id.

    `total_latency_ms` is used only when `result` is None (a failed query),
    to record whatever elapsed time was captured before the failure.
    """
    session = get_session()
    try:
        log = QueryLog(
            session_id=session_id,
            query_text=query_text,
            model_name=result.model if result else None,
            embedding_model=embedding_model,
            companies=companies or [],
            retrieved_chunk_ids=result.retrieved_chunk_ids if result else None,
            top_k=top_k,
            retrieval_latency_ms=result.retrieval_latency_ms if result else None,
            generation_latency_ms=result.generation_latency_ms if result else None,
            total_latency_ms=result.total_latency_ms if result else total_latency_ms,
            prompt_tokens=result.prompt_tokens if result else None,
            completion_tokens=result.completion_tokens if result else None,
            total_tokens=result.total_tokens if result else None,
            cache_hit=cache_hit,
            cost_usd=cost_usd,
            answer_text=result.answer_text if result else None,
            error=error,
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return log.id
    finally:
        session.close()
