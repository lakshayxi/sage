"""Cross-encoder reranking — the final narrowing step after hybrid retrieval
fetches a wide candidate set.

Retrieval (vector + BM25 fusion) is cheap but coarse; a cross-encoder scores
the query against each candidate's full text jointly, which is more accurate
but too slow to run over the whole corpus -- hence running it only over the
candidates already narrowed by `retrieve_hybrid`.

Kept identical to the reference project (same model, same sigmoid-scored
[0, 1] range, same `MIN_RERANK_SCORE` gate applied by the caller) -- this
step operates purely on already-retrieved chunk text and is unaffected by
Sage's multi-company retrieval change or its Gemini-only generation layer.
"""

import threading

from sentence_transformers import CrossEncoder

from config import settings
from sage.retrieval.retriever import RetrievedChunk

_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        # FastAPI's sync `def` route handlers run in a thread pool, so
        # concurrent requests right after a cold start can otherwise both
        # see `_model is None` and each build a full model -- double-checked
        # locking so only the first one actually constructs it.
        with _model_lock:
            if _model is None:
                _model = CrossEncoder(settings.RERANKER_MODEL)
    return _model


def rerank(query_text: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Score each candidate against the query and return the top_k, best first.

    `score` on the returned chunks is overwritten with the cross-encoder's
    relevance score (higher is better), replacing whatever hybrid RRF score
    it carried in from retrieval.
    """
    if not candidates:
        return []
    model = _get_model()
    pairs = [(query_text, c.text) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)

    reranked = []
    for chunk, score in ranked[:top_k]:
        chunk.score = float(score)
        reranked.append(chunk)
    return reranked
