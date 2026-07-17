"""Local sentence-transformers embeddings (BAAI/bge-small-en-v1.5 by default).

Replaces the earlier Gemini-embeddings path (`GeminiEmbedder`, removed) after
live testing hit a hard wall on Gemini's free embedding quota: a single
batched `embed_content` call over 30+ chunk texts from one real 10-K PDF hit
`429 RESOURCE_EXHAUSTED`, and the quota didn't recover -- not even after
generating a supposedly-new API key (almost certainly still the same
underlying GCP project). Embeddings are needed not just once at ingest time
but on every single query too (retrieval embeds the query text every call),
so this quota risk was structural, not a one-off. Running embeddings locally
removes it entirely. Generation stays on Gemini via API key -- that path was
separately live-validated and has no equivalent quota problem in practice
(see docs/llm-engineer-work-log.md for the full story).

Model choice: BAAI/bge-small-en-v1.5 (384-d output). Same vendor as the
cross-encoder reranker already in use (BAAI/bge-reranker-base) and
well-regarded for retrieval; small/fast enough (~130MB) to load and run
in-process alongside the reranker at no new heavy-dependency cost --
sentence-transformers (and its PyTorch dependency) is already installed
for the reranker. Loaded via sentence_transformers.SentenceTransformer,
mirroring the reranker's lazy-singleton pattern in
sage/retrieval/reranker.py exactly: no external embedding server (e.g. no
Ollama) needed, just an in-process model load on first use.

Batching: sentence-transformers' own `.encode()` handles batching internally
(default `batch_size=32`), so unlike the Gemini path there's no need to
chunk the input list ourselves to stay under a remote rate limit -- this is
a local, synchronous computation with no quota/429 concern at all, so the
`call_with_retry` wrapper from sage/retry.py is deliberately not used here
(it's now scoped to GeminiChatClient only, which still legitimately talks
to a remote API).

NOT implemented: BGE models are documented by BAAI to benefit from an
asymmetric "Represent this sentence for searching relevant passages: "
instruction prefix on the *query* side only (not on indexed passages) for
retrieval tasks -- typically a measurable recall improvement. This was
deliberately left out to keep `embed_text`/`embed_texts` a drop-in
replacement with the exact same call shape for every existing caller
(ingest, retrieval, semantic cache), matching how this swap was scoped.
Worth revisiting as a retrieval-quality follow-up: would need query-embedding
call sites (sage/retrieval/retriever.py, sage/generation/cache.py) to route
through a distinct query-prefixed function while document/passage embedding
(sage/ingest/pipeline.py) stays unprefixed.

Dimension note: switching embedding models changes vector dimensionality
(this model: 384-d, vs. the earlier Gemini path's 768-d). Any existing
Chroma collection built under the old model must be rebuilt (delete
data/chroma/ and re-ingest), not reused -- Chroma will error or silently
mismatch on a dimension change otherwise.
"""

import threading

from sentence_transformers import SentenceTransformer

from config import settings

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # See sage/retrieval/reranker.py's identical pattern: FastAPI's sync
        # route handlers run in a thread pool, so an unlocked check here lets
        # concurrent requests right after a cold start each build (and
        # discard) their own full model.
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(settings.LOCAL_EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()
