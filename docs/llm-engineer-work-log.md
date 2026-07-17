# Backend/CLI Work Log

Scope: ingestion, retrieval, generation, caching, data model, and CLI (`sage/`, `config/`).
Not covered here: the HTTP API layer (`api/`) — built separately/concurrently by another
session; this log doesn't speak to its decisions or issues.

## Objective

Build Sage's backend/CLI core from scratch, reimplementing (not copying) the proven
retrieval/ingestion approach from a reference local-first RAG project
(`/Users/shay/Desktop/projects/edge`), but swapped to Gemini-only generation/embeddings and
extended with multi-company comparison retrieval and conversation history — with no HTTP
API or frontend layer, and a CLI as the primary proof the pipeline works end-to-end.

## What was implemented

- **Ingestion** (`sage/ingest/`): PyMuPDF paragraph-reconstructing PDF loader, paragraph-boundary-aware
  chunker, filename-convention metadata parser, full ingest pipeline (load → chunk → SQLite → embed → Chroma).
- **Retrieval** (`sage/retrieval/`): hybrid BM25 + vector search fused via RRF, cross-encoder
  reranking (`BAAI/bge-reranker-base`) with a relevance gate. `retrieve_hybrid` takes
  `companies: list[str] | None` — for 2+ companies it runs one full retrieval per company and
  merges round-robin, guaranteeing balanced representation regardless of per-company corpus size.
- **Generation** (`sage/generation/`): `GeminiChatClient` wrapping `google-genai` directly (no
  provider abstraction — chat generation is Gemini-only by design), a single-company and a
  comparison-mode system prompt, citation-fence parsing with layered fallbacks, exact-match +
  semantic query cache, cost estimation, and the orchestration (`answer_engine.py`) tying
  cache → retrieve → rerank → generate → parse → cache-store → log together. Embeddings are
  **not** Gemini — see the "Local embeddings swap" update below.
- **Data model** (`sage/db/`): `Document`, `Chunk`, `QueryLog`, `QueryCache` (mirroring the
  reference) plus new `Conversation`/`Message` tables for resumable multi-turn sessions
  (linear history only, no branching).
- **CLI** (`sage/cli.py`): `sage ingest`, `sage ask` (repeatable `--company` for comparison
  mode, `--conversation-id`/`--new-conversation` for multi-turn), `sage conversations`.
- **Retry/backoff** (`sage/retry.py`): added mid-session after live testing surfaced a real
  failure mode — see Issues below.
- 105 tests (`tests/`), all passing offline (no `GEMINI_API_KEY`, network forcibly broken).

## Key decisions

- **Multi-company merge is round-robin, not fused RRF across companies.** RRF scores from
  independently-scored per-company retrieval runs aren't comparable; fusing them would let a
  larger per-company corpus silently dominate. Round-robin guarantees each requested company
  gets shelf space in the candidate set. See `sage/retrieval/retriever.py` module docstring.
- **`GeminiChatClient` takes an injectable `client` param**, defaulting to a real `genai.Client`
  only when omitted, mirroring the reference project's stub-provider pattern — tests inject a
  network-free fake (`tests/fakes.py`) and never hit the network.
- **History-bearing queries bypass the query cache entirely** (both read and write) — a cache
  keyed on literal query text would return a wrong answer for a context-dependent follow-up
  ("what about last year?").
- **Default chat model is `gemini-flash-latest`** (a rolling alias), not a pinned version —
  chosen after live testing showed pinned model names go stale faster than expected (see
  Issues). `gemini-3.1-flash-lite` / `gemini-3.5-flash` are documented alternatives if a
  cost-predictable pinned model is preferred over the alias.
- **Test isolation goes further than the reference project**: `tests/conftest.py` patches
  `SQLITE_PATH`, `CHROMA_DIR`, `RAW_DIR`, and `PROCESSED_DIR` to `tmp_path` and resets both
  process-wide singletons (DB engine, Chroma client) per test — added after catching a real
  leak (ingestion tests were writing debug JSON into the repo's real `data/processed/`).

## Issues encountered and how each was resolved

1. **WebFetched model names were already stale at build time.** `gemini-2.5-flash` and
   `gemini-2.5-flash-lite` (both shown as current in `ai.google.dev` docs on 2026-07-17)
   returned live `404 "no longer available to new users"`; `gemini-2.0-flash` was
   quota-blocked (429). *Resolved* by live-testing several candidates against a real key and
   switching the default to `gemini-flash-latest` (confirmed working), with
   `gemini-3.1-flash-lite`/`gemini-3.5-flash` noted as pinned alternatives.
2. **Free-tier embedding quota is tighter than assumed.** A single batched `embed_content`
   call over 30+ chunk texts from one real 10-K PDF (well under any documented request-size
   limit) hit `429 RESOURCE_EXHAUSTED`; batches of 20 succeeded. Repeated testing then
   exhausted the key's quota outright for the session (retries didn't recover it — looks like
   a daily, not per-minute, cap). *Resolved* by adding `sage/retry.py` (exponential backoff on
   429/500/503, non-retryable errors pass straight through) wired into both Gemini clients,
   and reducing the embedder's default batch size from 100 to 20 (empirically tuned, not a
   documented cap — see code comment).
3. **Ingestion tests were leaking real files into the repo's `data/processed/`.** Caught
   before it reached the final report. *Resolved* by extending `tests/conftest.py` to also
   patch `RAW_DIR`/`PROCESSED_DIR` to `tmp_path`.
4. **A second agent started building the HTTP API layer (`api/`) in this same repo mid-session**,
   adding `fastapi`/`uvicorn`/`slowapi` to `pyproject.toml` and its own test file. *Resolved* by
   leaving `api/`/`tests/test_api.py` untouched and confirming (via `git status` / diff
   inspection) that neither session's edits to the shared `pyproject.toml` or
   `config/settings.py` clobbered the other's.
5. **A live Gemini API key was pasted into chat in plaintext.** Treated as exposed by
   definition. *Resolved* by using it only as an in-process env var for smoke testing —
   never written to disk (`.env`, settings, or otherwise) — and flagging it to the user for
   rotation.

## Remaining limitations / TODOs

- Citation-fence **fallback parsers** (dropped fence, truncated JSON — carried over from the
  reference project's llama3.1-specific observations) are unverified against genuine Gemini
  failure output; only the well-formed happy path was observed live.
- Exact free-tier RPM/TPM/RPD numbers for Gemini *chat* are still unknown (would need the AI
  Studio dashboard) — moot for embeddings now that they're local (see update below).
- No retrieval-quality eval harness (hand-labeled Q&A set) — out of scope for this task; the
  reference project's `eval/` would be the template if one is wanted later.
- `chat_stream`'s retry only covers the call that opens the stream, not a 429 mid-stream
  (would risk duplicating already-yielded tokens) — see `sage/generation/gemini_client.py`.
- BGE's asymmetric query-instruction prefix not implemented for local embeddings — see the
  "Local embeddings swap" update below.

## Files created or modified

Everything under `sage/`, `config/`, `tests/`, `pyproject.toml`, `.gitignore` is new this
session (fresh repo). Notable individual files:

- `sage/retrieval/retriever.py` — multi-company hybrid retrieval + round-robin merge
- `sage/generation/{gemini_client,prompts,answer_engine,cache,cost}.py`
- `sage/embed/local_embedder.py` — local embedding path (see "Local embeddings swap" below)
- `sage/retry.py` — added mid-session (see Issue 2); later re-scoped to `GeminiChatClient` only
- `sage/db/{models,database,conversations,query_log}.py`
- `sage/ingest/{pdf_loader,chunker,metadata,pipeline}.py`
- `sage/cli.py`
- `config/settings.py` — updated mid-session (see Issue 1) to change `GEMINI_CHAT_MODEL` default
  and extend `MODEL_COST_PER_1K_TOKENS`; note the `DEMO_ACCESS_KEY`/`ALLOW_UPLOADS`/`CHAT_RATE_LIMIT`
  entries at the bottom of this file belong to the concurrent API-layer session, not this work.
- `tests/` — 105 tests, `conftest.py`, `fakes.py`

## Update: local embeddings swap (same repo, follow-up session)

**Objective:** Gemini's free embedding quota turned out to be a hard, non-recoverable wall —
`429 RESOURCE_EXHAUSTED` on the very first real ingest batch, and it persisted even after
generating a new API key (almost certainly still the same underlying GCP project). Embeddings
are needed on every query (not just at ingest), so this was a structural risk, not a one-off.
Directive: move embeddings to a local model; keep generation on Gemini (already validated live).

**What changed:**
- `sage/embed/gemini_embedder.py` (`GeminiEmbedder`) deleted. New
  `sage/embed/local_embedder.py` wraps `sentence_transformers.SentenceTransformer`
  (`BAAI/bge-small-en-v1.5`, 384-d), lazy-singleton-loaded exactly like the reranker's own
  `_get_model()` pattern in `sage/retrieval/reranker.py` — no new dependency, since
  sentence-transformers/PyTorch was already installed for the reranker.
- `embed_text`/`embed_texts` call shape preserved exactly, so `sage/ingest/pipeline.py`,
  `sage/retrieval/retriever.py`, and `sage/generation/cache.py` only needed a one-line import
  change each (`sage.embed.gemini_embedder` → `sage.embed.local_embedder`), no logic changes.
- `config/settings.py`: `GEMINI_EMBEDDING_MODEL`/`EMBEDDING_DIMENSIONS` replaced with
  `LOCAL_EMBEDDING_MODEL` (env: `SAGE_EMBEDDING_MODEL`). `GEMINI_CHAT_MODEL` untouched.
  `MODEL_COST_PER_1K_TOKENS`'s `gemini-embedding-001` entry removed (local embeddings are
  genuinely $0, not a fallback default).
- `embedding_model` observability field (`QueryLog`, via `answer_engine.py`'s two
  `record_query_log` call sites) now reports `settings.LOCAL_EMBEDDING_MODEL` instead of a
  stale Gemini model name — the same "observability field silently mis-reports which model
  actually ran" bug class flagged for the API layer, fixed here too.
- `sage/retry.py` (exponential backoff on 429/5xx) re-scoped to `GeminiChatClient` only — a
  local model call can't 429/have a quota, and its exception types don't match
  `google.genai.errors.APIError` anyway, so wrapping it there was both unnecessary and
  wouldn't have caught anything real. The artificial `_BATCH_SIZE=20` chunking loop is gone
  too — `SentenceTransformer.encode()` batches internally with no rate limit to chunk around.
- **Deliberately not implemented:** BGE's documented asymmetric query-instruction prefix
  (`"Represent this sentence for searching relevant passages: "`, recommended on the query
  side only, not on indexed passages) — would improve retrieval quality but requires
  query-embedding call sites to diverge from document-embedding call sites, which conflicts
  with keeping this a drop-in, zero-call-site-behavior-change swap. Flagged as a follow-up.
- Tests: `tests/test_gemini_embedder.py` deleted, replaced by `tests/test_local_embedder.py` —
  fast fake-`_get_model()` tests (mirroring `tests/test_reranker.py`'s existing convention)
  plus one deliberate real-model integration test (no API key or per-call quota risk locally,
  unlike the removed Gemini path, so there's no reason to fake the whole thing — only a
  one-time ~130MB model download, cached after). `tests/fakes.py`'s embed-related fake classes
  (`FakeModelsEmbed`, `FakeEmbedding`, etc.) removed as dead weight.

**Verification:**
- 105 tests passing (`pytest tests/ --ignore=tests/test_api.py`), `ruff check .` and
  `ruff format --check .` both clean, no `GEMINI_API_KEY` needed for any of it.
- **Live ingestion re-run against the real, previously-blocked corpus** (`Apple_FY25_filing.pdf`,
  `Microsoft_FY25_filing.pdf`, `NVIDIA_FY26_filing.pdf` — genuine SEC EDGAR filings, 77/137/113
  pages): succeeded completely in ~15s, zero embedding-related errors. 301 chunks across 3
  documents, all embedded and stored in Chroma (confirmed via `collection.count() == 301`).
- **Live `sage ask` end-to-end check did not complete**: the previously-working Gemini key now
  returns `401 UNAUTHENTICATED` (not `429` — a different failure mode), consistent with the key
  having been rotated/revoked, which was the recommended response to it having been pasted in
  plaintext chat earlier. Correctly *not* retried (401 isn't in `_RETRYABLE_STATUS_CODES`).
  Ingestion — the actual blocker this update targeted — is fully resolved and proven live; only
  the unrelated final generation-side smoke test is blocked on a fresh key.
