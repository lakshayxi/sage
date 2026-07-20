# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sage — a Gemini-based financial research app: PDF ingestion → local `sentence-transformers` embeddings → Chroma vector store → hybrid BM25+vector retrieval (RRF fusion) → cross-encoder reranking → Gemini-generated cited answers → FastAPI backend with SSE-style streaming → React/TypeScript frontend. This is a from-scratch reimplementation of the same approach as a sibling lab project, not a port. Full build history, technical decisions, and known issues are in `docs/llm-engineer-work-log.md` — check it before assuming something is a new bug.

## Environment setup

- Requires a `GEMINI_API_KEY` in `.env` (repo root) — generation is Gemini-only, nothing answers without it. Embeddings and reranking are fully local (`sentence-transformers`), no network dependency there.
- `config/settings.py` explicitly calls `load_dotenv(BASE_DIR / ".env")` — this was itself a real bug fix (the key used to sit on disk but never get loaded into the process env, producing a confusing generic error).
- Python 3.11 is the production/CI target (`requirements-lock.txt` is generated against 3.11, CI pins `python-version: "3.11"`, and the Docker image is `python:3.11-slim-bookworm`) — install from the lock file for a reproducible environment: `pip install -r requirements-lock.txt && pip install -e . --no-deps`.

## Running

- Server: `.venv/bin/uvicorn api.main:app --reload`, then open `http://localhost:8000/`
- If the server is backgrounded without `--reload`, it keeps serving stale code after a fix lands on disk with no error — restart before re-testing a fix.
- CLI: `.venv/bin/sage ingest|ask|conversations`
- Tests: `.venv/bin/python -m pytest tests/` — no live Gemini/network needed (the Gemini client is fully faked via `tests/fakes.py`), but the first run downloads the real `BAAI/bge-small-en-v1.5` model (~130MB, cached after) since one embedder test isn't mocked.
- Eval harness: `.venv/bin/sage-eval [--limit N] [--id ITEM_ID]` — scores the RAG pipeline against `eval/dataset.py`'s hand-curated ~19-item Q&A set (check the file for the exact count, it will drift). Every run passes `use_cache=False` to `generate_answer` so cached answers can't mask a same-day prompt/parsing fix as passing. Requires a live `GEMINI_API_KEY` and costs real Gemini calls; not part of `pytest tests/`.
- Frontend: `cd frontend && npm ci` then:
  - `npm run dev` for local dev
  - `npm run build` — runs `tsc -b` (typecheck) then `vite build`, producing `frontend/dist` (served by the same FastAPI process in prod)
  - `npm run test` — Vitest (`vitest run`); covers things like the `/chat/stream` fetch/ReadableStream parser (frame reassembly across chunk/UTF-8 boundaries, a stream that ends without a terminal event still resolving instead of hanging)
  - `npm run lint` — runs `oxlint`, not eslint

## CI

`.github/workflows/ci.yml` runs on push to `main` and on PRs, as two jobs:

- **backend** — installs from `requirements-lock.txt` (not `pip install -e ".[dev]"` directly against `pyproject.toml`'s `>=` floors, for a reproducible resolution), then `ruff check .`, `ruff format --check .`, and `python -m pytest tests/`. No `GEMINI_API_KEY` is needed since the Gemini client is faked; this does not run the live eval harness, only `eval/run_eval.py`'s network-free scoring-logic unit tests as part of the same `pytest tests/` invocation.
- **frontend** — `npm ci`, then `npm run lint`, `npm run test`, `npm run build` (typecheck + build in one step).

No other CI is configured — these two jobs plus local `ruff`/eval runs are the full set of gates before considering a change done.

## Architecture notes

- Embeddings run locally (`sage/embed/local_embedder.py`) but generation stays on Gemini — a deliberate split after Gemini's free embedding tier hit a hard, non-recoverable quota wall. Query text gets BGE's asymmetric instruction prefix (`embed_query()`); indexed passage text does not (`embed_text()`/`embed_texts()`) — don't swap these, or retrieval quality degrades silently with no crash.
- Two process-wide singletons need an explicit reset in tests, not just settings monkeypatching: `sage/db/database.py`'s engine (`reset_engine()`) and `sage/retrieval/store.py`'s Chroma client (set `_client = None`). Patching only `settings.SQLITE_PATH`/`settings.CHROMA_DIR` without resetting the singleton still silently touches the old store — see `tests/conftest.py`'s `isolated_storage` fixture. The reranker and local embedder's own `_get_model()` singletons use double-checked locking and are warmed up at FastAPI startup (`api/main.py`'s `lifespan()`), so cold-start latency never lands on a real request.
- Retrieval is hybrid (BM25 + vector via RRF, `sage/retrieval/retriever.py:retrieve_hybrid`) then cross-encoder reranked (`sage/retrieval/reranker.py`, `BAAI/bge-reranker-base`). Single-company and unfiltered queries retain the `MIN_RERANK_SCORE` (0.1) gate and cleaned-query retry. Explicit comparisons are detected from the trimmed, case-insensitively deduplicated caller-provided `companies` list (resolved to corpus metadata casing and capped at 10 entries), reuse the original candidate pools, rerank once per requested company with a deterministic company-local fact query for recognized single-metric comparison shapes (otherwise the original query), and keep a validated `top_k` of 1–30 per company by rank. The engine, CLI, HTTP API, and cache all use that same meaning and range for `top_k`. Every requested group must be present. Raw sigmoid scores are not calibrated probabilities or answerability judgments; do not lower the global threshold to fix comparison score collapse.
- Citations are resolved positionally, not by trusting the model: `_resolve_citations()` (`sage/generation/answer_engine.py`) treats citation number `n` as always meaning `chunks[n - 1]` — the exact chunk shown to the model as `[1]`, `[2]`, ... in the prompt (`prompts.py`'s `build_context_block`). Any `chunk_id` the model echoes back in its citation JSON is never consulted to pick the chunk — trusting it would let the model remap `[1]` in the visible text to a different chunk than the one actually shown as `[1]`. An entry is also dropped if it's malformed, out of range, a duplicate, or not actually referenced inline as an `[n]` marker in the answer text.
- Uploads (`POST /documents/upload`, `api/routes/documents.py`) default to disabled — `settings.ALLOW_UPLOADS` must be explicitly set `true`, failing closed rather than open. When enabled, uploads are streamed to disk in bounded chunks (never fully buffered via `file.file.read()`) with a hard `MAX_UPLOAD_BYTES` cap enforced as bytes arrive, the request body additionally capped at the ASGI layer (`api.middleware.MaxUploadBodySizeMiddleware`, since Starlette's multipart parser has no size limit of its own), and the file verified to actually be a readable PDF (magic-byte check + PyMuPDF open) with a `MAX_UPLOAD_PAGES` ceiling before it reaches the ingest pipeline.
- Streaming (`POST /chat/stream`) is implemented with `fetch()` + a manually-read `ReadableStream` on the frontend (`frontend/src/api/chat.ts`), not the browser's native `EventSource` — `EventSource` can only issue GET requests and can't set custom headers, which this endpoint needs.
- Both query caches (exact-match SQLite + semantic Chroma, `sage/generation/cache.py`) expire after `CACHE_TTL_SECONDS` (6 hours) — long enough that re-testing a fix against a previously-asked query can still return a stale cached answer; use a reworded query or check the response's `cache_hit` field.
- `api/` (FastAPI layer) was built concurrently with `sage/` (backend/CLI core) in a separate session sharing the same `pyproject.toml`/`config/settings.py` — the `DEMO_ACCESS_KEY`/`ALLOW_UPLOADS`/`CHAT_RATE_LIMIT` settings belong to the API layer, not the CLI/backend core.
- API responses carry a `schema_version` field (`api/schemas.py`) — bump it on any breaking response-shape change.

## Conventions

- Linting/formatting: `ruff` only (no black/isort/mypy) — `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`. Line length 100, `py311` target, rule set `E, F, I, UP, B`.
- Modern PEP 604 union type hints (`str | None`, not `Optional[str]`) throughout.
- Three data-modeling idioms by layer, don't mix them: SQLAlchemy `DeclarativeBase` ORM for persistence (`sage/db/models.py`), Pydantic `BaseModel` for API request/response schemas (`api/schemas.py`), plain `@dataclass` for internal cross-module value objects that are neither persisted nor serialized over HTTP (`RetrievedChunk`, `Citation`, `AnswerResult`).
- Module/function docstrings are expected to explain *why*, often citing the specific bug or decision that motivated a choice, not just what the code does — match this in new code rather than writing purely descriptive docstrings.
- Commit style: all lowercase, no conventional-commit prefixes, one descriptive line by default, no body unless the change is genuinely substantial (then exactly one paragraph, no bullets) — the `git-committer` agent already encodes this.
