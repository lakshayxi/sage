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
  reference project's llama3.1-specific observations): verified live on 2026-07-18 against 25
  real, successful `GeminiChatClient().chat()` calls (model `gemini-flash-lite-latest`) spanning
  single-company, genuine multi-company comparison-mode (2 distinct companies in context, real
  `### Company` section headers), long/detailed answers, many-citation answers, malformed/vague
  questions, and multi-turn follow-ups with conversation history. All 25 raw outputs matched
  `CITATIONS_FENCED_RE` (the happy path) on the first try — none of the fallback regexes
  (`CITATIONS_UNFENCED_RE`, `UNCLOSED_CITATIONS_RE`, `BARE_TRAILING_FENCE_RE`) were ever
  exercised, and no leaked fence/JSON fragments appeared in the cleaned answer text. Gemini also
  held up against adversarial-looking prose that could confuse a naive parser (multi-number
  citation brackets like `[2, 3, 5]` inside bullet points, markdown headers/bold immediately
  before the fence). Two of these real captured outputs (the comparison-mode one and the
  long/bulleted one) are now locked in as regression fixtures in
  `tests/test_answer_engine.py`. The fallback parsers remain defensively in place (unverified
  Gemini *failure* output is still just absence of evidence, not evidence of absence, and the
  parsers are cheap insurance), but the happy path is now confirmed to be what Gemini reliably
  produces, not just what was hoped for.
- Exact free-tier RPM/TPM/RPD numbers for Gemini *chat* are still unknown (would need the AI
  Studio dashboard) — moot for embeddings now that they're local (see update below).
- ~~No retrieval-quality eval harness~~ — built in a later follow-up session, see "Update: eval
  harness" below.
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
  **Update:** implemented in a later follow-up session — `sage/embed/local_embedder.py` gained
  `embed_query()` (prefixes then delegates to `embed_text`); `sage/retrieval/retriever.py` and
  both query-embedding call sites in `sage/generation/cache.py` now call `embed_query` instead
  of `embed_text`. `embed_text`/`embed_texts` are unchanged (still unprefixed) for
  `sage/ingest/pipeline.py`'s passage embedding. Any query embeddings already sitting in a
  dev-local `finresearch_query_cache` Chroma collection from before this change were written
  unprefixed, so they're now slightly inconsistent with newly-prefixed query lookups — not a
  crash, just degraded semantic-cache recall until those entries expire via
  `CACHE_TTL_SECONDS` or the collection is rebuilt; no migration written for this local dev
  data.
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

## Update: basic eval harness (same repo, later follow-up session)

**Objective:** address the "no retrieval-quality eval harness" TODO above, adapted from the
reference project's `eval/` (hand-curated Q&A set + DeepEval/local-judge pattern) but
reimplemented for Sage's actual architecture, not ported.

**What was built:**
- `eval/dataset.py` — 14 hand-curated questions against the real ingested corpus
  (`Apple_FY25_filing.pdf`, `Microsoft_FY25_filing.pdf`, `NVIDIA_FY26_filing.pdf`). Every expected
  figure was verified by reading the actual filing text via PyMuPDF (Consolidated
  Statements of Income/segment tables), not recalled from memory or guessed. Mix: 9
  single-company factual lookups (revenue/net income/R&D/segment), 3 multi-company
  comparison-mode questions (`companies` with 2-3 entries, exercising
  `retrieve_hybrid`'s round-robin merge path), 2 deliberately out-of-corpus questions (Tesla,
  Amazon — no such filing is ingested) to check the relevance gate declines rather than
  hallucinates.
- `eval/scoring.py` — deterministic heuristic scorer, **not** an LLM judge. Two independent
  checks per item: `correct` (a dollar figure within 2% tolerance of the expected value, via a
  regex that normalizes "$416,161 million" / "$416.2 billion" / raw-dollar phrasing to a common
  millions unit; or, for out-of-corpus items, refusal-phrase detection) and `grounded` (citation
  filenames actually belong to the company/companies the question is about; empty for
  out-of-corpus items). `passed = correct and grounded`. Chosen over an LLM judge (the reference
  project's DeepEval + local-Ollama-judge pattern) because: (1) Sage has no free/unmetered local
  judge the way Ollama was — an LLM judge here means *doubling* live Gemini calls against a
  free-tier key already documented above (Issue 2) to have hit a hard `429` wall once; (2) this
  dataset's answers are objectively-correct single numbers, which a heuristic checks more
  reliably than a judge model can; (3) it avoids DeepEval's ~24-package dependency footprint for
  a harness explicitly scoped as basic/correctness-only. See `eval/scoring.py`'s module docstring
  for the full reasoning.
- `eval/run_eval.py` — runs each item through the real `generate_answer()`, scores it, writes a
  timestamped CSV + Markdown report to `eval/results/` (gitignored, like the reference project's).
  A 1s courtesy pause between items guards against bursting the free-tier Gemini rate limit.
- `sage-eval` console script (`pyproject.toml`: `eval*` added to `packages.find`, matching the
  reference project's own `eval*` include) — `.venv/bin/sage-eval [--limit N]`, or
  `.venv/bin/python -m eval.run_eval` without reinstalling.
- `tests/test_eval_scoring.py` — 21 no-network unit tests on the pure scoring functions
  (`extract_amounts_millions`, `numeric_match`, `keywords_match`, `refusal_detected`,
  `allowed_filenames_for`, `score_item`), deliberately not importing `sage.generation` to stay
  fast and import-safe. `eval/run_eval.py` itself is inherently live-network and isn't exercised
  by `pytest tests/` (172 tests total now, unchanged pass rate, `ruff check .` /
  `ruff format --check .` clean).

**Real run results (2026-07-18, live `GEMINI_API_KEY`, model `gemini-flash-lite-latest`, real
corpus, `.venv/bin/python -m eval.run_eval`, no `--limit`):**

**13/14 passed**, 68.9s wall time, $0.00 (still under free-tier quota), 0/14 cache hits (fresh
queries). All 9 single-company and all 3 comparison-mode questions passed on the first live run,
including `cross-netincome-leader` (NVIDIA has the highest net income of the three despite Apple
having the highest revenue — the model got the correct, non-obvious ranking rather than just
naming the biggest company) and `cross-operating-income` (NVIDIA $130,387M vs. Microsoft
$128,528M, a deliberately close ~1.4% gap used as a harder precision stress test).

The one failure, `unans-tesla-revenue`, is a genuine, useful finding, not an eval-harness bug:
asked "What was Tesla's total revenue in fiscal year 2025?" (no company filter — Tesla isn't
ingested), the model correctly refused ("The provided context does not contain information
regarding Tesla's total revenue for fiscal year 2025; the document provided pertains to Apple
Inc.") — no hallucinated figure, `correct=True`. But the answer still carried a resolved citation
to `Apple_FY25_filing.pdf`, `grounded=False`. This shows the rerank gate (`MIN_RERANK_SCORE`)
let at least one Apple chunk through for an unfiltered Tesla query — refusal safety here came
from the LLM's own judgment during generation, not from the pre-generation relevance gate, and
that Apple chunk still got attached as a citation on an answer that explicitly says it can't
answer. Not a hallucination risk (the visible answer text is honest), but a real UI/UX
inconsistency: citations shouldn't render on a declined-to-answer response. **Fixed**:
`_resolve_citations` (`sage/generation/answer_engine.py`) now cross-checks each citation entry's
`n` against the actual inline `[n]` markers in the answer text (parsing every number out of every
`[...]` bracket group, so multi-number brackets like `[2, 3, 5]` still resolve correctly) and
drops any entry that was never really referenced in the prose — live-verified this exact Tesla
query now returns an empty `citations` array.
`unans-amazon-netincome` (also out-of-corpus, otherwise identical shape) passed cleanly with
zero citations, so this isn't systematic — it's specific to whatever scored that one Apple chunk
above `MIN_RERANK_SCORE` for the Tesla query's embedding/BM25 signal.

**How to reproduce:** `.venv/bin/sage-eval` (or `--limit N` for a faster/cheaper subset). Requires
a live `GEMINI_API_KEY` and the corpus already ingested (`sage ingest`) — costs ~14 real Gemini
chat calls per full run, so this is meant for occasional regression checks, not a tuning loop.

## Update: explicit comparison rerank-gate correction (2026-07-20)

The comparison candidate-balancing fix did not solve absolute-score collapse. The answer engine
still inferred comparison mode from candidate metadata and filtered every company with
`MIN_RERANK_SCORE=0.1`. Real-corpus measurements showed valid three-company revenue evidence below
the cutoff for every company, while Apple FY2020 and NVIDIA Automotive distractors scored 0.94 and
0.998. Lowering the global threshold was therefore rejected: the one-label BGE sigmoid output is
bounded but not a calibrated probability or factual-answerability classifier.

The implemented design:

- Explicit comparison mode comes only from two or more normalized caller-selected companies.
  Unfiltered mixed-company candidate pools and single-company requests retain the ordinary global
  threshold path and cleaned-query retry.
- Hybrid retrieval runs once with the original user query and its already-shared embedding. Each
  requested company is reranked once over its existing candidate pool with a deterministic
  company-local fact query. Up to `top_k` chunks are selected per company by rank, without an
  absolute comparison cutoff; every requested group must be nonempty or the whole comparison
  refuses. The original query remains unchanged for Gemini.
- Reranking now returns scored copies instead of mutating `RetrievedChunk.score`. Exact and
  semantic cache scope now includes `top_k`. Streaming and non-streaming continue through the same
  retrieval-selection and deterministic scope-validation helpers.
- Named fiscal years and singular `X segment` requests receive narrow lexical evidence checks
  before generation. This blocks the observed FY2020 distractor and prevents NVIDIA's Automotive
  end-market revenue from being mislabeled as a reportable segment.
- The CLI now forwards its existing `--fiscal-year` and `--doc-type` flags. Gemini generation uses
  temperature 0 for reproducible financial extraction/citation formatting.
- The eval set now has 19 items, including the exact long revenue-ranking reproduction.
  Comparison scoring requires every company/amount/source, recall checks gold evidence for all
  companies, and `--id` supports targeted live reruns.

**Real-corpus retrieval-only results:** all four target comparisons selected exactly five chunks
per requested company and included every gold amount. Warm-process retrieval latency was
8.20s (three-company revenue leader), 7.20s (net-income leader), 4.77s (two-company operating
income), and 7.21s (full revenue ranking). Explicit Tesla/Amazon/FY20 filters selected zero chunks
in 6.5–7.9ms. The exact live ranking completed in 7.63s (5.57s retrieval, 1.71s generation) and
cited the correct Apple FY25, Microsoft FY25, and NVIDIA FY26 chunks.

**Final validation:** 261 tests passed; Ruff lint/format and compileall passed. The final uncached
live eval, rerun after adding company/amount association scoring, passed 19/19 in 86.7s with
answerable gold recall 15/15, answerable citation mapping 15/15, and citation-text support 19/19
(including clean no-citation refusals). Generated eval reports remain ignored runtime artifacts.
