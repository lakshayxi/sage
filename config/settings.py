"""Centralized configuration for Sage.

Env-overridable with sensible local defaults, mirroring the pattern used by
the reference local-first project this was built alongside. Generation is
Gemini-only (via API key) — no local/cloud chat-provider switch or tier
system to configure. Embeddings, by contrast, run locally (sentence-
transformers) rather than through Gemini's embedding API — see
sage/embed/local_embedder.py's module docstring for why (Gemini's free
embedding quota turned out to be a hard, non-recoverable wall, and
embeddings are needed on every single query, not just at ingest time).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Nothing in this app previously loaded .env into the process environment --
# GEMINI_API_KEY silently resolved to "" for every real run (uvicorn, the
# CLI), which surfaced as a confusing "No API key was provided" ValueError
# from google-genai deep in the call stack, easily mistaken for a quota/auth
# problem with the key itself. load_dotenv() here is a no-op if the caller's
# shell already exports these vars (e.g. in a deployed container), so this is
# safe in both local and production environments.
load_dotenv(BASE_DIR / ".env")

# --- Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
DB_DIR = BASE_DIR / "db"
SQLITE_PATH = DB_DIR / "sage.db"

# --- Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# WebFetch against ai.google.dev/gemini-api/docs on 2026-07-17 showed
# gemini-2.5-flash as a current, stable model -- but a LIVE call against a
# real API key the same day got back "404 This model models/gemini-2.5-flash
# is no longer available to new users" (gemini-2.5-flash-lite: same 404;
# gemini-2.0-flash: quota-blocked, 429). Docs pages lag live model
# availability more than expected -- don't trust WebFetch alone for a model
# id that a real request will actually hit; live-verify before shipping.
# "gemini-flash-latest" (a rolling alias Google documents for "current
# recommended Flash model") worked live initially, but on 2026-07-18 started
# returning a live "503 UNAVAILABLE / high demand" on every call against the
# same key -- a Google-side capacity issue, not a quota/auth problem (ruled
# out by testing gemini-2.0-flash on the same key in the same breath, which
# correctly came back 429 quota-exhausted instead, a different error).
# "gemini-flash-lite-latest" (Google's rolling alias for its lite Flash tier)
# was live-verified working on the same key at the same time, so it's the
# default until flash-latest's availability is confirmed stable again -- at
# the cost of the concrete model (and therefore its exact price) potentially
# changing without a code change on Google's side.
GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-flash-lite-latest")

# --- Embeddings (local, not Gemini -- see sage/embed/local_embedder.py) ---
# BAAI/bge-small-en-v1.5, 384-d: same vendor as RERANKER_MODEL below, small
# and fast enough to load in-process with no new heavy dependency
# (sentence-transformers/PyTorch is already installed for the reranker).
# Switching this changes vector dimensionality -- any existing Chroma
# collection built under a different embedding model must be rebuilt
# (delete data/chroma/ and re-ingest), not reused.
LOCAL_EMBEDDING_MODEL = os.environ.get("SAGE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking ---
CHUNK_TOKENS = 650  # target chunk size (word-count approximation), ~500-800 window
CHUNK_OVERLAP_TOKENS = 120

# --- Retrieval ---
DEFAULT_TOP_K = 5
# Wide candidate set fetched by hybrid (vector + BM25) retrieval before the
# cross-encoder narrows it down to DEFAULT_TOP_K (or a caller-supplied top_k).
# In multi-company mode this is the per-company candidate count, not a global
# total -- see sage/retrieval/retriever.py.
RERANK_CANDIDATE_K = 30
RERANKER_MODEL = "BAAI/bge-reranker-base"
# Public comparison scope is intentionally bounded before the per-company
# retrieval loop: total work/context otherwise scales as companies × the
# candidate/selection budgets.
MAX_COMPARISON_COMPANIES = 10
MAX_COMPANY_FILTER_LENGTH = 200
# Minimum cross-encoder score for ordinary single-company and unfiltered
# queries. The sigmoid output is bounded but not a calibrated probability or
# factual-answerability classifier: Sage's measured answerable/unanswerable
# distributions overlap heavily. Explicit comparisons therefore use balanced
# company-local rank selection instead of this cutoff; see answer_engine.py.
# Do not lower this globally to fix a comparison false negative, because
# wrong-period and semantically-adjacent nonexistent facts can score far above
# it while valid comparison evidence can score below it.
MIN_RERANK_SCORE = 0.1

# --- Chroma ---
CHROMA_COLLECTION = "sage_chunks"
CHROMA_QUERY_CACHE_COLLECTION = "sage_query_cache"
# Max squared-L2 distance (Chroma's default hnsw:space) between a new query's
# embedding and the closest cached one to still count as a semantic cache
# hit. Unvalidated starting point (mirrors MIN_RERANK_SCORE's provenance) --
# chosen to only catch near-identical phrasing, not tuned against real query
# traffic yet.
SEMANTIC_CACHE_THRESHOLD = 0.05

# Max age (seconds) a cached answer stays valid before a lookup treats it as
# a miss. Protecting the free Gemini quota matters more here than in the
# reference (local, unmetered) project, so the cache is relied on more
# heavily -- but a long TTL still risks masking a same-day prompt/parsing fix
# behind a stale cached answer, so this keeps the reference's 6-hour value
# rather than going longer.
CACHE_TTL_SECONDS = 6 * 60 * 60

# --- Cost estimation ---
# $/1K-token rates, used to compute QueryLog.cost_usd. Checked via WebFetch
# against ai.google.dev/gemini-api/docs/pricing on 2026-07-17 (paid-tier
# rates; actual billing stays $0 while under the free-tier quota). Update
# here if GEMINI_CHAT_MODEL is swapped to a model not listed below -- an
# unlisted model falls back to a 0.0 rate rather than raising (see
# sage/generation/cost.py). gemini-2.5-flash/-lite are kept here for
# reference even though live-testing found them 404 for new users (see
# GEMINI_CHAT_MODEL's comment) -- an existing account with prior access might
# still be routed to them. "gemini-flash-latest" (the default) has no fixed
# rate here since it's a rolling alias -- its cost estimate falls back to
# 0.0 until it's pinned to a concrete model id. No entry for
# LOCAL_EMBEDDING_MODEL: local embeddings run on this machine's own
# CPU/GPU, so their cost is genuinely and always $0, not merely
# falling back to the same default an unlisted cloud model would.
MODEL_COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"prompt": 0.0003, "completion": 0.0025},
    "gemini-2.5-flash-lite": {"prompt": 0.0001, "completion": 0.0004},
    "gemini-3.5-flash": {"prompt": 0.0015, "completion": 0.009},
}

for _dir in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR, DB_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- API ---
# Shared demo-access key for the public deployment: unset (None) locally, so
# api/middleware.py's DemoKeyMiddleware is a complete no-op by default. Not a
# real secret (Vite compiles VITE_DEMO_ACCESS_KEY into the public frontend
# bundle) -- see api/middleware.py's module docstring.
DEMO_ACCESS_KEY = os.environ.get("DEMO_ACCESS_KEY") or None
# Curated-demo boundary for the public deployment. Defaults to disabled --
# uploads must be explicitly opted into with ALLOW_UPLOADS=true, not
# explicitly opted out of, so a deployment that forgets to set this env var
# fails closed (no arbitrary-PDF-processing service) rather than failing
# open. Previously defaulted to enabled, which every deployment had to
# remember to override.
ALLOW_UPLOADS = os.environ.get("ALLOW_UPLOADS", "false").lower() == "true"
# Hard ceiling on a single upload's size, enforced while the file streams to
# disk (api/routes/documents.py) so an oversized upload is rejected without
# ever buffering the whole thing in memory. Default 25MB comfortably covers
# a real multi-hundred-page 10-K (the demo corpus's largest filing is well
# under this) while still bounding worst-case disk/memory use per upload.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
# Hard ceiling on a single upload's page count, checked after the file is
# confirmed to be a readable PDF but before it's handed to the (synchronous,
# in-request) ingest pipeline -- keeps a single upload's ingestion latency
# bounded on the public deployment, which has no background job queue.
MAX_UPLOAD_PAGES = int(os.environ.get("MAX_UPLOAD_PAGES", "500"))
# Hard ceiling on a single chat query's length -- rejects a pathological
# multi-megabyte "query" string before it reaches embedding/retrieval/the
# Gemini prompt, where it would otherwise burn real compute/quota on
# obvious junk input.
MAX_QUERY_LENGTH = int(os.environ.get("MAX_QUERY_LENGTH", "2000"))
# slowapi rate-limit spec string, applied to POST /chat, POST /chat/stream,
# and POST /documents/upload to protect the free Gemini quota (and, for
# uploads, ingestion capacity) behind the public deployment.
CHAT_RATE_LIMIT = os.environ.get("CHAT_RATE_LIMIT", "10/minute")
