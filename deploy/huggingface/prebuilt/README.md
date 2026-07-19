# Pre-ingested demo corpus

This directory is baked into the Hugging Face Space image unconditionally
(see the root `Dockerfile`) so the public demo works with zero setup —
no upload step, no ingestion wait, no `GEMINI_API_KEY`-backed ingestion call
on first visit.

## What goes here

- `chroma/` — a full copy of a `data/chroma/` Chroma persistent-client
  directory (the `sage_chunks` and `sage_query_cache` collections) produced
  by running `sage ingest` locally against the demo PDFs (Apple, Microsoft,
  and NVIDIA's latest 10-Ks).
- `sage.db` — the matching SQLite database (`db/sage.db`) from that same
  ingest run: `Document`/`Chunk` rows the Chroma vectors point back into,
  plus (optionally) any warmed query-cache rows.

Both **must** come from the same ingest run — Chroma only stores vectors and
filter metadata, SQLite is the source of truth for chunk text (see
`CLAUDE.md`'s architecture notes in the reference project this pattern is
borrowed from); mismatched copies will retrieve vectors whose chunk text no
longer exists, or vice versa.

## Current status

Placeholder only. `chroma/` currently holds just a `.gitkeep`, and `sage.db`
is a 0-byte file (a valid, empty SQLite database — Chroma/SQLAlchemy both
open it fine, it just has no data). This is intentional: it lets the
Dockerfile's `COPY` steps succeed unconditionally, whether or not the real
data has landed yet, so the image always builds — it just serves an empty
corpus until the real files replace these placeholders.

Generating the real data isn't blocked on a Gemini quota issue anymore —
embeddings now run locally (`sage/embed/local_embedder.py`), so `sage ingest`
makes no Gemini calls at all. The only remaining blocker is that nobody has
run the three-command sequence below yet.

## How to (re)generate this

From the repo root, with `GEMINI_API_KEY` set and quota available:

```bash
.venv/bin/sage ingest data/raw/apple-10k.pdf --company Apple --fiscal-year 2025 --doc-type 10-K
.venv/bin/sage ingest data/raw/microsoft-10k.pdf --company Microsoft --fiscal-year 2025 --doc-type 10-K
.venv/bin/sage ingest data/raw/nvidia-10k.pdf --company NVIDIA --fiscal-year 2025 --doc-type 10-K

rm -rf deploy/huggingface/prebuilt/chroma
cp -r data/chroma deploy/huggingface/prebuilt/chroma
cp db/sage.db deploy/huggingface/prebuilt/sage.db
```

Then rebuild the image (`docker build .`) and spot-check a query against the
container before pushing to the Space — see `deploy/huggingface/DEPLOY.md`.
