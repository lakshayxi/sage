# Pre-ingested demo corpus

This directory contains the deployment-ready Apple FY2025, Microsoft FY2025,
and NVIDIA FY2026 corpus baked into the Docker image:

- `sage.db`: 3 ready documents and 421 source chunks.
- `chroma/`: the matching 421 embeddings in the `sage_chunks` collection.

The artifact intentionally contains no conversations, messages, session
tokens, query logs, exact-answer cache, or semantic-answer cache. Those tables
exist in SQLite for runtime use but are empty in the committed copy.

## Rebuild safely

After ingesting or changing the local corpus, run:

```bash
.venv/bin/python scripts/build_demo_corpus.py
.venv/bin/python scripts/deploy_hf_space.py --repo-id <username>/sage --dry-run
```

The exporter uses a SQLite backup, removes private runtime rows, copies only
the chunk collection into a fresh Chroma store, and validates that every
SQLite embedding ID has exactly one matching Chroma ID before atomically
replacing this directory.

Do not copy `db/sage.db` or `data/chroma/` here by hand. Development stores
contain private query text and opaque browser session tokens, and separately
copied stores can point at different chunk sets.
