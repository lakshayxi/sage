# Deploying Sage to Hugging Face Spaces

Target: a single Docker Space (free CPU tier, 16GB RAM) serving both the
API and the built frontend from one FastAPI process, one URL. No other
deployment target exists for this repo — this is the only path.

## Prerequisites (not yet done as of this writing)

1. **Real ingested demo data** in `deploy/huggingface/prebuilt/` (currently
   placeholders — see `deploy/huggingface/prebuilt/README.md`). The Space
   will build and run without this, it'll just answer from an empty corpus.
2. **A Hugging Face account and an empty Docker Space** created at
   `huggingface.co/new-space` (SDK: Docker) to push to.
3. **The frontend (`frontend/`) built and committed** — this Dockerfile's
   first stage runs `npm ci && npm run build` against it.

## One-time Space configuration

In the Space's **Settings → Variables and secrets**, set:

| Name | Type | Required | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | Secret | Yes | Sage is Gemini-only (chat + embeddings) — nothing works without this. |
| `DEMO_ACCESS_KEY` | Secret | Recommended | Gates `/chat`, `/conversations`, `/documents` behind an `X-Demo-Key` header (`api/middleware.py`). Leave unset only if the demo is meant to be fully open. |
| `ALLOW_UPLOADS` | Variable | Yes — set to `false` | Curated-demo boundary: `POST /documents/upload` 403s instead of accepting arbitrary PDFs from the public internet. Defaults to `true` if unset — **must** be explicitly set on this deployment. |
| `CHAT_RATE_LIMIT` | Variable | Optional | slowapi rate-limit spec (default `10/minute`) applied to `/chat` and `/chat/stream`, protecting the shared Gemini quota behind a public URL. |

These map directly to `config/settings.py`'s `os.environ.get(...)` calls — no
other wiring is needed; HF injects both Variables and Secrets as container
environment variables at runtime (confirmed against
huggingface.co/docs/hub/spaces-sdks-docker, "Secrets and Variables
Management" → Runtime).

**Never commit real values for any of these** — set them only in the
Space's Settings UI.

### `DEMO_ACCESS_KEY` also needs `VITE_DEMO_ACCESS_KEY` at *build* time

If `DEMO_ACCESS_KEY` is set, the frontend must be built with a matching
`VITE_DEMO_ACCESS_KEY` so it can actually supply the key back — as an
`X-Demo-Key` header from `conversations.ts`/`documents.ts`, and as a `key`
query param from `chat.ts`'s `GET /chat/stream` call, since the browser's
native `EventSource` can't set custom headers at all (see
`api/middleware.py` and `frontend/src/api/chat.ts`). Without this, setting
only `DEMO_ACCESS_KEY` locks the deployed frontend out of its own API.

Vite inlines `VITE_*` vars into the built JS at build time, not runtime, so
this can't be set as a Space Secret/Variable the way `DEMO_ACCESS_KEY`
is — those only become container environment variables *after* the image
is already built. The Dockerfile's frontend-build stage accepts it as a
build-arg (`ARG VITE_DEMO_ACCESS_KEY`), but **HF Spaces' own Docker build
does not accept custom `--build-arg` values** (Secrets/Variables are
runtime-only there). Until this deployment has its own CI/build pipeline,
setting a real `DEMO_ACCESS_KEY` on this Space means building and pushing
the image yourself instead of letting HF build it from source:

```bash
docker build --build-arg VITE_DEMO_ACCESS_KEY=<same value as DEMO_ACCESS_KEY> -t <your-image> .
```

then push `<your-image>` to the Space's registry per HF's Docker Spaces
docs, rather than using `scripts/deploy_hf_space.py --push` (which uploads
source for HF to build itself, with no build-arg). If this key isn't set at
all, skip this section entirely — the demo stays fully open.

## Deploying

```bash
.venv/bin/python scripts/deploy_hf_space.py --repo-id <your-username>/sage --push
```

This stages `api/`, `sage/`, `config/`, `frontend/` (source, not `dist/` —
the Space builds it itself from the Dockerfile), `deploy/huggingface/prebuilt/`,
`pyproject.toml`, `Dockerfile`, and `.dockerignore` into a temp folder,
renames `deploy/huggingface/README.md` to that folder's `README.md` (the
file HF reads for the Space's YAML config), and uploads it via
`huggingface_hub`'s `HfApi().upload_folder(...)`. Omit `--push` (or pass
`--dry-run` explicitly) to only print what would be uploaded without
contacting the Hub — do this first.

HF then builds the `Dockerfile` server-side. Watch the build in the Space's
**Logs** tab; a Docker CPU-tier build of this image (torch +
sentence-transformers + a frontend npm build) can take several minutes.

**This pushes to a shared, public HF repo and (re)deploys the live Space —
confirm with the user before running with `--push` for real.**

## Verifying after deploy

1. Space status shows **Running** (not "Build error" / "Runtime error") in
   the Space UI.
2. `curl -s -o /dev/null -w '%{http_code}\n' https://<your-username>-sage.hf.space/documents`
   returns `200` (or `401` if `DEMO_ACCESS_KEY` is set and no header was
   sent — either confirms the app is actually serving, not crash-looping).
3. Open `https://<your-username>-sage.hf.space/` in a browser — the built
   frontend should load (confirms the `StaticFiles` mount in `api/main.py`
   is serving `frontend/dist` correctly), and a real query against the demo
   corpus returns a cited answer.
4. Tail the Space's **Logs** tab for a few minutes after first traffic —
   confirms no repeated tracebacks on real requests.

## Rollback

HF Spaces keeps every previous commit to the Space's git repo. To roll back:

- **Via UI**: Space → **Files and versions** → find the last known-good
  commit → **Revert**. HF rebuilds and redeploys automatically from that
  commit.
- **Via git**: `git -C <space-repo-clone> revert <bad-commit>` then
  `git push`, or reset to a known-good commit and force-push (HF Spaces
  repos are single-branch deployment targets, not shared collaborative
  history, so a force-push here doesn't carry the usual shared-branch
  risk — but confirm this is really a Space-only clone, not this project's
  own repo, before force-pushing anything).
- Since the demo corpus is baked into the image rather than mounted from
  persistent storage, a rollback also rolls back the data — there's no
  separate "bad code, good data" split to worry about.

## Known limitations

- **Free Spaces have ephemeral container storage.** Anything written at
  runtime (new conversations in `db/sage.db`, cache writes) is lost on
  every Space restart/rebuild — expected and acceptable for a stateless
  public demo; don't treat this deployment as durable storage for
  conversation history.
- **No persistent-storage add-on is configured** (paid HF feature) — not
  needed here since the whole point is a fixed, baked-in demo corpus, not
  live ingestion.
- **The cross-encoder reranker model (`BAAI/bge-reranker-base`) is not baked
  into the image** — `sage/retrieval/reranker.py` downloads it from the HF
  Hub lazily on first use and caches it under `$HOME/.cache/huggingface`
  (writable by `appuser`). Expect the *first* query after a cold start to be
  noticeably slower than subsequent ones; this also means the container
  needs outbound network access at runtime, not just at build time (true by
  default on HF Spaces).
