# Deploying Sage to Hugging Face Spaces

Sage ships as one Docker Space: Vite builds the frontend, FastAPI serves the
static application and API on port 7860, and the clean 3-filing corpus is baked
into the image. Check Hugging Face's current Docker Space eligibility and
hardware pricing before creating the Space.

## 1. Create and configure the Space

Create an empty Space with the **Docker** SDK. In **Settings → Variables and
secrets**, configure:

| Name | Type | Required | Purpose |
|---|---|---:|---|
| `GEMINI_API_KEY` | Secret | Yes | Gemini answer generation. |
| `GEMINI_CHAT_MODEL` | Variable | No | Defaults to the pinned `gemini-3.5-flash-lite`. |
| `ALLOW_UPLOADS` | Variable | No | Leave unset/`false` for the curated public demo. |
| `CHAT_RATE_LIMIT` | Variable | No | Shared `/chat` + `/chat/stream` limit; default `10/minute` per client IP. |
| `CONVERSATION_RATE_LIMIT` | Variable | No | Conversation-create limit; default `10/minute`. |
| `MAX_CONCURRENT_GENERATIONS` | Variable | No | Process-wide expensive-work cap; default `2`. |
| `CORS_ORIGINS` | Variable | No | Leave empty for the same-origin Space frontend. |

`DEMO_ACCESS_KEY` is only a casual deterrent, not authentication, because a
browser client must receive the same value. The published image deliberately
does not compile one into its JavaScript. Leave it unset for this recruiter
demo and rely on request bounds, shared generation throttling, the global
concurrency cap, Gemini project quotas, and monitoring.

Never commit real keys. The deployment corpus is already scrubbed of local
queries, caches, conversations, and session tokens.

## 2. Validate and deploy

```bash
.venv/bin/python scripts/build_demo_corpus.py
.venv/bin/python scripts/deploy_hf_space.py \
  --repo-id <your-username>/sage \
  --dry-run

# Requires an HF_TOKEN with write access. This mutates the live Space.
.venv/bin/python scripts/deploy_hf_space.py \
  --repo-id <your-username>/sage \
  --push
```

The dry run refuses an empty, inconsistent, or privacy-contaminated corpus.
It stages only the Docker build inputs, lockfile, application source, frontend,
Space README, and prebuilt corpus. `--push` uploads that staging directory and
triggers the Space build.

Both local models are pinned to immutable Hugging Face revisions and downloaded
at image-build time. Runtime starts in Hugging Face/Transformers offline mode,
warms both models before reporting ready, and does not make the first visitor
wait on model downloads.

The production image disables durable query logging and stores only a digest of
each anonymous browser session token. Runtime answer caches remain ephemeral and
are never copied back into the committed corpus.

## 3. Verify the live deployment

Replace `<space-host>` with the generated `*.hf.space` host:

```bash
curl -i https://<space-host>/healthz
curl -i https://<space-host>/readyz
curl -i https://<space-host>/documents
```

Expected results:

- `/healthz`: `200 {"status":"ok"}`.
- `/readyz`: `200` with `documents: 3`; `503` means the corpus or model warmup
  is not ready.
- `/documents`: three ready filings.
- The root URL loads the UI, a sample prompt streams an answer, citation cards
  expand, and each curated filing offers an SEC source link.

Also inspect the Space build/runtime logs and make one real Apple, Microsoft,
and three-company comparison query before sharing the URL.

## Rollback and persistence

Use **Files and versions → Revert** in the Space UI to restore the previous
Space commit; this rebuilds code and baked data together. Container storage is
ephemeral, so runtime conversations and caches may disappear after a restart.
That is intentional for this anonymous portfolio demo. Uploads remain disabled
and the committed corpus remains the source of truth.
