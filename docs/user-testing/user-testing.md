# User Testing Log

**Date:** 2026-07-18
**Environment:** Real backend (`.venv/bin/uvicorn api.main:app`) + real frontend (`npm run dev`), against the real ingested corpus (Apple/Microsoft/NVIDIA 10-Ks, 301 chunks). First time the app was driven by hand in a browser rather than through curl/pytest.

Three real bugs surfaced this session, two fixed, one deliberately left open. Logged here rather than folded silently into commit messages, since none of them were caught by the existing test suite or code review — only live use surfaced them.

## 1. `GEMINI_API_KEY` silently empty at runtime (fixed)

**Symptom:** Every `/chat` call failed with a generic 500. Looked like a quota/auth problem with the key.

**Root cause:** `config/settings.py` read `os.environ.get("GEMINI_API_KEY", "")` but never called `load_dotenv()` anywhere in the app. The key sat in `.env` on disk but was never loaded into the process environment for uvicorn or the CLI — every real run silently had `GEMINI_API_KEY = ""`. Diagnostic scripts that manually called `load_dotenv()` first worked fine, which is what made this look like an intermittent key/quota issue rather than a config bug.

**Fix:** added `python-dotenv` as an explicit dependency and `load_dotenv(BASE_DIR / ".env")` at the top of `config/settings.py`. No-op if the shell/container already exports these vars, so safe in deployed environments too.

**Verified:** live `/chat` call after the fix returned a real, correctly-cited answer.

## 2. Model-specific transient failures misread as quota exhaustion

**Symptom:** Believed the Gemini free tier was fully exhausted with no working key.

**Root cause:** Different failure modes were being lumped together. Live-tested all three in one session, same key:
| Model | Result |
|---|---|
| `gemini-flash-latest` | `503 UNAVAILABLE` — Google-side capacity issue, not this account |
| `gemini-2.0-flash` | `429 RESOURCE_EXHAUSTED` — genuinely quota-exhausted |
| `gemini-flash-lite-latest` | Worked |

**Fix:** switched `GEMINI_CHAT_MODEL` default to `gemini-flash-lite-latest`.

**Lesson:** a 401/429/503 on one model doesn't mean the key or account is dead — worth testing sibling models on the same key before concluding it's a quota/auth problem.

## 3. SSE stream drops the last ~11 characters when no citations fence appears (fixed)

**Symptom:** Reported by the user from the live UI — the no-relevant-context refusal message rendered as `"...relevant to th"`, cut off mid-word.

**Root cause:** `api/routes/chat.py`'s `chat_stream` holds back the last `max_marker_len` (12) characters of the buffer at all times, in case they're the start of a ` ```citations ` fence marker split across two delta chunks. That tail is only flushed once a fence is actually found. The refusal message is a fixed string with no fence at all, so its last 12 characters were held back forever and never sent — silently dropped from the live stream (the `done` payload's `result.answer_text` was always correct; only the live-typed text was wrong).

**Fix:** two changes —
- Backend: after the stream ends, if no fence was ever found, flush whatever's left in the buffer (`api/routes/chat.py`).
- Frontend: on the `done` event, overwrite the accumulated streamed text with the authoritative `result.answer` rather than trusting the deltas alone (`useChatSession.ts`) — a defensive fix against this whole bug class, not just this instance.

**Verified:** raw SSE stream re-tested via curl post-fix — the held-back tail now arrives as a second `delta` event.

## 4. Vague/conversational queries get falsely refused (known limitation, left as-is)

**Symptom:** "how did apple do in FY25?" (Apple filter on) → refused with "no relevant information," even though Apple FY25 data is definitely ingested. "tell me about apple financials" and "what was Apple's total net sales..." on the same corpus both worked fine.

**Root cause:** `MIN_RERANK_SCORE = 0.1` (`config/settings.py`) is a hard cutoff — if every reranked candidate scores below it, the pipeline refuses without calling the LLM at all. For "how did apple do in FY25?", the top score across 30 candidates was **0.0512** — under threshold, despite genuinely relevant chunks existing in the corpus. The threshold value was carried over from the reference project's corpus/queries, and its own code comment already flagged it as unvalidated against this corpus. Vague, conversational phrasing just doesn't lexically/semantically match dense financial-statement prose as well as the reranker expects.

**Status: intentionally left unfixed for now** (explicit call — not an oversight). Two candidate fixes for later, not mutually exclusive:
1. Retune `MIN_RERANK_SCORE` down for this corpus.
2. Replace the hard cutoff with "always pass the top-N chunks through, let the LLM itself judge and say 'insufficient information' if the context doesn't answer the question" — more robust to phrasing variance, moves the relevance judgment call from a brittle score threshold to the model.

## 5. No balanced retrieval across companies without explicit selection (expected — this is task #9)

**Symptom:** "compare apple, nvidia and microsoft" with no company checkboxes selected returned an NVIDIA-only answer, ignoring Apple/Microsoft entirely.

**Root cause:** with no company filter, retrieval runs one flat search over the whole corpus rather than a balanced per-company search, so results skew toward whichever company's chunks happen to match best. Selecting all three companies explicitly does activate the balanced per-company retrieval path (`_merge_balanced` in `sage/retrieval/retriever.py`), but even then, answer quality was weak — each company only gets a thin slice of the usual `top_k`, and there's no dedicated prompt for synthesizing a structured cross-company comparison.

**Status:** not a bug — this is exactly the scope of task #9 (Compare Mode), which hasn't been built yet.
