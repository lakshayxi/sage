"""Retry-with-backoff for transient Gemini API errors (429 rate/quota limits,
5xx server errors). Used by GeminiChatClient only.

Originally added for `GeminiEmbedder` (since removed -- embeddings now run
locally, see sage/embed/local_embedder.py) after a real failure mode
observed during live testing (2026-07-17, against a real free-tier
GEMINI_API_KEY): a single batched `embed_content` call over ~30+ chunk
texts from one real 10-K-sized PDF (well under the SDK's own request-size
limits) was rejected with `429 RESOURCE_EXHAUSTED`, while the same call
over 20 texts succeeded. That turned out to be the shallow end of a much
harder wall: the embedding quota never recovered even after generating a
new API key (almost certainly the same underlying GCP project), which is
what ultimately forced the move to local embeddings entirely rather than
just tuning the batch size further. This module is kept for
`GeminiChatClient`, which still legitimately talks to a remote API and can
hit transient 429/5xx errors independent of that embedding-specific quota
history -- without retry logic here, a single such error propagates all the
way up and fails the whole request over what may just be a momentary blip,
exactly the "assume the model always cooperates" failure mode to avoid.

Not retried: non-429 4xx errors (bad request, auth failure, invalid model
name) -- retrying those just wastes quota on a request that will never
succeed differently.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai import errors

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = {429, 500, 503}
_MAX_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 2.0


def call_with_retry(fn: Callable[[], T], *, what: str = "Gemini API call") -> T:
    """Call `fn()`, retrying on 429/5xx with exponential backoff.

    `what` is used only for the log message so retries are identifiable in
    logs (e.g. "embed_content batch of 20", "generate_content").
    """
    last_error: errors.APIError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn()
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_ATTEMPTS:
                raise
            last_error = e
            delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s failed with %s (attempt %d/%d); retrying in %.0fs",
                what,
                code,
                attempt,
                _MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
    # Unreachable in practice (the loop always returns or raises), but keeps
    # type checkers happy and fails loudly instead of returning None.
    raise last_error  # type: ignore[misc]
