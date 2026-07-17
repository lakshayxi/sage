"""Estimated $ cost from token counts and config/settings.MODEL_COST_PER_1K_TOKENS.

Unlike the reference (local Ollama) project, where every rate is 0.0 and
cost tracking exists mostly to prove the plumbing works, Sage's rates are
Gemini's real published per-1K-token pricing -- actual billing stays $0
while under the free-tier quota, but the estimate is meaningful (and matters
more here since it's also what protects the free quota from being burned
carelessly; see generation/cache.py).
"""

from config import settings


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Falls back to a zero-rate for an unconfigured model rather than raising,
    # so a new/renamed model doesn't break answer generation over a missing
    # pricing entry.
    rates = settings.MODEL_COST_PER_1K_TOKENS.get(model, {"prompt": 0.0, "completion": 0.0})
    return (prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates[
        "completion"
    ]
