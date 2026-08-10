"""Small process-local admission gate for expensive generation work.

Per-IP throttles protect ordinary traffic, but many clients can still arrive at
once. A CPU demo should reject excess work immediately instead of queueing an
unbounded number of reranker/Gemini jobs in the server thread pool.
"""

import threading

from config import settings

_lock = threading.Lock()
_active = 0


def try_acquire_generation_slot() -> bool:
    global _active
    with _lock:
        if _active >= max(1, settings.MAX_CONCURRENT_GENERATIONS):
            return False
        _active += 1
        return True


def release_generation_slot() -> None:
    global _active
    with _lock:
        _active = max(0, _active - 1)


def reset_generation_capacity() -> None:
    """Test-only reset for a request interrupted before normal cleanup."""
    global _active
    with _lock:
        _active = 0
