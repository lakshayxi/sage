import pytest
from google.genai import errors

from sage.retry import call_with_retry


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": "boom", "status": "ERR"}})


def test_call_with_retry_returns_result_on_first_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert call_with_retry(fn) == "ok"
    assert calls["n"] == 1


def test_call_with_retry_retries_on_429_then_succeeds(monkeypatch):
    import sage.retry as retry_mod

    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _api_error(429)
        return "recovered"

    assert call_with_retry(fn) == "recovered"
    assert calls["n"] == 3


def test_call_with_retry_gives_up_after_max_attempts(monkeypatch):
    import sage.retry as retry_mod

    monkeypatch.setattr(retry_mod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _api_error(429)

    with pytest.raises(errors.APIError):
        call_with_retry(fn)
    assert calls["n"] == retry_mod._MAX_ATTEMPTS


def test_call_with_retry_does_not_retry_non_retryable_status():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _api_error(404)

    with pytest.raises(errors.APIError):
        call_with_retry(fn)
    assert calls["n"] == 1  # no retry attempted for a 404


def test_call_with_retry_retries_on_503():
    import sage.retry as retry_mod

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 2:
            raise _api_error(503)
        return "ok"

    import time as time_mod

    orig_sleep = time_mod.sleep
    retry_mod.time.sleep = lambda s: None
    try:
        assert call_with_retry(fn) == "ok"
    finally:
        retry_mod.time.sleep = orig_sleep
    assert calls["n"] == 2
