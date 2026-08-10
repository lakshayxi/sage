from config import settings
from sage.db.database import get_session
from sage.db.models import QueryLog
from sage.db.query_log import record_query_log


def test_query_logging_can_be_disabled_for_public_deployments(monkeypatch):
    monkeypatch.setattr(settings, "QUERY_LOGGING_ENABLED", False)

    assert record_query_log("private question", None) == 0

    session = get_session()
    try:
        assert session.query(QueryLog).count() == 0
    finally:
        session.close()
