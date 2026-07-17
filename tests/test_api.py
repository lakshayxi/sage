import json

import pytest
from fastapi.testclient import TestClient
from fpdf import FPDF

import api.routes.chat as chat_routes
import api.routes.documents as documents_routes
from api.limiter import limiter
from api.main import app
from config import settings
from sage.db import conversations
from sage.generation.answer_engine import AnswerResult, Citation


@pytest.fixture(autouse=True)
def _reset_limiter():
    # The Limiter's in-memory storage is a module-level singleton, so a
    # rate-limit test earlier in the run would otherwise bleed into later
    # tests hitting the same /chat* routes.
    limiter.reset()
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _fake_answer(**overrides) -> AnswerResult:
    defaults = dict(
        answer_text="Margins declined due to component costs [1].",
        citations=[
            Citation(
                n=1,
                chunk_id=1,
                text="chunk text",
                page_number=1,
                company="Apple",
                fiscal_year="FY24",
                doc_type="10-K",
                filename="Apple_FY24_10-K.pdf",
            )
        ],
        model="gemini-2.5-flash",
        retrieval_latency_ms=10.0,
        generation_latency_ms=20.0,
        total_latency_ms=30.0,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        retrieved_chunk_ids=[1],
    )
    defaults.update(overrides)
    return AnswerResult(**defaults)


def _make_sample_pdf(tmp_path, filename="TestCo_FY24_annual_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, "Test Co reported FY24 results with strong revenue growth.")
    pdf_path = tmp_path / filename
    pdf.output(str(pdf_path))
    return pdf_path


# --- /chat ---


def test_chat_non_streaming(monkeypatch, client):
    monkeypatch.setattr(chat_routes, "generate_answer", lambda *a, **kw: _fake_answer())

    response = client.post("/chat", json={"query": "Why did Apple margins decline?"})

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["answer"] == "Margins declined due to component costs [1]."
    assert body["citations"][0]["chunk_id"] == 1
    assert body["tokens"]["total_tokens"] == 150
    assert body["latency_ms"]["total_ms"] == 30.0
    assert body["session_id"] is None


def test_chat_with_session_id_loads_history_and_persists_turn(monkeypatch, client):
    conversation_id, token = conversations.create_conversation(title="Apple margins")
    captured = {}

    def fake_generate_answer(*args, **kwargs):
        captured["history"] = kwargs.get("history")
        return _fake_answer()

    monkeypatch.setattr(chat_routes, "generate_answer", fake_generate_answer)

    response = client.post(
        "/chat",
        json={"query": "What about last year?", "session_id": conversation_id},
        headers={"X-Session-Token": token},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == conversation_id
    assert captured["history"] == []  # brand-new conversation, no prior turns

    history = conversations.get_history(conversation_id)
    assert [h.role for h in history] == ["user", "assistant"]
    assert history[0].content == "What about last year?"
    assert history[1].content == "Margins declined due to component costs [1]."


def test_chat_returns_500_on_generation_failure(monkeypatch, client):
    def failing_generate_answer(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(chat_routes, "generate_answer", failing_generate_answer)

    response = client.post("/chat", json={"query": "Why did Apple margins decline?"})

    assert response.status_code == 500


def test_chat_stream_emits_deltas_then_done_event(monkeypatch, client):
    def fake_generate_answer_stream(*args, **kwargs):
        yield "Margins "
        yield "declined."
        yield _fake_answer(answer_text="Margins declined.")

    monkeypatch.setattr(chat_routes, "generate_answer_stream", fake_generate_answer_stream)

    with client.stream("GET", "/chat/stream", params={"query": "Why did margins decline?"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    assert "event: done" in body
    done_line = next(
        line for line in body.splitlines() if line.startswith("data: ") and "answer" in line
    )
    payload = json.loads(done_line.removeprefix("data: "))
    assert payload["answer"] == "Margins declined."
    assert payload["schema_version"] == 1


def test_chat_stream_strips_citations_fence_from_live_deltas(monkeypatch, client):
    def fake_generate_answer_stream(*args, **kwargs):
        yield "Answer text [1]."
        yield '\n```citations\n[{"n": 1, "chunk_id": 1}]\n```'
        yield _fake_answer(answer_text="Answer text [1].")

    monkeypatch.setattr(chat_routes, "generate_answer_stream", fake_generate_answer_stream)

    with client.stream("GET", "/chat/stream", params={"query": "q"}) as r:
        body = "".join(r.iter_text())

    delta_lines = [
        line for line in body.splitlines() if line.startswith("data: ") and "delta" in line
    ]
    streamed_text = "".join(
        json.loads(line.removeprefix("data: "))["delta"] for line in delta_lines
    )
    assert "```citations" not in streamed_text
    assert "chunk_id" not in streamed_text


def test_chat_rate_limit_returns_429_past_threshold(monkeypatch, client):
    monkeypatch.setattr(chat_routes, "generate_answer", lambda *a, **kw: _fake_answer())

    responses = [
        client.post("/chat", json={"query": "q"})
        for _ in range(11)  # default limit is 10/minute
    ]

    assert [r.status_code for r in responses[:10]] == [200] * 10
    assert responses[10].status_code == 429


# --- /conversations ---


def test_conversation_create_list_get_round_trip(client):
    create_response = client.post("/conversations", json={"title": "Apple margins"})
    assert create_response.status_code == 200
    body = create_response.json()
    conversation_id = body["conversation_id"]
    token = body["session_token"]
    assert token

    list_response = client.get("/conversations", headers={"X-Session-Token": token})
    assert list_response.status_code == 200
    ids = [c["id"] for c in list_response.json()]
    assert conversation_id in ids

    conversations.append_message(conversation_id, "user", "What were margins?")
    conversations.append_message(conversation_id, "assistant", "They declined.")

    detail_response = client.get(
        f"/conversations/{conversation_id}", headers={"X-Session-Token": token}
    )
    assert detail_response.status_code == 200
    body = detail_response.json()
    assert body["id"] == conversation_id
    assert [m["content"] for m in body["messages"]] == ["What were margins?", "They declined."]


def test_second_conversation_reuses_the_same_session_token(client):
    first = client.post("/conversations", json={"title": "first"}).json()
    second = client.post(
        "/conversations",
        json={"title": "second"},
        headers={"X-Session-Token": first["session_token"]},
    ).json()

    assert second["session_token"] == first["session_token"]

    list_response = client.get(
        "/conversations", headers={"X-Session-Token": first["session_token"]}
    )
    ids = [c["id"] for c in list_response.json()]
    assert set(ids) == {first["conversation_id"], second["conversation_id"]}


def test_get_unknown_conversation_returns_404(client):
    token = client.post("/conversations", json={}).json()["session_token"]
    response = client.get("/conversations/999999", headers={"X-Session-Token": token})
    assert response.status_code == 404


# --- session-token isolation (IDOR) ---


def test_conversations_list_is_empty_with_no_session_token(client):
    client.post("/conversations", json={"title": "someone else's"})
    response = client.get("/conversations")
    assert response.status_code == 200
    assert response.json() == []


def test_get_conversation_404s_with_wrong_session_token(client):
    created = client.post("/conversations", json={"title": "private"}).json()
    conversation_id = created["conversation_id"]

    response = client.get(
        f"/conversations/{conversation_id}", headers={"X-Session-Token": "not-my-token"}
    )
    assert response.status_code == 404


def test_get_conversation_404s_with_no_session_token(client):
    created = client.post("/conversations", json={"title": "private"}).json()
    response = client.get(f"/conversations/{created['conversation_id']}")
    assert response.status_code == 404


def test_two_sessions_each_only_see_their_own_conversations(client):
    a = client.post("/conversations", json={"title": "alice's"}).json()
    b = client.post("/conversations", json={"title": "bob's"}).json()
    assert a["session_token"] != b["session_token"]

    alice_list = client.get("/conversations", headers={"X-Session-Token": a["session_token"]})
    bob_list = client.get("/conversations", headers={"X-Session-Token": b["session_token"]})

    assert [c["id"] for c in alice_list.json()] == [a["conversation_id"]]
    assert [c["id"] for c in bob_list.json()] == [b["conversation_id"]]

    # Alice can't read Bob's conversation by guessing its id, and vice versa.
    assert (
        client.get(
            f"/conversations/{b['conversation_id']}",
            headers={"X-Session-Token": a["session_token"]},
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/conversations/{a['conversation_id']}",
            headers={"X-Session-Token": b["session_token"]},
        ).status_code
        == 404
    )


def test_chat_with_session_id_404s_when_token_does_not_match(monkeypatch, client):
    conversation_id, token = conversations.create_conversation(title="private")
    monkeypatch.setattr(chat_routes, "generate_answer", lambda *a, **kw: _fake_answer())

    response = client.post(
        "/chat",
        json={"query": "what about it", "session_id": conversation_id},
        headers={"X-Session-Token": "wrong-token"},
    )

    assert response.status_code == 404


def test_chat_stream_404s_when_session_token_missing(client):
    conversation_id, _token = conversations.create_conversation(title="private")

    with client.stream(
        "GET",
        "/chat/stream",
        params={"query": "q", "session_id": conversation_id},
    ) as r:
        assert r.status_code == 404


# --- /documents ---


def test_documents_list_empty(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_documents_upload_success(monkeypatch, tmp_path, client):
    monkeypatch.setattr(documents_routes, "ingest_pdf", lambda *a, **kw: _FakeDocument())
    pdf_path = _make_sample_pdf(tmp_path)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/documents/upload",
            files={"file": (pdf_path.name, f, "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["filename"] == "TestCo_FY24_annual_report.pdf"
    assert body["status"] == "ready"


def test_documents_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_documents_upload_403_when_uploads_disabled(monkeypatch, tmp_path, client):
    monkeypatch.setattr(settings, "ALLOW_UPLOADS", False)
    pdf_path = _make_sample_pdf(tmp_path)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/documents/upload",
            files={"file": (pdf_path.name, f, "application/pdf")},
        )

    assert response.status_code == 403


def test_documents_upload_rate_limit_returns_429_past_threshold(monkeypatch, tmp_path, client):
    monkeypatch.setattr(documents_routes, "ingest_pdf", lambda *a, **kw: _FakeDocument())
    pdf_path = _make_sample_pdf(tmp_path)

    def _upload():
        with open(pdf_path, "rb") as f:
            return client.post(
                "/documents/upload",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )

    responses = [_upload() for _ in range(11)]  # default limit is 10/minute

    assert [r.status_code for r in responses[:10]] == [200] * 10
    assert responses[10].status_code == 429


class _FakeDocument:
    id = 1
    filename = "TestCo_FY24_annual_report.pdf"
    status = "ready"
    page_count = 1
    company = "TestCo"
    fiscal_year = "FY24"
    doc_type = "annual_report"


# --- demo-key middleware ---


def test_demo_key_middleware_is_a_noop_when_unset(monkeypatch, client):
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", None)
    response = client.get("/documents")
    assert response.status_code == 200


def test_demo_key_middleware_401s_when_set_and_header_missing(monkeypatch, client):
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", "secret")
    response = client.get("/documents")
    assert response.status_code == 401


def test_demo_key_middleware_401s_when_set_and_header_wrong(monkeypatch, client):
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", "secret")
    response = client.get("/documents", headers={"X-Demo-Key": "wrong"})
    assert response.status_code == 401


def test_demo_key_middleware_allows_matching_header(monkeypatch, client):
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", "secret")
    response = client.get("/documents", headers={"X-Demo-Key": "secret"})
    assert response.status_code == 200


def test_demo_key_middleware_allows_matching_query_param(monkeypatch, client):
    # GET /chat/stream is consumed via the browser's native EventSource,
    # which can't set custom headers -- a `key` query param is the only way
    # it can supply this at all (see api/middleware.py's module docstring).
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", "secret")
    response = client.get("/documents", params={"key": "secret"})
    assert response.status_code == 200


def test_demo_key_middleware_401s_with_wrong_query_param(monkeypatch, client):
    monkeypatch.setattr(settings, "DEMO_ACCESS_KEY", "secret")
    response = client.get("/documents", params={"key": "wrong"})
    assert response.status_code == 401


# --- routing: wrong-method requests should 405, not fall through to the
# frontend static mount and 404 ---


def test_wrong_method_on_known_route_returns_405_not_404(client):
    response = client.get("/chat")  # only POST is defined
    assert response.status_code == 405
