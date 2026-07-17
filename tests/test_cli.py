from sage import cli
from sage.db import conversations
from sage.generation.answer_engine import AnswerResult


def _fake_result(text="An answer.", citations=None):
    return AnswerResult(
        answer_text=text,
        citations=citations or [],
        model="gemini-test",
        retrieval_latency_ms=1.0,
        generation_latency_ms=2.0,
        total_latency_ms=3.0,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        retrieved_chunk_ids=[1],
    )


def test_build_parser_ask_supports_repeated_company_flag():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["ask", "compare capex", "--company", "Apple", "--company", "Microsoft"]
    )
    assert args.company == ["Apple", "Microsoft"]


def test_build_parser_ask_defaults_company_to_none():
    parser = cli.build_parser()
    args = parser.parse_args(["ask", "what were margins"])
    assert args.company is None


def test_cmd_ask_non_streaming_prints_answer_and_citations(monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate_answer", lambda *a, **kw: _fake_result("The answer text."))

    parser = cli.build_parser()
    args = parser.parse_args(["ask", "what were margins"])
    cli.cmd_ask(args)

    out = capsys.readouterr().out
    assert "The answer text." in out
    assert "(no citations resolved)" in out


def test_cmd_ask_passes_multiple_companies_through(monkeypatch):
    captured = {}

    def fake_generate_answer(query, top_k, companies=None, history=None):
        captured["companies"] = companies
        return _fake_result()

    monkeypatch.setattr(cli, "generate_answer", fake_generate_answer)

    parser = cli.build_parser()
    args = parser.parse_args(["ask", "compare", "--company", "Apple", "--company", "Microsoft"])
    cli.cmd_ask(args)

    assert captured["companies"] == ["Apple", "Microsoft"]


def test_cmd_ask_new_conversation_persists_turns(monkeypatch):
    monkeypatch.setattr(cli, "generate_answer", lambda *a, **kw: _fake_result("Persisted answer."))

    parser = cli.build_parser()
    args = parser.parse_args(["ask", "what were margins", "--new-conversation", "Margins thread"])
    cli.cmd_ask(args)

    convs = conversations.list_conversations()
    assert len(convs) == 1
    assert convs[0].title == "Margins thread"

    history = conversations.get_history(convs[0].id)
    assert [h.role for h in history] == ["user", "assistant"]
    assert history[1].content == "Persisted answer."


def test_cmd_ask_continues_existing_conversation_with_history(monkeypatch):
    conversation_id, _ = conversations.create_conversation(title="existing")
    conversations.append_message(conversation_id, "user", "first turn")
    conversations.append_message(conversation_id, "assistant", "first answer")

    captured = {}

    def fake_generate_answer(query, top_k, companies=None, history=None):
        captured["history"] = history
        return _fake_result("second answer")

    monkeypatch.setattr(cli, "generate_answer", fake_generate_answer)

    parser = cli.build_parser()
    args = parser.parse_args(["ask", "second question", "--conversation-id", str(conversation_id)])
    cli.cmd_ask(args)

    assert captured["history"] is not None
    assert [h.content for h in captured["history"]] == ["first turn", "first answer"]

    history_after = conversations.get_history(conversation_id)
    assert len(history_after) == 4


def test_cmd_conversations_lists_created_conversations(capsys):
    conversations.create_conversation(title="Apple margins")

    parser = cli.build_parser()
    args = parser.parse_args(["conversations"])
    cli.cmd_conversations(args)

    out = capsys.readouterr().out
    assert "Apple margins" in out


def test_cmd_conversations_handles_empty_state(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["conversations"])
    cli.cmd_conversations(args)

    out = capsys.readouterr().out
    assert "No conversations yet." in out
