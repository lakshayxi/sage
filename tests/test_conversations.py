from sage.db import conversations


def test_create_conversation_returns_an_id_and_a_session_token():
    conversation_id, token = conversations.create_conversation(title="Apple margins")
    assert isinstance(conversation_id, int)
    assert isinstance(token, str)
    assert token  # non-empty


def test_create_conversation_mints_a_fresh_token_by_default():
    _, first_token = conversations.create_conversation(title="a")
    _, second_token = conversations.create_conversation(title="b")
    assert first_token != second_token


def test_create_conversation_reuses_a_given_session_token():
    conversation_id, token = conversations.create_conversation(
        title="second in same session", session_token="existing-token"
    )
    assert token == "existing-token"
    row = conversations.get_conversation(conversation_id)
    assert row.session_token == "existing-token"


def test_append_message_auto_increments_ordered():
    conversation_id, _ = conversations.create_conversation(title="test")

    first_id = conversations.append_message(conversation_id, "user", "What were margins?")
    second_id = conversations.append_message(conversation_id, "assistant", "They declined.")

    assert first_id != second_id
    history = conversations.get_history(conversation_id)
    assert [h.content for h in history] == ["What were margins?", "They declined."]
    assert [h.role for h in history] == ["user", "assistant"]


def test_get_history_is_empty_for_new_conversation():
    conversation_id, _ = conversations.create_conversation(title="empty")
    assert conversations.get_history(conversation_id) == []


def test_list_conversations_orders_newest_first():
    first, _ = conversations.create_conversation(title="first")
    second, _ = conversations.create_conversation(title="second")

    result = conversations.list_conversations()

    ids = [c.id for c in result]
    assert ids.index(second) < ids.index(first)


def test_list_conversations_scoped_to_session_token():
    first_id, first_token = conversations.create_conversation(title="mine")
    second_id, second_token = conversations.create_conversation(title="theirs")
    assert first_token != second_token

    mine = conversations.list_conversations(session_token=first_token)
    theirs = conversations.list_conversations(session_token=second_token)

    assert [c.id for c in mine] == [first_id]
    assert [c.id for c in theirs] == [second_id]


def test_list_conversations_with_unknown_token_is_empty():
    conversations.create_conversation(title="someone else's")
    assert conversations.list_conversations(session_token="no-such-token") == []


def test_get_conversation_includes_messages_in_order():
    conversation_id, _ = conversations.create_conversation(title="ordered")
    conversations.append_message(conversation_id, "user", "first turn")
    conversations.append_message(conversation_id, "assistant", "second turn", citations=[{"n": 1}])

    conv = conversations.get_conversation(conversation_id)

    assert conv is not None
    assert [m.content for m in conv.messages] == ["first turn", "second turn"]
    assert conv.messages[1].citations == [{"n": 1}]


def test_get_conversation_returns_none_for_unknown_id():
    assert conversations.get_conversation(999999) is None


def test_get_conversation_scoped_to_session_token():
    conversation_id, token = conversations.create_conversation(title="mine")

    assert conversations.get_conversation(conversation_id, session_token=token) is not None
    assert conversations.get_conversation(conversation_id, session_token="wrong-token") is None


def test_conversation_belongs_to_session():
    conversation_id, token = conversations.create_conversation(title="mine")

    assert conversations.conversation_belongs_to_session(conversation_id, token) is True
    assert conversations.conversation_belongs_to_session(conversation_id, "wrong-token") is False
    assert conversations.conversation_belongs_to_session(999999, token) is False
