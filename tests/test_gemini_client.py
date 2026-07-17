from sage.generation.gemini_client import GeminiChatClient, StreamDone, StreamToken
from tests.fakes import FakeGenaiClient


def test_chat_returns_content_and_token_usage():
    fake_client = FakeGenaiClient(
        response_text="The margins declined.", prompt_tokens=42, completion_tokens=7
    )
    client = GeminiChatClient(client=fake_client, model="gemini-test")

    result = client.chat([{"role": "user", "content": "Why did margins decline?"}])

    assert result.content == "The margins declined."
    assert result.model == "gemini-test"
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7
    assert result.total_tokens == 49


def test_chat_splits_system_message_into_system_instruction():
    fake_client = FakeGenaiClient(response_text="ok")
    client = GeminiChatClient(client=fake_client)

    client.chat(
        [
            {"role": "system", "content": "You are a financial assistant."},
            {"role": "user", "content": "hello"},
        ]
    )

    call = fake_client._generate.last_call
    assert call["config"].system_instruction == "You are a financial assistant."
    # Only the non-system turn should remain in `contents`.
    assert len(call["contents"]) == 1
    assert call["contents"][0].role == "user"


def test_chat_maps_assistant_role_to_gemini_model_role():
    fake_client = FakeGenaiClient(response_text="ok")
    client = GeminiChatClient(client=fake_client)

    client.chat(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
            {"role": "user", "content": "follow-up"},
        ]
    )

    roles = [c.role for c in fake_client._generate.last_call["contents"]]
    assert roles == ["user", "model", "user"]


def test_chat_stream_yields_tokens_then_a_done_event():
    fake_client = FakeGenaiClient(
        response_text="one two three", prompt_tokens=10, completion_tokens=3
    )
    client = GeminiChatClient(client=fake_client, model="gemini-test")

    events = list(client.chat_stream([{"role": "user", "content": "hi"}]))

    tokens = [e for e in events if isinstance(e, StreamToken)]
    done = [e for e in events if isinstance(e, StreamDone)]
    assert "".join(t.content for t in tokens) == "one two three"
    assert len(done) == 1
    assert done[0].model == "gemini-test"
    assert done[0].prompt_tokens == 10
    assert done[0].completion_tokens == 3
    assert done[0].total_tokens == 13


def test_chat_stream_with_no_system_message_passes_none_config():
    fake_client = FakeGenaiClient(response_text="ok")
    client = GeminiChatClient(client=fake_client)

    list(client.chat_stream([{"role": "user", "content": "hi"}]))

    assert fake_client._generate.last_call["config"] is None


def test_injected_client_means_no_real_client_is_ever_constructed():
    # GEMINI_API_KEY is not set in the test environment; if GeminiChatClient
    # ever fell through to constructing a real genai.Client with a missing
    # key it would still succeed here (the SDK doesn't validate the key at
    # construction time) -- what actually matters is that `_client` stays
    # exactly the injected fake instance and no network call is attempted,
    # which the fake's deterministic response above already proves.
    fake_client = FakeGenaiClient()
    client = GeminiChatClient(client=fake_client)
    assert client._client is fake_client
