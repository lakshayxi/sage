"""Gemini chat client (google-genai SDK).

Sage's generation is Gemini-only by design -- unlike the reference project's
`LLMProvider` Protocol (built to support multiple backends: Ollama + a stub
+ future providers), there's no provider abstraction here, just one
concrete class wrapping `client.models.generate_content` /
`generate_content_stream`. Embeddings, unlike chat, are NOT Gemini -- see
sage/embed/local_embedder.py for why (a hard, non-recoverable free-tier
embedding quota wall) and note `call_with_retry` (sage/retry.py) is
consequently scoped to this client only now.

`GeminiChatClient` takes an injectable `client` constructor param, defaulting
to a real `genai.Client(api_key=settings.GEMINI_API_KEY)` only when none is
given -- so tests can inject a network-free fake (see tests/fakes.py) and
never need GEMINI_API_KEY set or hit the network.

NOTE on live validation (2026-07-17, real GEMINI_API_KEY): chat(), chat_stream(),
citation-fence parsing (see answer_engine.py), and the comparison-mode
prompt (see prompts.py) were all exercised live and worked exactly as
designed on the first try -- `usage_metadata.prompt_token_count`/
`candidates_token_count` populate correctly, `.text` is clean on both
plain and streamed responses, and the model reliably emitted the exact
```citations fence format on both a single-company and a multi-company
(comparison) prompt. What that same session found NOT to hold up: the
originally-WebFetched default model (gemini-2.5-flash) 404s live as
"no longer available to new users" -- see GEMINI_CHAT_MODEL's comment in
config/settings.py for the model that replaced it and why. Not yet
exercised live: the citation-fallback parsing paths for a *malformed*
response (dropped fence, truncated JSON) -- only well-formed output was
observed in this session's live calls, so those fallbacks (carried over
from the reference project's llama3.1-specific observations) remain
unverified against genuine Gemini failure output.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from google import genai
from google.genai import types

from config import settings
from sage.retry import call_with_retry


@dataclass
class ChatResult:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class StreamToken:
    content: str


@dataclass
class StreamDone:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _split_system_and_turns(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Gemini takes a single `system_instruction`, not a "system" turn in the
    conversation -- pull the (at most one, expected-first) system message out
    of the role/content list our callers build, and translate the rest into
    Gemini `Content` turns. Our "assistant" role maps to Gemini's "model".
    """
    system: str | None = None
    turns: list[types.Content] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system = content
            continue
        gemini_role = "model" if role == "assistant" else "user"
        turns.append(types.Content(role=gemini_role, parts=[types.Part(text=content)]))
    return system, turns


def _usage_tokens(usage) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    completion_tokens = getattr(usage, "candidates_token_count", None) or 0
    return prompt_tokens, completion_tokens


class GeminiChatClient:
    def __init__(
        self,
        client: genai.Client | None = None,
        model: str = settings.GEMINI_CHAT_MODEL,
    ):
        self._client = client
        self.model = model

    def _client_or_default(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _config(self, system: str | None) -> types.GenerateContentConfig | None:
        if system is None:
            return None
        return types.GenerateContentConfig(system_instruction=system)

    def chat(self, messages: list[dict]) -> ChatResult:
        client = self._client_or_default()
        system, turns = _split_system_and_turns(messages)
        response = call_with_retry(
            lambda: client.models.generate_content(
                model=self.model,
                contents=turns,
                config=self._config(system),
            ),
            what="generate_content",
        )
        prompt_tokens, completion_tokens = _usage_tokens(getattr(response, "usage_metadata", None))
        return ChatResult(
            content=response.text or "",
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def chat_stream(self, messages: list[dict]) -> Iterator[StreamToken | StreamDone]:
        client = self._client_or_default()
        system, turns = _split_system_and_turns(messages)
        # Only the call that opens the stream is retried -- once iteration
        # below has yielded a token to the caller, retrying the whole
        # request on a later mid-stream error would duplicate content, so a
        # 429 partway through a stream still propagates rather than retrying.
        stream = call_with_retry(
            lambda: client.models.generate_content_stream(
                model=self.model,
                contents=turns,
                config=self._config(system),
            ),
            what="generate_content_stream",
        )
        prompt_tokens = completion_tokens = 0
        for chunk in stream:
            usage = getattr(chunk, "usage_metadata", None)
            if usage is not None:
                p, c = _usage_tokens(usage)
                prompt_tokens = p or prompt_tokens
                completion_tokens = c or completion_tokens
            text = getattr(chunk, "text", None)
            if text:
                yield StreamToken(content=text)
        yield StreamDone(
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
