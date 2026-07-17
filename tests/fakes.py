"""Network-free fakes for the google-genai SDK surface GeminiChatClient uses.

Mirrors the real SDK's response shapes closely enough (verified against the
installed `google-genai` package's actual pydantic model fields -- see
GeminiChatClient's docstring) that GeminiChatClient exercises its real
parsing code against these fakes, not a simplified stand-in. No real client
is ever constructed in tests using these.

Embeddings are no longer part of this surface (they run locally via
sentence-transformers now, not through google-genai -- see
sage/embed/local_embedder.py), so this file no longer fakes
`embed_content`; see tests/test_local_embedder.py for how that module is
tested instead.
"""

from dataclasses import dataclass, field


@dataclass
class FakeUsage:
    prompt_token_count: int = 0
    candidates_token_count: int = 0


@dataclass
class FakeGenerateContentResponse:
    text: str
    usage_metadata: FakeUsage = field(default_factory=FakeUsage)


class FakeModelsGenerate:
    """Fakes `client.models.generate_content` / `generate_content_stream`."""

    def __init__(
        self, response_text: str = "An answer.", prompt_tokens: int = 10, completion_tokens: int = 5
    ):
        self.response_text = response_text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.last_call: dict | None = None

    def generate_content(self, *, model: str, contents, config=None):
        self.last_call = {"model": model, "contents": contents, "config": config}
        return FakeGenerateContentResponse(
            text=self.response_text,
            usage_metadata=FakeUsage(
                prompt_token_count=self.prompt_tokens,
                candidates_token_count=self.completion_tokens,
            ),
        )

    def generate_content_stream(self, *, model: str, contents, config=None):
        self.last_call = {"model": model, "contents": contents, "config": config}
        words = self.response_text.split(" ")
        for i, word in enumerate(words):
            piece = word if i == len(words) - 1 else word + " "
            usage = (
                FakeUsage(
                    prompt_token_count=self.prompt_tokens,
                    candidates_token_count=self.completion_tokens,
                )
                if i == len(words) - 1
                else None
            )
            yield FakeGenerateContentResponse(text=piece, usage_metadata=usage or FakeUsage())


class FakeGenaiModels:
    def __init__(self, generate: FakeModelsGenerate):
        self.generate_content = generate.generate_content
        self.generate_content_stream = generate.generate_content_stream


class FakeGenaiClient:
    """Stands in for `genai.Client` — no network calls, no API key needed."""

    def __init__(
        self,
        response_text: str = "An answer.",
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ):
        self._generate = FakeModelsGenerate(response_text, prompt_tokens, completion_tokens)
        self.models = FakeGenaiModels(self._generate)
